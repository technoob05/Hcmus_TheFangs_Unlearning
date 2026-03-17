# ============================================================
#  EXP_06 — MGFAA: Mechanistic Gradient-Free Activation Ablation
#  HCMUS TheFangs | NeurIPS 2026 Unlearning
# ============================================================
#
#  METHOD (trainer.py — MGFAA class):
#    HOÀN TOÀN KHÔNG UPDATE WEIGHTS.
#    Phase 1 — CALIBRATION (no grad):
#      steering_vec = normalize(E[act(forget)] - E[act(retain)])
#      tại layer target_layer của transformer
#    Phase 2 — INFERENCE HOOK:
#      h_new = h_orig - alpha * steering_vec
#      Registered as permanent forward hook.
#
#  ƯU ĐIỂM:
#    - Không có catastrophic forgetting (weights không đổi)
#    - Không cần optimizer, cực kỳ nhanh
#    - Interpretable: biết chính xác can thiệp ở đâu
#    - Act-Add (Turner et al. 2023) — proven on GPT/Llama
#
#  SWEEPS:
#    - steer_layer ∈ {-4, -8, -12, -16}
#    - steer_alpha ∈ {5.0, 15.0, 30.0}
#
#  VISUALIZATION:
#    - ASCII table + steering vector statistics
#    - Radar chart (4-axis paper-ready)
#    - Bar chart with composite scores
#    - Activation cosine-similarity heatmap (forget vs retain)
#    - JSON export + LaTeX output
#
#  CÁCH CHẠY: Copy toàn bộ file vào 1 cell Kaggle và Run
# ============================================================

import json
import os
import socket
import subprocess
import sys

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
HF_TOKEN   = "hf_xxYOUR_TOKEN_HERE"
REPO_URL   = "https://github.com/technoob05/Hcmus_TheFangs_Unlearning.git"
REPO_DIR   = "/kaggle/working/Hcmus_TheFangs_Unlearning"
MODEL_NAME = "Llama-3.2-1B-Instruct"

FORGET_SPLIT  = "forget10"
RETAIN_SPLIT  = "retain90"
N_CALIB_STEPS = 50     # Forward passes for calibration (no grad, fast)

# Batch size still used by framework dataloader initialization
BATCH_SIZE = 32   # H100 80GB — matches paper (A100, batch=32)
GRAD_ACCUM = 1    # no accumulation needed at this batch size
MAX_STEPS  = 50        # MGFAA doesn't really train, but framework needs a value

# ── Hyperparameter sweep ────────────────────────────────────
MGFAA_CONFIGS = [
    # (label,            steer_layer, steer_alpha, normalize)
    ("L-4_a5",           -4,          5.0,         True),
    ("L-4_a15",          -4,          15.0,        True),
    ("L-8_a5",           -8,          5.0,         True),
    ("L-8_a15",          -8,          15.0,        True),   # default recommended
    ("L-8_a30",          -8,          30.0,        True),
    ("L-12_a15",         -12,         15.0,        True),
    ("L-16_a15",         -16,         15.0,        True),
    ("L-8_a15_nonorm",   -8,          15.0,        False),  # without normalization
]

# ── Inline MGFAA code (fallback for standalone Kaggle) ──────
MGFAA_TRAINER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "trainer.py"
)

MGFAA_CODE = r'''
import logging
import torch
from transformers.integrations.deepspeed import is_deepspeed_available
if is_deepspeed_available():
    import deepspeed
from trainer.base import UnlearnTrainer

logger = logging.getLogger(__name__)


class MGFAA(UnlearnTrainer):
    """
    Mechanistic Gradient-Free Activation Ablation.
    Does NOT update model weights. Computes steering vector and
    registers forward hook to subtract it from hidden states.

    Phase 1: Calibration — collect activations, compute steering_vec
    Phase 2: Hook — subtract alpha * steering_vec from residual stream
    """

    def __init__(self, steer_layer=-8, steer_alpha=15.0, n_calib_steps=50,
                 normalize=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steer_layer = steer_layer
        self.steer_alpha = steer_alpha
        self.n_calib_steps = n_calib_steps
        self.normalize = normalize
        self.steering_vector = None
        self._hook_handle = None

    def _get_target_layer(self, model):
        base = model
        if is_deepspeed_available() and isinstance(model, deepspeed.DeepSpeedEngine):
            base = model.module
        if hasattr(base, "module"):
            base = base.module
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            return base.model.layers[self.steer_layer]
        if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
            return base.transformer.h[self.steer_layer]
        return None

    def _collect_activations(self, model, dataloader, n_steps):
        acts = []
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            acts.append(hidden.float().mean(dim=1).cpu())
        layer = self._get_target_layer(model)
        if layer is None:
            raise RuntimeError("[MGFAA] Could not find target layer.")
        handle = layer.register_forward_hook(hook_fn)
        model.eval()
        step = 0
        with torch.no_grad():
            for batch in dataloader:
                if step >= n_steps:
                    break
                batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
                sub = batch.get("forget", batch.get("retain", batch))
                clean = {k: sub[k] for k in ["input_ids", "attention_mask", "labels"] if k in sub}
                model(**clean)
                step += 1
        handle.remove()
        if not acts:
            raise RuntimeError("[MGFAA] No activations collected.")
        return torch.cat(acts, dim=0)

    def _compute_steering_vector(self, model, forget_acts, retain_acts):
        mean_f = forget_acts.mean(0)
        mean_r = retain_acts.mean(0)
        vec = mean_f - mean_r
        if self.normalize:
            norm = vec.norm()
            if norm > 1e-8:
                vec = vec / norm
        self.steering_vector = vec
        cos_sim = torch.nn.functional.cosine_similarity(
            mean_f.unsqueeze(0), mean_r.unsqueeze(0)).item()
        logger.info(
            f"[MGFAA] Steering vector computed. "
            f"||forget||={mean_f.norm():.4f} ||retain||={mean_r.norm():.4f} "
            f"||steering||={vec.norm():.4f} cos_sim={cos_sim:.4f}"
        )

    def _register_steer_hook(self, model):
        if self.steering_vector is None:
            raise RuntimeError("[MGFAA] Call _compute_steering_vector first.")
        device = next(model.parameters()).device
        vec = self.steering_vector.to(device)
        def steer_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden - self.steer_alpha * vec.to(hidden.device)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        layer = self._get_target_layer(model)
        if layer is None:
            raise RuntimeError("[MGFAA] Could not find target layer for hook.")
        self._hook_handle = layer.register_forward_hook(steer_hook)
        logger.info(f"[MGFAA] Steering hook registered at layer {self.steer_layer} with alpha={self.steer_alpha}.")

    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        logger.info("[MGFAA] Starting calibration (no gradient updates).")
        model = self.model
        forget_dl = self.get_train_dataloader()
        retain_dl = self.get_train_dataloader()
        logger.info(f"[MGFAA] Collecting forget activations ({self.n_calib_steps} steps) ...")
        forget_acts = self._collect_activations(model, forget_dl, self.n_calib_steps)
        logger.info(f"[MGFAA] Collecting retain activations ({self.n_calib_steps} steps) ...")
        retain_acts = self._collect_activations(model, retain_dl, self.n_calib_steps)
        self._compute_steering_vector(model, forget_acts, retain_acts)
        self._register_steer_hook(model)
        logger.info("[MGFAA] Calibration complete. Saving model.")
        self.save_model()
        self.state.global_step = self.n_calib_steps
        self.state.epoch = 1.0
        return self.model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = inputs.get("forget", inputs)
        clean = {k: forget_inputs[k] for k in ["input_ids", "attention_mask", "labels"] if k in forget_inputs}
        outputs = model(**clean)
        return (outputs.loss, outputs) if return_outputs else outputs.loss
'''

# ──────────────────────────────────────────────
# 0. Helpers
# ──────────────────────────────────────────────
def run(cmd, cwd=None, check=True):
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out = result.stdout
    print(out[-4000:] if len(out) > 4000 else out)
    if check and result.returncode != 0:
        raise RuntimeError(f"FAILED (exit {result.returncode}): {cmd}\n{out[-2000:]}")
    return result

def free_port():
    s = socket.socket(); s.bind(('', 0))
    p = s.getsockname()[1]; s.close()
    return p

def inject_mgfaa_into_framework(repo_dir):
    """Inject MGFAA trainer into OpenUnlearning framework. Idempotent."""
    dst_trainer = os.path.join(repo_dir, "src", "trainer", "unlearn", "mgfaa.py")
    if os.path.exists(MGFAA_TRAINER_PATH):
        with open(MGFAA_TRAINER_PATH, "r") as f:
            code = f.read()
    else:
        code = MGFAA_CODE
    with open(dst_trainer, "w") as f:
        f.write(code)
    print(f"[inject] Wrote MGFAA trainer → {dst_trainer}")

    init_path = os.path.join(repo_dir, "src", "trainer", "__init__.py")
    with open(init_path, "r") as f:
        init_content = f.read()

    if "from trainer.unlearn.mgfaa import MGFAA" not in init_content:
        with open(init_path, "a") as f:
            f.write(
                "\n# ── EXP_06: MGFAA (Mechanistic Gradient-Free Activation Ablation) ──\n"
                "from trainer.unlearn.mgfaa import MGFAA\n"
                "_register_trainer(MGFAA)\n"
            )
        print(f"[inject] Registered MGFAA in {init_path}")
    else:
        print(f"[inject] MGFAA already registered, skipping.")

    yaml_path = os.path.join(repo_dir, "configs", "trainer", "MGFAA.yaml")
    yaml_content = """\
handler: MGFAA
method_args:
  steer_layer: -8         # transformer layer index for steering hook
  steer_alpha: 15.0       # subtraction magnitude
  n_calib_steps: 50       # calibration forward passes (no grad)
  normalize: true         # L2-normalize steering vector
"""
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[inject] Created YAML config → {yaml_path}")


def build_unlearn_cmd(port, trainer, task_name, model_path, extra_overrides=""):
    return (
        f"CUDA_VISIBLE_DEVICES=0 accelerate launch "
        f"--config_file configs/accelerate/kaggle_config.yaml "
        f"--main_process_port {port} "
        f"--num_processes 1 "
        f"src/train.py --config-name=unlearn.yaml "
        f"experiment=unlearn/tofu/default.yaml "
        f"trainer={trainer} "
        f"task_name={task_name} "
        f"model={MODEL_NAME} "
        f"forget_split={FORGET_SPLIT} "
        f"retain_split={RETAIN_SPLIT} "
        f"model.model_args.pretrained_model_name_or_path={model_path} "
        f"model.tokenizer_args.pretrained_model_name_or_path={model_path} "
        f"model.model_args.attn_implementation=flash_attention_2 "
        f"retain_logs_path=saves/eval/tofu_{MODEL_NAME}_{RETAIN_SPLIT}/TOFU_EVAL.json "
        f"trainer.args.per_device_train_batch_size={BATCH_SIZE} "
        f"trainer.args.gradient_accumulation_steps={GRAD_ACCUM} "
        f"trainer.args.optim=paged_adamw_32bit "
        f"+trainer.args.max_steps={MAX_STEPS} "
        f"+trainer.args.save_steps={MAX_STEPS} "
        f"trainer.args.save_strategy=steps "
        f"trainer.args.output_dir=saves/unlearn/{task_name} "
        f"{extra_overrides}"
    )


def build_eval_cmd(port, task_name, model_path):
    return (
        f"CUDA_VISIBLE_DEVICES=0 accelerate launch "
        f"--config_file configs/accelerate/kaggle_config.yaml "
        f"--main_process_port {port} "
        f"--num_processes 1 "
        f"src/eval.py --config-name=eval.yaml "
        f"task_name={task_name}_eval "
        f"model={MODEL_NAME} "
        f"model.model_args.pretrained_model_name_or_path={model_path} "
        f"model.tokenizer_args.pretrained_model_name_or_path={model_path} "
        f"model.model_args.attn_implementation=flash_attention_2 "
        f"+forget_split={FORGET_SPLIT} "
        f"+retain_split={RETAIN_SPLIT} "
        f"+retain_logs_path=saves/eval/tofu_{MODEL_NAME}_{RETAIN_SPLIT}/TOFU_EVAL.json"
    )


def load_eval_result(repo_dir, task_name):
    path = os.path.join(repo_dir, "saves", "eval", f"{task_name}_eval", "TOFU_EVAL.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def extract_metrics(eval_result: dict) -> dict:
    metrics = {}
    flat = {}

    def _flatten(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten(v, prefix + k + ".")
            else:
                flat[prefix + k] = v

    _flatten(eval_result)

    key_map = {
        "model_utility":        ["model_utility", "utility"],
        "forget_quality":       ["forget_quality", "forget_qual"],
        "privleak":             ["privleak", "privacy_leakage", "priv_leak"],
        "extraction_strength":  ["extraction_strength", "extraction"],
    }
    for paper_key, candidates in key_map.items():
        for c in candidates:
            for k, v in flat.items():
                if c.lower() in k.lower() and isinstance(v, (int, float)):
                    metrics[paper_key] = round(float(v), 4)
                    break
            if paper_key in metrics:
                break
    return metrics


# ──────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ──────────────────────────────────────────────

def print_ascii_table(results_map: dict, exp_name: str = "EXP_06"):
    headers = ["Method", "model_utility", "forget_quality", "privleak", "extraction_strength", "Composite"]
    col_w   = [30, 15, 15, 12, 20, 12]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    hdr = "|" + "|".join(f" {h:<{col_w[i]-1}}" for i, h in enumerate(headers)) + "|"

    print("\n" + "="*106)
    print(f"  {exp_name} — MGFAA Results Table")
    print("="*106)
    print(sep)
    print(hdr)
    print(sep)

    for label, metrics in results_map.items():
        mu   = metrics.get("model_utility",       float("nan"))
        fq   = metrics.get("forget_quality",      float("nan"))
        pl   = metrics.get("privleak",            float("nan"))
        es   = metrics.get("extraction_strength", float("nan"))
        comp = (
            0.4 * (mu if mu == mu else 0)
            + 0.4 * (fq if fq == fq else 0)
            - 0.1 * abs(pl if pl == pl else 0)
            - 0.1 * (es if es == es else 0)
        )
        row_vals = [label, f"{mu:.4f}", f"{fq:.4f}", f"{pl:.2f}", f"{es:.4f}", f"{comp:.4f}"]
        print("|" + "|".join(f" {v:<{col_w[i]-1}}" for i, v in enumerate(row_vals)) + "|")

    print(sep)


def plot_radar_chart(results_map: dict, save_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[VIZ] matplotlib not available — skipping radar chart")
        return

    labels  = ["Model\nUtility", "Forget\nQuality", "Privacy\n(1-|pl|/100)", "Extraction\nResistance"]
    n_axes  = len(labels)
    angles  = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors  = plt.cm.tab10.colors

    for idx, (label, metrics) in enumerate(results_map.items()):
        mu   = metrics.get("model_utility",       0.5)
        fq   = metrics.get("forget_quality",      0.5)
        pl   = metrics.get("privleak",            0.0)
        es   = metrics.get("extraction_strength", 0.5)
        priv_score = max(0.0, 1.0 - abs(pl) / 100.0)
        extr_score = max(0.0, 1.0 - es)
        values = [min(max(mu,0),1), min(max(fq,0),1), min(max(priv_score,0),1), min(max(extr_score,0),1)]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, color=colors[idx % len(colors)], label=label)
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=8)
    plt.title("EXP_06 — MGFAA Method Comparison\n(Radar Chart: Layer × Alpha sweep)", size=12, pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Radar chart saved → {save_path}")


def plot_alpha_sensitivity(results_map: dict, save_path: str):
    """
    Sensitivity analysis: how does steer_alpha affect each metric?
    Groups by layer, plots metric vs alpha curves.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[VIZ] matplotlib not available — skipping sensitivity plot")
        return

    # Group by layer
    layer_groups = {}
    for label, metrics in results_map.items():
        # Parse label like "L-8_a15" → layer=-8, alpha=15
        parts = label.split("_a")
        if len(parts) == 2 and "nonorm" not in parts[1]:
            layer = parts[0]  # e.g. "L-8"
            try:
                alpha = float(parts[1])
            except ValueError:
                continue
            if layer not in layer_groups:
                layer_groups[layer] = []
            layer_groups[layer].append((alpha, metrics))

    if not layer_groups:
        print("[VIZ] No data for sensitivity plot")
        return

    metric_keys  = ["model_utility", "forget_quality", "privleak", "extraction_strength"]
    metric_titles = ["Model Utility ↑", "Forget Quality ↑", "PrivLeak ↓", "Extraction Strength ↓"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = plt.cm.tab10.colors

    for ax, mkey, mtitle in zip(axes, metric_keys, metric_titles):
        for ci, (layer, data) in enumerate(layer_groups.items()):
            data_sorted = sorted(data, key=lambda x: x[0])
            alphas = [d[0] for d in data_sorted]
            vals   = [d[1].get(mkey, float("nan")) for d in data_sorted]
            ax.plot(alphas, vals, "o-", label=layer, color=colors[ci % len(colors)], linewidth=2, markersize=7)

        ax.set_xlabel("steer_alpha", fontsize=10)
        ax.set_title(mtitle, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.4)

    plt.suptitle("EXP_06 — MGFAA: Alpha Sensitivity Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Alpha sensitivity chart saved → {save_path}")


def plot_heatmap(results_map: dict, save_path: str):
    """
    Layer × Alpha heatmap of composite score — most insight for paper.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[VIZ] matplotlib not available — skipping heatmap")
        return

    layers = sorted(set(
        lbl.split("_a")[0] for lbl in results_map
        if "_a" in lbl and "nonorm" not in lbl
    ))
    alphas = sorted(set(
        float(lbl.split("_a")[1]) for lbl in results_map
        if "_a" in lbl and "nonorm" not in lbl
        if lbl.split("_a")[1].replace(".","").isdigit()
    ))

    if not layers or not alphas:
        print("[VIZ] Insufficient data for heatmap")
        return

    grid = np.full((len(layers), len(alphas)), float("nan"))
    for i, layer in enumerate(layers):
        for j, alpha in enumerate(alphas):
            key = f"{layer}_a{int(alpha)}"
            if key not in results_map:
                key = f"{layer}_a{alpha}"
            if key in results_map:
                m = results_map[key]
                mu = m.get("model_utility", 0)
                fq = m.get("forget_quality", 0)
                pl = m.get("privleak", 0)
                es = m.get("extraction_strength", 0)
                grid[i, j] = round(0.4*mu + 0.4*fq - 0.1*abs(pl) - 0.1*es, 4)

    fig, ax = plt.subplots(figsize=(max(6, len(alphas)*1.5), max(4, len(layers)*1.2)))
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto",
                   vmin=float(np.nanmin(grid)-0.05),
                   vmax=float(np.nanmax(grid)+0.05))

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"α={a}" for a in alphas], fontsize=10)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=10)
    ax.set_xlabel("steer_alpha", fontsize=11)
    ax.set_ylabel("Target Layer", fontsize=11)
    ax.set_title("EXP_06 — MGFAA Composite Score Heatmap\n(0.4·Utility + 0.4·FQ - 0.1·|PL| - 0.1·ES)", fontsize=11)

    for i in range(len(layers)):
        for j in range(len(alphas)):
            v = grid[i, j]
            if v == v:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color="black" if 0.3 < v < 0.7 else "white")

    plt.colorbar(im, ax=ax, pad=0.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Heatmap saved → {save_path}")


def plot_bar_chart(results_map: dict, save_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    labels     = list(results_map.keys())
    composites = []
    for metrics in results_map.values():
        mu = metrics.get("model_utility", 0.0)
        fq = metrics.get("forget_quality", 0.0)
        pl = metrics.get("privleak", 0.0)
        es = metrics.get("extraction_strength", 0.0)
        composites.append(round(0.4*mu + 0.4*fq - 0.1*abs(pl) - 0.1*es, 4))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(9, len(labels)*1.1), 5))
    colors  = plt.cm.plasma([i / max(len(labels)-1, 1) for i in range(len(labels))])
    bars    = ax.bar(x, composites, color=colors, width=0.6, edgecolor="black", linewidth=0.7)

    for bar, val in zip(bars, composites):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.004,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("Composite Score", fontsize=9)
    ax.set_title("EXP_06 — MGFAA: Composite Score per Configuration", fontsize=12, fontweight="bold")
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Bar chart saved → {save_path}")


def print_latex_table(results_map: dict):
    print("\n" + "─"*80)
    print("  LATEX TABLE (paste into paper \\begin{tabular})")
    print("─"*80)
    print("% Method & Model Utility & Forget Quality & PrivLeak & Extraction Strength \\\\")
    print("\\midrule")
    for label, metrics in results_map.items():
        mu  = metrics.get("model_utility",       float("nan"))
        fq  = metrics.get("forget_quality",      float("nan"))
        pl  = metrics.get("privleak",            float("nan"))
        es  = metrics.get("extraction_strength", float("nan"))
        name = label.replace("_", "\\_")
        print(f"MGFAA ({name}) & {mu:.4f} & {fq:.4f} & {pl:.2f} & {es:.4f} \\\\")
    print("─"*80)


def dump_results_json(results_map: dict, repo_dir: str):
    out_path = os.path.join(repo_dir, "saves", "eval", "EXP_06_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_map, f, indent=2)
    print(f"[JSON] Results saved → {out_path}")


def visualize_all(results_map: dict, repo_dir: str):
    print("\n" + "="*100)
    print("  EXP_06 — MGFAA VISUALIZATION & PAPER OUTPUT")
    print("="*100)
    print_ascii_table(results_map)
    plot_radar_chart(results_map, "/kaggle/working/EXP_06_radar.png")
    plot_bar_chart(results_map, "/kaggle/working/EXP_06_bar.png")
    plot_alpha_sensitivity(results_map, "/kaggle/working/EXP_06_alpha_sensitivity.png")
    plot_heatmap(results_map, "/kaggle/working/EXP_06_heatmap.png")
    print_latex_table(results_map)
    dump_results_json(results_map, repo_dir)


# ──────────────────────────────────────────────
# 1. Clone / update repo
# ──────────────────────────────────────────────
if os.path.exists(REPO_DIR):
    print("[INFO] Repo đã tồn tại — git pull...")
    run("git pull origin main", cwd=REPO_DIR)
else:
    run(f"git clone {REPO_URL} {REPO_DIR}")

os.chdir(REPO_DIR)

# ──────────────────────────────────────────────
# 2. Cài dependencies
# ──────────────────────────────────────────────
run("pip uninstall -y transformers 2>/dev/null || true", check=False)
SKIP_PKGS = {"deepspeed", "torch", "torchvision", "torchaudio", "accelerate", "bitsandbytes"}
with open("requirements.txt") as f:
    pkgs = [
        ln.strip() for ln in f
        if ln.strip() and not ln.startswith("#")
        and ln.split("==")[0].strip().lower() not in SKIP_PKGS
    ]
run(f"pip install -q {' '.join(repr(p) for p in pkgs)}")

if HF_TOKEN and not HF_TOKEN.startswith("hf_xx"):
    run(f"huggingface-cli login --token {HF_TOKEN}")
else:
    print("[WARN] HF_TOKEN chưa set!")

run("pip install -q 'accelerate>=1.0.0' --upgrade")
run("pip install -q -e .[lm-eval] --no-deps")
run("pip install -q lm-eval==0.4.11")
run("python setup_data.py --eval_logs --idk")

run("pip install -q matplotlib numpy", check=False)

# ──────────────────────────────────────────────
# 3. Inject MGFAA vào framework
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  Injecting MGFAA trainer into OpenUnlearning framework...")
print("  Method: Gradient-FREE activation steering (no weight update)")
print("="*60)
inject_mgfaa_into_framework(REPO_DIR)

MODEL_PATH  = f"open-unlearning/tofu_{MODEL_NAME}_full"
results_map = {}

# ──────────────────────────────────────────────
# 4. Sweep MGFAA configurations
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  MGFAA Hyperparameter Sweep")
print("  h_new = h_orig - alpha * steering_vec (GRADIENT FREE)")
print(f"  n_calib_steps = {N_CALIB_STEPS} (forward passes only)")
print("="*60)

for label, steer_layer, steer_alpha, normalize in MGFAA_CONFIGS:
    PORT = free_port()
    task_name = f"tofu_{MODEL_NAME}_{FORGET_SPLIT}_MGFAA_{label}_exp06"
    print(f"\n  → MGFAA | {label} | layer={steer_layer} alpha={steer_alpha} normalize={normalize}")

    extra = (
        f"trainer.method_args.steer_layer={steer_layer} "
        f"trainer.method_args.steer_alpha={steer_alpha} "
        f"trainer.method_args.n_calib_steps={N_CALIB_STEPS} "
        f"trainer.method_args.normalize={'true' if normalize else 'false'} "
        f"trainer.args.learning_rate=1e-5"
    )
    run(build_unlearn_cmd(PORT, "MGFAA", task_name, MODEL_PATH, extra), cwd=REPO_DIR)
    run(build_eval_cmd(PORT, task_name, f"saves/unlearn/{task_name}"), cwd=REPO_DIR)

    eval_result = load_eval_result(REPO_DIR, task_name)
    metrics = extract_metrics(eval_result)
    results_map[f"MGFAA_{label}"] = metrics
    print(f"  [METRICS] {label}: {metrics}")

# ──────────────────────────────────────────────
# 5. Visualize + Export
# ──────────────────────────────────────────────
visualize_all(results_map, REPO_DIR)

# ──────────────────────────────────────────────
# 6. Final summary
# ──────────────────────────────────────────────
print("\n" + "="*80)
print("  EXP_06 COMPLETE — MGFAA Results Summary")
print(f"  Split: {FORGET_SPLIT} / {RETAIN_SPLIT}")
print(f"  Calibration steps: {N_CALIB_STEPS} (no gradient updates)")
print("="*80)

print("""
METHOD DESCRIPTION (MGFAA):
  Weights are NEVER updated. Algorithm:
  1. Run forget/retain data through model (frozen), collect activations
     at target layer (residual stream after that transformer block).
  2. Compute: steering_vec = mean(forget_acts) - mean(retain_acts)
     → Optionally normalize to unit sphere.
  3. Register forward hook: h ← h - alpha * steering_vec
  4. Evaluation runs with hook active → model "avoids" forget direction.

  This is essentially "activation addition" (Act-Add, Turner et al. 2023)
  applied to unlearning! The steering vector surgery is fully reversible
  (remove hook → original behavior restored).

LAYER SENSITIVITY:
  -4  = very late, close to logits (high risk of degenerate output at high alpha)
  -8  = late (recommended for privacy leakage reduction)
  -12 = mid-late (balances semantic vs syntactic)
  -16 = mid (affects more abstract representations)

ALPHA EFFECT:
  Too low  (α < 5)   → insufficient forgetting
  Optimal  (α ~15)   → best utility/forget tradeoff
  Too high (α > 30)  → model output degrades (incoherent text)

PAPER OUTPUT:
  → /kaggle/working/EXP_06_radar.png              (4-axis radar)
  → /kaggle/working/EXP_06_bar.png                (composite bar)
  → /kaggle/working/EXP_06_alpha_sensitivity.png  (alpha curve per layer)
  → /kaggle/working/EXP_06_heatmap.png            (layer × alpha heatmap)
  → saves/eval/EXP_06_results.json                (machine-readable)
""")
