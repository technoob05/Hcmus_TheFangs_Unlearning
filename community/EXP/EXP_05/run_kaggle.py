# ============================================================
#  EXP_05 — C-SAES: Contrastive Sparse Autoencoder Suppression
#  HCMUS TheFangs | NeurIPS 2026 Unlearning
# ============================================================
#
#  METHOD (trainer.py — CSAES class):
#    Loss = γ · L_grad_ascent(forget) + proj_coeff · L_supp + α · L_retain(NLL) + contrast_coeff · L_contrastive
#
#    L_supp      = mean(||A_forget · d_forget||²)   ← suppress forget direction
#    L_contrastive = -mean(||A_retain · d_forget||) ← keep retain directionally stable
#    d_forget = normalize(mean_forget_act - mean_retain_act) ← computed during warmup
#
#  SWEEPS:
#    - target_layer ∈ {-4, -8, -12}  (late, mid, mid-early layers)
#    - proj_coeff ∈ {0.5, 1.0, 2.0}
#
#  VISUALIZATION:
#    - ASCII table: method × metric
#    - Radar chart (matplotlib): 4-axis paper-ready figure
#    - Bar chart: composite score per method
#    - JSON export to saves/eval/EXP_05_results.json
#    - LaTeX table rows printed to stdout
#
#  CÁCH CHẠY: Copy toàn bộ file vào 1 cell Kaggle và Run
# ============================================================

import json
import os
import shutil
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

FORGET_SPLIT = "forget10"
RETAIN_SPLIT = "retain90"

BATCH_SIZE = 32   # H100 80GB — matches paper (A100, batch=32)
GRAD_ACCUM = 1    # no accumulation needed at this batch size
MAX_STEPS  = 200
WARMUP_STEPS = 20           # Steps to accumulate forget/retain directions

# ── Hyperparameter sweep ────────────────────────────────────
CSAES_CONFIGS = [
    # (label,         target_layer, proj_coeff, contrast_coeff)
    ("L-4_p0.5",     -4,           0.5,        0.3),   # late layer, weak suppression
    ("L-8_p1.0",     -8,           1.0,        0.5),   # mid layer, default
    ("L-12_p1.0",    -12,          1.0,        0.5),   # earlier layer 
    ("L-8_p2.0",     -8,           2.0,        0.5),   # stronger suppression
    ("L-8_p0.5",     -8,           0.5,        0.5),   # weaker suppression
]

# ── Inline CSAES trainer code (fallback for standalone Kaggle) ──
CSAES_TRAINER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "trainer.py"
)

CSAES_CODE = r'''
import logging
import torch
import torch.nn.functional as F
from transformers.integrations.deepspeed import is_deepspeed_available
if is_deepspeed_available():
    import deepspeed
from trainer.unlearn.grad_diff import GradDiff

logger = logging.getLogger(__name__)


class CSAES(GradDiff):
    """
    Contrastive Sparse Autoencoder Suppression (C-SAES).
    Approximates SAE-based feature suppression using contrastive
    mean-activation directions in residual stream.

    Loss = gamma*L_forget + proj_coeff*L_supp + alpha*L_retain + contrast_coeff*L_contrastive
    """

    def __init__(self, target_layer=-8, proj_coeff=1.0, contrast_coeff=0.5,
                 warmup_steps=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_layer = target_layer
        self.proj_coeff = proj_coeff
        self.contrast_coeff = contrast_coeff
        self.warmup_steps = warmup_steps
        self.forget_direction = None
        self._forget_act_sum = None
        self._retain_act_sum = None
        self._forget_act_count = 0
        self._retain_act_count = 0
        self._hook_handle = None
        self._hook_output = None

    def _get_target_layer(self, model):
        base = model
        if is_deepspeed_available() and isinstance(model, deepspeed.DeepSpeedEngine):
            base = model.module
        if hasattr(base, "module"):
            base = base.module
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            return base.model.layers[self.target_layer]
        if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
            return base.transformer.h[self.target_layer]
        return None

    def _register_hook(self, model):
        layer = self._get_target_layer(model)
        if layer is None:
            return
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._hook_output = hidden.mean(dim=1).detach()
        self._hook_handle = layer.register_forward_hook(hook_fn)

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _accumulate_direction(self, activation, is_forget):
        act = activation.float()
        if is_forget:
            if self._forget_act_sum is None:
                self._forget_act_sum = act.sum(0)
            else:
                self._forget_act_sum += act.sum(0)
            self._forget_act_count += act.shape[0]
        else:
            if self._retain_act_sum is None:
                self._retain_act_sum = act.sum(0)
            else:
                self._retain_act_sum += act.sum(0)
            self._retain_act_count += act.shape[0]

    def _finalize_direction(self):
        if (self._forget_act_sum is None or self._retain_act_sum is None or
                self._forget_act_count == 0 or self._retain_act_count == 0):
            return
        mean_forget = self._forget_act_sum / self._forget_act_count
        mean_retain = self._retain_act_sum / self._retain_act_count
        direction = mean_forget - mean_retain
        norm = direction.norm()
        if norm > 1e-8:
            direction = direction / norm
        self.forget_direction = direction
        logger.info(f"[C-SAES] Forget direction computed at step {self.state.global_step}. ||direction||={norm.item():.4f}")

    def _suppression_loss(self, activations):
        d = self.forget_direction.to(activations.device)
        proj = (activations.float() @ d).pow(2)
        return proj.mean()

    def _contrastive_loss(self, activations):
        d = self.forget_direction.to(activations.device)
        proj = (activations.float() @ d).abs()
        return -proj.mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        step = self.state.global_step if self.state is not None else 0
        forget_inputs = inputs["forget"]
        retain_inputs_raw = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs_raw["input_ids"],
            "attention_mask": retain_inputs_raw["attention_mask"],
            "labels": retain_inputs_raw["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        forget_clean = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_outputs = model(**forget_clean)
        forget_loss = -forget_outputs.loss
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        if step < self.warmup_steps:
            self._register_hook(model)
            with torch.no_grad():
                model(**forget_clean)
            if self._hook_output is not None:
                self._accumulate_direction(self._hook_output, is_forget=True)
            with torch.no_grad():
                model(**retain_inputs)
            if self._hook_output is not None:
                self._accumulate_direction(self._hook_output, is_forget=False)
            self._remove_hook()
            if step == self.warmup_steps - 1:
                self._finalize_direction()

        elif self.forget_direction is not None:
            self._register_hook(model)
            _ = model(**forget_clean)
            supp_loss = self._suppression_loss(self._hook_output) if self._hook_output is not None else torch.tensor(0.0)
            if self._hook_output is not None:
                loss = loss + self.proj_coeff * supp_loss
            with torch.no_grad():
                model(**retain_inputs)
            cont_loss = self._contrastive_loss(self._hook_output) if self._hook_output is not None else torch.tensor(0.0)
            if self._hook_output is not None:
                loss = loss + self.contrast_coeff * cont_loss
            self._remove_hook()

            if step % 10 == 0:
                logger.info(
                    f"[C-SAES] step={step:4d} | "
                    f"forget={forget_loss.item():.4f} | retain={retain_loss.item():.4f} | "
                    f"suppress={supp_loss.item() if hasattr(supp_loss, 'item') else 0:.4f} | "
                    f"contrastive={cont_loss.item() if hasattr(cont_loss, 'item') else 0:.4f} | "
                    f"total={loss.item():.4f}"
                )

        return (loss, forget_outputs) if return_outputs else loss
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

def inject_csaes_into_framework(repo_dir):
    """Inject CSAES trainer into OpenUnlearning framework. Idempotent."""
    dst_trainer = os.path.join(repo_dir, "src", "trainer", "unlearn", "c_saes.py")
    if os.path.exists(CSAES_TRAINER_PATH):
        with open(CSAES_TRAINER_PATH, "r") as f:
            code = f.read()
    else:
        code = CSAES_CODE
    with open(dst_trainer, "w") as f:
        f.write(code)
    print(f"[inject] Wrote CSAES trainer → {dst_trainer}")

    init_path = os.path.join(repo_dir, "src", "trainer", "__init__.py")
    with open(init_path, "r") as f:
        init_content = f.read()

    if "from trainer.unlearn.c_saes import CSAES" not in init_content:
        with open(init_path, "a") as f:
            f.write(
                "\n# ── EXP_05: CSAES (Contrastive Sparse Autoencoder Suppression) ──\n"
                "from trainer.unlearn.c_saes import CSAES\n"
                "_register_trainer(CSAES)\n"
            )
        print(f"[inject] Registered CSAES in {init_path}")
    else:
        print(f"[inject] CSAES already registered, skipping.")

    yaml_path = os.path.join(repo_dir, "configs", "trainer", "CSAES.yaml")
    yaml_content = """\
defaults:
  - GradDiff     # kế thừa GradDiff defaults (alpha, gamma, retain_loss_type)

handler: CSAES
method_args:
  alpha: 1.0           # retain NLL weight
  gamma: 1.0           # forget gradient ascent weight
  target_layer: -8     # transformer layer for activation hook
  proj_coeff: 1.0      # weight of forget-direction suppression loss
  contrast_coeff: 0.5  # weight of retain-direction contrastive loss
  warmup_steps: 20     # steps to accumulate directions before suppression
  retain_loss_type: NLL
"""
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[inject] Created YAML config → {yaml_path}")


def build_unlearn_cmd(port, trainer, task_name, model_path, extra_overrides=""):
    return f"""CUDA_VISIBLE_DEVICES=0 accelerate launch \
--config_file configs/accelerate/kaggle_config.yaml \
--main_process_port {port} \
--num_processes 1 \
src/train.py --config-name=unlearn.yaml \
experiment=unlearn/tofu/default.yaml \
trainer={trainer} \
task_name={task_name} \
model={MODEL_NAME} \
forget_split={FORGET_SPLIT} \
retain_split={RETAIN_SPLIT} \
model.model_args.pretrained_model_name_or_path={model_path} \
model.tokenizer_args.pretrained_model_name_or_path={model_path} \
model.model_args.attn_implementation=flash_attention_2 \
retain_logs_path=saves/eval/tofu_{MODEL_NAME}_{RETAIN_SPLIT}/TOFU_EVAL.json \
trainer.args.per_device_train_batch_size={BATCH_SIZE} \
trainer.args.gradient_accumulation_steps={GRAD_ACCUM} \
trainer.args.optim=paged_adamw_32bit \
+trainer.args.max_steps={MAX_STEPS} \
+trainer.args.save_steps={MAX_STEPS} \
trainer.args.save_strategy=steps \
trainer.args.output_dir=saves/unlearn/{task_name} \
{extra_overrides}""".replace('\n', ' \\' + '\n    ')


def build_eval_cmd(port, task_name, model_path):
    return f"""CUDA_VISIBLE_DEVICES=0 accelerate launch \
--config_file configs/accelerate/kaggle_config.yaml \
--main_process_port {port} \
--num_processes 1 \
src/eval.py --config-name=eval.yaml \
task_name={task_name}_eval \
model={MODEL_NAME} \
model.model_args.pretrained_model_name_or_path={model_path} \
model.tokenizer_args.pretrained_model_name_or_path={model_path} \
model.model_args.attn_implementation=flash_attention_2 \
+forget_split={FORGET_SPLIT} \
+retain_split={RETAIN_SPLIT} \
+retain_logs_path=saves/eval/tofu_{MODEL_NAME}_{RETAIN_SPLIT}/TOFU_EVAL.json""".replace('\n', ' \\' + '\n    ')


def load_eval_result(repo_dir, task_name):
    """Load TOFU_EVAL.json from eval output dir."""
    path = os.path.join(repo_dir, "saves", "eval", f"{task_name}_eval", "TOFU_EVAL.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def extract_metrics(eval_result: dict) -> dict:
    """Extract the 4 key metrics from TOFU_EVAL.json."""
    metrics = {}
    flat = {}

    def _flatten(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten(v, prefix + k + ".")
            else:
                flat[prefix + k] = v

    _flatten(eval_result)

    # Map common TOFU metric keys to paper names
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

def print_ascii_table(results_map: dict):
    """Print a rich ASCII table of all results for quick inspection."""
    headers = ["Method", "model_utility", "forget_quality", "privleak", "extraction_strength", "Composite"]
    col_w = [28, 15, 15, 12, 20, 12]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    hdr = "|" + "|".join(f" {h:<{col_w[i]-1}}" for i, h in enumerate(headers)) + "|"

    print("\n" + "="*100)
    print("  EXP_05 — C-SAES Results Table")
    print("="*100)
    print(sep)
    print(hdr)
    print(sep)

    for label, metrics in results_map.items():
        mu   = metrics.get("model_utility",       float("nan"))
        fq   = metrics.get("forget_quality",      float("nan"))
        pl   = metrics.get("privleak",            float("nan"))
        es   = metrics.get("extraction_strength", float("nan"))
        # Composite: higher utility + higher forget_quality is better
        comp = (
            0.4 * (mu if mu == mu else 0)
            + 0.4 * (fq if fq == fq else 0)
            - 0.1 * abs(pl if pl == pl else 0)
            - 0.1 * (es if es == es else 0)
        )
        row_vals = [label, f"{mu:.4f}", f"{fq:.4f}", f"{pl:.2f}", f"{es:.4f}", f"{comp:.4f}"]
        print("|" + "|".join(f" {v:<{col_w[i]-1}}" for i, v in enumerate(row_vals)) + "|")

    print(sep)


def plot_radar_chart(results_map: dict, save_path: str = None):
    """Paper-ready radar chart: 4-axis (Utility, ForgetQuality, Privacy, Extraction)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[VIZ] matplotlib not available — skipping radar chart")
        return

    labels  = ["Model\nUtility", "Forget\nQuality", "Privacy\n(1-|privleak|/100)", "Extraction\nResistance"]
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

        # Normalize to [0, 1] for radar readability
        priv_score = max(0.0, 1.0 - abs(pl) / 100.0)  # higher = less leakage
        extr_score = max(0.0, 1.0 - es)                # higher = less extractable

        values = [
            min(max(mu, 0), 1),
            min(max(fq, 0), 1),
            min(max(priv_score, 0), 1),
            min(max(extr_score, 0), 1),
        ]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, color=colors[idx % len(colors)], label=label)
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.title("EXP_05 — C-SAES Method Comparison\n(Radar Chart)", size=13, pad=20)
    plt.tight_layout()

    if save_path is None:
        save_path = "/kaggle/working/EXP_05_radar.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Radar chart saved → {save_path}")


def plot_bar_chart(results_map: dict, save_path: str = None):
    """Bar chart: composite score per method with value annotations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[VIZ] matplotlib not available — skipping bar chart")
        return

    labels   = list(results_map.keys())
    composites = []
    for metrics in results_map.values():
        mu = metrics.get("model_utility",       0.0)
        fq = metrics.get("forget_quality",      0.0)
        pl = metrics.get("privleak",            0.0)
        es = metrics.get("extraction_strength", 0.0)
        c  = 0.4 * mu + 0.4 * fq - 0.1 * abs(pl) - 0.1 * es
        composites.append(round(c, 4))

    x = np.arange(len(labels))
    colors = plt.cm.viridis([i / max(len(labels) - 1, 1) for i in range(len(labels))])

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(x, composites, color=colors, width=0.6, edgecolor="black", linewidth=0.7)

    for bar, val in zip(bars, composites):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Composite Score\n(0.4·Utility + 0.4·ForgetQ - 0.1·|privleak| - 0.1·Extraction)", fontsize=9)
    ax.set_title("EXP_05 — C-SAES Configuration Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(min(0, min(composites)) - 0.05, max(composites) + 0.08)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is None:
        save_path = "/kaggle/working/EXP_05_bar.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Bar chart saved → {save_path}")


def plot_metric_breakdown(results_map: dict, save_path: str = None):
    """4-panel breakdown: one subplot per metric, all methods side by side."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[VIZ] matplotlib not available — skipping breakdown chart")
        return

    metrics_keys = ["model_utility", "forget_quality", "privleak", "extraction_strength"]
    titles       = ["Model Utility ↑", "Forget Quality ↑", "Privacy Leak ↓ (closer to 0)", "Extraction Strength ↓"]
    labels       = list(results_map.keys())
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    colors = plt.cm.tab10.colors

    for ax, metric, title in zip(axes, metrics_keys, titles):
        values = [results_map[lbl].get(metric, float("nan")) for lbl in labels]
        bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
        ax.bar(x, values, color=bar_colors, width=0.6, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for i, v in enumerate(values):
            if v == v:  # not nan
                ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("EXP_05 — C-SAES: Per-Metric Breakdown", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path is None:
        save_path = "/kaggle/working/EXP_05_breakdown.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Metric breakdown chart saved → {save_path}")


def print_latex_table(results_map: dict):
    """Print LaTeX table rows ready to paste into paper."""
    print("\n" + "─"*80)
    print("  LATEX TABLE (paste into paper \\begin{tabular})")
    print("─"*80)
    print("% Method & Model Utility & Forget Quality & PrivLeak & Extraction Strength \\\\")
    print("\\midrule")
    for label, metrics in results_map.items():
        mu   = metrics.get("model_utility",       float("nan"))
        fq   = metrics.get("forget_quality",      float("nan"))
        pl   = metrics.get("privleak",            float("nan"))
        es   = metrics.get("extraction_strength", float("nan"))
        name = label.replace("_", "\\_")
        print(f"C-SAES ({name}) & {mu:.4f} & {fq:.4f} & {pl:.2f} & {es:.4f} \\\\")
    print("─"*80)


def dump_results_json(results_map: dict, repo_dir: str):
    """Dump all results to JSON for later aggregation."""
    out_path = os.path.join(repo_dir, "saves", "eval", "EXP_05_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_map, f, indent=2)
    print(f"[JSON] Results saved → {out_path}")


def visualize_all(results_map: dict, repo_dir: str):
    """Run all visualization functions."""
    print("\n" + "="*100)
    print("  EXP_05 — VISUALIZATION & PAPER OUTPUT")
    print("="*100)
    print_ascii_table(results_map)
    plot_radar_chart(results_map, save_path="/kaggle/working/EXP_05_radar.png")
    plot_bar_chart(results_map, save_path="/kaggle/working/EXP_05_bar.png")
    plot_metric_breakdown(results_map, save_path="/kaggle/working/EXP_05_breakdown.png")
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

# Install matplotlib for visualization
run("pip install -q matplotlib numpy", check=False)

# ──────────────────────────────────────────────
# 3. Inject CSAES vào framework
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  Injecting C-SAES trainer into OpenUnlearning framework...")
print("  Method: Contrastive activation direction suppression")
print("="*60)
inject_csaes_into_framework(REPO_DIR)

MODEL_PATH  = f"open-unlearning/tofu_{MODEL_NAME}_full"
results_map = {}  # label → metrics dict

# ──────────────────────────────────────────────
# 4. Sweep C-SAES configurations
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  C-SAES Hyperparameter Sweep")
print("  L=γ·L_forget + proj·||A_f·d||² + α·L_retain - cont·||A_r·d||")
print("="*60)

for label, target_layer, proj_coeff, contrast_coeff in CSAES_CONFIGS:
    PORT = free_port()
    task_name = f"tofu_{MODEL_NAME}_{FORGET_SPLIT}_CSAES_{label}_exp05"
    print(f"\n  → C-SAES | {label} | layer={target_layer} proj={proj_coeff} contrast={contrast_coeff}")

    extra = (
        f"trainer.method_args.target_layer={target_layer} "
        f"trainer.method_args.proj_coeff={proj_coeff} "
        f"trainer.method_args.contrast_coeff={contrast_coeff} "
        f"trainer.method_args.warmup_steps={WARMUP_STEPS} "
        f"trainer.args.learning_rate=1e-5"
    )
    run(build_unlearn_cmd(PORT, "CSAES", task_name, MODEL_PATH, extra), cwd=REPO_DIR)
    run(build_eval_cmd(PORT, task_name, f"saves/unlearn/{task_name}"), cwd=REPO_DIR)

    eval_result = load_eval_result(REPO_DIR, task_name)
    metrics = extract_metrics(eval_result)
    results_map[f"CSAES_{label}"] = metrics
    print(f"  [METRICS] {label}: {metrics}")

# ──────────────────────────────────────────────
# 5. Visualize + Export
# ──────────────────────────────────────────────
visualize_all(results_map, REPO_DIR)

# ──────────────────────────────────────────────
# 6. Final summary
# ──────────────────────────────────────────────
print("\n" + "="*80)
print("  EXP_05 COMPLETE — C-SAES Results Summary")
print(f"  Split: {FORGET_SPLIT} / {RETAIN_SPLIT}  |  Steps: {MAX_STEPS}")
print("="*80)

print("""
METHOD DESCRIPTION (C-SAES):
  Instead of weight-space intervention (→ collapse), C-SAES works in
  ACTIVATION SPACE by learning a "forget direction" d = normalize(
    E[A(forget)] - E[A(retain)]
  ) then suppressing the projection of forget activations onto d.

  L_supp = mean(||A_forget · d||²)       ← push forget acts away from d
  L_cont = -mean(||A_retain · d||)       ← keep retain acts aligned with d
  (This contrastive term is KEY: ensures d stays forget-specific)

LAYERS:
  -4  = very late (close to logits) → aggressive suppression, risk of hallucination
  -8  = mid-late (recommended for 1B models)
  -12 = mid (more semantic, less syntactic)

EXPECTED PATTERN:
  Best layer is typically -8 to -12 for Llama-3 style models.
  proj_coeff=1.0 balanced vs 2.0 (aggressive) vs 0.5 (conservative).
  Look for: high forget_quality + high model_utility (both > 0.5).

PAPER OUTPUT:
  → /kaggle/working/EXP_05_radar.png       (radar chart — 4 axes)
  → /kaggle/working/EXP_05_bar.png         (composite score bar)
  → /kaggle/working/EXP_05_breakdown.png   (per-metric 4-panel)
  → saves/eval/EXP_05_results.json          (machine-readable)
""")
