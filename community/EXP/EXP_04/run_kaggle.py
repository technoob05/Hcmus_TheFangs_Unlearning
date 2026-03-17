# ============================================================
#  EXP_04 — ER-BWP: Entropy-Reservoir BW Projection (NOVEL)
#  HCMUS TheFangs | NeurIPS 2026 Unlearning
# ============================================================
#
#  MỤC ĐÍCH:
#    Implement và chạy ER-BWP (Entropy-Reservoir Bregman-Wasserstein
#    Projection) — phương pháp MỚI được thiết kế để giải quyết
#    "representation collapse" trong sequential unlearning.
#
#  METHOD (trainer.py — ERBWP class):
#    Loss = γ·L_NPO(forget) + α·L_retain(NLL) + λ(t)·L_entropy(forget)
#
#    L_entropy(forget) = -H(p_θ(·|forget)) = maximize entropy của
#    token-level predictions trên forget set → push model toward
#    "maximal entropy prior" (uncertainty about forgotten data).
#
#    λ(t) = cosine decay từ λ_max → λ_min theo training step.
#    Entropy coupling mạnh ở đầu (prevent collapse), yếu dần sau
#    khi model đã stable (allow targeted forgetting to deepen).
#
#  CƠ SỞ LÝ THUYẾT (PreRound_1.md — Alternative 1: ER-BWP):
#    "continuous entropy flux provably stabilizes the closed-loop
#     dynamics of sequential unlearning"
#    "eliminates the need for strict inverse-Hessian-vector product
#     computations by utilizing first-order entropy approximations"
#
#  PHẦN B: PDU và SatImp — decomposition/stopping-based methods
#    như là comparison baseline cho ER-BWP.
#
#  KẾT QUẢ MONG ĐỢI:
#    ERBWP (lambda_max=0.5, cosine decay) sẽ:
#    - utility cao hơn GradAscent (EXP_00) vì no collapse
#    - forgetting tốt hơn SimNPO có entropy term
#    - sequential robustness: chạy lại từ checkpoint → stable
#
#  CÁCH CHẠY: Copy toàn bộ file vào 1 cell Kaggle và Run
#    Script sẽ tự copy trainer.py vào framework và register ERBWP.
# ============================================================

import os
import shutil
import subprocess
import sys
import socket

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

# ── Part A: ER-BWP hyperparameter sweep ─────────────────────
# Sweep lambda_max (strength of entropy reservoir coupling)
# và lambda_decay (speed at which coupling fades)
ERBWP_CONFIGS = [
    # (label,        lambda_max, lambda_min, lambda_decay)
    ("lmax0.1_d1.0", 0.1,       0.001,      1.0),   # weak coupling
    ("lmax0.5_d1.0", 0.5,       0.001,      1.0),   # medium coupling (default)
    ("lmax1.0_d1.0", 1.0,       0.001,      1.0),   # strong coupling
    ("lmax0.5_d0.5", 0.5,       0.001,      0.5),   # fast decay
]

# ── Part B: Comparison baselines ────────────────────────────
# PDU + SatImp: decomposition/stopping-based methods
COMPARISON_METHODS = [
    (
        "PDU",
        "unlearn/tofu/default.yaml",
        "trainer.args.learning_rate=1e-5",
        "Parameter Decomposition Unlearning"
    ),
    (
        "SatImp",
        "unlearn/tofu/default.yaml",
        "trainer.args.learning_rate=1e-5",
        "Satisfaction Improvement stopping criterion"
    ),
]

# ── Nội dung trainer.py (ER-BWP) — đọc từ file cùng thư mục ─
# Khi upload lên Kaggle, dán nội dung trainer.py vào ERBWP_CODE bên dưới
# HOẶC để script tự đọc nếu chạy từ local → Kaggle via dataset
ERBWP_TRAINER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "trainer.py"
)

# Fallback: inline code nếu file không tồn tại (khi chạy standalone trên Kaggle)
ERBWP_CODE = r'''
import logging
import math

import torch
import torch.nn.functional as F

from trainer.utils import compute_dpo_loss
from trainer.unlearn.npo import NPO

logger = logging.getLogger(__name__)


class ERBWP(NPO):
    """
    Entropy-Reservoir Bregman-Wasserstein Projection (ER-BWP).

    Loss = gamma * L_NPO(forget) + alpha * L_retain(NLL) + lambda(t) * L_entropy(forget)

    L_entropy(forget) = -H(p_theta(.|forget)) — maximize entropy = push toward uniform.
    lambda(t) = cosine decay from lambda_max to lambda_min.

    Reference: PreRound_1.md, Alternative 1: ER-BWP
    """

    def __init__(self, lambda_max=0.5, lambda_min=0.001, lambda_decay=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.lambda_decay = lambda_decay

    def _get_lambda(self):
        step = self.state.global_step if self.state is not None else 0
        max_steps = max(int(self.args.max_steps), 1)
        t_decay = max(self.lambda_decay * max_steps, 1)
        progress = min(step / t_decay, 1.0)
        lam = self.lambda_min + 0.5 * (self.lambda_max - self.lambda_min) * (
            1.0 + math.cos(math.pi * progress)
        )
        return float(lam)

    def _compute_entropy_loss(self, model, forget_inputs):
        outputs = model(
            input_ids=forget_inputs["input_ids"],
            attention_mask=forget_inputs["attention_mask"],
        )
        logits = outputs.logits  # [B, T, V]
        probs = F.softmax(logits, dim=-1)
        token_entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1)  # [B, T]
        labels = forget_inputs.get("labels")
        if labels is not None:
            valid_mask = (labels != -100).float()
        else:
            valid_mask = forget_inputs["attention_mask"].float()
        denom = valid_mask.sum().clamp(min=1.0)
        mean_entropy = (token_entropy * valid_mask).sum() / denom
        return -mean_entropy  # minimize -H = maximize H

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = inputs["forget"]
        lam = self._get_lambda()

        # 1. NPO forget loss
        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )

        # 2. Retain NLL
        retain_inputs = inputs["retain"]
        retain_clean = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_clean)

        # 3. Entropy reservoir
        forget_clean = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs.get("labels"),
        }
        entropy_loss = self._compute_entropy_loss(model, forget_clean)

        loss = self.gamma * forget_loss + self.alpha * retain_loss + lam * entropy_loss

        if self.state is not None and self.state.global_step % 10 == 0:
            logger.info(
                f"[ERBWP] step={self.state.global_step:4d} | lam={lam:.4f} | "
                f"forget={forget_loss.item():.4f} | retain={retain_loss.item():.4f} | "
                f"entropy={entropy_loss.item():.4f} | total={loss.item():.4f}"
            )

        return (loss, forget_outputs) if return_outputs else loss
'''

# ──────────────────────────────────────────────
# 0. Helper
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
        raise RuntimeError(
            f"FAILED (exit {result.returncode}): {cmd}\n{out[-2000:]}"
        )
    return result

def free_port():
    s = socket.socket(); s.bind(('', 0))
    p = s.getsockname()[1]; s.close()
    return p

def inject_erbwp_into_framework(repo_dir):
    """
    Inject ERBWP trainer vào OpenUnlearning framework:
      1. Copy er_bwp.py vào src/trainer/unlearn/
      2. Append import + _register_trainer(ERBWP) vào src/trainer/__init__.py
      3. Tạo configs/trainer/ERBWP.yaml

    Idempotent: chạy nhiều lần không bị duplicate.
    """
    # ── 1. Ghi er_bwp.py ───────────────────────────────────────
    dst_trainer = os.path.join(repo_dir, "src", "trainer", "unlearn", "er_bwp.py")
    if os.path.exists(ERBWP_TRAINER_PATH):
        # Chạy local: đọc từ file thật
        with open(ERBWP_TRAINER_PATH, "r") as f:
            code = f.read()
    else:
        # Chạy standalone Kaggle: dùng inline code
        code = ERBWP_CODE
    with open(dst_trainer, "w") as f:
        f.write(code)
    print(f"[inject] Wrote ERBWP trainer → {dst_trainer}")

    # ── 2. Append vào __init__.py (chỉ một lần) ─────────────────
    init_path = os.path.join(repo_dir, "src", "trainer", "__init__.py")
    with open(init_path, "r") as f:
        init_content = f.read()

    if "from trainer.unlearn.er_bwp import ERBWP" not in init_content:
        append_lines = (
            "\n# ── EXP_04: ERBWP (Entropy-Reservoir BW Projection) ──\n"
            "from trainer.unlearn.er_bwp import ERBWP\n"
            "_register_trainer(ERBWP)\n"
        )
        with open(init_path, "a") as f:
            f.write(append_lines)
        print(f"[inject] Registered ERBWP in {init_path}")
    else:
        print(f"[inject] ERBWP already registered, skipping.")

    # ── 3. Tạo YAML config ───────────────────────────────────────
    yaml_path = os.path.join(repo_dir, "configs", "trainer", "ERBWP.yaml")
    yaml_content = """\
defaults:
  - NPO      # kế thừa NPO defaults (beta, alpha, gamma, retain_loss_type)

handler: ERBWP
method_args:
  beta: 0.1          # NPO temperature
  alpha: 1.0         # retain loss weight
  gamma: 1.0         # forget loss weight
  lambda_max: 0.5    # entropy reservoir max coupling
  lambda_min: 0.001  # entropy reservoir min coupling
  lambda_decay: 1.0  # cosine decay over lambda_decay * max_steps
  retain_loss_type: NLL
"""
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[inject] Created YAML config → {yaml_path}")

def build_unlearn_cmd(port, trainer, experiment, task_name, model_path, extra_overrides=""):
    return f"""
CUDA_VISIBLE_DEVICES=0 accelerate launch \\
    --config_file configs/accelerate/kaggle_config.yaml \\
    --main_process_port {port} \\
    --num_processes 1 \\
    src/train.py --config-name=unlearn.yaml \\
    experiment={experiment} \\
    trainer={trainer} \\
    task_name={task_name} \\
    model={MODEL_NAME} \\
    forget_split={FORGET_SPLIT} \\
    retain_split={RETAIN_SPLIT} \\
    model.model_args.pretrained_model_name_or_path={model_path} \\
    model.tokenizer_args.pretrained_model_name_or_path={model_path} \\
    model.model_args.attn_implementation=flash_attention_2 \\
    retain_logs_path=saves/eval/tofu_{MODEL_NAME}_{RETAIN_SPLIT}/TOFU_EVAL.json \\
    trainer.args.per_device_train_batch_size={BATCH_SIZE} \\
    trainer.args.gradient_accumulation_steps={GRAD_ACCUM} \\
    trainer.args.optim=paged_adamw_32bit \\
    +trainer.args.max_steps={MAX_STEPS} \\
    +trainer.args.save_steps={MAX_STEPS} \\
    trainer.args.save_strategy=steps \\
    trainer.args.output_dir=saves/unlearn/{task_name} \\
    {extra_overrides}
""".strip()

def build_eval_cmd(port, task_name, model_path):
    return f"""
CUDA_VISIBLE_DEVICES=0 accelerate launch \\
    --config_file configs/accelerate/kaggle_config.yaml \\
    --main_process_port {port} \\
    --num_processes 1 \\
    src/eval.py --config-name=eval.yaml \\
    task_name={task_name}_eval \\
    model={MODEL_NAME} \\
    model.model_args.pretrained_model_name_or_path={model_path} \\
    model.tokenizer_args.pretrained_model_name_or_path={model_path} \\
    model.model_args.attn_implementation=flash_attention_2 \\
    +forget_split={FORGET_SPLIT} \\
    +retain_split={RETAIN_SPLIT} \\
    +retain_logs_path=saves/eval/tofu_{MODEL_NAME}_{RETAIN_SPLIT}/TOFU_EVAL.json
""".strip()

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

# ──────────────────────────────────────────────
# 3. Inject ERBWP vào framework
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  Injecting ERBWP trainer into OpenUnlearning framework...")
print("="*60)
inject_erbwp_into_framework(REPO_DIR)

MODEL_PATH = f"open-unlearning/tofu_{MODEL_NAME}_full"
results    = {}

# ──────────────────────────────────────────────
# 4. Part A — ER-BWP hyperparameter sweep
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  PART A: ER-BWP Lambda Sweep (Novel Method)")
print("  Arch: L = γ·L_NPO + α·L_retain + λ(t)·L_entropy")
print("="*60)

for label, lambda_max, lambda_min, lambda_decay in ERBWP_CONFIGS:
    PORT = free_port()
    task_name = f"tofu_{MODEL_NAME}_{FORGET_SPLIT}_ERBWP_{label}_exp04"
    print(f"\n  → ERBWP | {label} | λ_max={lambda_max} λ_min={lambda_min} decay={lambda_decay}")

    extra = (
        f"trainer.method_args.lambda_max={lambda_max} "
        f"trainer.method_args.lambda_min={lambda_min} "
        f"trainer.method_args.lambda_decay={lambda_decay} "
        f"trainer.args.learning_rate=1e-5"
    )
    run(build_unlearn_cmd(PORT, "ERBWP", "unlearn/tofu/default.yaml",
                          task_name, MODEL_PATH, extra), cwd=REPO_DIR)
    run(build_eval_cmd(PORT, task_name, f"saves/unlearn/{task_name}"), cwd=REPO_DIR)
    results[f"ERBWP_{label}"] = task_name

# ──────────────────────────────────────────────
# 5. Part B — Comparison: PDU + SatImp
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  PART B: Comparison Methods (PDU, SatImp)")
print("="*60)

for trainer, experiment, extra_overrides, desc in COMPARISON_METHODS:
    PORT = free_port()
    task_name = f"tofu_{MODEL_NAME}_{FORGET_SPLIT}_{trainer}_exp04"
    print(f"\n  → {trainer}: {desc}")

    run(build_unlearn_cmd(PORT, trainer, experiment,
                          task_name, MODEL_PATH, extra_overrides), cwd=REPO_DIR)
    run(build_eval_cmd(PORT, task_name, f"saves/unlearn/{task_name}"), cwd=REPO_DIR)
    results[trainer] = task_name

# ──────────────────────────────────────────────
# 6. Summary
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  EXP_04 DONE — ER-BWP Novel Method + Comparisons")
print(f"  Split: {FORGET_SPLIT} / {RETAIN_SPLIT}")
print("="*60)
for label, task in results.items():
    print(f"  {label:30s} → {REPO_DIR}/saves/eval/{task}_eval/")

print(f"""
NOVEL METHOD — ER-BWP (trainer.py → ERBWP class):
  Loss = γ·L_NPO(forget) + α·L_retain + λ(t)·L_entropy(forget)
  λ(t) = cosine decay from lambda_max → lambda_min

  Cơ chế: entropy maximization trên forget set phòng ngừa
  "representation collapse" (PreRound_1: "exponential decay in entropy
  causes dimensional collapse of latent space").

SO SÁNH:
  ERBWP_lmax0.1  : weak entropy coupling → behavior gần NPO nhất
  ERBWP_lmax0.5  : medium (paper recommended)
  ERBWP_lmax1.0  : strong → entropy dominates, có thể over-regularize
  ERBWP_lmax0.5_d0.5: fast decay → entropy effect chỉ ở đầu

  PDU   : weight decomposition (no entropy)
  SatImp: stopping criterion (no entropy)

Target: ERBWP_lmax0.5 vượt SimNPO về model_utility trong khi giữ
được forgetting quality tương đương.
""")

# ══════════════════════════════════════════════════════════════
#  VISUALIZATION — Paper-ready output
# ══════════════════════════════════════════════════════════════
import json as _json

def _load_metrics04(repo_dir, task_name):
    path = os.path.join(repo_dir, "saves", "eval", f"{task_name}_eval", "TOFU_EVAL.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = _json.load(f)
    flat = {}
    def _flatten(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict): _flatten(v, prefix + k + ".")
            else: flat[prefix + k] = v
    _flatten(raw)
    key_map = {
        "model_utility": ["model_utility", "utility"],
        "forget_quality": ["forget_quality", "forget_qual"],
        "privleak": ["privleak", "privacy_leakage"],
        "extraction_strength": ["extraction_strength", "extraction"],
    }
    out = {}
    for pk, cands in key_map.items():
        for c in cands:
            for k, v in flat.items():
                if c.lower() in k.lower() and isinstance(v, (int, float)):
                    out[pk] = round(float(v), 4); break
            if pk in out: break
    return out

metrics_map = {label: _load_metrics04(REPO_DIR, task) for label, task in results.items()}

# ── ASCII table ───────────────────────────────────────────────
_COL_W = [30, 15, 15, 12, 20, 12]
_HEADERS = ["Method", "model_utility", "forget_quality", "privleak", "extraction_strength", "Composite"]
_SEP = "+" + "+".join("-" * w for w in _COL_W) + "+"
print("\n" + "=" * 106)
print("  EXP_04 — ER-BWP Novel Method: Results")
print("=" * 106)
print(_SEP)
print("|" + "|".join(f" {h:<{_COL_W[i]-1}}" for i, h in enumerate(_HEADERS)) + "|")
print(_SEP)
for label, m in metrics_map.items():
    mu=m.get("model_utility",float("nan")); fq=m.get("forget_quality",float("nan"))
    pl=m.get("privleak",float("nan")); es=m.get("extraction_strength",float("nan"))
    comp = 0.4*(mu if mu==mu else 0)+0.4*(fq if fq==fq else 0)-0.1*abs(pl if pl==pl else 0)-0.1*(es if es==es else 0)
    row = [label, f"{mu:.4f}", f"{fq:.4f}", f"{pl:.2f}", f"{es:.4f}", f"{comp:.4f}"]
    print("|" + "|".join(f" {v:<{_COL_W[i]-1}}" for i, v in enumerate(row)) + "|")
print(_SEP)

print("\n% LATEX TABLE ROWS (EXP_04):")
for label, m in metrics_map.items():
    mu=m.get("model_utility",float("nan")); fq=m.get("forget_quality",float("nan"))
    pl=m.get("privleak",float("nan")); es=m.get("extraction_strength",float("nan"))
    print(f"ER-BWP ({label.replace('_', chr(92)+'_')}) & {mu:.4f} & {fq:.4f} & {pl:.2f} & {es:.4f} \\\\")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Radar chart
    _labels4 = ["Model\nUtility", "Forget\nQuality", "Privacy\n(1-|pl|/100)", "Extraction\nResist."]
    _ang = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist(); _ang += _ang[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    _colors = plt.cm.tab10.colors
    for ci, (lbl, m) in enumerate(metrics_map.items()):
        mu=m.get("model_utility",0.5); fq=m.get("forget_quality",0.5)
        pl=m.get("privleak",0); es=m.get("extraction_strength",0.5)
        vals = [min(max(mu,0),1), min(max(fq,0),1), min(max(1-abs(pl)/100,0),1), min(max(1-es,0),1)]
        vals += vals[:1]
        # Highlight ERBWP_lmax0.5_d1.0 (recommended) with thicker line
        lw = 3.5 if "lmax0.5_d1" in lbl else 2
        ax.plot(_ang, vals, "o-", linewidth=lw, color=_colors[ci%10], label=lbl)
        ax.fill(_ang, vals, alpha=0.1, color=_colors[ci%10])
    ax.set_xticks(_ang[:-1]); ax.set_xticklabels(_labels4, size=11)
    ax.set_ylim(0, 1); ax.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=8)
    plt.title("EXP_04 — ER-BWP Lambda Sweep\n(Novel Method: entropy reservoir)", size=12, pad=20)
    plt.tight_layout()
    plt.savefig("/kaggle/working/EXP_04_radar.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[VIZ] Radar chart → /kaggle/working/EXP_04_radar.png")

    # Lambda sensitivity: group ERBWP by lambda_max, plot utility & forget_quality
    erbwp_data = {k: v for k, v in metrics_map.items() if k.startswith("ERBWP_")}
    _labs = list(erbwp_data.keys()); _x = np.arange(len(_labs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mkey, mtitle, color in zip(axes,
            ["model_utility", "forget_quality"],
            ["Model Utility ↑", "Forget Quality ↑"],
            ["#4CAF50", "#2196F3"]):
        vals = [erbwp_data[l].get(mkey, float("nan")) for l in _labs]
        bars = ax.bar(_x, vals, color=color, width=0.6, edgecolor="black", alpha=0.85)
        for b, v in zip(bars, vals):
            if v == v: ax.text(b.get_x()+b.get_width()/2., v+0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(_x); ax.set_xticklabels(_labs, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"ER-BWP Lambda Sweep: {mtitle}", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("/kaggle/working/EXP_04_lambda_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[VIZ] Lambda sensitivity → /kaggle/working/EXP_04_lambda_sensitivity.png")
except Exception as e:
    print(f"[VIZ] Visualization skipped: {e}")

_out_path = os.path.join(REPO_DIR, "saves", "eval", "EXP_04_results.json")
os.makedirs(os.path.dirname(_out_path), exist_ok=True)
with open(_out_path, "w") as f:
    _json.dump(metrics_map, f, indent=2)
print(f"[JSON] Results → {_out_path}")
print("\n[EXP_04] Paper output complete.")
