# ============================================================
#  EXP_01 — Standard Methods Benchmark (Paper Reproduction)
#  HCMUS TheFangs | NeurIPS 2026 Unlearning
# ============================================================
#
#  MỤC ĐÍCH:
#    Reproduce kết quả Table 3 trong OpenUnlearning paper bằng
#    cách chạy 4 phương pháp chuẩn: GradDiff, NPO, SimNPO, RMU
#    trên TOFU forget10 với Llama-3.2-1B-Instruct.
#
#  CƠ SỞ LÝ THUYẾT (Sample_paper.md — Table 3):
#    - GradDiff : Gradient Difference — loss trên forget + retain
#    - NPO       : Negative Preference Optimization — dùng KL divergence
#                  từ reference model, geometry-aware hơn GradAscent
#    - SimNPO    : Simplified NPO — best performer trong paper
#                  (Agg=0.53, giữ utility=1.00)
#    - RMU       : Representation Misdirection Unlearning — strong
#                  memorization removal (Agg=0.52, Mem=0.47)
#
#  KẾT QUẢ MONG ĐỢI (theo paper):
#    SimNPO > RMU > GradDiff về composite score
#    GradDiff sẽ over-unlearn (Mem cao nhưng utility drop)
#
#  CÁCH CHẠY: Copy toàn bộ file vào 1 cell Kaggle và Run
# ============================================================

import os
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

# forget10 = forget fraction lớn nhất, paper dùng để benchmark chính
FORGET_SPLIT = "forget10"
RETAIN_SPLIT = "retain90"

# Các methods cần so sánh — (trainer_name, experiment_yaml)
METHODS = [
    ("GradDiff",  "unlearn/tofu/default.yaml"),
    ("NPO",       "unlearn/tofu/default.yaml"),
    ("SimNPO",    "unlearn/tofu/default.yaml"),
    ("RMU",       "unlearn/tofu/default.yaml"),
]

# Hyperparams — dùng paper defaults để reproduce
BATCH_SIZE = 4
GRAD_ACCUM = 4   # effective = 16
MAX_STEPS  = 200

# Per-method LR (theo paper hyperparams AppendixF)
METHOD_LR = {
    "GradDiff": "1e-5",
    "NPO":      "1e-5",
    "SimNPO":   "1e-5",
    "RMU":      "5e-5",
}

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
# 3. Main loop — chạy từng method
# ──────────────────────────────────────────────
MODEL_PATH = f"open-unlearning/tofu_{MODEL_NAME}_full"
results    = {}

for trainer, experiment in METHODS:
    PORT = free_port()
    lr   = METHOD_LR.get(trainer, "1e-5")

    print(f"\n{'='*60}")
    print(f"  EXP_01 | {trainer} | {FORGET_SPLIT} | lr={lr}")
    print(f"{'='*60}")

    task_name = f"tofu_{MODEL_NAME}_{FORGET_SPLIT}_{trainer}_exp01"

    # ── 3a. Unlearn ──────────────────────────────
    unlearn_cmd = f"""
CUDA_VISIBLE_DEVICES=0 accelerate launch \\
    --config_file configs/accelerate/kaggle_config.yaml \\
    --main_process_port {PORT} \\
    --num_processes 1 \\
    src/train.py --config-name=unlearn.yaml \\
    experiment={experiment} \\
    trainer={trainer} \\
    task_name={task_name} \\
    model={MODEL_NAME} \\
    forget_split={FORGET_SPLIT} \\
    retain_split={RETAIN_SPLIT} \\
    model.model_args.pretrained_model_name_or_path={MODEL_PATH} \\
    model.tokenizer_args.pretrained_model_name_or_path={MODEL_PATH} \\
    model.model_args.attn_implementation=eager \\
    retain_logs_path=saves/eval/tofu_{MODEL_NAME}_{RETAIN_SPLIT}/TOFU_EVAL.json \\
    trainer.args.per_device_train_batch_size={BATCH_SIZE} \\
    trainer.args.gradient_accumulation_steps={GRAD_ACCUM} \\
    trainer.args.optim=adamw_torch \\
    trainer.args.learning_rate={lr} \\
    +trainer.args.max_steps={MAX_STEPS} \\
    +trainer.args.save_steps={MAX_STEPS} \\
    trainer.args.save_strategy=steps \\
    trainer.args.output_dir=saves/unlearn/{task_name}
""".strip()
    run(unlearn_cmd, cwd=REPO_DIR)

    # ── 3b. Eval ─────────────────────────────────
    eval_cmd = f"""
CUDA_VISIBLE_DEVICES=0 accelerate launch \\
    --config_file configs/accelerate/kaggle_config.yaml \\
    --main_process_port {PORT} \\
    --num_processes 1 \\
    src/eval.py --config-name=eval.yaml \\
    task_name={task_name}_eval \\
    model={MODEL_NAME} \\
    model.model_args.pretrained_model_name_or_path=saves/unlearn/{task_name} \\
    model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/{task_name} \\
    model.model_args.attn_implementation=eager \\
    +forget_split={FORGET_SPLIT} \\
    +retain_split={RETAIN_SPLIT} \\
    +retain_logs_path=saves/eval/tofu_{MODEL_NAME}_{RETAIN_SPLIT}/TOFU_EVAL.json
""".strip()
    run(eval_cmd, cwd=REPO_DIR)
    results[trainer] = task_name

# ──────────────────────────────────────────────
# 4. Summary
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  EXP_01 DONE — Standard Methods Benchmark")
print(f"  Split: {FORGET_SPLIT} / {RETAIN_SPLIT}")
print("="*60)
for method, task in results.items():
    print(f"  {method:12s} → {REPO_DIR}/saves/eval/{task}_eval/")

print("""
PAPER REFERENCE (Table 3, forget10, Llama-3.2-1B):
  SimNPO  Agg=0.53  Mem=0.32  Priv=0.63  Utility=1.00
  RMU     Agg=0.52  Mem=0.47  Priv=0.50  Utility=0.61
  GradDiff Agg=9e-3 Mem=0.97  Priv=3e-3  Utility=0.79
So sánh số của bạn với bảng trên để verify reproduction.
""")

# ══════════════════════════════════════════════════════════════
#  VISUALIZATION — Paper-ready output
# ══════════════════════════════════════════════════════════════
import json as _json

def _load_metrics(repo_dir, task_name):
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

metrics_map = {method: _load_metrics(REPO_DIR, task) for method, task in results.items()}

# Paper reference values for overlay
PAPER_REF = {
    "SimNPO (paper)": {"model_utility": 1.00, "forget_quality": 0.53, "privleak": -0.63*100, "extraction_strength": 0.0},
    "RMU (paper)":    {"model_utility": 0.61, "forget_quality": 0.52, "privleak": -0.50*100, "extraction_strength": 0.0},
}

# ── ASCII table ───────────────────────────────────────────────
_COL_W = [28, 15, 15, 12, 20, 12]
_HEADERS = ["Method", "model_utility", "forget_quality", "privleak", "extraction_strength", "Composite"]
_SEP = "+" + "+".join("-" * w for w in _COL_W) + "+"
print("\n" + "=" * 104)
print("  EXP_01 — Standard Methods Benchmark (TOFU forget10)")
print("=" * 104)
print(_SEP)
print("|" + "|".join(f" {h:<{_COL_W[i]-1}}" for i, h in enumerate(_HEADERS)) + "|")
print(_SEP)
for label, m in {**metrics_map, **PAPER_REF}.items():
    mu=m.get("model_utility",float("nan")); fq=m.get("forget_quality",float("nan"))
    pl=m.get("privleak",float("nan")); es=m.get("extraction_strength",float("nan"))
    comp = 0.4*(mu if mu==mu else 0)+0.4*(fq if fq==fq else 0)-0.1*abs(pl if pl==pl else 0)-0.1*(es if es==es else 0)
    row = [label, f"{mu:.4f}", f"{fq:.4f}", f"{pl:.2f}", f"{es:.4f}", f"{comp:.4f}"]
    print("|" + "|".join(f" {v:<{_COL_W[i]-1}}" for i, v in enumerate(row)) + "|")
print(_SEP)

# ── LaTeX rows ────────────────────────────────────────────────
print("\n% LATEX TABLE ROWS (EXP_01):")
for label, m in metrics_map.items():
    mu=m.get("model_utility",float("nan")); fq=m.get("forget_quality",float("nan"))
    pl=m.get("privleak",float("nan")); es=m.get("extraction_strength",float("nan"))
    print(f"{label.replace('_', chr(92)+'_')} & {mu:.4f} & {fq:.4f} & {pl:.2f} & {es:.4f} \\\\")

# ── Radar + Bar chart ─────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    all_map = {**metrics_map, **PAPER_REF}
    _labels4 = ["Model\nUtility", "Forget\nQuality", "Privacy\n(1-|pl|/100)", "Extraction\nResist."]
    _ang = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist(); _ang += _ang[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    _colors = plt.cm.tab10.colors
    line_styles = ["-", "--"]
    for ci, (lbl, m) in enumerate(all_map.items()):
        mu=m.get("model_utility",0.5); fq=m.get("forget_quality",0.5)
        pl=m.get("privleak",0); es=m.get("extraction_strength",0.5)
        vals = [min(max(mu,0),1), min(max(fq,0),1), min(max(1-abs(pl)/100,0),1), min(max(1-es,0),1)]
        vals += vals[:1]
        ls = "--" if "paper" in lbl else "-"
        ax.plot(_ang, vals, f"o{ls}", linewidth=2, color=_colors[ci%10], label=lbl, alpha=0.9)
        ax.fill(_ang, vals, alpha=0.07, color=_colors[ci%10])
    ax.set_xticks(_ang[:-1]); ax.set_xticklabels(_labels4, size=10)
    ax.set_ylim(0, 1); ax.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=8)
    plt.title("EXP_01 — Standard Methods vs Paper Reference\n(Radar Chart)", size=12, pad=20)
    plt.tight_layout()
    plt.savefig("/kaggle/working/EXP_01_radar.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[VIZ] Radar chart → /kaggle/working/EXP_01_radar.png")

    # 4-panel metric breakdown
    _mkeys = ["model_utility","forget_quality","privleak","extraction_strength"]
    _mtitles = ["Model Utility ↑","Forget Quality ↑","PrivLeak ↓","Extraction Strength ↓"]
    _labs = list(metrics_map.keys()); _x = np.arange(len(_labs))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, mkey, mtitle in zip(axes, _mkeys, _mtitles):
        vals = [metrics_map[l].get(mkey, float("nan")) for l in _labs]
        bc = [_colors[i % 10] for i in range(len(_labs))]
        ax.bar(_x, vals, color=bc, width=0.6, edgecolor="black", linewidth=0.5)
        ax.set_xticks(_x); ax.set_xticklabels(_labs, rotation=30, ha="right", fontsize=8)
        ax.set_title(mtitle, fontsize=10, fontweight="bold"); ax.grid(axis="y", linestyle="--", alpha=0.3)
        for i, v in enumerate(vals):
            if v == v: ax.text(i, v+0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("EXP_01 — Per-Metric Breakdown", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/kaggle/working/EXP_01_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[VIZ] Breakdown chart → /kaggle/working/EXP_01_breakdown.png")
except Exception as e:
    print(f"[VIZ] Visualization skipped: {e}")

# ── JSON dump ─────────────────────────────────────────────────
_out_path = os.path.join(REPO_DIR, "saves", "eval", "EXP_01_results.json")
os.makedirs(os.path.dirname(_out_path), exist_ok=True)
with open(_out_path, "w") as f:
    _json.dump(metrics_map, f, indent=2)
print(f"[JSON] Results → {_out_path}")
print("\n[EXP_01] Paper output complete.")
