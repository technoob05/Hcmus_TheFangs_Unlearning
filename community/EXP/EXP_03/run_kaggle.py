# ============================================================
#  EXP_03 — Geometry-Aware Community Methods
#  HCMUS TheFangs | NeurIPS 2026 Unlearning
# ============================================================
#
#  MỤC ĐÍCH:
#    Đánh giá các community methods có thiết kế geometry-aware
#    hoặc influence-function-based: WGA, UNDIAL, AltPO.
#    Đây là approximation gần nhất với BWGP framework trong
#    OpenUnlearning hiện tại.
#
#  CƠ SỞ LÝ THUYẾT (PreRound_2.md):
#    - WGA (Weight-Guided Ascent): cân nhắc curvature của weight
#      space khi ascent → gần với "weight-centric geometry"
#    - UNDIAL: dùng influence function để xác định contribution
#      của từng training sample → gần với LANCET/LinFAC approach
#      (PreRound_2: "influence functions to attribute model behavior")
#    - AltPO (Alternative Preference Optimization): alternative
#      data approach, giảm thiểu language collapse bằng cách dùng
#      IDK-style responses thay vì direct gradient ascent
#
#  KẾT QUẢ MONG ĐỢI:
#    WGA/UNDIAL sẽ bảo toàn utility tốt hơn GradAscent (EXP_00)
#    vì geometry-aware. AltPO sẽ có privleak tốt nhất trong nhóm
#    (AltPO paper: Agg=0.15 nhưng Mem=0.63, utility=0.95).
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

FORGET_SPLIT = "forget10"
RETAIN_SPLIT = "retain90"

# Community geometry-aware methods
# (trainer_name, experiment_yaml, extra_hydra_overrides)
METHODS = [
    (
        "WGA",
        "unlearn/tofu/default.yaml",
        # WGA hyperparams: beta1=1.0, alpha=1.0, lr=1e-5
        "trainer.args.learning_rate=1e-5 "
        "trainer.method_args.beta1=1.0 "
        "trainer.method_args.alpha=1.0"
    ),
    (
        "UNDIAL",
        "unlearn/tofu/default.yaml",
        # UNDIAL hyperparams: alpha=2, beta=10, lr=1e-5
        "trainer.args.learning_rate=1e-5 "
        "trainer.method_args.alpha=2 "
        "trainer.method_args.beta=10"
    ),
    (
        "DPO",   # AltPO uses DPO trainer with generate.yaml data
        "unlearn/tofu/default.yaml",
        # AltPO/DPO: beta=0.1, alpha=2, lr=1e-5
        "trainer.args.learning_rate=1e-5 "
        "trainer.method_args.beta=0.1 "
        "trainer.method_args.alpha=2"
    ),
]

BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_STEPS  = 200

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
# 3. Main loop
# ──────────────────────────────────────────────
MODEL_PATH = f"open-unlearning/tofu_{MODEL_NAME}_full"
results    = {}

for trainer, experiment, extra_overrides in METHODS:
    PORT = free_port()
    label = trainer  # dùng làm key

    print(f"\n{'='*60}")
    print(f"  EXP_03 | {trainer} | {FORGET_SPLIT}")
    print(f"{'='*60}")

    task_name = f"tofu_{MODEL_NAME}_{FORGET_SPLIT}_{trainer}_exp03"

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
    +trainer.args.max_steps={MAX_STEPS} \\
    +trainer.args.save_steps={MAX_STEPS} \\
    trainer.args.save_strategy=steps \\
    trainer.args.output_dir=saves/unlearn/{task_name} \\
    {extra_overrides}
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
    results[label] = task_name

# ──────────────────────────────────────────────
# 4. Summary
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  EXP_03 DONE — Geometry-Aware Community Methods")
print(f"  Split: {FORGET_SPLIT} / {RETAIN_SPLIT}")
print("="*60)
for method, task in results.items():
    print(f"  {method:10s} → {REPO_DIR}/saves/eval/{task}_eval/")

print("""
PHÂN TÍCH SO SÁNH:
  WGA   : Weight-guided curvature → utility preserved?
  UNDIAL: Influence-function based → surgical forgetting?
  DPO   : Alternative preference (AltPO style) → privacy?

  Reference: PreRound_2.md — BWGP "weight-centric → representation-centric"
  Reference: PreRound_2.md — LANCET/SCRUB influence function approach
""")

# ══════════════════════════════════════════════════════════════
#  VISUALIZATION — Paper-ready output
# ══════════════════════════════════════════════════════════════
import json as _json

def _load_metrics03(repo_dir, task_name):
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

metrics_map = {method: _load_metrics03(REPO_DIR, task) for method, task in results.items()}

# ── ASCII table ───────────────────────────────────────────────
_COL_W = [26, 15, 15, 12, 20, 12]
_HEADERS = ["Method", "model_utility", "forget_quality", "privleak", "extraction_strength", "Composite"]
_SEP = "+" + "+".join("-" * w for w in _COL_W) + "+"
print("\n" + "=" * 102)
print("  EXP_03 — Geometry-Aware Methods (WGA / UNDIAL / DPO)")
print("=" * 102)
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

print("\n% LATEX TABLE ROWS (EXP_03):")
for label, m in metrics_map.items():
    mu=m.get("model_utility",float("nan")); fq=m.get("forget_quality",float("nan"))
    pl=m.get("privleak",float("nan")); es=m.get("extraction_strength",float("nan"))
    print(f"{label.replace('_', chr(92)+'_')} & {mu:.4f} & {fq:.4f} & {pl:.2f} & {es:.4f} \\\\")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _labels4 = ["Model\nUtility", "Forget\nQuality", "Privacy\n(1-|pl|/100)", "Extraction\nResist."]
    _ang = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist(); _ang += _ang[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    _colors = plt.cm.Set2.colors
    for ci, (lbl, m) in enumerate(metrics_map.items()):
        mu=m.get("model_utility",0.5); fq=m.get("forget_quality",0.5)
        pl=m.get("privleak",0); es=m.get("extraction_strength",0.5)
        vals = [min(max(mu,0),1), min(max(fq,0),1), min(max(1-abs(pl)/100,0),1), min(max(1-es,0),1)]
        vals += vals[:1]
        ax.plot(_ang, vals, "o-", linewidth=2.5, color=_colors[ci%8], label=lbl)
        ax.fill(_ang, vals, alpha=0.12, color=_colors[ci%8])
    ax.set_xticks(_ang[:-1]); ax.set_xticklabels(_labels4, size=11)
    ax.set_ylim(0, 1); ax.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15), fontsize=10)
    plt.title("EXP_03 — Geometry-Aware Methods Comparison\n(WGA / UNDIAL / DPO)", size=12, pad=20)
    plt.tight_layout()
    plt.savefig("/kaggle/working/EXP_03_radar.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[VIZ] Radar chart → /kaggle/working/EXP_03_radar.png")

    # Bar
    _labs = list(metrics_map.keys()); _x = np.arange(len(_labs))
    _comps = []
    for m in metrics_map.values():
        mu=m.get("model_utility",0); fq=m.get("forget_quality",0); pl=m.get("privleak",0); es=m.get("extraction_strength",0)
        _comps.append(round(0.4*mu+0.4*fq-0.1*abs(pl)-0.1*es, 4))
    fig, ax = plt.subplots(figsize=(max(7, len(_labs)*2), 4))
    _bc = [_colors[i%8] for i in range(len(_labs))]
    _bars = ax.bar(_x, _comps, color=_bc, width=0.5, edgecolor="black")
    for b, v in zip(_bars, _comps):
        ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.004, f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(_x); ax.set_xticklabels(_labs, fontsize=11)
    ax.set_title("EXP_03 — Geometry-Aware Methods: Composite Score", fontweight="bold", fontsize=12)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4); plt.tight_layout()
    plt.savefig("/kaggle/working/EXP_03_bar.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[VIZ] Bar chart → /kaggle/working/EXP_03_bar.png")
except Exception as e:
    print(f"[VIZ] Visualization skipped: {e}")

_out_path = os.path.join(REPO_DIR, "saves", "eval", "EXP_03_results.json")
os.makedirs(os.path.dirname(_out_path), exist_ok=True)
with open(_out_path, "w") as f:
    _json.dump(metrics_map, f, indent=2)
print(f"[JSON] Results → {_out_path}")
print("\n[EXP_03] Paper output complete.")
