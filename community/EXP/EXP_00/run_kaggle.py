# ============================================================
#  EXP_00 — Baseline: GradAscent Full Sweep
#  HCMUS TheFangs | NeurIPS 2026 Unlearning
# ============================================================
#
#  MỤC ĐÍCH:
#    Thiết lập reference point bằng cách chạy GradAscent
#    (phương pháp Euclidean đơn giản nhất) trên cả 3 splits:
#    forget01, forget05, forget10.
#
#  CƠ SỞ LÝ THUYẾT (PreRound_2.md):
#    GradAscent hoạt động trong không gian Euclidean phẳng,
#    xử lý tất cả chiều weight như nhau → dễ gây "representation
#    collapse" khi forget fraction lớn. Đây là baseline để chứng
#    minh sự cần thiết của các phương pháp geometry-aware hơn.
#
#  KẾT QUẢ MONG ĐỢI:
#    - forget01: OK, utility giữ được tốt
#    - forget05: model_utility bắt đầu drop
#    - forget10: rủi ro collapse cao, utility giảm mạnh
#
#  CÁCH CHẠY: Copy toàn bộ file vào 1 cell Kaggle và Run
# ============================================================

import os
import subprocess
import sys
import socket
import json

# ──────────────────────────────────────────────
# CONFIG — đổi token của bạn vào đây
# ──────────────────────────────────────────────
HF_TOKEN   = "hf_xxYOUR_TOKEN_HERE"
REPO_URL   = "https://github.com/technoob05/Hcmus_TheFangs_Unlearning.git"
REPO_DIR   = "/kaggle/working/Hcmus_TheFangs_Unlearning"
MODEL_NAME = "Llama-3.2-1B-Instruct"
TRAINER    = "GradAscent"

# Tất cả 3 splits để vẽ được degradation curve
FORGET_SPLITS = [
    ("forget01", "retain99"),
    ("forget05", "retain95"),
    ("forget10", "retain90"),
]

# Hyperparams — GradAscent cơ bản, không cần tune nhiều
LR            = "1e-5"
MAX_STEPS     = 100   # đủ để thấy pattern, không quá lâu
BATCH_SIZE    = 4
GRAD_ACCUM    = 4     # effective batch = 16

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
            f"Command failed (exit {result.returncode}): {cmd}\n"
            f"--- tail ---\n{out[-2000:]}"
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
print(f"[INFO] Working dir: {os.getcwd()}")

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
    print("[WARN] HF_TOKEN chưa set — nếu model bị gated sẽ lỗi!")

run("pip install -q 'accelerate>=1.0.0' --upgrade")
run("pip install -q -e .[lm-eval] --no-deps")
run("pip install -q lm-eval==0.4.11")

# ──────────────────────────────────────────────
# 3. Download eval data
# ──────────────────────────────────────────────
run("python setup_data.py --eval_logs --idk")

# ──────────────────────────────────────────────
# 4. Main loop — unlearn + eval mỗi split
# ──────────────────────────────────────────────
PORT       = free_port()
MODEL_PATH = f"open-unlearning/tofu_{MODEL_NAME}_full"
results    = {}

for forget_split, retain_split in FORGET_SPLITS:
    print(f"\n{'='*60}")
    print(f"  EXP_00 | GradAscent | {forget_split} / {retain_split}")
    print(f"{'='*60}")

    task_name = f"tofu_{MODEL_NAME}_{forget_split}_{TRAINER}_exp00"

    # ── 4a. Unlearn ──────────────────────────────
    unlearn_cmd = f"""
CUDA_VISIBLE_DEVICES=0 accelerate launch \\
    --config_file configs/accelerate/kaggle_config.yaml \\
    --main_process_port {PORT} \\
    --num_processes 1 \\
    src/train.py --config-name=unlearn.yaml \\
    experiment=unlearn/tofu/default.yaml \\
    trainer={TRAINER} \\
    task_name={task_name} \\
    model={MODEL_NAME} \\
    forget_split={forget_split} \\
    retain_split={retain_split} \\
    model.model_args.pretrained_model_name_or_path={MODEL_PATH} \\
    model.tokenizer_args.pretrained_model_name_or_path={MODEL_PATH} \\
    model.model_args.attn_implementation=eager \\
    retain_logs_path=saves/eval/tofu_{MODEL_NAME}_{retain_split}/TOFU_EVAL.json \\
    trainer.args.per_device_train_batch_size={BATCH_SIZE} \\
    trainer.args.gradient_accumulation_steps={GRAD_ACCUM} \\
    trainer.args.optim=adamw_torch \\
    trainer.args.learning_rate={LR} \\
    +trainer.args.max_steps={MAX_STEPS} \\
    +trainer.args.save_steps={MAX_STEPS} \\
    trainer.args.save_strategy=steps \\
    trainer.args.output_dir=saves/unlearn/{task_name}
""".strip()
    run(unlearn_cmd, cwd=REPO_DIR)

    # ── 4b. Eval ─────────────────────────────────
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
    +forget_split={forget_split} \\
    +retain_split={retain_split} \\
    +retain_logs_path=saves/eval/tofu_{MODEL_NAME}_{retain_split}/TOFU_EVAL.json
""".strip()
    run(eval_cmd, cwd=REPO_DIR)

    results[forget_split] = {
        "task_name": task_name,
        "model_path": f"saves/unlearn/{task_name}",
        "eval_path":  f"saves/eval/{task_name}_eval/",
    }
    # Free port cho vòng tiếp
    PORT = free_port()

# ──────────────────────────────────────────────
# 5. Summary
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  EXP_00 DONE — GradAscent Baseline Sweep")
print("="*60)
for split, info in results.items():
    print(f"\n  [{split}]")
    print(f"    Model : {REPO_DIR}/{info['model_path']}")
    print(f"    Eval  : {REPO_DIR}/{info['eval_path']}")

print("""
NOTE: GradAscent = Euclidean Gradient Ascent (flat metric tensor).
  Dự kiến: forget10 sẽ thể hiện utility collapse rõ nhất.
  Dùng kết quả này làm baseline so sánh với EXP_01..EXP_04.
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

# Build metrics dict {split: metrics}
metrics_map = {}
for split, info in results.items():
    m = _load_metrics(REPO_DIR, info["task_name"])
    metrics_map[f"GradAscent_{split}"] = m

# ── ASCII table ───────────────────────────────────────────────
_COL_W = [32, 15, 15, 12, 20, 12]
_HEADERS = ["Method", "model_utility", "forget_quality", "privleak", "extraction_strength", "Composite"]
_SEP = "+" + "+".join("-" * w for w in _COL_W) + "+"
print("\n" + "=" * 108)
print("  EXP_00 — GradAscent Baseline: Results across Forget Splits")
print("=" * 108)
print(_SEP)
print("|" + "|".join(f" {h:<{_COL_W[i]-1}}" for i, h in enumerate(_HEADERS)) + "|")
print(_SEP)
for label, m in metrics_map.items():
    mu = m.get("model_utility", float("nan"))
    fq = m.get("forget_quality", float("nan"))
    pl = m.get("privleak", float("nan"))
    es = m.get("extraction_strength", float("nan"))
    comp = 0.4*(mu if mu==mu else 0) + 0.4*(fq if fq==fq else 0) - 0.1*abs(pl if pl==pl else 0) - 0.1*(es if es==es else 0)
    row = [label, f"{mu:.4f}", f"{fq:.4f}", f"{pl:.2f}", f"{es:.4f}", f"{comp:.4f}"]
    print("|" + "|".join(f" {v:<{_COL_W[i]-1}}" for i, v in enumerate(row)) + "|")
print(_SEP)

# ── LaTeX rows ────────────────────────────────────────────────
print("\n% LATEX TABLE ROWS:")
for label, m in metrics_map.items():
    mu=m.get("model_utility",float("nan")); fq=m.get("forget_quality",float("nan"))
    pl=m.get("privleak",float("nan")); es=m.get("extraction_strength",float("nan"))
    print(f"GradAscent ({label.replace('_', chr(92)+'_')}) & {mu:.4f} & {fq:.4f} & {pl:.2f} & {es:.4f} \\\\")

# ── Radar chart ───────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _labels = ["Model\nUtility", "Forget\nQuality", "Privacy\n(1-|pl|/100)", "Extraction\nResist."]
    _ang = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist(); _ang += _ang[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    _colors = plt.cm.tab10.colors
    for ci, (lbl, m) in enumerate(metrics_map.items()):
        mu=m.get("model_utility",0.5); fq=m.get("forget_quality",0.5)
        pl=m.get("privleak",0); es=m.get("extraction_strength",0.5)
        vals = [min(max(mu,0),1), min(max(fq,0),1), min(max(1-abs(pl)/100,0),1), min(max(1-es,0),1)]
        vals += vals[:1]
        ax.plot(_ang, vals, "o-", linewidth=2, color=_colors[ci % 10], label=lbl)
        ax.fill(_ang, vals, alpha=0.1, color=_colors[ci % 10])
    ax.set_xticks(_ang[:-1]); ax.set_xticklabels(_labels, size=11)
    ax.set_ylim(0, 1); ax.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15), fontsize=9)
    plt.title("EXP_00 — GradAscent Baseline (Forget Split Comparison)", size=12, pad=20)
    plt.tight_layout()
    plt.savefig("/kaggle/working/EXP_00_radar.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[VIZ] Radar chart → /kaggle/working/EXP_00_radar.png")

    # Bar chart
    _labs = list(metrics_map.keys()); _comps = []
    for m in metrics_map.values():
        mu=m.get("model_utility",0); fq=m.get("forget_quality",0); pl=m.get("privleak",0); es=m.get("extraction_strength",0)
        _comps.append(round(0.4*mu+0.4*fq-0.1*abs(pl)-0.1*es, 4))
    fig, ax = plt.subplots(figsize=(max(7, len(_labs)*1.5), 4))
    _x = np.arange(len(_labs))
    _bc = plt.cm.viridis([i/max(len(_labs)-1,1) for i in range(len(_labs))])
    _bars = ax.bar(_x, _comps, color=_bc, width=0.6, edgecolor="black", linewidth=0.7)
    for b, v in zip(_bars, _comps):
        ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.004, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(_x); ax.set_xticklabels(_labs, rotation=20, ha="right")
    ax.set_title("EXP_00 — GradAscent: Composite Score per Split", fontweight="bold")
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4); plt.tight_layout()
    plt.savefig("/kaggle/working/EXP_00_bar.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[VIZ] Bar chart → /kaggle/working/EXP_00_bar.png")
except Exception as e:
    print(f"[VIZ] Visualization skipped: {e}")

# ── JSON dump ─────────────────────────────────────────────────
_out_path = os.path.join(REPO_DIR, "saves", "eval", "EXP_00_results.json")
os.makedirs(os.path.dirname(_out_path), exist_ok=True)
with open(_out_path, "w") as f:
    _json.dump(metrics_map, f, indent=2)
print(f"[JSON] Results → {_out_path}")
print("\n[EXP_00] Paper output complete.")
