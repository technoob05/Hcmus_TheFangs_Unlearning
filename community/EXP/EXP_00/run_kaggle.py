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
