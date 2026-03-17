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
