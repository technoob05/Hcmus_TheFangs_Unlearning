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
