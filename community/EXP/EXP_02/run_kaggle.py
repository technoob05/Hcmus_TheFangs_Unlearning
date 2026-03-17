# ============================================================
#  EXP_02 — Sequential Unlearning Experiment
#  HCMUS TheFangs | NeurIPS 2026 Unlearning
# ============================================================
#
#  MỤC ĐÍCH:
#    Nghiên cứu hiện tượng "catastrophic forgetting of alignment"
#    khi unlearning được thực hiện tuần tự (sequential) thay vì
#    một lần (single-shot). Đây là vấn đề cốt lõi trong PreRound_1.
#
#  CƠ SỞ LÝ THUYẾT (PreRound_1.md — Sequential Unlearning):
#    Trong môi trường production, unlearning phải sequential:
#    mỗi bước unlearn sẽ alter probability measure của latent
#    space → Fisher Information Matrix (FIM) ước tính ở bước
#    trước sẽ "stale" ở bước sau → non-geodesic drift tích lũy.
#
#    Experiment này verify trực tiếp:
#      - Chạy NPO forget01 → lưu checkpoint_1
#      - Tiếp tục NPO forget05 từ checkpoint_1 → lưu checkpoint_2
#      - Tiếp tục NPO forget10 từ checkpoint_2 → lưu checkpoint_3
#    So sánh checkpoint_3 với direct NPO forget10 (từ EXP_01):
#      → Nếu sequential tệ hơn: chứng minh error accumulation
#      → Nếu tương đương: sequential là viable
#
#  KẾT QUẢ MONG ĐỢI:
#    Sequential sẽ có utility thấp hơn và extraction_strength
#    cao hơn (unlearning kém hiệu quả hơn) so với direct forget10.
#
#  NOTE: Trong TOFU, forget01 ⊂ forget05 ⊂ forget10, nên chain
#        sequential là: full → unlearn01 → unlearn05 → unlearn10
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

# Dùng NPO vì geometry-aware hơn GradAscent — ít collapse hơn
TRAINER    = "NPO"

# Sequential chain: từng bước mở rộng forget set
SEQ_STEPS = [
    # (step_id, forget_split, retain_split, max_steps)
    (1, "forget01", "retain99", 80),
    (2, "forget05", "retain95", 80),
    (3, "forget10", "retain90", 80),
]

BATCH_SIZE = 4
GRAD_ACCUM = 4
LR         = "1e-5"

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
# 3. Sequential unlearn chain
# ──────────────────────────────────────────────
BASE_MODEL_PATH = f"open-unlearning/tofu_{MODEL_NAME}_full"
prev_checkpoint = BASE_MODEL_PATH   # bắt đầu từ fully fine-tuned model
sequential_results = {}

for step_id, forget_split, retain_split, max_steps in SEQ_STEPS:
    PORT = free_port()

    print(f"\n{'='*60}")
    print(f"  EXP_02 | Sequential Step {step_id}/3")
    print(f"  Input  : {prev_checkpoint}")
    print(f"  Forget : {forget_split} / {retain_split}")
    print(f"{'='*60}")

    task_name = f"tofu_{MODEL_NAME}_{forget_split}_{TRAINER}_exp02_seq{step_id}"
    output_dir = f"saves/unlearn/{task_name}"

    # retain_logs: dùng đúng split tương ứng
    retain_eval_path = f"saves/eval/tofu_{MODEL_NAME}_{retain_split}/TOFU_EVAL.json"

    # ── 3a. Unlearn từ prev_checkpoint ───────────
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
    model.model_args.pretrained_model_name_or_path={prev_checkpoint} \\
    model.tokenizer_args.pretrained_model_name_or_path={prev_checkpoint} \\
    model.model_args.attn_implementation=eager \\
    retain_logs_path={retain_eval_path} \\
    trainer.args.per_device_train_batch_size={BATCH_SIZE} \\
    trainer.args.gradient_accumulation_steps={GRAD_ACCUM} \\
    trainer.args.optim=adamw_torch \\
    trainer.args.learning_rate={LR} \\
    +trainer.args.max_steps={max_steps} \\
    +trainer.args.save_steps={max_steps} \\
    trainer.args.save_strategy=steps \\
    trainer.args.output_dir={output_dir}
""".strip()
    run(unlearn_cmd, cwd=REPO_DIR)

    # ── 3b. Eval checkpoint này ──────────────────
    eval_cmd = f"""
CUDA_VISIBLE_DEVICES=0 accelerate launch \\
    --config_file configs/accelerate/kaggle_config.yaml \\
    --main_process_port {PORT} \\
    --num_processes 1 \\
    src/eval.py --config-name=eval.yaml \\
    task_name={task_name}_eval \\
    model={MODEL_NAME} \\
    model.model_args.pretrained_model_name_or_path={output_dir} \\
    model.tokenizer_args.pretrained_model_name_or_path={output_dir} \\
    model.model_args.attn_implementation=eager \\
    +forget_split={forget_split} \\
    +retain_split={retain_split} \\
    +retain_logs_path={retain_eval_path}
""".strip()
    run(eval_cmd, cwd=REPO_DIR)

    sequential_results[f"step{step_id}_{forget_split}"] = {
        "input_checkpoint": prev_checkpoint,
        "output_checkpoint": output_dir,
        "eval_dir": f"saves/eval/{task_name}_eval/",
    }

    # Chain: output của bước này là input của bước tiếp theo
    prev_checkpoint = output_dir
    PORT = free_port()

# ──────────────────────────────────────────────
# 4. Summary
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  EXP_02 DONE — Sequential Unlearning Chain")
print("="*60)
print(f"\n  Chain: {MODEL_NAME}_full → seq1 → seq2 → seq3")
for step_key, info in sequential_results.items():
    print(f"\n  [{step_key}]")
    print(f"    In  : {info['input_checkpoint']}")
    print(f"    Out : {REPO_DIR}/{info['output_checkpoint']}")
    print(f"    Eval: {REPO_DIR}/{info['eval_dir']}")

print("""
PHÂN TÍCH:
  So sánh step3 (forget10 sequential) với EXP_01 NPO forget10 direct:
  - model_utility   : sequential thấp hơn? → error accumulation ✓
  - extraction_strength: sequential cao hơn? → unlearning kém sâu ✓
  - privleak        : sequential tệ hơn? → alignment drift ✓
  Nếu sequential = direct → NPO đủ robust cho sequential setting.
  Reference: PreRound_1.md — "non-geodesic drift and error accumulation"
""")
