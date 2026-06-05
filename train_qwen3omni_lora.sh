#!/bin/bash
# =============================================================================
# Qwen3-Omni-30B LoRA SFT — 8x H800 80G
# 启动: bash train_qwen3omni_lora.sh
# 原理: NPROC_PER_NODE=4 + 8 卡可见 → ms-swift 自动 DDP+MP 混合并行
#       4 个 DDP 进程，每进程内 2 卡 device_map（160 GiB/进程）
#       比单进程 device_map 快 ~4x，无需 DeepSpeed
# =============================================================================

# ── 环境 ──────────────────────────────────────────────────────────────────────
CONDA_ENV=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/conda_env/ms-swift-latent
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=2       # 4 DDP 进程，每进程内 2 卡 device_map

# ── 路径 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/model_warehouse/Qwen3-Omni-30B-A3B-Instruct
RAW_DATA=/apdcephfs_fsgm3/share_303840540/users/tatelqzhang/data/train_data/sft_data/full_think_bus_20260521_merge_add_sp_final/train_s2t_nothink.jsonl
OUTPUT_DIR=${SCRIPT_DIR}/output/qwen3omni_lora_sft
DATASET_PATH=${OUTPUT_DIR}/sft_data.jsonl

# ── LoRA 参数 ─────────────────────────────────────────────────────────────────
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
TARGET_MODULES=all-linear
FREEZE_VIT=true          # 冻结 audio_tower
FREEZE_ALIGNER=true      # 冻结 audio proj1/proj2

# ── 训练超参 ──────────────────────────────────────────────────────────────────
NUM_EPOCHS=2
PER_DEVICE_BATCH=4
GRAD_ACCUM=4             # global batch = 4进程 × 4 batch × 4 accum = 64
MAX_LENGTH=8192
LEARNING_RATE=1e-4
LR_SCHEDULER=cosine
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01

# ── 日志 / 保存 ───────────────────────────────────────────────────────────────
LOGGING_STEPS=10
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=3

# =============================================================================

set -e

# ── 激活 conda 环境 ────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Step 1: 数据格式转换（已存在则跳过）──────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${DATASET_PATH}" ]; then
    echo ">>> [Step 1] 转换数据格式..."
    python - <<EOF
import json, os

raw_path  = "${RAW_DATA}"
out_path  = "${DATASET_PATH}"

records, skipped = [], 0
with open(raw_path, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        msgs      = d.get('messages', [])
        user_msgs = [m for m in msgs if m['role'] == 'user']
        asst_msgs = [m for m in msgs if m['role'] == 'assistant']

        # 只保留单轮
        if len(user_msgs) != 1 or len(asst_msgs) != 1:
            skipped += 1
            continue

        wav  = user_msgs[0].get('wav_path', '').strip()
        text = user_msgs[0].get('content', '')
        gt   = asst_msgs[0].get('content', '')

        if wav:
            record = {
                'messages': [
                    {'role': 'user',      'content': '<audio>'},
                    {'role': 'assistant', 'content': gt},
                ],
                'audios': [wav],
            }
        else:
            record = {
                'messages': [
                    {'role': 'user',      'content': text},
                    {'role': 'assistant', 'content': gt},
                ],
            }
        records.append(record)

with open(out_path, 'w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f'[data] 写入 {len(records)} 条（跳过多轮 {skipped} 条）-> {out_path}')
EOF
else
    echo ">>> [Step 1] 已有转换数据，跳过: ${DATASET_PATH}"
fi

# ── Step 2: 启动训练 ──────────────────────────────────────────────────────────
# NPROC_PER_NODE=4 + 8 卡可见 → ms-swift 自动 DDP+MP 混合并行
#   4 个 DDP 进程，每进程内 2 卡 device_map，等效 160 GiB/进程
#   比单进程 device_map 快 ~4x，无需 DeepSpeed
echo ">>> [Step 2] 启动训练 (NPROC_PER_NODE=${NPROC_PER_NODE}, DDP+MP 混合并行)..."

NPROC_PER_NODE=${NPROC_PER_NODE} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
swift sft \
    --model                        "${MODEL_PATH}" \
    --model_type                   qwen3_omni_moe \
    --template                     qwen3_omni \
    --torch_dtype                  bfloat16 \
    --attn_impl                    flash_attn \
    \
    --dataset                      "${DATASET_PATH}" \
    --split_dataset_ratio          0.0 \
    --dataset_num_proc             4 \
    --max_length                   ${MAX_LENGTH} \
    --load_from_cache_file         true \
    \
    --tuner_type                   lora \
    --lora_rank                    ${LORA_RANK} \
    --lora_alpha                   ${LORA_ALPHA} \
    --lora_dropout                 ${LORA_DROPOUT} \
    --target_modules               all-linear \
    --freeze_vit                   ${FREEZE_VIT} \
    --freeze_aligner               ${FREEZE_ALIGNER} \
    \
    --output_dir                   "${OUTPUT_DIR}" \
    --num_train_epochs             ${NUM_EPOCHS} \
    --per_device_train_batch_size  ${PER_DEVICE_BATCH} \
    --per_device_eval_batch_size   ${PER_DEVICE_BATCH} \
    --gradient_accumulation_steps  ${GRAD_ACCUM} \
    --learning_rate                ${LEARNING_RATE} \
    --lr_scheduler_type            ${LR_SCHEDULER} \
    --warmup_ratio                 ${WARMUP_RATIO} \
    --weight_decay                 ${WEIGHT_DECAY} \
    --gradient_checkpointing       true \
    # --padding_free                 true \
    \
    --logging_steps                ${LOGGING_STEPS} \
    --logging_first_step           true \
    --save_strategy                steps \
    --save_steps                   ${SAVE_STEPS} \
    --save_total_limit             ${SAVE_TOTAL_LIMIT} \
    --report_to                    tensorboard \
    \
    --dataloader_num_workers       4

echo ">>> 训练完成！checkpoint 保存在: ${OUTPUT_DIR}"
