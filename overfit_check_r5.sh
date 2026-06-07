#!/bin/bash
# =============================================================================
# CoLaR overfit 自检（r=5 固定）：在 r=2 已打通后，验证更高压缩率的 latent rollout。
#
# 默认沿用刚在 r=2 上成功的配置：
#   - COLAR_EMBED_LOSS=mse
#   - COLAR_SQRT_MEAN=1
#   - COLAR_FIXED_R=5
#   - NPROC_PER_NODE=4
#
# 用法:
#   bash overfit_check_r5.sh
#   MAX_STEPS=300 SAVE_STEPS=50 SAVE_TOTAL_LIMIT=6 bash overfit_check_r5.sh
# =============================================================================
set -e

CONDA_ENV=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/conda_env/ms-swift-latent
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/model_warehouse/Qwen3-Omni-30B-A3B-Instruct
: "${OUTPUT_DIR:=${SCRIPT_DIR}/output/qwen3omni_colar_r5_mse_sqrt}"
: "${DATASET_PATH:=${SCRIPT_DIR}/output/qwen3omni_colar/overfit_text.jsonl}"
PLUGIN=${SCRIPT_DIR}/colar_plugin/plugin.py

: "${COLAR_MAX_R:=5}"
: "${COLAR_FIXED_R:=5}"
: "${COLAR_CE_WEIGHT:=1.0}"
: "${COLAR_EMBED_WEIGHT:=1.0}"
: "${COLAR_ENTROPY_WEIGHT:=0.0}"
: "${COLAR_EMBED_LOSS:=mse}"
: "${COLAR_DETERMINISTIC:=0}"
: "${COLAR_SQRT_MEAN:=1}"
export COLAR_MAX_R
export COLAR_FIXED_R
export COLAR_CE_WEIGHT
export COLAR_EMBED_WEIGHT
export COLAR_ENTROPY_WEIGHT
export COLAR_EMBED_LOSS
export COLAR_DETERMINISTIC
export COLAR_SQRT_MEAN

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${NPROC_PER_NODE:=4}"
: "${SAVE_STEPS:=50}"
: "${SAVE_TOTAL_LIMIT:=6}"
: "${SAVE_ONLY_MODEL:=true}"
: "${WARMUP_RATIO:=0.05}"
: "${MAX_STEPS:=300}"
export CUDA_VISIBLE_DEVICES

echo ">>> overfit: 2 samples, r=5 (FIXED), mse + sqrt_mean"
if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    : "${DDP_FIND_UNUSED_PARAMETERS:=true}"
    echo ">>> launch: NPROC_PER_NODE=${NPROC_PER_NODE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo ">>> ddp_find_unused_parameters=${DDP_FIND_UNUSED_PARAMETERS}"
    LAUNCH_ENV=("NPROC_PER_NODE=${NPROC_PER_NODE}" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
else
    : "${DDP_FIND_UNUSED_PARAMETERS:=false}"
    echo ">>> launch: single process device_map, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    LAUNCH_ENV=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi
echo ">>> scheduler: cosine, warmup_ratio=${WARMUP_RATIO}"
echo ">>> train: max_steps=${MAX_STEPS}"
echo ">>> colar: fixed_r=${COLAR_FIXED_R}, embed_loss=${COLAR_EMBED_LOSS}, sqrt_mean=${COLAR_SQRT_MEAN}, deterministic=${COLAR_DETERMINISTIC}"
echo ">>> dataset: ${DATASET_PATH}"
echo ">>> output_dir: ${OUTPUT_DIR}/overfit_ckpt_r5"
echo ">>> checkpoint: every ${SAVE_STEPS} steps, keep ${SAVE_TOTAL_LIMIT}, save_only_model=${SAVE_ONLY_MODEL}"

env "${LAUNCH_ENV[@]}" \
swift sft \
    --model                        "${MODEL_PATH}" \
    --model_type                   qwen3_omni_moe \
    --template                     colar_qwen3_omni \
    --external_plugins             "${PLUGIN}" \
    --torch_dtype                  bfloat16 \
    --attn_impl                    flash_attn \
    \
    --dataset                      "${DATASET_PATH}" \
    --split_dataset_ratio          0.0 \
    --dataset_num_proc             1 \
    --max_length                   8192 \
    --load_from_cache_file         false \
    \
    --tuner_type                   lora \
    --lora_rank                    16 \
    --lora_alpha                   32 \
    --lora_dropout                 0.0 \
    --target_modules               all-linear \
    --freeze_vit                   true \
    --freeze_aligner               true \
    \
    --output_dir                   "${OUTPUT_DIR}/overfit_ckpt_r5" \
    --num_train_epochs             600 \
    --max_steps                    "${MAX_STEPS}" \
    --per_device_train_batch_size  1 \
    --gradient_accumulation_steps  1 \
    --learning_rate                5e-4 \
    --lr_scheduler_type            cosine \
    --warmup_ratio                 "${WARMUP_RATIO}" \
    --weight_decay                 0.0 \
    --gradient_checkpointing       true \
    --padding_free                 false \
    --ddp_find_unused_parameters   "${DDP_FIND_UNUSED_PARAMETERS}" \
    \
    --logging_steps                1 \
    --logging_first_step           true \
    --save_strategy                steps \
    --save_steps                   "${SAVE_STEPS}" \
    --save_total_limit             "${SAVE_TOTAL_LIMIT}" \
    --save_only_model              "${SAVE_ONLY_MODEL}" \
    --dataloader_num_workers       1

echo ">>> overfit (r=5) 完成"
