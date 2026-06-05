#!/bin/bash
# =============================================================================
# CoLaR overfit 自检：在 2 条样本上反复训练，验证整条链路（forward→loss→backward→optim）正确。
#
# 判据：
#   - r=1（不压缩、确定性）下，train/ce_loss 应快速下降到接近 0（能把答案背下来）。
#   - train/embed_loss 是高斯 NLL，会持续下降甚至变负，这正常，不作为判据。
#   - 若 ce_loss 不降 → 管线有 bug（梯度没回流 / label 错位 / forward 不对）。
#
# 用法: bash overfit_check.sh
# =============================================================================
set -e

CONDA_ENV=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/conda_env/ms-swift-latent
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/model_warehouse/Qwen3-Omni-30B-A3B-Instruct
OUTPUT_DIR=${SCRIPT_DIR}/output/qwen3omni_colar
DATASET_PATH=${OUTPUT_DIR}/overfit_text.jsonl
PLUGIN=${SCRIPT_DIR}/colar_plugin/plugin.py

# ── CoLaR 超参：overfit 用 r=1，对齐原版 SFT 前提（latent head 非确定性，log_std 可学）──
# 注意：deterministic=1 + nll 是病态组合——高斯 NLL 含 0.5*((x-μ)/σ)²，σ→0 会发散到 1e17，
#       淹没 CE 梯度（原版 colar.py 的 SFT 也不用此组合）。判据仍只看 train/ce_loss 是否→~0。
export COLAR_MAX_R=1
export COLAR_CE_WEIGHT=1.0
export COLAR_EMBED_WEIGHT=1.0
export COLAR_ENTROPY_WEIGHT=0.0
export COLAR_EMBED_LOSS=nll
export COLAR_DETERMINISTIC=0

# 单进程多卡 device_map（30B 单卡放不下）；不要设 NPROC_PER_NODE
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo ">>> overfit: 2 samples, r=1, 高 LR, 多 epoch"

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
    --output_dir                   "${OUTPUT_DIR}/overfit_ckpt" \
    --num_train_epochs             60 \
    --per_device_train_batch_size  1 \
    --gradient_accumulation_steps  1 \
    --learning_rate                5e-4 \
    --lr_scheduler_type            constant \
    --warmup_ratio                 0.0 \
    --weight_decay                 0.0 \
    --gradient_checkpointing       true \
    --padding_free                 false \
    \
    --logging_steps                1 \
    --logging_first_step           true \
    --save_strategy                no \
    --report_to                    tensorboard \
    \
    --dataloader_num_workers       1

echo ">>> overfit 完成"
