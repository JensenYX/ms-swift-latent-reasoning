#!/bin/bash
# =============================================================================
# CoLaR overfit 自检（r=2 固定）：验证“压缩路径”本身正确。
#
# 与 overfit_check.sh（r=1）的区别：r=1 走 colar_trainer.py 的 `if r==1` 捷径，
# 整个 _compress_batch（reshape 池化 / multinomial 采标签 / 变长重 pad）从未执行。
# 本脚本用 COLAR_FIXED_R=2 把 r 钉死为 2，第一次真正跑压缩 forward→loss→backward。
#
# 判据（注意与 r=1 不同！）：
#   - 必须跑得通、不 NaN/Inf，grad_norm 正常有限 → 压缩 forward/backward 链路对。
#   - train/ce_loss 单调下降，但会停在一个 >0 的“地板”，不会到 ~0。这是正常的：
#       压缩后思考段的 CE 标签是每步从每个 r-token 组里随机抽一个代表 token
#       （colar_trainer.py 的 multinomial，无 seed），监督目标本身在抖动，
#       构造上就不可能降到 0；只有答案段标签稳定、能被背下来。
#     => 千万别套用 r=1 的“ce_loss→~0”判据，否则会把健康的压缩管线误判为坏。
#   - train/embed_loss（高斯 NLL）持续下降甚至变负，正常，不作判据。
#   - train/r 应恒为 2.0（确认 FIXED_R 生效）。
#
# 用法: bash overfit_check_r2.sh
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

# ── CoLaR 超参：r 钉死为 2（COLAR_FIXED_R 覆盖随机采样）。其余对齐 r=1 自检 ──
# latent head 非确定性（log_std 可学）；deterministic=1 + nll 是病态组合，勿用。
export COLAR_MAX_R=2
export COLAR_FIXED_R=2
export COLAR_CE_WEIGHT=1.0
export COLAR_EMBED_WEIGHT=1.0
export COLAR_ENTROPY_WEIGHT=0.0
export COLAR_EMBED_LOSS=nll
export COLAR_DETERMINISTIC=0

# 单进程多卡 device_map（30B 单卡放不下）；不要设 NPROC_PER_NODE
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo ">>> overfit: 2 samples, r=2 (FIXED), 高 LR, 多 epoch"

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
    --output_dir                   "${OUTPUT_DIR}/overfit_ckpt_r2" \
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

echo ">>> overfit (r=2) 完成"
