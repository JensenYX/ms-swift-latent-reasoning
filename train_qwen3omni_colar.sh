#!/bin/bash
# =============================================================================
# Qwen3-Omni-30B + CoLaR latent-reasoning SFT —— 里程碑1（纯文本，单卡 device_map）
# 启动: bash train_qwen3omni_colar.sh
#
# 与普通 LoRA SFT 的差异：
#   --template colar_qwen3_omni              使用 CoLaR 注册的模板
#   --external_plugins .../colar_plugin/plugin.py   注册模板 + 替换 Trainer
#   COLAR_* 环境变量                         CoLaR 专属超参（压缩率/各 loss 权重）
#
# 里程碑1先验证核心机制：embedding 压缩 + CE + latent head(NLL) 双 loss 能前向/反向/保存。
# 先用 COLAR_MAX_R=1（不压缩）把管线打通，再调成 COLAR_MAX_R=4 验证压缩分支。
# 纯文本里程碑先关 padding_free（逐样本压缩重建与 packing 冲突）。
# =============================================================================

set -e

# ── 环境 ──────────────────────────────────────────────────────────────────────
CONDA_ENV=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/conda_env/ms-swift-latent
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 路径 ──────────────────────────────────────────────────────────────────────
MODEL_PATH=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/model_warehouse/Qwen3-Omni-30B-A3B-Instruct
RAW_DATA=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/git_warehouse/Latent-reasoning-emotion/data/qwen3omnithinking.jsonl.rank00000-of00016.jsonl
OUTPUT_DIR=${SCRIPT_DIR}/output/qwen3omni_colar
DATASET_PATH=${OUTPUT_DIR}/colar_text.jsonl
PLUGIN=${SCRIPT_DIR}/colar_plugin/plugin.py

mkdir -p "${OUTPUT_DIR}"

# ── Step 1: 数据转换（reasoning_content -> <think> 块）────────────────────────
if [ ! -f "${DATASET_PATH}" ]; then
    echo ">>> [Step 1] 转换数据（纯文本模式）..."
    python "${SCRIPT_DIR}/colar_plugin/prepare_data_colar.py" \
        --raw  "${RAW_DATA}" \
        --out  "${DATASET_PATH}" \
        --mode text
else
    echo ">>> [Step 1] 已有数据，跳过: ${DATASET_PATH}"
fi

# ── CoLaR 超参（环境变量传入 Trainer）─────────────────────────────────────────
export COLAR_MAX_R=4            # 先用 1 打通管线，再改 4 验证压缩
export COLAR_CE_WEIGHT=1.0
export COLAR_EMBED_WEIGHT=1.0
export COLAR_ENTROPY_WEIGHT=0.0
export COLAR_EMBED_LOSS=nll     # nll | mse
# export COLAR_LP_INTERMEDIATE=2048
# export COLAR_DETERMINISTIC=0
# export COLAR_SQRT_MEAN=0

# ── 单进程 + 多卡 device_map（模型并行）─────────────────────────────────────
# 之所以不用 DDP/torchrun：本实现的 compute_loss 直接调用 thinker 文本主干子模块
# （绕过顶层 forward），单进程 device_map 没有 DDP wrapper，行为最干净。
# 30B MoE 单卡 80G 放不下，用 8 卡 device_map 自动模型并行。
# latent head 固定在 embedding 所在卡，跨卡 hidden 在 _latent_loss 里搬运（已处理）。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 注意：不要设 NPROC_PER_NODE（设了会触发 DDP，与直接调子模块的方式冲突）

# ── Step 2: 启动训练 ──────────────────────────────────────────────────────────
echo ">>> [Step 2] 启动 CoLaR SFT 训练 (COLAR_MAX_R=${COLAR_MAX_R})..."

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
    --dataset_num_proc             4 \
    --max_length                   8192 \
    --load_from_cache_file         true \
    \
    --tuner_type                   lora \
    --lora_rank                    16 \
    --lora_alpha                   32 \
    --lora_dropout                 0.05 \
    --target_modules               all-linear \
    --freeze_vit                   true \
    --freeze_aligner               true \
    \
    --output_dir                   "${OUTPUT_DIR}/ckpt" \
    --num_train_epochs             1 \
    --per_device_train_batch_size  1 \
    --gradient_accumulation_steps  8 \
    --learning_rate                1e-4 \
    --lr_scheduler_type            cosine \
    --warmup_ratio                 0.05 \
    --weight_decay                 0.01 \
    --gradient_checkpointing       true \
    --padding_free                 false \
    \
    --logging_steps                5 \
    --logging_first_step           true \
    --save_strategy                steps \
    --save_steps                   200 \
    --save_total_limit             3 \
    --report_to                    tensorboard \
    \
    --dataloader_num_workers       4

echo ">>> 训练完成！checkpoint: ${OUTPUT_DIR}/ckpt （应含 adapter + latent_policy.pt）"
