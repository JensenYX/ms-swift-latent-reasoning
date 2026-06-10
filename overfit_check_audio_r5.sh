#!/bin/bash
# =============================================================================
# CoLaR overfit 自检（audio input -> text output, r=5 固定）。
#
# 数据仍使用当前两条 overfit 样本，只是把 user 输入替换为 <audio>，
# 顶层 audios 指向原始 wav_path；assistant 保持 <think>...</think> + answer。
#
# 用法:
#   bash overfit_check_audio_r5.sh
#   MAX_STEPS=300 SAVE_STEPS=50 SAVE_TOTAL_LIMIT=6 bash overfit_check_audio_r5.sh
# =============================================================================
set -e

CONDA_ENV=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/conda_env/ms-swift-latent
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/model_warehouse/Qwen3-Omni-30B-A3B-Instruct
RAW_DATA=${SCRIPT_DIR}/overfit_data.jsonl
: "${OUTPUT_DIR:=${SCRIPT_DIR}/output/qwen3omni_colar_audio_r5_mse_sqrt}"
: "${DATASET_PATH:=${SCRIPT_DIR}/output/qwen3omni_colar/overfit_audio.jsonl}"
: "${OVERFIT_LIMIT:=2}"
PLUGIN=${SCRIPT_DIR}/colar_plugin/plugin.py

mkdir -p "$(dirname "${DATASET_PATH}")"
if [ ! -f "${DATASET_PATH}" ]; then
    echo ">>> [data] build audio overfit dataset: ${DATASET_PATH}"
    python "${SCRIPT_DIR}/colar_plugin/prepare_data_colar.py" \
        --raw "${RAW_DATA}" \
        --out "${DATASET_PATH}" \
        --mode audio \
        --limit "${OVERFIT_LIMIT}"
else
    echo ">>> [data] reuse dataset: ${DATASET_PATH}"
fi

: "${COLAR_MAX_R:=5}"
: "${COLAR_FIXED_R:=5}"
: "${COLAR_CE_WEIGHT:=1.0}"
: "${COLAR_EMBED_WEIGHT:=1.0}"
: "${COLAR_ENTROPY_WEIGHT:=0.0}"
: "${COLAR_EMBED_LOSS:=mse}"
: "${COLAR_DETERMINISTIC:=0}"
: "${COLAR_SQRT_MEAN:=1}"
: "${COLAR_SS_PROB:=0.0}"
: "${COLAR_SS_TEMPERATURE:=1.0}"
: "${COLAR_SS_WARMUP_STEPS:=0}"
: "${COLAR_SS_CACHE_MAX:=128}"
export COLAR_MAX_R
export COLAR_FIXED_R
export COLAR_CE_WEIGHT
export COLAR_EMBED_WEIGHT
export COLAR_ENTROPY_WEIGHT
export COLAR_EMBED_LOSS
export COLAR_DETERMINISTIC
export COLAR_SQRT_MEAN
export COLAR_SS_PROB
export COLAR_SS_TEMPERATURE
export COLAR_SS_WARMUP_STEPS
export COLAR_SS_CACHE_MAX

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${NUMBA_CACHE_DIR:=/tmp/numba_cache_colar}"
# 训练显存贴近 80GB 上限；expandable_segments 可缓解 reserved-but-unallocated
# 碎片导致的小额分配 OOM。
: "${PYTORCH_ALLOC_CONF:=expandable_segments:True}"
: "${PYTORCH_CUDA_ALLOC_CONF:=${PYTORCH_ALLOC_CONF}}"
export PYTORCH_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF
: "${NPROC_PER_NODE:=4}"
: "${SAVE_STEPS:=50}"
: "${SAVE_TOTAL_LIMIT:=6}"
: "${SAVE_ONLY_MODEL:=true}"
: "${WARMUP_RATIO:=0.05}"
: "${MAX_STEPS:=300}"
: "${GRADIENT_CHECKPOINTING:=true}"
export CUDA_VISIBLE_DEVICES
export NUMBA_CACHE_DIR

DATASET_LINES="$(python - <<PY
from pathlib import Path
p = Path("${DATASET_PATH}")
print(sum(1 for line in p.open(encoding="utf-8") if line.strip()))
PY
)"
echo ">>> overfit: ${DATASET_LINES} audio samples, r=5 (FIXED), mse + sqrt_mean"
if [ "${DATASET_LINES}" -lt "${NPROC_PER_NODE}" ]; then
    echo "ERROR: dataset has ${DATASET_LINES} samples but NPROC_PER_NODE=${NPROC_PER_NODE}."
    echo "       ms-swift BatchSamplerShard uses len(dataset)//world_size, so this would produce 0 training batches."
    echo "       Add samples to ${DATASET_PATH}, or regenerate it with OVERFIT_LIMIT>=${NPROC_PER_NODE}."
    exit 1
fi
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
echo ">>> colar: ss_prob=${COLAR_SS_PROB}, ss_temperature=${COLAR_SS_TEMPERATURE}, ss_warmup_steps=${COLAR_SS_WARMUP_STEPS}, ss_cache_max=${COLAR_SS_CACHE_MAX}"
echo ">>> overfit_limit=${OVERFIT_LIMIT}"
echo ">>> dataset: ${DATASET_PATH}"
echo ">>> output_dir: ${OUTPUT_DIR}/overfit_ckpt_audio_r5"
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
    --output_dir                   "${OUTPUT_DIR}/overfit_ckpt_audio_r5" \
    --num_train_epochs             600 \
    --max_steps                    "${MAX_STEPS}" \
    --per_device_train_batch_size  1 \
    --gradient_accumulation_steps  1 \
    --learning_rate                5e-4 \
    --lr_scheduler_type            cosine \
    --warmup_ratio                 "${WARMUP_RATIO}" \
    --weight_decay                 0.0 \
    --gradient_checkpointing       "${GRADIENT_CHECKPOINTING}" \
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

echo ">>> overfit audio (r=5) 完成"
