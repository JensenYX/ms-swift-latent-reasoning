#!/bin/bash
# =============================================================================
# Query-anchored CoLaR overfit check (mixed audio/text input -> text output, r=10).
#
# This is an isolated experiment path:
#   - plugin: colar_plugin/query_anchor_plugin.py
#   - output: output/qwen3omni_query_anchor_hidden_r10_mse_sqrt
# It does not overwrite the baseline CoLaR checkpoints.
# =============================================================================
set -e

CONDA_ENV=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/conda_env/ms-swift-latent
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/model_warehouse/Qwen3-Omni-30B-A3B-Instruct
RAW_DATA=${SCRIPT_DIR}/overfit_data.jsonl
: "${OUTPUT_DIR:=${SCRIPT_DIR}/output/qwen3omni_query_anchor_hidden_r10_mse_sqrt}"
: "${DATASET_PATH:=${SCRIPT_DIR}/output/qwen3omni_colar/overfit_mixed_audio_text.jsonl}"
: "${OVERFIT_LIMIT:=0}"
PLUGIN=${SCRIPT_DIR}/colar_plugin/query_anchor_plugin.py

mkdir -p "$(dirname "${DATASET_PATH}")"
if [ ! -f "${DATASET_PATH}" ]; then
    echo ">>> [data] build mixed audio/text overfit dataset: ${DATASET_PATH}"
    python "${SCRIPT_DIR}/colar_plugin/prepare_data_colar.py" \
        --raw "${RAW_DATA}" \
        --out "${DATASET_PATH}" \
        --mode audio \
        --limit "${OVERFIT_LIMIT}"
else
    echo ">>> [data] reuse dataset: ${DATASET_PATH}"
fi

: "${COLAR_MAX_R:=10}"
: "${COLAR_FIXED_R:=10}"
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
: "${QUERY_ANCHOR_CONDITION_HEAD:=1}"
: "${QUERY_ANCHOR_CONDITION_INTERACTION:=1}"
: "${QUERY_ANCHOR_CONDITION_INPUT_NORM:=0}"
: "${QUERY_ANCHOR_RESIDUAL:=1}"
: "${QUERY_ANCHOR_NORM_PRESERVE:=1}"
: "${QUERY_ANCHOR_GATE_INIT:=0.1}"
: "${QUERY_ANCHOR_GATE_L2:=0.0}"
: "${QUERY_ANCHOR_NOISE_STD:=0.0}"
: "${QUERY_ANCHOR_RESIDUAL_SOURCE:=hidden}"
: "${QUERY_ANCHOR_HIDDEN_SOURCE:=user_last}"
: "${QUERY_ANCHOR_PROJ_INIT:=identity}"
export COLAR_MAX_R COLAR_FIXED_R COLAR_CE_WEIGHT COLAR_EMBED_WEIGHT COLAR_ENTROPY_WEIGHT
export COLAR_EMBED_LOSS COLAR_DETERMINISTIC COLAR_SQRT_MEAN
export COLAR_SS_PROB COLAR_SS_TEMPERATURE COLAR_SS_WARMUP_STEPS COLAR_SS_CACHE_MAX
export QUERY_ANCHOR_CONDITION_HEAD QUERY_ANCHOR_CONDITION_INTERACTION QUERY_ANCHOR_CONDITION_INPUT_NORM
export QUERY_ANCHOR_RESIDUAL QUERY_ANCHOR_NORM_PRESERVE
export QUERY_ANCHOR_GATE_INIT QUERY_ANCHOR_GATE_L2 QUERY_ANCHOR_NOISE_STD QUERY_ANCHOR_RESIDUAL_SOURCE
export QUERY_ANCHOR_HIDDEN_SOURCE QUERY_ANCHOR_PROJ_INIT

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${NUMBA_CACHE_DIR:=/tmp/numba_cache_colar}"
: "${NPROC_PER_NODE:=4}"
: "${SAVE_STEPS:=50}"
: "${SAVE_TOTAL_LIMIT:=6}"
: "${SAVE_ONLY_MODEL:=true}"
: "${WARMUP_RATIO:=0.05}"
: "${MAX_STEPS:=300}"
: "${GRADIENT_CHECKPOINTING:=false}"
: "${GRADIENT_CHECKPOINTING_KWARGS:=}"
export CUDA_VISIBLE_DEVICES
export NUMBA_CACHE_DIR

DATASET_LINES="$(python - <<PY
from pathlib import Path
p = Path("${DATASET_PATH}")
print(sum(1 for line in p.open(encoding="utf-8") if line.strip()))
PY
)"
echo ">>> overfit: ${DATASET_LINES} mixed audio/text samples, query-anchor r=10, mse + sqrt_mean"
if [ "${DATASET_LINES}" -lt "${NPROC_PER_NODE}" ]; then
    echo "ERROR: dataset has ${DATASET_LINES} samples but NPROC_PER_NODE=${NPROC_PER_NODE}."
    exit 1
fi
if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    : "${DDP_FIND_UNUSED_PARAMETERS:=true}"
    echo ">>> launch: NPROC_PER_NODE=${NPROC_PER_NODE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    LAUNCH_ENV=("NPROC_PER_NODE=${NPROC_PER_NODE}" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
else
    : "${DDP_FIND_UNUSED_PARAMETERS:=false}"
    echo ">>> launch: single process device_map, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    LAUNCH_ENV=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

echo ">>> scheduler: cosine, warmup_ratio=${WARMUP_RATIO}"
echo ">>> train: max_steps=${MAX_STEPS}"
echo ">>> gradient_checkpointing: ${GRADIENT_CHECKPOINTING}${GRADIENT_CHECKPOINTING_KWARGS:+, kwargs=${GRADIENT_CHECKPOINTING_KWARGS}}"
echo ">>> colar: ss_prob=${COLAR_SS_PROB}, ss_temperature=${COLAR_SS_TEMPERATURE}, ss_warmup_steps=${COLAR_SS_WARMUP_STEPS}, ss_cache_max=${COLAR_SS_CACHE_MAX}"
echo ">>> query-anchor: condition_head=${QUERY_ANCHOR_CONDITION_HEAD}, condition_interaction=${QUERY_ANCHOR_CONDITION_INTERACTION}, condition_input_norm=${QUERY_ANCHOR_CONDITION_INPUT_NORM}, residual=${QUERY_ANCHOR_RESIDUAL}, residual_source=${QUERY_ANCHOR_RESIDUAL_SOURCE}, hidden_source=${QUERY_ANCHOR_HIDDEN_SOURCE}, gate_init=${QUERY_ANCHOR_GATE_INIT}, gate_l2=${QUERY_ANCHOR_GATE_L2}, noise_std=${QUERY_ANCHOR_NOISE_STD}, proj_init=${QUERY_ANCHOR_PROJ_INIT}"
echo ">>> output_dir: ${OUTPUT_DIR}/overfit_ckpt_audio_r10"

CMD=(
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
    --output_dir                   "${OUTPUT_DIR}/overfit_ckpt_audio_r10" \
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
)
if [ -n "${GRADIENT_CHECKPOINTING_KWARGS}" ]; then
    CMD+=(--gradient_checkpointing_kwargs "${GRADIENT_CHECKPOINTING_KWARGS}")
fi

env "${LAUNCH_ENV[@]}" "${CMD[@]}"

echo ">>> query-anchor overfit mixed audio/text (r=10) 完成"
