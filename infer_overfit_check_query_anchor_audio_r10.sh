#!/bin/bash
# =============================================================================
# Qwen3-Omni query-anchored CoLaR mixed audio/text overfit inference check (r=10).
# =============================================================================
set -e

CONDA_ENV=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/conda_env/ms-swift-latent
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${NUMBA_CACHE_DIR:=/tmp/numba_cache_colar}"
: "${CHECKPOINT_ROOT:=${SCRIPT_DIR}/output/qwen3omni_query_anchor_hidden_r10_mse_sqrt/overfit_ckpt_audio_r10}"
: "${DATASET:=${SCRIPT_DIR}/output/qwen3omni_colar/overfit_mixed_audio_text.jsonl}"
: "${OUTPUT:=${SCRIPT_DIR}/output/qwen3omni_query_anchor_hidden_r10_mse_sqrt/overfit_infer_query_anchor_r10.jsonl}"
: "${PLUGIN:=${SCRIPT_DIR}/colar_plugin/query_anchor_plugin.py}"
: "${MODEL_PATH:=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/model_warehouse/Qwen3-Omni-30B-A3B-Instruct}"
: "${MAX_NEW_TOKENS:=512}"
: "${MAX_LATENT_FORWARD:=400}"
: "${BATCH_SIZE:=1}"
: "${DEVICE_MAP:=auto}"
: "${ATTN_IMPL:=auto}"
: "${PROGRESS_EVERY:=128}"
: "${LATENT_TEMPERATURE:=1.0}"
: "${EOL_TEMPERATURE:=1.0}"

export CUDA_VISIBLE_DEVICES
export NUMBA_CACHE_DIR

echo ">>> inference: Qwen3-Omni query-anchored CoLaR mixed audio/text overfit r=10"
echo ">>> checkpoint_root=${CHECKPOINT_ROOT}"
echo ">>> dataset=${DATASET}"
echo ">>> output=${OUTPUT}"
echo ">>> max_latent_forward=${MAX_LATENT_FORWARD}"
echo ">>> latent_temperature=${LATENT_TEMPERATURE}"
echo ">>> eol_temperature=${EOL_TEMPERATURE}"

CMD=(
    python "${SCRIPT_DIR}/infer_qwen3omni_query_anchor_overfit.py"
    --checkpoint-root "${CHECKPOINT_ROOT}"
    --dataset "${DATASET}"
    --output "${OUTPUT}"
    --plugin "${PLUGIN}"
    --model "${MODEL_PATH}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --max-latent-forward "${MAX_LATENT_FORWARD}"
    --batch-size "${BATCH_SIZE}"
    --device-map "${DEVICE_MAP}"
    --attn-impl "${ATTN_IMPL}"
    --progress-every "${PROGRESS_EVERY}"
    --latent-temperature "${LATENT_TEMPERATURE}"
    --eol-temperature "${EOL_TEMPERATURE}"
)

if [ -n "${CHECKPOINT:-}" ]; then
    CMD+=(--checkpoint "${CHECKPOINT}")
fi
if [ -n "${LIMIT:-}" ]; then
    CMD+=(--limit "${LIMIT}")
fi
if [ -n "${TEMPERATURE:-}" ]; then
    CMD+=(--temperature "${TEMPERATURE}")
fi

"${CMD[@]}"

echo ">>> query-anchor inference 完成: ${OUTPUT}"
