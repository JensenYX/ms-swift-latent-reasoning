#!/bin/bash
# =============================================================================
# Qwen3-Omni CoLaR overfit inference check (r=2 run)
#
# Default mode is COLAR latent_generate (not plain text generate):
#   prompt -> latent_policy compressed reasoning -> answer generation
#
# GPU: 推理默认单卡即可（LoRA + bf16 30B 通常一张卡够）。训练 overfit_check_r2.sh
# 才需要 8 卡 device_map。多卡推理可显式传入，例如：
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash infer_overfit_check_r2.sh
#
# Usage:
#   bash infer_overfit_check_r2.sh
#   CHECKPOINT=/path/to/checkpoint-600 bash infer_overfit_check_r2.sh
#   MAX_NEW_TOKENS=8192 MAX_LATENT_FORWARD=2048 LIMIT=2 bash infer_overfit_check_r2.sh
#   MODE=text bash infer_overfit_check_r2.sh   # legacy baseline only
#
# Output:
#   output/qwen3omni_colar/overfit_infer_r2.jsonl
#
# Pass criteria for r=2 overfit:
#   - script completes without error
#   - hit_eol=true on most samples (model predicts </think>)
#   - match.answer_expected_in_response or match.answer_response_in_expected ~ 2/2
#   Do NOT expect full_expected_in_response (thinking is latent/silent).
# =============================================================================
set -e

CONDA_ENV=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/conda_env/ms-swift-latent
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${CHECKPOINT_ROOT:=${SCRIPT_DIR}/output/qwen3omni_colar/overfit_ckpt_r2}"
: "${DATASET:=${SCRIPT_DIR}/output/qwen3omni_colar/overfit_text.jsonl}"
: "${OUTPUT:=${SCRIPT_DIR}/output/qwen3omni_colar/overfit_infer_r2.jsonl}"
: "${PLUGIN:=${SCRIPT_DIR}/colar_plugin/plugin.py}"
: "${MODEL_PATH:=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/model_warehouse/Qwen3-Omni-30B-A3B-Instruct}"
: "${MODE:=latent}"
: "${MAX_NEW_TOKENS:=8192}"
: "${MAX_LATENT_FORWARD:=2048}"
: "${BATCH_SIZE:=1}"
: "${DEVICE_MAP:=auto}"
: "${ATTN_IMPL:=auto}"
: "${PROGRESS_EVERY:=0}"

export CUDA_VISIBLE_DEVICES

echo ">>> inference: Qwen3-Omni CoLaR overfit r=2 (mode=${MODE})"
echo ">>> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ">>> checkpoint_root=${CHECKPOINT_ROOT}"
if [ -n "${CHECKPOINT:-}" ]; then
    echo ">>> checkpoint=${CHECKPOINT}"
fi
echo ">>> dataset=${DATASET}"
echo ">>> output=${OUTPUT}"
echo ">>> max_new_tokens=${MAX_NEW_TOKENS}"
echo ">>> max_latent_forward=${MAX_LATENT_FORWARD}"
echo ">>> attn_impl=${ATTN_IMPL}"
echo ">>> progress_every=${PROGRESS_EVERY}"

CMD=(
    python "${SCRIPT_DIR}/infer_qwen3omni_colar_overfit.py"
    --mode "${MODE}"
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
    --require-latent-policy
)

if [ -n "${CHECKPOINT:-}" ]; then
    CMD+=(--checkpoint "${CHECKPOINT}")
fi
if [ -n "${LIMIT:-}" ]; then
    CMD+=(--limit "${LIMIT}")
fi
if [ "${DISABLE_THINKING:-0}" = "1" ]; then
    CMD+=(--disable-thinking)
fi
if [ "${RETURN_DETAILS:-0}" = "1" ]; then
    CMD+=(--return-details)
fi
if [ -n "${LATENT_TEMPERATURE:-}" ]; then
    CMD+=(--latent-temperature "${LATENT_TEMPERATURE}")
fi
if [ -n "${EOL_TEMPERATURE:-}" ]; then
    CMD+=(--eol-temperature "${EOL_TEMPERATURE}")
fi
if [ -n "${TEMPERATURE:-}" ]; then
    CMD+=(--temperature "${TEMPERATURE}")
fi

"${CMD[@]}"

echo ">>> inference 完成: ${OUTPUT}"
