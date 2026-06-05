"""
SFT for Qwen3-Omni-30B-A3B-Instruct using ms-swift.
Input : audio (wav_path in user message) or plain text (content)
GT    : assistant content

启动方式（单进程 + 8 卡可见 → ms-swift 自动 device_map 模型并行，无需 DeepSpeed/torchrun）:
  conda activate /apdcephfs_tj6/share_303840540/hunyuan/jensenwang/conda_env/ms-swift-latent
  python train_qwen3omni_lora.py
"""

import os
import json

# ── 单进程启动，ms-swift 根据可见 GPU 数量自动做 device_map 模型并行 ────────
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = '/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/model_warehouse/Qwen3-Omni-30B-A3B-Instruct'
RAW_DATA    = '/apdcephfs_fsgm3/share_303840540/users/tatelqzhang/data/train_data/sft_data/full_think_bus_20260521_merge_add_sp_final/train_s2t_nothink.jsonl'
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, 'output', 'qwen3omni_lora_sft')
DATASET_PATH = os.path.join(OUTPUT_DIR, 'sft_data.jsonl')


# ── Step 1: 数据格式转换 ───────────────────────────────────────────────────────
def prepare_dataset(raw_path: str, out_path: str) -> str:
    """
    将原始 JSONL 转成 ms-swift 标准多模态 SFT 格式：
      - 只保留单轮对话（跳过多轮）
      - 有 wav_path → audios 字段 + user content 用 <audio> 占位符
      - 无 wav_path → 纯文本 content
    """
    if os.path.exists(out_path):
        print(f'[data] 检测到已有数据文件，跳过转换: {out_path}')
        return out_path

    records = []
    skipped = 0
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            msgs = d.get('messages', [])

            user_msgs = [m for m in msgs if m['role'] == 'user']
            asst_msgs = [m for m in msgs if m['role'] == 'assistant']

            # 只保留单轮
            if len(user_msgs) != 1 or len(asst_msgs) != 1:
                skipped += 1
                continue

            user_msg = user_msgs[0]
            gt       = asst_msgs[0].get('content', '')
            wav_path = user_msg.get('wav_path', '').strip()
            text     = user_msg.get('content', '')

            if wav_path:
                record = {
                    'messages': [
                        {'role': 'user',      'content': '<audio>'},
                        {'role': 'assistant', 'content': gt},
                    ],
                    'audios': [wav_path],
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

    print(f'[data] 写入 {len(records)} 条样本（跳过多轮 {skipped} 条）→ {out_path}')
    return out_path


# ── Step 2: SFT 训练 ───────────────────────────────────────────────────────────
def main():
    from swift import SftArguments, sft_main

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 数据准备
    prepare_dataset(RAW_DATA, DATASET_PATH)

    args = SftArguments(
        # ── 模型 ──────────────────────────────────────────────────────────────
        model=MODEL_PATH,
        model_type='qwen3_omni_moe',
        template='qwen3_omni',
        torch_dtype='bfloat16',
        attn_impl='flash_attn',
        experts_impl='grouped_mm',       # MoE grouped GEMM，提升专家层效率

        # ── 数据集 ────────────────────────────────────────────────────────────
        dataset=[DATASET_PATH],
        split_dataset_ratio=0,           # 不切验证集
        dataset_num_proc=4,
        max_length=8192,
        load_from_cache_file=True,

        # ── LoRA（只训练 LLM，冻结音频编码器和对齐层）──────────────────────────
        tuner_type='lora',
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['all-linear'],
        freeze_vit=True,                 # 冻结 audio_tower
        freeze_aligner=True,             # 冻结 audio proj1/proj2 + visual merger

        # ── 训练超参 ──────────────────────────────────────────────────────────
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,   # 等效 global batch = 8卡×4×4 = 128
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        padding_free=True,               # 变长 batch，避免 padding 浪费

        # ── 日志 / 保存 ────────────────────────────────────────────────────────
        logging_steps=10,
        logging_first_step=True,
        save_strategy='steps',
        save_steps=200,
        save_total_limit=3,
        report_to=['tensorboard'],

        # ── 其他 ──────────────────────────────────────────────────────────────
        dataloader_num_workers=4,
    )

    sft_main(args)


if __name__ == '__main__':
    main()
