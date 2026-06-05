#!/usr/bin/env python3
"""
将原始数据集转换为 ms-swift 标准的多模态 SFT 格式。
- 只保留单轮对话（跳过多轮）
- 若 user 消息有 wav_path，则用 <audio> token + audios 字段
- 若无 wav_path，则用 content 字段（纯文本）
- gt 为 assistant 的 content 字段
"""

import json
import os
import argparse

def convert(input_path: str, output_path: str):
    total = skipped_multi = written_audio = written_text = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            d = json.loads(line)
            msgs = d.get("messages", [])

            user_msgs  = [m for m in msgs if m["role"] == "user"]
            asst_msgs  = [m for m in msgs if m["role"] == "assistant"]

            # 只保留单轮对话
            if len(user_msgs) != 1 or len(asst_msgs) != 1:
                skipped_multi += 1
                continue

            user_msg = user_msgs[0]
            asst_msg = asst_msgs[0]

            wav_path = user_msg.get("wav_path", "")
            text_content = user_msg.get("content", "")
            gt = asst_msg.get("content", "")

            if wav_path and wav_path.strip():
                # 音频样本：用 <audio> 占位符，audios 字段存路径
                new_sample = {
                    "messages": [
                        {"role": "user",      "content": "<audio>"},
                        {"role": "assistant", "content": gt},
                    ],
                    "audios": [wav_path.strip()],
                }
                written_audio += 1
            else:
                # 纯文本样本
                new_sample = {
                    "messages": [
                        {"role": "user",      "content": text_content},
                        {"role": "assistant", "content": gt},
                    ],
                }
                written_text += 1

            fout.write(json.dumps(new_sample, ensure_ascii=False) + "\n")

    print(f"总行数: {total}")
    print(f"跳过多轮对话: {skipped_multi}")
    print(f"写入音频样本: {written_audio}")
    print(f"写入文本样本: {written_text}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="/apdcephfs_fsgm3/share_303840540/users/tatelqzhang/data/train_data/sft_data/full_think_bus_20260521_merge_add_sp_final/train_s2t_nothink.jsonl")
    parser.add_argument("--output", default="/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/git_warehouse/ms-swift-latent-reasoning/data/qwen3omni_sft.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    convert(args.input, args.output)
