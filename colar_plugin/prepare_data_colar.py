"""
数据转换：把带 thinking 的原始 jsonl 转成 ms-swift 标准 SFT 格式。

输入（你的数据，每行）：
  {"messages":[
     {"role":"user","content":"问题文本","wav_path":"/path.wav"},   # wav_path 可选
     {"role":"assistant","content":"最终回答","reasoning_content":"思考过程"}
  ]}

输出（ms-swift 标准格式）：
  - 纯文本模式（--mode text，里程碑1）：
      {"messages":[
         {"role":"user","content":"问题文本"},
         {"role":"assistant","content":"<think>\n思考过程\n</think>\n\n最终回答"}
      ]}
  - 音频模式（--mode audio，里程碑2）：
      {"messages":[
         {"role":"user","content":"<audio>"},
         {"role":"assistant","content":"<think>\n思考过程\n</think>\n\n最终回答"}
      ],
       "audios":["/path.wav"]}

为什么要把 reasoning 包进 <think>：
  ms-swift 的 _swift_encode 只 tokenize message['content']，不会自动把 reasoning_content
  拼进训练文本。CoLaR 要压缩的“思考段”必须出现在 input_ids 里，因此手动包进 <think>...</think>，
  与 Qwen3-Omni 的 thinking 模板（thinking_prefix='<think>\n'）token 边界一致。自定义 Template 再用
  <think>(151667) / </think>(151668) 这两个单 token 定位要压缩的区间。

用法：
  python colar_plugin/prepare_data_colar.py \
      --raw  /path/to/qwen3omnithinking.jsonl.rank00000-of00016.jsonl \
      --out  output/qwen3omni_colar/colar_text.jsonl \
      --mode text
  # 多个分片：--raw a.jsonl b.jsonl ...  或  --raw "dir/qwen3omnithinking.jsonl.rank*.jsonl"
"""
import argparse
import glob
import json
import os
from typing import List


def _think_wrap(reasoning: str, answer: str) -> str:
    reasoning = (reasoning or '').strip()
    answer = (answer or '').strip()
    if reasoning:
        return f'<think>\n{reasoning}\n</think>\n\n{answer}'
    # 没有 reasoning 的样本退化成普通回答（不带 think 块）
    return answer


def convert(raw_paths: List[str], out_path: str, mode: str, limit: int = 0) -> None:
    assert mode in ('text', 'audio')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    n_written = 0
    n_skipped = 0
    n_no_think = 0
    with open(out_path, 'w', encoding='utf-8') as fout:
        for raw_path in raw_paths:
            with open(raw_path, encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    msgs = d.get('messages', [])
                    user_msgs = [m for m in msgs if m.get('role') == 'user']
                    asst_msgs = [m for m in msgs if m.get('role') == 'assistant']
                    # 只保留单轮
                    if len(user_msgs) != 1 or len(asst_msgs) != 1:
                        n_skipped += 1
                        continue

                    u = user_msgs[0]
                    a = asst_msgs[0]
                    reasoning = a.get('reasoning_content', '')
                    answer = a.get('content', '')
                    if not reasoning:
                        n_no_think += 1
                    asst_content = _think_wrap(reasoning, answer)

                    wav = (u.get('wav_path') or '').strip()
                    user_text = u.get('content', '')

                    if mode == 'audio' and wav:
                        record = {
                            'messages': [
                                {'role': 'user', 'content': '<audio>'},
                                {'role': 'assistant', 'content': asst_content},
                            ],
                            'audios': [wav],
                        }
                    else:
                        # text 模式，或 audio 模式但本条无 wav -> 退化为纯文本
                        record = {
                            'messages': [
                                {'role': 'user', 'content': user_text},
                                {'role': 'assistant', 'content': asst_content},
                            ],
                        }
                    fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                    n_written += 1
                    if limit > 0 and n_written >= limit:
                        break
            if limit > 0 and n_written >= limit:
                break

    print(f'[prepare_data_colar] mode={mode} 写入 {n_written} 条 '
          f'(跳过非单轮 {n_skipped} 条, 无 reasoning {n_no_think} 条) -> {out_path}')


def _expand(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        matched = glob.glob(p)
        out.extend(sorted(matched) if matched else [p])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', nargs='+', required=True, help='原始 jsonl，可多个或用通配符')
    ap.add_argument('--out', required=True, help='输出 jsonl 路径')
    ap.add_argument('--mode', choices=['text', 'audio'], default='text')
    ap.add_argument('--limit', type=int, default=0, help='最多写入 N 条；0 表示不限制')
    args = ap.parse_args()
    raw_paths = _expand(args.raw)
    assert raw_paths, f'no input files matched: {args.raw}'
    print(f'[prepare_data_colar] inputs ({len(raw_paths)}): {raw_paths}')
    convert(raw_paths, args.out, args.mode, args.limit)


if __name__ == '__main__':
    main()
