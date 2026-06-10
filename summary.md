# Qwen3-Omni CoLaR Latent Reasoning 进展总结

更新时间：2026-06-10

仓库：

```text
/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/git_warehouse/ms-swift-latent-reasoning
```

原始 CoLaR 仓库：

```text
/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/git_warehouse/colar
```

目标：把 CoLaR latent reasoning 方法迁移到 Qwen3-Omni。此前 overfit 检查已打通纯 text 输入、纯 text 输出；2026-06-07 已新增并验证 audio input -> text output 的 overfit 链路。

## 当前核心结论

1. r=2 的训练/推理主链路已经基本打通。
2. 最初 `nll + sqrt_mean=0 + deterministic/argmax inference` 会出现严重 free rollout drift：teacher-forcing 正常，但 latent 自回归生成预测不出 `</think>`。
3. 改成更接近原始 CoLaR 的配置后成功：

```text
COLAR_EMBED_LOSS=mse
COLAR_SQRT_MEAN=1
LATENT_TEMPERATURE=1.0
EOL_TEMPERATURE=1.0
```

4. 成功的 r=2 实验中，两个 overfit sample 都能 hit `</think>`：

```text
sample 0: n_latent_forward=235, hit_eol=True
sample 1: n_latent_forward=890, hit_eol=True
answer-ish matches: 2/2
eol hits: 2/2
```

5. r=5 也已 overfit 成功：

```text
sample 0: n_latent_forward=94, hit_eol=True
sample 1: n_latent_forward=356, hit_eol=True
answer-ish matches: 2/2
eol hits: 2/2
```

6. audio input -> text output 的 r=5 overfit 也已成功。当前 audio 数据集 4 行，实际是两个 overfit 样本各重复一次，用来满足 `NPROC_PER_NODE=4` 时每个 rank 至少有样本：

```text
checkpoint: output/qwen3omni_colar_audio_r5_mse_sqrt/overfit_ckpt_audio_r5/v1-20260607-135625/checkpoint-300
infer output: output/qwen3omni_colar_audio_r5_mse_sqrt/overfit_infer_audio_r5.jsonl
sample 0: n_latent_forward=94, hit_eol=True, answer_response_in_expected=True
sample 1: n_latent_forward=356, hit_eol=True, answer_response_in_expected=True
sample 2: n_latent_forward=94, hit_eol=True, answer_response_in_expected=True
sample 3: n_latent_forward=356, hit_eol=True, answer_response_in_expected=True
answer-ish matches: 4/4
eol hits: 4/4
```

7. mixed audio/text 的 r=5 和 r=10 推理也已完成，整体基本成功：

```text
mixed r=5:
  answer_response_in_expected: 4/4
  eol hits: 3/4
  未停样本: idx=1，中文化学音频样本，n_latent_forward=400, hit_eol=False
  该样本虽然没有正确预测停止，但 response 仍是 gold answer 的前缀。

mixed r=10:
  answer_response_in_expected: 4/4
  eol hits: 4/4
```

8. 这说明之前不是简单的 label/prompt/checkpoint 加载 bug，而主要是训练目标/压缩归一化/采样方式导致 free rollout 不稳定。

## Audio input -> text output 新增进展

2026-06-07 已在不改变纯文本默认路径的前提下接入音频输入：

- `colar_plugin/qwen3omni_audio.py` 新增 Qwen3-Omni audio embedding helper。
- `colar_plugin/colar_trainer.py` 在存在 `input_features` 时调用 `thinker.get_audio_features(...)`，并把输出 scatter 到展开后的 `<|audio_pad|>` token；无音频时仍直接 `embed_tokens(input_ids)`。
- `colar_plugin/colar_infer.py` 的 latent inference 支持 `InferRequest(..., audios=[...])`，首段 prompt embedding 会合并 audio tower 输出；latent rollout 和 answer decode 仍沿用文本主干 KV cache。
- `infer_qwen3omni_colar_overfit.py` 会从顶层 `audios` 或 user message 里的 `wav_path` 识别音频输入。
- 新增 `overfit_check_audio_r5.sh` 和 `infer_overfit_check_audio_r5.sh`，默认复用 r=5 成功配置 `mse + sqrt_mean + sampling`。
- `colar_plugin/prepare_data_colar.py` 已支持 `--mode audio --limit N`。

已完成检查：

```text
python -m py_compile ... 通过
bash -n overfit_check_audio_r5.sh / infer_overfit_check_audio_r5.sh 通过
template encode 音频样本通过：
  input_ids_len=57
  n_audio_tokens=45
  input_features_shape=(1, 128, 344)
  feature_attention_mask_shape=(1, 344)
  last_tokens 包含 <think>\n
```

注意：当前执行环境里 `librosa/numba` 需要可写 cache，所以音频脚本默认设置：

```text
NUMBA_CACHE_DIR=/tmp/numba_cache_colar
```

audio overfit 训练和推理已完成：

```text
训练：300/300 step 完成，最后 ce_loss ~= 0.1969，embed_loss ~= 0.00219
推理：LIMIT=4 bash infer_overfit_check_audio_r5.sh
结果：answer-ish matches 4/4，eol hits 4/4
```

## Mixed audio/text overfit 准备与结果

2026-06-08 用户把 `overfit_data.jsonl` 改成音频和纯文本混合数据。已生成新的 mixed dataset，避免覆盖前面 audio-only 成功实验：

```text
dataset: output/qwen3omni_colar/overfit_mixed_audio_text.jsonl
logical rows: 4
row 0: audio, user content=<audio>
row 1: audio, user content=<audio>
row 2: text, user content=原始文本问题
row 3: audio, user content=<audio>
```

转换逻辑来自 `prepare_data_colar.py --mode audio`：

```text
有 wav_path 的样本 -> user content 替换为 <audio>，顶层 audios=[wav_path]
没有 wav_path 的样本 -> 保留 user 原始文本，不写 audios 字段
assistant 都保持 <think>...</think> + answer
```

`OVERFIT_LIMIT=0` 表示不截断样本，导出所有可用记录；`OVERFIT_LIMIT=2/4` 才表示只取前 2/4 条。这里使用 `0` 是为了避免脚本默认值 `2` 把 mixed 数据截成前两条 audio-only。

用户本地启动 mixed r=5 训练建议使用：

```bash
DATASET_PATH=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/git_warehouse/ms-swift-latent-reasoning/output/qwen3omni_colar/overfit_mixed_audio_text.jsonl \
OUTPUT_DIR=/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/git_warehouse/ms-swift-latent-reasoning/output/qwen3omni_colar_mixed_r5_mse_sqrt \
OVERFIT_LIMIT=0 \
NPROC_PER_NODE=4 \
bash overfit_check_audio_r5.sh
```

训练完成后推理建议使用：

```bash
CHECKPOINT_ROOT=output/qwen3omni_colar_mixed_r5_mse_sqrt/overfit_ckpt_audio_r5 \
DATASET=output/qwen3omni_colar/overfit_mixed_audio_text.jsonl \
OUTPUT=output/qwen3omni_colar_mixed_r5_mse_sqrt/overfit_infer_mixed_r5.jsonl \
LIMIT=4 \
bash infer_overfit_check_audio_r5.sh
```

注意：不要直接无参数运行 `bash overfit_check_audio_r5.sh` 来做 mixed 实验；脚本默认 `DATASET_PATH` 仍是 `output/qwen3omni_colar/overfit_audio.jsonl`，会复用旧 audio-only 数据集。

2026-06-08 另增 mixed r=10 脚本，默认已指向 mixed dataset 和独立输出目录：

```text
train script: overfit_check_audio_r10.sh
infer script: infer_overfit_check_audio_r10.sh
dataset: output/qwen3omni_colar/overfit_mixed_audio_text.jsonl
output dir: output/qwen3omni_colar_mixed_r10_mse_sqrt
checkpoint dir: output/qwen3omni_colar_mixed_r10_mse_sqrt/overfit_ckpt_audio_r10
```

r=10 训练：

```bash
bash overfit_check_audio_r10.sh
```

r=10 推理：

```bash
LIMIT=4 bash infer_overfit_check_audio_r10.sh
```

2026-06-08 mixed r=5 和 r=10 推理都已完成。

mixed r=5：

```text
checkpoint: output/qwen3omni_colar_mixed_r5_mse_sqrt/overfit_ckpt_audio_r5/v0-20260607-235138/checkpoint-300
infer output: output/qwen3omni_colar_mixed_r5_mse_sqrt/overfit_infer_mixed_r5.jsonl
训练 step 300: ce_loss ~= 0.3054, embed_loss ~= 0.00618, r = 5.0

idx 0: audio, n_latent_forward=94, hit_eol=True, answer_response_in_expected=True
idx 1: audio, n_latent_forward=400, hit_eol=False, answer_response_in_expected=True
idx 2: text,  n_latent_forward=121, hit_eol=True, answer_response_in_expected=True
idx 3: audio, n_latent_forward=104, hit_eol=True, answer_response_in_expected=True

answer_response_in_expected: 4/4
eol hits: 3/4
```

mixed r=5 唯一未正确停止的是 `idx=1` 中文化学音频样本。它跑满了 `MAX_LATENT_FORWARD=400`，但生成答案仍是 gold answer 的前缀，因此可视为答案 overfit 成功、停止预测存在一例 free rollout/EOL miss。

mixed r=10：

```text
checkpoint: output/qwen3omni_colar_mixed_r10_mse_sqrt/overfit_ckpt_audio_r10/v0-20260608-003027/checkpoint-300
infer output: output/qwen3omni_colar_mixed_r10_mse_sqrt/overfit_infer_mixed_r10.jsonl
训练 step 300: ce_loss ~= 0.2440, embed_loss ~= 0.00263, r = 10.0

idx 0: audio, n_latent_forward=47, hit_eol=True, answer_response_in_expected=True
idx 1: audio, n_latent_forward=178, hit_eol=True, answer_response_in_expected=True
idx 2: text,  n_latent_forward=61, hit_eol=True, answer_response_in_expected=True
idx 3: audio, n_latent_forward=52, hit_eol=True, answer_response_in_expected=True

answer_response_in_expected: 4/4
eol hits: 4/4
```

对比 r=5，r=10 在这个 mixed overfit 集合上停止预测更稳，4 条样本全部 hit `</think>`，且 latent 步数约为 r=5 的一半。

## 关键代码文件

### 训练

```text
overfit_check_r2.sh
overfit_check_r5.sh
colar_plugin/colar_trainer.py
```

`overfit_check_r2.sh`：

- r=2 固定 overfit 训练脚本。
- 已改成 `OUTPUT_DIR`、`DATASET_PATH`、`COLAR_EMBED_LOSS`、`COLAR_SQRT_MEAN` 等都可被环境变量覆盖。
- 默认仍是 r=2；如果要复现实验，建议显式传入关键环境变量。

`overfit_check_r5.sh`：

- 新增的 r=5 固定 overfit 脚本。
- 默认配置：

```text
COLAR_FIXED_R=5
COLAR_MAX_R=5
COLAR_EMBED_LOSS=mse
COLAR_SQRT_MEAN=1
NPROC_PER_NODE=4
MAX_STEPS=300
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=6
OUTPUT_DIR=output/qwen3omni_colar_r5_mse_sqrt
checkpoint dir=output/qwen3omni_colar_r5_mse_sqrt/overfit_ckpt_r5
```

`colar_plugin/colar_trainer.py` 的重要修改：

- 修复 r>1 时 `<think>\n` 和压缩正文的边界：
  - prefix 保留到 `<think>\n`。
  - compressed reason 从换行之后开始。
  - latent loss 从 prefix 最后一个 token 的 hidden state 开始预测第一个 compressed embedding。
  - 最后一个 compressed embedding 预测 `</think>` embedding。
- 通过 top-level forward hook 让 DDP 可见自定义 loss。
- checkpoint 额外保存 `latent_policy.pt` 和完整 `colar_config.json`。

### 推理

```text
infer_overfit_check_r2.sh
infer_qwen3omni_colar_overfit.py
colar_plugin/colar_infer.py
```

`infer_overfit_check_r2.sh`：

- 名字虽然是 r2，但可用于 r5，只要设置 `CHECKPOINT_ROOT` 和 `OUTPUT`。
- 支持：

```text
LIMIT
MAX_LATENT_FORWARD
MAX_NEW_TOKENS
LATENT_TEMPERATURE
EOL_TEMPERATURE
TEMPERATURE
PROGRESS_EVERY
CHECKPOINT_ROOT
CHECKPOINT
```

`colar_plugin/colar_infer.py`：

- 实现 Qwen3-Omni 的 CoLaR latent_generate。
- latent loop 和 answer loop 都使用 KV cache。
- prompt 编码到 `<think>\n` 后开始 latent rollout。
- 每个 latent step 用 `latent_policy(hidden[-1])` 生成 latent embedding，再喂回文本主干。
- 每步用 `lm_head` 检查是否预测 `</think>`。
- latent 结束后显式 append `</think>` embedding，然后生成 answer tokens。

### 诊断

```text
diagnose_colar_teacher_forcing.py
```

用途：

- 重建训练时的 gold compressed path。
- 在 teacher-forcing 下检查：
  - inference prompt prefix 是否和训练 prefix 对齐。
  - `</think>` label 是否被监督。
  - latent_policy 预测 gold compressed embedding 的 cosine/MSE/NLL。
  - gold path 上 `</think>` token 的 logit rank/prob。

## 重要参数含义

```text
MAX_LATENT_FORWARD
```

latent reasoning 最大步数。每一步是一次 latent embedding 自回归 forward。当前实现有 KV cache，但每步仍要过一次 Qwen3-Omni text backbone。

```text
MAX_NEW_TOKENS
```

latent reasoning 结束、强制 append `</think>` 后，生成可见 answer 的最大 token 数。

```text
LIMIT
```

从 overfit jsonl 里取前 N 条样本做推理。当前 overfit 数据只有两条。

样本对应：

```text
sample 0: "拉拉是如何发现自己的拉拉身份的？"
sample 1: "248. 镧系元素的特征氧化态是+3价，但Ce、Pr、Tb、Dy却可以呈现+4价氧化态，而Sm、Eu、Tm、Yb却又有+2价氧化态，为什么?"
```

## 实验记录

### 1. 旧 r=2 checkpoint 推理

checkpoint：

```text
output/qwen3omni_colar/overfit_ckpt_r2/v6-20260605-160142/checkpoint-250
```

现象：

- sample0 有时能生成 answer 的前缀。
- sample1 失败，输出重复/乱码。
- 两个样本都基本不 hit `</think>`。

结论：

- 仅 loss 下降不能证明 latent rollout 成功。
- 当时怀疑训练/推理边界不一致。

### 2. 修复 newline/latent loss 边界后的 r=2，仍使用 nll/sqrt_mean=0/deterministic

训练命令：

```bash
MAX_STEPS=300 SAVE_STEPS=50 SAVE_TOTAL_LIMIT=6 bash overfit_check_r2.sh
```

checkpoint：

```text
output/qwen3omni_colar_r2_fix/overfit_ckpt_r2/v1-20260606-191152/checkpoint-300
```

训练配置：

```text
embed_loss=nll
sqrt_mean=false
fixed_r=2
```

推理命令：

```bash
CHECKPOINT_ROOT=output/qwen3omni_colar_r2_fix/overfit_ckpt_r2 \
LIMIT=2 \
MAX_LATENT_FORWARD=512 \
MAX_NEW_TOKENS=512 \
PROGRESS_EVERY=128 \
OUTPUT=output/qwen3omni_colar_r2_fix/overfit_infer_r2_limit2_512latent_512ans.jsonl \
bash infer_overfit_check_r2.sh
```

结果：

```text
sample 0: n_latent_forward=512, hit_eol=False, answer prefix ok
sample 1: n_latent_forward=512, hit_eol=False, bad/repeated output
answer-ish matches: 1/2
eol hits: 0/2
```

再把 `MAX_LATENT_FORWARD` 提到 900：

```bash
CHECKPOINT_ROOT=output/qwen3omni_colar_r2_fix/overfit_ckpt_r2 \
LIMIT=2 \
MAX_LATENT_FORWARD=900 \
MAX_NEW_TOKENS=512 \
PROGRESS_EVERY=128 \
OUTPUT=output/qwen3omni_colar_r2_fix/overfit_infer_r2_limit2_900latent_512ans.jsonl \
bash infer_overfit_check_r2.sh
```

结果：

```text
sample 0: n_latent_forward=900, hit_eol=False, answer prefix ok
sample 1: n_latent_forward=900, hit_eol=False, output比512时好，但仍不匹配
answer-ish matches: 1/2
eol hits: 0/2
```

结论：

- 512 对 sample1 本来不够，因为 sample1 gold compressed length 约 890。
- 但 sample0 gold length 只有 235，跑到 900 仍不 hit eol，说明 free rollout drift 仍然存在。

### 3. Teacher-forcing 诊断

命令：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && \
conda activate /apdcephfs_tj6/share_303840540/hunyuan/jensenwang/conda_env/ms-swift-latent && \
CUDA_VISIBLE_DEVICES=0 python diagnose_colar_teacher_forcing.py \
  --checkpoint-root output/qwen3omni_colar_r2_fix/overfit_ckpt_r2 \
  --dataset output/qwen3omni_colar/overfit_text.jsonl \
  --output output/qwen3omni_colar_r2_fix/teacher_forcing_diag_ckpt300.jsonl \
  --limit 2
```

结果：

```text
sample 0:
  prefix_match=True
  reason_tokens=470
  compressed_tokens=235
  labels: close=151668 trained=True
  latent cos_mean=0.9159
  last_close_cos=0.9981
  gold_eol rank=1 token='</think>'

sample 1:
  prefix_match=True
  reason_tokens=1777
  compressed_tokens=889
  labels: close=151668 trained=True
  latent cos_mean=0.8361
  last_close_cos=0.9980
  gold_eol rank=1 token='</think>'
```

结论：

- prompt prefix 没错。
- `</think>` label 确实训练了。
- gold compressed path 上 `</think>` 是 rank 1。
- 问题是 free rollout 自回归漂移，而不是 stop token 没学到。

### 4. r=2，mse + sqrt_mean + sampling 成功实验

训练命令：

```bash
OUTPUT_DIR=output/qwen3omni_colar_r2_mse_sqrt \
COLAR_EMBED_LOSS=mse \
COLAR_SQRT_MEAN=1 \
MAX_STEPS=300 \
SAVE_STEPS=50 \
SAVE_TOTAL_LIMIT=6 \
bash overfit_check_r2.sh
```

checkpoint：

```text
output/qwen3omni_colar_r2_mse_sqrt/overfit_ckpt_r2/v0-20260606-222252/checkpoint-300
```

训练配置确认：

```json
{
  "embed_loss": "mse",
  "sqrt_mean": true,
  "fixed_r": 2
}
```

推理命令：

```bash
CHECKPOINT_ROOT=output/qwen3omni_colar_r2_mse_sqrt/overfit_ckpt_r2 \
LIMIT=2 \
MAX_LATENT_FORWARD=900 \
MAX_NEW_TOKENS=512 \
LATENT_TEMPERATURE=1.0 \
EOL_TEMPERATURE=1.0 \
PROGRESS_EVERY=128 \
OUTPUT=output/qwen3omni_colar_r2_mse_sqrt/overfit_infer_r2_limit2_900latent_sampled_512ans.jsonl \
bash infer_overfit_check_r2.sh
```

结果：

```text
sample 0:
  n_latent_forward=235
  hit_eol=True
  answer_response_in_expected=True

sample 1:
  n_latent_forward=890
  hit_eol=True
  answer_response_in_expected=True

answer-ish matches: 2/2
eol hits: 2/2
```

注意：

- 日志里的 `answer_match=False` 指的是完整 expected answer 是否完全包含在 response 里。
- 因为 `MAX_NEW_TOKENS=512` 截断了 answer，所以完整答案不会全包含。
- 真实应看 `answer_response_in_expected=True`，即生成答案是 gold answer 的前缀。

结论：

- 这组说明 r=2 latent reasoning overfit 成功。
- 关键配置是 `mse + sqrt_mean=1 + latent/eol sampling temperature=1.0`。

### 5. r=5，mse + sqrt_mean + sampling 成功实验

训练脚本：

```text
overfit_check_r5.sh
```

训练命令：

```bash
bash overfit_check_r5.sh
```

checkpoint：

```text
output/qwen3omni_colar_r5_mse_sqrt/overfit_ckpt_r5/v0-20260607-104348/checkpoint-300
```

训练配置确认：

```json
{
  "embed_loss": "mse",
  "sqrt_mean": true,
  "fixed_r": 5
}
```

最后几步训练状态：

```text
step 300:
  ce_loss ~= 0.1961
  embed_loss ~= 0.00204
  r = 5.0
```

推理命令：

```bash
CHECKPOINT_ROOT=output/qwen3omni_colar_r5_mse_sqrt/overfit_ckpt_r5 \
LIMIT=2 \
MAX_LATENT_FORWARD=400 \
MAX_NEW_TOKENS=512 \
LATENT_TEMPERATURE=1.0 \
EOL_TEMPERATURE=1.0 \
PROGRESS_EVERY=128 \
OUTPUT=output/qwen3omni_colar_r5_mse_sqrt/overfit_infer_r5_limit2_400latent_sampled_512ans.jsonl \
bash infer_overfit_check_r2.sh
```

结果：

```text
sample 0:
  n_latent_forward=94
  hit_eol=True
  answer_response_in_expected=True

sample 1:
  n_latent_forward=356
  hit_eol=True
  answer_response_in_expected=True

answer-ish matches: 2/2
eol hits: 2/2
```

输出文件：

```text
output/qwen3omni_colar_r5_mse_sqrt/overfit_infer_r5_limit2_400latent_sampled_512ans.jsonl
```

r=5 的 expected latent length：

```text
sample 0: ceil(470 / 5) = 94
sample 1: ceil(1777 / 5) = 356
```

实际推理得到的 `n_latent_forward` 正好是 `94/356`，说明 r=5 压缩率生效，且 free rollout 没有漂。相比 r=2 的 `235/890`，latent reasoning 步数约为 40%。

## r=5 后续可选检查

如果要看完整答案，而不是 512 token 前缀，可以跑：

```bash
CHECKPOINT_ROOT=output/qwen3omni_colar_r5_mse_sqrt/overfit_ckpt_r5 \
LIMIT=2 \
MAX_LATENT_FORWARD=400 \
MAX_NEW_TOKENS=2048 \
LATENT_TEMPERATURE=1.0 \
EOL_TEMPERATURE=1.0 \
PROGRESS_EVERY=128 \
OUTPUT=output/qwen3omni_colar_r5_mse_sqrt/overfit_infer_r5_limit2_400latent_sampled_2048ans.jsonl \
bash infer_overfit_check_r2.sh
```

## 实用注意事项

1. 训练用 8 卡 DDP：

```text
NPROC_PER_NODE=4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

2. 推理单卡可运行：

```text
CUDA_VISIBLE_DEVICES=0
DEVICE_MAP=auto
```

3. 如果要判断 overfit 是否成功，主要看：

```text
eol hits 是否 2/2
n_latent_forward 是否接近 gold compressed length
answer_response_in_expected 是否 True
```

4. `MAX_NEW_TOKENS=512` 只是快速检查答案前缀。若想完整生成答案，调大到 2048 或 4096。

5. 若后续出现 teacher-forcing 好但 free rollout 坏，先检查：

```text
COLAR_EMBED_LOSS 是否 mse
COLAR_SQRT_MEAN 是否 1
LATENT_TEMPERATURE 是否 1.0
EOL_TEMPERATURE 是否 1.0
MAX_LATENT_FORWARD 是否大于 gold compressed length
```

6. 原始 CoLaR 默认配置参考：

```text
embed_modeling_loss: mse
sqrt_mean: True
latent_temperature: 1.0
eol_temperature: 1.0
```

这与成功的 r=2 迁移实验一致。

## 2026-06-09 Query-anchor / Res-CoLaR 实验记录

动机：当前 baseline CoLaR 已能在 Qwen3-Omni mixed audio/text overfit 上跑通，但 Qwen3-Omni 的 CoT 很长，latent rollout 较长时仍有 exposure bias 风险。因此新增一个独立 query-anchor 实验路径，尝试在 latent reasoning 阶段引入 query hidden 作为稳定条件。

新增文件均为独立实验路径，没有覆盖 baseline CoLaR 代码：

```text
colar_plugin/query_anchor_policy.py
colar_plugin/query_anchor_trainer.py
colar_plugin/query_anchor_plugin.py
colar_plugin/query_anchor_infer.py
infer_qwen3omni_query_anchor_overfit.py
overfit_check_query_anchor_audio_r5.sh
overfit_check_query_anchor_audio_r10.sh
infer_overfit_check_query_anchor_audio_r5.sh
infer_overfit_check_query_anchor_audio_r10.sh
```

核心实现：

```text
1. latent head 可条件化：
   latent_head input = [h_t, q, h_t * q]

2. residual anchor 可选：
   anchored_latent = (latent + alpha * query_anchor) / sqrt(1 + alpha^2)

3. 训练 target 仍是 clean compressed latent：
   gold = gold_source[t+1] / embeds_std
   不把 query anchor 加到 latent target 里。

4. inference 仍使用 KV cache：
   prompt forward use_cache=True
   每个 latent step 只喂一个 latent embedding + past_key_values
```

重要配置：

```text
QUERY_ANCHOR_CONDITION_HEAD=1  # latent head 看到 query hidden
QUERY_ANCHOR_RESIDUAL=1/0      # 是否把 query residual 加回 latent input
QUERY_ANCHOR_RESIDUAL_SOURCE=hidden/embed
QUERY_ANCHOR_HIDDEN_SOURCE=prompt_last/user_last
QUERY_ANCHOR_GATE_INIT=0.1
QUERY_ANCHOR_NOISE_STD=0.0
GRADIENT_CHECKPOINTING=false   # hidden-source 版本建议关闭
```

关于 `QUERY_ANCHOR_HIDDEN_SOURCE`：

```text
prompt_last:
  取整个 prompt 的最后一个 token hidden，通常是 <think>\n 的换行 token。
  这和 latent rollout 第一步的起点天然对齐。

user_last:
  取 user turn 最后一个 token hidden。
  纯文本样本中是题目最后一个 token；
  音频样本中实际是 <|audio_end|>，因为 user content=<audio> 会展开成
  <|audio_start|><|audio_pad|>...<|audio_end|>。
```

已确认音频 encode 细节：

```text
idx 0 audio:
  audio_pad_count=45
  audio_pad_range=4..48
  user_last anchor=<|audio_end|> at pos 49

idx 1 audio:
  audio_pad_count=257
  audio_pad_range=4..260
  user_last anchor=<|audio_end|> at pos 261

idx 2 text:
  user_last anchor=题目最后一个 token "统一"

idx 3 audio:
  audio_pad_count=118
  audio_pad_range=4..121
  user_last anchor=<|audio_end|> at pos 122
```

因此，音频样本的 `user_last` hidden 理论上能 attend 到前面的 audio soft tokens，但它本身是音频结束符，不一定是最好的 query summary。

### Query-anchor 已完成实验

#### A. hidden residual 版本，旧 full run

checkpoint:

```text
output/qwen3omni_query_anchor_hidden_r5_mse_sqrt/overfit_ckpt_audio_r5/v0-20260608-192850/checkpoint-300
```

注意：该 checkpoint 的 `colar_config.json` 没有 `query_anchor_hidden_source` 字段，因为当时代码尚未显式保存该字段；按旧逻辑，它等价于 `prompt_last`。

配置摘要：

```text
query_anchor_condition_head=true
query_anchor_residual=true
query_anchor_residual_source=hidden
query_anchor_proj_init=identity
query_anchor_alpha ~= 0.1086
query_anchor_noise_std=0.0
```

推理结果：

```text
answer-ish matches: 2/4
eol hits: 2/4

idx 0: audio, hit_eol=True,  n_latent_forward=94,  script match=False
idx 1: audio, hit_eol=False, n_latent_forward=400, script match=False
idx 2: text,  hit_eol=True,  n_latent_forward=121, script match=True
idx 3: audio, hit_eol=False, n_latent_forward=400, script match=False
```

观察：hidden residual 直接加回 latent embedding 后，训练和推理都比 baseline 更不稳。主要怀疑点是 hidden-state 空间和 compressed embedding/latent 空间未必几何对齐，identity projection + alpha=0.1 可能把 latent 拉出原 manifold。

#### B. condition head only，prompt_last，r=5

训练命令：

```bash
QUERY_ANCHOR_CONDITION_HEAD=1 \
QUERY_ANCHOR_RESIDUAL=0 \
QUERY_ANCHOR_HIDDEN_SOURCE=prompt_last \
QUERY_ANCHOR_NOISE_STD=0.0 \
OUTPUT_DIR=output/qwen3omni_query_anchor_condonly_promptlast_r5_mse_sqrt \
bash overfit_check_query_anchor_audio_r5.sh
```

checkpoint:

```text
output/qwen3omni_query_anchor_condonly_promptlast_r5_mse_sqrt/overfit_ckpt_audio_r5/v0-20260608-222109/checkpoint-300
```

config 摘要：

```text
query_anchor_condition_head=true
query_anchor_residual=false
query_anchor_residual_source=hidden
query_anchor_hidden_source=prompt_last
query_anchor_noise_std=0.0
query_anchor_alpha=0.1  # residual=false 时实际不生效
```

推理命令：

```bash
CHECKPOINT_ROOT=output/qwen3omni_query_anchor_condonly_promptlast_r5_mse_sqrt/overfit_ckpt_audio_r5 \
OUTPUT=output/qwen3omni_query_anchor_condonly_promptlast_r5_mse_sqrt/overfit_infer_query_anchor_r5.jsonl \
LIMIT=4 \
MAX_LATENT_FORWARD=400 \
MAX_NEW_TOKENS=512 \
LATENT_TEMPERATURE=1.0 \
EOL_TEMPERATURE=1.0 \
PROGRESS_EVERY=128 \
bash infer_overfit_check_query_anchor_audio_r5.sh
```

推理结果：

```text
answer-ish matches: 3/4
eol hits: 2/4

idx 0: audio, n_latent_forward=94,  hit_eol=True,  answer_response_in_expected=True
idx 1: audio, n_latent_forward=400, hit_eol=False, 内容方向正确但 match=False
idx 2: text,  n_latent_forward=121, hit_eol=True,  answer_response_in_expected=True
idx 3: audio, n_latent_forward=400, hit_eol=False, answer_response_in_expected=True
```

结论：

```text
condition-only 比 hidden residual 版本更稳：
  hidden residual:    answer-ish 2/4, eol 2/4
  condition-only:     answer-ish 3/4, eol 2/4

但仍不如 baseline CoLaR r=5：
  baseline mixed r=5: answer-ish 4/4, eol 3/4
```

当前判断：

```text
1. query hidden 作为 latent head 条件是相对安全的；
2. 直接把 hidden residual 加回 latent embedding 空间大概率有害；
3. 当前 query-anchor 的主要剩余问题是停止预测不稳，尤其音频样本容易跑满 MAX_LATENT_FORWARD=400；
4. 音频 query summary 也可能需要更好的 pooling，而不是只依赖 <|audio_end|> 或 prompt_last 单 token。
```

下一步建议先不重新训练，直接用 condition-only checkpoint 跑 deterministic inference：

```bash
CHECKPOINT_ROOT=output/qwen3omni_query_anchor_condonly_promptlast_r5_mse_sqrt/overfit_ckpt_audio_r5 \
OUTPUT=output/qwen3omni_query_anchor_condonly_promptlast_r5_mse_sqrt/overfit_infer_query_anchor_r5_greedy.jsonl \
LIMIT=4 \
MAX_LATENT_FORWARD=400 \
MAX_NEW_TOKENS=512 \
LATENT_TEMPERATURE=0 \
EOL_TEMPERATURE=0 \
PROGRESS_EVERY=128 \
bash infer_overfit_check_query_anchor_audio_r5.sh
```

如果 greedy 后 EOL 明显改善，问题主要是 sampling 放大了停止漂移；如果 greedy 也不行，再考虑训练侧：

```text
QUERY_ANCHOR_NOISE_STD=0.05/0.1
或单独加强 EOL/stop 监督
或新增 user/audio span pooling 的 query anchor
```

#### B2. condition head only，prompt_last，r=5，greedy inference

推理命令：

```bash
CHECKPOINT_ROOT=output/qwen3omni_query_anchor_condonly_promptlast_r5_mse_sqrt/overfit_ckpt_audio_r5 \
OUTPUT=output/qwen3omni_query_anchor_condonly_promptlast_r5_mse_sqrt/overfit_infer_query_anchor_r5_greedy.jsonl \
LIMIT=4 \
MAX_LATENT_FORWARD=400 \
MAX_NEW_TOKENS=512 \
LATENT_TEMPERATURE=0 \
EOL_TEMPERATURE=0 \
PROGRESS_EVERY=128 \
bash infer_overfit_check_query_anchor_audio_r5.sh
```

推理结果：

```text
answer-ish matches: 4/4
eol hits: 2/4

idx 0: audio, n_latent_forward=94,  hit_eol=True,  answer_match=False, answer_response_in_expected=True
idx 1: audio, n_latent_forward=400, hit_eol=False, answer_match=False, answer_response_in_expected=True
idx 2: text,  n_latent_forward=121, hit_eol=True,  answer_match=True
idx 3: audio, n_latent_forward=400, hit_eol=False, answer_match=True
```

观察：

```text
1. greedy 后答案匹配恢复到 4/4，说明 condition-only checkpoint 的答案内容能力并没有完全坏掉；
2. 但 EOL 仍是 2/4，两个音频样本继续跑满 400 latent step；
3. 因此停止失败不是单纯由 sampling 随机性导致，而是 free rollout 下 stop/EOL 决策本身不稳；
4. 与 baseline mixed r=5 的 4/4 answer-ish、3/4 EOL 相比，condition-only query anchor 仍然没有超过原版 CoLaR。
```

#### B3. query-anchor EOL 失败诊断与 condition input norm 修复

2026-06-09 对 query-anchor 的 EOL 掉点做了 teacher-forcing 诊断。结论不是 prompt/EOL label bug：

```text
old condition-only prompt_last checkpoint:
  gold path teacher forcing 下 4 个样本的 gold </think> rank 都是 1
  prefix/audio merge 也能对齐
```

真正可疑点是 condition head 的输入尺度。旧实现直接拼：

```text
[h, q, h * q]
```

诊断发现 `h` / `q` RMS 约 2-4，但 `h*q` RMS 可到 24-90，interaction 项尺度过大，容易主导 condition head。对应训练上旧 condition-only 的 `embed_loss ~= 0.0500`，明显差于 baseline mixed r=5 的 `embed_loss ~= 0.00618`。

因此新增配置：

```text
QUERY_ANCHOR_CONDITION_INTERACTION=1/0
QUERY_ANCHOR_CONDITION_INPUT_NORM=1/0
```

实现要点：

```text
condition_input_norm=1 时：
  使用 layer_norm(h), layer_norm(q)
  如果保留 interaction，则使用 layer_norm(layer_norm(h) * layer_norm(q))

condition_interaction=0 时：
  condition head 输入只拼 [h, q]
```

相关文件：

```text
diagnose_query_anchor_teacher_forcing.py
colar_plugin/query_anchor_policy.py
colar_plugin/query_anchor_trainer.py
colar_plugin/query_anchor_infer.py
overfit_check_query_anchor_audio_r5.sh
overfit_check_query_anchor_audio_r10.sh
```

已验证：

```text
python -m py_compile ... 通过
bash -n overfit/infer query-anchor 脚本通过
旧 checkpoint 兼容加载通过
```

#### B4. condition norm 五组 r=5 复测

五个 condition-norm r=5 实验均训练到 checkpoint-300，并用相同推理设置复测：

```text
LIMIT=4
MAX_LATENT_FORWARD=400
MAX_NEW_TOKENS=512
LATENT_TEMPERATURE=1.0
EOL_TEMPERATURE=1.0
```

实验列表：

```text
1. output/qwen3omni_query_anchor_condnorm_promptlast_r5_mse_sqrt
   condition_interaction=true,  condition_input_norm=true, residual=false, hidden_source=prompt_last

2. output/qwen3omni_query_anchor_condnorm_nointer_promptlast_r5_mse_sqrt
   condition_interaction=false, condition_input_norm=true, residual=false, hidden_source=prompt_last

3. output/qwen3omni_query_anchor_condnorm_userlast_r5_mse_sqrt
   condition_interaction=true,  condition_input_norm=true, residual=false, hidden_source=user_last

4. output/qwen3omni_query_anchor_condnorm_nointer_userlast_r5_mse_sqrt
   condition_interaction=false, condition_input_norm=true, residual=false, hidden_source=user_last

5. output/qwen3omni_query_anchor_condnorm_promptlast_noise005_r5_mse_sqrt
   condition_interaction=true,  condition_input_norm=true, residual=false, hidden_source=prompt_last, noise_std=0.05
```

五组推理结果完全一致：

```text
answer-ish: 4/4
eol hits:   3/4
latent steps: [94, 400, 121, 104]

idx 0: audio, n_latent_forward=94,  hit_eol=True
idx 1: audio, n_latent_forward=400, hit_eol=False
idx 2: text,  n_latent_forward=121, hit_eol=True
idx 3: audio, n_latent_forward=104, hit_eol=True
```

对比：

```text
baseline mixed r=5:
  answer-ish 4/4, eol 3/4, steps [94, 400, 121, 104]

old condition-only prompt_last:
  sampled: answer-ish 3/4, eol 2/4, steps [94, 400, 121, 400]
  greedy:  answer-ish 4/4, eol 2/4, steps [94, 400, 121, 400]
```

训练末尾 loss：

```text
baseline mixed r=5:
  ce_loss ~= 0.3054, embed_loss ~= 0.00618

old condition-only prompt_last:
  ce_loss ~= 0.3444, embed_loss ~= 0.0500

new condnorm r=5:
  condnorm_promptlast:              ce_loss ~= 0.3065, embed_loss ~= 0.0181
  condnorm_nointer_promptlast:      ce_loss ~= 0.3058, embed_loss ~= 0.0168
  condnorm_userlast:                ce_loss ~= 0.3066, embed_loss ~= 0.0280
  condnorm_nointer_userlast:        ce_loss ~= 0.3060, embed_loss ~= 0.0232
  condnorm_promptlast_noise005:     ce_loss ~= 0.3044, embed_loss ~= 0.0173
```

结论：

```text
1. condition_input_norm 修掉了旧 query-anchor 在 idx 3 上的 EOL 退化；
2. query-anchor condition-only 不再拖 baseline r=5 后腿；
3. 但 condition-only r=5 也没有超过 baseline，idx 1 仍是 baseline 自身也失败的 400-step EOL miss；
4. no-inter + prompt_last 的 embed_loss 最低，后续默认优先使用：
   QUERY_ANCHOR_CONDITION_INTERACTION=0
   QUERY_ANCHOR_CONDITION_INPUT_NORM=1
   QUERY_ANCHOR_RESIDUAL=0
   QUERY_ANCHOR_HIDDEN_SOURCE=prompt_last
```

#### B5. 新五组：r10/nores 与 residual anchor 对照

随后又跑了五组新实验，并用五张卡并行推理。

推理设置：

```text
LIMIT=4
MAX_LATENT_FORWARD=400
MAX_NEW_TOKENS=512
LATENT_TEMPERATURE=1.0
EOL_TEMPERATURE=1.0
```

实验与结果：

```text
r10_nores:
  dir: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_r10_mse_sqrt
  checkpoint: overfit_ckpt_audio_r10/v0-20260609-134706/checkpoint-300
  config: r=10, condition_input_norm=true, condition_interaction=false, residual=false
  answer-ish: 4/4
  eol hits:   4/4
  steps: [47, 178, 61, 52]

r5_res_hidden:
  dir: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_residual_r5_mse_sqrt
  checkpoint: overfit_ckpt_audio_r5/v0-20260609-134733/checkpoint-300
  config: r=5, residual=true, residual_source=hidden, gate_init=0.1
  answer-ish: 3/4
  eol hits:   3/4
  steps: [94, 400, 121, 154]

r10_res_hidden:
  dir: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_residual_r10_mse_sqrt
  checkpoint: overfit_ckpt_audio_r10/v0-20260609-134908/checkpoint-300
  config: r=10, residual=true, residual_source=hidden, gate_init=0.1
  answer-ish: 3/4
  eol hits:   2/4
  steps: [47, 400, 61, 400]

r5_res_embed:
  dir: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_residual_embed_r5_mse_sqrt
  checkpoint: overfit_ckpt_audio_r5/v0-20260609-134921/checkpoint-300
  config: r=5, residual=true, residual_source=embed, gate_init=0.1
  answer-ish: 4/4
  eol hits:   3/4
  steps: [94, 400, 121, 104]

r5_res_hidden_g003:
  dir: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_residual_g003_r5_mse_sqrt
  checkpoint: overfit_ckpt_audio_r5/v0-20260609-134930/checkpoint-300
  config: r=5, residual=true, residual_source=hidden, gate_init=0.03
  answer-ish: 4/4
  eol hits:   1/4
  steps: [94, 400, 400, 400]
```

baseline 对照：

```text
baseline mixed r=10:
  answer-ish 4/4, eol 4/4, steps [47, 178, 61, 52]

baseline mixed r=5:
  answer-ish 4/4, eol 3/4, steps [94, 400, 121, 104]
```

训练末尾 loss：

```text
r10_nores:
  ce_loss ~= 0.2438, embed_loss ~= 0.00718

r5_res_hidden:
  ce_loss ~= 0.3352, embed_loss ~= 0.05844

r10_res_hidden:
  ce_loss ~= 0.2842, embed_loss ~= 0.04013

r5_res_embed:
  ce_loss ~= 0.3062, embed_loss ~= 0.01681

r5_res_hidden_g003:
  ce_loss ~= 0.3297, embed_loss ~= 0.06205
```

结论：

```text
1. r10_nores 完全追平 baseline r=10，说明 condition-only + input_norm + no-inter 是安全主线；
2. hidden residual 是明显负向：
   - r10 baseline/nores 能 4/4 EOL，但 r10_res_hidden 只有 2/4 EOL；
   - r5_res_hidden 在 idx 1 上 answer 也掉到 False；
   - hidden residual 的 embed_loss 明显升高到 0.04-0.06；
3. embed residual 没明显伤害，但也没有收益，基本复刻 baseline r=5；
4. 小 gate=0.03 并未解决 hidden residual 问题，反而 EOL 更差；
5. 当前主线应收敛到 condition-only，不建议继续直接走 hidden residual anchor。
```

#### B6. residual anchor gate 训练后数值

直接读取 checkpoint-300 的 `latent_policy.pt` 中 `anchor_gate`，确认 gate 是可训练且有更新的。注意：当前 `colar_config.json` 没保存 `query_anchor_gate_init`，只保存了 `query_anchor_gate_l2`；真实 gate 值需从 `latent_policy.pt` 读取。

checkpoint-300 结果：

```text
r5_res_hidden_g010:
  init=0.10
  checkpoint-300 anchor_gate=0.108856268
  delta=+0.008856

r10_res_hidden_g010:
  init=0.10
  checkpoint-300 anchor_gate=0.108835079
  delta=+0.008835

r5_res_embed_g010:
  init=0.10
  checkpoint-300 anchor_gate=0.083219431
  delta=-0.016781

r5_res_hidden_g003:
  init=0.03
  checkpoint-300 anchor_gate=0.047263760
  delta=+0.017264
```

gate 曲线：

```text
r5_res_hidden_g010:
  ckpt-50  0.108497441
  ckpt-100 0.109239988
  ckpt-150 0.109544449
  ckpt-200 0.108642094
  ckpt-250 0.109050550
  ckpt-300 0.108856268

r10_res_hidden_g010:
  ckpt-50  0.107159667
  ckpt-100 0.107834123
  ckpt-150 0.108418323
  ckpt-200 0.108574271
  ckpt-250 0.108885556
  ckpt-300 0.108835079

r5_res_embed_g010:
  ckpt-50  0.091047741
  ckpt-100 0.086415298
  ckpt-150 0.085062288
  ckpt-200 0.084085286
  ckpt-250 0.083329201
  ckpt-300 0.083219431

r5_res_hidden_g003:
  ckpt-50  0.040755257
  ckpt-100 0.043727878
  ckpt-150 0.046002377
  ckpt-200 0.046821374
  ckpt-250 0.047472868
  ckpt-300 0.047263760
```

gate 观察：

```text
1. hidden residual 的 gate 会被训练往上推：
   0.10 -> ~0.109；0.03 -> ~0.047
2. 但 hidden residual 推理表现更差，说明优化正在增强一个对 free rollout 有害的方向；
3. embed residual 的 gate 会被训练往下压：
   0.10 -> ~0.083
4. 小 gate 仍然被推大且 EOL 更差，所以问题不是简单的 gate 初始值过大，而是 hidden residual 方向/空间对齐本身有问题。
```

后续记录改进建议：

```text
把 query_anchor_gate_init 写入 colar_config.json，避免之后只能从 latent_policy.pt 反查。
```

#### B7. r=5 EOL miss 的训练侧消融

在确认 residual anchor 不值得继续走后，围绕 `condition-only + input_norm + no-inter + no residual` 继续做 r=5 训练侧消融，目标是修复 mixed r=5 中 `idx=1` 的 `400-step EOL miss`。

共同配置：

```text
QUERY_ANCHOR_CONDITION_HEAD=1
QUERY_ANCHOR_CONDITION_INTERACTION=0
QUERY_ANCHOR_CONDITION_INPUT_NORM=1
QUERY_ANCHOR_RESIDUAL=0
QUERY_ANCHOR_HIDDEN_SOURCE=prompt_last
COLAR_EMBED_LOSS=mse
COLAR_SQRT_MEAN=1
```

五组实验：

```text
r5_s600:
  dir: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_r5_s600_mse_sqrt
  checkpoint: overfit_ckpt_audio_r5/v0-20260609-163320/checkpoint-600
  change: MAX_STEPS=600

r5_embedw2:
  dir: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_r5_embedw2_mse_sqrt
  checkpoint: overfit_ckpt_audio_r5/v0-20260609-163334/checkpoint-300
  change: COLAR_EMBED_WEIGHT=2.0

r5_embedw4:
  dir: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_r5_embedw4_mse_sqrt
  checkpoint: overfit_ckpt_audio_r5/v0-20260609-163343/checkpoint-300
  change: COLAR_EMBED_WEIGHT=4.0

r5_noise005:
  dir: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_r5_noise005_mse_sqrt
  checkpoint: overfit_ckpt_audio_r5/v0-20260609-163350/checkpoint-300
  change: QUERY_ANCHOR_NOISE_STD=0.05

r5_deterministic:
  dir: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_r5_deterministic_mse_sqrt
  checkpoint: overfit_ckpt_audio_r5/v0-20260609-163358/checkpoint-300
  change: COLAR_DETERMINISTIC=1
```

推理设置：

```text
LIMIT=4
MAX_LATENT_FORWARD=400
MAX_NEW_TOKENS=512
LATENT_TEMPERATURE=1.0
EOL_TEMPERATURE=1.0
```

推理结果：

```text
r5_s600:
  answer-ish: 4/4
  eol hits:   4/4
  steps: [94, 356, 121, 104]

r5_embedw2:
  answer-ish: 4/4
  eol hits:   4/4
  steps: [94, 356, 121, 104]

r5_embedw4:
  answer-ish: 4/4
  eol hits:   4/4
  steps: [94, 356, 121, 104]

r5_noise005:
  answer-ish: 4/4
  eol hits:   3/4
  steps: [94, 400, 121, 104]

r5_deterministic:
  answer-ish: 4/4
  eol hits:   3/4
  steps: [94, 400, 121, 104]
```

baseline / previous 对照：

```text
baseline mixed r=5:
  answer-ish 4/4, eol 3/4, steps [94, 400, 121, 104]

previous condnorm nores r=5:
  answer-ish 4/4, eol 3/4, steps [94, 400, 121, 104]

baseline mixed r=10:
  answer-ish 4/4, eol 4/4, steps [47, 178, 61, 52]
```

训练末尾 loss：

```text
r5_s600:
  ce_loss ~= 0.3011, embed_loss ~= 0.000817

r5_embedw2:
  ce_loss ~= 0.3046, embed_loss ~= 0.007464

r5_embedw4:
  ce_loss ~= 0.3074, embed_loss ~= 0.003672

r5_noise005:
  ce_loss ~= 0.3041, embed_loss ~= 0.016016

r5_deterministic:
  ce_loss ~= 0.3050, embed_loss ~= 0.010648
```

结论：

```text
1. r=5 的 idx=1 EOL miss 可以通过更强 latent 对齐修掉，不需要 residual anchor；
2. MAX_STEPS=600 最稳，embed_loss 压到 0.000817，并得到 4/4 EOL；
3. 从训练成本看，COLAR_EMBED_WEIGHT=2.0 更划算：300 step 就得到 4/4 EOL；
4. COLAR_EMBED_WEIGHT=4.0 也能 4/4 EOL，但 ce_loss 稍高，没有明显优先级；
5. QUERY_ANCHOR_NOISE_STD=0.05 和 COLAR_DETERMINISTIC=1 没修掉 idx=1 EOL miss，暂时不作为主线。
```

后续推荐主线：

```text
QUERY_ANCHOR_CONDITION_HEAD=1
QUERY_ANCHOR_CONDITION_INTERACTION=0
QUERY_ANCHOR_CONDITION_INPUT_NORM=1
QUERY_ANCHOR_RESIDUAL=0
QUERY_ANCHOR_HIDDEN_SOURCE=prompt_last
COLAR_EMBED_WEIGHT=2.0
```

#### B8. RMS drift 诊断、推理 RMS renorm 与 scheduled sampling 接线

按新的排查路线先做了 free-rollout latent RMS 诊断和 EOL seed sweep：

```text
diagnostic:
  script: diagnose_latent_rms_drift.py
  output: output/qwen3omni_colar_mixed_r5_mse_sqrt/rms_drift_diag_r5.jsonl
  checkpoint: output/qwen3omni_colar_mixed_r5_mse_sqrt/overfit_ckpt_audio_r5/v0-20260607-235138/checkpoint-300

seed sweep:
  script: sweep_eol_seeds.py
  output: output/qwen3omni_colar_mixed_r5_mse_sqrt/eol_seed_sweep_idx1.jsonl
  sample: idx=1
  seeds: 0..9
```

关键诊断结果：

```text
idx=0: gold_rms=1.0080, tf_pred_rms=1.0075, ratio=0.999, rollout_cos_mean=0.9995
idx=1: gold_rms=0.9063, tf_pred_rms=0.8934, ratio=0.986, rollout_cos_mean=0.9857
idx=2: gold_rms=1.0017, tf_pred_rms=1.0001, ratio=0.998, rollout_cos_mean=0.9988
idx=3: gold_rms=0.7918, tf_pred_rms=0.7893, ratio=0.997, rollout_cos_mean=0.9970
```

idx=1 的 10-seed sweep：

```text
P(hit_eol) = 8/10
hit seeds: steps all 356
miss seeds: 2, 3
miss seed 2: eol_logprob@gold_end=-22.16, cos_gold_last8 ~= 0.15..0.35
miss seed 3: eol_logprob@gold_end=-23.29, cos_gold_last8 ~= 0.10..0.31
```

结论：

```text
1. baseline r=5 的 free rollout 没有系统性 RMS 漂移；teacher-forced pred RMS 与 gold RMS 基本贴合。
2. idx=1 的 EOL miss 是采样后方向崩掉/knife-edge stop 事件，不是模长单调缩放问题。
3. 因此推理时 RMS renorm 可以保留为零成本对照，但不是主线解法。
4. 主线应继续做 scheduled sampling 或更强 latent 对齐（例如已验证的 COLAR_EMBED_WEIGHT=2.0）。
```

已接线的推理 RMS renorm：

```text
baseline infer:
  colar_plugin/colar_infer.py
  infer_qwen3omni_colar_overfit.py
  infer_overfit_check_audio_r5.sh
  infer_overfit_check_audio_r10.sh

query-anchor infer:
  colar_plugin/query_anchor_infer.py
  infer_qwen3omni_query_anchor_overfit.py
  infer_overfit_check_query_anchor_audio_r5.sh
  infer_overfit_check_query_anchor_audio_r10.sh

env/CLI:
  LATENT_RMS_TARGET / --latent-rms-target
  0 = disabled
```

scheduled sampling 实现：

```text
env:
  COLAR_SS_PROB
  COLAR_SS_TEMPERATURE
  COLAR_SS_WARMUP_STEPS
  COLAR_SS_CACHE_MAX

implementation:
  cached scheduled sampling, not same-step extra text-model forward
  reuse current latent-loss prediction to populate a detached per-sample cache
  next visit to the same sample can replace compressed latent input positions
  no extra backbone forward, so it works with gradient_checkpointing
```

为什么用 cached 版本：

```text
原始 same-step no-grad text-model forward 在 Qwen3-Omni 30B 上会 OOM，并且容易和
gradient checkpointing 的 recompute 记账冲突。cached 版本保持每个训练 step 只有一次
主干 forward，适合当前 4 条 overfit 数据反复训练的场景。
```

smoke 结果：

```text
wrong topology smoke:
  CUDA_VISIBLE_DEVICES=0,1,2,3
  NPROC_PER_NODE=4
  result: OOM at optimizer state init
  note: this is 4-card pure DDP, not the established 8-card DDP=4/MP=2 topology.

correct topology smoke:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  NPROC_PER_NODE=4
  output_dir: output/qwen3omni_colar_ss_cached_smoke
  checkpoint: overfit_ckpt_audio_r5/v1-20260610-120507/checkpoint-3
  COLAR_SS_PROB=0.25
  MAX_STEPS=3
  result: pass, checkpoint saved
```

3-step smoke losses:

```text
step 1: loss=4.9551, ce_loss=1.8194, embed_loss=3.1356
step 2: loss=4.9839, ce_loss=1.8451, embed_loss=3.1388
step 3: loss=3.0122, ce_loss=1.5622, embed_loss=1.4501
train_loss=4.3171
```

Next recommended run:

```text
DATASET_PATH=output/qwen3omni_colar/overfit_mixed_audio_text.jsonl \
OUTPUT_DIR=output/qwen3omni_colar_ss025_r5_mse_sqrt \
COLAR_SS_PROB=0.25 \
COLAR_SS_TEMPERATURE=1.0 \
COLAR_SS_WARMUP_STEPS=20 \
COLAR_EMBED_WEIGHT=2.0 \
MAX_STEPS=300 \
SAVE_STEPS=50 \
SAVE_TOTAL_LIMIT=6 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash overfit_check_audio_r5.sh
```

2026-06-10 五实验训练后推理结果：

```text
dataset: output/qwen3omni_colar/overfit_mixed_audio_text.jsonl
limit: 4
inference setting:
  MAX_LATENT_FORWARD=400
  MAX_NEW_TOKENS=512
  LATENT_TEMPERATURE=1.0
  EOL_TEMPERATURE=1.0
  LATENT_RMS_TARGET=0

parallelism:
  five inference processes were run concurrently on GPU0..GPU4
```

实验 checkpoint：

```text
ss010:
  checkpoint: output/qwen3omni_colar_ss010_r5_mse_sqrt/overfit_ckpt_audio_r5/v0-20260610-122331/checkpoint-300
  infer output: output/qwen3omni_colar_ss010_r5_mse_sqrt/overfit_infer_r5.jsonl

ss025:
  checkpoint: output/qwen3omni_colar_ss025_r5_mse_sqrt/overfit_ckpt_audio_r5/v0-20260610-122346/checkpoint-300
  infer output: output/qwen3omni_colar_ss025_r5_mse_sqrt/overfit_infer_r5.jsonl

ss010_embedw2:
  checkpoint: output/qwen3omni_colar_ss010_embedw2_r5_mse_sqrt/overfit_ckpt_audio_r5/v0-20260610-122356/checkpoint-300
  infer output: output/qwen3omni_colar_ss010_embedw2_r5_mse_sqrt/overfit_infer_r5.jsonl

ss025_embedw2:
  checkpoint: output/qwen3omni_colar_ss025_embedw2_r5_mse_sqrt/overfit_ckpt_audio_r5/v0-20260610-122422/checkpoint-300
  infer output: output/qwen3omni_colar_ss025_embedw2_r5_mse_sqrt/overfit_infer_r5.jsonl

query_anchor_ss010_embedw2:
  checkpoint: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_ss010_embedw2_r5_mse_sqrt/overfit_ckpt_audio_r5/v0-20260610-122433/checkpoint-300
  infer output: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_ss010_embedw2_r5_mse_sqrt/overfit_infer_query_anchor_r5.jsonl
```

单次推理结果：

```text
all five experiments:
  eol hits: 4/4
  answer-ish matches: 4/4
  latent steps: [94, 356, 121, 104]

match detail:
  response_in_full_expected: 4/4
  answer_response_in_expected: 4/4
  answer_expected_in_response: 2/4
  full_expected_in_response: 0/4
```

解释：

```text
日志中的 answer_match=False/False/True/True 是较严格口径；
JSONL 的 answer-ish 口径只要 answer_response_in_expected 或
answer_expected_in_response 为真即算匹配，因此五个实验都是 4/4。
```

idx=1 seed sweep：

```text
script: sweep_eol_seeds.py
sample: idx=1
seeds: 0..9
parallelism: five sweep processes were run concurrently on GPU0..GPU4

code update:
  sweep_eol_seeds.py now supports:
    --backend colar
    --backend query_anchor
    --latent-rms-target
  query-anchor sweep uses the same query_hidden/residual-anchor path as
  colar_plugin/query_anchor_infer.py.
```

10-seed EOL sweep results：

```text
ss010:
  output: output/qwen3omni_colar_ss010_r5_mse_sqrt/eol_seed_sweep_idx1.jsonl
  P(hit_eol): 10/10
  steps: 356..356
  gold-end eol logprob mean: -2.16e-05
  best pre-gold-end eol logprob max: -16.10

ss025:
  output: output/qwen3omni_colar_ss025_r5_mse_sqrt/eol_seed_sweep_idx1.jsonl
  P(hit_eol): 10/10
  steps: 356..356
  gold-end eol logprob mean: -1.06e-04
  best pre-gold-end eol logprob max: -13.59

ss010_embedw2:
  output: output/qwen3omni_colar_ss010_embedw2_r5_mse_sqrt/eol_seed_sweep_idx1.jsonl
  P(hit_eol): 10/10
  steps: 356..356
  gold-end eol logprob mean: -8.45e-06
  best pre-gold-end eol logprob max: -15.32

ss025_embedw2:
  output: output/qwen3omni_colar_ss025_embedw2_r5_mse_sqrt/eol_seed_sweep_idx1.jsonl
  P(hit_eol): 10/10
  steps: 356..356
  gold-end eol logprob mean: -1.85e-06
  best pre-gold-end eol logprob max: -15.17

query_anchor_ss010_embedw2:
  output: output/qwen3omni_query_anchor_condnorm_nointer_promptlast_ss010_embedw2_r5_mse_sqrt/eol_seed_sweep_idx1.jsonl
  P(hit_eol): 10/10
  steps: 356..356
  gold-end eol logprob mean: -2.02e-05
  best pre-gold-end eol logprob max: -14.49
```

当前结论：

```text
1. 原始 mixed r=5 baseline 在 idx=1 上是 8/10 EOL hit；这五个 scheduled-sampling
   版本均达到 10/10。
2. 五个实验的 idx=1 都精确在 gold compressed length 356 处停止，没有早停。
3. 因此 cached scheduled sampling 对该随机 EOL miss 有明确修复效果。
4. 单看当前 4-sample overfit 和 idx=1 10-seed sweep，五个实验都已满分，无法证明
   query-anchor condition head 在 overfit setting 下有额外收益。
5. 从 gold-end EOL spike 强度看，ss025_embedw2 最强；但这只是 overfit 诊断指标，
   不能直接外推到全量数据。
```
