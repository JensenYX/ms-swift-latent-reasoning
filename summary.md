# Qwen3-Omni CoLaR Latent Reasoning 进展总结

更新时间：2026-06-07

仓库：

```text
/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/git_warehouse/ms-swift-latent-reasoning
```

原始 CoLaR 仓库：

```text
/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/git_warehouse/colar
```

目标：把 CoLaR latent reasoning 方法迁移到 Qwen3-Omni。当前 overfit 检查只做纯 text 输入、纯 text 输出，不涉及音频。

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

6. 这说明之前不是简单的 label/prompt/checkpoint 加载 bug，而主要是训练目标/压缩归一化/采样方式导致 free rollout 不稳定。

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
