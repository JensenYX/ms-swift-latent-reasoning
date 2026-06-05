# 用 ms-swift 给 Qwen3-Omni 实现 CoLaR (SFT)

## 目标与范围

把 CoLaR 的 **latent reasoning 压缩 SFT** 移植到 ms-swift + Qwen3-Omni-30B-A3B。

**本次范围（已与用户确认）：**
- ✅ 只做 **SFT 训练**（embedding 压缩 + CE loss + latent head NLL loss + 保存 latent head）。
- ✅ 用 Qwen3-Omni 原生 `<think>...</think>` 块承载要压缩的 reasoning（不照搬 CoLaR 的 `###/Answer:` 脚手架）。
- ✅ **里程碑 1 先纯文本跑通核心机制**；音频 merge / mrope 放里程碑 2。
- ❌ 不做 RL（GRPO 对开放式中文任务不可用）。
- ❌ 本次不做 latent 自回归推理生成（`latent_generate`），留作后续。

**关键技术决定（已由代码确认）：**
1. `--external_plugins <file.py>` 在训练前 import 任意 py 文件 → 用它注册自定义 Template + 替换 Trainer，**不改 swift 源码**。(`swift/arguments/base_args/base_args.py:90,136`)
2. `--loss_type` 的 loss plugin **不够用**：它只在 `model(**inputs)` 之后拿到 `(outputs, labels)`，无法控制 forward（压缩 / 音频 merge / latent head / 第二次 forward 都需要改 forward）。(`swift/loss/base.py:10`，`swift/trainers/seq2seq_trainer.py:139,192`) → 必须**自定义 Trainer 覆写 `compute_loss`**。
3. ms-swift **不会自动**把 `reasoning_content` 拼进训练文本：`_swift_encode` 只 tokenize `message['content']`（`swift/template/base.py:1281`）。→ 需要一个数据准备步骤把 reasoning 合进 content。
4. Trainer 类由 `TrainerFactory.get_trainer_cls(args)` 选取，`SwiftSft.run()` 里 `trainer_cls(...)` 实例化（`swift/pipelines/train/sft.py:188`）。`TRAINER_MAPPING` 是硬编码 dict（`swift/trainers/trainer_factory.py:14`）→ 用 plugin 在 import 时**改写这个 dict**指向我们的子类。
5. PEFT 保存只存 LoRA adapter（`swift/trainers/mixin.py` `_save_model` → `save_pretrained(selected_adapters=['default'])`）→ latent head 要在 `_save_model` override 里 `torch.save` 额外存。
6. position_ids：无图像/视频时 `get_rope_index` 走 `else` 分支 = `cumsum(attention_mask)-1` 广播成 4 行（`qwen3omni_model/modeling_qwen3_omni_moe.py:296,505`）。压缩后只需在新 mask 上重新 cumsum。

---

## 文件清单（全部新建，零改动 swift 源码）

```
ms-swift-latent-reasoning/
├── colar_plugin/
│   ├── __init__.py
│   ├── prepare_data_colar.py     # 数据转换：reasoning_content -> <think> 块
│   ├── latent_policy.py          # LatentPolicy MLP（从 colar 移植）
│   ├── colar_template.py         # 自定义 Template：切 think 段边界 + 音频
│   ├── colar_trainer.py          # 自定义 Trainer：compute_loss 压缩+双loss / _save_model
│   └── plugin.py                 # 入口：注册 template + 改写 TRAINER_MAPPING
├── train_qwen3omni_colar.sh      # 里程碑1：纯文本启动脚本
└── (里程碑2 再在 sh 里加音频数据)
```

---

## 数据格式

**输入**（你的 jsonl，已确认字段）：
```json
{"messages":[
  {"role":"user","content":"...","wav_path":"/path.wav"},
  {"role":"assistant","content":"最终回答","reasoning_content":"思考过程"}
]}
```

**转换后（里程碑1 纯文本）：** `prepare_data_colar.py` 产出 ms-swift 标准格式，把 reasoning 包进 `<think>`：
```json
{"messages":[
  {"role":"user","content":"<问题文本>"},
  {"role":"assistant","content":"<think>\n思考过程\n</think>\n\n最终回答"}
]}
```
**里程碑2（音频）：** user 用 `<audio>` 占位 + `audios:[wav_path]`，assistant 同上。

> 为什么要 `<think>` 包裹：`Qwen3OmniTemplate` 继承链会按 thinking 模板处理 `<think>...</think>`，token 边界与预训练一致；我们在 Template 里再用 think 块的 token 边界切出"要压缩的段"。

---

## 序列结构（Qwen3-Omni 版的 CoLaR 三段）

CoLaR 原版： `[question][steps][answer]`，压缩中间 steps。
我们映射到 Qwen chat 序列上的三段（按 token 区间）：

```
[ ...system+user+assistant_header...  <think>\n ] [ 思考 tokens ] [ \n</think>\n\n + 最终回答 + eos ]
└──────────── PREFIX 段（不压缩，不算CE）──────────┘ └─ THINK 段(压缩) ─┘ └──────── ANSWER 段（不压缩，算CE）────────┘
```

- **PREFIX 段**：到 `<think>\n` 为止。含音频（里程碑2）。labels 全 -100。
- **THINK 段**：`<think>` 与 `</think>` 之间的 reasoning tokens。**这是要做 embedding 压缩的部分**。
- **ANSWER 段**：`</think>` 起到 eos。labels 监督。

Template 的核心职责：在 encode 后，**额外吐出 THINK 段的 `[start, end)` 边界索引**（以及 PREFIX/ANSWER 长度），供 Trainer 做压缩。

---

## 里程碑 1：纯文本跑通核心机制

### M1.1 `latent_policy.py`
直接移植 `colar/src/modules/projector.py` 的 `LatentPolicy`（fc: Linear→GELU→Linear→LayerNorm；mean head；log_std head；`forward` 返回 `Normal(mean, std)`，deterministic 时 std=1e-9）。新增一个工具函数 `compute_embeds_std(model)`：
```python
emb = model.get_input_embeddings().weight
embeds_std = emb.float().view(-1).std().item()  # Qwen3-Omni 自己的值，替代硬编码 MODEL_EMB_STD
```

### M1.2 `colar_template.py`
- 子类化 `Qwen3OmniTemplate`（里程碑1纯文本也走它，音频字段自然为空）。
- 覆写 `_encode`：先 `encoded = super()._encode(inputs)` 拿到 `input_ids/labels`，然后在 `input_ids` 里定位 `<think>` / `</think>` 的 token 位置，写入：
  - `encoded['think_start']`, `encoded['think_end']`（THINK 段区间）。
  - 用 token id 定位：`think_open_id = tokenizer.encode('<think>')`，`think_close_id = tokenizer.encode('</think>')`（实现时确认是否单 token / 多 token，做子串匹配）。
- 覆写 `_data_collator`：`super()` 之后，把每条样本的 `think_start/think_end` 收集成 batch 张量（含 padding 偏移修正），放进 batch dict 传给 Trainer。

> 注意：collator 会做 padding，think_start/end 必须按**最终 padding 后**的位置对齐（左/右 pad 影响偏移）。Qwen 模板默认右 pad，相对 question 起点的偏移稳定，但要在 collator 里按实际 pad 重算。

### M1.3 `colar_trainer.py` —— 核心
子类 `Seq2SeqTrainer`，覆写 `compute_loss(self, model, inputs, ...)`：

```
1. 取出 think_start/think_end，从 inputs 弹出 labels。
2. inputs_embeds = base_model.get_input_embeddings()(input_ids)        # 纯文本：查表即可
3. 采样压缩率 r = randint(1, max_compression_factor)
4. 对每条样本的 THINK 段 [start,end) 做压缩（CoLaR r>1 算法）：
   - 取 think 段 embeds，按 r 分组、pad 到 r 的倍数、reshape (·, L/r, r, H)、mean-pool
   - 重建该样本序列： [prefix_embeds | compressed_think_embeds | answer_embeds]
   - THINK 段被压短 → 整条序列变短 → 需要逐样本重建后再 pad 成 batch
   - labels：prefix=-100；compressed think 段=每组随机采 1 个 token（sample_indices_from_attention_mask_3d）；answer 段=原 labels
5. position_ids = cumsum(new_attention_mask)-1，广播成 4 行（mrope 退化）
6. 第一次 forward： outputs = model(inputs_embeds=..., attention_mask=..., position_ids=..., labels=new_labels, output_hidden_states=True)
   ce_loss = outputs.loss
7. latent head loss：
   - steps_hidden = outputs.hidden_states[-1][think 段区间]
   - dist = latent_policy(steps_hidden)
   - gold = inputs_embeds[think 段 +1 偏移]   # 预测"下一个压缩 embedding"
   - nll = -dist.log_prob(gold.detach()/embeds_std).mean(-1)，按 think mask 平均
8. total = ce_weight*ce_loss + embed_modeling_weight*nll  (+ 可选 entropy)
9. return (total, outputs) if return_outputs else total
```

实现要点：
- `r==1` 时退化为不压缩（直接整段做 CE + latent head），先用 `r==1` 把管线打通，再开 `r>1`。
- `latent_policy` 作为 `self.latent_policy` 挂在 trainer，`__init__` 里 `compute_embeds_std` 并 `.to(model.device/dtype)`；用 `model.get_input_embeddings()` 拿 hidden_size。
- latent_policy 的参数要进 optimizer：override `create_optimizer` 或在 trainer init 后把 `self.latent_policy.parameters()` 加入。**最稳妥**：把 latent_policy 注册为 model 的子模块（`model.latent_policy = ...`）并设 `requires_grad=True`，这样 HF optimizer 自动收（需确认 LoRA 冻结逻辑不会把它冻掉——它不在 LoRA target 里，默认 requires_grad 由我们显式设 True）。
- `_save_model` override：`super()._save_model(...)` 后 `torch.save(self.latent_policy.state_dict(), output_dir/'latent_policy.pt')`。

### M1.4 `plugin.py` —— 注册入口
```python
from swift.template import register_template
from swift.trainers import trainer_factory
from .colar_template import ColarQwen3OmniTemplate, COLAR_TEMPLATE_META
from .colar_trainer import ColarSeq2SeqTrainer

register_template(COLAR_TEMPLATE_META, exist_ok=True)
# 改写工厂指向我们的 trainer
trainer_factory.TrainerFactory.TRAINER_MAPPING['causal_lm'] = \
    'colar_plugin.colar_trainer.ColarSeq2SeqTrainer'
```
（也可直接 monkeypatch `get_trainer_cls`；二选一，改 MAPPING 最简单。）

### M1.5 `train_qwen3omni_colar.sh`
基于现有 `train_qwen3omni_lora.sh`，关键增量：
```bash
swift sft \
  --model $MODEL_PATH --model_type qwen3_omni_moe \
  --template colar_qwen3_omni \                 # 我们注册的
  --external_plugins $SCRIPT_DIR/colar_plugin/plugin.py \
  --dataset $COLAR_TEXT_DATA \                  # prepare_data 产出的纯文本版
  --tuner_type lora --target_modules all-linear \
  --freeze_vit true --freeze_aligner true \
  ... (其余超参沿用) \
  --max_compression_factor 4 \                  # 通过 args 传，或写死在 plugin 读环境变量
  --embed_modeling_weight 1.0 --ce_weight 1.0
```
> CoLaR 专属超参（max_compression_factor / 各 loss 权重）ms-swift 不认识 → 用**环境变量**或在 plugin.py 里读一个 json config，避免改 argument dataclass。

### M1.6 验证标准（纯文本）
- 能正常启动、loss 下降；`ce_loss` 和 `embed_modeling_loss` 分别打印。
- `r==1` 与 `r=4` 都能跑一个 epoch 不报错。
- checkpoint 目录里同时有 `adapter_model.safetensors` 和 `latent_policy.pt`。
- 压缩后序列长度确实变短（日志打印平均 think 段压缩比）。

---

## 里程碑 2：接入音频 + mrope

### M2.1 音频先 merge
THINK / ANSWER 是纯文本（查表 OK），**只有 PREFIX 段含 `<audio>`**。在 compute_loss 里构造 inputs_embeds 时：
```python
emb = base.get_input_embeddings()(input_ids)                 # 文字对，音频位是垃圾
if input_features is not None:
    audio_feats = thinker.get_audio_features(input_features, feature_attention_mask, audio_feature_lengths).last_hidden_state
    _,_,audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=emb)
    emb = emb.masked_scatter(audio_mask, audio_feats)        # 音频注入
# 之后再用 emb 去切三段 / 压缩 / 拼接
```
`thinker = base_model`（Qwen3-Omni 的 `ForConditionalGeneration` 即 thinker；确认 `get_base_model(model)` 返回的对象上有 `get_audio_features/get_placeholder_mask` 或经 `.thinker`/`.model`）。

### M2.2 position_ids
压缩改变序列长度 → 不能用 collator 预算的 position_ids。改为压缩后用 `cumsum(new_mask)-1` 广播 4 行（无图无视频，mrope 退化成顺序，已验证）。音频 token 在 `get_rope_index` 里也是顺序 arange，故顺序编码与之一致。

### M2.3 验证
- 单条音频样本前向不报错，audio placeholder 数量与 audio_features 对齐（masked_scatter 不报 size mismatch）。
- 音频+文本混合 batch loss 正常下降。

---

## 主要风险与对策

| 风险 | 说明 | 对策 |
|---|---|---|
| think 段 token 边界定位 | `<think>`/`</think>` 可能是多 token，padding 偏移 | 用 id 子串搜索；collator 里按实际 pad 重算；先用 batch=1 验证 |
| 逐样本变长重建后再 pad | 压缩使每条长度不同 | 在 compute_loss 内重新 pad + 重建 attention_mask（参考 CoLaR 但要右 pad 对齐 Qwen） |
| latent_policy 入 optimizer / dtype | 不在 LoRA target | 挂为 model 子模块 + 显式 requires_grad=True，确认被 optimizer 收；bf16 下数值稳定性看 NLL |
| embeds_std 尺度 | 硬编码常数不适用 Qwen3-Omni | 启动时实测 `get_input_embeddings().weight.std()` |
| device_map/DDP 多卡 | latent head 要和 hidden states 同卡 | 先单卡（里程碑1）跑通，多卡时确保 latent_policy 在最后一层 hidden 所在 device |
| padding_free / packing | 你现有脚本用了 padding_free | 里程碑1先关掉 padding_free（与逐样本压缩重建冲突），稳定后再评估 |
| gradient_checkpointing + 双 forward | 第二次 forward（pred_embed_forward）很贵 | 本次默认 `pred_embed_forward_weight=0`，不做第二次 forward |

---

## 实施顺序（TODO）
1. `prepare_data_colar.py` + 跑一遍生成纯文本数据集，肉眼检查 `<think>` 包裹正确。
2. `latent_policy.py`（移植 + compute_embeds_std）。
3. `colar_template.py`（think 边界，先 batch=1 单测 encode 输出）。
4. `colar_trainer.py`：先 `r==1` 打通 forward+CE+latent loss+save，再加 `r>1` 压缩。
5. `plugin.py` + `train_qwen3omni_colar.sh`，纯文本小数据 smoke test（里程碑1验收）。
6. 里程碑2：音频 merge + position_ids 重建，音频 smoke test。

> 后续（不在本次）：latent 自回归推理 `latent_generate`、压缩率 auto 提示、多卡调优、与基线 CoT SFT 的指标对比。
