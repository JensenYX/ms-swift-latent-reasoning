"""
自定义 Trainer：在 ms-swift 的 Seq2SeqTrainer 上实现 CoLaR 的 latent-reasoning SFT。

为什么必须自定义 Trainer（而不是 --loss_type 插件）：
  CoLaR 需要在“喂进 LLM 之前”就改 inputs_embeds（压缩思考段），并在 forward 后用
  最后一层 hidden state 训练一个额外的 latent head。--loss_type 插件只在 model(**inputs)
  之后拿到 (outputs, labels)，无法控制 forward，因此不够用。

核心流程（compute_loss）：
  1. 从 input_ids 定位每条样本的思考段 [<think> ... </think>]。
  2. embeds = thinker.embed_tokens(input_ids)   （里程碑1纯文本：查表即可；里程碑2再加音频 merge）
  3. 采样压缩率 r：
       r==1  -> 不压缩，整段直接前向
       r>1   -> 仅把“思考段”按 r 做 mean-pool 压缩，前缀/答案保持不变，逐样本重建后再 pad
  4. 直接调 thinker.model（文本主干）拿 last_hidden_state，再过 lm_head 得 logits。
     position_ids 用 cumsum(attention_mask)-1（无图无视频时 mrope 退化为顺序编码）。
  5. 两个 loss：
       - ce_loss：标准 causal LM shift CE（思考段+答案段监督，前缀 -100）
       - embed_modeling_loss：latent head 预测“下一个（压缩）embedding”的 NLL/MSE
  6. total = ce_weight*ce + embed_modeling_weight*nll (+ entropy_weight*entropy)

latent head（LatentPolicy）作为 model 的子模块挂载，显式 requires_grad=True，
并在 create_optimizer 里确保进入优化器；_save_model 里额外 torch.save 到 latent_policy.pt。

CoLaR 专属超参通过环境变量传入（ms-swift 的 argument dataclass 不认识它们）：
  COLAR_MAX_R            最大压缩率（int），默认 1（=不压缩，先打通管线）
  COLAR_FIXED_R         >0 则把每步 r 钉死为该值（调试/自检用，绕过随机采样）；默认 0=随机
  COLAR_CE_WEIGHT        CE loss 权重，默认 1.0
  COLAR_EMBED_WEIGHT     latent NLL/MSE 权重，默认 1.0
  COLAR_ENTROPY_WEIGHT   entropy 正则权重，默认 0.0
  COLAR_EMBED_LOSS       'nll' 或 'mse'，默认 'nll'
  COLAR_LP_INTERMEDIATE  latent head 中间维度，默认 = hidden_size
  COLAR_DETERMINISTIC    '1' 则 latent head 退化为确定性（std≈0），默认 '0'
  COLAR_SQRT_MEAN        '1' 则压缩归一化用 sqrt(count)，默认 '0'
"""
import json
import os
import random
from functools import wraps
from types import MethodType
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
from swift.utils import get_logger

from .latent_policy import LatentPolicy, compute_embeds_std
from .colar_template import THINK_OPEN_ID, THINK_CLOSE_ID

logger = get_logger()


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, '') else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, '') else default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v in (None, ''):
        return default
    return str(v).lower() in ('1', 'true', 'yes', 'y')


class ColarSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ---- 读取 CoLaR 超参 ----
        self.colar_max_r = _env_int('COLAR_MAX_R', 1)
        self.colar_fixed_r = _env_int('COLAR_FIXED_R', 0)  # >0 则把 r 钉死，调试/自检用
        self.colar_ce_weight = _env_float('COLAR_CE_WEIGHT', 1.0)
        self.colar_embed_weight = _env_float('COLAR_EMBED_WEIGHT', 1.0)
        self.colar_entropy_weight = _env_float('COLAR_ENTROPY_WEIGHT', 0.0)
        self.colar_embed_loss = os.environ.get('COLAR_EMBED_LOSS', 'nll').lower()
        self.colar_sqrt_mean = _env_bool('COLAR_SQRT_MEAN', False)
        self.colar_deterministic = _env_bool('COLAR_DETERMINISTIC', False)

        self.think_open_id = THINK_OPEN_ID
        self.think_close_id = THINK_CLOSE_ID
        self.colar_pad_id = self.tokenizer.pad_token_id
        newline_ids = self.tokenizer.encode('\n', add_special_tokens=False)
        self.colar_newline_id = newline_ids[0] if len(newline_ids) == 1 else None

        # ---- 定位 thinker / 文本主干 / lm_head / embedding ----
        base_model = self.template.get_base_model(self.model)  # 解 PEFT 包装
        thinker = getattr(base_model, 'thinker', base_model)
        self._thinker = thinker
        self._text_model = thinker.model          # Qwen3OmniMoeThinkerTextModel（文本主干）
        self._lm_head = thinker.lm_head
        self._embed_tokens = thinker.get_input_embeddings()

        hidden_size = self._embed_tokens.weight.shape[1]
        self.colar_hidden_size = hidden_size
        self.colar_lp_intermediate = _env_int('COLAR_LP_INTERMEDIATE', hidden_size)

        # ---- 实测 embeds_std（替代硬编码 MODEL_EMB_STD）----
        self.embeds_std = compute_embeds_std(thinker)

        # ---- 构建 latent head 并挂为 model 子模块（fp32，数值稳定）----
        latent_policy = LatentPolicy(
            feature_size=hidden_size,
            intermediate_size=self.colar_lp_intermediate,
            deterministic=self.colar_deterministic)
        dev = self._embed_tokens.weight.device
        latent_policy = latent_policy.to(device=dev, dtype=torch.float32)
        for p in latent_policy.parameters():
            p.requires_grad_(True)
        # 挂在 PEFT 外层 model 上，确保 state_dict / optimizer 能看到
        self.model.latent_policy = latent_policy
        self.latent_policy = latent_policy
        self._install_colar_forward()

        logger.info(f'[colar] max_r={self.colar_max_r} fixed_r={self.colar_fixed_r} '
                    f'ce_w={self.colar_ce_weight} '
                    f'embed_w={self.colar_embed_weight} embed_loss={self.colar_embed_loss} '
                    f'deterministic={self.colar_deterministic} embeds_std={self.embeds_std:.5f} '
                    f'hidden={hidden_size} lp_inter={self.colar_lp_intermediate}')

    # ---- 确保 latent head 进入优化器 ----
    def create_optimizer(self, *args, **kwargs):
        optimizer = super().create_optimizer(*args, **kwargs)
        if optimizer is None:
            return optimizer
        existing = {id(p) for g in optimizer.param_groups for p in g['params']}
        extra = [p for p in self.latent_policy.parameters() if p.requires_grad and id(p) not in existing]
        if extra:
            optimizer.add_param_group({'params': extra})
            logger.info(f'[colar] added {len(extra)} latent_policy params to optimizer')
        return optimizer

    # ---- 额外保存 latent head ----
    def _save_model(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save_model(output_dir=output_dir, state_dict=state_dict)
        try:
            if output_dir is not None and self.is_world_process_zero():
                torch.save(self.latent_policy.state_dict(), os.path.join(output_dir, 'latent_policy.pt'))
                meta = {
                    'embeds_std': self.embeds_std,
                    'hidden_size': self.colar_hidden_size,
                    'lp_intermediate': self.colar_lp_intermediate,
                    'deterministic': self.colar_deterministic,
                    'max_r': self.colar_max_r,
                    'fixed_r': self.colar_fixed_r,
                    'ce_weight': self.colar_ce_weight,
                    'embed_weight': self.colar_embed_weight,
                    'entropy_weight': self.colar_entropy_weight,
                    'embed_loss': self.colar_embed_loss,
                    'sqrt_mean': self.colar_sqrt_mean,
                    'think_open_id': self.think_open_id,
                    'think_close_id': self.think_close_id,
                }
                with open(os.path.join(output_dir, 'colar_config.json'), 'w') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                logger.info(f'[colar] saved latent_policy.pt -> {output_dir}')
        except Exception as e:  # noqa
            logger.warning(f'[colar] save latent_policy failed: {e}')

    # ===================== 核心 =====================
    def _install_colar_forward(self):
        """
        Route CoLaR's custom forward through the top-level model object.

        In DDP, gradients are only reduced reliably when the wrapped module's
        forward is entered. Calling cached submodules directly from
        compute_loss bypasses that wrapper, which is fine for single-process
        device_map but fragile for MP+DDP.
        """
        if getattr(self.model, '_colar_forward_installed', False):
            return

        original_forward = self.model.forward
        trainer = self

        @wraps(original_forward)
        def colar_forward(model_self, *args, **kwargs):
            if args or 'labels' not in kwargs:
                return original_forward(*args, **kwargs)
            return trainer._colar_forward_impl(kwargs)

        self.model._colar_original_forward = original_forward
        self.model.forward = MethodType(colar_forward, self.model)
        self.model._colar_forward_installed = True
        logger.info('[colar] installed top-level forward hook for DDP-visible custom loss')

    def _colar_forward_impl(self, inputs):
        # ms-swift 会注入这些，CoLaR 自管 loss，全部弹出，避免传进模型
        inputs.pop('compute_loss_func', None)
        inputs.pop('loss_scale', None)
        inputs.pop('text_position_ids', None)
        inputs.pop('channel', None)
        inputs.pop('position_ids', None)            # 压缩会改长度，position_ids 需重算
        output_router_logits = inputs.pop('output_router_logits', None)

        input_ids = inputs['input_ids']             # [B, L]
        labels = inputs.pop('labels')               # [B, L]，前缀已 -100
        attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = (input_ids != self.colar_pad_id).long()

        device = input_ids.device

        # 1) 基础 embedding（里程碑1纯文本：直接查表）
        #    里程碑2：在此后插入 audio_tower -> masked_scatter（见 plan M2.1）
        base_embeds = self._embed_tokens(input_ids)  # [B, L, H]

        # 2) 采样压缩率
        #    COLAR_FIXED_R>0 时把 r 钉死（overfit 自检/调试用，曲线不抖、可复现）；
        #    默认 0 -> 维持原行为：在 [1, max_r] 上每步随机采样（正式 SFT 形态）。
        if self.colar_fixed_r > 0:
            r = self.colar_fixed_r
        else:
            r = random.randint(1, max(1, self.colar_max_r))

        # 3) 构造（可能压缩后的）embeds / attention_mask / labels / think_mask
        if r == 1:
            embeds = base_embeds
            attn = attention_mask
            new_labels = labels
            think_mask = self._build_think_mask(input_ids, attention_mask)  # [B, L]
            gold_source = base_embeds
        else:
            embeds, attn, new_labels, think_mask, gold_source = self._compress_batch(
                input_ids, base_embeds, labels, attention_mask, r)

        # 4) position_ids：cumsum(mask)-1（mrope 退化，传 2D，文本主干自动扩成 4 行）
        position_ids = torch.clamp_min(attn.long().cumsum(dim=1) - 1, 0)

        # 5) 文本主干前向，取 last_hidden_state
        text_kwargs = {}
        if output_router_logits is not None:
            text_kwargs['output_router_logits'] = output_router_logits
        text_out = self._text_model(
            inputs_embeds=embeds,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,            # 与 gradient_checkpointing 兼容
            **text_kwargs,
        )
        hidden = text_out.last_hidden_state          # [B, L', H]
        logits = self._lm_head(hidden)               # [B, L', V]

        # 6) CE loss（标准 causal LM shift）
        ce_loss = self._causal_ce(logits, new_labels)

        # 7) latent head loss（在思考段上预测“下一个 embedding”）
        embed_loss, entropy = self._latent_loss(hidden, gold_source, think_mask)

        # device_map 下 ce_loss / embed_loss 可能在不同卡，统一搬到 ce_loss 设备再相加
        loss_dev = ce_loss.device
        total = self.colar_ce_weight * ce_loss
        if self.colar_embed_weight != 0:
            total = total + self.colar_embed_weight * embed_loss.to(loss_dev)
        if self.colar_entropy_weight != 0:
            total = total + self.colar_entropy_weight * entropy.to(loss_dev)

        return {
            'loss': total.to(device),
            'logits': logits,
            'ce_loss': ce_loss,
            'embed_loss': embed_loss,
            'entropy': entropy,
            'r': torch.tensor(float(r), device=loss_dev),
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)

        # 记录指标
        mode = 'train' if self.model.training else 'eval'
        self.custom_metrics[mode]['ce_loss'].update(outputs['ce_loss'].detach())
        self.custom_metrics[mode]['embed_loss'].update(outputs['embed_loss'].detach())
        if self.colar_entropy_weight != 0:
            self.custom_metrics[mode]['entropy'].update(outputs['entropy'].detach())
        self.custom_metrics[mode]['r'].update(outputs['r'].detach())

        if return_outputs:
            return outputs['loss'], outputs
        return outputs['loss']   # HF Trainer 要求 loss 回到原始输入设备（cuda:0）

    # ---------- 工具 ----------
    def _build_think_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """think_mask[b,t]=1 当 t 落在 <think> 与 </think> 之间（不含两端标记本身）。"""
        B, L = input_ids.shape
        mask = torch.zeros((B, L), dtype=torch.float32, device=input_ids.device)
        ids = input_ids.tolist()
        for b in range(B):
            row = ids[b]
            try:
                o = row.index(self.think_open_id)
                c = row.index(self.think_close_id)
            except ValueError:
                continue
            if c > o + 1:
                mask[b, o + 1:c] = 1.0
        return mask

    def _causal_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous().to(shift_logits.device)  # device_map: 对齐到 logits 卡
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    def _latent_loss(self, hidden: torch.Tensor, gold_source: torch.Tensor,
                     think_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在思考段位置 t：hidden[t] 预测 embed[t+1]。
        dist = latent_policy(hidden[t])，gold = gold_source[t+1] / embeds_std。
        用 think_mask（已对齐 hidden 长度）做掩码平均。
        """
        # 设备对齐：device_map 下 hidden（最后一层）与 latent_policy（固定在初始化设备）
        # 可能不在同一张卡。注意：绝不能移动 latent_policy（会与 optimizer 状态错位），
        # 而是把 hidden/gold/mask 搬到 latent_policy 所在设备来算 loss。
        lp_dev = next(self.latent_policy.parameters()).device
        h = hidden.to(lp_dev).float()
        dist = self.latent_policy(h)                                  # Normal, [B, L', H]
        gold = torch.roll(gold_source.to(lp_dev), shifts=-1, dims=1).float() / self.embeds_std  # [B, L', H]
        think_mask = think_mask.to(lp_dev)

        if self.colar_embed_loss == 'mse':
            pred = dist.rsample()
            per = F.mse_loss(pred, gold.detach(), reduction='none').mean(dim=-1)   # [B, L']
        else:  # nll
            per = -dist.log_prob(gold.detach()).mean(dim=-1)                       # [B, L']

        denom = think_mask.sum().clamp_min(1.0)
        embed_loss = (per * think_mask).sum() / denom
        ent = dist.entropy().mean(dim=-1)
        entropy = (ent * think_mask).sum() / denom
        return embed_loss, entropy

    def _compress_batch(self, input_ids, base_embeds, labels, attention_mask, r):
        """
        仅压缩“思考段”，前缀（含 <think>）与答案段（含 </think>）保持不变。
        逐样本重建后右 pad 成 batch。embedding 取片/池化全程带梯度。
        返回: embeds[B,L',H], attn[B,L'], new_labels[B,L'], think_mask[B,L'], gold_source[B,L',H]
        """
        B, L, H = base_embeds.shape
        device = base_embeds.device
        pad_id = self.colar_pad_id

        ids = input_ids.tolist()
        seqs_embed: List[torch.Tensor] = []
        seqs_label: List[torch.Tensor] = []
        seqs_think: List[torch.Tensor] = []
        seqs_goldsrc: List[torch.Tensor] = []

        for b in range(B):
            row = ids[b]
            try:
                o = row.index(self.think_open_id)
                c = row.index(self.think_close_id)
            except ValueError:
                o, c = -1, -1

            emb_b = base_embeds[b]               # [L, H]
            lab_b = labels[b]                    # [L]
            valid_len = int(attention_mask[b].sum().item())

            if o < 0 or c <= o + 1 or c >= valid_len:
                # 无有效 think 段：整条不压缩（去掉右 pad）
                seqs_embed.append(emb_b[:valid_len])
                seqs_label.append(lab_b[:valid_len])
                seqs_think.append(torch.zeros(valid_len, device=device))
                seqs_goldsrc.append(emb_b[:valid_len])
                continue

            # 推理 prompt 由 template 生成到 `<think>\n` 为止；latent_policy 的第一步
            # 输入是这个换行 token 的 hidden state。因此 r>1 压缩时必须把换行留在
            # uncompressed prefix 中，从换行之后的真实思考内容开始压缩。否则训练的
            # 第一个 compressed embedding 会是 mean('\n', first_reason_token)，而推理
            # 已经显式喂入了 '\n'，第一步 rollout 从一开始就错位。
            reason_start = o + 1
            if self.colar_newline_id is not None and reason_start < c and row[reason_start] == self.colar_newline_id:
                reason_start += 1

            prefix_embed = emb_b[:reason_start]          # 含 <think> 以及模板注入的换行
            prefix_label = lab_b[:reason_start]
            reason_embed = emb_b[reason_start:c]         # 不含 prompt 里的换行
            reason_ids = input_ids[b, reason_start:c]
            answer_embed = emb_b[c:valid_len]            # 含 </think> + 答案
            answer_label = lab_b[c:valid_len]

            n = reason_embed.shape[0]
            if n == 0:
                seqs_embed.append(torch.cat([prefix_embed, answer_embed], dim=0))
                seqs_label.append(torch.cat([prefix_label, answer_label], dim=0))
                seqs_think.append(torch.zeros(prefix_embed.shape[0] + answer_embed.shape[0], device=device))
                seqs_goldsrc.append(torch.cat([prefix_embed, answer_embed], dim=0))
                continue
            rem = (-n) % r                               # 右 pad 到 r 的倍数
            if rem > 0:
                reason_embed_p = torch.cat([reason_embed, reason_embed.new_zeros(rem, H)], dim=0)
                pad_valid = torch.cat([torch.ones(n, device=device), torch.zeros(rem, device=device)])
                reason_ids_p = torch.cat([reason_ids, reason_ids.new_full((rem,), pad_id)])
            else:
                reason_embed_p = reason_embed
                pad_valid = torch.ones(n, device=device)
                reason_ids_p = reason_ids
            k = reason_embed_p.shape[0] // r

            grp_embed = reason_embed_p.reshape(k, r, H)
            grp_valid = pad_valid.reshape(k, r)
            cnt = grp_valid.sum(dim=1)
            cnt_norm = cnt.sqrt() if self.colar_sqrt_mean else cnt
            comp_embed = (grp_embed * grp_valid.unsqueeze(-1)).sum(dim=1) / (cnt_norm.unsqueeze(-1) + 1e-5)  # [k,H]

            grp_ids = reason_ids_p.reshape(k, r)
            probs = grp_valid + 1e-5
            probs = probs / probs.sum(dim=1, keepdim=True)
            sel = torch.multinomial(probs, num_samples=1).squeeze(1)
            comp_label = grp_ids.gather(1, sel.unsqueeze(1)).squeeze(1)  # [k]

            emb_seq = torch.cat([prefix_embed, comp_embed, answer_embed], dim=0)
            lab_seq = torch.cat([prefix_label, comp_label, answer_label], dim=0)
            think_seq = torch.zeros(emb_seq.shape[0], device=device)
            # 训练 latent_policy(hidden[t]) -> embedding[t+1]。推理第一步使用
            # prompt 最后一个 token（通常是 '\n'）的 hidden state 预测第一个
            # compressed embedding，因此这里要把 prefix 的最后一个 token 也纳入
            # latent loss mask；之后每个 compressed token 预测下一个 compressed token，
            # 最后一个 compressed token 预测 </think> embedding。
            anchor = prefix_embed.shape[0] - 1
            think_seq[anchor: prefix_embed.shape[0] + k] = 1.0

            seqs_embed.append(emb_seq)
            seqs_label.append(lab_seq)
            seqs_think.append(think_seq)
            seqs_goldsrc.append(emb_seq)         # gold 用压缩后序列自身 shift

        Lmax = max(s.shape[0] for s in seqs_embed)
        embeds = base_embeds.new_zeros(B, Lmax, H)
        attn = torch.zeros(B, Lmax, dtype=torch.long, device=device)
        new_labels = torch.full((B, Lmax), -100, dtype=labels.dtype, device=device)
        think_mask = torch.zeros(B, Lmax, dtype=torch.float32, device=device)
        gold_source = base_embeds.new_zeros(B, Lmax, H)
        for b in range(B):
            ln = seqs_embed[b].shape[0]
            embeds[b, :ln] = seqs_embed[b]
            attn[b, :ln] = 1
            new_labels[b, :ln] = seqs_label[b]
            think_mask[b, :ln] = seqs_think[b]
            gold_source[b, :ln] = seqs_goldsrc[b]
        return embeds, attn, new_labels, think_mask, gold_source
