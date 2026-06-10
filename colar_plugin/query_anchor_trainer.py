"""Query-anchored CoLaR trainer.

This trainer is a parallel experiment implementation.  It subclasses the
existing CoLaR trainer but replaces the latent policy and custom forward path
with a query-anchored variant.  Existing CoLaR files are not modified.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .colar_trainer import ColarSeq2SeqTrainer, _env_bool, _env_float
from .query_anchor_policy import QueryAnchoredLatentPolicy


class QueryAnchoredColarSeq2SeqTrainer(ColarSeq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.query_anchor_condition_head = _env_bool('QUERY_ANCHOR_CONDITION_HEAD', True)
        self.query_anchor_condition_interaction = _env_bool('QUERY_ANCHOR_CONDITION_INTERACTION', True)
        self.query_anchor_condition_input_norm = _env_bool('QUERY_ANCHOR_CONDITION_INPUT_NORM', False)
        self.query_anchor_residual = _env_bool('QUERY_ANCHOR_RESIDUAL', True)
        self.query_anchor_norm_preserve = _env_bool('QUERY_ANCHOR_NORM_PRESERVE', True)
        self.query_anchor_gate_init = _env_float('QUERY_ANCHOR_GATE_INIT', 0.1)
        self.query_anchor_gate_l2 = _env_float('QUERY_ANCHOR_GATE_L2', 0.0)
        self.query_anchor_noise_std = _env_float('QUERY_ANCHOR_NOISE_STD', 0.0)
        self.query_anchor_proj_init = os.environ.get('QUERY_ANCHOR_PROJ_INIT', 'identity').lower()
        self.query_anchor_residual_source = os.environ.get('QUERY_ANCHOR_RESIDUAL_SOURCE', 'hidden').lower()
        self.query_anchor_hidden_source = os.environ.get('QUERY_ANCHOR_HIDDEN_SOURCE', 'user_last').lower()
        self.query_anchor_im_end_id = getattr(self.tokenizer, 'im_end_token_id', None)
        if self.query_anchor_im_end_id is None:
            self.query_anchor_im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        if self.query_anchor_residual_source not in ('embed', 'hidden'):
            raise ValueError(
                'QUERY_ANCHOR_RESIDUAL_SOURCE must be "embed" or "hidden", '
                f'got {self.query_anchor_residual_source!r}')
        if self.query_anchor_hidden_source not in ('prompt_last', 'user_last'):
            raise ValueError(
                'QUERY_ANCHOR_HIDDEN_SOURCE must be "prompt_last" or "user_last", '
                f'got {self.query_anchor_hidden_source!r}')

        latent_policy = QueryAnchoredLatentPolicy(
            feature_size=self.colar_hidden_size,
            intermediate_size=self.colar_lp_intermediate,
            deterministic=self.colar_deterministic,
            condition_head=self.query_anchor_condition_head,
            condition_interaction=self.query_anchor_condition_interaction,
            condition_input_norm=self.query_anchor_condition_input_norm,
            residual_anchor=self.query_anchor_residual,
            norm_preserve=self.query_anchor_norm_preserve,
            gate_init=self.query_anchor_gate_init,
            proj_init=self.query_anchor_proj_init,
        )
        dev = self._embed_tokens.weight.device
        latent_policy = latent_policy.to(device=dev, dtype=torch.float32)
        for p in latent_policy.parameters():
            p.requires_grad_(True)
        self.model.latent_policy = latent_policy
        self.latent_policy = latent_policy

        self._log_query_anchor_config()

    def _log_query_anchor_config(self):
        from swift.utils import get_logger

        logger = get_logger()
        logger.info(
            '[query-anchor-colar] condition_head=%s condition_interaction=%s condition_input_norm=%s '
            'residual=%s norm_preserve=%s '
            'gate_init=%.6f gate_l2=%.6g noise_std=%.6g proj_init=%s hidden_source=%s',
            self.query_anchor_condition_head,
            self.query_anchor_condition_interaction,
            self.query_anchor_condition_input_norm,
            self.query_anchor_residual,
            self.query_anchor_norm_preserve,
            self.query_anchor_gate_init,
            self.query_anchor_gate_l2,
            self.query_anchor_noise_std,
            self.query_anchor_proj_init,
            self.query_anchor_hidden_source,
        )
        if self.query_anchor_residual_source == 'hidden':
            logger.info(
                '[query-anchor-colar] residual_source=hidden uses an extra prefix forward; '
                'disable gradient_checkpointing if PyTorch checkpoint metadata errors occur.')
        else:
            logger.info('[query-anchor-colar] residual_source=embed')

    def _save_model(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save_model(output_dir=output_dir, state_dict=state_dict)
        if output_dir is None or not self.is_world_process_zero():
            return
        meta = {
            'latent_policy_type': 'query_anchor',
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
            'query_anchor_condition_head': self.query_anchor_condition_head,
            'query_anchor_condition_interaction': self.query_anchor_condition_interaction,
            'query_anchor_condition_input_norm': self.query_anchor_condition_input_norm,
            'query_anchor_residual': self.query_anchor_residual,
            'query_anchor_norm_preserve': self.query_anchor_norm_preserve,
            'query_anchor_gate_l2': self.query_anchor_gate_l2,
            'query_anchor_noise_std': self.query_anchor_noise_std,
            'query_anchor_proj_init': self.query_anchor_proj_init,
            'query_anchor_residual_source': self.query_anchor_residual_source,
            'query_anchor_hidden_source': self.query_anchor_hidden_source,
            'query_anchor_im_end_id': self.query_anchor_im_end_id,
            'query_anchor_alpha': float(self.latent_policy.alpha().detach().cpu()),
        }
        with open(os.path.join(output_dir, 'colar_config.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _colar_forward_impl(self, inputs):
        inputs.pop('compute_loss_func', None)
        inputs.pop('loss_scale', None)
        inputs.pop('text_position_ids', None)
        inputs.pop('channel', None)
        inputs.pop('position_ids', None)
        output_router_logits = inputs.pop('output_router_logits', None)

        input_ids = inputs['input_ids']
        labels = inputs.pop('labels')
        attention_mask = inputs.get('attention_mask')
        input_features = inputs.pop('input_features', None)
        feature_attention_mask = inputs.pop('feature_attention_mask', None)
        for mm_key in (
                'pixel_values',
                'pixel_values_videos',
                'image_grid_thw',
                'video_grid_thw',
                'video_second_per_grid',
                'audio_feature_lengths',
                'rope_deltas',
                'use_audio_in_video',
                'cache_position',
        ):
            inputs.pop(mm_key, None)
        if attention_mask is None:
            attention_mask = (input_ids != self.colar_pad_id).long()

        device = input_ids.device
        base_embeds = self._build_base_embeds(input_ids, input_features, feature_attention_mask)

        if self.colar_fixed_r > 0:
            r = self.colar_fixed_r
        else:
            import random
            r = random.randint(1, max(1, self.colar_max_r))

        prefix_lengths = self._query_prefix_lengths(input_ids, attention_mask)
        query_anchor_positions = self._query_anchor_positions(input_ids, attention_mask, prefix_lengths)
        residual_query = None
        if self.query_anchor_residual:
            residual_query = self._build_residual_query(
                input_ids, base_embeds, attention_mask, prefix_lengths, query_anchor_positions)

        if r == 1:
            embeds = base_embeds
            attn = attention_mask
            new_labels = labels
            think_mask = self._build_think_mask(input_ids, attention_mask)
            gold_source = base_embeds
            latent_input_mask = torch.zeros_like(think_mask)
        else:
            embeds, attn, new_labels, think_mask, gold_source = self._compress_batch(
                input_ids, base_embeds, labels, attention_mask, r)
            latent_input_mask = self._latent_input_mask_from_think_mask(think_mask)
            embeds = self._apply_latent_noise(embeds, latent_input_mask)
            embeds = self.latent_policy.apply_anchor_to_embeddings(
                embeds,
                query_hidden=residual_query,
                latent_input_mask=latent_input_mask,
                embeds_std=self.embeds_std,
            )

        position_ids = torch.clamp_min(attn.long().cumsum(dim=1) - 1, 0)

        text_kwargs = {}
        if output_router_logits is not None:
            text_kwargs['output_router_logits'] = output_router_logits
        text_out = self._text_model(
            inputs_embeds=embeds,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
            **text_kwargs,
        )
        hidden = text_out.last_hidden_state
        logits = self._lm_head(hidden)

        ce_loss = self._causal_ce(logits, new_labels)
        condition_query = self._gather_query_hidden(hidden, query_anchor_positions)
        embed_loss, entropy = self._latent_loss(hidden, gold_source, think_mask, condition_query)

        loss_dev = ce_loss.device
        total = self.colar_ce_weight * ce_loss
        if self.colar_embed_weight != 0:
            total = total + self.colar_embed_weight * embed_loss.to(loss_dev)
        if self.colar_entropy_weight != 0:
            total = total + self.colar_entropy_weight * entropy.to(loss_dev)
        if self.query_anchor_gate_l2 != 0:
            alpha = self.latent_policy.alpha().to(loss_dev)
            total = total + self.query_anchor_gate_l2 * alpha.square()

        return {
            'loss': total.to(device),
            'logits': logits,
            'ce_loss': ce_loss,
            'embed_loss': embed_loss,
            'entropy': entropy,
            'r': torch.tensor(float(r), device=loss_dev),
        }

    def _apply_latent_noise(self, embeds: torch.Tensor, latent_input_mask: torch.Tensor) -> torch.Tensor:
        if (not self.model.training) or self.query_anchor_noise_std <= 0:
            return embeds
        if latent_input_mask.sum().item() <= 0:
            return embeds
        noise = torch.randn_like(embeds.float()) * (self.query_anchor_noise_std * float(self.embeds_std))
        mask = latent_input_mask.to(device=embeds.device, dtype=embeds.dtype).unsqueeze(-1)
        return embeds + noise.to(embeds.dtype) * mask

    def _latent_input_mask_from_think_mask(self, think_mask: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(think_mask)
        for b in range(think_mask.shape[0]):
            idx = torch.nonzero(think_mask[b] > 0, as_tuple=False).flatten()
            if idx.numel() > 1:
                mask[b, idx[1:]] = 1.0
        return mask

    def _build_residual_query(
        self,
        input_ids: torch.Tensor,
        base_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_lengths,
        query_anchor_positions,
    ) -> torch.Tensor:
        if self.query_anchor_residual_source == 'embed':
            gather = torch.tensor(query_anchor_positions, device=base_embeds.device, dtype=torch.long)
            q = base_embeds[torch.arange(base_embeds.shape[0], device=base_embeds.device), gather, :].unsqueeze(1)
            return q.detach()
        return self._build_query_hidden(input_ids, base_embeds, attention_mask, prefix_lengths, query_anchor_positions)

    def _build_query_hidden(
        self,
        input_ids: torch.Tensor,
        base_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_lengths=None,
        query_anchor_positions=None,
    ) -> torch.Tensor:
        if prefix_lengths is None:
            prefix_lengths = self._query_prefix_lengths(input_ids, attention_mask)
        if query_anchor_positions is None:
            query_anchor_positions = self._query_anchor_positions(input_ids, attention_mask, prefix_lengths)
        max_len = max(prefix_lengths)
        B, _, H = base_embeds.shape
        prefix_embeds = base_embeds.new_zeros(B, max_len, H)
        prefix_attn = torch.zeros(B, max_len, dtype=torch.long, device=base_embeds.device)
        for b, ln in enumerate(prefix_lengths):
            prefix_embeds[b, :ln] = base_embeds[b, :ln]
            prefix_attn[b, :ln] = 1
        prefix_pos = torch.clamp_min(prefix_attn.long().cumsum(dim=1) - 1, 0)
        # The text backbone is used twice in this custom forward.  Disable all
        # layer-level checkpoint flags for the detached prefix pass so PyTorch
        # does not register checkpoint frames with a different sequence length.
        restore_gradient_checkpointing = self._temporarily_disable_gradient_checkpointing(self._text_model)
        try:
            with torch.no_grad():
                out = self._text_model(
                    inputs_embeds=prefix_embeds,
                    attention_mask=prefix_attn,
                    position_ids=prefix_pos,
                    use_cache=False,
                )
        finally:
            restore_gradient_checkpointing()
        hidden = out.last_hidden_state
        gather = torch.tensor(query_anchor_positions, device=hidden.device, dtype=torch.long)
        q = hidden[torch.arange(B, device=hidden.device), gather, :].unsqueeze(1)
        return q.detach()

    def _gather_query_hidden(self, hidden: torch.Tensor, query_anchor_positions) -> torch.Tensor:
        gather = torch.tensor(query_anchor_positions, device=hidden.device, dtype=torch.long)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), gather, :].unsqueeze(1).detach()

    def _temporarily_disable_gradient_checkpointing(self, module):
        saved_flags: Dict[torch.nn.Module, bool] = {}
        for m in module.modules():
            if hasattr(m, 'gradient_checkpointing'):
                saved_flags[m] = bool(getattr(m, 'gradient_checkpointing'))
                setattr(m, 'gradient_checkpointing', False)

        def restore():
            for m, value in saved_flags.items():
                setattr(m, 'gradient_checkpointing', value)

        return restore

    def _query_prefix_lengths(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        rows = input_ids.tolist()
        out = []
        for b, row in enumerate(rows):
            valid_len = int(attention_mask[b].sum().item())
            try:
                o = row.index(self.think_open_id)
                c = row.index(self.think_close_id)
            except ValueError:
                out.append(max(1, valid_len))
                continue
            if c <= o + 1 or c >= valid_len:
                out.append(max(1, valid_len))
                continue
            reason_start = o + 1
            if self.colar_newline_id is not None and reason_start < c and row[reason_start] == self.colar_newline_id:
                reason_start += 1
            out.append(max(1, reason_start))
        return out

    def _query_anchor_positions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, prefix_lengths):
        if self.query_anchor_hidden_source == 'prompt_last':
            return [max(0, ln - 1) for ln in prefix_lengths]

        rows = input_ids.tolist()
        out = []
        for b, row in enumerate(rows):
            valid_len = int(attention_mask[b].sum().item())
            try:
                o = row.index(self.think_open_id)
                c = row.index(self.think_close_id)
                im_end_positions = [i for i, tok in enumerate(row[:o]) if tok == self.query_anchor_im_end_id]
                user_end = im_end_positions[-1]
            except ValueError:
                out.append(max(0, min(prefix_lengths[b], valid_len) - 1))
                continue
            except IndexError:
                out.append(max(0, min(prefix_lengths[b], valid_len) - 1))
                continue
            # The user query ends at the final <|im_end|> before assistant/<think>.
            if user_end <= 0 or user_end >= c:
                out.append(max(0, min(prefix_lengths[b], valid_len) - 1))
            else:
                out.append(user_end - 1)
        return out

    def _latent_loss(
        self,
        hidden: torch.Tensor,
        gold_source: torch.Tensor,
        think_mask: torch.Tensor,
        query_hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lp_dev = next(self.latent_policy.parameters()).device
        h = hidden.to(lp_dev).float()
        q = query_hidden.to(lp_dev).float() if query_hidden is not None else None
        dist = self.latent_policy(h, query_hidden=q)
        gold = torch.roll(gold_source.to(lp_dev), shifts=-1, dims=1).float() / self.embeds_std
        think_mask = think_mask.to(lp_dev)

        if self.colar_embed_loss == 'mse':
            pred = dist.rsample()
            per = F.mse_loss(pred, gold.detach(), reduction='none').mean(dim=-1)
        else:
            per = -dist.log_prob(gold.detach()).mean(dim=-1)

        denom = think_mask.sum().clamp_min(1.0)
        embed_loss = (per * think_mask).sum() / denom
        ent = dist.entropy().mean(dim=-1)
        entropy = (ent * think_mask).sum() / denom
        return embed_loss, entropy
