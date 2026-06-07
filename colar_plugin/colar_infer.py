"""
CoLaR latent inference for Qwen3-Omni on ms-swift.

Mirrors the original COLAR `latent_generate` loop (colar/src/models/model_base.py):
  1) encode user prompt up to `<think>\\n`
  2) forward the question prefix
  3) autoregressively sample compressed embeddings with `latent_policy`
  4) stop when lm_head predicts `</think>` (or max latent steps)
  5) append the close-tag embedding and generate the answer tokens
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from swift.infer_engine import InferRequest
from swift.tuners import Swift

from .colar_template import THINK_CLOSE_ID, THINK_OPEN_ID
from .latent_policy import LatentPolicy


@dataclass
class ColarInferConfig:
    embeds_std: float
    hidden_size: int
    lp_intermediate: int
    deterministic: bool
    think_open_id: int = THINK_OPEN_ID
    think_close_id: int = THINK_CLOSE_ID
    max_n_latent_forward: int = 2048
    latent_temperature: float = 0.0
    eol_temperature: float = 0.0
    max_new_tokens: int = 8192
    answer_temperature: float = 0.0
    progress_every: int = 0


def load_colar_infer_config(checkpoint: Path, overrides: Optional[Dict[str, Any]] = None) -> ColarInferConfig:
    overrides = overrides or {}
    meta_path = checkpoint / 'colar_config.json'
    meta = {}
    if meta_path.exists():
        with meta_path.open('r', encoding='utf-8') as f:
            meta = json.load(f)
    return ColarInferConfig(
        embeds_std=float(overrides.get('embeds_std', meta.get('embeds_std', 1.0))),
        hidden_size=int(overrides.get('hidden_size', meta.get('hidden_size', 2048))),
        lp_intermediate=int(overrides.get('lp_intermediate', meta.get('lp_intermediate', 2048))),
        deterministic=bool(overrides.get('deterministic', meta.get('deterministic', False))),
        think_open_id=int(overrides.get('think_open_id', meta.get('think_open_id', THINK_OPEN_ID))),
        think_close_id=int(overrides.get('think_close_id', meta.get('think_close_id', THINK_CLOSE_ID))),
        max_n_latent_forward=int(overrides.get('max_n_latent_forward', 2048)),
        latent_temperature=float(overrides.get('latent_temperature', 0.0)),
        eol_temperature=float(overrides.get('eol_temperature', 0.0)),
        max_new_tokens=int(overrides.get('max_new_tokens', 8192)),
        answer_temperature=float(overrides.get('answer_temperature', 0.0)),
        progress_every=int(overrides.get('progress_every', 0)),
    )


def load_latent_policy(checkpoint: Path, config: ColarInferConfig) -> LatentPolicy:
    policy_path = checkpoint / 'latent_policy.pt'
    if not policy_path.exists():
        raise FileNotFoundError(f'latent_policy.pt not found in checkpoint: {policy_path}')
    latent_policy = LatentPolicy(
        feature_size=config.hidden_size,
        intermediate_size=config.lp_intermediate,
        deterministic=config.deterministic,
    )
    try:
        state_dict = torch.load(policy_path, map_location='cpu', weights_only=True)
    except TypeError:
        state_dict = torch.load(policy_path, map_location='cpu')
    latent_policy.load_state_dict(state_dict)
    latent_policy.eval()
    for p in latent_policy.parameters():
        p.requires_grad_(False)
    return latent_policy


def _resolve_thinker_modules(model, template):
    base_model = template.get_base_model(model)
    thinker = getattr(base_model, 'thinker', base_model)
    text_model = thinker.model
    lm_head = thinker.lm_head
    embed_tokens = thinker.get_input_embeddings()
    return thinker, text_model, lm_head, embed_tokens


def _module_device(module) -> torch.device:
    for p in module.parameters(recurse=True):
        return p.device
    for b in module.buffers(recurse=True):
        return b.device
    return torch.device('cpu')


def _text_input_device(text_model) -> torch.device:
    layers = getattr(text_model, 'layers', None)
    if layers is not None and len(layers) > 0:
        return _module_device(layers[0])
    return _module_device(text_model)


def _position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(attention_mask.long().cumsum(dim=1) - 1, 0)


class ColarLatentGenerator:

    def __init__(
        self,
        model,
        template,
        latent_policy: LatentPolicy,
        config: ColarInferConfig,
    ):
        self.model = model
        self.template = template
        self.latent_policy = latent_policy
        self.config = config
        self.tokenizer = template.tokenizer
        self.thinker, self.text_model, self.lm_head, self.embed_tokens = _resolve_thinker_modules(model, template)
        self.embed_device = self.embed_tokens.weight.device
        self.text_device = _text_input_device(self.text_model)
        self.lm_head_device = self.lm_head.weight.device
        self.lp_device = next(latent_policy.parameters()).device

    def _encode_prompt(self, messages: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.template.set_mode('transformers')
        encoded = self.template.encode(
            InferRequest(
                messages=messages,
                chat_template_kwargs={'enable_thinking': True},
            ))
        input_ids = torch.tensor([encoded['input_ids']], device=self.embed_device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return input_ids, attention_mask

    @torch.inference_mode()
    def latent_generate(self, messages: List[Dict[str, Any]]) -> Tuple[str, int, bool]:
        cfg = self.config
        input_ids, attention_mask = self._encode_prompt(messages)
        question_embeds = self.embed_tokens(input_ids).to(self.text_device)
        attention_mask = attention_mask.to(self.text_device)
        position_ids = _position_ids_from_mask(attention_mask)

        all_inputs_embeds = [question_embeds]
        all_attention_mask = attention_mask
        current_position_ids = position_ids[:, -1:]

        text_out = self.text_model(
            inputs_embeds=question_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past_key_values = text_out.past_key_values
        hidden = text_out.last_hidden_state

        batch_size = input_ids.shape[0]
        assert batch_size == 1, 'CoLaR latent inference currently supports batch_size=1 only.'
        is_done = torch.zeros((batch_size, 1), device=self.text_device, dtype=torch.bool)
        n_latent_forward = 0
        hit_eol = False

        for _ in range(cfg.max_n_latent_forward):
            lp_hidden = hidden[:, -1:, :].to(device=self.lp_device, dtype=torch.float32)
            if cfg.latent_temperature <= 0:
                # Deterministic/greedy latent step: we only need the distribution mean
                # (= loc), which is independent of the scale. Build the Normal with a
                # valid positive temperature so torch.distributions.Normal doesn't reject
                # scale=0 (log_std.exp() * 0). Do NOT pass temperature=0 here.
                dist = self.latent_policy(lp_hidden, temperature=1.0)
                sampled = dist.mean
            else:
                dist = self.latent_policy(lp_hidden, temperature=cfg.latent_temperature)
                sampled = dist.rsample()
            current_inputs_embeds = sampled.to(device=self.text_device, dtype=question_embeds.dtype) * cfg.embeds_std
            all_inputs_embeds.append(current_inputs_embeds)

            active = (~is_done).long()
            all_attention_mask = torch.cat([all_attention_mask, active], dim=1)
            current_position_ids = current_position_ids + active
            n_latent_forward += int(active.item())

            text_out = self.text_model(
                inputs_embeds=current_inputs_embeds,
                attention_mask=all_attention_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = text_out.past_key_values
            hidden = text_out.last_hidden_state

            logits = self.lm_head(hidden[:, -1:, :].to(self.lm_head_device))
            last_logits = logits[:, -1, :].float()
            if cfg.eol_temperature <= 0:
                next_token = last_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(last_logits / cfg.eol_temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.to(self.text_device)

            is_eol = next_token == cfg.think_close_id
            if is_eol.any():
                hit_eol = True
            is_done = is_done | is_eol
            if is_done.all():
                break

        close_ids = torch.full((batch_size, 1), cfg.think_close_id, device=self.embed_device, dtype=torch.long)
        close_embeds = self.embed_tokens(close_ids).to(self.text_device)
        all_inputs_embeds.append(close_embeds)
        all_attention_mask = torch.cat(
            [all_attention_mask, torch.ones((batch_size, 1), device=self.text_device, dtype=torch.long)],
            dim=1,
        )
        current_position_ids = current_position_ids + 1

        text_out = self.text_model(
            inputs_embeds=close_embeds,
            attention_mask=all_attention_mask,
            position_ids=current_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = text_out.past_key_values
        hidden = text_out.last_hidden_state

        generated_ids: List[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        for _ in range(cfg.max_new_tokens):
            logits = self.lm_head(hidden[:, -1:, :].to(self.lm_head_device))
            last_logits = logits[:, -1, :].float()
            if cfg.answer_temperature <= 0:
                next_token = last_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(last_logits / cfg.answer_temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = int(next_token.item())
            generated_ids.append(token_id)
            if cfg.progress_every > 0 and len(generated_ids) % cfg.progress_every == 0:
                print(f'[infer:latent] answer decode progress: {len(generated_ids)}/{cfg.max_new_tokens} tokens')
            if eos_token_id is not None and token_id == eos_token_id:
                break

            next_token = next_token.to(device=self.embed_device, dtype=torch.long)
            next_embeds = self.embed_tokens(next_token).to(self.text_device)
            all_attention_mask = torch.cat(
                [all_attention_mask, torch.ones((batch_size, 1), device=self.text_device, dtype=torch.long)],
                dim=1,
            )
            current_position_ids = current_position_ids + 1
            text_out = self.text_model(
                inputs_embeds=next_embeds,
                attention_mask=all_attention_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = text_out.past_key_values
            hidden = text_out.last_hidden_state

        answer_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return answer_text, n_latent_forward, hit_eol


def build_colar_latent_generator(
    *,
    model_path: str,
    checkpoint: Path,
    ckpt_args: Dict[str, Any],
    plugin_path: Path,
    torch_dtype: Optional[torch.dtype] = None,
    model_type: Optional[str] = None,
    attn_impl: Optional[str] = None,
    device_map: Optional[str] = 'auto',
    infer_overrides: Optional[Dict[str, Any]] = None,
):
    from swift.arguments import BaseArguments
    from swift.model import get_model_processor
    from swift.utils import import_external_file

    import_external_file(str(plugin_path))
    config = load_colar_infer_config(checkpoint, infer_overrides)
    latent_policy = load_latent_policy(checkpoint, config)

    model, processor = get_model_processor(
        model_path,
        torch_dtype=torch_dtype,
        model_type=model_type or ckpt_args.get('model_type') or 'qwen3_omni_moe',
        attn_impl=attn_impl or ckpt_args.get('attn_impl'),
        device_map=device_map,
        load_model=True,
    )
    model = Swift.from_pretrained(model, str(checkpoint))
    model.eval()

    args = BaseArguments.from_pretrained(str(checkpoint))
    template = args.get_template(processor)
    template.set_mode('transformers')

    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        lp_device = next(model.parameters()).device
    else:
        lp_device = model.device
    latent_policy = latent_policy.to(device=lp_device, dtype=torch.float32)

    return ColarLatentGenerator(model, template, latent_policy, config)
