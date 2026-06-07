"""Qwen3-Omni audio embedding helpers for CoLaR.

The CoLaR trainer/inferencer call the text backbone directly with
``inputs_embeds``. For audio samples we therefore need to reproduce the small
part of Qwen3-Omni's thinker forward that scatters audio tower outputs into the
expanded ``<|audio_pad|>`` token positions.
"""
from __future__ import annotations

from typing import Optional

import torch


def module_device(module) -> torch.device:
    for p in module.parameters(recurse=True):
        return p.device
    for b in module.buffers(recurse=True):
        return b.device
    return torch.device('cpu')


def resolve_audio_token_id(thinker, tokenizer=None, model=None) -> Optional[int]:
    for obj in (
            getattr(thinker, 'config', None),
            getattr(getattr(model, 'config', None), 'thinker_config', None),
            getattr(model, 'config', None),
    ):
        if obj is None:
            continue
        for name in ('audio_token_id', 'audio_token_index'):
            value = getattr(obj, name, None)
            if value is not None:
                return int(value)
    if tokenizer is not None:
        ids = tokenizer.encode('<|audio_pad|>', add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def merge_audio_embeds(
    *,
    thinker,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    input_features: Optional[torch.Tensor],
    feature_attention_mask: Optional[torch.Tensor],
    audio_token_id: Optional[int],
) -> torch.Tensor:
    """Return ``inputs_embeds`` with audio features scattered into audio tokens.

    Pure text samples pass ``input_features=None`` and are returned unchanged.
    """
    if input_features is None:
        return inputs_embeds
    if audio_token_id is None:
        raise ValueError('Cannot merge Qwen3-Omni audio: audio token id was not resolved.')
    if not hasattr(thinker, 'get_audio_features'):
        raise AttributeError('Qwen3-Omni thinker has no get_audio_features method.')

    audio_tower = getattr(thinker, 'audio_tower', thinker)
    audio_device = module_device(audio_tower)
    audio_dtype = getattr(audio_tower, 'dtype', inputs_embeds.dtype)
    input_features = input_features.to(device=audio_device, dtype=audio_dtype)
    if feature_attention_mask is not None:
        feature_attention_mask = feature_attention_mask.to(device=audio_device)

    audio_res = thinker.get_audio_features(input_features, feature_attention_mask)
    audio_embeds = audio_res.last_hidden_state if hasattr(audio_res, 'last_hidden_state') else audio_res
    audio_embeds = audio_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    audio_mask = (input_ids.to(inputs_embeds.device) == audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    n_audio_slots = int(audio_mask[..., 0].sum().item())
    if n_audio_slots == 0:
        raise ValueError('Audio features were provided, but no <|audio_pad|> tokens were found in input_ids.')
    if audio_embeds.numel() != n_audio_slots * inputs_embeds.shape[-1]:
        raise ValueError(
            'Audio features and audio tokens do not match: '
            f'tokens={n_audio_slots}, audio_embeds_shape={tuple(audio_embeds.shape)}, '
            f'hidden_size={inputs_embeds.shape[-1]}')
    return inputs_embeds.masked_scatter(audio_mask, audio_embeds)
