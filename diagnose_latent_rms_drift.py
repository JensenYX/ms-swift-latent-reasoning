#!/usr/bin/env python
"""Free-rollout latent RMS drift diagnostics for Qwen3-Omni CoLaR checkpoints.

Answers one question: when free rollout fails to hit `</think>`, is the
predicted latent drifting in MAGNITUDE (RMS) or in DIRECTION?

For each sample this script reports, in the normalized latent space
(`embedding / embeds_std`, the space latent_policy is trained in):

1. gold:     per-position RMS of the gold compressed latents
2. teacher:  per-position RMS of `latent_policy(hidden).mean` along the GOLD
             path (one-step predictions; systematically low RMS here means
             MSE shrinkage bias, before any compounding)
3. rollout:  per-step RMS of the actually-fed latent during free rollout,
             plus cosine to the gold latent at the same index and the
             `</think>` logprob at every step

If teacher/rollout RMS tracks gold but cosine decays -> direction drift
(scheduled sampling territory). If RMS systematically shrinks or grows ->
magnitude drift (inference-time RMS renorm should help).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from colar_plugin.colar_infer import build_colar_latent_generator
from colar_plugin.qwen3omni_audio import merge_audio_embeds
from infer_qwen3omni_colar_overfit import (
    DEFAULT_MODEL,
    DEFAULT_PLUGIN,
    _canonicalize_path,
    find_latest_checkpoint,
    iter_jsonl,
    load_checkpoint_args,
    parse_torch_dtype,
    split_prompt_and_expected,
)


def _load_colar_meta(checkpoint: Path) -> Dict[str, Any]:
    meta_path = checkpoint / "colar_config.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _encode_training_sample(template, record: Dict[str, Any]) -> Dict[str, Any]:
    template.set_mode("train")
    return template.encode(copy.deepcopy(record))


@torch.inference_mode()
def _build_base_embeds(generator, encoded: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    embed_device = generator.embed_device
    text_device = generator.text_device
    input_ids = torch.tensor([encoded["input_ids"]], device=embed_device, dtype=torch.long)
    input_features = encoded.get("input_features")
    feature_attention_mask = encoded.get("feature_attention_mask")
    if input_features is not None:
        input_features = input_features.to(embed_device)
    if feature_attention_mask is not None:
        feature_attention_mask = feature_attention_mask.to(embed_device)
    inputs_embeds = generator.embed_tokens(input_ids)
    base_embeds = merge_audio_embeds(
        thinker=generator.thinker,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        audio_token_id=generator.audio_token_id,
    ).to(text_device)
    return input_ids[0], base_embeds[0]


@torch.inference_mode()
def build_gold_path(generator, record: Dict[str, Any], r: int, sqrt_mean: bool) -> Dict[str, Any]:
    """Rebuild the training-time compressed sequence and gold latent targets."""
    tokenizer = generator.tokenizer
    cfg = generator.config
    text_device = generator.text_device

    encoded = _encode_training_sample(generator.template, record)
    input_ids, base_embeds = _build_base_embeds(generator, encoded)

    ids = input_ids.tolist()
    open_idx = ids.index(cfg.think_open_id)
    close_idx = ids.index(cfg.think_close_id)
    if close_idx <= open_idx + 1:
        raise ValueError("empty thinking span")

    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    newline_id = newline_ids[0] if len(newline_ids) == 1 else None
    reason_start = open_idx + 1
    if newline_id is not None and reason_start < close_idx and ids[reason_start] == newline_id:
        reason_start += 1

    prefix_embed = base_embeds[:reason_start]
    reason_embed = base_embeds[reason_start:close_idx]
    answer_embed = base_embeds[close_idx:]

    n = int(reason_embed.shape[0])
    hidden_size = int(base_embeds.shape[-1])
    rem = (-n) % r
    if rem:
        reason_embed_p = torch.cat([reason_embed, reason_embed.new_zeros(rem, hidden_size)], dim=0)
        valid = torch.cat([torch.ones(n, device=text_device), torch.zeros(rem, device=text_device)])
    else:
        reason_embed_p = reason_embed
        valid = torch.ones(n, device=text_device)
    k = int(reason_embed_p.shape[0] // r)

    group_embed = reason_embed_p.reshape(k, r, hidden_size)
    group_valid = valid.reshape(k, r)
    count = group_valid.sum(dim=1)
    count_norm = count.sqrt() if sqrt_mean else count
    comp_embed = (group_embed * group_valid.unsqueeze(-1)).sum(dim=1) / (count_norm.unsqueeze(-1) + 1e-5)
    comp_embed = comp_embed.to(dtype=prefix_embed.dtype)

    compressed_embeds = torch.cat([prefix_embed, comp_embed, answer_embed], dim=0).to(dtype=prefix_embed.dtype)
    # Normalized-space gold latents: what latent_policy is trained to produce.
    gold_norm = comp_embed.float() / float(cfg.embeds_std)  # [k, H]

    return {
        "compressed_embeds": compressed_embeds,
        "gold_norm": gold_norm,
        "prefix_len": int(prefix_embed.shape[0]),
        "compressed_tokens": k,
        "reason_tokens": n,
        "has_audio": encoded.get("input_features") is not None,
    }


def _rms(x: torch.Tensor) -> torch.Tensor:
    """Per-row RMS over the last dim. x: [..., H] -> [...]"""
    return x.float().pow(2).mean(dim=-1).sqrt()


@torch.inference_mode()
def teacher_forced_stats(generator, gold_path: Dict[str, Any]) -> Dict[str, Any]:
    """One-step policy predictions along the GOLD compressed path."""
    cfg = generator.config
    compressed = gold_path["compressed_embeds"].unsqueeze(0)
    attention_mask = torch.ones((1, compressed.shape[1]), device=compressed.device, dtype=torch.long)
    position_ids = torch.clamp_min(attention_mask.long().cumsum(dim=1) - 1, 0)
    text_out = generator.text_model(
        inputs_embeds=compressed,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    hidden = text_out.last_hidden_state[0]

    prefix_len = gold_path["prefix_len"]
    k = gold_path["compressed_tokens"]
    # Positions whose policy output is supervised against gold latents:
    # prefix_len-1 (predicts gold[0]) .. prefix_len+k-2 (predicts gold[k-1]).
    positions = torch.arange(prefix_len - 1, prefix_len + k - 1, device=hidden.device)
    lp_hidden = hidden.index_select(0, positions).unsqueeze(0).to(
        device=generator.lp_device, dtype=torch.float32)
    dist = generator.latent_policy(lp_hidden, temperature=1.0)
    pred_mean = dist.mean[0]                                     # [k, H]
    pred_std = dist.scale[0]                                     # [k, H]
    gold = gold_path["gold_norm"].to(pred_mean.device)           # [k, H]

    return {
        "pred_rms": _rms(pred_mean).cpu().tolist(),
        "pred_std_mean": pred_std.mean(dim=-1).cpu().tolist(),
        "cos": F.cosine_similarity(pred_mean, gold, dim=-1).cpu().tolist(),
    }


@torch.inference_mode()
def free_rollout_stats(
    generator,
    messages: List[Dict[str, Any]],
    audios: List[str],
    gold_norm: torch.Tensor,
    max_latent_forward: int,
    latent_temperature: float,
    eol_temperature: float,
) -> Dict[str, Any]:
    """Replicates ColarLatentGenerator.latent_generate latent loop with per-step stats."""
    cfg = generator.config
    input_ids, attention_mask, input_features, feature_attention_mask = generator._encode_prompt(messages, audios)
    question_embeds = generator._build_prompt_embeds(input_ids, input_features, feature_attention_mask).to(
        generator.text_device)
    attention_mask = attention_mask.to(generator.text_device)
    position_ids = torch.clamp_min(attention_mask.long().cumsum(dim=1) - 1, 0)

    text_out = generator.text_model(
        inputs_embeds=question_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    past_key_values = text_out.past_key_values
    hidden = text_out.last_hidden_state
    all_attention_mask = attention_mask
    current_position_ids = position_ids[:, -1:]

    gold_norm = gold_norm.to(generator.lp_device)
    gold_len = int(gold_norm.shape[0])

    rms_mean: List[float] = []
    rms_sampled: List[float] = []
    std_mean: List[float] = []
    cos_gold_mean: List[float] = []
    cos_gold_sampled: List[float] = []
    eol_logprob: List[float] = []
    eol_rank: List[int] = []

    hit_eol = False
    n_latent_forward = 0

    for step in range(max_latent_forward):
        lp_hidden = hidden[:, -1:, :].to(device=generator.lp_device, dtype=torch.float32)
        if latent_temperature <= 0:
            dist = generator.latent_policy(lp_hidden, temperature=1.0)
            sampled = dist.mean
        else:
            dist = generator.latent_policy(lp_hidden, temperature=latent_temperature)
            sampled = dist.rsample()

        mean_vec = dist.mean[0, 0]
        sampled_vec = sampled[0, 0]
        rms_mean.append(float(_rms(mean_vec).item()))
        rms_sampled.append(float(_rms(sampled_vec).item()))
        std_mean.append(float(dist.scale[0, 0].mean().item()))
        if step < gold_len:
            g = gold_norm[step]
            cos_gold_mean.append(float(F.cosine_similarity(mean_vec, g, dim=-1).item()))
            cos_gold_sampled.append(float(F.cosine_similarity(sampled_vec, g, dim=-1).item()))

        current_inputs_embeds = sampled.to(
            device=generator.text_device, dtype=question_embeds.dtype) * cfg.embeds_std
        all_attention_mask = torch.cat(
            [all_attention_mask,
             torch.ones((1, 1), device=generator.text_device, dtype=torch.long)], dim=1)
        current_position_ids = current_position_ids + 1
        n_latent_forward += 1

        text_out = generator.text_model(
            inputs_embeds=current_inputs_embeds,
            attention_mask=all_attention_mask,
            position_ids=current_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = text_out.past_key_values
        hidden = text_out.last_hidden_state

        logits = generator.lm_head(hidden[:, -1:, :].to(generator.lm_head_device))
        last_logits = logits[0, -1, :].float()
        logprobs = torch.log_softmax(last_logits, dim=-1)
        eol_logprob.append(float(logprobs[cfg.think_close_id].item()))
        eol_rank.append(int((last_logits > last_logits[cfg.think_close_id]).sum().item()) + 1)

        if eol_temperature <= 0:
            next_token = int(last_logits.argmax().item())
        else:
            probs = torch.softmax(last_logits / eol_temperature, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())
        if next_token == cfg.think_close_id:
            hit_eol = True
            break

    return {
        "n_latent_forward": n_latent_forward,
        "hit_eol": hit_eol,
        "rms_mean": rms_mean,
        "rms_sampled": rms_sampled,
        "std_mean": std_mean,
        "cos_gold_mean": cos_gold_mean,
        "cos_gold_sampled": cos_gold_sampled,
        "eol_logprob": eol_logprob,
        "eol_rank": eol_rank,
    }


def _probe(values: List[float], n_points: int = 8) -> List[Tuple[int, float]]:
    if not values:
        return []
    if len(values) <= n_points:
        return list(enumerate(values))
    idxs = [round(i * (len(values) - 1) / (n_points - 1)) for i in range(n_points)]
    return [(i, values[i]) for i in idxs]


def _fmt_probe(pairs: List[Tuple[int, float]], fmt: str = "{:7.3f}") -> Tuple[str, str]:
    steps = "".join(f"{i:>8d}" for i, _ in pairs)
    vals = "".join(f"{fmt.format(v):>8s}" for _, v in pairs)
    return steps, vals


def print_report(row: Dict[str, Any]) -> None:
    gold = row["gold"]
    tf = row["teacher_forced"]
    ro = row["rollout"]
    print(f"[rms] sample {row['idx']} audio={row['has_audio']} "
          f"gold_len={gold['len']} rollout={ro['n_latent_forward']} hit_eol={ro['hit_eol']}")
    print(f"  gold_rms:    mean={gold['rms_stats']['mean']:.4f} "
          f"min={gold['rms_stats']['min']:.4f} max={gold['rms_stats']['max']:.4f}")
    print(f"  tf_pred_rms: mean={tf['rms_stats']['mean']:.4f} "
          f"min={tf['rms_stats']['min']:.4f} max={tf['rms_stats']['max']:.4f} "
          f"(shrinkage ratio vs gold: {tf['rms_stats']['mean'] / max(gold['rms_stats']['mean'], 1e-9):.3f})")
    print(f"  tf_cos:      mean={tf['cos_stats']['mean']:.4f} min={tf['cos_stats']['min']:.4f}")

    steps, vals = _fmt_probe(_probe(ro["rms_sampled"]))
    print(f"  rollout step:     {steps}")
    print(f"  rollout rms_samp: {vals}")
    _, vals = _fmt_probe(_probe(ro["rms_mean"]))
    print(f"  rollout rms_mean: {vals}")
    if ro["cos_gold_sampled"]:
        _, vals = _fmt_probe(_probe(ro["cos_gold_sampled"]))
        print(f"  cos_to_gold:      {vals}")
    _, vals = _fmt_probe(_probe(ro["eol_logprob"]), fmt="{:7.2f}")
    print(f"  eol_logprob:      {vals}")


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    t = torch.tensor(values)
    return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-root", required=False, default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN))
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--attn-impl", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-latent-forward", type=int, default=400)
    parser.add_argument("--latent-temperature", type=float, default=1.0)
    parser.add_argument("--eol-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_arg = _canonicalize_path(args.checkpoint) if args.checkpoint else None
    if checkpoint_arg is None:
        if not args.checkpoint_root:
            raise ValueError("either --checkpoint or --checkpoint-root is required")
        checkpoint = find_latest_checkpoint(Path(args.checkpoint_root))
    elif checkpoint_arg.is_dir() and (checkpoint_arg / "adapter_config.json").exists():
        checkpoint = checkpoint_arg
    else:
        checkpoint = find_latest_checkpoint(checkpoint_arg)

    ckpt_args = load_checkpoint_args(checkpoint)
    model = _canonicalize_path(args.model or ckpt_args.get("model") or str(DEFAULT_MODEL))
    plugin = _canonicalize_path(args.plugin)
    dataset = _canonicalize_path(args.dataset)
    for name, p in (("model", model), ("plugin", plugin), ("dataset", dataset)):
        if p is None or not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    meta = _load_colar_meta(checkpoint)
    r = int(meta.get("fixed_r") or meta.get("max_r") or 2)
    sqrt_mean = bool(meta.get("sqrt_mean", False))

    torch_dtype = parse_torch_dtype(args.torch_dtype or ckpt_args.get("torch_dtype"))
    model_type = args.model_type or ckpt_args.get("model_type") or "qwen3_omni_moe"
    attn_impl = args.attn_impl if args.attn_impl != "auto" else ckpt_args.get("attn_impl")
    device_map = None if args.device_map == "none" else args.device_map

    print(f"[rms] checkpoint: {checkpoint}")
    print(f"[rms] dataset: {dataset}")
    print(f"[rms] r={r} sqrt_mean={sqrt_mean} latent_temp={args.latent_temperature} "
          f"eol_temp={args.eol_temperature} max_latent_forward={args.max_latent_forward}")

    generator = build_colar_latent_generator(
        model_path=str(model),
        checkpoint=checkpoint,
        ckpt_args=ckpt_args,
        plugin_path=plugin,
        torch_dtype=torch_dtype,
        model_type=model_type,
        attn_impl=attn_impl,
        device_map=device_map,
        infer_overrides={},
    )

    records = list(iter_jsonl(dataset))
    if args.limit:
        records = records[:args.limit]

    out_f = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    try:
        for idx, record in enumerate(records):
            torch.manual_seed(args.seed + idx)
            prompt_messages, _expected, audios = split_prompt_and_expected(record)
            gold_path = build_gold_path(generator, record, r=r, sqrt_mean=sqrt_mean)
            gold_rms = _rms(gold_path["gold_norm"]).cpu().tolist()
            tf = teacher_forced_stats(generator, gold_path)
            rollout = free_rollout_stats(
                generator,
                prompt_messages,
                audios,
                gold_path["gold_norm"],
                max_latent_forward=args.max_latent_forward,
                latent_temperature=args.latent_temperature,
                eol_temperature=args.eol_temperature,
            )
            row = {
                "idx": idx,
                "has_audio": gold_path["has_audio"],
                "gold": {
                    "len": gold_path["compressed_tokens"],
                    "reason_tokens": gold_path["reason_tokens"],
                    "rms": gold_rms,
                    "rms_stats": _stats(gold_rms),
                },
                "teacher_forced": {
                    "rms": tf["pred_rms"],
                    "rms_stats": _stats(tf["pred_rms"]),
                    "std_mean_stats": _stats(tf["pred_std_mean"]),
                    "cos": tf["cos"],
                    "cos_stats": _stats(tf["cos"]),
                },
                "rollout": rollout,
            }
            print_report(row)
            if out_f is not None:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        if out_f is not None:
            out_f.close()
            print(f"[rms] wrote: {args.output}")


if __name__ == "__main__":
    main()
