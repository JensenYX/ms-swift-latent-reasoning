#!/usr/bin/env python
"""Seed sweep for the knife-edge EOL stop event.

diagnose_latent_rms_drift.py showed that the idx=1 EOL miss is stochastic:
the rollout tracks the gold trajectory (cos ~0.99, stable RMS) and the stop
decision is a single sharp spike at the gold end step. This script estimates
P(hit_eol) for one sample across seeds, and for missed runs reports the best
post-gold-end `</think>` logprob (is the miss recoverable at all?).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from diagnose_latent_rms_drift import (
    _load_colar_meta,
    _rms,
    build_gold_path,
    free_rollout_stats,
)
from colar_plugin.colar_infer import build_colar_latent_generator
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


DEFAULT_QUERY_ANCHOR_PLUGIN = Path(__file__).resolve().parent / "colar_plugin/query_anchor_plugin.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("colar", "query_anchor"), default="colar")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN))
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--attn-impl", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--idx", type=int, default=1, help="Sample index in the dataset.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds (0..N-1).")
    parser.add_argument("--max-latent-forward", type=int, default=400)
    parser.add_argument("--latent-temperature", type=float, default=1.0)
    parser.add_argument("--eol-temperature", type=float, default=1.0)
    parser.add_argument("--latent-rms-target", type=float, default=0.0)
    return parser.parse_args()


@torch.inference_mode()
def query_anchor_free_rollout_stats(
    generator,
    messages,
    audios,
    gold_norm: torch.Tensor,
    max_latent_forward: int,
    latent_temperature: float,
    eol_temperature: float,
):
    """Query-anchor version of diagnose_latent_rms_drift.free_rollout_stats."""
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

    query_pos = generator._query_anchor_position(input_ids)
    query_hidden = hidden[:, query_pos:query_pos + 1, :].detach()
    if cfg.residual_source == "hidden":
        residual_query = query_hidden
    else:
        residual_query = question_embeds[:, query_pos:query_pos + 1, :].detach()

    gold_norm = gold_norm.to(generator.lp_device)
    gold_len = int(gold_norm.shape[0])

    rms_mean = []
    rms_sampled = []
    std_mean = []
    cos_gold_mean = []
    cos_gold_sampled = []
    eol_logprob = []
    eol_rank = []

    hit_eol = False
    n_latent_forward = 0

    for step in range(max_latent_forward):
        lp_hidden = hidden[:, -1:, :].to(device=generator.lp_device, dtype=torch.float32)
        lp_query = query_hidden.to(device=generator.lp_device, dtype=torch.float32)
        if latent_temperature <= 0:
            dist = generator.latent_policy(lp_hidden, query_hidden=lp_query, temperature=1.0)
            sampled = dist.mean
        else:
            dist = generator.latent_policy(lp_hidden, query_hidden=lp_query, temperature=latent_temperature)
            sampled = dist.rsample()
        if cfg.latent_rms_target > 0:
            rms = sampled.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            sampled = sampled * (cfg.latent_rms_target / rms).to(sampled.dtype)

        lp_residual_query = residual_query.to(device=generator.lp_device, dtype=torch.float32)
        anchored = generator.latent_policy.apply_anchor_normalized(sampled, lp_residual_query)

        mean_vec = dist.mean[0, 0]
        fed_vec = anchored[0, 0]
        rms_mean.append(float(_rms(mean_vec).item()))
        rms_sampled.append(float(_rms(fed_vec).item()))
        std_mean.append(float(dist.scale[0, 0].mean().item()))
        if step < gold_len:
            g = gold_norm[step]
            cos_gold_mean.append(float(F.cosine_similarity(mean_vec, g, dim=-1).item()))
            cos_gold_sampled.append(float(F.cosine_similarity(fed_vec, g, dim=-1).item()))

        current_inputs_embeds = anchored.to(device=generator.text_device, dtype=question_embeds.dtype) * cfg.embeds_std
        all_attention_mask = torch.cat(
            [all_attention_mask, torch.ones((1, 1), device=generator.text_device, dtype=torch.long)], dim=1)
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


def main() -> None:
    args = parse_args()
    checkpoint_arg = _canonicalize_path(args.checkpoint) if args.checkpoint else None
    if checkpoint_arg is None:
        checkpoint = find_latest_checkpoint(Path(args.checkpoint_root))
    elif checkpoint_arg.is_dir() and (checkpoint_arg / "adapter_config.json").exists():
        checkpoint = checkpoint_arg
    else:
        checkpoint = find_latest_checkpoint(checkpoint_arg)

    ckpt_args = load_checkpoint_args(checkpoint)
    model = _canonicalize_path(args.model or ckpt_args.get("model") or str(DEFAULT_MODEL))
    plugin = _canonicalize_path(args.plugin)
    dataset = _canonicalize_path(args.dataset)

    meta = _load_colar_meta(checkpoint)
    r = int(meta.get("fixed_r") or meta.get("max_r") or 2)
    sqrt_mean = bool(meta.get("sqrt_mean", False))

    torch_dtype = parse_torch_dtype(args.torch_dtype or ckpt_args.get("torch_dtype"))
    model_type = args.model_type or ckpt_args.get("model_type") or "qwen3_omni_moe"
    attn_impl = args.attn_impl if args.attn_impl != "auto" else ckpt_args.get("attn_impl")
    device_map = None if args.device_map == "none" else args.device_map

    print(f"[eol-sweep] backend: {args.backend}")
    print(f"[eol-sweep] checkpoint: {checkpoint}")
    print(f"[eol-sweep] dataset: {dataset} idx={args.idx} seeds=0..{args.seeds - 1}")
    print(f"[eol-sweep] r={r} sqrt_mean={sqrt_mean} latent_temp={args.latent_temperature} "
          f"eol_temp={args.eol_temperature} max_latent_forward={args.max_latent_forward}")

    infer_overrides = {"latent_rms_target": args.latent_rms_target}
    if args.backend == "query_anchor":
        from colar_plugin.query_anchor_infer import build_query_anchor_latent_generator

        if args.plugin == str(DEFAULT_PLUGIN):
            plugin = DEFAULT_QUERY_ANCHOR_PLUGIN
        generator = build_query_anchor_latent_generator(
            model_path=str(model),
            checkpoint=checkpoint,
            ckpt_args=ckpt_args,
            plugin_path=plugin,
            torch_dtype=torch_dtype,
            model_type=model_type,
            attn_impl=attn_impl,
            device_map=device_map,
            infer_overrides=infer_overrides,
        )
        rollout_fn = query_anchor_free_rollout_stats
    else:
        generator = build_colar_latent_generator(
            model_path=str(model),
            checkpoint=checkpoint,
            ckpt_args=ckpt_args,
            plugin_path=plugin,
            torch_dtype=torch_dtype,
            model_type=model_type,
            attn_impl=attn_impl,
            device_map=device_map,
            infer_overrides=infer_overrides,
        )
        rollout_fn = free_rollout_stats

    records = list(iter_jsonl(dataset))
    record = records[args.idx]
    prompt_messages, _expected, audios = split_prompt_and_expected(record)
    gold_path = build_gold_path(generator, record, r=r, sqrt_mean=sqrt_mean)
    gold_len = gold_path["compressed_tokens"]
    print(f"[eol-sweep] gold_len={gold_len} has_audio={gold_path['has_audio']}")

    rows = []
    hits = 0
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        ro = rollout_fn(
            generator,
            prompt_messages,
            audios,
            gold_path["gold_norm"],
            max_latent_forward=args.max_latent_forward,
            latent_temperature=args.latent_temperature,
            eol_temperature=args.eol_temperature,
        )
        hits += int(ro["hit_eol"])
        # At/after the gold end: how close did `</think>` ever get?
        tail_logprob = ro["eol_logprob"][gold_len - 1:]
        best_tail = max(tail_logprob) if tail_logprob else float("-inf")
        best_tail_step = (gold_len - 1 + tail_logprob.index(best_tail)) if tail_logprob else -1
        # Before the gold end: any premature near-stop?
        head_logprob = ro["eol_logprob"][:gold_len - 1]
        best_head = max(head_logprob) if head_logprob else float("-inf")
        row = {
            "seed": seed,
            "hit_eol": ro["hit_eol"],
            "n_latent_forward": ro["n_latent_forward"],
            "eol_logprob_at_gold_end": ro["eol_logprob"][gold_len - 1] if len(ro["eol_logprob"]) >= gold_len else None,
            "best_tail_logprob": best_tail,
            "best_tail_step": best_tail_step,
            "best_head_logprob": best_head,
            "cos_gold_last8": [round(c, 4) for c in ro["cos_gold_sampled"][-8:]],
        }
        rows.append(row)
        print(f"[eol-sweep] seed={seed} hit={ro['hit_eol']} steps={ro['n_latent_forward']} "
              f"eol_lp@gold_end={row['eol_logprob_at_gold_end']} "
              f"best_tail_lp={best_tail:.2f}@{best_tail_step} best_head_lp={best_head:.2f}")

    print(f"[eol-sweep] P(hit_eol) = {hits}/{args.seeds}")
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[eol-sweep] wrote: {args.output}")


if __name__ == "__main__":
    main()
