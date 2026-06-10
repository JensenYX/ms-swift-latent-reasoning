#!/usr/bin/env python
"""Teacher-forcing diagnostics for query-anchored CoLaR checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from swift.infer_engine import InferRequest

from colar_plugin.colar_template import THINK_CLOSE_ID, THINK_OPEN_ID
from colar_plugin.qwen3omni_audio import merge_audio_embeds
from colar_plugin.query_anchor_infer import build_query_anchor_latent_generator
from infer_qwen3omni_colar_overfit import (
    DEFAULT_MODEL,
    _canonicalize_path,
    find_latest_checkpoint,
    iter_jsonl,
    load_checkpoint_args,
    parse_torch_dtype,
    split_prompt_and_expected,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT_ROOT = REPO_ROOT / "output/qwen3omni_query_anchor_condonly_promptlast_r5_mse_sqrt/overfit_ckpt_audio_r5"
DEFAULT_DATASET = REPO_ROOT / "output/qwen3omni_colar/overfit_mixed_audio_text.jsonl"
DEFAULT_PLUGIN = REPO_ROOT / "colar_plugin/query_anchor_plugin.py"


@dataclass
class CompressedPath:
    input_ids: torch.Tensor
    labels: Optional[torch.Tensor]
    compressed_embeds: torch.Tensor
    attention_mask: torch.Tensor
    mask_positions: torch.Tensor
    target_embeds: torch.Tensor
    prefix_len: int
    reason_start: int
    close_index_original: int
    reason_tokens: int
    compressed_tokens: int
    answer_ids: torch.Tensor
    query_pos: int


def _as_tensor_1d(values: Iterable[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(list(values), device=device, dtype=torch.long)


def _token_text(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except Exception:
        return f"<decode-error:{token_id}>"


def _top_tokens(logits: torch.Tensor, tokenizer, k: int = 5) -> List[Dict[str, Any]]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    vals, ids = torch.topk(log_probs, k=min(k, log_probs.numel()))
    return [{
        "id": int(tid),
        "text": _token_text(tokenizer, int(tid)),
        "logprob": float(logp),
        "prob": float(torch.exp(torch.tensor(logp)).item()),
    } for logp, tid in zip(vals.tolist(), ids.tolist())]


def _rank_and_logprob(logits: torch.Tensor, target_id: int) -> Tuple[int, float]:
    logits = logits.float()
    target_logit = logits[target_id]
    rank = int((logits > target_logit).sum().item()) + 1
    logprob = float(torch.log_softmax(logits, dim=-1)[target_id].item())
    return rank, logprob


def _stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float().cpu()
    if x.numel() == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "rms": float("nan")}
    return {
        "mean": float(x.mean().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "rms": float(x.pow(2).mean().sqrt().item()),
    }


def _load_colar_meta(checkpoint: Path) -> Dict[str, Any]:
    meta_path = checkpoint / "colar_config.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _query_anchor_position(row: List[int], prefix_len: int, cfg) -> int:
    if getattr(cfg, "hidden_source", "prompt_last") == "prompt_last":
        return max(0, prefix_len - 1)
    try:
        think_open = row.index(cfg.think_open_id)
        im_end_positions = [i for i, tok in enumerate(row[:think_open]) if tok == cfg.im_end_id]
        return max(0, im_end_positions[-1] - 1)
    except (ValueError, IndexError):
        return max(0, prefix_len - 1)


def _encode_training_sample(template, record: Dict[str, Any]) -> Dict[str, Any]:
    template.set_mode("train")
    return template.encode(copy.deepcopy(record))


def _encode_infer_prompt_ids(generator, messages: List[Dict[str, Any]], audios: List[str]) -> List[int]:
    input_ids, _attention_mask, _input_features, _feature_attention_mask = generator._encode_prompt(messages, audios)
    return input_ids[0].detach().cpu().tolist()


@torch.inference_mode()
def _build_base_embeds(generator, encoded: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
    return input_ids[0], base_embeds[0], input_features


def build_compressed_path(generator, record: Dict[str, Any], r: int, sqrt_mean: bool) -> Tuple[CompressedPath, Dict[str, Any]]:
    tokenizer = generator.tokenizer
    cfg = generator.config
    embed_device = generator.embed_device
    text_device = generator.text_device

    encoded = _encode_training_sample(generator.template, record)
    input_ids, base_embeds, input_features = _build_base_embeds(generator, encoded)
    labels = encoded.get("labels")
    label_tensor = _as_tensor_1d(labels, embed_device) if labels is not None else None

    ids = input_ids.tolist()
    try:
        open_idx = ids.index(cfg.think_open_id)
        close_idx = ids.index(cfg.think_close_id)
    except ValueError as exc:
        raise ValueError("sample does not contain expected thinking markers") from exc
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
    answer_ids = input_ids[close_idx:].to(text_device)

    n = int(reason_embed.shape[0])
    hidden = int(base_embeds.shape[-1])
    rem = (-n) % r
    if rem:
        reason_embed_p = torch.cat([reason_embed, reason_embed.new_zeros(rem, hidden)], dim=0)
        valid = torch.cat([torch.ones(n, device=text_device), torch.zeros(rem, device=text_device)])
    else:
        reason_embed_p = reason_embed
        valid = torch.ones(n, device=text_device)
    k = int(reason_embed_p.shape[0] // r)

    group_embed = reason_embed_p.reshape(k, r, hidden)
    group_valid = valid.reshape(k, r)
    count = group_valid.sum(dim=1)
    count_norm = count.sqrt() if sqrt_mean else count
    comp_embed = (group_embed * group_valid.unsqueeze(-1)).sum(dim=1) / (count_norm.unsqueeze(-1) + 1e-5)
    comp_embed = comp_embed.to(dtype=prefix_embed.dtype)

    compressed_embeds = torch.cat([prefix_embed, comp_embed, answer_embed], dim=0).to(dtype=prefix_embed.dtype)
    attention_mask = torch.ones((1, compressed_embeds.shape[0]), device=text_device, dtype=torch.long)

    anchor = int(prefix_embed.shape[0] - 1)
    mask_positions = torch.arange(anchor, int(prefix_embed.shape[0]) + k, device=text_device, dtype=torch.long)
    target_embeds = torch.roll(compressed_embeds, shifts=-1, dims=0)[mask_positions]
    query_pos = _query_anchor_position(ids, int(prefix_embed.shape[0]), cfg)

    debug = {
        "open_index_original": open_idx,
        "close_index_original": close_idx,
        "newline_id": newline_id,
        "prefix_len": int(prefix_embed.shape[0]),
        "reason_start": reason_start,
        "reason_tokens": n,
        "compressed_tokens": k,
        "compressed_seq_len": int(compressed_embeds.shape[0]),
        "full_seq_len": int(input_ids.numel()),
        "query_pos": query_pos,
        "query_token": _token_text(tokenizer, ids[query_pos]),
        "has_audio_features": input_features is not None,
    }
    return (
        CompressedPath(
            input_ids=input_ids,
            labels=label_tensor,
            compressed_embeds=compressed_embeds,
            attention_mask=attention_mask,
            mask_positions=mask_positions,
            target_embeds=target_embeds,
            prefix_len=int(prefix_embed.shape[0]),
            reason_start=reason_start,
            close_index_original=close_idx,
            reason_tokens=n,
            compressed_tokens=k,
            answer_ids=answer_ids,
            query_pos=query_pos,
        ),
        debug,
    )


@torch.inference_mode()
def diagnose_one(generator, record: Dict[str, Any], idx: int, r: int, sqrt_mean: bool, top_k: int) -> Dict[str, Any]:
    tokenizer = generator.tokenizer
    cfg = generator.config
    text_model = generator.text_model
    lm_head = generator.lm_head
    latent_policy = generator.latent_policy

    prompt_messages, expected, audios = split_prompt_and_expected(record)
    path, debug = build_compressed_path(generator, record, r, sqrt_mean)
    prompt_ids = _encode_infer_prompt_ids(generator, prompt_messages, audios)
    prefix_ids = path.input_ids[:path.prefix_len].detach().cpu().tolist()

    position_ids = torch.clamp_min(path.attention_mask.long().cumsum(dim=1) - 1, 0)
    text_out = text_model(
        inputs_embeds=path.compressed_embeds.unsqueeze(0),
        attention_mask=path.attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    hidden = text_out.last_hidden_state[0]
    query_hidden = hidden[path.query_pos:path.query_pos + 1].unsqueeze(0).detach()

    lp_hidden = hidden.index_select(0, path.mask_positions).unsqueeze(0).to(
        device=generator.lp_device, dtype=torch.float32)
    lp_query = query_hidden.to(device=generator.lp_device, dtype=torch.float32)
    dist = latent_policy(lp_hidden, query_hidden=lp_query, temperature=1.0)
    pred = dist.mean[0]
    gold = path.target_embeds.to(device=generator.lp_device, dtype=torch.float32) / cfg.embeds_std
    cos = F.cosine_similarity(pred, gold, dim=-1)
    mse = F.mse_loss(pred, gold, reduction="none").mean(dim=-1)
    nll = -dist.log_prob(gold).mean(dim=-1).squeeze(0)

    last_comp_pos = path.prefix_len + path.compressed_tokens - 1
    close_pos = path.prefix_len + path.compressed_tokens
    close_logits = lm_head(hidden[last_comp_pos:last_comp_pos + 1].to(generator.lm_head_device))[0].float()
    close_rank, close_logprob = _rank_and_logprob(close_logits, cfg.think_close_id)

    first_logits = lm_head(hidden[path.mask_positions[0]:path.mask_positions[0] + 1].to(generator.lm_head_device))[0].float()
    first_target_id = int(path.input_ids[path.reason_start].item())
    first_rank, first_logprob = _rank_and_logprob(first_logits, first_target_id)

    h_stats = _stats(lp_hidden)
    q_expanded = lp_query.expand(-1, lp_hidden.shape[1], -1)
    q_stats = _stats(q_expanded)
    prod_stats = _stats(lp_hidden * q_expanded)

    return {
        "idx": idx,
        "has_audio": bool(audios),
        "prefix_match": prompt_ids == prefix_ids,
        "prompt_len": len(prompt_ids),
        "prefix_len": path.prefix_len,
        "prompt_tail": [_token_text(tokenizer, x) for x in prompt_ids[-8:]],
        "prefix_tail": [_token_text(tokenizer, x) for x in prefix_ids[-8:]],
        "expected_prefix": (expected or "")[:240],
        "debug": debug,
        "latent": {
            "steps": int(path.mask_positions.numel()),
            "cos": _stats(cos),
            "mse": _stats(mse),
            "nll": _stats(nll),
            "first_cos": float(cos[0].item()),
            "last_close_target_cos": float(cos[-1].item()),
            "first_nll": float(nll[0].item()),
            "last_close_target_nll": float(nll[-1].item()),
        },
        "condition_input_stats": {
            "h": h_stats,
            "q": q_stats,
            "h_times_q": prod_stats,
        },
        "gold_first_reason_logits": {
            "target_id": first_target_id,
            "target_text": _token_text(tokenizer, first_target_id),
            "rank": first_rank,
            "logprob": first_logprob,
            "top": _top_tokens(first_logits, tokenizer, top_k),
        },
        "gold_eol_logits": {
            "target_id": int(cfg.think_close_id),
            "target_text": _token_text(tokenizer, int(cfg.think_close_id)),
            "rank": close_rank,
            "logprob": close_logprob,
            "top": _top_tokens(close_logits, tokenizer, top_k),
        },
        "close_pos": close_pos,
    }


def print_report(row: Dict[str, Any]) -> None:
    lat = row["latent"]
    eol = row["gold_eol_logits"]
    first = row["gold_first_reason_logits"]
    debug = row["debug"]
    stats = row["condition_input_stats"]
    print(f"[qa-tf] sample {row['idx']} audio={row['has_audio']}")
    print(
        f"  prefix_match={row['prefix_match']} prompt_len={row['prompt_len']} "
        f"prefix_len={row['prefix_len']} tail={row['prefix_tail']}")
    print(
        f"  full_len={debug['full_seq_len']} compressed_len={debug['compressed_seq_len']} "
        f"reason_tokens={debug['reason_tokens']} compressed_tokens={debug['compressed_tokens']} "
        f"query_pos={debug['query_pos']} query_token={debug['query_token']!r}")
    print(
        f"  latent: steps={lat['steps']} cos_mean={lat['cos']['mean']:.4f} "
        f"cos_min={lat['cos']['min']:.4f} first_cos={lat['first_cos']:.4f} "
        f"last_close_cos={lat['last_close_target_cos']:.4f}")
    print(
        f"  latent: mse_mean={lat['mse']['mean']:.5f} nll_mean={lat['nll']['mean']:.4f} "
        f"first_nll={lat['first_nll']:.4f} last_close_nll={lat['last_close_target_nll']:.4f}")
    print(
        f"  input_rms: h={stats['h']['rms']:.4f} q={stats['q']['rms']:.4f} "
        f"h*q={stats['h_times_q']['rms']:.4f}")
    print(
        f"  gold_first_reason: rank={first['rank']} logprob={first['logprob']:.4f} "
        f"top1={first['top'][0]['id']} {first['top'][0]['text']!r}")
    print(
        f"  gold_eol: rank={eol['rank']} logprob={eol['logprob']:.4f} "
        f"top1={eol['top'][0]['id']} {eol['top'][0]['text']!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Adapter checkpoint dir, run dir, or root to search.")
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=None, help="Optional JSONL diagnostics output.")
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN))
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--attn-impl", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--r", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


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
    meta = _load_colar_meta(checkpoint)
    model = _canonicalize_path(args.model or ckpt_args.get("model") or str(DEFAULT_MODEL))
    plugin = _canonicalize_path(args.plugin or str(DEFAULT_PLUGIN))
    dataset = _canonicalize_path(args.dataset)
    if model is None or not model.exists():
        raise FileNotFoundError(f"model not found: {model}")
    if plugin is None or not plugin.exists():
        raise FileNotFoundError(f"plugin not found: {plugin}")
    if dataset is None or not dataset.exists():
        raise FileNotFoundError(f"dataset not found: {dataset}")

    torch_dtype = parse_torch_dtype(args.torch_dtype or ckpt_args.get("torch_dtype"))
    model_type = args.model_type or ckpt_args.get("model_type") or "qwen3_omni_moe"
    attn_impl = args.attn_impl if args.attn_impl != "auto" else ckpt_args.get("attn_impl")
    device_map = None if args.device_map == "none" else args.device_map
    r = int(args.r if args.r is not None else meta.get("fixed_r", 5))
    sqrt_mean = bool(meta.get("sqrt_mean", False))

    print(f"[qa-tf] checkpoint: {checkpoint}")
    print(f"[qa-tf] dataset: {dataset}")
    print(f"[qa-tf] r={r} sqrt_mean={sqrt_mean}")
    print(f"[qa-tf] model_type={model_type} torch_dtype={torch_dtype} attn_impl={attn_impl} device_map={device_map}")

    generator = build_query_anchor_latent_generator(
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
    if args.limit is not None:
        records = records[:args.limit]

    out_f = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")
    try:
        for idx, record in enumerate(records):
            row = diagnose_one(generator, record, idx=idx, r=r, sqrt_mean=sqrt_mean, top_k=args.top_k)
            print_report(row)
            if out_f is not None:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        if out_f is not None:
            out_f.close()
            print(f"[qa-tf] wrote: {args.output}")


if __name__ == "__main__":
    main()
