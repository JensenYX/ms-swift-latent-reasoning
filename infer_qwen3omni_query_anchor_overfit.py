#!/usr/bin/env python
"""Run query-anchored CoLaR latent inference for Qwen3-Omni overfit checkpoints."""

import argparse
import json
import sys
from pathlib import Path

from infer_qwen3omni_colar_overfit import (
    DEFAULT_MODEL,
    _canonicalize_path,
    _is_adapter_dir,
    find_latest_checkpoint,
    iter_jsonl,
    load_checkpoint_args,
    match_flags,
    parse_torch_dtype,
    split_prompt_and_expected,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT_ROOT = REPO_ROOT / "output/qwen3omni_query_anchor_mixed_r10_mse_sqrt/overfit_ckpt_audio_r10"
DEFAULT_DATASET = REPO_ROOT / "output/qwen3omni_colar/overfit_mixed_audio_text.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "output/qwen3omni_query_anchor_mixed_r10_mse_sqrt/overfit_infer_query_anchor_r10.jsonl"
DEFAULT_PLUGIN = REPO_ROOT / "colar_plugin/query_anchor_plugin.py"


def build_query_anchor_generator(args, checkpoint: Path, ckpt_args, plugin: Path):
    from colar_plugin.query_anchor_infer import build_query_anchor_latent_generator

    model = _canonicalize_path(args.model or ckpt_args.get("model") or DEFAULT_MODEL)
    if model is None or not model.exists():
        raise FileNotFoundError(f"Base model not found: {model}")

    torch_dtype = parse_torch_dtype(args.torch_dtype or ckpt_args.get("torch_dtype"))
    model_type = args.model_type or ckpt_args.get("model_type") or "qwen3_omni_moe"
    attn_impl = args.attn_impl if args.attn_impl != "auto" else ckpt_args.get("attn_impl")
    device_map = None if args.device_map == "none" else args.device_map

    infer_overrides = {
        "max_n_latent_forward": args.max_latent_forward,
        "latent_temperature": args.latent_temperature,
        "eol_temperature": args.eol_temperature,
        "max_new_tokens": args.max_new_tokens,
        "answer_temperature": args.temperature,
        "progress_every": args.progress_every,
        "latent_rms_target": args.latent_rms_target,
    }

    print(f"[infer:query-anchor] model: {model}")
    print(f"[infer:query-anchor] checkpoint: {checkpoint}")
    print(
        f"[infer:query-anchor] model_type={model_type} torch_dtype={torch_dtype} "
        f"attn_impl={attn_impl} device_map={device_map}"
    )
    print(
        f"[infer:query-anchor] max_latent_forward={args.max_latent_forward} "
        f"latent_temperature={args.latent_temperature} eol_temperature={args.eol_temperature} "
        f"max_new_tokens={args.max_new_tokens} latent_rms_target={args.latent_rms_target}"
    )

    return build_query_anchor_latent_generator(
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


def run_inference(args) -> None:
    checkpoint_arg = _canonicalize_path(args.checkpoint) if args.checkpoint else None
    if checkpoint_arg is None:
        checkpoint = find_latest_checkpoint(Path(args.checkpoint_root))
    elif _is_adapter_dir(checkpoint_arg):
        checkpoint = checkpoint_arg
    else:
        checkpoint = find_latest_checkpoint(checkpoint_arg)

    ckpt_args = load_checkpoint_args(checkpoint)
    plugin = _canonicalize_path(args.plugin or str(DEFAULT_PLUGIN))
    if plugin is None or not plugin.exists():
        raise FileNotFoundError(f"Query-anchor plugin not found: {plugin}")

    latent_policy = checkpoint / "latent_policy.pt"
    if not latent_policy.exists():
        raise FileNotFoundError(f"[infer:query-anchor] latent policy not found: {latent_policy}")
    print(f"[infer:query-anchor] latent policy found: {latent_policy}")

    colar_config = checkpoint / "colar_config.json"
    if colar_config.exists():
        print(f"[infer:query-anchor] colar config found: {colar_config}")

    dataset = _canonicalize_path(args.dataset)
    if dataset is None or not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    records = list(iter_jsonl(dataset))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"No records to infer from: {dataset}")

    prompts, expected, audios_list = [], [], []
    for record in records:
        prompt, exp, audios = split_prompt_and_expected(record)
        prompts.append(prompt)
        expected.append(exp)
        audios_list.append(audios)

    output = _canonicalize_path(args.output) or Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    generator = build_query_anchor_generator(args, checkpoint, ckpt_args, plugin)

    n_answer_match = 0
    n_eol_hit = 0
    errors = []
    with output.open("w", encoding="utf-8") as f:
        for idx, (prompt, exp, audios) in enumerate(zip(prompts, expected, audios_list)):
            error = None
            response_text = ""
            n_latent_forward = 0
            hit_eol = False
            try:
                response_text, n_latent_forward, hit_eol = generator.latent_generate(prompt, audios=audios)
                n_eol_hit += int(hit_eol)
            except Exception as e:
                error = repr(e)
                errors.append((idx, error))
                print(f"[infer:query-anchor] sample {idx} error: {error}", file=sys.stderr)
            flags = match_flags(exp, response_text)
            n_answer_match += int(flags["answer_expected_in_response"] or flags["answer_response_in_expected"])
            row = {
                "idx": idx,
                "mode": "query_anchor_latent",
                "prompt_messages": prompt,
                "audios": audios,
                "expected": exp,
                "response": response_text,
                "n_latent_forward": n_latent_forward,
                "hit_eol": hit_eol,
                "match": flags,
                "error": error,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[infer:query-anchor] sample {idx}: n_latent_forward={n_latent_forward} "
                f"hit_eol={hit_eol} answer_match={flags['answer_expected_in_response']}"
            )
            if response_text:
                print(f"[infer:query-anchor] sample {idx} response head: {response_text[:500]!r}")

    print(f"[infer:query-anchor] wrote: {output}")
    print(f"[infer:query-anchor] answer-ish matches: {n_answer_match}/{len(records)}")
    print(f"[infer:query-anchor] eol hits: {n_eol_hit}/{len(records)}")
    if errors:
        details = "; ".join(f"sample {idx}: {error}" for idx, error in errors[:3])
        raise RuntimeError(f"query-anchor latent inference failed on {len(errors)}/{len(records)} samples: {details}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Adapter checkpoint dir, run dir, or root to search.")
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN))
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--attn-impl", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-latent-forward", type=int, default=2048)
    parser.add_argument("--latent-temperature", type=float, default=0.0)
    parser.add_argument(
        "--latent-rms-target", type=float, default=0.0,
        help=">0: renormalize each rollout latent to this RMS (normalized space) before anchoring/feeding back. 0=off.")
    parser.add_argument("--eol-temperature", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=0)
    return parser.parse_args()


def main():
    try:
        run_inference(parse_args())
    except Exception as e:
        print(f"[infer:query-anchor][error] {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
