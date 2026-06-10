#!/usr/bin/env python
"""Run CoLaR latent inference for the Qwen3-Omni overfit checkpoint.

Default mode uses the original COLAR `latent_generate` loop:
  prompt -> latent_policy compressed reasoning -> answer generation

Use `--mode text` only when you explicitly want the legacy ms-swift text generate
baseline (no latent head).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    "/apdcephfs_tj6/share_303840540/hunyuan/jensenwang/"
    "model_warehouse/Qwen3-Omni-30B-A3B-Instruct"
)
DEFAULT_CKPT_ROOT = REPO_ROOT / "output/qwen3omni_colar/overfit_ckpt_r2"
DEFAULT_DATASET = REPO_ROOT / "output/qwen3omni_colar/overfit_text.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "output/qwen3omni_colar/overfit_infer_r2.jsonl"
DEFAULT_PLUGIN = REPO_ROOT / "colar_plugin/plugin.py"


def _canonicalize_path(path: Optional[str]) -> Optional[Path]:
    if path in (None, ""):
        return None
    p = Path(os.path.expanduser(path))
    if p.exists():
        return p

    text = str(p)
    legacy_prefix = "/apdcephfs/tj6/"
    mounted_prefix = "/apdcephfs/tj/apdcephfs_tj6/"
    if text.startswith(legacy_prefix):
        alt = Path(mounted_prefix + text[len(legacy_prefix):])
        if alt.exists():
            return alt
        direct = Path("/" + text[len(legacy_prefix):])
        if direct.exists():
            return direct
    return p


def _is_adapter_dir(path: Path) -> bool:
    return (
        (path / "adapter_config.json").exists()
        or (path / "default/adapter_config.json").exists()
        or (path / "reft").exists()
    )


def _checkpoint_step(path: Path) -> int:
    match = re.match(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(root: Path) -> Path:
    root = _canonicalize_path(str(root))
    assert root is not None
    if _is_adapter_dir(root):
        return root

    candidates = [p for p in root.rglob("checkpoint-*") if p.is_dir() and _is_adapter_dir(p)]
    if not candidates:
        raise FileNotFoundError(
            f"No adapter checkpoint found under {root}. "
            "Wait until training reaches save_steps, or pass --checkpoint explicitly."
        )
    return max(candidates, key=lambda p: (p.stat().st_mtime, _checkpoint_step(p), str(p)))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_args(checkpoint: Path) -> Dict[str, Any]:
    args_path = checkpoint / "args.json"
    if args_path.exists():
        return load_json(args_path)
    parent_args = checkpoint.parent / "args.json"
    if parent_args.exists():
        return load_json(parent_args)
    return {}


def parse_torch_dtype(value: Optional[str]):
    if value in (None, "", "auto"):
        return None
    value = str(value).lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported torch dtype from args.json: {value}")
    return mapping[value]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e


def split_prompt_and_expected(record: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str], List[str]]:
    messages = record.get("messages") or []
    prompt_messages: List[Dict[str, Any]] = []
    expected = None
    for message in messages:
        if message.get("role") == "assistant":
            expected = message.get("content")
            break
        prompt_messages.append(dict(message))
    if not prompt_messages:
        prompt_messages = [dict(message) for message in messages]
    audios = record.get("audios") or []
    if isinstance(audios, str):
        audios = [audios]
    if not audios:
        audios = [m.get("wav_path", "").strip() for m in prompt_messages if m.get("wav_path")]
        audios = [p for p in audios if p]
        if audios:
            for message in prompt_messages:
                message.pop("wav_path", None)
            user_messages = [m for m in prompt_messages if m.get("role") == "user"]
            if len(user_messages) == 1 and "<audio>" not in str(user_messages[0].get("content", "")):
                user_messages[0]["content"] = "<audio>"
    return prompt_messages, expected, audios


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def strip_think(text: Optional[str]) -> str:
    text = text or ""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def match_flags(expected: Optional[str], response: str) -> Dict[str, bool]:
    expected_norm = normalize_text(expected)
    response_norm = normalize_text(response)
    expected_answer_norm = normalize_text(strip_think(expected))
    response_answer_norm = normalize_text(strip_think(response))
    return {
        "full_expected_in_response": bool(expected_norm and expected_norm in response_norm),
        "response_in_full_expected": bool(response_norm and response_norm in expected_norm),
        "answer_expected_in_response": bool(expected_answer_norm and expected_answer_norm in response_answer_norm),
        "answer_response_in_expected": bool(response_answer_norm and response_answer_norm in expected_answer_norm),
    }


def import_plugin(plugin: Path) -> None:
    plugin = _canonicalize_path(str(plugin))
    if plugin is None or not plugin.exists():
        raise FileNotFoundError(f"CoLaR plugin not found: {plugin}")
    from swift.utils import import_external_file

    import_external_file(str(plugin))


def build_text_engine(args, checkpoint: Path, ckpt_args: Dict[str, Any]):
    from swift.infer_engine import TransformersEngine

    model = _canonicalize_path(args.model or ckpt_args.get("model") or DEFAULT_MODEL)
    if model is None or not model.exists():
        raise FileNotFoundError(f"Base model not found: {model}")

    torch_dtype = parse_torch_dtype(args.torch_dtype or ckpt_args.get("torch_dtype"))
    model_type = args.model_type or ckpt_args.get("model_type") or "qwen3_omni_moe"
    attn_impl = args.attn_impl if args.attn_impl != "auto" else ckpt_args.get("attn_impl")
    device_map = None if args.device_map == "none" else args.device_map

    print(f"[infer:text] model: {model}")
    print(f"[infer:text] checkpoint: {checkpoint}")
    print(
        f"[infer:text] model_type={model_type} torch_dtype={torch_dtype} "
        f"attn_impl={attn_impl} device_map={device_map}"
    )

    return TransformersEngine(
        str(model),
        adapters=[str(checkpoint)],
        max_batch_size=args.batch_size,
        torch_dtype=torch_dtype,
        model_type=model_type,
        attn_impl=attn_impl,
        device_map=device_map,
    )


def build_latent_generator(args, checkpoint: Path, ckpt_args: Dict[str, Any], plugin: Path):
    from colar_plugin.colar_infer import build_colar_latent_generator

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

    print(f"[infer:latent] model: {model}")
    print(f"[infer:latent] checkpoint: {checkpoint}")
    print(
        f"[infer:latent] model_type={model_type} torch_dtype={torch_dtype} "
        f"attn_impl={attn_impl} device_map={device_map}"
    )
    print(
        f"[infer:latent] max_latent_forward={args.max_latent_forward} "
        f"latent_temperature={args.latent_temperature} eol_temperature={args.eol_temperature} "
        f"max_new_tokens={args.max_new_tokens} latent_rms_target={args.latent_rms_target}"
    )

    return build_colar_latent_generator(
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


def run_text_inference(args, checkpoint, ckpt_args, records, prompts, expected, audios_list, output):
    from swift.infer_engine import InferRequest, RequestConfig

    engine = build_text_engine(args, checkpoint, ckpt_args)
    requests = []
    for prompt, audios in zip(prompts, audios_list):
        requests.append(
            InferRequest(
                messages=prompt,
                audios=audios,
                chat_template_kwargs={"enable_thinking": not args.disable_thinking},
            ))
    request_config = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        return_details=args.return_details,
    )
    if args.top_p is not None:
        request_config.top_p = args.top_p
    if args.top_k is not None:
        request_config.top_k = args.top_k
    if args.repetition_penalty is not None:
        request_config.repetition_penalty = args.repetition_penalty
    responses = engine.infer(requests, request_config=request_config, use_tqdm=True)

    n_answer_match = 0
    with output.open("w", encoding="utf-8") as f:
        for idx, (prompt, exp, audios, resp) in enumerate(zip(prompts, expected, audios_list, responses)):
            if isinstance(resp, Exception):
                response_text = ""
                error = repr(resp)
                extra = {}
            else:
                response_text = resp.choices[0].message.content
                error = None
                extra = {}
                if args.return_details:
                    extra["prompt_token_ids"] = resp.prompt_token_ids
                    extra["completion_token_ids"] = resp.choices[0].token_ids
            flags = match_flags(exp, response_text)
            n_answer_match += int(flags["answer_expected_in_response"] or flags["answer_response_in_expected"])
            row = {
                "idx": idx,
                "mode": "text",
                "prompt_messages": prompt,
                "audios": audios,
                "expected": exp,
                "response": response_text,
                "match": flags,
                "error": error,
                **extra,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[infer:text] wrote: {output}")
    print(f"[infer:text] answer-ish matches: {n_answer_match}/{len(records)}")
    for i, resp in enumerate(responses[: min(2, len(responses))]):
        if isinstance(resp, Exception):
            print(f"[infer:text] sample {i} error: {resp!r}")
            continue
        print(f"[infer:text] sample {i} response head: {resp.choices[0].message.content[:500]!r}")


def run_latent_inference(args, checkpoint, ckpt_args, plugin, records, prompts, expected, audios_list, output):
    generator = build_latent_generator(args, checkpoint, ckpt_args, plugin)

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
                print(f"[infer:latent] sample {idx} error: {error}", file=sys.stderr)
            flags = match_flags(exp, response_text)
            n_answer_match += int(flags["answer_expected_in_response"] or flags["answer_response_in_expected"])
            row = {
                "idx": idx,
                "mode": "latent",
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
                f"[infer:latent] sample {idx}: n_latent_forward={n_latent_forward} "
                f"hit_eol={hit_eol} answer_match={flags['answer_expected_in_response']}"
            )
            if response_text:
                print(f"[infer:latent] sample {idx} response head: {response_text[:500]!r}")

    print(f"[infer:latent] wrote: {output}")
    print(f"[infer:latent] answer-ish matches: {n_answer_match}/{len(records)}")
    print(f"[infer:latent] eol hits: {n_eol_hit}/{len(records)}")
    if errors:
        details = "; ".join(f"sample {idx}: {error}" for idx, error in errors[:3])
        raise RuntimeError(f"latent inference failed on {len(errors)}/{len(records)} samples: {details}")


def run_inference(args) -> None:
    checkpoint_arg = _canonicalize_path(args.checkpoint) if args.checkpoint else None
    if checkpoint_arg is None:
        checkpoint = find_latest_checkpoint(Path(args.checkpoint_root))
    elif _is_adapter_dir(checkpoint_arg):
        checkpoint = checkpoint_arg
    else:
        checkpoint = find_latest_checkpoint(checkpoint_arg)

    ckpt_args = load_checkpoint_args(checkpoint)
    plugin = _canonicalize_path(args.plugin or ckpt_args.get("external_plugins", [None])[0] or str(DEFAULT_PLUGIN))
    import_plugin(plugin)

    latent_policy = checkpoint / "latent_policy.pt"
    if not latent_policy.exists():
        msg = f"[infer] latent policy not found in checkpoint: {latent_policy}"
        if args.require_latent_policy or args.mode == "latent":
            raise FileNotFoundError(msg)
        print(msg)
    else:
        print(f"[infer] latent policy found: {latent_policy}")

    colar_config = checkpoint / "colar_config.json"
    if colar_config.exists():
        print(f"[infer] colar config found: {colar_config}")

    dataset = _canonicalize_path(args.dataset)
    if dataset is None or not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    records = list(iter_jsonl(dataset))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"No records to infer from: {dataset}")

    prompts = []
    expected = []
    audios_list = []
    for record in records:
        prompt, exp, audios = split_prompt_and_expected(record)
        prompts.append(prompt)
        expected.append(exp)
        audios_list.append(audios)

    output = _canonicalize_path(args.output) or Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "text":
        run_text_inference(args, checkpoint, ckpt_args, records, prompts, expected, audios_list, output)
    else:
        run_latent_inference(args, checkpoint, ckpt_args, plugin, records, prompts, expected, audios_list, output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("latent", "text"),
        default="latent",
        help="latent=COLAR latent_generate (default); text=legacy ms-swift generate baseline.",
    )
    parser.add_argument("--checkpoint", default=None, help="Adapter checkpoint dir, run dir, or root to search.")
    parser.add_argument(
        "--checkpoint-root",
        default=str(DEFAULT_CKPT_ROOT),
        help="Root searched when checkpoint is omitted.",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Overfit JSONL dataset.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL path.")
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN), help="Path to colar_plugin/plugin.py.")
    parser.add_argument("--model", default=None, help="Base model path. Defaults to checkpoint args.json or local default.")
    parser.add_argument("--model-type", default=None, help="Override model_type.")
    parser.add_argument("--torch-dtype", default=None, help="Override torch dtype, e.g. bfloat16.")
    parser.add_argument("--attn-impl", default="auto", help="Override attention implementation. Use 'auto' for checkpoint value.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map. Use 'none' to leave unset.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-latent-forward", type=int, default=2048, help="Max latent reasoning steps.")
    parser.add_argument("--latent-temperature", type=float, default=0.0, help="LatentPolicy sampling temperature.")
    parser.add_argument(
        "--latent-rms-target", type=float, default=0.0,
        help=">0: renormalize each rollout latent to this RMS (normalized space) before feeding back. 0=off.")
    parser.add_argument("--eol-temperature", type=float, default=0.0, help="Temperature for </think> EOL check.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Answer generation temperature.")
    parser.add_argument("--progress-every", type=int, default=0, help="Print answer decode progress every N tokens.")
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--disable-thinking", action="store_true", help="Text mode only: skip thinking prefix.")
    parser.add_argument("--require-latent-policy", action="store_true", help="Fail if latent_policy.pt is absent.")
    parser.add_argument("--return-details", action="store_true", help="Text mode only: save prompt/completion token ids.")
    return parser.parse_args()


def main():
    try:
        run_inference(parse_args())
    except Exception as e:
        print(f"[infer][error] {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
