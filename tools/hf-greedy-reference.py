#!/usr/bin/env python3
"""Generate deterministic Hugging Face token ids for a memra argmax gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--tokens-out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer
    except ImportError as exc:
        print(
            "missing HF reference dependencies; install current torch, transformers, "
            f"and accelerate in an isolated venv: {exc}",
            file=sys.stderr,
        )
        return 2

    prompt = args.prompt_file.read_text()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=not args.no_thinking,
    )
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_tokens = encoded["input_ids"][0].tolist()

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    input_device = next(model.parameters()).device
    encoded = {name: value.to(input_device) for name, value in encoded.items()}

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    prompt_length = encoded["input_ids"].shape[1]
    generated = output[0, prompt_length:].detach().cpu().tolist()
    record = {
        "model": str(args.model.resolve()),
        "prompt_file": str(args.prompt_file.resolve()),
        "thinking": not args.no_thinking,
        "prompt_tokens": prompt_length,
        "input_tokens": input_tokens,
        "tokens": generated,
    }
    args.tokens_out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"prompt tokens: {prompt_length}")
    print(f"tokens: {generated}")
    print(f"wrote: {args.tokens_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
