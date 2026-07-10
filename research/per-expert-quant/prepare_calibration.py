#!/usr/bin/env python3
"""Freeze deterministic, tokenized Hy3 routing-calibration requests.

Requires `datasets` and `transformers`. Dataset revisions and sampling parameters come from
calibration.lock.json; output requests contain prompt ids so both uniform control arms receive the
exact same tokens. Public evaluation datasets are never read here.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


SWE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "Custom editing tool for viewing, creating and editing files.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    },
                    "path": {"type": "string"},
                    "file_text": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "insert_line": {"type": "integer"},
                    "view_range": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["command", "path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "_state_anthropic",
            "description": "Internal helper to manage persistent editor state across tool calls.",
            "strict": True,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submits the current file.",
            "strict": True,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Runs the given command directly in bash.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def maybe_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def normalize_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, sort_keys=True))
        return "\n".join(part for part in parts if part)
    return str(value)


def deterministic_tool_calls(calls: Any, source_id: str) -> list[dict[str, Any]]:
    out = []
    for i, call in enumerate(maybe_json(calls) or []):
        fn = call.get("function", call)
        out.append(
            {
                "type": "function",
                "id": f"calib-{source_id[:12]}-{i}",
                "function": {
                    "name": fn["name"],
                    "arguments": maybe_json(fn.get("arguments", {})),
                },
            }
        )
    return out


def map_row(stratum: str, row: dict[str, Any]) -> tuple[list[dict[str, Any]], Any, str]:
    row_hash = hashlib.sha256(canonical(row)).hexdigest()
    if stratum == "general_code":
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ]
        return messages, None, row_hash
    if stratum == "single_turn_tools":
        sid = str(row.get("id", row_hash))
        messages = [
            {"role": "user", "content": row["query"]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": deterministic_tool_calls(row["answers"], sid),
            },
        ]
        return messages, maybe_json(row["tools"]), sid
    if stratum.startswith("reasoning_"):
        return row["messages"], None, f"{row.get('source', 'unknown')}:{row_hash}"
    if stratum == "agentic_code_tools":
        sid = str(row.get("traj_id", row_hash))
        messages = []
        for message in maybe_json(row["messages"]):
            mapped = {"role": message["role"], "content": normalize_content(message.get("content"))}
            if message.get("tool_calls"):
                mapped["tool_calls"] = deterministic_tool_calls(message["tool_calls"], sid)
            messages.append(mapped)
        return messages, SWE_TOOLS, sid
    raise ValueError(f"unknown calibration stratum {stratum}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lock", type=Path, default=Path(__file__).with_name("calibration.lock.json"))
    p.add_argument("--tokenizer", required=True, help="pinned Hy3 source/tokenizer directory")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    lock = json.loads(args.lock.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    per = int(lock["samples_per_stratum"])
    seed = int(lock["seed"])
    max_tokens = int(lock["max_input_tokens"])
    buffer_size = int(lock["shuffle_buffer"])
    requests = []
    sample_manifest = []

    for source_index, source in enumerate(lock["sources"]):
        kwargs = {
            "path": source["name"],
            "split": source["split"],
            "revision": source["revision"],
            "streaming": True,
        }
        if source["config"] is not None:
            kwargs["name"] = source["config"]
        if args.cache_dir is not None:
            kwargs["cache_dir"] = str(args.cache_dir)
        dataset = load_dataset(**kwargs).shuffle(seed=seed + source_index, buffer_size=buffer_size)
        rows = list(itertools.islice(dataset, per))
        if len(rows) != per:
            raise RuntimeError(f"{source['stratum']}: expected {per} rows, got {len(rows)}")
        for sample_index, row in enumerate(rows):
            messages, tools, source_id = map_row(source["stratum"], row)
            template_kwargs = {
                "conversation": messages,
                "tokenize": True,
                "add_generation_prompt": bool(lock["add_generation_prompt"]),
                "truncation": True,
                "max_length": max_tokens,
            }
            if tools is not None:
                template_kwargs["tools"] = tools
            prompt_ids = tokenizer.apply_chat_template(**template_kwargs)
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()
            prompt_ids = [int(token) for token in prompt_ids]
            if not prompt_ids or len(prompt_ids) > max_tokens:
                raise RuntimeError(f"invalid token count for {source_id}: {len(prompt_ids)}")
            content_sha = hashlib.sha256(canonical({"messages": messages, "tools": tools})).hexdigest()
            record = {
                "ordinal": len(requests),
                "stratum": source["stratum"],
                "source_id": source_id,
                "content_sha256": content_sha,
                "prompt_tokens": len(prompt_ids),
                "prompt_ids": prompt_ids,
            }
            requests.append(record)
            sample_manifest.append({k: record[k] for k in record if k != "prompt_ids"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    requests_path = args.out_dir / "requests.jsonl"
    body = b"".join(canonical(record) + b"\n" for record in requests)
    requests_path.write_bytes(body)
    manifest = {
        "format": "bw24-hy3-routing-calibration-corpus-v1",
        "recipe": lock,
        "tokenizer_dir": str(Path(args.tokenizer).resolve()),
        "request_count": len(requests),
        "total_prompt_tokens": sum(record["prompt_tokens"] for record in requests),
        "requests_sha256": hashlib.sha256(body).hexdigest(),
        "samples": sample_manifest,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {len(requests)} requests / {manifest['total_prompt_tokens']} tokens "
        f"sha256={manifest['requests_sha256']}"
    )


if __name__ == "__main__":
    main()
