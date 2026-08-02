#!/usr/bin/env python3
"""OpenAI streaming tool_calls schema checker (serve-tools gate b).

Reads a raw SSE capture (the exact bytes memra-server sent) and validates every chunk
against the OpenAI chat.completion.chunk shape, with tool-call specifics:

  - every `data:` payload before [DONE] is JSON with object=="chat.completion.chunk",
    choices[0].index==0, and delta keys within {role, content, tool_calls};
  - tool_calls delta entries carry an integer `index`; the FIRST chunk of a call carries
    id (string, non-empty) + type=="function" + function.name; continuation chunks carry
    function.arguments fragments; per-index accumulated arguments parse as a JSON object;
  - exactly one terminal chunk with finish_reason in {"stop","length","tool_calls"} and a
    usage block carrying integer prompt_tokens/completion_tokens/total_tokens
    (total == prompt + completion);
  - stream ends with `data: [DONE]`.

Usage: check_stream_schema.py <capture.sse> [--expect-tool-calls]
Exit 0 = PASS; prints a JSON verdict row either way.
"""

import json
import sys


def main():
    path = sys.argv[1]
    expect_calls = "--expect-tool-calls" in sys.argv[2:]
    raw = open(path, "rb").read().decode()
    payloads = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            payloads.append(line[len("data:"):].strip())
    problems = []
    if not payloads:
        problems.append("no data: lines")
    if payloads and payloads[-1] != "[DONE]":
        problems.append("stream does not end with [DONE]")
    calls = {}          # index -> {id, name, arguments}
    call_order = []
    content = ""
    finish = None
    usage = None
    for k, p in enumerate(payloads[:-1] if payloads and payloads[-1] == "[DONE]" else payloads):
        try:
            d = json.loads(p)
        except json.JSONDecodeError as e:
            problems.append(f"chunk {k}: not JSON ({e})")
            continue
        if d.get("object") != "chat.completion.chunk":
            problems.append(f"chunk {k}: object={d.get('object')!r}")
        ch = (d.get("choices") or [{}])[0]
        if ch.get("index") != 0:
            problems.append(f"chunk {k}: choice index {ch.get('index')!r}")
        delta = ch.get("delta", {})
        extra = set(delta.keys()) - {"role", "content", "tool_calls"}
        if extra:
            problems.append(f"chunk {k}: unexpected delta keys {sorted(extra)}")
        if isinstance(delta.get("content"), str):
            content += delta["content"]
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index")
            if not isinstance(idx, int):
                problems.append(f"chunk {k}: tool_calls entry without integer index")
                continue
            if idx not in calls:
                # first chunk of a call: id + type + function.name required
                if not (isinstance(tc.get("id"), str) and tc["id"]):
                    problems.append(f"chunk {k}: first tool_call chunk (index {idx}) missing id")
                if tc.get("type") != "function":
                    problems.append(f"chunk {k}: tool_call type={tc.get('type')!r}")
                fn = tc.get("function", {})
                if not fn.get("name"):
                    problems.append(f"chunk {k}: first tool_call chunk missing function.name")
                calls[idx] = {"id": tc.get("id"), "name": fn.get("name"),
                              "arguments": fn.get("arguments", "")}
                call_order.append(idx)
            else:
                fn = tc.get("function", {})
                calls[idx]["arguments"] += fn.get("arguments", "")
        if ch.get("finish_reason") is not None:
            if finish is not None:
                problems.append(f"chunk {k}: second finish_reason")
            finish = ch["finish_reason"]
            usage = d.get("usage")
    if finish not in ("stop", "length", "tool_calls"):
        problems.append(f"finish_reason={finish!r}")
    if not isinstance(usage, dict):
        problems.append("final chunk has no usage block")
    else:
        for f in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if not isinstance(usage.get(f), int):
                problems.append(f"usage.{f} missing or not int")
        if isinstance(usage.get("prompt_tokens"), int) \
                and isinstance(usage.get("completion_tokens"), int) \
                and usage.get("total_tokens") != usage["prompt_tokens"] + usage["completion_tokens"]:
            problems.append("usage.total_tokens != prompt + completion")
    for idx in call_order:
        try:
            parsed = json.loads(calls[idx]["arguments"])
            if not isinstance(parsed, dict):
                problems.append(f"call {idx}: arguments not a JSON object")
        except json.JSONDecodeError as e:
            problems.append(f"call {idx}: accumulated arguments not JSON ({e})")
    if expect_calls:
        if not calls:
            problems.append("expected tool_calls, saw none")
        if finish != "tool_calls":
            problems.append(f"expected finish_reason tool_calls, got {finish!r}")
    verdict = "PASS" if not problems else "FAIL"
    print(json.dumps({
        "file": path, "verdict": verdict, "n_chunks": max(len(payloads) - 1, 0),
        "n_calls": len(calls), "finish_reason": finish, "usage": usage,
        "calls": [calls[i] for i in call_order], "content_chars": len(content),
        "problems": problems,
    }))
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
