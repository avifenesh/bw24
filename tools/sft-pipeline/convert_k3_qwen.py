#!/usr/bin/env python3
"""convert_k3_qwen.py — K3/Moonshot trace JSONL -> Qwen3.6 ChatML training text.

Target format: the TOOLS branch of the qwen3.5/3.6-class ChatML template, byte-identical
to memra's serve surface (`crates/memra-tokenizer/src/chat.rs::apply_chat_template_tools`,
lane/sft-pipeline @ restructure/public-split). The laws ported here, exactly:

  - tools present  -> `<|im_start|>system\n# Tools\n\nYou have access to the following
    functions:\n\n<tools>` + `\n{tool json}` each (python-style JSON: `", "`/`": "`
    separators, insertion-order keys, non-ASCII raw — the serve `pyjson` law) +
    `\n</tools>` + the fixed instruction block; a leading system turn's trimmed content
    folds in after `\n\n`; `<|im_end|>\n`.
  - assistant turns with tool_calls -> trimmed content, then per call
    `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>...\n</function>\n</tool_call>`
    (`\n\n` before the first call when content non-empty, `\n` between calls), `<|im_end|>\n`.
    Argument value law (serve `render_req_tool_call`): strings raw, objects/arrays
    python-style JSON, other scalars their JSON spelling (`true`/`3`/`null`).
  - consecutive `tool` turns group into ONE user turn of
    `\n<tool_response>\n{content}\n</tool_response>` blocks.
  - training text renders the full conversation, NO generation prompt (the final
    assistant turn is the target). `--gen-prompt` adds `<|im_start|>assistant\n<think>\n`
    for serve-parity checks.

K3/Moonshot specifics handled:
  - `function.arguments` as a JSON object STRING (their API shape) or an object.
  - `reasoning_content` (K3 thinking): embedded as `<think>\n...\n</think>\n\n` in the
    FINAL assistant turn (qwen3.6 keeps thinking only on the last turn); on earlier
    turns it is moved to meta (`conversion.dropped_reasoning`) — never dropped silently.
  - Unknown message/top-level fields are preserved under `meta.conversion` — never
    dropped silently.

Output JSONL per input trace:
  {"task_id", "seed_source", "text": <ChatML string>, "outcome", "meta"}

`--roundtrip` parses each rendered text back into turns and verifies roles, contents,
and tool calls survive (exit non-zero on mismatch).
"""
import argparse
import json
import re
import sys

# --- byte-for-byte constants from crates/memra-tokenizer/src/chat.rs ---

QWEN_TOOLS_INSTRUCTION = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML "
    "tags\n- Required parameters MUST be specified\n- You may provide optional reasoning "
    "for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls\n</IMPORTANT>"
)

TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"

KNOWN_TOP_FIELDS = {"task_id", "seed_source", "tools", "messages", "outcome", "meta", "verify"}
KNOWN_MSG_FIELDS = {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"}
KNOWN_CALL_FIELDS = {"id", "type", "function", "index"}


def pyjson(v):
    """The serve surface's `pyjson` law: python-style JSON — `", "`/`": "` separators,
    insertion-order keys, non-ASCII raw, JSON string escaping."""
    return json.dumps(v, ensure_ascii=False, separators=(", ", ": "))


def render_arg_value(v):
    """The serve surface's `render_req_tool_call` value law: strings raw, mappings and
    sequences python-style JSON, other scalars their JSON spelling."""
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return pyjson(v)
    return json.dumps(v)  # true / 3 / 3.5 / null — JSON spelling


def k3_call_to_params(call, notes):
    """K3/Moonshot tool call -> (name, [(key, rendered_value)]). Unknown fields on the
    call object are recorded in `notes` (never dropped silently)."""
    fn = call.get("function") or {}
    name = fn.get("name", "")
    args = fn.get("arguments")
    if args is None or (isinstance(args, str) and not args.strip()):
        parsed = {}
    elif isinstance(args, str):
        parsed = json.loads(args)  # validator guarantees this parses to an object
    elif isinstance(args, dict):
        parsed = args
    else:
        raise ValueError(f"tool call {name!r}: arguments must be object or object-string")
    if not isinstance(parsed, dict):
        raise ValueError(f"tool call {name!r}: arguments must decode to a JSON object")
    unknown = {k: call[k] for k in call if k not in KNOWN_CALL_FIELDS}
    if unknown:
        notes.setdefault("unknown_call_fields", []).append({"call": name, "fields": unknown})
    params = [(k, render_arg_value(v)) for k, v in parsed.items()]
    return name, params


def render_qwen_chatml(turns, tools_json, add_generation_prompt=False):
    """Port of apply_chat_template_tools (qwen tools branch), add_generation_prompt as
    given, ThinkMode::Default. `turns` = list of dicts {role, content, tool_calls:[(name,
    [(k,v)])]}. Returns the rendered string."""
    out = []
    skip_leading_system = False
    if tools_json:
        out.append("<|im_start|>system\n")
        out.append(TOOLS_HEADER)
        for tool in tools_json:
            out.append("\n")
            out.append(tool)
        out.append("\n</tools>")
        out.append(QWEN_TOOLS_INSTRUCTION)
        if turns and turns[0]["role"] == "system":
            skip_leading_system = True
            content = turns[0]["content"].strip()
            if content:
                out.append("\n\n")
                out.append(content)
        out.append("<|im_end|>\n")

    n = len(turns)
    for i, turn in enumerate(turns):
        if i == 0 and skip_leading_system:
            continue
        content = turn["content"].strip()
        role = turn["role"]
        if role in ("system", "user"):
            out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        elif role == "assistant":
            out.append("<|im_start|>assistant\n")
            out.append(content)
            for k, (name, params) in enumerate(turn.get("tool_calls", [])):
                if k == 0:
                    if content:
                        out.append("\n\n")
                else:
                    out.append("\n")
                out.append(f"<tool_call>\n<function={name}>\n")
                for key, value in params:
                    out.append(f"<parameter={key}>\n{value}\n</parameter>\n")
                out.append("</function>\n</tool_call>")
            out.append("<|im_end|>\n")
        elif role == "tool":
            if i == 0 or turns[i - 1]["role"] != "tool":
                out.append("<|im_start|>user")
            out.append(f"\n<tool_response>\n{content}\n</tool_response>")
            if i + 1 >= n or turns[i + 1]["role"] != "tool":
                out.append("<|im_end|>\n")
        else:  # parity with the legacy renderer's generic-turn arm
            out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    if add_generation_prompt:
        out.append("<|im_start|>assistant\n<think>\n")
    return "".join(out)


def convert_trace(trace, embed_think=True, add_generation_prompt=False):
    """One trace object -> (converted record, template turns). Raises ValueError on
    malformed input (run validate_trace.py first)."""
    notes = {"template": "qwen3.6-chatml-tools",
             "renderer": "crates/memra-tokenizer/src/chat.rs::apply_chat_template_tools"}
    unknown_top = {k: trace[k] for k in trace if k not in KNOWN_TOP_FIELDS}
    if unknown_top:
        notes["unknown_fields"] = unknown_top

    tools_json = [pyjson(t) for t in trace.get("tools") or []]
    messages = trace["messages"]
    last_assistant = max((i for i, m in enumerate(messages)
                          if m.get("role") == "assistant"), default=-1)

    turns = []
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg.get("content") or ""
        unknown = {k: msg[k] for k in msg if k not in KNOWN_MSG_FIELDS}
        if unknown:
            notes.setdefault("unknown_message_fields", []).append(
                {"index": i, "fields": unknown})
        reasoning = msg.get("reasoning_content")
        if role == "assistant" and reasoning:
            if i == last_assistant and embed_think:
                # qwen3.6 keeps thinking only on the final turn
                content = f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"
                notes["think"] = "embedded_final_turn"
            else:
                notes.setdefault("dropped_reasoning", []).append(
                    {"index": i, "reasoning_content": reasoning})
        calls = []
        for call in msg.get("tool_calls") or []:
            calls.append(k3_call_to_params(call, notes))
        turns.append({"role": role, "content": content, "tool_calls": calls})

    text = render_qwen_chatml(turns, tools_json, add_generation_prompt)
    meta = dict(trace.get("meta") or {})
    meta["conversion"] = notes
    record = {
        "task_id": trace["task_id"],
        "seed_source": trace["seed_source"],
        "text": text,
        "outcome": trace.get("outcome"),
        "meta": meta,
    }
    return record, turns


# --- round-trip: parse rendered ChatML back into turns ---

_CALL_RE = re.compile(
    r"<tool_call>\n<function=(?P<name>[^>]*)>\n(?P<params>.*?)</function>\n</tool_call>",
    re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=(?P<key>[^>]*)>\n(?P<val>.*?)\n</parameter>\n", re.DOTALL)
_RESP_RE = re.compile(r"<tool_response>\n(?P<body>.*?)\n</tool_response>", re.DOTALL)


def parse_qwen_chatml(text):
    """Recover turns (+ tools_json) from a rendered string. Inverse of
    render_qwen_chatml for the shapes this pipeline emits."""
    turns, tools_json = [], []
    blocks = text.split("<|im_start|>")
    for block in blocks:
        if not block:
            continue
        if not block.endswith("<|im_end|>\n"):
            raise ValueError(f"unterminated block: {block[:60]!r}")
        block = block[: -len("<|im_end|>\n")]
        role, _, body = block.partition("\n")
        if role == "system" and body.startswith(TOOLS_HEADER):
            rest = body[len(TOOLS_HEADER):]
            tools_part, sep, tail = rest.partition("\n</tools>")
            if not sep:
                raise ValueError("tools header without </tools>")
            tools_json = [ln for ln in tools_part.split("\n") if ln]
            if not tail.startswith(QWEN_TOOLS_INSTRUCTION):
                raise ValueError("tools header without the fixed instruction block")
            folded = tail[len(QWEN_TOOLS_INSTRUCTION):]
            if folded.startswith("\n\n"):
                turns.append({"role": "system", "content": folded[2:], "tool_calls": []})
            continue
        if role == "user" and body.startswith("<tool_response>\n"):
            for m in _RESP_RE.finditer(body):
                turns.append({"role": "tool", "content": m.group("body"), "tool_calls": []})
            continue
        if role == "assistant":
            calls = []
            first = _CALL_RE.search(body)
            content = body[: first.start()] if first else body
            for m in _CALL_RE.finditer(body):
                params = [(p.group("key"), p.group("val"))
                          for p in _PARAM_RE.finditer(m.group("params"))]
                calls.append((m.group("name"), params))
            if calls:
                content = content[:-2] if content.endswith("\n\n") else content
            turns.append({"role": "assistant", "content": content, "tool_calls": calls})
            continue
        turns.append({"role": role, "content": body, "tool_calls": []})
    return turns, tools_json


def roundtrip_check(turns, tools_json, text):
    """Render -> parse -> compare. Returns a list of mismatch strings (empty = pass)."""
    errs = []
    back_turns, back_tools = parse_qwen_chatml(text)
    if back_tools != tools_json:
        errs.append(f"tools_json mismatch: {back_tools!r} != {tools_json!r}")
    want = [{"role": t["role"], "content": t["content"].strip(),
             "tool_calls": [tuple(c) if isinstance(c, tuple) else c
                            for c in t["tool_calls"]]} for t in turns]
    got = [{"role": t["role"], "content": t["content"].strip(),
            "tool_calls": t["tool_calls"]} for t in back_turns]
    if len(want) != len(got):
        errs.append(f"turn count mismatch: rendered {len(want)}, parsed back {len(got)}")
        return errs
    for i, (w, g) in enumerate(zip(want, got)):
        if w["role"] != g["role"]:
            errs.append(f"turn {i}: role {g['role']!r} != {w['role']!r}")
        if w["content"] != g["content"]:
            errs.append(f"turn {i}: content mismatch: {g['content']!r} != {w['content']!r}")
        wc = [(n, list(p)) for n, p in w["tool_calls"]]
        gc = [(n, list(p)) for n, p in g["tool_calls"]]
        if wc != gc:
            errs.append(f"turn {i}: tool_calls mismatch: {gc!r} != {wc!r}")
    return errs


def selftest():
    """Golden byte-parity vector: the exact expected string of the Rust test
    `tools_header_and_tool_response_render_per_template_law` in
    crates/memra-tokenizer/src/chat.rs. Any drift from the serve renderer fails here."""
    turns = [
        {"role": "system", "content": "Be terse.", "tool_calls": []},
        {"role": "user", "content": "Weather in Paris?", "tool_calls": []},
        {"role": "assistant", "content": "",
         "tool_calls": [("get_weather", [("city", "Paris")])]},
        {"role": "tool", "content": '{"temp_c": 21}', "tool_calls": []},
    ]
    tools = ['{"type": "function", "function": {"name": "get_weather"}}']
    got = render_qwen_chatml(turns, tools, add_generation_prompt=True)
    expected = (
        "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n"
        '<tools>\n{"type": "function", "function": {"name": "get_weather"}}\n</tools>'
        + QWEN_TOOLS_INSTRUCTION
        + "\n\nBe terse.<|im_end|>\n"
        "<|im_start|>user\nWeather in Paris?<|im_end|>\n"
        "<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\n"
        "Paris\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        '<|im_start|>user\n<tool_response>\n{"temp_c": 21}\n</tool_response><|im_end|>\n'
        "<|im_start|>assistant\n<think>\n"
    )
    if got != expected:
        # locate the first divergent byte for a usable failure message
        for i, (a, b) in enumerate(zip(got, expected)):
            if a != b:
                print(f"selftest FAIL: first divergence at byte {i}: "
                      f"got {a!r}, want {b!r}\ncontext: ...{got[max(0, i-40):i+40]!r}...",
                      file=sys.stderr)
                break
        else:
            print(f"selftest FAIL: length mismatch {len(got)} != {len(expected)}",
                  file=sys.stderr)
        return 1
    # content + multi-call + grouped tool turns (Rust test
    # `assistant_content_plus_calls_and_consecutive_tool_turns_group`)
    turns2 = [
        {"role": "user", "content": "both", "tool_calls": []},
        {"role": "assistant", "content": "checking",
         "tool_calls": [("a", [("x", "1")]), ("b", [])]},
        {"role": "tool", "content": "r1", "tool_calls": []},
        {"role": "tool", "content": "r2", "tool_calls": []},
    ]
    got2 = render_qwen_chatml(turns2, [], add_generation_prompt=False)
    expected2 = (
        "<|im_start|>user\nboth<|im_end|>\n"
        "<|im_start|>assistant\nchecking\n\n"
        "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=b>\n</function>\n</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\nr1\n</tool_response>"
        "\n<tool_response>\nr2\n</tool_response><|im_end|>\n"
    )
    if got2 != expected2:
        print(f"selftest FAIL: multi-call vector diverged:\n{got2!r}", file=sys.stderr)
        return 1
    errs = roundtrip_check(turns2, [], got2)
    if errs:
        print("selftest FAIL: roundtrip on golden vector:", *errs, sep="\n  ",
              file=sys.stderr)
        return 1
    print("selftest OK: byte-identical to chat.rs golden vectors (2/2), roundtrip OK",
          file=sys.stderr)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", nargs="?", help="validated trace JSONL ('-' = stdin)")
    ap.add_argument("-o", "--output", default="-", help="converted JSONL (default stdout)")
    ap.add_argument("--no-think", action="store_true",
                    help="do not embed reasoning_content as a <think> block (moved to meta)")
    ap.add_argument("--gen-prompt", action="store_true",
                    help="append the generation prompt (serve-parity checks, not training)")
    ap.add_argument("--roundtrip", action="store_true",
                    help="parse each rendered text back and verify structure survives")
    ap.add_argument("--selftest", action="store_true",
                    help="run the chat.rs golden byte-parity vectors and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.input is None:
        ap.error("input is required (or use --selftest)")

    fin = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    fout = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    n = bad = 0
    with fin, fout:
        for ln, line in enumerate(fin, 1):
            if not line.strip():
                continue
            n += 1
            try:
                trace = json.loads(line)
                record, turns = convert_trace(
                    trace, embed_think=not args.no_think,
                    add_generation_prompt=args.gen_prompt)
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                print(f"{args.input}:{ln}: conversion failed: {e}", file=sys.stderr)
                bad += 1
                continue
            if args.roundtrip and not args.gen_prompt:
                tools_json = [pyjson(t) for t in trace.get("tools") or []]
                errs = roundtrip_check(turns, tools_json, record["text"])
                for e in errs:
                    print(f"{args.input}:{ln}: roundtrip: {e}", file=sys.stderr)
                if errs:
                    bad += 1
                    continue
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    if bad:
        print(f"FAIL: {bad}/{n} trace(s) failed conversion", file=sys.stderr)
        return 1
    print(f"converted {n} trace(s)"
          + (" (roundtrip verified)" if args.roundtrip else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
