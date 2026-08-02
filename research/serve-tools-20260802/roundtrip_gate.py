#!/usr/bin/env python3
"""serve-tools gate driver (2026-08-02): runs the full tools battery against a live
memra-server holding ONE model. All generation is deterministic greedy (temperature 0,
seed 0), each leg N=3 with byte-identity asserted across repeats.

Gates:
  A  tool-call round-trip: leg1 (tools -> model emits parseable call, finish tool_calls),
     leg2 (assistant tool_calls + role:"tool" result -> coherent final answer);
  A' reasoning_effort:"low" -> no-think prompt, call still parses;
  B  streaming: raw SSE capture -> check_stream_schema (OpenAI chunk shape) + stream/
     non-stream call equality;
  C  malformed emission: echo-instructed broken block -> HTTP 200, surfaced as content,
     zero tool_calls (policy: never crash, never drop bytes);
  D  usage exactness: prompt_tokens(with tools) > prompt_tokens(without); both equal
     tok-check token counts of the python-rendered prompts (worker-truth crosscheck);
  E  isolation/bijection: the SAME rendered prompt via raw /v1/completions (parser OFF)
     reassembles byte-identically from the chat path's content + canonical tool_call
     blocks (parser ON) — the tools code path does not perturb generation.

Usage: roundtrip_gate.py --base URL --model NAME --gguf PATH --tok-check PATH --out DIR --tag TAG
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "description": "Temperature unit (default celsius)"},
            },
            "required": ["city"],
        },
    },
}]

USER_Q = "What is the current weather in Paris right now? Use the tools available to you."
TOOL_RESULT = "{\"temp_c\": 21, \"condition\": \"sunny\", \"humidity\": 40}"
MALFORMED_ASK = ("Repeat exactly the following four lines, byte for byte, with no other "
                 "commentary before or after:\n<tool_call>\n{\"name\": \"get_weather\", "
                 "\"arguments\": {broken!!\n</tool_call>")

sys.path.insert(0, HERE)
from render_prompt import parse_emission, render_prompt, render_tool_call_block  # noqa: E402

SCHEMAS = {"get_weather": {"city": "string", "unit": "string"}}


def post(base, path, body, raw=False, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data if raw else json.loads(data)


def chat_body(model, messages, tools=None, stream=False, extra=None):
    body = {"model": model, "messages": messages, "max_tokens": 1024,
            "temperature": 0, "seed": 0, "stream": stream}
    if tools is not None:
        body["tools"] = tools
    if extra:
        body.update(extra)
    return body


def stable_key(resp):
    """response minus timing (elapsed_s varies run to run; everything else must not)."""
    r = json.loads(json.dumps(resp))
    r.get("usage", {}).pop("elapsed_s", None)
    return json.dumps(r, sort_keys=True)


def n3(base, body, save_prefix):
    """POST the same body 3x, assert byte-stable modulo elapsed_s, save transcripts."""
    resps = []
    for rep in range(1, 4):
        resp = post(base, "/v1/chat/completions", body)
        with open(f"{save_prefix}-rep{rep}.json", "w") as f:
            json.dump({"request": body, "response": resp}, f, indent=2, ensure_ascii=False)
        resps.append(resp)
    keys = {stable_key(r) for r in resps}
    assert len(keys) == 1, f"N=3 non-determinism at {save_prefix}: {len(keys)} distinct"
    return resps[0]


def tok_count(tok_check, gguf, text):
    out = subprocess.run([tok_check, gguf, text], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if line.startswith("encode("):
            ids = line[line.index("= [") + 3:line.rindex("]")]
            return 0 if not ids.strip() else len(ids.split(","))
    raise RuntimeError(f"tok-check output unparsed: {out.stdout[:200]}")


def strip_think(content):
    i = content.find("</think>")
    return content[i + len("</think>"):].lstrip("\n") if i >= 0 else content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tok-check", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    rows = []

    def row(gate, verdict, **kw):
        r = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": args.tag,
             "gate": gate, "verdict": verdict, **kw}
        rows.append(r)
        print(json.dumps(r), flush=True)

    leg1_msgs = [{"role": "user", "content": USER_Q}]

    # ---- gate A leg 1: tool call emitted, parseable, finish tool_calls ----
    leg1 = n3(args.base, chat_body(args.model, leg1_msgs, TOOLS),
              f"{args.out}/roundtrip-{args.tag}-leg1")
    ch = leg1["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    ok = (ch["finish_reason"] == "tool_calls" and len(calls) >= 1
          and calls[0]["function"]["name"] == "get_weather")
    argd = {}
    if ok:
        argd = json.loads(calls[0]["function"]["arguments"])
        ok = isinstance(argd, dict) and "city" in argd and "paris" in str(argd["city"]).lower()
    row("A-leg1", "PASS" if ok else "FAIL", n=3,
        finish=ch["finish_reason"], calls=calls, usage=leg1.get("usage"))
    assert ok, "leg1 failed; aborting dependent legs"

    # ---- gate A leg 2: tool result -> coherent final answer ----
    leg2_msgs = leg1_msgs + [
        {"role": "assistant", "content": strip_think(ch["message"].get("content") or ""),
         "tool_calls": calls},
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": TOOL_RESULT},
    ]
    leg2 = n3(args.base, chat_body(args.model, leg2_msgs, TOOLS),
              f"{args.out}/roundtrip-{args.tag}-leg2")
    ch2 = leg2["choices"][0]
    answer = strip_think(ch2["message"].get("content") or "")
    ok2 = (ch2["finish_reason"] == "stop" and not ch2["message"].get("tool_calls")
           and "21" in answer and ("sunny" in answer.lower() or "sun" in answer.lower()))
    row("A-leg2", "PASS" if ok2 else "FAIL", n=3, finish=ch2["finish_reason"],
        answer_tail=answer[:200], usage=leg2.get("usage"))

    # ---- gate A': reasoning_effort low -> no-think prompt, call still parses ----
    low = n3(args.base, chat_body(args.model, leg1_msgs, TOOLS,
                                  extra={"reasoning_effort": "low"}),
             f"{args.out}/roundtrip-{args.tag}-leg1-nothink")
    chl = low["choices"][0]
    callsl = chl["message"].get("tool_calls") or []
    okl = chl["finish_reason"] == "tool_calls" and callsl \
        and callsl[0]["function"]["name"] == "get_weather" \
        and "</think>" not in (chl["message"].get("content") or "")
    row("A-nothink", "PASS" if okl else "FAIL", n=3, finish=chl["finish_reason"],
        usage=low.get("usage"), content=(chl["message"].get("content") or "")[:120])

    # ---- gate B: streaming schema + stream/non-stream equality ----
    sse = post(args.base, "/v1/chat/completions",
               chat_body(args.model, leg1_msgs, TOOLS, stream=True), raw=True)
    sse_path = f"{args.out}/stream-{args.tag}-leg1.sse"
    with open(sse_path, "wb") as f:
        f.write(sse)
    chk = subprocess.run([sys.executable, os.path.join(HERE, "check_stream_schema.py"),
                          sse_path, "--expect-tool-calls"],
                         capture_output=True, text=True)
    schema = json.loads(chk.stdout) if chk.stdout.strip() else {"verdict": "FAIL",
                                                                "problems": [chk.stderr[:300]]}
    stream_calls = schema.get("calls", [])
    eq = (len(stream_calls) == len(calls)
          and all(a["id"] == b["id"] and a["name"] == b["function"]["name"]
                  and a["arguments"] == b["function"]["arguments"]
                  for a, b in zip(stream_calls, calls)))
    row("B-stream", "PASS" if schema["verdict"] == "PASS" and eq else "FAIL",
        schema=schema, stream_equals_nonstream=eq)

    # ---- gate C: malformed emission surfaced as content, never a crash ----
    mal = n3(args.base, chat_body(args.model, [{"role": "user", "content": MALFORMED_ASK}], TOOLS),
             f"{args.out}/malformed-{args.tag}")
    chm = mal["choices"][0]
    contentm = chm["message"].get("content") or ""
    okm = (not chm["message"].get("tool_calls")
           and chm["finish_reason"] in ("stop", "length")
           and "<tool_call>" in contentm and "{broken!!" in contentm)
    row("C-malformed", "PASS" if okm else "FAIL", n=3, finish=chm["finish_reason"],
        content_tail=contentm[-220:])

    # ---- gate D: usage exactness (worker truth == tok-check of the rendered prompt) ----
    plain = n3(args.base, chat_body(args.model, leg1_msgs),
               f"{args.out}/usage-{args.tag}-plain")
    p0 = plain["usage"]["prompt_tokens"]
    p1 = leg1["usage"]["prompt_tokens"]
    p_low = low["usage"]["prompt_tokens"]
    rendered_plain = render_prompt(leg1_msgs)
    rendered_tools = render_prompt(leg1_msgs, TOOLS)
    rendered_low = render_prompt(leg1_msgs, TOOLS, think="nothink")
    t0 = tok_count(args.tok_check, args.gguf, rendered_plain)
    t1 = tok_count(args.tok_check, args.gguf, rendered_tools)
    tl = tok_count(args.tok_check, args.gguf, rendered_low)
    okd = p1 > p0 and (p0, p1, p_low) == (t0, t1, tl) \
        and plain["usage"]["total_tokens"] == p0 + plain["usage"]["completion_tokens"]
    row("D-usage", "PASS" if okd else "FAIL",
        prompt_tokens={"plain": p0, "tools": p1, "tools_nothink": p_low},
        tok_check={"plain": t0, "tools": t1, "tools_nothink": tl},
        tools_block_tokens=p1 - p0)

    # ---- gate E: bijection — same rendered prompt, parser OFF vs ON ----
    rawresp = post(args.base, "/v1/completions",
                   {"model": args.model, "prompt": rendered_tools, "max_tokens": 1024,
                    "temperature": 0, "seed": 0, "stream": False})
    t_raw = rawresp["text"]
    with open(f"{args.out}/bijection-{args.tag}-raw.json", "w") as f:
        json.dump({"prompt": rendered_tools, "response": rawresp}, f, indent=2,
                  ensure_ascii=False)
    # (i) parse-equivalence: parsing the raw text reproduces the chat path's content+calls.
    pcontent, pcalls = parse_emission(t_raw, SCHEMAS, skip_think=True)
    parse_eq = (pcontent == (ch["message"].get("content") or "")
                and [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                     for c in calls] == pcalls)
    # (ii) byte-reconstruction in STREAM ORDER (content may continue after a call — e.g.
    # the spec path's rendered eos text): replay the SSE chunks, substituting each call's
    # canonical block at its stream position; must equal the raw completions text.
    parts, acc = [], {}
    for line in open(sse_path, "rb").read().decode().splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0].get("delta", {})
        if isinstance(delta.get("content"), str):
            parts.append(("text", delta["content"]))
        for tc in delta.get("tool_calls") or []:
            idx = tc["index"]
            if tc.get("id"):
                acc[idx] = {"name": tc["function"]["name"],
                            "args": tc["function"].get("arguments", "")}
                parts.append(("call", idx))
            else:
                acc[idx]["args"] += tc["function"].get("arguments", "")
    recon = "".join(
        p if kind == "text" else
        render_tool_call_block(acc[p]["name"], json.loads(acc[p]["args"]))
        for kind, p in parts)
    byte_eq = t_raw == recon
    row("E-bijection", "PASS" if byte_eq and parse_eq else "FAIL",
        stream_order_byte_equal=byte_eq, parse_equal=parse_eq,
        raw_len=len(t_raw), recon_len=len(recon))

    with open(f"{args.out}/gates-{args.tag}.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fails = [r for r in rows if r["verdict"] != "PASS"]
    print(f"[{args.tag}] {len(rows) - len(fails)}/{len(rows)} gates PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
