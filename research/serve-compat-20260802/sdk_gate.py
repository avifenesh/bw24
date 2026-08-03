#!/usr/bin/env python3
"""serve-compat SDK gate (lane/serve-compat, 2026-08-03): the OFFICIAL `openai` Python SDK
round-trips completion + stream + tool-call against a live memra-server without pydantic
errors — the acceptance criterion for gap-scan F1 (the SDK used to REJECT every response:
ChatCompletion/ChatCompletionChunk pydantic-require `id: str` and `created: int`).

Gates (every row -> sdk-gates.jsonl; exit nonzero on any FAIL):
  G1  non-stream completion validates; id chatcmpl-*, created > 0, system_fingerprint,
      x-request-id header present.
  G2  stream validates chunk-by-chunk; FIRST delta carries role:"assistant"; every chunk
      shares one id/created/fingerprint; final chunk has finish_reason + usage; [DONE]
      terminates (SDK iterator exhausts cleanly).
  G3  tool-call round-trip: tools request -> finish_reason "tool_calls", parsed function
      name/arguments; STREAMING tool-call deltas accumulate under the SDK.
  G4  reasoning separation (F13): message carries `reasoning` (+ reasoning_details),
      content has NO think text; include_reasoning:false drops it.
  G5  max_tokens omitted (F2): generation not cut at the old 128 default.
  G6  error shape (F1): unsupported param -> BadRequestError with the OpenAI error object
      (message/type/param/code parseable by the SDK).
  G7  penalties accepted end-to-end (F3): frequency/presence/repetition_penalty -> 200.
  G8  disconnect probe (F8): open a stream, read a little, hang up; server stays healthy
      (the [abort] log line is asserted by the runner, which owns the server log).

Usage: sdk_gate.py --base http://127.0.0.1:PORT --model q9 --out DIR
"""

import argparse
import json
import sys
import time

import httpx
import openai
from openai import OpenAI

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
            },
            "required": ["city"],
        },
    },
}]

ROWS = []
FAILS = 0


def row(gate, verdict, **kw):
    global FAILS
    r = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gate": gate, "verdict": verdict, **kw}
    ROWS.append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
    if verdict != "PASS":
        FAILS += 1


def check(gate, cond, detail):
    row(gate, "PASS" if cond else "FAIL", detail=detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    client = OpenAI(base_url=args.base + "/v1", api_key="gate", timeout=600.0)
    m = args.model

    # ---- G1: non-stream envelope through the official SDK (pydantic validates) ----
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=m, temperature=0, seed=0, max_tokens=32,
            messages=[{"role": "user", "content": "Say OK."}])
        resp = raw.parse()  # pydantic ChatCompletion — the old server FAILED here
        check("G1-sdk-validates", True, "ChatCompletion parsed")
        check("G1-id", resp.id.startswith("chatcmpl-"), resp.id[:24])
        check("G1-created", resp.created > 1_700_000_000, str(resp.created))
        check("G1-fingerprint", (resp.system_fingerprint or "").startswith("memra-"),
              str(resp.system_fingerprint))
        check("G1-role", resp.choices[0].message.role == "assistant",
              resp.choices[0].message.role)
        check("G1-x-request-id", raw.headers.get("x-request-id", "") == resp.id,
              raw.headers.get("x-request-id", "<missing>"))
        check("G1-usage", resp.usage.total_tokens
              == resp.usage.prompt_tokens + resp.usage.completion_tokens,
              f"{resp.usage.prompt_tokens}+{resp.usage.completion_tokens}")
    except Exception as e:  # noqa: BLE001 — any SDK rejection IS the gate failure
        row("G1-sdk-validates", "FAIL", error=f"{type(e).__name__}: {e}")

    # ---- G2: stream envelope; first-delta role; single id; final usage ----
    try:
        stream = client.chat.completions.create(
            model=m, temperature=0, seed=0, max_tokens=32, stream=True,
            messages=[{"role": "user", "content": "Say OK."}])
        chunks = list(stream)  # each is a pydantic-validated ChatCompletionChunk
        check("G2-sdk-validates", len(chunks) > 0, f"{len(chunks)} chunks")
        ids = {c.id for c in chunks}
        check("G2-one-id", len(ids) == 1 and next(iter(ids)).startswith("chatcmpl-"),
              str(ids))
        check("G2-fingerprint-all", all((c.system_fingerprint or "").startswith("memra-")
                                        for c in chunks), "all chunks")
        first_delta = next(c.choices[0].delta for c in chunks if c.choices)
        check("G2-first-role", first_delta.role == "assistant", str(first_delta.role))
        fin = [c for c in chunks if c.choices and c.choices[0].finish_reason]
        check("G2-finish", len(fin) == 1 and fin[0].choices[0].finish_reason
              in ("stop", "length"), str([c.choices[0].finish_reason for c in fin]))
        check("G2-usage-final", fin and fin[0].usage is not None
              and fin[0].usage.prompt_tokens > 0, "usage on finish chunk")
    except Exception as e:  # noqa: BLE001
        row("G2-sdk-validates", "FAIL", error=f"{type(e).__name__}: {e}")

    # ---- G3: tool calls, non-stream + stream accumulation ----
    try:
        resp = client.chat.completions.create(
            model=m, temperature=0, seed=0, max_tokens=1024, tools=TOOLS,
            messages=[{"role": "user", "content":
                       "What is the weather in Paris right now? Use the tools."}])
        ch = resp.choices[0]
        calls = ch.message.tool_calls or []
        check("G3-finish", ch.finish_reason == "tool_calls", ch.finish_reason)
        check("G3-call", len(calls) == 1 and calls[0].function.name == "get_weather"
              and "paris" in json.loads(calls[0].function.arguments)
                  .get("city", "").lower(),
              calls[0].function.arguments if calls else "<none>")
        with open(f"{args.out}/sdk-g3-toolcall.json", "w") as f:
            f.write(resp.model_dump_json(indent=2))

        stream = client.chat.completions.create(
            model=m, temperature=0, seed=0, max_tokens=1024, tools=TOOLS, stream=True,
            messages=[{"role": "user", "content":
                       "What is the weather in Paris right now? Use the tools."}])
        name, arg_buf, finish = None, "", None
        for c in stream:
            if not c.choices:
                continue
            d = c.choices[0].delta
            for tc in d.tool_calls or []:
                if tc.function and tc.function.name:
                    name = tc.function.name
                if tc.function and tc.function.arguments:
                    arg_buf += tc.function.arguments
            finish = c.choices[0].finish_reason or finish
        check("G3-stream", finish == "tool_calls" and name == "get_weather"
              and "paris" in json.loads(arg_buf).get("city", "").lower(),
              f"finish={finish} name={name} args={arg_buf[:60]}")
    except Exception as e:  # noqa: BLE001
        row("G3-toolcall", "FAIL", error=f"{type(e).__name__}: {e}")

    # ---- G4: reasoning separation (raw JSON: the OR dialect fields ride model_extra) ----
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=m, temperature=0, seed=0, max_tokens=512,
            messages=[{"role": "user", "content": "What is 17 + 25?"}])
        body = json.loads(raw.text)
        msg = body["choices"][0]["message"]
        reasoning = msg.get("reasoning") or ""
        content = msg.get("content") or ""
        check("G4-reasoning-field", len(reasoning) > 0, f"{len(reasoning)} chars")
        check("G4-content-clean", "</think>" not in content and "<think>" not in content
              and "<|im_end|>" not in content,  # EOS text leak: spec-burst divergence,
              content[:60])                      # caught by this receipt on try2
        check("G4-details", isinstance(msg.get("reasoning_details"), list)
              and msg["reasoning_details"][0].get("text") == reasoning,
              "reasoning_details mirrors")
        with open(f"{args.out}/sdk-g4-reasoning.json", "w") as f:
            json.dump(body, f, indent=2, ensure_ascii=False)

        raw2 = client.chat.completions.with_raw_response.create(
            model=m, temperature=0, seed=0, max_tokens=512,
            messages=[{"role": "user", "content": "What is 17 + 25?"}],
            extra_body={"include_reasoning": False})
        msg2 = json.loads(raw2.text)["choices"][0]["message"]
        check("G4-exclude", "reasoning" not in msg2
              and "</think>" not in (msg2.get("content") or ""),
              "include_reasoning:false drops it")

        # streaming: think text arrives as delta.reasoning, never delta.content.
        stream_raw = ""
        with httpx.stream(
                "POST", args.base + "/v1/chat/completions", timeout=600.0,
                json={"model": m, "temperature": 0, "seed": 0, "max_tokens": 512,
                      "stream": True,
                      "messages": [{"role": "user", "content": "What is 17 + 25?"}]},
        ) as r:
            stream_raw = r.read().decode()
        deltas = [json.loads(line[6:]) for line in stream_raw.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        s_reason = "".join(d["choices"][0]["delta"].get("reasoning") or ""
                           for d in deltas if d.get("choices"))
        s_content = "".join(d["choices"][0]["delta"].get("content") or ""
                            for d in deltas if d.get("choices"))
        check("G4-stream-split", len(s_reason) > 0 and "</think>" not in s_content,
              f"reasoning={len(s_reason)}ch content={len(s_content)}ch")
    except Exception as e:  # noqa: BLE001
        row("G4-reasoning", "FAIL", error=f"{type(e).__name__}: {e}")

    # ---- G5: max_tokens omitted -> not the old 128 truncation ----
    try:
        resp = client.chat.completions.create(
            model=m, temperature=0, seed=0,
            messages=[{"role": "user", "content":
                       "Count from 1 to 200, separated by spaces. No other text."}])
        n = resp.usage.completion_tokens
        fr = resp.choices[0].finish_reason
        check("G5-not-128", not (fr == "length" and n <= 128), f"finish={fr} tokens={n}")
        check("G5-past-128", n > 128, f"{n} completion tokens")
    except Exception as e:  # noqa: BLE001
        row("G5-max-tokens", "FAIL", error=f"{type(e).__name__}: {e}")

    # ---- G6: unsupported params -> OpenAI error object the SDK can parse ----
    for param, kwargs in [
        ("logit_bias", {"logit_bias": {"50256": -100}}),
        ("n", {"n": 3}),
        ("response_format", {"response_format": {"type": "json_object"}}),
    ]:
        try:
            client.chat.completions.create(
                model=m, max_tokens=8,
                messages=[{"role": "user", "content": "hi"}], **kwargs)
            row(f"G6-{param}", "FAIL", detail="request was accepted (silent downgrade)")
        except openai.BadRequestError as e:
            body = e.body if isinstance(e.body, dict) else {}
            ok = (e.status_code == 400 and body.get("type") == "invalid_request_error"
                  and param in str(body.get("message", "")) and "param" in body
                  and "code" in body)
            check(f"G6-{param}", ok, str(body)[:120])
        except Exception as e:  # noqa: BLE001
            row(f"G6-{param}", "FAIL", error=f"{type(e).__name__}: {e}")

    # ---- G7: penalties plumb end-to-end (200 + generation) ----
    try:
        resp = client.chat.completions.create(
            model=m, temperature=0.7, seed=0, max_tokens=48,
            frequency_penalty=0.5, presence_penalty=0.2,
            extra_body={"repetition_penalty": 1.1},
            messages=[{"role": "user", "content": "Say OK."}])
        check("G7-penalties", resp.usage.completion_tokens > 0,
              f"{resp.usage.completion_tokens} tokens")
    except Exception as e:  # noqa: BLE001
        row("G7-penalties", "FAIL", error=f"{type(e).__name__}: {e}")

    # ---- G8: disconnect probe — hang up mid-stream, server must stay healthy ----
    try:
        got = 0
        with httpx.stream(
                "POST", args.base + "/v1/chat/completions", timeout=600.0,
                json={"model": m, "temperature": 0, "seed": 0, "max_tokens": 4096,
                      "stream": True,
                      "messages": [{"role": "user", "content":
                                    "Count from 1 to 2000, separated by spaces."}]},
        ) as r:
            for _line in r.iter_lines():
                got += 1
                if got >= 5:
                    break  # context-manager exit closes the connection mid-generation
        time.sleep(3)  # give the worker ticks to notice the closed channel
        health = httpx.get(args.base + "/health", timeout=10.0)
        check("G8-disconnect", health.status_code == 200,
              f"read {got} lines, hung up, health={health.status_code}"
              " ([abort] log line asserted by the runner)")
    except Exception as e:  # noqa: BLE001
        row("G8-disconnect", "FAIL", error=f"{type(e).__name__}: {e}")

    with open(f"{args.out}/sdk-gates.jsonl", "a") as f:
        for r in ROWS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"SDK-GATE {'FAIL' if FAILS else 'ALL GREEN'} ({len(ROWS)} rows, {FAILS} fails)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
