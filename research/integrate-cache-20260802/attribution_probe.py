#!/usr/bin/env python3
"""Attribution probe for the leg-R cold-vs-hit think-text divergence (2026-08-02).

Modes (server booted by the runner; this script only drives requests):
  cold2   — send the leg-R TOOLS chat request twice against a cache-OFF bulk server:
            is the cold path deterministic? (expect: byte-identical)
  raw3    — send the RENDERED leg-R prompt via /v1/completions three times against a
            cache-ON bulk server: rep1 cold seeds, rep3 full-prefix hit. Works on BOTH
            the merged binary and the pre-merge cache-lane binary (no tools API needed)
            — the prefix cache keys on token ids, not on the chat surface.
Output: one JSON row per comparison to stdout + transcripts under --out.
"""

import argparse
import json
import sys
import urllib.request

HERE = "/home/avifenesh/projects/bw24-integrate-cache/research/integrate-cache-20260802"
sys.path.insert(0, HERE + "/../serve-tools-20260802")
sys.path.insert(0, HERE)
from render_prompt import render_prompt  # noqa: E402
from intersection_gate import TOOLS, USER_Q  # noqa: E402


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["cold2", "raw3"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.mode == "cold2":
        body = {"model": args.model, "messages": [{"role": "user", "content": USER_Q}],
                "tools": TOOLS, "max_tokens": 1024, "temperature": 0, "seed": 0,
                "stream": False}
        texts = []
        for rep in (1, 2):
            resp = post(args.base, "/v1/chat/completions", body)
            with open(f"{args.out}/attr-{args.tag}-rep{rep}.json", "w") as f:
                json.dump(resp, f, indent=2, ensure_ascii=False)
            ch = resp["choices"][0]
            texts.append(json.dumps({"c": ch["message"].get("content"),
                                     "t": ch["message"].get("tool_calls")}, sort_keys=True))
            print(json.dumps({"tag": args.tag, "rep": rep,
                              "completion": resp["usage"]["completion_tokens"],
                              "cached": resp["usage"]["prompt_tokens_details"]["cached_tokens"]}))
        print(json.dumps({"tag": args.tag, "probe": "cold-determinism",
                          "identical": texts[0] == texts[1]}))
        sys.exit(0 if texts[0] == texts[1] else 1)

    rendered = render_prompt([{"role": "user", "content": USER_Q}], TOOLS)
    texts = []
    for rep in (1, 2, 3):
        resp = post(args.base, "/v1/completions",
                    {"model": args.model, "prompt": rendered, "max_tokens": 1024,
                     "temperature": 0, "seed": 0, "stream": False})
        with open(f"{args.out}/attr-{args.tag}-rep{rep}.json", "w") as f:
            json.dump(resp, f, indent=2, ensure_ascii=False)
        texts.append(resp["text"])
        print(json.dumps({"tag": args.tag, "rep": rep, "n_tokens": resp["n_tokens"],
                          "prompt_tokens": resp.get("prompt_tokens"),
                          "cached": resp.get("cached_tokens")}))
    same13 = texts[0] == texts[2]
    same23 = texts[1] == texts[2]
    n = next((i for i in range(min(len(texts[0]), len(texts[2])))
              if texts[0][i] != texts[2][i]), min(len(texts[0]), len(texts[2])))
    print(json.dumps({"tag": args.tag, "probe": "cold-vs-hit-raw",
                      "rep1_eq_rep3": same13, "rep2_eq_rep3": same23,
                      "diverge_at_char": None if same13 else n}))
    sys.exit(0)


if __name__ == "__main__":
    main()
