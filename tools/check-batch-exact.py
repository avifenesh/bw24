#!/usr/bin/env python3
"""check-batch-exact: serving-level batched-decode exactness check.

Phase A: run N distinct greedy prompts SEQUENTIALLY (c=1 -> B=1 decode, the bit-exact
decode_step_h contract) and record each completion.
Phase B: fire the same N prompts CONCURRENTLY so the scheduler batches them
(B up to the active chunk cap) and compare each completion to its isolated reference.

Under the v1 exactness tier (chunk<=8, batched mmvq b2/b4/b8) every output must be
byte-identical to isolated. Chunk>8 (MEMRA_DECODE_BATCH_CAP probe) crosses numeric
tiers — this tool measures exactly how many outputs move.

Usage: check-batch-exact.py --base http://127.0.0.1:8085 --model qwen --n 20 \
          [--max-tokens 64] [--label chunk16] [--out results.jsonl] [--ref refs.json]
If --ref exists it is loaded instead of re-running phase A (so all chunk sizes are
compared against the SAME isolated references).
"""

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request

TOPICS = [
    "the roman aqueduct system", "how a transistor amplifies current",
    "the lifecycle of a monarch butterfly", "why the sky appears blue",
    "the rules of the game go", "how yeast leavens bread",
    "the water cycle", "plate tectonics and earthquakes",
    "how vaccines train immunity", "the doppler effect",
    "photosynthesis in c4 plants", "the history of the metric system",
    "how gps triangulates position", "the physics of a curveball",
    "why ice floats on water", "the invention of movable type",
    "how lithium batteries store energy", "the great barrier reef ecosystem",
    "the halting problem", "how compilers optimize loops",
    "the formation of hurricanes", "the krebs cycle",
    "how noise-cancelling headphones work", "the gold standard in economics",
    "the structure of dna", "how radar measures speed",
    "the causes of seasons", "how a laser produces coherent light",
    "the digestion of starch", "the tragedy of the commons",
    "how sonar maps the seafloor", "the properties of graphene",
]


def ask(base, model, prompt, max_tokens, timeout=600):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 0, "stream": False}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        # evidence discipline: quote the server's error body, not just the status line.
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None
    return d["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None,
                    help="isolated-reference cache file (json); reused if present")
    args = ap.parse_args()

    prompts = [f"Explain {t} in two sentences." for t in TOPICS[:args.n]]

    # Phase A: isolated references (sequential -> B=1)
    if args.ref and os.path.exists(args.ref):
        with open(args.ref) as f:
            refs = json.load(f)
        assert len(refs) >= args.n, "ref file has fewer prompts than --n"
        refs = refs[:args.n]
        print(f"[ref] loaded {len(refs)} isolated references from {args.ref}")
    else:
        refs = []
        for p in prompts:
            refs.append(ask(args.base, args.model, p, args.max_tokens))
        if args.ref:
            with open(args.ref, "w") as f:
                json.dump(refs, f)
        print(f"[ref] collected {len(refs)} isolated references")

    # Phase B: concurrent (batched)
    outs = [None] * args.n
    errs = [None] * args.n

    def worker(i):
        try:
            outs[i] = ask(args.base, args.model, prompts[i], args.max_tokens)
        except Exception as e:
            errs[i] = str(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.n)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0

    n_match = sum(1 for i in range(args.n) if outs[i] == refs[i])
    n_err = sum(1 for e in errs if e)
    mismatches = []
    for i in range(args.n):
        if errs[i] or outs[i] == refs[i]:
            continue
        # first divergence point
        a, b = refs[i], outs[i]
        k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        mismatches.append({"i": i, "diverge_at_char": k,
                           "ref_sha": hashlib.sha256(a.encode()).hexdigest()[:12],
                           "out_sha": hashlib.sha256(b.encode()).hexdigest()[:12],
                           "ref_tail": a[k:k + 40], "out_tail": b[k:k + 40]})

    # evidence discipline: failure causes are quoted, never inferred — carry the raw
    # per-request error strings into the row (they were previously swallowed).
    errors = [{"i": i, "err": e} for i, e in enumerate(errs) if e]

    verdict = "PASS" if (n_match == args.n and n_err == 0) else "FAIL"
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": args.label,
           "base": args.base, "n": args.n, "max_tokens": args.max_tokens,
           "wall_s": wall, "n_match": n_match, "n_mismatch": args.n - n_match - n_err,
           "n_err": n_err, "verdict": verdict, "mismatches": mismatches[:8],
           "errors": errors[:8]}
    print(json.dumps(row))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
