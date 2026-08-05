#!/usr/bin/env python3
"""Session-affinity harness: the OWNER REGIME (pi's history rewrite) against a
chat-template-rendered conversation, with a byte-identity gate.

Descends from the F5 lane's drive-session.py. Two things are new, both required
by the affinity lane:

 1. TEMPLATE MARKERS. pi renders the chat template CLIENT-side and posts raw
    /v1/completions, so the token stream the worker sees carries the template's
    own <|im_start|>/<|im_end|> control tokens. The implicit affinity tier
    segments the conversation at exactly those tokens, so the harness must
    render them too — F5's plain-prose driver has no segment structure at all
    (one segment, no identity, affinity correctly declines).

 2. THE THINK-STRIP, not a char chop. F5's --rewrite dropped the last 5 chars of
    each response, which mutates the TAIL. pi strips <think> blocks out of prior
    assistant turns — an INTERIOR mutation of a turn whose boundaries survive.
    This harness reproduces that: each turn asks the model to open with a
    parenthetical aside, and the aside is deleted from history on the next turn.

BYTE-IDENTITY GATE (--gate): the contract is that a resumed session emits
byte-identical output to a fresh full prime of the same request. Run the same
conversation twice against two servers — one with the pool live (resume path),
one with MEMRA_REUSE_POOL=0 (every turn cold-primes) — and compare per-turn
text_sha. Any mismatch is a lane-blocking failure.

Usage:
  drive-affinity.py <port> <out.jsonl> [turns] [--session-id ID] [--no-rewrite]
  drive-affinity.py --gate <resume.jsonl> <fresh.jsonl>
"""
import hashlib, json, sys, time, urllib.request

KEY = "aviary-local"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"

# ~8k-token deterministic base document (~4 chars/token), same shape as the F5 driver
# so prompt sizes stay comparable across the two lanes' logs.
PARA = ("Section {i}: The pipeline stages data from storage through pinned host "
        "buffers into device memory, overlapping transfer with compute so that "
        "neither the copy engines nor the SMs sit idle while the other works. "
        "Careful accounting of bytes per token keeps the budget honest. ")
BASE = "".join(PARA.format(i=i) for i in range(220))
SYSTEM = ("You are a careful technical assistant. Read the document and answer "
          "questions about it concisely. Begin every answer with a short "
          "parenthetical aside in (parentheses), then the answer.\n\nDOCUMENT:\n" + BASE)


def render(turns):
    """Client-side chat template (ChatML), exactly as pi does it."""
    out = [f"{IM_START}system\n{SYSTEM}{IM_END}\n"]
    for role, content in turns:
        out.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    out.append(f"{IM_START}assistant\n")
    return "".join(out)


def strip_aside(text):
    """pi's think-strip analogue: delete the leading parenthetical aside — an
    INTERIOR edit of an assistant turn whose boundaries (role marker, <|im_end|>)
    are untouched."""
    s = text.lstrip()
    if s.startswith("("):
        close = s.find(")")
        if close != -1:
            return s[close + 1:].lstrip()
    return text


def drive(port, out_path, n_turns, session_id, rewrite):
    url = f"http://127.0.0.1:{port}/v1/completions"
    history = []
    rows = []
    for turn in range(n_turns):
        history.append(("user", f"Summarize section {3 + turn} in one sentence, then "
                                f"relate it to section {4 + turn}."))
        prompt = render(history)
        payload = {"model": "qwen36-27b", "prompt": prompt,
                   "max_tokens": 100, "temperature": 0}
        if session_id:
            payload["session_id"] = session_id
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {KEY}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            print(f"# HTTP {e.code} at turn {turn}: "
                  f"{e.read().decode(errors='replace')[:500]}", flush=True)
            raise
        dt = time.time() - t0
        text = resp["choices"][0]["text"]
        usage = resp.get("usage", {})
        row = {"turn": turn, "wall_s": round(dt, 3),
               "prompt_chars": len(prompt),
               "prompt_tokens": usage.get("prompt_tokens"),
               "cached_tokens": (usage.get("prompt_tokens_details") or {})
                                .get("cached_tokens"),
               "completion_tokens": usage.get("completion_tokens"),
               "gen_chars": len(text),
               "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16]}
        rows.append(row)
        print(json.dumps(row), flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        # THE REWRITE: history keeps the answer with its aside REMOVED, so the next
        # turn re-sends a mutated interior for this turn — pi's think-strip.
        history.append(("assistant", strip_aside(text) if rewrite else text))
    tot = sum(r["wall_s"] for r in rows)
    print(f"# total {tot:.1f}s over {len(rows)} turns", flush=True)


def gate(resume_path, fresh_path):
    def load(p):
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip() and not l.startswith("#")]
    a, b = load(resume_path), load(fresh_path)
    if len(a) != len(b):
        print(f"FAIL: turn count {len(a)} vs {len(b)}")
        return 1
    bad = [(x["turn"], x["text_sha"], y["text_sha"])
           for x, y in zip(a, b) if x["text_sha"] != y["text_sha"]]
    for turn, sa, sb in bad:
        print(f"MISMATCH turn {turn}: resume {sa} != fresh {sb}")
    print(f"{'FAIL' if bad else 'PASS'}: {len(a) - len(bad)}/{len(a)} turns "
          f"byte-identical (resume vs fresh full-prime)")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "--gate":
        sys.exit(gate(sys.argv[2], sys.argv[3]))
    port = int(sys.argv[1])
    out = sys.argv[2]
    turns = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 25
    sid = None
    if "--session-id" in sys.argv:
        sid = sys.argv[sys.argv.index("--session-id") + 1]
    drive(port, out, turns, sid, "--no-rewrite" not in sys.argv)
