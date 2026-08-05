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
    This harness reproduces that literally: the emitted <think>...</think> span
    is deleted from history before the next turn re-sends it.

    The daily model is a reasoning model, so the strip needs a token budget large
    enough for the block to CLOSE (--max-tokens; a budget that truncates mid-think
    leaves nothing to strip and the run silently degrades into F5's pure-extension
    pattern — measured, that is exactly what a 100-token budget does). Every row
    records `rewrote` so a run can never claim the regime it did not reproduce.

BYTE-IDENTITY GATE. The contract: a resumed session emits byte-identical output
to a fresh full prime of THE SAME REQUEST. Two arms driven independently do not
test that — the moment one turn differs, the two conversations have different
histories and every later turn compares different prompts, so one divergence
cascades into 20 uninterpretable rows.

So the gate is a REPLAY, and each turn is an independent same-input comparison:
  phase 1  drive the conversation against the resume-arm server (MEMRA_AFFINITY=1),
           recording each turn's kept (think-stripped) assistant text into a
           transcript.
  phase 2  --replay that transcript against the control-arm server
           (MEMRA_AFFINITY=0). Prompts are rebuilt from the RECORDED history, not
           from the control server's own output, so both arms see byte-identical
           prompts at every turn regardless of what the control arm generates.
  phase 3  --gate compares per-turn text.

TOLERANCE: burst overshoot, exactly as tools/serve-st-gate.sh check 4 defines it.
A spec burst emits in bursts of up to K, so a run may stop up to K tokens past
max_tokens, and a resumed session's bursts need not align with a cold one's
(measured: 602 vs 600 completion tokens on the same 12317-token prompt). The
shorter text must be a PREFIX of the longer; anything else is a real divergence.

Usage:
  drive-affinity.py <port> <out.jsonl> [turns] [--session-id ID] [--no-rewrite]
                    [--max-tokens N] [--transcript FILE]
  drive-affinity.py --replay <transcript> <port> <out.jsonl> [--max-tokens N]
                    [--session-id ID] [--cold]
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
          "questions about it concisely.\n\nDOCUMENT:\n" + BASE)


def render(turns):
    """Client-side chat template (ChatML), exactly as pi does it."""
    out = [f"{IM_START}system\n{SYSTEM}{IM_END}\n"]
    for role, content in turns:
        out.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    out.append(f"{IM_START}assistant\n")
    return "".join(out)


def strip_think(text):
    """pi's think-strip: delete the <think>...</think> span from an assistant turn
    before re-sending it. An INTERIOR edit — the turn's boundaries (role marker,
    <|im_end|>) are untouched, which is why the prefix probes miss but the
    structural fingerprint still matches. Returns (text, did_strip)."""
    open_i = text.find("<think>")
    close_i = text.find("</think>")
    if open_i != -1 and close_i > open_i:
        return (text[:open_i] + text[close_i + len("</think>"):]).lstrip(), True
    return text, False


def ask(url, prompt, max_tokens, session_id, salt=None):
    payload = {"model": "qwen36-27b", "prompt": prompt,
               "max_tokens": max_tokens, "temperature": 0}
    if session_id:
        payload["session_id"] = session_id
    # --cold: a per-turn cache_salt puts every request in its own PC-ISO namespace, so no
    # pool probe (token-prefix, text-prefix, or affinity) can hit and every turn primes
    # cold. This is how a cold arm is obtained on a binary that predates MEMRA_AFFINITY —
    # and it does not depend on MEMRA_REUSE_POOL=0, which panics the pre-lane worker.
    if salt:
        payload["cache_salt"] = salt
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"# HTTP {e.code}: {e.read().decode(errors='replace')[:500]}", flush=True)
        raise
    return resp, time.time() - t0


def row_for(turn, prompt, text, resp, dt, stripped):
    usage = resp.get("usage", {})
    return {"turn": turn, "wall_s": round(dt, 3),
            "prompt_chars": len(prompt),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {})
                             .get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "gen_chars": len(text),
            # rewrote=false on a --rewrite run means the think block never CLOSED
            # inside the budget: this turn's history is a pure extension and the
            # affinity regime was not exercised. Never summarize past this field.
            "rewrote": stripped,
            # The gate needs the TEXT, not only its digest: burst overshoot is a
            # tolerated difference (see the module docstring) and a prefix test
            # cannot be run on hashes.
            "text": text,
            "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16]}


def emit(out_path, row, rows):
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def drive(port, out_path, n_turns, session_id, rewrite, max_tokens, transcript,
          long_answers=False):
    url = f"http://127.0.0.1:{port}/v1/completions"
    history, rows = [], []
    for turn in range(n_turns):
        # --long forces every turn to RUN THE BUDGET OUT. Needed to test the long-window
        # near-tie class: the default one-sentence question stops at ~325 tokens, well
        # short of where resumed-vs-cold FP divergence appears.
        ask_more = (" Then, separately, restate each of the following in its own "
                    "sentence: the storage stage, the pinned-host stage, the device "
                    "stage, the overlap argument, and the byte budget. Be thorough."
                    if long_answers else "")
        history.append(("user", f"Summarize section {3 + turn} in one sentence, then "
                                f"relate it to section {4 + turn}.{ask_more}"))
        prompt = render(history)
        resp, dt = ask(url, prompt, max_tokens, session_id)
        text = resp["choices"][0]["text"]
        # THE REWRITE: history keeps the answer with its <think> span REMOVED, so the
        # next turn re-sends a mutated interior for this turn — pi's think-strip.
        kept, stripped = strip_think(text) if rewrite else (text, False)
        emit(out_path, row_for(turn, prompt, text, resp, dt, stripped), rows)
        history.append(("assistant", kept))
    tot = sum(r["wall_s"] for r in rows)
    nrw = sum(1 for r in rows if r["rewrote"])
    print(f"# total {tot:.1f}s over {len(rows)} turns; rewrote {nrw}/{len(rows)}", flush=True)
    if transcript:
        with open(transcript, "w") as f:
            json.dump({"max_tokens": max_tokens, "history": history}, f)
        print(f"# transcript -> {transcript}", flush=True)


def replay(transcript_path, port, out_path, max_tokens, session_id, cold=False):
    """Re-issue the recorded conversation turn by turn. Each request's prompt is
    rebuilt from the RECORDED history, so this arm sees byte-identical prompts to
    the arm that produced the transcript no matter what it generates itself."""
    url = f"http://127.0.0.1:{port}/v1/completions"
    with open(transcript_path) as f:
        t = json.load(f)
    history = [tuple(x) for x in t["history"]]
    max_tokens = max_tokens or t["max_tokens"]
    rows = []
    for turn in range(0, len(history), 2):
        prompt = render(history[:turn + 1])
        resp, dt = ask(url, prompt, max_tokens, session_id,
                       salt=f"cold-{turn}" if cold else None)
        text = resp["choices"][0]["text"]
        # `rewrote` is a property of the RECORDED history (was this turn's stored
        # answer think-stripped?), not of what this arm just generated.
        stripped = "<think>" not in history[turn + 1][1] if turn + 1 < len(history) else False
        emit(out_path, row_for(turn // 2, prompt, text, resp, dt, stripped), rows)
    tot = sum(r["wall_s"] for r in rows)
    print(f"# total {tot:.1f}s over {len(rows)} replayed turns", flush=True)


def gate(resume_path, fresh_path):
    def load(p):
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip() and not l.startswith("#")]
    a, b = load(resume_path), load(fresh_path)
    if len(a) != len(b):
        print(f"FAIL: turn count {len(a)} vs {len(b)}")
        return 1
    def agree(x, y):
        if x["text_sha"] == y["text_sha"]:
            return True
        # TOLERATED: burst overshoot only (serve-st-gate check 4's rule) — the shorter
        # text must be an exact prefix of the longer.
        s, t = x.get("text"), y.get("text")
        if s is None or t is None:
            return False
        return t.startswith(s) or s.startswith(t)

    bad = [(x["turn"], x["text_sha"], y["text_sha"])
           for x, y in zip(a, b) if not agree(x, y)]
    for turn, sa, sb in bad:
        print(f"MISMATCH turn {turn}: resume {sa} != fresh {sb}")
    over = sum(1 for x, y in zip(a, b) if x["text_sha"] != y["text_sha"] and agree(x, y))
    if over:
        print(f"note: {over} turn(s) matched by burst-overshoot prefix, not exact sha")
    # A run whose history was never rewritten is a pure prefix extension: the prefix
    # probes carry it and affinity is never asked, so identical shas would prove nothing.
    # Refuse to call that a pass.
    rw = sum(1 for x in a if x.get("rewrote"))
    if rw == 0:
        print("FAIL: no turn rewrote its history — the affinity regime was never exercised")
        return 1
    print(f"{'FAIL' if bad else 'PASS'}: {len(a) - len(bad)}/{len(a)} turns "
          f"byte-identical (resume vs fresh full-prime); {rw}/{len(a)} turns rewrote history")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "--gate":
        sys.exit(gate(sys.argv[2], sys.argv[3]))
    sid = None
    if "--session-id" in sys.argv:
        sid = sys.argv[sys.argv.index("--session-id") + 1]
    maxtok = 0
    if "--max-tokens" in sys.argv:
        maxtok = int(sys.argv[sys.argv.index("--max-tokens") + 1])
    if sys.argv[1] == "--replay":
        sys.exit(replay(sys.argv[2], int(sys.argv[3]), sys.argv[4], maxtok, sid,
                        "--cold" in sys.argv) or 0)
    port = int(sys.argv[1])
    out = sys.argv[2]
    turns = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 25
    tr = None
    if "--transcript" in sys.argv:
        tr = sys.argv[sys.argv.index("--transcript") + 1]
    drive(port, out, turns, sid, "--no-rewrite" not in sys.argv, maxtok or 600, tr,
          "--long" in sys.argv)
