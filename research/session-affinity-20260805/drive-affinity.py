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
                    [--max-tokens N] [--transcript FILE] [--stream]
  drive-affinity.py --replay <transcript> <port> <out.jsonl> [--max-tokens N]
                    [--session-id ID] [--cold] [--only TURN] [--stream]
  drive-affinity.py --gate <resume.jsonl> <fresh.jsonl>
  drive-affinity.py --curve <field> <on-r*.jsonl> -- <off-r*.jsonl>

--stream requests SSE so ttft_s is measured (see ask()). --curve prints the per-turn
median of `field` across the N replicate files of each arm.
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


def ask(url, prompt, max_tokens, session_id, salt=None, stream=False):
    """Returns (resp, wall_s, ttft_s). ttft_s is None unless stream=True.

    TTFT is the number the lane exists to move, and it is NOT wall_s: wall_s bundles
    prefill with the whole generation, so a turn that also generates fewer tokens looks
    faster for the wrong reason. Only a streamed request can time the FIRST token —
    the clock stops on the first SSE chunk carrying non-empty text, which is prefill
    (+ one decode step), i.e. exactly the quantity the resume path shortcuts."""
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
    if stream:
        payload["stream"] = True
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            if not stream:
                return json.loads(r.read()), time.time() - t0, None
            return read_sse(r, t0)
    except urllib.error.HTTPError as e:
        # The BODY is the diagnostic, not the status code — print it, and print the
        # request shape that produced it (never the 60k-char prompt itself).
        print(f"# HTTP {e.code}: {e.read().decode(errors='replace')[:800]}", flush=True)
        print(f"# request: prompt_chars={len(prompt)} max_tokens={max_tokens} "
              f"salt={salt!r} session_id={session_id!r}", flush=True)
        raise


def read_sse(r, t0):
    """Reassemble an OpenAI-compat text_completion SSE stream into the non-stream shape,
    so the SAME row_for()/gate() path serves both modes and a streamed arm's text_sha is
    directly comparable to a blocking arm's. Returns (resp, wall_s, ttft_s)."""
    text, ttft, usage, finish = [], None, {}, None
    for raw in r:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue          # SSE keep-alive comment (`: ping`) or blank separator
        body = line[5:].strip()
        if body == "[DONE]":
            break
        ev = json.loads(body)
        if "error" in ev:
            raise RuntimeError(f"stream error: {ev['error']}")
        ch = (ev.get("choices") or [{}])[0]
        piece = ch.get("text") or ""
        if piece and ttft is None:
            ttft = time.time() - t0
        text.append(piece)
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
            usage = ev.get("usage") or {}
    return ({"choices": [{"text": "".join(text), "finish_reason": finish}],
             "usage": usage}, time.time() - t0, ttft)


def row_for(turn, prompt, text, resp, dt, stripped, ttft=None):
    usage = resp.get("usage", {})
    return {"turn": turn, "wall_s": round(dt, 3),
            # THE lane number (streamed arms only): prefill + first decode step.
            "ttft_s": None if ttft is None else round(ttft, 3),
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
          long_answers=False, stream=False):
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
        resp, dt, ttft = ask(url, prompt, max_tokens, session_id, stream=stream)
        text = resp["choices"][0]["text"]
        # THE REWRITE: history keeps the answer with its <think> span REMOVED, so the
        # next turn re-sends a mutated interior for this turn — pi's think-strip.
        kept, stripped = strip_think(text) if rewrite else (text, False)
        emit(out_path, row_for(turn, prompt, text, resp, dt, stripped, ttft), rows)
        history.append(("assistant", kept))
    tot = sum(r["wall_s"] for r in rows)
    nrw = sum(1 for r in rows if r["rewrote"])
    print(f"# total {tot:.1f}s over {len(rows)} turns; rewrote {nrw}/{len(rows)}", flush=True)
    if transcript:
        with open(transcript, "w") as f:
            json.dump({"max_tokens": max_tokens, "history": history}, f)
        print(f"# transcript -> {transcript}", flush=True)


def replay(transcript_path, port, out_path, max_tokens, session_id, cold=False,
           only=None, stream=False):
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
        # --only N: issue ONLY turn N's request. The cleanest possible cold control — a
        # fresh server serving exactly one request has no parked session to resume from
        # and no accumulated namespaces. (Per-turn cache_salt does force cold priming, but
        # each namespace parks its own ~4.2GB session, so on the 24GB card it OOMs by turn
        # 4: captured "step error: DriverError(CUDA_ERROR_OUT_OF_MEMORY)" at salt cold-4.)
        if only is not None and turn // 2 != only:
            continue
        prompt = render(history[:turn + 1])
        resp, dt, ttft = ask(url, prompt, max_tokens, session_id,
                             salt=f"cold-{turn}" if cold else None, stream=stream)
        text = resp["choices"][0]["text"]
        # `rewrote` is a property of the RECORDED history (was this turn's stored
        # answer think-stripped?), not of what this arm just generated.
        stripped = "<think>" not in history[turn + 1][1] if turn + 1 < len(history) else False
        emit(out_path, row_for(turn // 2, prompt, text, resp, dt, stripped, ttft), rows)
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


def curve(field, on_paths, off_paths):
    """Per-turn median of `field` across replicates, one row per turn, both arms + ratio.

    Median over replicates PER TURN, never a mean over turns: turn 0 is a cold prime in
    both arms (nothing to resume) and the rewrite turns are the interesting ones, so a
    single aggregate would hide the whole shape. N is printed with the table because a
    median without its N is not a number (evidence discipline)."""
    def load(paths):
        out = {}
        for p in paths:
            for l in open(p):
                if not l.strip() or l.startswith("#"):
                    continue
                r = json.loads(l)
                v = r.get(field)
                if v is not None:
                    out.setdefault(r["turn"], []).append(v)
        return out
    a, b = load(on_paths), load(off_paths)
    def med(xs):
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    print(f"# {field}: affinity ON (n={len(on_paths)} reps) vs OFF (n={len(off_paths)} reps)")
    print("# turn  on_med  off_med  speedup")
    for turn in sorted(set(a) | set(b)):
        if turn not in a or turn not in b:
            continue
        ma, mb = med(a[turn]), med(b[turn])
        sp = f"{mb / ma:.2f}x" if ma else "n/a"
        print(f"{turn:5d}  {ma:7.3f} {mb:8.3f}  {sp:>7}")
    tot_a = sum(med(a[t]) for t in a if t in b)
    tot_b = sum(med(b[t]) for t in b if t in a)
    print(f"# sum-of-medians: on {tot_a:.2f} off {tot_b:.2f} "
          f"({tot_b / tot_a:.2f}x)" if tot_a else "")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "--gate":
        sys.exit(gate(sys.argv[2], sys.argv[3]))
    if sys.argv[1] == "--curve":
        split = sys.argv.index("--")
        sys.exit(curve(sys.argv[2], sys.argv[3:split], sys.argv[split + 1:]))
    sid = None
    if "--session-id" in sys.argv:
        sid = sys.argv[sys.argv.index("--session-id") + 1]
    maxtok = 0
    if "--max-tokens" in sys.argv:
        maxtok = int(sys.argv[sys.argv.index("--max-tokens") + 1])
    if sys.argv[1] == "--replay":
        only = int(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None
        sys.exit(replay(sys.argv[2], int(sys.argv[3]), sys.argv[4], maxtok, sid,
                        "--cold" in sys.argv, only, "--stream" in sys.argv) or 0)
    port = int(sys.argv[1])
    out = sys.argv[2]
    turns = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 25
    tr = None
    if "--transcript" in sys.argv:
        tr = sys.argv[sys.argv.index("--transcript") + 1]
    drive(port, out, turns, sid, "--no-rewrite" not in sys.argv, maxtok or 600, tr,
          "--long" in sys.argv, "--stream" in sys.argv)
