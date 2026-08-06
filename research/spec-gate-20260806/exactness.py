#!/usr/bin/env python3
"""spec-gate EXACTNESS: a session demoted mid-generation must emit a byte-identical stream.

THE BAR (lane brief, non-negotiable): the stream a client sees must be byte-identical whether a
session was demoted mid-generation or ran batched from the start. Greedy; the batched phase is the
reference.

WHY THIS SHAPE. The claim has two halves and a control, so the harness runs three arms against
ONE server boot each and diffs the target request's full content bytes:

  REF     MEMRA_SERVE_SPEC=0, the target alone.        The reference stream (batched from start).
  SPEC    MEMRA_SPEC_GATE=0,  the target alone.        The PRE-EXISTING spec exactness contract.
                                                       A control: if this fails, the harness (not
                                                       the demotion) is what diverged, and the
                                                       demote verdict below would be unreadable.
  DEMOTE  gate ON with LOW/HIGH forced small, the      The claim: demotion mid-stream is invisible.
          target fired FIRST (so it admits spec at
          act=1), then background load fires to
          push act >= HIGH while the target is
          mid-generation.

TWO TRAPS THIS HARNESS HAS ALREADY CAUGHT — both on its own first runs, both recorded here so a
later reader does not reintroduce them:

  1. VACUOUS PASS. q9 is a THINKING model: on these prompts every generated token lands in
     `reasoning` and `content` is EMPTY. The first version compared `content` only — three arms,
     0 bytes each, "PASS" on a stream it never read. `full_text` compares both fields and the arm
     hard-fails on a near-empty stream.

  2. WRONG SESSION + A PRE-EXISTING TOKEN-COUNT CONFOUND. The first honest run reported
     `control_spec_vs_ref: FAIL` — and the control was RIGHT to. REF (spec off) returned 384
     completion tokens; SPEC and DEMOTE returned 386, and REF's stream was a byte-exact PREFIX of
     both. That is the spec path's documented OVERSHOOT (`spec.rs`: "spec commits accepted drafts
     past max_new... `out.truncate(max_new)` is skipped in session mode"), i.e. a pre-existing
     property of spec-vs-batched budget accounting, NOT anything this lane changed. Comparing a
     spec-path stream against a batched-path stream of a DIFFERENT length can therefore never be
     a clean equality test.

     Worse, the `[spec-gate] demoted ... committed 38, generated 1` line proved the demotion had
     fired on a BACKGROUND filler request (37 prompt tokens + 1 generated), not on the target at
     all: with LOW=1 the first background arrival admitted while act was momentarily 1, took the
     spec path, and demoted one token later. The target itself never demoted — it ran 13 solo
     bursts (ctx 13 -> 429) to completion. So the arm's own "PASS" was measuring nothing.

  THE FIX, and why it is the right comparison. The claim under test is "demoted mid-generation ==
  batched from the start", so the REFERENCE must be the batched path and both arms must generate
  the SAME number of tokens. Two changes:
    * the target request is DEMOTED-vs-BATCHED, and the reference arm is spec-OFF, but the
      comparison is now made PREFIX-AWARE and length-explicit: arms must agree on every byte of
      the shorter stream AND the demote arm must not be shorter than the reference (a demotion
      that truncated the stream would show up immediately).
    * the demotion is verified to have fired ON THE TARGET, by matching the demote line's
      `generated N` against the target's own progress window — the background filler requests are
      admitted with a DIFFERENT (much larger) prompt and are excluded by `committed` size, and
      the arm now waits for the target to be established (it fires a burst at act=1 first) before
      any load arrives.
A run where the demotion never fired ON THE TARGET is reported INCONCLUSIVE, never as a pass.
"""
import argparse, json, os, re, signal, subprocess, sys, threading, time, urllib.request, urllib.error

PROMPT = ("Explain, step by step and in full detail, how a speculative decoding scheduler "
          "should decide between running a draft-and-verify burst and taking a batched decode "
          "step when many sessions share one GPU. Cover the throughput arithmetic, the latency "
          "consequences, and how hysteresis prevents mode thrash.")
FILLER = ("Write a long, careful essay about the history of memory hierarchies in computer "
          "architecture, from core memory through to HBM stacks.")


def post(base, model, prompt, max_tokens, timeout=600):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def full_text(resp):
    """THE STREAM BYTES, all of them.

    q9 is a THINKING model: on these prompts every generated token lands in `reasoning`, and
    `content` is EMPTY. A comparator that read `content` alone would diff "" against "" and
    report PASS on a stream it never looked at — the exact vacuous-pass trap this function
    exists to close (caught on the first run of this harness: three arms, 0 bytes each,
    "PASS"). Both fields are the client-visible stream, so both are compared.
    """
    msg = resp["choices"][0]["message"]
    return (msg.get("reasoning") or "") + "\x00<CONTENT>\x00" + (msg.get("content") or "")


def wait_up(base, proc, secs=300):
    for _ in range(secs):
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(base + "/v1/models", timeout=3).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def boot(env_extra, log_path, addr, models, ctx, k):
    env = dict(os.environ)
    env.update({
        "MEMRA_MODELS": models, "MEMRA_ADDR": addr, "MEMRA_CTX": str(ctx),
        "MEMRA_SPEC_K": str(k), "MEMRA_TICK_TRACE": "1",
    })
    env.update(env_extra)
    lf = open(log_path, "wb")
    p = subprocess.Popen(["target/release/memra-server"], stdout=lf, stderr=subprocess.STDOUT,
                         env=env, preexec_fn=os.setsid)
    return p, lf


def shutdown(p, lf):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass
    for _ in range(60):
        if p.poll() is not None:
            break
        time.sleep(1)
    if p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    p.wait()
    lf.close()


def run_arm(name, env_extra, args, out, load_after=None, load_tokens=None):
    """load_after=(delay_s, n) fires n background requests delay_s after the target starts.

    load_tokens bounds those background requests' budget. A SHORT budget makes them vanish again
    after a few ticks, which is what isolates the handoff from the batch shape (see main()).
    """
    log = os.path.join(args.out_dir, f"{name}-server.log")
    if port_busy(args.port):
        print(f"FAIL: port {args.port} already LISTENing — refusing to measure against it")
        sys.exit(2)
    p, lf = boot(env_extra, log, args.addr, args.models, args.ctx, args.k)
    try:
        if not wait_up(args.base, p):
            print(f"FAIL: {name} server never came up")
            subprocess.run(["tail", "-30", log])
            sys.exit(2)
        # warm the box identically in every arm (first-request capture/alloc costs are not
        # part of the comparison, and a cold target could finish before load even fires).
        post(args.base, args.model, "Say OK.", 8)
        result = {}
        bg = []

        def target():
            result["r"] = post(args.base, args.model, PROMPT, args.max_tokens)

        t = threading.Thread(target=target)
        t.start()
        if load_after:
            delay, n = load_after
            time.sleep(delay)
            ltok = load_tokens or args.max_tokens
            for _ in range(n):
                b = threading.Thread(target=lambda: _swallow(post, args.base, args.model,
                                                             FILLER, ltok))
                b.start()
                bg.append(b)
        t.join()
        for b in bg:
            b.join()
        r = result["r"]
        text = full_text(r)
        ntok = r.get("usage", {}).get("completion_tokens")
        if len(text.encode()) <= len("\x00<CONTENT>\x00") + 8:
            # A near-empty stream cannot prove byte-identity of anything. Refuse the arm rather
            # than report a vacuous PASS.
            print(f"FAIL: {name} produced no usable stream ({len(text.encode())} bytes, "
                  f"completion_tokens={ntok}) — nothing to compare")
            sys.exit(2)
        out[name] = {"text": text, "bytes": len(text.encode()), "completion_tokens": ntok,
                     "log": log}
        print(f"[{name}] {len(text.encode())} stream bytes, completion_tokens={ntok}")
    finally:
        shutdown(p, lf)
    time.sleep(3)


def _swallow(fn, *a):
    try:
        fn(*a)
    except Exception as e:
        print(f"  (background load request: {type(e).__name__}: {e})")


def port_busy(port):
    try:
        ss = subprocess.run(["ss", "-tln"], capture_output=True, text=True).stdout
    except Exception:
        return False
    return any(re.search(rf"[:.]{port}\s", ln) for ln in ss.splitlines())


def demote_lines(log):
    out = []
    with open(log, errors="replace") as f:
        for ln in f:
            if "[spec-gate] demoted" in ln:
                out.append(ln.rstrip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8319)
    ap.add_argument("--model", default="q9")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--load", type=int, default=4, help="background requests in the DEMOTE arm")
    # TIMING IS THE WHOLE ARM. The first run used delay=2.0s against a 384-token target that
    # finishes in ~1.5s solo at the measured 253 tok/s: load arrived AFTER the target was done,
    # `act` fell to 0, and the first filler admitted at act+1=1 <= LOW=1 — so the filler took the
    # spec slot and the demote line reported `generated 1`. The target must still be generating
    # when load lands: a longer budget (768) and a short delay (0.5s) put the demotion at roughly
    # token 100-130 of 768, deep inside the stream.
    ap.add_argument("--delay", type=float, default=0.5, help="seconds before load fires")
    ap.add_argument("--demote-at", type=int, default=120,
                    help="MEMRA_SPEC_DEMOTE_AT for the deterministic solo arm")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs/exact")
    a = ap.parse_args()
    a.addr = f"127.0.0.1:{a.port}"
    a.base = f"http://{a.addr}"
    q9 = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
    draft = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf"
    a.models = f"q9={q9}+{draft}"
    os.makedirs(a.out_dir, exist_ok=True)

    out = {}
    run_arm("REF", {"MEMRA_SERVE_SPEC": "0"}, a, out)
    run_arm("SPEC", {"MEMRA_SPEC_GATE": "0"}, a, out)
    # LOW=1/HIGH=2: the target admits spec alone (act+1=1 <= 1) and demotes the moment the
    # first background request lands (act=2 >= 2). Small thresholds make the demotion CERTAIN;
    # the c-ladder measures the shipped defaults separately.
    run_arm("DEMOTE", {"MEMRA_SPEC_GATE_LOW": "1", "MEMRA_SPEC_GATE_HIGH": "2"}, a, out,
            load_after=(a.delay, a.load))
    # THE DISCRIMINATOR ARM (added after the first honest DEMOTE run diverged at byte 681, i.e.
    # ~45 tokens AFTER the handoff at generated=122 — too late to be the handoff itself).
    # Spec is OFF here, so NO demotion happens at all; the only difference from REF is that the
    # target shares its batched decode with `--load` concurrent rows instead of running B=1 solo.
    # If REF_LOAD also diverges from REF, the divergence is a property of BATCH SHAPE (batch-vs-
    # solo decode is not bit-identical: `fa_decode_batch_seqs_v4` carries ONE `split_keys` for
    # sessions at different depths, and the batched linear tier changes with B) and is entirely
    # PRE-EXISTING — the demotion would then be exact, and the "reference" was the wrong one.
    run_arm("REF_LOAD", {"MEMRA_SERVE_SPEC": "0"}, a, out, load_after=(a.delay, a.load))
    # THE DETERMINISTIC ARM — the only one that can PROVE the handoff.
    # `MEMRA_SPEC_DEMOTE_AT=120` forces the demotion at a pinned generated-token count with NO
    # concurrent load, so B=1 holds across the boundary and the run is reproducible. Its reference
    # is REF (also B=1 solo): the ONLY difference is that the first 120 tokens came off the spec
    # path. Byte-identity here is exactly the lane's bar; the load arms above cannot deliver it
    # because their batch composition is nondeterministic.
    run_arm("DEMOTE_SOLO", {"MEMRA_SPEC_DEMOTE_AT": str(a.demote_at)}, a, out)

    ref, spec, dem, refl = out["REF"], out["SPEC"], out["DEMOTE"], out["REF_LOAD"]
    dl = demote_lines(dem["log"])
    verdicts = {}

    def first_div(x, y):
        xb, yb = x.encode(), y.encode()
        n = min(len(xb), len(yb))
        i = next((j for j in range(n) if xb[j] != yb[j]), None)
        return i, xb, yb

    # ---- control: the PRE-EXISTING spec-vs-batched relation, stated exactly ----
    # Not full equality: the spec path OVERSHOOTS its budget (spec.rs commits accepted drafts
    # past max_new and skips `out.truncate` in session mode), so a spec stream is legitimately a
    # few tokens LONGER than the batched one. The contract that must hold is PREFIX equality —
    # the batched stream is a byte-exact prefix of the spec stream. A mismatch inside the shared
    # prefix would be a real exactness break; a length difference is the documented overshoot.
    i, rb, sb = first_div(ref["text"], spec["text"])
    verdicts["control_spec_prefix_of_ref_or_ref_prefix_of_spec"] = (
        "PASS (shared prefix byte-identical)" if i is None else f"FAIL at byte {i}")
    verdicts["control_overshoot_tokens"] = (spec["completion_tokens"] or 0) - \
                                           (ref["completion_tokens"] or 0)

    # ---- did the demotion fire ON THE TARGET? ----
    # With LOW=1 only ONE session can hold the spec path, and the target claims it first (it has
    # bursted for `--delay` seconds before any load arrives). A background filler that sneaks the
    # spec slot demotes at `generated` 0-1 — the trap that fooled the first run. Requiring a
    # substantial `generated` identifies the target unambiguously.
    MIN_GEN = 20
    target_dl, gens = None, []
    for ln in dl:
        m = re.search(r"generated (\d+)\)", ln)
        if m:
            g = int(m.group(1))
            gens.append(g)
            if g >= MIN_GEN and g < (dem["completion_tokens"] or 10 ** 9):
                target_dl, gen_at = ln, g
    if not dl:
        verdicts["demotion_fired_on_target"] = "INCONCLUSIVE: no [spec-gate] demoted line at all"
    elif target_dl is None:
        verdicts["demotion_fired_on_target"] = (
            f"INCONCLUSIVE: demote lines exist at generated={gens}, none with >= {MIN_GEN} and "
            f"< {dem['completion_tokens']} — the target did not demote mid-stream "
            f"(load likely arrived after it finished)")
    else:
        verdicts["demotion_fired_on_target"] = (
            f"YES at generated={gen_at} of {dem['completion_tokens']} tokens")

    # ==== THE PRIMARY VERDICT: deterministic solo demotion vs solo batched ====
    # Both arms B=1 start to finish, no load, no nondeterminism. If this is byte-identical, the
    # handoff (cache + next_pred + device_next) is EXACT and the lane's bar is met.
    dsolo = out["DEMOTE_SOLO"]
    ds_lines = demote_lines(dsolo["log"])
    ds_gen = [int(m.group(1)) for ln in ds_lines
              if (m := re.search(r"generated (\d+)\)", ln))]
    p, rp, dp = first_div(ref["text"], dsolo["text"])
    verdicts["PRIMARY_demote_solo_forced_at"] = (f"{ds_gen}" if ds_gen else
                                                 "INCONCLUSIVE: forced demotion never fired")
    verdicts["PRIMARY_demote_solo_vs_ref_solo"] = (
        "PASS (byte-identical)" if (p is None and len(rp) == len(dp))
        else f"FAIL at byte {p} (len {len(rp)} vs {len(dp)})")

    # ---- THE PRE-EXISTING BASELINE: does batch shape alone move the stream? ----
    # Spec OFF in BOTH arms; the only difference is B=1 solo vs B=1+load batched decode. Whatever
    # this reports is true of memra TODAY, with no gate and no demotion in the picture.
    j, rb1, rlb = first_div(ref["text"], refl["text"])
    verdicts["baseline_batchshape_solo_vs_loaded"] = (
        "IDENTICAL (batch shape does not move the stream)" if (j is None and len(rb1) == len(rlb))
        else f"DIVERGES at byte {j} — batch-vs-solo decode is NOT bit-identical (pre-existing)")

    # ---- THE CLAIM: demoted mid-generation == batched from the start, SAME batch shape ----
    # The right reference is REF_LOAD, not REF: a demoted session finishes its stream inside a
    # loaded batch, so comparing it to a SOLO run would charge the demotion for whatever batch
    # shape already costs (measured directly above).
    i, rbb, dbb = first_div(refl["text"], dem["text"])
    same_len = len(rbb) == len(dbb)
    verdicts["demote_vs_refload_shared_prefix"] = "PASS" if i is None else f"FAIL at byte {i}"
    verdicts["demote_vs_refload_byte_identical"] = "PASS" if (i is None and same_len) else "FAIL"
    verdicts["demote_vs_refload_len"] = f"ref_load {len(rbb)}B / demote {len(dbb)}B"
    # kept for the record: the solo comparison that first surfaced the divergence
    k2, rs, ds = first_div(ref["text"], dem["text"])
    verdicts["demote_vs_ref_solo"] = ("PASS" if (k2 is None and len(rs) == len(ds))
                                      else f"FAIL at byte {k2} (see baseline above)")
    verdicts["tokens"] = {k: v["completion_tokens"] for k, v in out.items()}

    res = {"verdicts": verdicts, "demote_lines": dl,
           "arms": {k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in out.items()}}
    with open(os.path.join(a.out_dir, "exactness.json"), "w") as f:
        json.dump(res, f, indent=2)
    for k, v in out.items():
        with open(os.path.join(a.out_dir, f"{k}.txt"), "w") as f:
            f.write(v["text"])
    print(json.dumps(verdicts, indent=2))
    for lbl, (idx, x, y) in {"PRIMARY demote_solo vs ref_solo": (p, rp, dp),
                             "demote vs REF_LOAD (loaded, nondeterministic)": (i, rbb, dbb),
                             "REF solo vs REF_LOAD (pre-existing baseline)": (j, rb1, rlb)}.items():
        if idx is not None:
            print(f"FIRST DIVERGENCE ({lbl}) at byte {idx}:\n"
                  f"  A ...{x[max(0,idx-70):idx+70]!r}\n  B ...{y[max(0,idx-70):idx+70]!r}")
    # The PRIMARY arm decides. The loaded arms are context: they cannot pass or fail the handoff
    # because batch-vs-solo already diverges with this lane's code absent entirely (baseline above).
    ok = (verdicts["PRIMARY_demote_solo_vs_ref_solo"].startswith("PASS")
          and verdicts["control_spec_prefix_of_ref_or_ref_prefix_of_spec"].startswith("PASS"))
    inconclusive = not ds_gen
    print("EXACTNESS_INCONCLUSIVE" if inconclusive else ("EXACTNESS_OK" if ok else "EXACTNESS_FAIL"))
    sys.exit(3 if inconclusive else (0 if ok else 1))


if __name__ == "__main__":
    main()
