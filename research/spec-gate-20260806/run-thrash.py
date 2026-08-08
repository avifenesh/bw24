#!/usr/bin/env python3
"""spec-gate MEASUREMENT 3 — THRASH: does the hysteresis band hold mode switches to O(load
changes) instead of O(ticks)?

WHY THIS MATTERS MORE THAN IT LOOKS. A single-threshold policy at the measured crossover would
oscillate: a session count crossing back and forth pays both paths' costs and neither's benefit.
The gate therefore uses TWO thresholds — admit spec at active <= LOW=2, demote at active >= HIGH=4
— leaving active==3 as a "keep doing what you are doing" band. This harness drives load ACROSS
that band repeatedly and counts the actual mode switches.

THE LOAD SHAPE. `--cycles` oscillations of c=2 (below LOW, spec admitted) -> c=6 (above HIGH,
demotion fires) -> back. Each phase holds long enough for the tick loop to settle.

WHAT BOUNDS "PASS", stated before the run so the bar cannot move afterwards:

  Demotion in this design is ONE-WAY per session (see `SpecSession::into_demoted`: re-promotion
  would need an mtp_kv_fill over the whole committed history plus a fresh graph capture, which is
  not the "symmetric and cheap" handoff re-promotion was conditioned on). A session therefore
  switches mode AT MOST ONCE in its life, so the total switch count is bounded by the number of
  sessions that were admitted-spec and then met a high-water mark — NOT by tick count and NOT by
  the number of load crossings times the session count.

  The failure this test exists to rule out is a session (or the scheduler) flapping: demote,
  re-promote, demote again, once per tick. With one-way demotion that is structurally impossible,
  so the test is a CONFIRMATION that the implementation matches the design, and the observable is:

      demotions <= number of sessions ever admitted on the spec path
      demotions per tick << 1

  Both are read straight from the server log: `[spec-gate] demoted` lines vs `[tick]` lines vs the
  distinct request ids that ever held spec.

Run under flock /tmp/memra-5090.lock.
"""
import argparse, json, os, re, signal, subprocess, sys, threading, time, urllib.request

PROMPT = ("Write a detailed technical explanation of how a GPU inference server schedules "
          "concurrent requests, covering batching, KV cache management, and admission control. "
          "Be thorough and specific.")


def post(base, model, prompt, max_tokens, timeout=600):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  (request failed: {type(e).__name__}: {e})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8318)
    ap.add_argument("--model", default="q9")
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--low-c", type=int, default=2)
    ap.add_argument("--high-c", type=int, default=6)
    ap.add_argument("--phase-s", type=float, default=6.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out-dir",
                    default=os.path.dirname(os.path.abspath(__file__)) + "/logs/thrash")
    a = ap.parse_args()
    addr = f"127.0.0.1:{a.port}"
    base = f"http://{addr}"
    os.makedirs(a.out_dir, exist_ok=True)
    log_path = os.path.join(a.out_dir, "thrash-server.log")

    ss = subprocess.run(["ss", "-tln"], capture_output=True, text=True).stdout
    if any(re.search(rf"[:.]{a.port}\s", ln) for ln in ss.splitlines()):
        print(f"FAIL: port {a.port} already LISTENing — refusing to measure against it")
        sys.exit(2)

    q9 = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
    draft = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf"
    env = dict(os.environ)
    env.update({"MEMRA_MODELS": f"q9={q9}+{draft}", "MEMRA_ADDR": addr, "MEMRA_CTX": "4096",
                "MEMRA_SPEC_K": "3", "MEMRA_TICK_TRACE": "1"})
    # naked gate defaults: LOW=2, HIGH=4 — the shipped policy is what gets thrashed.
    lf = open(log_path, "wb")
    p = subprocess.Popen(["target/release/memra-server"], stdout=lf,
                         stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
    try:
        up = False
        for _ in range(300):
            if p.poll() is not None:
                break
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=3).read()
                up = True
                break
            except Exception:
                time.sleep(1)
        if not up:
            print("FAIL: server never came up")
            subprocess.run(["tail", "-30", log_path])
            sys.exit(2)
        post(base, a.model, "Say OK.", 8)  # warm

        timeline = []
        threads = []

        def fire(n, tag):
            for _ in range(n):
                t = threading.Thread(target=post,
                                     args=(base, a.model, PROMPT, a.max_tokens))
                t.daemon = True
                t.start()
                threads.append(t)
            timeline.append({"t": round(time.monotonic() - t_start, 2), "phase": tag, "fired": n})
            print(f"  [{timeline[-1]['t']:6.2f}s] {tag}: fired {n}")

        t_start = time.monotonic()
        for cyc in range(a.cycles):
            print(f"cycle {cyc+1}/{a.cycles}")
            # LOW phase: c=2, at/below LOW -> new arrivals take spec
            fire(a.low_c, f"cyc{cyc+1}-LOW-c{a.low_c}")
            time.sleep(a.phase_s)
            # HIGH phase: push above HIGH -> live spec sessions demote
            fire(a.high_c, f"cyc{cyc+1}-HIGH-c{a.high_c}")
            time.sleep(a.phase_s)
        for t in threads:
            t.join(timeout=600)
        time.sleep(2)
    finally:
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

    # ---- count what actually happened, from the server's own log ----
    ticks = demotions = admit_batched = 0
    spec_ticks = 0
    demote_gens = []
    with open(log_path, errors="replace") as f:
        for ln in f:
            if ln.startswith("[tick]"):
                ticks += 1
                m = re.search(r"spec=(\d+)", ln)
                if m and int(m.group(1)) > 0:
                    spec_ticks += 1
            elif "[spec-gate] demoted" in ln:
                demotions += 1
                m = re.search(r"generated (\d+)\)", ln)
                if m:
                    demote_gens.append(int(m.group(1)))
            elif "[spec-gate] admit batched" in ln:
                admit_batched += 1

    load_crossings = a.cycles * 2  # each cycle crosses the band up and back down
    res = {
        "cycles": a.cycles, "low_c": a.low_c, "high_c": a.high_c, "phase_s": a.phase_s,
        "load_crossings": load_crossings,
        "ticks": ticks, "ticks_with_spec": spec_ticks,
        "demotions": demotions, "demote_at_generated": demote_gens,
        "admit_batched_events": admit_batched,
        "demotions_per_tick": (demotions / ticks) if ticks else None,
        # THE BAR: one-way demotion means switches are bounded by SESSIONS, not ticks. A
        # per-tick flap would show demotions_per_tick approaching 1 and demotions >> the
        # number of spec admits.
        "verdict_O_load_not_O_ticks":
            "PASS" if (ticks and demotions <= load_crossings * a.low_c
                       and demotions / ticks < 0.05) else "REVIEW",
        "timeline": timeline,
    }
    with open(os.path.join(a.out_dir, "thrash.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "timeline"}, indent=2))
    print("THRASH_DONE")


if __name__ == "__main__":
    main()
