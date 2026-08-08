#!/usr/bin/env python3
"""iso-gap serve A/B (task #91): X alone (O1) vs X with a staggered-depth co-resident Y (O2).

THE ATTRIBUTION FORK (PROGRESS.md §1.5). The engine probe killed H-A: at fixed B on the batched
body, a co-resident at a straddling depth moves ZERO bits of X's logits (5 arms, incl. a 300-step
rung crossing). So if the serve-level receipt (spec-gate REF vs REF_LOAD, byte 1347/2379)
reproduces here, the carrier must be H-B: co-RESIDENCE flips the session's PROGRAM between the
solo family (b1fast m=1 fused trunk / GraphSession replay — mutually bit-identical, gated) and
the batched body (a documented FP-composition gap, decode-batch-gate gate1 config jurisdiction).

ARMS (one server boot per arm — the reuse pool would otherwise serve arm 2's X prime from arm
1's cache; the spec-gate harness precedent):
  O1  default env, X alone (768 greedy tokens).           The solo program family.
  O1R default env, X alone again.                         Determinism control (must == O1).
  O2  default env, Y fired first (greedy, long budget), X fires once Y is mid-generation.
      X shares every tick with Y at a different, moving depth (Y crosses the 512 rung).
      H-B predicts O2 != O1 (config flip + near-tie roulette); the equal-depth serve gate
      cannot see this shape.
  O3S MEMRA_SERVE_B1FAST=0 MEMRA_SERVE_GS=0, X alone.     Solo forced onto the batched body.
  O3L MEMRA_SERVE_B1FAST=0 MEMRA_SERVE_GS=0, Y then X.    Co-resident on the batched body.
      H-B predicts O3L == O3S BYTE-IDENTICAL (the engine probe's serve-level echo). If these
      differ, something beyond the config flip is live and H-A reopens at serve granularity.

q9 is a THINKING model — compare reasoning AND content (the spec-gate vacuous-pass trap).
Greedy throughout. Every response's bytes land in raw/ next to the server log.
"""
import json, os, signal, subprocess, sys, threading, time, urllib.request

Q9 = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
PORT = 8323
BASE = f"http://127.0.0.1:{PORT}"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
X_PROMPT = ("Explain, step by step and in full detail, how a memory allocator for GPU "
            "inference should decide between a pooled arena and direct cudaMalloc calls. "
            "Cover fragmentation, capture-graph address stability, and growth policy.")
Y_PROMPT = ("Write a long, careful essay about the history of instruction set architectures, "
            "from the IBM 360 through RISC-V vector extensions.")
X_TOKENS = 768
Y_TOKENS = 2400   # Y must outlive X's whole stream so B stays >= 2 for every X tick


def post(prompt, max_tokens, timeout=900):
    body = json.dumps({"model": "q9", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0,
                       "stream": False}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def full_text(resp):
    msg = resp["choices"][0]["message"]
    return (msg.get("reasoning") or "") + "\x00<CONTENT>\x00" + (msg.get("content") or "")


def boot(env_extra, log_path):
    env = dict(os.environ)
    env.update({"MEMRA_MODELS": f"q9={Q9}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
                "MEMRA_CTX": "8192", "MEMRA_SERVE_SPEC": "0"})
    env.update(env_extra)
    lf = open(log_path, "wb")
    p = subprocess.Popen(["target/release/memra-server"], stdout=lf,
                         stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
    for _ in range(300):
        if p.poll() is not None:
            raise RuntimeError(f"server died booting; see {log_path}")
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=3).read()
            return p, lf
        except Exception:
            time.sleep(1)
    raise RuntimeError("server never came up")


def stop(p, lf):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    p.wait(timeout=30)
    lf.close()
    time.sleep(1)


def run_arm(name, env_extra, with_y):
    p, lf = boot(env_extra, os.path.join(OUT, f"serveab-{name}-server.log"))
    try:
        y_resp = {}
        if with_y:
            yt = threading.Thread(target=lambda: y_resp.update(post(Y_PROMPT, Y_TOKENS)))
            yt.start()
            time.sleep(4.0)   # Y mid-generation (a few hundred tokens deep) before X fires
        r = post(X_PROMPT, X_TOKENS)
        text = full_text(r)
        if with_y:
            yt.join()
            ytoks = y_resp.get("usage", {}).get("completion_tokens", -1)
            print(f"  [{name}] Y completion_tokens={ytoks} (must be > X's total ticks)")
        toks = r.get("usage", {}).get("completion_tokens", -1)
        with open(os.path.join(OUT, f"serveab-{name}.txt"), "w") as f:
            f.write(text)
        assert len(text) > 200, f"{name}: near-empty stream ({len(text)}B) — vacuous-pass trap"
        print(f"  [{name}] X completion_tokens={toks} bytes={len(text)}")
        return text
    finally:
        stop(p, lf)


def diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def main():
    os.makedirs(OUT, exist_ok=True)
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    v = {}
    print("== arm O1 (default, solo) ==");   o1 = run_arm("O1", {}, False)
    print("== arm O1R (default, solo, determinism control) ==")
    o1r = run_arm("O1R", {}, False)
    v["O1_determinism"] = "PASS (byte-identical)" if o1 == o1r else \
        f"FAIL at byte {diff(o1, o1r)} — solo is not deterministic, arms unreadable"
    print("== arm O2 (default, Y co-resident at staggered/moving depth) ==")
    o2 = run_arm("O2", {}, True)
    d12 = diff(o1, o2)
    v["O2_vs_O1"] = "IDENTICAL (receipt did not reproduce)" if d12 is None else \
        f"DIVERGES at byte {d12} of {len(o1)}/{len(o2)} — the REF/REF_LOAD class reproduced"
    env_h = {"MEMRA_SERVE_B1FAST": "0", "MEMRA_SERVE_GS": "0"}
    print("== arm O3S (batched-body-everywhere, solo) ==")
    o3s = run_arm("O3S", env_h, False)
    print("== arm O3L (batched-body-everywhere, Y co-resident) ==")
    o3l = run_arm("O3L", env_h, True)
    d3 = diff(o3s, o3l)
    v["O3L_vs_O3S"] = "IDENTICAL — H-B confirmed (one program => co-resident invisible)" \
        if d3 is None else \
        f"DIVERGES at byte {d3} — a non-config-flip carrier is live, H-A reopens at serve level"
    print(json.dumps(v, indent=1))
    with open(os.path.join(OUT, "serveab-verdicts.json"), "w") as f:
        json.dump(v, f, indent=1)


if __name__ == "__main__":
    main()
