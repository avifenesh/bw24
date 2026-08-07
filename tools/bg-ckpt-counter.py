#!/usr/bin/env python3
"""bg-ckpt-counter — the toy checkpoint/resume proof for the dead-darklane runner
(lane/darklane-training, 2026-08-07).

A stand-in for a training-class background job: increments a counter ("training steps"),
checkpoints it to disk, and speaks the runner's checkpoint protocol
(docs/SERVING.md "Dead-darklane background jobs"):

  * SIGUSR1  -> write checkpoint NOW, exit 75 (EX_TEMPFAIL: "preempted, resume me").
  * relaunch -> resume from the checkpoint file (never from zero).
  * exit 0   -> the job is complete (--steps reached); the runner never relaunches it.

Every step also appends a JSONL row (--log) stamping pid/step/monotonic time — the receipt
that resume continued from the checkpoint instead of restarting.

Usage:
  MEMRA_BG_JOB="python3 tools/bg-ckpt-counter.py --ckpt /tmp/ck --steps 1000" memra-server
  (standalone test: python3 tools/bg-ckpt-counter.py --ckpt /tmp/ck --steps 50 --dt 0.01)
"""

import argparse
import json
import os
import signal
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint file (counter state)")
    ap.add_argument("--steps", type=int, default=10_000, help="total steps to 'train'")
    ap.add_argument("--dt", type=float, default=0.05, help="seconds per step")
    ap.add_argument("--log", default=None, help="JSONL step log (default: <ckpt>.jsonl)")
    args = ap.parse_args()
    log_path = args.log or args.ckpt + ".jsonl"

    # resume: the checkpoint file is authoritative. At-least-once semantics — a dirty
    # preempt (SIGKILL past the grace window) repeats the steps since the last write.
    step = 0
    if os.path.exists(args.ckpt):
        with open(args.ckpt) as f:
            step = int(f.read().strip() or 0)
        print(f"[ckpt-counter] pid {os.getpid()}: RESUMED at step {step}", flush=True)
    else:
        print(f"[ckpt-counter] pid {os.getpid()}: fresh start", flush=True)

    def checkpoint() -> None:
        # atomic write: a preempt mid-write must never leave a torn checkpoint.
        tmp = args.ckpt + ".tmp"
        with open(tmp, "w") as f:
            f.write(str(step))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, args.ckpt)

    preempted = False

    def on_usr1(_sig, _frm):
        nonlocal preempted
        preempted = True  # handled at the step boundary — a step is the atomic unit

    signal.signal(signal.SIGUSR1, on_usr1)

    with open(log_path, "a") as log:
        while step < args.steps:
            if preempted:
                checkpoint()
                print(f"[ckpt-counter] pid {os.getpid()}: preempted at step {step}, "
                      f"checkpointed, exit 75", flush=True)
                return 75  # EX_TEMPFAIL — the runner relaunches next valley
            time.sleep(args.dt)  # the "work"
            step += 1
            log.write(json.dumps({"pid": os.getpid(), "step": step,
                                  "t_mono": time.monotonic()}) + "\n")
            log.flush()
            if step % 20 == 0:
                checkpoint()  # periodic, so even SIGKILL loses at most 20 steps

    checkpoint()
    print(f"[ckpt-counter] pid {os.getpid()}: COMPLETE at step {step}, exit 0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
