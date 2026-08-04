#!/usr/bin/env bash
# Slice 4b (27B) — CROSS-ARM prefill exactness for the per-block FP8 MMQ kernel.
#
# WHY THIS EXISTS, and why the first 27B "accuracy" leg did not measure exactness:
# run-gen's ST gate prints `verify-prefill argmax=A decode argmax=B logit maxdiff=D`, where the
# comparison is batched prefill vs THAT SAME RUN's own m=1 decode chain. Both arms therefore print
# `maxdiff=0.000e0 MATCH` no matter what the prefill kernel does, because each arm is only ever
# compared against itself. And the MEMRA_NGEN=128 stream that follows runs entirely on the decode
# cache at m=1, below GEMM_M_THRESHOLD=16 — the prefill kernel CANNOT dispatch there. The 27B
# floor/mmq logs differing only in the timing line (`2.429s = 52.70` vs `2.426s = 52.76`) is
# exactly what a never-dispatched kernel would also produce; it is not evidence.
#
# What IS the cross-arm instrument: MEMRA_PREFILL_LOGITS dumps the 512-token batched prefill's
# last-position logit row as raw LE f32. That vector is the only model-level quantity this kernel
# can change on the 27B. Plus `fp8-mmq dispatches: N` right after it, so a zero count is visible
# instead of masquerading as agreement.
#
# SERIALIZATION: the perf battery is running on GPU 0 right now, and a concurrent 27 GB model load
# would perturb its pp medians. This takes the SAME flock the battery uses, so it waits its turn
# and runs alone. GPU 1 is used only because it is the free die once the lock is held.
#
# BUDGET: same 3072 MB as the perf leg, so the exactness verdict and the throughput number
# describe the SAME configuration. At ~355 MiB of e4m3 per layer (measured from the checkpoint
# header: q 62.9 + k 5.2 + v 5.2 + o 31.5 + gate 89.1 + up 89.1 + down 89.1 MB) that budget covers
# a PREFIX of ~8.6 of 64 layers. Partial coverage is stated, not hidden.
set -uo pipefail
cd /root/memra-fp8mmq
OUT=research/fp8st-20260804/mmq/vast27b-exact
mkdir -p "$OUT"
CKPT=/root/models/qwen36-27b-fp8
P512=research/e2e/prompts/pp512.txt
LOCK=/tmp/memra-bench.lock
BIN=/root/target-instr/release/run-gen   # instrumented build, separate target dir: does NOT
                                         # replace the binary the running perf battery is using
BUDGET=3072
DLOG=$OUT/driver.log
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$DLOG"; }

run(){ # arm, env...
  local arm=$1; shift
  log "$arm waiting for $LOCK"
  flock "$LOCK" env CUDA_VISIBLE_DEVICES=1 MEMRA_NGEN=16 MEMRA_PROMPT_FILE=$P512 \
    MEMRA_PREFILL_LOGITS=$OUT/prefill-$arm.f32 "$@" timeout 7200 "$BIN" "$CKPT" \
    > "$OUT/$arm.log" 2>&1
  local rc=$?
  log "$arm rc=$rc | $(grep -a 'fp8-mmq dispatches after prefill' "$OUT/$arm.log" | head -1) | $(grep -a verify-prefill "$OUT/$arm.log" | head -1) | oom=$(grep -ac 'out of memory' "$OUT/$arm.log")"
}

run floor MEMRA_PP_X=0
run mmq   MEMRA_FP8_MMQ=1 MEMRA_PP_FP8_BUDGET_MB=$BUDGET
run arma  MEMRA_FP8_FOLD=1 MEMRA_ST_E4M3=1

log "== cross-arm prefill logit drift =="
python3 - "$OUT" <<'PY' 2>&1 | tee -a "$DLOG"
import struct, sys, math, pathlib
out = pathlib.Path(sys.argv[1])
def rd(p):
    b = p.read_bytes()
    return list(struct.unpack("<%df" % (len(b)//4), b))
ref = rd(out/"prefill-floor.f32")
rms_ref = math.sqrt(sum(v*v for v in ref)/len(ref))
print(f"n={len(ref)}  rms(floor)={rms_ref:.4f}")
order = sorted(range(len(ref)), key=lambda i: -ref[i])
for arm in ("floor", "mmq", "arma"):
    p = out/f"prefill-{arm}.f32"
    if not p.exists():
        print(f"{arm}: MISSING"); continue
    g = rd(p)
    d = [a-b for a, b in zip(ref, g)]
    mx = max(abs(v) for v in d)
    rmsd = math.sqrt(sum(v*v for v in d)/len(d))
    nbits = sum(1 for a, b in zip(ref, g)
                if struct.pack("<f", a) != struct.pack("<f", b))
    go = sorted(range(len(g)), key=lambda i: -g[i])
    print(f"{arm}: max_abs={mx:.4e} rms(diff)={rmsd:.4e} rms_rel={rmsd/rms_ref:.3e} "
          f"differing_f32={nbits}/{len(ref)} top1={'MATCH' if go[0]==order[0] else f'FLIP {order[0]}->{go[0]}'} "
          f"top10_overlap={len(set(go[:10])&set(order[:10]))}/10")
PY
log "27B EXACTNESS DONE"
