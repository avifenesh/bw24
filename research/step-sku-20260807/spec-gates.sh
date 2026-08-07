#!/usr/bin/env bash
# step-sku item 3: full run-spec K=1..8 self-consistency WITH the MTP drafter over PP-2,
# plus an ACCEPTANCE-DELTA assertion against the pinned baseline.
#
# Baseline: research/step37-p2-20260806/raw/mtp-draft-PASS-20260806T215132Z.log (the fix
# commit's own gate), independently reproduced digit-for-digit by lane/step35-chunkfix S1
# (raw/spec35-20260807T005750Z.log). Same short prompt, same n=32, same artifact.
#
# The delta assertion is the f8f4-flip lesson made mechanical: self-consistency stays green
# under acceptance regressions (the verify arbitrates), so the gate must FAIL LOUDLY when
# accepted counts move against the pinned baseline, not just when tokens diverge. Tokenizer
# and serve changes landed since the baseline (tilde \p{S} fix, reasoning_effort surface);
# neither touches this prompt's ids or this path — the assertion PROVES that instead of
# assuming it.
#
# Run ON THE BOX: bash spec-gates.sh   (takes the flock itself)
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/tokparity-memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
RAW=$HOME/step37/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/specgate-$TS.log
P="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."

{
echo "=== step-sku run-spec K=1..8 + acceptance-delta $TS commit=$(git -C ~/tokparity-memra rev-parse --short HEAD 2>/dev/null || echo rsync-tree)"
(
  flock -w 3600 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_MTP_DRAFT="$D" MEMRA_NGEN=32 \
    MEMRA_PROMPT="$P" timeout 3600 ./target/release/run-spec "$M"
  echo "run-spec exit=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== specgate rc=$?"
} > "$LOG" 2>&1

# ---- acceptance-delta assertion (parses the log it just wrote) ----
python3 - "$LOG" <<'PY'
import re, sys
log = open(sys.argv[1]).read()

# pinned baseline: (accepted, drafted) per K from mtp-draft-PASS-20260806T215132Z.log
BASE = {1: (14, 18), 2: (15, 34), 3: (15, 51), 4: (15, 68),
        5: (15, 85), 6: (15, 102), 7: (15, 119), 8: (15, 136)}
TOL_PP = 5.0  # percentage-point tolerance before the gate goes red

rows = re.findall(r"\[generate_spec K=(\d+)\].*?\n\s*acceptance: (\d+)/(\d+) = ([\d.]+)%"
                  r"\s+self-consistency: (\S+)", log)
if len(rows) != 8:
    print(f"DELTA-GATE FAIL: expected 8 K rows, parsed {len(rows)}")
    sys.exit(1)
if "=== SELF-CONSISTENCY PASS ===" not in log:
    print("DELTA-GATE FAIL: self-consistency battery did not PASS")
    sys.exit(1)

fail = False
print(f"{'K':>2} {'acc/draft':>10} {'rate':>7} {'baseline':>10} {'delta_pp':>9}  verdict")
for k_s, acc_s, dr_s, rate_s, sc in rows:
    k, acc, dr = int(k_s), int(acc_s), int(dr_s)
    b_acc, b_dr = BASE[k]
    rate = 100.0 * acc / dr
    b_rate = 100.0 * b_acc / b_dr
    delta = rate - b_rate
    ok = abs(delta) <= TOL_PP and sc.startswith("PASS")
    fail |= not ok
    print(f"{k:>2} {f'{acc}/{dr}':>10} {rate:>6.1f}% {f'{b_acc}/{b_dr}':>10} {delta:>+8.1f}pp"
          f"  {'OK' if ok else 'FAIL'}")
print(f"ACCEPTANCE-DELTA GATE: {'FAIL' if fail else 'PASS'} (tolerance +/-{TOL_PP}pp vs "
      f"mtp-draft-PASS-20260806T215132Z baseline)")
sys.exit(1 if fail else 0)
PY
RC=$?
echo "delta-gate rc=$RC" >> "$LOG"
echo "LOG=$LOG rc=$RC"
