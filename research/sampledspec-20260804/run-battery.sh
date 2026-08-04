#!/usr/bin/env bash
# Part 3 (dogfood F4 daily-driver verification) — the owner's EXACT serve config, new binary.
#
# Verifies the two F4 fixes on the real daily driver and measures what they cost:
#   A. LOOP:   the actual failing pattern — repeated tool-check prompts, omitted temperature
#              AND omitted seed (what pi/pill sends). Pre-fix this returned byte-identical
#              tool calls forever. PASS = the responses are not all identical.
#   B. GREEDY: explicit temperature 0 still byte-identical across repeats (the gates' contract).
#   C. SEED:   explicit seed reproduces exactly; two omitted-seed runs differ.
#   D. PERF:   tok/s for sampled-spec (the new default) vs greedy-spec vs plain-sampled
#              (MEMRA_SERVE_SPEC=0), same prompt, same server, interleaved.
#
# Protocol (repo evidence discipline): every run's raw JSON is kept; medians state N; the
# server log is tee'd, never piped-and-parsed; failure text is captured verbatim.
# The perf arms are INTERLEAVED per repetition, not blocked, so clock/thermal drift hits all
# three arms equally (the cross-run-comparison law from the H100 lane).
set -uo pipefail

OUT="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8102}"
BASE="http://127.0.0.1:$PORT"
AUTH="Authorization: Bearer aviary-local"
CT="Content-Type: application/json"
REPS="${REPS:-5}"
NGEN="${NGEN:-512}"
RAW="$OUT/raw"
mkdir -p "$RAW"

say() { echo "[$(date +%H:%M:%S)] $*"; }
gpu() { nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,power.draw \
          --format=csv,noheader | tr -d ' '; }

# ---- the owner's actual agentic shape: a tool-enabled repeated version-check turn.
# This is the transcript pattern that looped (~10 identical `npm view version` cycles).
read -r -d '' AGENTIC <<'EOF'
You are a coding agent with shell access. The user asked you to check whether the
project's dependencies are current. You have already run `npm view react version`
three times and each time it printed 19.2.0. The package.json pins react 18.2.0.
Decide what to do next and explain your reasoning in detail, then state the single
concrete next action. Do not repeat a command you have already run.
EOF

# body <temp-json> <seed-json> <maxtok> <prompt>
body() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
t, s, n, p = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
b = {"model": "qwen36-27b", "messages": [{"role": "user", "content": p}], "max_tokens": n}
if t != "omit": b["temperature"] = float(t)
if s != "omit": b["seed"] = int(s)
print(json.dumps(b))
PY
}

# ask <tag> <temp> <seed> <maxtok> <prompt>  -> writes raw json, echoes "sha16 ntok ms"
ask() {
  local tag=$1 t=$2 s=$3 n=$4 p=$5 f="$RAW/$1.json" t0 t1
  t0=$(date +%s%N)
  curl -sS -m 900 "$BASE/v1/chat/completions" -H "$AUTH" -H "$CT" \
       -d "$(body "$t" "$s" "$n" "$p")" > "$f" 2>"$RAW/$tag.err"
  local rc=$?
  t1=$(date +%s%N)
  if [ $rc -ne 0 ]; then
    echo "CURL-FAIL rc=$rc $(head -c 300 "$RAW/$tag.err")"; return 1
  fi
  python3 - "$f" "$(( (t1-t0)/1000000 ))" <<'PY'
import json, sys, hashlib
f, ms = sys.argv[1], int(sys.argv[2])
try:
    d = json.load(open(f))
except Exception as e:
    print("PARSE-FAIL", type(e).__name__, open(f).read()[:300].replace("\n", " ")); sys.exit(0)
if "error" in d:
    print("ERR", json.dumps(d["error"])[:300]); sys.exit(0)
c = d["choices"][0]["message"].get("content") or ""
r = d["choices"][0]["message"].get("reasoning") or ""
txt = r + c
nt = d.get("usage", {}).get("completion_tokens", 0)
print(hashlib.sha256(txt.encode()).hexdigest()[:16], nt, ms,
      f"{nt/(ms/1000):.2f}" if ms and nt else "0")
PY
}

say "=== SERVER HEALTH ==="
curl -sS -m 30 "$BASE/health" -H "$AUTH" | tee "$RAW/health.json"; echo
say "gpu at start: $(gpu)"

# ============================================================ A. THE LOOP
say ""
say "=== A. LOOP TEST — pi's exact request shape (temperature OMITTED, seed OMITTED) ==="
: > "$OUT/loop.txt"
for i in $(seq 1 4); do
  r=$(ask "loop-$i" omit omit 400 "$AGENTIC")
  echo "run$i $r" | tee -a "$OUT/loop.txt"
done
UNIQ=$(awk '{print $2}' "$OUT/loop.txt" | sort -u | wc -l)
say "A VERDICT: $UNIQ distinct outputs of 4 -> $([ "$UNIQ" -gt 1 ] && echo 'PASS (no loop)' || echo 'FAIL (still pinned)')"

# ============================================================ B. GREEDY UNCHANGED
say ""
say "=== B. GREEDY — explicit temperature 0 must stay byte-identical (gate contract) ==="
: > "$OUT/greedy.txt"
for i in $(seq 1 3); do
  r=$(ask "greedy-$i" 0 0 200 "List the first eight prime numbers, comma-separated, and nothing else.")
  echo "run$i $r" | tee -a "$OUT/greedy.txt"
done
GU=$(awk '{print $2}' "$OUT/greedy.txt" | sort -u | wc -l)
say "B VERDICT: $GU distinct of 3 -> $([ "$GU" -eq 1 ] && echo 'PASS (greedy still deterministic)' || echo 'FAIL')"

# ============================================================ C. SEED SEMANTICS
say ""
say "=== C. SEED — explicit reproduces, omitted varies (both at temperature 1.0) ==="
: > "$OUT/seed.txt"
for i in 1 2; do
  r=$(ask "seed-fixed-$i" 1.0 4242 200 "Pick one fruit at random and say only its name.")
  echo "explicit-4242-$i $r" | tee -a "$OUT/seed.txt"
done
for i in 1 2 3 4; do
  r=$(ask "seed-omit-$i" 1.0 omit 200 "Pick one fruit at random and say only its name.")
  echo "omitted-$i $r" | tee -a "$OUT/seed.txt"
done
FU=$(grep '^explicit' "$OUT/seed.txt" | awk '{print $2}' | sort -u | wc -l)
OU=$(grep '^omitted'  "$OUT/seed.txt" | awk '{print $2}' | sort -u | wc -l)
say "C VERDICT: explicit-seed distinct=$FU (want 1), omitted-seed distinct=$OU of 4 (want >1) -> $([ "$FU" -eq 1 ] && [ "$OU" -gt 1 ] && echo PASS || echo FAIL)"

# ============================================================ D. PERF, INTERLEAVED
say ""
say "=== D. PERF — sampled-spec vs greedy-spec, INTERLEAVED x$REPS, ngen=$NGEN ==="
say "    (plain-sampled needs MEMRA_SERVE_SPEC=0 = a server restart; run-noscpec.sh does that arm)"
: > "$OUT/perf.txt"
PROMPT="Write a detailed technical explanation of how speculative decoding works, covering drafting, verification, and acceptance. Be thorough."
for i in $(seq 1 "$REPS"); do
  a=$(ask "perf-sampled-$i" omit omit "$NGEN" "$PROMPT")
  echo "sampled-spec rep$i $a  gpu=$(gpu)" | tee -a "$OUT/perf.txt"
  b=$(ask "perf-greedy-$i" 0 0 "$NGEN" "$PROMPT")
  echo "greedy-spec  rep$i $b  gpu=$(gpu)" | tee -a "$OUT/perf.txt"
done

say ""
say "=== MEDIANS (N=$REPS interleaved, same server, same prompt) ==="
python3 - "$OUT/perf.txt" <<'PY' | tee "$OUT/perf-medians.txt"
import sys, statistics as st, collections
rows = collections.defaultdict(list)
for ln in open(sys.argv[1]):
    p = ln.split()
    if len(p) < 6 or p[2] in ("CURL-FAIL", "ERR", "PARSE-FAIL"): continue
    rows[p[0]].append(float(p[5]))
for k, v in rows.items():
    print(f"{k:14s} N={len(v)} median={st.median(v):.2f} tok/s  min={min(v):.2f} max={max(v):.2f}")
PY
say "gpu at end: $(gpu)"
say "=== BATTERY DONE ==="
