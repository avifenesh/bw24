#!/usr/bin/env bash
# accept-gate — the in-battery ACCEPTANCE-DELTA + LONG-TEXT assertion for served spec configs.
#
#   tools/accept-gate.sh [--full] [--cells a,b] [--control] [--teeth]
#   tools/accept-gate.sh --pin [--full] [--cells a,b]
#
# WHY THIS GATE EXISTS (research/f8f4-flip-20260806/MATRIX.md, merged c506317e).
# A kernel-arm change (MEMRA_MMQ_F8F4=1) moved SERVED GREEDY TEXT in 4 of 6 regime cells at
# temperature 0 and moved spec acceptance by up to -9.5pp — and the whole battery stayed green:
#
#   1. The token goldens are 20 tokens. Both of that lane's greedy divergences landed at
#      generated index 22 and 38, i.e. PAST the pins. A 20-token golden is structurally
#      incapable of seeing a change that starts at token 22.
#   2. Worse, `fast-gate --refresh-goldens` after such a change would have silently RE-PINNED
#      the new arm's tokens — the gate would have absorbed the regression and then defended it.
#   3. Acceptance is invisible to every exactness gate BY CONSTRUCTION. Each arm is internally
#      self-consistent (run-spec PASSes both arms: spec output == plain output within an arm),
#      each arm reproduces its own goldens, and argmax MATCHes. Nothing in the battery compares
#      HOW MANY DRAFT TOKENS WERE ACCEPTED — which is spec throughput, i.e. the product.
#
# So this gate asserts the two properties the battery was blind to, at the serve config:
#
#   A. ACCEPTANCE COUNTS, EXACT. (rounds, drafted, accepted) must integer-match a pinned
#      reference. Not the rate, and not a tolerance band: at temperature 0 drafting is
#      deterministic, so these are hard integers, and a band would just relabel the blind spot
#      at a coarser grain. `accepted/drafted` is derived and reported for humans only.
#   B. THE FULL GENERATED TEXT, to ngen >= 100 tokens (default 128), sha256-pinned — 6.4x past
#      the 20-token window, and covering the exact indices (22, 38) the receipted divergences
#      landed at. Mismatch quotes the first differing character with context.
#
# WHAT MAKES A SINGLE-SHOT READ LEGITIMATE (inherited from the seed harness,
# research/f8f4-flip-20260806/tools/regime_accept_ab.sh): temperature 0 => drafting is
# deterministic => acceptance is a hard number, not a sample. That lane's OFF/ON/OFF2 protocol
# measured it: the two independent OFF passes (separate server boots, separate processes) were
# byte-identical in ALL SIX cells — same rounds/drafted/accepted, same text sha256. Acceptance
# here is a property of (build x model x drafter x prompt x K), not of the box's mood, so one
# pass is evidence. `--control` re-runs each cell in a SECOND server boot and asserts
# self-agreement, which is that OFF-vs-OFF2 drift bound available on demand.
#
# THE LAW THIS GATE ENCODES (the lane's law-shaped finding): acceptance sign follows
# (model x drafter x prompt), NOT the model. It INVERTED between the GGUF's embedded MTP head
# and the production regime drafter on the same two models the same day (q27 -1.9pp bare /
# +0.45pp regime; q9 +1.6pp bare / -3.05pp regime, worst cell -9.5pp). Therefore every cell
# here attaches the artifact's REAL PRODUCTION DRAFTER via MEMRA_MODELS "+draft" (which replaces
# the embedded head at load, worker.rs load_draft) and runs through the SERVER, not run-spec. A
# bare-head acceptance number is not evidence about a served config, so this gate does not
# collect one.
#
# CELLS: tools/fast-gate/accept-cells.tsv (registry, same shape as fast-gate/models.tsv).
# REFS:  tools/fast-gate/accept-refs/<cell>.ref  (counts + config fingerprint + text sha256)
#        tools/fast-gate/accept-refs/<cell>.text (the full pinned completion, for diffing)
# Two files per cell mirrors the goldens/<id>.tokens + <id>.perf convention.
#
# ---- --pin: THE SILENT-RE-PIN TRAP, CLOSED ----
# The receipted failure mode is not "someone forgot to run the gate". It is: run the kernel
# change, watch the battery go green, refresh the references, commit. The reference now encodes
# the regression and the gate defends it forever. `--pin` therefore REFUSES to run when
# crates/ is dirty (staged or unstaged) — references may only be minted from committed engine
# code, so every reference is attributable to a SHA someone can review. There is NO --force
# override for the crates/ check: an escape hatch on this specific check is the entire bug.
# (Dirt outside crates/ — docs, research/, tools/ — is fine and is only reported.)
# `--pin` also refuses to run when MEMRA_MMQ_F8F4 or another arm env is set, because pinning an
# opt-in arm's numbers as the default reference is the same trap wearing a different hat.
#
# ---- TEETH ----
# A gate only ever observed PASSING proves nothing. `--teeth` sets MEMRA_MMQ_F8F4=1 — the
# merged opt-in arm that provably moves q27's p1 cell (drafted 126->129, rounds 42->43, text
# sha fddd...->33d6...) — and INVERTS the verdict: the run MUST FAIL. If a known-moving arm
# still passes, this gate is not measuring acceptance and its green is worthless.
#
# NOT flock-wrapped: callers own the GPU lock (fast-gate's lockrun, local-ci's window
# discipline) — self-locking here would self-deadlock under fast-gate. Standalone runs should
# be wrapped: flock /tmp/gpu5090.lock tools/accept-gate.sh
#
# SKIPs cleanly (exit 0 + a "accept-gate: SKIP" line, fast-gate's verdict-word contract) when a
# model or drafter artifact is absent.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

CELLS_TSV=tools/fast-gate/accept-cells.tsv
REFS=tools/fast-gate/accept-refs
PORT="${MEMRA_ACCEPT_PORT:-8317}"
KEY=acceptgate
LOGDIR="${MEMRA_ACCEPT_LOGDIR:-/tmp/accept-gate-$(date +%Y%m%d-%H%M%S)}"

FULL=0; PIN=0; CONTROL=0; TEETH=0; CELLS_OVERRIDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --full)    FULL=1; shift ;;
        --cells)   CELLS_OVERRIDE="$2"; shift 2 ;;
        --pin)     PIN=1; shift ;;
        --control) CONTROL=1; shift ;;
        --teeth)   TEETH=1; export MEMRA_MMQ_F8F4=1; shift ;;
        *) echo "accept-gate: unknown arg $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$LOGDIR"

[ -f "$CELLS_TSV" ] || { echo "accept-gate: FAIL (missing cell registry $CELLS_TSV)"; exit 1; }
[ -x target/release/memra-server ] || cargo build --release -p memra-server || exit 1

cell_field() { awk -F'\t' -v id="$1" -v f="$2" '$0 !~ /^#/ && $1 == id { print $f; exit }' "$CELLS_TSV"; }
# col7 = smoke|full tier. The DEFAULT arm runs only `smoke` cells (battery time budget ~3 min);
# --full runs every registered cell.
select_cells() {
    if [ -n "$CELLS_OVERRIDE" ]; then echo "${CELLS_OVERRIDE//,/ }"; return; fi
    if [ "$FULL" = 1 ]; then awk -F'\t' '$0 !~ /^#/ && NF >= 7 { print $1 }' "$CELLS_TSV"
    else awk -F'\t' '$0 !~ /^#/ && NF >= 7 && $7 == "smoke" { print $1 }' "$CELLS_TSV"; fi
}
SEL=$(select_cells | tr '\n' ' ')
[ -n "${SEL// /}" ] || { echo "accept-gate: FAIL (no cells selected)"; exit 1; }

# ---------- --pin guards (the silent-re-pin trap) ----------
if [ "$PIN" = 1 ]; then
    if ! git diff --quiet -- crates/ || ! git diff --cached --quiet -- crates/; then
        echo "accept-gate: REFUSING --pin — crates/ is dirty."
        echo "  Acceptance references may only be minted from COMMITTED engine code. Pinning"
        echo "  references while a kernel change is uncommitted is EXACTLY the receipted failure"
        echo "  mode (research/f8f4-flip-20260806): the new arm's numbers become the reference and"
        echo "  the gate then defends the regression. There is deliberately no --force here."
        git status --porcelain -- crates/ | sed 's/^/    /'
        exit 2
    fi
    for v in MEMRA_MMQ_F8F4 MEMRA_MMQ_F8F4_PLAIN MEMRA_MMQ_FP8BLK_PLAIN MEMRA_FAST MEMRA_PRIME_F32CHUNK0; do
        if [ -n "${!v:-}" ]; then
            echo "accept-gate: REFUSING --pin — $v=${!v} is set in the environment."
            echo "  References must describe the NAKED default build (flags doctrine: winners are"
            echo "  defaults). Pinning an opt-in arm's acceptance as the default reference is the"
            echo "  same silent-re-pin trap wearing a different hat."
            exit 2
        fi
    done
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "accept-gate: note — tree is dirty OUTSIDE crates/ (allowed; engine code is clean):"
        git status --porcelain | grep -v '^.. crates/' | head -8 | sed 's/^/    /'
    fi
    echo "accept-gate: --pin at $(git rev-parse --short HEAD) (crates/ clean)"
fi

# ---------- server lifecycle (one boot per model+draft+K+ctx group) ----------
# PRE-FLIGHT PORT GUARD (found the hard way on this lane's first pin run): the rig's idle
# llama-server happened to hold 8181, so /health answered INSTANTLY ("up in 0s") from a foreign
# process, our own server was never waited for, and all six cells failed with HTTP 500 from a
# server that does not speak this API. Had that foreign process instead answered 200 with a
# plausible body, the gate would have measured SOMEONE ELSE'S MODEL and pinned it. An occupied
# port is therefore a hard abort, never a wait: we cannot prove the responder is ours.
port_busy() { ss -tln 2>/dev/null | grep -q "[:.]$PORT "; }
if port_busy; then
    echo "accept-gate: FAIL — port $PORT is already LISTENing before we start a server."
    ss -tlnp 2>/dev/null | grep "[:.]$PORT " | sed 's/^/    /'
    echo "  Refusing to run: a foreign responder on our port can answer /health and be measured"
    echo "  as if it were the model under test (this lane's first pin run hit exactly that)."
    echo "  Free the port or set MEMRA_ACCEPT_PORT=<free port>."
    exit 1
fi
SPID=""
serve_up() { # serve_up <model> <draft> <k> <ctx> <tag>
    local model=$1 draft=$2 k=$3 ctx=$4 tag=$5
    local spec="m=$model"
    [ -n "$draft" ] && [ "$draft" != "-" ] && spec="m=$model+$draft"
    MEMRA_MODELS="$spec" MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_API_KEY=$KEY \
    MEMRA_CTX="$ctx" MEMRA_MAX_SESSIONS=1 MEMRA_REUSE_POOL=1 MEMRA_PRIME_CHUNK=2048 \
    MEMRA_SPEC_K="$k" \
        target/release/memra-server > "$LOGDIR/server-$tag.log" 2>&1 &
    SPID=$!
    local i
    for i in $(seq 1 180); do
        if curl -sf -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            # Belt and braces on top of the pre-flight guard: the healthy responder must BE our
            # child. A racing process that grabbed the port after the pre-flight check would
            # otherwise get measured as the model under test.
            if ! ss -tlnp 2>/dev/null | grep "[:.]$PORT " | grep -q "pid=$SPID,"; then
                echo "  FAIL: port $PORT answers /health but is NOT owned by our server (pid $SPID)"
                ss -tlnp 2>/dev/null | grep "[:.]$PORT " | sed 's/^/      /'
                return 1
            fi
            return 0
        fi
        kill -0 "$SPID" 2>/dev/null || { echo "  server died during boot (log $LOGDIR/server-$tag.log)"; return 1; }
        sleep 2
    done
    echo "  server never became healthy (log $LOGDIR/server-$tag.log)"; return 1
}
serve_down() {
    [ -n "$SPID" ] || return 0
    kill "$SPID" 2>/dev/null
    local i; for i in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
    SPID=""; sleep 2
}
trap serve_down EXIT

# ---------- one cell: request -> measured json ----------
# Writes $LOGDIR/<cell>[.<suffix>].json = {"spec":{...},"completion_tokens":N,"text":"..."}.
# Raw-prompt /v1/completions is the pi contract these servers serve (the client renders the
# chat template), so no server-side template enters the comparison.
measure_cell() { # measure_cell <cell> <prompt> <ngen> <outjson>
    local pfile=$2 ngen=$3 out=$4
    python3 - "$pfile" "$ngen" "http://127.0.0.1:$PORT/v1/completions" "$KEY" "$out" <<'PY'
import hashlib, json, sys, urllib.request
pfile, ngen, url, key, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
body = {"model": "m", "prompt": open(pfile).read(),
        "max_tokens": ngen, "temperature": 0, "stream": False}
req = urllib.request.Request(url, data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + key})
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
except Exception as e:
    print(f"REQUEST-FAIL {type(e).__name__}: {e}"); sys.exit(1)
u = d.get("usage", {}) or {}
sp = u.get("spec")
if not sp:
    # No spec block = the request never ran spec rounds (drafter not attached, spec disabled,
    # or it fell back). An acceptance gate with no acceptance telemetry must not report PASS.
    print("NO-SPEC-USAGE (request ran no spec rounds — drafter attached? spec enabled?)")
    sys.exit(2)
text = d["choices"][0]["text"]
json.dump({"spec": {k: sp.get(k) for k in ("rounds", "drafted", "accepted")},
           "acceptance_rate": sp.get("acceptance_rate"),
           "completion_tokens": u.get("completion_tokens"),
           "prompt_tokens": u.get("prompt_tokens"),
           "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "text": text}, open(out, "w"))
print("OK rounds={rounds} drafted={drafted} accepted={accepted}".format(**sp))
PY
}

# ---------- compare measured vs pinned ----------
compare_cell() { # compare_cell <cell> <measured.json> <cfg-fingerprint>
    python3 - "$1" "$2" "$3" "$REFS" <<'PY'
import hashlib, json, sys
cell, mfile, cfgfp, refs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
m = json.load(open(mfile))
try:
    ref = dict(l.split("\t", 1) for l in open(f"{refs}/{cell}.ref")
               if l.strip() and not l.startswith("#"))
    ref = {k: v.rstrip("\n") for k, v in ref.items()}
    rtext = open(f"{refs}/{cell}.text").read()
except FileNotFoundError as e:
    print(f"  {cell}: NO REFERENCE PINNED ({e.filename}) — run --pin at a battery-green point")
    sys.exit(3)

# A reference is only meaningful for the config it was minted under. If the prompt file, K,
# ngen, model or drafter moved, the pinned integers describe a different experiment — say so
# loudly instead of reporting a divergence that is really a config edit.
if ref.get("cfg_fp") != cfgfp:
    print(f"  {cell}: FAIL — REFERENCE CONFIG MISMATCH (this cell's config changed since pinning)")
    print(f"      ref cfg: {ref.get('cfg')}")
    print(f"      now:     {ref.get('cfg_now', '(see accept-cells.tsv)')}")
    print("      The pinned counts describe a different experiment. Re-pin deliberately"
          " (--pin) after reviewing WHY the cell config moved.")
    sys.exit(1)

fails = []
rs = json.loads(ref["spec"])
ms = m["spec"]
for k in ("rounds", "drafted", "accepted"):
    if rs.get(k) != ms.get(k):
        fails.append(f"spec.{k}: pinned {rs.get(k)} -> got {ms.get(k)}")
if ref.get("text_sha256") != m["text_sha256"]:
    fails.append(f"text sha256: pinned {ref['text_sha256'][:16]} -> got {m['text_sha256'][:16]}")
if int(ref.get("completion_tokens", -1)) != int(m["completion_tokens"] or -1):
    fails.append(f"completion_tokens: pinned {ref.get('completion_tokens')} -> got {m['completion_tokens']}")

def rate(s):
    return s["accepted"] / s["drafted"] if s.get("drafted") else 0.0
if not fails:
    print(f"  {cell}: PASS (rounds={ms['rounds']} drafted={ms['drafted']} accepted={ms['accepted']}"
          f" accept={rate(ms):.4f}, {int(m['completion_tokens'])} tok text sha-identical)")
    sys.exit(0)

print(f"  {cell}: FAIL — {len(fails)} assertion(s):")
for f in fails:
    print(f"      {f}")
d_pp = (rate(ms) - rate(rs)) * 100
print(f"      acceptance {rate(rs):.4f} -> {rate(ms):.4f} ({d_pp:+.2f}pp)   <- clock-independent;"
      " a ratio, so this is real, not drift")
mtext = m["text"]
if mtext != rtext:
    # Quote the divergence (failure causes are quoted, never inferred) and report WHERE it
    # starts relative to the 20-token golden window this gate exists to see past.
    i = next((j for j in range(min(len(rtext), len(mtext))) if rtext[j] != mtext[j]),
             min(len(rtext), len(mtext)))
    approx_tok = len(rtext[:i].split())
    print(f"      TEXT DIVERGES at char {i} (~word {approx_tok}; pinned len {len(rtext)},"
          f" got {len(mtext)})")
    print(f"        common:  ...{rtext[max(0,i-60):i]!r}")
    print(f"        pinned:  {rtext[i:i+70]!r}")
    print(f"        got:     {mtext[i:i+70]!r}")
    if approx_tok >= 15:
        print("        NOTE: this divergence is at/past the 20-token golden window — the token"
              " goldens are BLIND to it. That blindness is why this gate exists.")
sys.exit(1)
PY
}

pin_cell() { # pin_cell <cell> <measured.json> <cfg-fingerprint> <cfg-human>
    python3 - "$1" "$2" "$3" "$4" "$REFS" "$(git rev-parse --short HEAD)" \
             "$(date -u +%Y-%m-%dT%H:%M:%SZ)" <<'PY'
import json, sys
cell, mfile, cfgfp, cfg, refs, sha, ts = sys.argv[1:8]
m = json.load(open(mfile))
with open(f"{refs}/{cell}.ref", "w") as f:
    f.write(f"# accept-ref {cell} @ {sha} {ts}\n")
    f.write("# ACCEPTANCE + LONG-TEXT reference for a SERVED spec config (tools/accept-gate.sh).\n")
    f.write("# Minted only from committed crates/ on the naked default build. Counts are EXACT\n")
    f.write("# assertions: temperature 0 => deterministic drafting => hard integers.\n")
    f.write(f"cfg\t{cfg}\n")
    f.write(f"cfg_fp\t{cfgfp}\n")
    f.write("spec\t" + json.dumps(m["spec"], sort_keys=True) + "\n")
    f.write(f"acceptance_rate\t{m['acceptance_rate']}\n")
    f.write(f"prompt_tokens\t{m['prompt_tokens']}\n")
    f.write(f"completion_tokens\t{m['completion_tokens']}\n")
    f.write(f"text_sha256\t{m['text_sha256']}\n")
open(f"{refs}/{cell}.text", "w").write(m["text"])
s = m["spec"]
print(f"  {cell}: pinned (rounds={s['rounds']} drafted={s['drafted']} accepted={s['accepted']},"
      f" {m['completion_tokens']} tok, sha {m['text_sha256'][:16]})")
PY
}

# ---------- run ----------
echo "== accept-gate: served-spec acceptance + long-text assertion =="
echo "   cells: ${SEL% } $([ "$FULL" = 1 ] && echo '(--full matrix)' || echo '(smoke tier; --full for the matrix)')"
echo "   arm env: MEMRA_MMQ_F8F4=${MEMRA_MMQ_F8F4:-unset}$([ "$TEETH" = 1 ] && echo '  [--teeth: verdict INVERTED, must FAIL]')"
echo "   logs: $LOGDIR"
mkdir -p "$REFS"

FAILS=0; PASSES=0; SKIPS=0; NOREF=0
# Group cells by (model,draft,k,ctx) so a 16G artifact is loaded once for all its prompts.
declare -A CELLGRP
for c in $SEL; do
    m=$(cell_field "$c" 2); d=$(cell_field "$c" 3); k=$(cell_field "$c" 5); x=$(cell_field "$c" 6)
    [ -n "$m" ] || { echo "  $c: UNKNOWN cell id (not in $CELLS_TSV)"; FAILS=$((FAILS+1)); continue; }
    key="$m|$d|$k|$x"
    CELLGRP["$key"]="${CELLGRP[$key]:-} $c"
done

for key in "${!CELLGRP[@]}"; do
    IFS='|' read -r MODEL DRAFT K CTX <<< "$key"
    group="${CELLGRP[$key]}"
    if [ ! -f "$MODEL" ]; then
        echo "accept-gate: SKIP (no model at $MODEL) — cells:$group"
        SKIPS=$((SKIPS + $(echo "$group" | wc -w))); continue
    fi
    if [ -n "$DRAFT" ] && [ "$DRAFT" != "-" ] && [ ! -f "$DRAFT" ]; then
        # A missing drafter must NOT silently degrade into a no-spec run that reports PASS —
        # this gate is about the PRODUCTION drafter (the acceptance sign follows the drafter).
        echo "accept-gate: SKIP (no drafter at $DRAFT) — cells:$group"
        SKIPS=$((SKIPS + $(echo "$group" | wc -w))); continue
    fi
    tag=$(basename "$MODEL" .gguf | cut -c1-24)
    echo "-- server: $(basename "$MODEL") + $(basename "${DRAFT:--}") K=$K ctx=$CTX"
    t0=$(date +%s)
    serve_up "$MODEL" "$DRAFT" "$K" "$CTX" "$tag" || {
        echo "accept-gate: FAIL (server boot, cells:$group)"; FAILS=$((FAILS+1)); serve_down; continue; }
    echo "   up in $(( $(date +%s) - t0 ))s"

    for c in $group; do
        # cols: 1 id, 2 model, 3 draft, 4 prompt, 5 k, 6 ctx, 7 tier, 8 ngen
        PROMPT=$(cell_field "$c" 4); NGEN=$(cell_field "$c" 8)
        [ -f "$PROMPT" ] || { echo "  $c: FAIL (missing prompt $PROMPT)"; FAILS=$((FAILS+1)); continue; }
        # Config fingerprint: any change to what the cell MEASURES invalidates its reference.
        CFG="model=$(basename "$MODEL") draft=$(basename "${DRAFT:--}") k=$K ngen=$NGEN prompt=$PROMPT"
        CFGFP=$(printf '%s|%s' "$CFG" "$(sha256sum "$PROMPT" | cut -d' ' -f1)" | sha256sum | cut -d' ' -f1)
        MJ="$LOGDIR/$c.json"
        out=$(measure_cell "$c" "$PROMPT" "$NGEN" "$MJ" 2>&1); rc=$?
        echo "$out" > "$LOGDIR/$c.measure.log"
        if [ $rc -ne 0 ]; then
            echo "  $c: FAIL (measurement) — $(echo "$out" | tail -1)"; FAILS=$((FAILS+1)); continue
        fi
        if [ "$PIN" = 1 ]; then
            pin_cell "$c" "$MJ" "$CFGFP" "$CFG"; PASSES=$((PASSES+1)); continue
        fi
        compare_cell "$c" "$MJ" "$CFGFP"; crc=$?
        case $crc in
            0) PASSES=$((PASSES+1)) ;;
            3) NOREF=$((NOREF+1)) ;;
            *) FAILS=$((FAILS+1)) ;;
        esac
    done
    serve_down

    # --control: the OFF-vs-OFF2 drift bound from the seed harness — re-measure every cell of
    # this group in a SECOND, independent server boot and require byte-identity. This is what
    # licenses the single-shot read; if it ever disagrees, acceptance here is NOT deterministic
    # and no single-pass verdict (pass OR fail) on this rig may be believed.
    if [ "$CONTROL" = 1 ] && [ "$PIN" = 0 ]; then
        echo "-- control boot (independent process; must reproduce byte-identically)"
        serve_up "$MODEL" "$DRAFT" "$K" "$CTX" "$tag-ctl" || {
            echo "  control: FAIL (server boot)"; FAILS=$((FAILS+1)); serve_down; continue; }
        for c in $group; do
            PROMPT=$(cell_field "$c" 4); NGEN=$(cell_field "$c" 8)
            [ -f "$PROMPT" ] || continue
            MJ2="$LOGDIR/$c.ctl.json"
            measure_cell "$c" "$PROMPT" "$NGEN" "$MJ2" > "$LOGDIR/$c.ctl.measure.log" 2>&1 || {
                echo "  $c control: FAIL (measurement)"; FAILS=$((FAILS+1)); continue; }
            if python3 -c 'import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
sys.exit(0 if (a["spec"],a["text_sha256"])==(b["spec"],b["text_sha256"]) else 1)' "$MJ" "$MJ2"; then
                echo "  $c control: PASS (boot 1 == boot 2 — single-shot read is valid here)"
            else
                echo "  $c control: FAIL — TWO IDENTICAL-CONFIG BOOTS DISAGREE."
                echo "      Acceptance is not deterministic on this rig/build, so NO single-pass"
                echo "      verdict from this gate is trustworthy. Root-cause before reading any"
                echo "      pass/fail above."
                python3 -c 'import json,sys
for n,p in (("boot1",sys.argv[1]),("boot2",sys.argv[2])):
    d=json.load(open(p)); print(f"        {n}: {d[\"spec\"]} sha={d[\"text_sha256\"][:16]}")' "$MJ" "$MJ2"
                FAILS=$((FAILS+1))
            fi
        done
        serve_down
    fi
done

echo
if [ "$PIN" = 1 ]; then
    echo "accept-gate: pinned $PASSES cell(s), $FAILS fail, $SKIPS skip."
    echo "  References are only valid at FULL-BATTERY GREEN points — commit them with the"
    echo "  battery log that justified them, never alongside an unproven engine change."
    [ "$FAILS" -eq 0 ] || exit 1
    exit 0
fi

# SKIP contract (fast-gate reads the verdict WORD, not just the exit code): if every cell
# skipped for missing artifacts, say SKIP — a 0-cell run must never read as a pass.
if [ "$PASSES" -eq 0 ] && [ "$FAILS" -eq 0 ] && [ "$NOREF" -eq 0 ] && [ "$SKIPS" -gt 0 ]; then
    echo "accept-gate: SKIP (no cell artifacts present on this rig: $SKIPS cell(s))"
    exit 0
fi

if [ "$TEETH" = 1 ]; then
    if [ "$FAILS" -gt 0 ]; then
        echo "accept-gate: TEETH OK ($FAILS cell(s) FAILED under MEMRA_MMQ_F8F4=1 — the gate detects"
        echo "  the receipted acceptance/text move that the 20-token goldens are blind to)."
        exit 0
    fi
    echo "accept-gate: TEETH FAIL — MEMRA_MMQ_F8F4=1 passed every cell. That arm is receipted to"
    echo "  move q27's p1 cell (drafted 126->129, rounds 42->43, text sha fddd->33d6), so this"
    echo "  gate is NOT measuring acceptance and its green is worthless. FIX THE GATE."
    exit 1
fi

echo "accept-gate: $PASSES pass, $FAILS fail, $NOREF unpinned, $SKIPS skip"
if [ "$NOREF" -gt 0 ]; then
    echo "  $NOREF cell(s) have no pinned reference — run 'tools/accept-gate.sh --pin' at a"
    echo "  full-battery green point with committed crates/."
fi
if [ "$FAILS" -gt 0 ]; then
    cat <<'ACCEPTRED'

  ^ An acceptance or long-text FAIL is NOT a drift tripwire. Acceptance is a RATIO
    (clock-independent) and greedy text at temperature 0 is deterministic, so neither can be
    explained by thermals, clocks or a contended window. Read it as a real behavior change.
    DO NOT re-pin to make it green: that is the receipted failure mode (a --refresh-goldens
    after a kernel change silently re-pins the new arm). Root-cause first; re-pin only after
    the change is understood, reviewed and committed.
ACCEPTRED
fi
[ "$FAILS" -eq 0 ] || exit 1
exit 0
