#!/usr/bin/env bash
# smoke_test.sh — run all four sft-pipeline tools over the fixtures.
# Exercises: validator accept + reject paths, converter golden byte-parity selftest,
# converter + roundtrip, patch + answer verification (real subprocesses, real tmp
# git worktree), rejects with reasons, and the stats report.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
FX="$HERE/fixtures"
OUT="$(mktemp -d /tmp/sft-smoke-XXXXXX)"
trap 'rm -rf "$OUT"' EXIT
PY=python3
fails=0
step() { printf '\n== %s ==\n' "$1"; }
expect() { # expect <want_rc> <label> <cmd...>
    local want=$1 label=$2; shift 2
    "$@"; local rc=$?
    if [ "$rc" -eq "$want" ]; then
        echo "PASS: $label (exit $rc)"
    else
        echo "FAIL: $label (exit $rc, wanted $want)"; fails=$((fails+1))
    fi
}

step "1a. validator accepts the valid fixtures"
expect 0 "validate valid.jsonl + toolcall.jsonl" \
    $PY "$HERE/validate_trace.py" "$FX/valid.jsonl" "$FX/toolcall.jsonl"

step "1b. validator rejects the broken fixtures (must exit non-zero, with reasons)"
expect 1 "validate broken.jsonl" \
    $PY "$HERE/validate_trace.py" "$FX/broken.jsonl"

step "2a. converter golden byte-parity selftest (chat.rs vectors)"
expect 0 "convert_k3_qwen --selftest" \
    $PY "$HERE/convert_k3_qwen.py" --selftest

step "2b. convert fixtures with round-trip verification"
cat "$FX/valid.jsonl" "$FX/toolcall.jsonl" > "$OUT/all.jsonl"
expect 0 "convert --roundtrip" \
    $PY "$HERE/convert_k3_qwen.py" --roundtrip "$OUT/all.jsonl" -o "$OUT/converted.jsonl"
n_conv=$(wc -l < "$OUT/converted.jsonl")
if [ "$n_conv" -eq 3 ]; then echo "PASS: 3 converted records"; else
    echo "FAIL: expected 3 converted records, got $n_conv"; fails=$((fails+1)); fi

step "3a. build a pinned throwaway repo for the patch arm"
REPO="$OUT/pinned-repo"
git init -q "$REPO"
git -C "$REPO" -c user.email=sft@local -c user.name=sft checkout -qb pin
cat > "$REPO/calc.py" <<'EOF'
def add(a, b):
    return a - b  # BUG
EOF
cat > "$REPO/test_calc.py" <<'EOF'
from calc import add
def test_add():
    assert add(2, 3) == 5
if __name__ == "__main__":
    test_add()
    print("test_add passed")
EOF
git -C "$REPO" add -A
git -C "$REPO" -c user.email=sft@local -c user.name=sft commit -qm "pinned: buggy add"
REV=$(git -C "$REPO" rev-parse HEAD)
echo "pinned repo at $REV"

step "3b. verify_outcome: passing patch + failing patch + answer arms"
GOOD_PATCH='--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b  # BUG
+    return a + b
'
BAD_PATCH='--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    THIS CONTEXT DOES NOT EXIST
+    return a + b
'
printf '%s' "$GOOD_PATCH" > "$OUT/good.patch"
printf '%s' "$BAD_PATCH" > "$OUT/bad.patch"
$PY - "$OUT" "$REPO" "$REV" <<'EOF'
import json, sys
out, repo, rev = sys.argv[1], sys.argv[2], sys.argv[3]
base_msgs = [{"role": "user", "content": "Fix add()"},
             {"role": "assistant", "content": "Patched: use a + b."}]
base_meta = {"model": "kimi-k3", "ts": "2026-08-02T15:00:00+00:00",
             "turns": 2, "sub_session": None}
pending = {"verified": False, "method": "tests_pass", "detail": "pending"}
good_patch = open(f"{out}/good.patch").read()
bad_patch = open(f"{out}/bad.patch").read()
traces = [
    {"task_id": "fx-patch-good", "seed_source": "fixtures", "messages": base_msgs,
     "outcome": pending, "meta": base_meta,
     "verify": {"type": "patch", "repo": repo, "rev": rev, "patch": good_patch,
                "test_cmd": ["python3", "test_calc.py"], "timeout": 60}},
    {"task_id": "fx-patch-bad", "seed_source": "fixtures", "messages": base_msgs,
     "outcome": pending, "meta": base_meta,
     "verify": {"type": "patch", "repo": repo, "rev": rev, "patch": bad_patch,
                "test_cmd": ["python3", "test_calc.py"], "timeout": 60}},
]
with open(f"{out}/patch_traces.jsonl", "w") as fh:
    for t in traces:
        fh.write(json.dumps(t) + "\n")
print("wrote patch_traces.jsonl (2 traces)")
EOF

cat "$FX/valid.jsonl" "$FX/toolcall.jsonl" "$OUT/patch_traces.jsonl" > "$OUT/to_verify.jsonl"
# fx-answer-002 says "Lyon." vs expected "Paris." and fx-patch-bad has a bad patch:
# exactly 2 rejects expected -> exit 1 is the CORRECT outcome here.
expect 1 "verify_outcome (2 expected rejects)" \
    $PY "$HERE/verify_outcome.py" "$OUT/to_verify.jsonl" \
        -o "$OUT/verified.jsonl" --rejects "$OUT/rejects.jsonl"
n_ok=$(wc -l < "$OUT/verified.jsonl"); n_rej=$(wc -l < "$OUT/rejects.jsonl")
if [ "$n_ok" -eq 3 ] && [ "$n_rej" -eq 2 ]; then
    echo "PASS: 3 verified, 2 rejected"
else
    echo "FAIL: expected 3 verified / 2 rejected, got $n_ok / $n_rej"; fails=$((fails+1))
fi
echo "-- reject reasons (quoted evidence lives in each record):"
$PY -c '
import json, sys
for line in open(sys.argv[1]):
    t = json.loads(line)
    print("  {}: {}".format(t.get("task_id"), t["reject"]["reason"]))' "$OUT/rejects.jsonl"
echo "-- captured test evidence from the verified patch trace:"
$PY -c '
import json, sys
for line in open(sys.argv[1]):
    t = json.loads(line)
    if t["task_id"] == "fx-patch-good":
        step = t["outcome"]["detail"]["steps"][-1]
        print("  stage={} exit={} stdout={!r}".format(
            step["stage"], step["exit_code"], step["stdout"].strip()))' \
    "$OUT/verified.jsonl"

step "3c. leftover worktrees check (harness must clean up)"
leftover=$(git -C "$REPO" worktree list | wc -l)
if [ "$leftover" -eq 1 ]; then echo "PASS: no leftover worktrees"; else
    echo "FAIL: leftover worktrees:"; git -C "$REPO" worktree list; fails=$((fails+1)); fi

step "4. corpus stats over verified + rejects"
expect 0 "corpus_stats" \
    $PY "$HERE/corpus_stats.py" "$OUT/verified.jsonl" \
        --rejects "$OUT/rejects.jsonl" -o "$OUT/report.md"
echo "-- report head:"
head -n 12 "$OUT/report.md"

printf '\n== SMOKE RESULT: '
if [ "$fails" -eq 0 ]; then echo "ALL PASS =="; exit 0; else
    echo "$fails FAILURE(S) =="; exit 1; fi
