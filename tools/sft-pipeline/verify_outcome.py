#!/usr/bin/env python3
"""verify_outcome.py — verified-outcome filter for SFT traces (sft-corpus-20260802).

Never infer success; quote it. Every verdict carries captured output. Traces that fail
verification are NOT deleted — they go to rejects.jsonl with the reason and the captured
evidence attached (owner rule: bad results are recorded too).

Each trace may carry a `verify` spec:

  {"type": "patch",
   "repo": "<path to pinned checkout>",        # required
   "rev": "<commit sha the patch applies to>", # required — the pin
   "patch": "<unified diff text>",             # or "patch_path"
   "test_cmd": ["pytest", "-x", "tests/..."],  # optional; argv list, no shell
   "timeout": 300}                             # seconds, default 300

  {"type": "answer",
   "expected": "<exact string>",               # or "pattern": "<python regex>"
   "source": "last_assistant"}                 # only supported source (default)

Patch flow: `git worktree add --detach <tmpdir> <rev>` from the pinned repo (the checkout
itself is never mutated), `git apply --check` then `git apply`, then the named test
command with cwd=tmpdir. method = tests_pass when a test_cmd ran, else patch_applies.
stdout/stderr are captured and quoted into outcome.detail (tail-truncated at 8 KiB each,
truncation marked). The tmp worktree is always removed.

Answer flow: match against the LAST assistant message's content, thinking stripped
(`<think>...</think>` blocks removed before matching). method = answer_check; the
matched/unmatched text is quoted into detail.

Traces with no `verify` spec and outcome.method == "manual" pass through only with
`--allow-manual`; otherwise they are rejected as unverifiable (reason quoted).

Output: verified traces (outcome rewritten with the evidence) -> --output;
rejects (reason + evidence in `reject`) -> --rejects. Exit 1 if anything was rejected,
2 on harness errors.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CAPTURE_LIMIT = 8192  # bytes of stdout/stderr tail kept as evidence


def _tail(text, limit=CAPTURE_LIMIT):
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return f"[...truncated {len(text) - limit} bytes...]" + text[-limit:]


def _run(argv, cwd, timeout):
    """Run argv (no shell), capture everything. Returns dict evidence, never raises on
    non-zero exit — the exit code IS the evidence."""
    try:
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {"argv": argv, "exit_code": proc.returncode,
                "stdout": _tail(proc.stdout), "stderr": _tail(proc.stderr)}
    except subprocess.TimeoutExpired as e:
        return {"argv": argv, "exit_code": None, "timeout_s": timeout,
                "stdout": _tail(e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout),
                "stderr": _tail(e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr),
                "error": f"timed out after {timeout}s"}
    except FileNotFoundError as e:
        return {"argv": argv, "exit_code": None, "error": f"command not found: {e}"}


def verify_patch(spec, workdir_root=None):
    """Apply the patch to a throwaway worktree of the pinned rev; optionally run tests.
    Returns (verified: bool, method, detail: dict)."""
    repo = spec.get("repo")
    rev = spec.get("rev")
    if not repo or not rev:
        return False, "patch_applies", {"error": "verify spec missing repo/rev pin"}
    repo = Path(repo)
    if not repo.is_dir():
        return False, "patch_applies", {"error": f"pinned repo not found: {repo}"}

    patch_text = spec.get("patch")
    if patch_text is None and spec.get("patch_path"):
        try:
            patch_text = Path(spec["patch_path"]).read_text(encoding="utf-8")
        except OSError as e:
            return False, "patch_applies", {"error": f"cannot read patch_path: {e}"}
    if not patch_text:
        return False, "patch_applies", {"error": "verify spec has no patch/patch_path"}

    timeout = spec.get("timeout", 300)
    evidence = {"repo": str(repo), "rev": rev, "steps": []}
    with tempfile.TemporaryDirectory(prefix="sft-verify-", dir=workdir_root) as tmp:
        wt = str(Path(tmp) / "wt")
        try:
            add = _run(["git", "-C", str(repo), "worktree", "add", "--detach", wt, rev],
                       cwd=None, timeout=120)
            evidence["steps"].append({"stage": "worktree_add", **add})
            if add.get("exit_code") != 0:
                return False, "patch_applies", evidence

            patch_file = Path(tmp) / "change.patch"
            patch_file.write_text(patch_text, encoding="utf-8")

            check = _run(["git", "apply", "--check", str(patch_file)], cwd=wt, timeout=60)
            evidence["steps"].append({"stage": "apply_check", **check})
            if check.get("exit_code") != 0:
                return False, "patch_applies", evidence

            apply_ = _run(["git", "apply", str(patch_file)], cwd=wt, timeout=60)
            evidence["steps"].append({"stage": "apply", **apply_})
            if apply_.get("exit_code") != 0:
                return False, "patch_applies", evidence

            test_cmd = spec.get("test_cmd")
            if not test_cmd:
                return True, "patch_applies", evidence
            if not isinstance(test_cmd, list) or not all(isinstance(a, str) for a in test_cmd):
                evidence["steps"].append(
                    {"stage": "test", "error": "test_cmd must be an argv list (no shell)"})
                return False, "tests_pass", evidence
            test = _run(test_cmd, cwd=wt, timeout=timeout)
            evidence["steps"].append({"stage": "test", **test})
            return test.get("exit_code") == 0, "tests_pass", evidence
        finally:
            subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", wt],
                           capture_output=True)
            subprocess.run(["git", "-C", str(repo), "worktree", "prune"],
                           capture_output=True)


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def verify_answer(spec, trace):
    """Exact / regex match against the last assistant message. Returns
    (verified, "answer_check", detail)."""
    source = spec.get("source", "last_assistant")
    if source != "last_assistant":
        return False, "answer_check", {"error": f"unsupported source {source!r}"}
    last = next((m for m in reversed(trace.get("messages", []))
                 if m.get("role") == "assistant"), None)
    if last is None:
        return False, "answer_check", {"error": "no assistant message to check"}
    text = _THINK_RE.sub("", last.get("content") or "").strip()
    detail = {"checked_text": _tail(text, 2048)}
    if "expected" in spec:
        detail["expected"] = spec["expected"]
        ok = text == spec["expected"].strip()
        detail["match"] = "exact" if ok else "none"
        return ok, "answer_check", detail
    if "pattern" in spec:
        detail["pattern"] = spec["pattern"]
        try:
            m = re.search(spec["pattern"], text, re.DOTALL)
        except re.error as e:
            detail["error"] = f"bad pattern: {e}"
            return False, "answer_check", detail
        detail["match"] = m.group(0)[:2048] if m else "none"
        return bool(m), "answer_check", detail
    return False, "answer_check", {"error": "verify spec needs expected or pattern"}


def verify_trace(trace, allow_manual=False, workdir_root=None):
    """Returns (verified, outcome_dict_or_none, reject_reason_or_none)."""
    spec = trace.get("verify")
    if spec is None:
        outcome = trace.get("outcome") or {}
        if outcome.get("method") == "manual" and allow_manual:
            if outcome.get("verified") is True and outcome.get("detail"):
                return True, outcome, None
            return False, None, ("manual outcome without verified=true + detail "
                                 "evidence; refusing to pass it through")
        return False, None, "no verify spec and not an allowed manual outcome"
    vtype = spec.get("type")
    if vtype == "patch":
        ok, method, detail = verify_patch(spec, workdir_root=workdir_root)
    elif vtype == "answer":
        ok, method, detail = verify_answer(spec, trace)
    else:
        return False, None, f"unknown verify.type {vtype!r}"
    outcome = {"verified": ok, "method": method, "detail": detail}
    if ok:
        return True, outcome, None
    return False, outcome, f"verification failed ({method})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="validated trace JSONL")
    ap.add_argument("-o", "--output", required=True, help="verified traces JSONL")
    ap.add_argument("--rejects", required=True, help="rejected traces JSONL (with reasons)")
    ap.add_argument("--allow-manual", action="store_true",
                    help="pass through outcome.method=manual traces that carry evidence")
    ap.add_argument("--workdir", default=None,
                    help="root for throwaway patch worktrees (default system tmp)")
    args = ap.parse_args(argv)

    now = datetime.now(timezone.utc).isoformat()
    n = kept = rejected = harness_errors = 0
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout, \
         open(args.rejects, "w", encoding="utf-8") as frej:
        for ln, line in enumerate(fin, 1):
            if not line.strip():
                continue
            n += 1
            try:
                trace = json.loads(line)
            except json.JSONDecodeError as e:
                harness_errors += 1
                frej.write(json.dumps({
                    "line": ln, "reject": {"reason": f"unparseable JSON: {e}", "ts": now},
                    "raw": line.rstrip("\n")}, ensure_ascii=False) + "\n")
                continue
            try:
                ok, outcome, reason = verify_trace(
                    trace, allow_manual=args.allow_manual, workdir_root=args.workdir)
            except Exception as e:  # harness bug — record, don't hide
                harness_errors += 1
                trace["reject"] = {"reason": f"harness error: {e!r}", "ts": now}
                frej.write(json.dumps(trace, ensure_ascii=False) + "\n")
                continue
            if ok:
                trace["outcome"] = outcome
                fout.write(json.dumps(trace, ensure_ascii=False) + "\n")
                kept += 1
            else:
                if outcome is not None:
                    trace["outcome"] = outcome
                trace["reject"] = {"reason": reason, "ts": now}
                frej.write(json.dumps(trace, ensure_ascii=False) + "\n")
                rejected += 1

    print(f"verified {kept}/{n} trace(s); rejected {rejected}; "
          f"harness errors {harness_errors} -> {args.rejects}", file=sys.stderr)
    if harness_errors:
        return 2
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())
