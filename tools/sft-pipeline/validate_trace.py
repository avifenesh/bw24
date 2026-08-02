#!/usr/bin/env python3
"""validate_trace.py — structural validator for SFT trace JSONL (sft-corpus-20260802).

One trace per line. Schema (see research/sft-corpus-20260802/README.md, the contract):

  task_id      non-empty string
  seed_source  non-empty string
  tools        optional list of OpenAI-shape tool schemas (each needs function.name)
  messages[]   role in {system,user,assistant,tool}; content string;
               assistant may carry tool_calls[] (id/type/function{name,arguments});
               tool must carry tool_call_id resolving to a pending call
  outcome      {verified: bool, method: tests_pass|patch_applies|answer_check|manual,
                detail: string|object}
  meta         {model, ts (ISO-8601), turns (== len(messages)), sub_session (str|null)}
  verify       optional verification spec consumed by verify_outcome.py

Legal role alternation (strict — corpus quality over tolerance):
  [system]? user (assistant tool*)* assistant     — i.e. optional leading system, then
  user; assistant follows user or a fully-consumed tool batch; every assistant
  tool_call must be answered by exactly one tool message before the next non-tool
  message; the trace ends on an assistant turn with no pending calls.

Exit non-zero on ANY invalid line; every problem is reported as `file:line: reason`.
Unknown top-level / message fields are allowed (the converter preserves them).
"""
import argparse
import json
import sys
from datetime import datetime

VALID_ROLES = {"system", "user", "assistant", "tool"}
VALID_METHODS = {"tests_pass", "patch_applies", "answer_check", "manual"}


def _is_nonempty_str(v):
    return isinstance(v, str) and v.strip() != ""


def _check_tool_calls(tool_calls, errs, where):
    """Validate an assistant message's tool_calls array; return the list of ids."""
    ids = []
    if not isinstance(tool_calls, list) or not tool_calls:
        errs.append(f"{where}: tool_calls must be a non-empty array when present")
        return ids
    for j, tc in enumerate(tool_calls):
        w = f"{where}.tool_calls[{j}]"
        if not isinstance(tc, dict):
            errs.append(f"{w}: must be an object")
            continue
        tc_id = tc.get("id")
        if not _is_nonempty_str(tc_id):
            errs.append(f"{w}: missing/empty id")
        else:
            ids.append(tc_id)
        if "type" in tc and tc["type"] != "function":
            errs.append(f"{w}: type must be \"function\" (got {tc['type']!r})")
        fn = tc.get("function")
        if not isinstance(fn, dict):
            errs.append(f"{w}: missing function object")
            continue
        if not _is_nonempty_str(fn.get("name")):
            errs.append(f"{w}: missing/empty function.name")
        args = fn.get("arguments")
        if args is None:
            pass  # absent arguments = no-arg call, allowed
        elif isinstance(args, str):
            if args.strip():
                try:
                    parsed = json.loads(args)
                    if not isinstance(parsed, dict):
                        errs.append(f"{w}: function.arguments must decode to a JSON object")
                except json.JSONDecodeError as e:
                    errs.append(f"{w}: function.arguments is not valid JSON: {e}")
        elif not isinstance(args, dict):
            errs.append(f"{w}: function.arguments must be a JSON object or object-string")
    return ids


def _check_messages(messages, errs):
    if not isinstance(messages, list) or not messages:
        errs.append("messages: must be a non-empty array")
        return
    pending = []          # unconsumed tool_call ids from the current assistant batch
    seen_ids = set()      # all tool_call ids in the trace (must be unique)
    prev_role = None
    for i, msg in enumerate(messages):
        w = f"messages[{i}]"
        if not isinstance(msg, dict):
            errs.append(f"{w}: must be an object")
            prev_role = None
            continue
        role = msg.get("role")
        if role not in VALID_ROLES:
            errs.append(f"{w}: bad role {role!r} (system|user|assistant|tool)")
            prev_role = None
            continue
        content = msg.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            errs.append(f"{w}: content must be a string (or null)")
            content = ""
        has_calls = "tool_calls" in msg
        if has_calls and role != "assistant":
            errs.append(f"{w}: tool_calls are only valid on assistant messages")

        # -- alternation state machine --
        if role == "system":
            if i != 0:
                errs.append(f"{w}: system is only legal at index 0")
        elif role == "user":
            if prev_role == "user":
                errs.append(f"{w}: user cannot follow user")
            elif prev_role == "tool":
                errs.append(f"{w}: user cannot follow a tool batch (assistant must respond)")
            elif prev_role == "assistant" and pending:
                errs.append(f"{w}: user cannot follow an assistant with pending tool_calls")
        elif role == "assistant":
            if i == 0 or prev_role == "assistant":
                errs.append(f"{w}: assistant must follow user or a consumed tool batch")
            elif prev_role == "tool" and pending:
                errs.append(f"{w}: assistant before tool batch consumed "
                            f"(unanswered ids: {sorted(pending)})")
            elif prev_role == "system":
                errs.append(f"{w}: assistant cannot directly follow system (user first)")
        elif role == "tool":
            if prev_role not in ("assistant", "tool"):
                errs.append(f"{w}: tool must follow an assistant tool_call batch")
            if not pending and prev_role in ("assistant", "tool"):
                errs.append(f"{w}: tool message but no pending tool_calls")

        # -- content emptiness --
        if role in ("system", "user", "tool") and not content.strip():
            errs.append(f"{w}: empty content is illegal for role {role}")
        if role == "assistant" and not content.strip() and not msg.get("tool_calls"):
            errs.append(f"{w}: assistant with empty content and no tool_calls")

        # -- tool_calls / tool_call_id bookkeeping --
        if role == "assistant" and has_calls:
            ids = _check_tool_calls(msg["tool_calls"], errs, w)
            for tc_id in ids:
                if tc_id in seen_ids:
                    errs.append(f"{w}: duplicate tool_call id {tc_id!r}")
                seen_ids.add(tc_id)
            pending = list(ids)
        if role == "tool":
            tcid = msg.get("tool_call_id")
            if not _is_nonempty_str(tcid):
                errs.append(f"{w}: tool message missing tool_call_id")
            elif tcid not in pending:
                errs.append(f"{w}: tool_call_id {tcid!r} does not resolve to a pending call "
                            f"(pending: {sorted(pending)})")
            else:
                pending.remove(tcid)
        prev_role = role

    last = messages[-1] if isinstance(messages[-1], dict) else {}
    if last.get("role") != "assistant":
        errs.append("messages: trace must end on an assistant turn")
    if pending:
        errs.append(f"messages: trace ends with unanswered tool_calls: {sorted(pending)}")


def _check_outcome(outcome, errs):
    if not isinstance(outcome, dict):
        errs.append("outcome: missing or not an object")
        return
    if not isinstance(outcome.get("verified"), bool):
        errs.append("outcome.verified: must be a bool")
    method = outcome.get("method")
    if method not in VALID_METHODS:
        errs.append(f"outcome.method: {method!r} not in {sorted(VALID_METHODS)}")
    if "detail" not in outcome:
        errs.append("outcome.detail: missing (record the evidence, even for failures)")
    elif not isinstance(outcome["detail"], (str, dict)):
        errs.append("outcome.detail: must be a string or object")


def _check_meta(meta, n_messages, errs):
    if not isinstance(meta, dict):
        errs.append("meta: missing or not an object")
        return
    if not _is_nonempty_str(meta.get("model")):
        errs.append("meta.model: missing/empty")
    ts = meta.get("ts")
    if not _is_nonempty_str(ts):
        errs.append("meta.ts: missing/empty")
    else:
        try:
            datetime.fromisoformat(ts)
        except ValueError:
            errs.append(f"meta.ts: not ISO-8601: {ts!r}")
    turns = meta.get("turns")
    if not isinstance(turns, int) or isinstance(turns, bool):
        errs.append("meta.turns: must be an int")
    elif turns != n_messages:
        errs.append(f"meta.turns: {turns} != len(messages) ({n_messages})")
    if "sub_session" not in meta:
        errs.append("meta.sub_session: missing (use null if none)")
    elif meta["sub_session"] is not None and not isinstance(meta["sub_session"], str):
        errs.append("meta.sub_session: must be a string or null")


def _check_tools(tools, errs):
    if tools is None:
        return
    if not isinstance(tools, list):
        errs.append("tools: must be an array when present")
        return
    for i, t in enumerate(tools):
        if not isinstance(t, dict) or not isinstance(t.get("function"), dict) \
                or not _is_nonempty_str(t["function"].get("name")):
            errs.append(f"tools[{i}]: each tool needs function.name")


def validate_line(obj):
    """Return a list of error strings for one parsed trace object."""
    errs = []
    if not isinstance(obj, dict):
        return ["trace must be a JSON object"]
    if not _is_nonempty_str(obj.get("task_id")):
        errs.append("task_id: missing/empty")
    if not _is_nonempty_str(obj.get("seed_source")):
        errs.append("seed_source: missing/empty")
    _check_tools(obj.get("tools"), errs)
    messages = obj.get("messages")
    if isinstance(messages, list) and messages:
        _check_messages(messages, errs)
        n = len(messages)
    else:
        errs.append("messages: must be a non-empty array")
        n = 0
    _check_outcome(obj.get("outcome"), errs)
    _check_meta(obj.get("meta"), n, errs)
    verify = obj.get("verify")
    if verify is not None:
        if not isinstance(verify, dict) or verify.get("type") not in ("patch", "answer"):
            errs.append("verify: must be an object with type patch|answer")
    return errs


def validate_file(path, out=sys.stdout):
    """Validate one JSONL file. Returns (n_lines, n_bad)."""
    n = bad = 0
    try:
        fh = sys.stdin if path == "-" else open(path, encoding="utf-8")
    except OSError as e:
        print(f"{path}: cannot open: {e}", file=out)
        return 0, 1
    with fh:
        for ln, line in enumerate(fh, 1):
            if not line.strip():
                continue
            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{path}:{ln}: not valid JSON: {e}", file=out)
                bad += 1
                continue
            errs = validate_line(obj)
            if errs:
                bad += 1
                for e in errs:
                    print(f"{path}:{ln}: {e}", file=out)
    return n, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="trace JSONL files ('-' = stdin)")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="suppress the per-file OK summary (errors always print)")
    args = ap.parse_args(argv)
    total = total_bad = 0
    for path in args.files:
        n, bad = validate_file(path)
        total += n
        total_bad += bad
        if not args.quiet:
            status = "OK" if bad == 0 else f"{bad} INVALID"
            print(f"{path}: {n} trace(s), {status}")
    if total_bad:
        print(f"FAIL: {total_bad}/{total} invalid trace line(s)", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"all {total} trace(s) valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
