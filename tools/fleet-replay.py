#!/usr/bin/env python3
"""Drive low-rate, agent-shaped replay traffic through the local memra server."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import random
import signal
import sys
import threading
import time
import urllib.error
import urllib.request


DEFAULT_BASE = "http://127.0.0.1:8002"
DEFAULT_MODEL = "qwen36-27b"
DEFAULT_DURATION_S = 300.0
DEFAULT_REQUESTS_PER_MINUTE = 3.0
DEFAULT_PROMPT_COMPLETION_RATIO = 89.5
DEFAULT_SESSION_COUNT = 12
DEFAULT_TENANT_COUNT = 4
RECEIPT_LABEL = "replay-calibrated"
BURST_TURNS_MIN = 2
BURST_TURNS_MAX = 4


def estimate_tokens(text: str) -> int:
    """Cheap budgeting estimate; response usage remains the authoritative count."""
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def completion_budget(prompt_tokens: int, ratio: float) -> int:
    return max(1, round(prompt_tokens / ratio))


def _tool_schema(name: str, description: str, argument: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    argument: {
                        "type": "string",
                        "description": f"The {argument.replace('_', ' ')} to operate on.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "A concise reason this tool call is the next action.",
                    },
                },
                "required": [argument, "reason"],
                "additionalProperties": False,
            },
        },
    }


_TEMPLATE_SPECS = (
    (
        "repository-maintainer",
        1100,
        "You maintain a production repository. Inspect before editing, preserve unrelated "
        "work, make narrow changes, and verify behavior with the repository's own tests.",
        (
            ("search_text", "Search tracked source files for a literal or regular expression.", "query"),
            ("read_file", "Read a bounded range from a repository file.", "path"),
            ("apply_patch", "Apply a reviewable source patch.", "patch"),
            ("run_command", "Run a non-interactive build, test, or inspection command.", "command"),
        ),
    ),
    (
        "incident-responder",
        1400,
        "You are the primary responder for a live inference service. Establish the timeline, "
        "separate symptoms from causes, avoid destructive recovery, and preserve raw evidence.",
        (
            ("query_metrics", "Query a bounded service-metrics time range.", "metric_query"),
            ("fetch_logs", "Fetch exact log lines for one service and time range.", "service"),
            ("inspect_deployment", "Read deployment, replica, and rollout state.", "deployment"),
            ("update_incident", "Append verified evidence or a decision to the incident record.", "incident_id"),
        ),
    ),
    (
        "data-analyst",
        1750,
        "You analyze operational data for an engineering team. State metric definitions, "
        "inspect schemas before querying, retain denominators, and distinguish observations "
        "from recommendations.",
        (
            ("inspect_schema", "Inspect tables, columns, and types without running a query.", "dataset"),
            ("run_sql", "Run a read-only SQL query with a bounded result set.", "sql"),
            ("profile_result", "Compute descriptive statistics for a saved query result.", "result_id"),
            ("render_chart", "Render a chart from a saved result using explicit encodings.", "result_id"),
        ),
    ),
    (
        "research-assistant",
        2100,
        "You research current technical questions. Prefer primary sources, track publication "
        "dates, quote sparingly, cite every material claim, and call out unresolved conflicts.",
        (
            ("web_search", "Search the public web for current primary sources.", "query"),
            ("open_source", "Open and inspect a specific source or document.", "url"),
            ("find_in_source", "Find a phrase or symbol inside an opened source.", "pattern"),
            ("save_citation", "Save a claim-to-source citation with a short evidence note.", "claim"),
        ),
    ),
    (
        "cloud-operator",
        2450,
        "You operate a multi-region GPU service. Read current state before mutation, keep "
        "capacity and rollback constraints explicit, and never widen a change beyond the "
        "named service and region.",
        (
            ("describe_service", "Read replicas, health, capacity, and current revision.", "service"),
            ("list_events", "List recent scheduler and infrastructure events.", "region"),
            ("plan_rollout", "Produce a dry-run rollout plan and rollback boundary.", "revision"),
            ("execute_rollout", "Apply an approved rollout plan.", "plan_id"),
        ),
    ),
    (
        "support-engineer",
        2800,
        "You resolve technical support cases using account facts and product documentation. "
        "Protect customer data, do not guess entitlement state, and record every external "
        "change in the case timeline.",
        (
            ("lookup_account", "Read the minimum account fields needed for the active case.", "account_id"),
            ("search_knowledge", "Search approved product and operations documentation.", "query"),
            ("inspect_request", "Inspect a redacted request trace by request identifier.", "request_id"),
            ("update_case", "Append a factual note or status change to the support case.", "case_id"),
        ),
    ),
    (
        "security-reviewer",
        3250,
        "You review application changes for concrete security impact. Trace trust boundaries, "
        "prove reachability, avoid speculative severity, and include a minimal reproduction "
        "and remediation for every finding.",
        (
            ("inspect_diff", "Inspect a bounded code diff with repository context.", "revision"),
            ("trace_symbol", "Find definitions and callers of a security-relevant symbol.", "symbol"),
            ("scan_dependencies", "Check declared dependencies against current advisories.", "manifest"),
            ("record_finding", "Record a reproducible security finding with evidence.", "title"),
        ),
    ),
    (
        "workflow-coordinator",
        3650,
        "You coordinate a long-running engineering workflow. Keep the task ledger current, "
        "route work to the right owner, surface blockers early, and never report completion "
        "without an attached verification result.",
        (
            ("list_tasks", "Read current tasks, owners, dependencies, and status.", "project"),
            ("inspect_artifact", "Inspect a produced artifact or verification receipt.", "artifact"),
            ("update_task", "Change one task state with a factual status note.", "task_id"),
            ("send_message", "Send a concise coordination message to a named owner.", "recipient"),
        ),
    ),
)


_PLAYBOOK_RULES = (
    "Read the smallest current-state surface that can answer the question before taking action.",
    "Treat tool output as evidence, not as permission to broaden the requested scope.",
    "When a command or query fails, preserve the exact error and test the narrowest hypothesis next.",
    "Do not claim a change succeeded until the relevant state is read back or a focused test passes.",
    "Keep identifiers, timestamps, revisions, and metric units exact in all summaries.",
    "Prefer one reversible operation over a batch of coupled mutations when the state is uncertain.",
    "Do not expose secrets, private payloads, access tokens, or unrelated customer data in output.",
    "Separate measured facts, inferences, and proposed next actions so another operator can audit them.",
    "If two sources disagree, inspect their collection time and authority before choosing either.",
    "Use bounded queries and reads first; expand only when the bounded result cannot resolve the task.",
    "Preserve raw evidence for failures and include the command, target, and observed result.",
    "A completion statement must name the verification that closed the task and any remaining risk.",
)


@dataclass(frozen=True)
class PrefixTemplate:
    name: str
    target_tokens: int
    text: str


def _build_prefix_template(spec: tuple, offset: int) -> PrefixTemplate:
    name, target_tokens, system_prompt, tools = spec
    schemas = [_tool_schema(*tool) for tool in tools]
    parts = [
        "<|im_start|>system\n",
        system_prompt,
        "\n\nAvailable tools are described by JSON Schema. Emit at most one tool call per "
        "assistant turn unless the user explicitly asks for a final answer.\n<tools>\n",
        json.dumps(schemas, indent=2, sort_keys=True),
        "\n</tools>\n\nOperational playbook:\n",
    ]
    end = "<|im_end|>\n"
    index = 0
    while estimate_tokens("".join(parts) + end) < target_tokens:
        rule = _PLAYBOOK_RULES[(index + offset) % len(_PLAYBOOK_RULES)]
        tool_name = tools[index % len(tools)][0]
        parts.append(
            f"{index + 1}. {rule} For the {name} workflow, record why `{tool_name}` is "
            "or is not the next justified action, what evidence it should return, and the "
            "condition that would stop further work.\n"
        )
        index += 1
    return PrefixTemplate(name, target_tokens, "".join(parts) + end)


PREFIX_TEMPLATES = tuple(
    _build_prefix_template(spec, offset)
    for offset, spec in enumerate(_TEMPLATE_SPECS)
)


_TURN_TASKS = (
    "Inspect the current state described above and emit the single best next tool call. "
    "Use a compact JSON object and spend the full response budget on concrete arguments.",
    "Continue from the prior assistant result. Verify one assumption before choosing the "
    "next action, then emit exactly one compact tool call.",
    "A new constraint arrived: preserve all unrelated state and avoid broad scans. Select "
    "the narrowest tool call that still advances the task.",
    "Review the conversation for an unsupported claim. Emit the tool call that would "
    "confirm or refute it with the least operational impact.",
    "Prepare the next handoff step. Use one tool call whose output would be sufficient for "
    "another engineer to continue without repeating earlier work.",
)


@dataclass
class AgentSession:
    name: str
    tenant: str
    cache_salt: str
    prefix: str
    turn: int = 0
    transcript: str = ""

    def next_prompt(self) -> str:
        base = self.transcript or self.prefix
        task = _TURN_TASKS[self.turn % len(_TURN_TASKS)]
        return (
            f"{base}<|im_start|>user\n"
            f"Synthetic session {self.name}, turn {self.turn + 1}. {task}\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        )

    def accept_reply(self, prompt: str, reply: str) -> None:
        ending = "" if reply.rstrip().endswith("<|im_end|>") else "\n<|im_end|>"
        self.transcript = f"{prompt}{reply}{ending}\n"
        self.turn += 1


def build_sessions(
    count: int,
    tenant_count: int,
    templates: tuple[PrefixTemplate, ...] = PREFIX_TEMPLATES,
) -> list[AgentSession]:
    sessions = []
    for index in range(count):
        pair = index // 2
        tenant_index = pair % tenant_count
        template = templates[pair % len(templates)]
        sessions.append(
            AgentSession(
                name=f"replay-session-{index + 1:03d}",
                tenant=f"synthetic-tenant-{tenant_index + 1:02d}",
                cache_salt=f"{RECEIPT_LABEL}-tenant-{tenant_index + 1:02d}",
                prefix=template.text,
            )
        )
    return sessions


@dataclass(frozen=True)
class ReplayConfig:
    base: str
    model: str
    duration_s: float
    requests_per_minute: float
    prompt_completion_ratio: float
    session_count: int
    tenant_count: int
    api_key: str | None
    timeout_s: float
    request_limit: int | None
    seed: int | None


def _nonnegative_int(value: object, fallback: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return fallback


def _completion_text(payload: dict) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        text = choices[0].get("text")
        if isinstance(text, str):
            return text
    text = payload.get("text")
    if isinstance(text, str):
        return text
    raise ValueError("response has no completion text")


def request_completion(
    config: ReplayConfig,
    session: AgentSession,
    prompt: str,
    max_tokens: int,
) -> dict:
    body = {
        "model": config.model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.95,
        "stream": False,
        "cache_salt": session.cache_salt,
        "session_id": session.name,
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "memra-fleet-replay/1",
        "X-Memra-Replay": RECEIPT_LABEL,
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = urllib.request.Request(
        config.base.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            payload = json.load(response)
        if not isinstance(payload, dict):
            raise ValueError("response must be a JSON object")
        text = _completion_text(payload)
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        details = usage.get("prompt_tokens_details")
        if not isinstance(details, dict):
            details = {}
        return {
            "ok": True,
            "text": text,
            "latency_s": time.monotonic() - started,
            "prompt_tokens": _nonnegative_int(
                usage.get("prompt_tokens"),
                _nonnegative_int(payload.get("prompt_tokens"), estimate_tokens(prompt)),
            ),
            "completion_tokens": _nonnegative_int(
                usage.get("completion_tokens"),
                _nonnegative_int(payload.get("n_tokens"), estimate_tokens(text)),
            ),
            "cached_tokens": _nonnegative_int(
                details.get("cached_tokens"),
                _nonnegative_int(payload.get("cached_tokens")),
            ),
        }
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read(500).decode(errors="replace")
        except OSError:
            detail = ""
        return {
            "ok": False,
            "fatal": exc.code in {400, 401, 403, 404},
            "latency_s": time.monotonic() - started,
            "error": f"HTTP {exc.code}: {detail or exc.reason}",
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "fatal": True,
            "latency_s": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _choose_session(
    rng: random.Random,
    sessions: list[AgentSession],
    previous_index: int | None,
) -> int:
    index = rng.randrange(len(sessions))
    if previous_index is not None and len(sessions) > 1 and index == previous_index:
        index = (index + 1 + rng.randrange(len(sessions) - 1)) % len(sessions)
    return index


def run_replay(
    config: ReplayConfig,
    stop_event: threading.Event,
    log=lambda message: print(message, file=sys.stderr, flush=True),
) -> dict:
    rng = random.Random(config.seed)
    sessions = build_sessions(config.session_count, config.tenant_count)
    started = time.monotonic()
    deadline = started + config.duration_s
    next_arrival = started
    active_index = None
    previous_index = None
    burst_remaining = 0
    attempts = 0
    successes = 0
    errors = 0
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    fatal_error = None
    stop_reason = "duration"

    log(
        f"[fleet-replay] label={RECEIPT_LABEL} base={config.base} "
        f"duration={config.duration_s:g}s rate={config.requests_per_minute:g}/min "
        f"sessions={config.session_count} tenants={config.tenant_count} "
        f"ratio={config.prompt_completion_ratio:g}:1"
    )

    while True:
        if stop_event.is_set():
            stop_reason = "signal"
            break
        if config.request_limit is not None and attempts >= config.request_limit:
            stop_reason = "request-limit"
            break
        now = time.monotonic()
        if now >= deadline:
            stop_reason = "duration"
            break
        wait_until = min(next_arrival, deadline)
        if wait_until > now and stop_event.wait(wait_until - now):
            stop_reason = "signal"
            break
        if time.monotonic() >= deadline and next_arrival >= deadline:
            stop_reason = "duration"
            break

        if burst_remaining == 0:
            active_index = _choose_session(rng, sessions, previous_index)
            previous_index = active_index
            burst_remaining = rng.randint(BURST_TURNS_MIN, BURST_TURNS_MAX)
        session = sessions[active_index]
        burst_remaining -= 1

        arrival = next_arrival
        next_arrival = arrival + rng.expovariate(config.requests_per_minute / 60.0)
        prompt = session.next_prompt()
        estimated_prompt_tokens = estimate_tokens(prompt)
        max_tokens = completion_budget(
            estimated_prompt_tokens, config.prompt_completion_ratio
        )
        attempts += 1
        result = request_completion(config, session, prompt, max_tokens)

        if result["ok"]:
            session.accept_reply(prompt, result["text"])
            successes += 1
            prompt_tokens += result["prompt_tokens"]
            completion_tokens += result["completion_tokens"]
            cached_tokens += result["cached_tokens"]
            log(
                f"[fleet-replay] ok request={attempts} session={session.name} "
                f"tenant={session.tenant} turn={session.turn} "
                f"prompt={result['prompt_tokens']} cached={result['cached_tokens']} "
                f"completion={result['completion_tokens']} max={max_tokens} "
                f"latency={result['latency_s']:.3f}s"
            )
        else:
            errors += 1
            log(
                f"[fleet-replay] error request={attempts} session={session.name}: "
                f"{result['error']}"
            )
            if result.get("fatal"):
                fatal_error = result["error"]
                stop_reason = "error"
                break

    elapsed = time.monotonic() - started
    return {
        "label": RECEIPT_LABEL,
        "base": config.base,
        "model": config.model,
        "duration_requested_s": config.duration_s,
        "elapsed_s": round(elapsed, 6),
        "requests_per_minute": config.requests_per_minute,
        "sessions": config.session_count,
        "tenants": config.tenant_count,
        "configured_prompt_completion_ratio": config.prompt_completion_ratio,
        "actual_prompt_completion_ratio": (
            round(prompt_tokens / completion_tokens, 6)
            if completion_tokens
            else None
        ),
        "requests_attempted": attempts,
        "requests_ok": successes,
        "requests_error": errors,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "stop_reason": stop_reason,
        "fatal_error": fatal_error,
    }


def parse_args(argv: list[str] | None = None) -> ReplayConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"wall-clock seconds to run (default {DEFAULT_DURATION_S:g})",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help=f"mean Poisson arrival rate (default {DEFAULT_REQUESTS_PER_MINUTE:g})",
    )
    parser.add_argument(
        "--prompt-completion-ratio",
        type=float,
        default=DEFAULT_PROMPT_COMPLETION_RATIO,
        help=f"target prompt:completion token ratio (default {DEFAULT_PROMPT_COMPLETION_RATIO:g}:1)",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=DEFAULT_SESSION_COUNT,
        help=f"synthetic conversation pool size (default {DEFAULT_SESSION_COUNT})",
    )
    parser.add_argument(
        "--tenants",
        type=int,
        default=DEFAULT_TENANT_COUNT,
        help=f"synthetic cache-salt tenant count (default {DEFAULT_TENANT_COUNT})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MEMRA_API_KEY"),
        help="Bearer key (default MEMRA_API_KEY; omitted for an open dev server)",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="optional request cap in addition to --duration",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.requests_per_minute <= 0:
        parser.error("--requests-per-minute must be positive")
    if args.prompt_completion_ratio <= 0:
        parser.error("--prompt-completion-ratio must be positive")
    if args.sessions <= 0:
        parser.error("--sessions must be positive")
    if args.tenants <= 0:
        parser.error("--tenants must be positive")
    if args.tenants > args.sessions:
        parser.error("--tenants cannot exceed --sessions")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.requests is not None and args.requests <= 0:
        parser.error("--requests must be positive")

    return ReplayConfig(
        base=args.base,
        model=args.model,
        duration_s=args.duration,
        requests_per_minute=args.requests_per_minute,
        prompt_completion_ratio=args.prompt_completion_ratio,
        session_count=args.sessions,
        tenant_count=args.tenants,
        api_key=args.api_key,
        timeout_s=args.timeout,
        request_limit=args.requests,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    stop_event = threading.Event()

    def stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    summary = run_replay(config, stop_event)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if summary["fatal_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
