#!/usr/bin/env bash
# Public OpenAI-compatible streaming smoke for a provisioned memra endpoint.

set -euo pipefail

BASE_URL="${MEMRA_BASE_URL:-${OPENAI_BASE_URL:-}}"
API_KEY="${MEMRA_API_KEY:-${OPENAI_API_KEY:-}}"
MODEL="${MEMRA_MODEL:-stepfun/step-3.7-flash}"
REQUESTS="${MEMRA_SMOKE_REQUESTS:-3}"
MAX_TOKENS="${MEMRA_SMOKE_MAX_TOKENS:-16}"
TIMEOUT="${MEMRA_SMOKE_TIMEOUT:-300}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: deploy/runpod/smoke.sh [options]

Options:
  --base-url URL       public API base, with or without trailing /v1
  --api-key KEY        bearer API key
  --model MODEL        default stepfun/step-3.7-flash
  -n, --requests N     request count, default 3
  --max-tokens N       generated-token cap, default 16
  --timeout SECONDS    per-request socket timeout, default 300
  --dry-run            print a sanitized request and exit without network access
  -h, --help

Environment equivalents:
  MEMRA_BASE_URL or OPENAI_BASE_URL
  MEMRA_API_KEY or OPENAI_API_KEY
  MEMRA_MODEL, MEMRA_SMOKE_REQUESTS, MEMRA_SMOKE_MAX_TOKENS,
  MEMRA_SMOKE_TIMEOUT

Only Bash and Python 3's standard library are required.
EOF
}

while (($#)); do
    case "$1" in
        --base-url)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            BASE_URL="$2"
            shift 2
            ;;
        --api-key)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            API_KEY="$2"
            shift 2
            ;;
        --model)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            MODEL="$2"
            shift 2
            ;;
        -n|--requests)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            REQUESTS="$2"
            shift 2
            ;;
        --max-tokens)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            MAX_TOKENS="$2"
            shift 2
            ;;
        --timeout)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            TIMEOUT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'smoke.sh: unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for pair in \
    "requests:$REQUESTS" \
    "max-tokens:$MAX_TOKENS" \
    "timeout:$TIMEOUT"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        printf 'smoke.sh: %s must be a positive integer: %s\n' "$name" "$value" >&2
        exit 2
    }
done

if ((DRY_RUN)); then
    base_display="${BASE_URL:-https://api.example.invalid/v1}"
    printf '%s\n' \
        '[smoke] DRY RUN - no network request will be sent' \
        "[smoke] base_url=${base_display%/}" \
        "[smoke] Authorization: Bearer <redacted>" \
        "[smoke] model=$MODEL requests=$REQUESTS max_tokens=$MAX_TOKENS timeout=${TIMEOUT}s"
    printf '%s\n' \
        '[smoke] body={"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with exactly: memra smoke ok"}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":'"$MAX_TOKENS"',"temperature":0,"cache_salt":"runpod-public-smoke"}'
    exit 0
fi

[[ -n "$BASE_URL" ]] || {
    printf 'smoke.sh: --base-url or MEMRA_BASE_URL is required\n' >&2
    exit 2
}
[[ -n "$API_KEY" ]] || {
    printf 'smoke.sh: --api-key or MEMRA_API_KEY is required\n' >&2
    exit 2
}
command -v python3 >/dev/null 2>&1 || {
    printf 'smoke.sh: python3 is required\n' >&2
    exit 1
}

MEMRA_SMOKE_API_KEY_INTERNAL="$API_KEY" \
python3 - "$BASE_URL" "$MODEL" "$REQUESTS" "$MAX_TOKENS" "$TIMEOUT" <<'PY'
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request


base_url, model = sys.argv[1:3]
request_count, max_tokens, timeout = map(int, sys.argv[3:6])
api_key = os.environ.pop("MEMRA_SMOKE_API_KEY_INTERNAL")

base_url = base_url.rstrip("/")
if not base_url.endswith("/v1"):
    base_url += "/v1"
endpoint = base_url + "/chat/completions"


def nonnegative_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AssertionError(f"{field} must be a non-negative integer, got {value!r}")
    return value


def validate_usage(usage):
    if not isinstance(usage, dict):
        raise AssertionError(f"usage must be an object, got {usage!r}")
    prompt = nonnegative_int(usage.get("prompt_tokens"), "usage.prompt_tokens")
    completion = nonnegative_int(
        usage.get("completion_tokens"), "usage.completion_tokens"
    )
    total = nonnegative_int(usage.get("total_tokens"), "usage.total_tokens")
    if total != prompt + completion:
        raise AssertionError(
            f"usage.total_tokens={total}, expected prompt+completion={prompt + completion}"
        )
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        raise AssertionError("usage.prompt_tokens_details must be an object")
    cached = nonnegative_int(
        details.get("cached_tokens"),
        "usage.prompt_tokens_details.cached_tokens",
    )
    if cached > prompt:
        raise AssertionError(f"cached_tokens={cached} exceeds prompt_tokens={prompt}")
    return prompt, completion, total, cached


ttfts = []
for index in range(1, request_count + 1):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: memra smoke ok",
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0,
        "cache_salt": "runpod-public-smoke",
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "memra-runpod-smoke/1",
        },
    )

    started = time.perf_counter()
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(
            f"[smoke] request {index}: HTTP {exc.code}, expected 200: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"[smoke] request {index}: connection failed: {exc}") from exc

    with response:
        status = getattr(response, "status", response.getcode())
        if status != 200:
            raise SystemExit(
                f"[smoke] request {index}: HTTP {status}, expected 200"
            )
        request_id = response.headers.get("x-request-id", "-")
        first_token_at = None
        final_usage = None
        saw_done = False

        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"invalid SSE JSON: {data!r}") from exc
            if "error" in event:
                raise AssertionError(f"stream returned an error object: {event['error']!r}")

            if event.get("usage") is not None:
                final_usage = event["usage"]

            choices = event.get("choices") or []
            if choices and first_token_at is None:
                delta = choices[0].get("delta") or {}
                generated = (
                    bool(delta.get("content"))
                    or bool(delta.get("reasoning"))
                    or bool(delta.get("reasoning_content"))
                    or bool(delta.get("tool_calls"))
                )
                if generated:
                    first_token_at = time.perf_counter()

        if not saw_done:
            raise AssertionError("stream closed without data: [DONE]")
        if first_token_at is None:
            raise AssertionError("stream completed without a generated token delta")
        if final_usage is None:
            raise AssertionError("stream completed without final usage")

        prompt, completion, total, cached = validate_usage(final_usage)
        ttft_ms = (first_token_at - started) * 1000.0
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ttfts.append(ttft_ms)
        print(
            f"[smoke] {index}/{request_count} HTTP 200 "
            f"ttft={ttft_ms:.1f}ms elapsed={elapsed_ms:.1f}ms "
            f"usage=prompt:{prompt},completion:{completion},total:{total},cached:{cached} "
            f"request_id={request_id}",
            flush=True,
        )

print(
    f"[smoke] PASS requests={request_count} "
    f"ttft_median={statistics.median(ttfts):.1f}ms "
    f"ttft_min={min(ttfts):.1f}ms ttft_max={max(ttfts):.1f}ms",
    flush=True,
)
PY
