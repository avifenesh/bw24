# CX bare-503 closure - 2026-08-07

Status: complete locally on `lane/cx-bare503`.

Train tip: `80f47796`.

## Commits

- `4409c50d fix(serve): add retry contract to liveness probes`
- `89936f75 fix(serve): add retry contract to readiness probes`

## Per-path dispositions

### `health_live` (`GET /health`, `GET /livez`)

- Disposition: retryable 503, not a permanent misconfiguration.
- Conditions: worker reload/respawn, worker death, scheduler stall, or a latched GPU fault.
- Body and status remain the health-probe contract: 503 with `status: "unhealthy"` and the
  existing quoted `detail`.
- Headers now pass through the shared retry-contract builder with the supervisor's first
  respawn delay: `Retry-After: 2` and `retry-after-ms: 2000`.
- `x-should-retry: false` is absent because retrying the probe after the supervisor has acted is
  valid.
- Regression test: `liveness_failure_obeys_the_retry_contract`.

### `health_ready` (`GET /readyz`)

- Disposition: retryable 503, not a permanent misconfiguration.
- Worker-unavailable conditions use `Retry-After: 2` and `retry-after-ms: 2000`, tied to the
  worker respawn base.
- Draining uses `MEMRA_DRAIN_S`, clamped to 1..=60 seconds, plus the matching millisecond twin.
- Body and status remain the readiness contract: 503 with `status: "not_ready"` and the existing
  `detail`.
- `x-should-retry: false` is absent.
- Regression tests: `readiness_failure_obeys_the_retry_contract` and the drain-specific
  assertions in `draining_rejects_new_requests_with_503_and_retry_after`.

No listed path represented a permanent configuration error, so no 503 was converted to a 4xx or
500 status.

## Implementation notes

- Extracted `retry_contract_response` so OpenAI error responses and health-probe JSON bodies share
  the exact header logic without changing their body schemas.
- The helper emits integer retry seconds in the 1..=60 window and an agreeing
  `retry-after-ms`.
- The existing drain completion response now uses the same helper, removing its duplicate header
  construction.

## Regression receipts

Before the fixes:

```text
$ cargo test -p memra-server liveness_failure_obeys_the_retry_contract -- --nocapture
assertion `left == right` failed
  left: None
 right: Some("2")
test result: FAILED. 0 passed; 1 failed
exit 101

$ cargo test -p memra-server readiness_failure_obeys_the_retry_contract -- --nocapture
assertion `left == right` failed
  left: None
 right: Some("2")
test result: FAILED. 0 passed; 1 failed
exit 101

$ cargo test -p memra-server draining_rejects_new_requests_with_503_and_retry_after -- --nocapture
assertion `left == right` failed
  left: None
 right: Some("30")
test result: FAILED. 0 passed; 1 failed
exit 101
```

Focused retry-contract suite after the fixes:

```text
$ cargo test -p memra-server retry_contract -- --nocapture
running 3 tests
test tests::command_send_failure_obeys_the_retry_contract ... ok
test tests::liveness_failure_obeys_the_retry_contract ... ok
test tests::readiness_failure_obeys_the_retry_contract ... ok
test result: ok. 3 passed; 0 failed; 110 filtered out
exit 0
```

Required gates:

```text
$ cargo check -p memra-server
Finished `dev` profile [unoptimized + debuginfo] target(s) in 7.14s
exit 0

$ cargo test -p memra-server
running 113 tests
test result: ok. 113 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.40s
exit 0
```

The full test run printed `User defined signal 1` from the dark-lane checkpoint-preemption
coverage. Cargo exited 0 and all 113 tests passed.
