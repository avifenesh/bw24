# CX retry contract progress - 2026-08-07

Status: complete locally on `lane/cx-retry-contract`.

## Fix

- Both `cmd_tx.send(Cmd::Generate(...))` failure branches now use the engine retry-contract
  response builder.
- The response is `503 server_error` with `code: "overloaded"`, `Retry-After: 2`, and
  `retry-after-ms: 2000`.
- The 2-second hint and the supervisor's `2 * attempt` ladder share
  `WORKER_RESPAWN_BACKOFF_BASE_S`, so the HTTP response cannot drift from the first/default
  respawn delay.
- Retryable responses continue to omit `x-should-retry`; the established contract uses
  `x-should-retry: false` only for requests whose identical bytes cannot succeed. The regression
  test proves this 503 never carries the contradictory `false` override.

Fix and regression-test commit: `f966805c fix(serve): honor retry contract on worker send failure`.

## Regression receipt

The handler-level test disconnects the worker receiver and calls both completion endpoints.

Before the fix:

```text
$ cargo test -p memra-server command_send_failure_obeys_the_retry_contract -- --nocapture
assertion `left == right` failed
  left: None
 right: Some("2")
test result: FAILED. 0 passed; 1 failed; 106 filtered out
```

After the fix:

```text
$ cargo test -p memra-server command_send_failure_obeys_the_retry_contract -- --nocapture
test tests::command_send_failure_obeys_the_retry_contract ... ok
test result: ok. 1 passed; 0 failed; 106 filtered out
```

Required gates:

```text
$ cargo check -p memra-server
Finished `dev` profile [unoptimized + debuginfo] target(s) in 6.15s
exit 0

$ cargo test -p memra-server
running 107 tests
test result: ok. 107 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.36s
exit 0
```

The full test run also printed `User defined signal 1`; the suite includes SIGUSR1 coverage in
the dark-lane checkpoint preemption tests. Cargo exited 0 with all 107 tests passing.

## Other bare 503 paths

The remaining direct `StatusCode::SERVICE_UNAVAILABLE` response producers are control-plane
probes, not OpenAI completion errors:

- `health_live`: an unhealthy worker returns a 503 health payload without retry headers.
- `health_ready`: loading, draining, or otherwise unready returns a 503 readiness payload without
  retry headers.

These are intentionally left unchanged. Drain rejections and engine `Overloaded` failures already
use retry-contract responses, and there are no other bare 503 completion paths in
`crates/memra-server/src`.
