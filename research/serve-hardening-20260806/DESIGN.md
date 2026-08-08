# serve-hardening — G5 / G6 / G24

Lane: `lane/serve-hardening` off `origin/restructure/public-split`. Rig: local RTX 5090 Laptop
(driver 595.84, CUDA 13.1). Scope: **memra-server crate only** — no CUDA kernel changes, so no
kernel-check/run-gen/run-spec numbers move.

Source of the gaps: `~/projects/darklanes/exp/provider-table-stakes-20260806.md` (G5 worker death
invisible, G6 every engine fault is HTTP 400, G24 no GPU-fault detection). Launch 2026-08-27;
OpenRouter delists below 80% uptime, where 401/402/404/5xx and mid-stream errors count against a
provider and 400/413/429/403 do not.

---

## 1. What each endpoint reports now

### `/health` and `/livez` (same handler)

"Should this process be restarted?" Derived **only** from worker state — never from "the HTTP task
is scheduled".

| worker phase | status | body `status` | reason |
|---|---|---|---|
| `loading` | 503 | `unhealthy` | weights not resident; the process answers nothing yet |
| `idle` | 200 (any beat age) | `ok` | the worker blocks in `rx.recv()`; an idle server legitimately stamps nothing for hours |
| `busy`, beat advancing | 200 | `ok` | progress |
| `busy`, beat older than `MEMRA_HEALTH_STALL_S` | 503 | `unhealthy` | wedged (hung kernel / deadlock / hung driver call) |
| `dead` (panic latch) | 503 | `unhealthy` | latched — instant, no threshold wait |
| GPU fault latched | 503 | `unhealthy` | fatal Xid or a probe that hung |
| draining | **200** | `draining` | a drain is a healthy deliberate shutdown; 503 here invites a SIGKILL mid-stream |

Bodies, captured from the wire (`logs/endpoints-live.txt`, `logs/worker-death.txt` — not
illustrative):

```
GET /health   200 {"status":"ok","models":["probe"],"worker":{"phase":"idle","beat_age_ms":63,
                   "tick_max_ms":0,"stall_threshold_ms":120000,"generation":0,"xid_warnings":0}}
GET /health   503 {"status":"unhealthy","models":["probe"],"worker":{"phase":"dead",...},
                   "detail":"worker thread panicked: <the panic payload, quoted>"}
GET /health   200 {"status":"draining","models":["probe"],"worker":{"phase":"busy",
                   "beat_age_ms":126,"tick_max_ms":253,...}}
GET /readyz   503 {"status":"not_ready",...,"detail":"draining (shutdown in progress)"}
```

`detail` is added on a red, carrying the **quoted** cause (the panic payload text, or the GPU
watcher's reason) — not an inferred one.

`PHASE_LOADING` is deliberately **not** reachable over HTTP on a first load: `main` binds the
listener only after `worker::spawn` returns ready, so the load window is connection-refused
rather than 503. k8s probes and `serve-fleet.sh` treat refused and 503 identically, so the
verdict is the same either way. It IS reachable during a **respawn** (socket already bound,
worker reloading weights) — which is the case that matters, since that is when a bound port
must not be reported ready.

### `/readyz` (new)

"Should traffic be routed here?" Ready = model loaded AND worker alive AND not draining. Unready is
**not** a restart request. 200 `{"status":"ready",...}` / 503 `{"status":"not_ready","detail":...}`.

Deliberately NOT readiness-affecting: queue depth. The interactive lane queues FIFO and never
sheds, so a deep queue is work in progress; capacity backpressure belongs on the request path where
a client can act on it (§2).

### Why the split, and where memra now sits

k8s deprecated `/healthz` at v1.16 in favor of `/livez` + `/readyz`; KServe's Open Inference
Protocol v2 (Triton) uses `v2/health/live` + `v2/health/ready`. vLLM has **no** readiness endpoint
(its `/health` 503s only on `EngineDeadError`), TGI has a single `/health`. memra now has both, and
`/health` keeps its historical name so every existing script keeps working.

Consumers updated to ask the right question:

* `tools/serve-proxy.py` → `/readyz` (rotation), with a `/health` fallback if a replica 404s the
  new route, so a mixed-version fleet degrades instead of marking every backend DOWN.
* `tools/serve-fleet.sh` → `/health` (restart decision) — unchanged call, but it now actually
  restarts a replica whose worker died instead of seeing green forever.

---

## 2. Error taxonomy — before / after

**Before**, `main.rs` had exactly one worker-error arm:

```rust
Event::Error(msg) => { return bad_request(&msg, None); }   // 400 invalid_request_error
```

Every CUDA fault, VRAM exhaustion, tokenizer failure, admission shed and graph fault arrived at the
client as `400 invalid_request_error`. Wrong in both directions, and both cost money:

* openai-python retries 408/409/429/≥500 **only** — a transient capacity blip became a hard
  user-visible failure with no retry;
* a router cannot tell "your request was malformed" from "my GPU fell over", so it keeps sending
  traffic to a broken box instead of failing over.

**After** — `worker::EngineError { class, message, param }`, with the class assigned by the
**producer** (a string classifier in the HTTP layer would drift from the code that raises the
error). One deliberate quoted-text rule: `is_cuda_oom(message)` promotes an engine fault to
`Overloaded`, using the same predicate as the step-OOM park path so the two can never disagree
about what an OOM is.

| condition | before | after | `type` | `code` | retry headers |
|---|---|---|---|---|---|
| chat template failure | 400 invalid_request_error | 400 | `invalid_request_error` | — | `x-should-retry: false` |
| empty prompt after tokenization | 400 | 400 | `invalid_request_error` | — | `x-should-retry: false` |
| `response_format` / grammar rejected | 400 | 400 | `invalid_request_error` | — | `x-should-retry: false` |
| prompt ≥ context cap | 400, no code | 400 | `invalid_request_error` | `context_length_exceeded` | `x-should-retry: false` |
| unknown model id | 400, no code | 400 | `invalid_request_error` | `model_not_found` | `x-should-retry: false` |
| dark-lane QoS shed | 429 + bare-string body | 429 | `rate_limit_error` | `rate_limit_exceeded` | `Retry-After: 2`, `retry-after-ms: 2000` |
| cache alloc failed (CUDA OOM text) | **400** | **503** | `server_error` | `overloaded` | `Retry-After: 5`, `retry-after-ms: 5000` |
| step OOM past its park budget | **400** | **503** | `server_error` | `overloaded` | same |
| worker channel closed mid-request | **500** "worker closed stream" | **503** | `server_error` | `overloaded` | same |
| step / prefill / batch-step error | **400** | **500** | `server_error` | `engine_error` | none |
| graph promote / graph step failed | **400** | **500** | `server_error` | `engine_error` | none |
| constraint mask / advance failed | **400** | **500** | `server_error` | `engine_error` | none |
| new request during a drain | 503, bare `Retry-After`, no code | 503 | `server_error` | `draining` | `Retry-After: 30`, `retry-after-ms: 30000` |
| unknown `x-lane` value | 400, **bare-string** body | 400 | `invalid_request_error` | `invalid_lane` | `x-should-retry: false` |
| batch-class key claiming `x-lane: interactive` | 403, no `x-should-retry` | 403 | `authentication_error` | — | `x-should-retry: false` |

The last two rows are the same kind of find as the drain row — swept up on a second pass over the
whole surface rather than only the engine's own errors. `lane_for_tenant` answered an unknown
`x-lane` with `{"error": "unknown x-lane \"turbo\""}`: a bare STRING where the rest of the surface
puts an object, so `e.body["error"]["type"]` is an index error and the message renders blank in
SDKs that read the standard shape. And the handler-layer refusals (auth, lane) never carried
`x-should-retry: false`, even though they are exactly as unretryable as the engine-layer 400s that
do — so `error_response` now routes through `error_response_coded`, which attaches the header on
any 4xx except the three genuinely-retryable client statuses (429, 408, 409).

The drain row is a gap this lane found while probing, not a pre-existing plan: `drain_response`
predates the taxonomy and was the last 503 on the surface emitting a bare `Retry-After` with no
`code` and no `retry-after-ms` twin — so a client that reads only the ms header (openai-python
reads it **first**) saw no window at all on memra's single most predictable outage. It now goes
through the same contract, clamped to 60 s.

Decisions worth stating, because each was a fork:

* **Unknown model is 400, not 404.** OpenRouter counts 404 against provider uptime and excludes
  400. "You asked for a model this endpoint does not serve" is squarely a client error; taking an
  uptime hit for someone else's typo would be self-punishment. Clients branch on `code` either way.
* **Shed is 429, capacity is 503.** A dark-lane shed is a QoS decision with a known short window —
  429 + Retry-After, uptime-neutral, and OpenRouter's own guidance prefers an early 429 to
  queueing. Being out of VRAM is not something a client can fix by waiting a fixed window; OpenAI
  itself serves overload as 503, and "a 429 a client cannot fix by waiting should not be a 429".
* **Engine faults carry no Retry-After.** They are not time-bounded — this process may need a
  restart. The SDK's own exponential backoff (500s are retryable by default) is the honest
  behavior; promising a window we cannot honor is not.
* **A closed worker channel is 503, not 500.** It means the worker thread is gone and the
  supervisor is already acting; a retry may well land on a restarted process.

### Retry contract (verified against client code, not docs)

* `Retry-After` is **integer** delay-seconds (RFC 9110 §10.2.3 — a float is unparseable) and always
  ≤ 60: litellm honors the header only for `0 < v ≤ 60`, and openai-python **abandons** the retry
  entirely past its 120 s `MAX_RETRY_AFTER_DELAY`.
* `retry-after-ms` is emitted alongside and always agrees (openai-python reads it **first**).
* `x-should-retry: false` on the unfixable 400 classes, so a client retrying by status alone does
  not hammer a request that can never succeed.

### Mid-stream failures

Once the first byte of a 200 is written the response is committed — there is no status code left to
change, and OpenRouter cannot fail over. Two consequences, both implemented:

1. `peek_shed` (dark lanes) resolves the admission verdict **before** headers, converting a
   would-be mid-stream death into a clean pre-header 429/503 the client's own retry handles. Its
   body used to be `{"error": "<string>"}` — a bare string where every SDK expects an object, so
   shed errors rendered blank client-side; it now goes through the same builder as every other
   error.
2. A genuine mid-stream fault emits the class-derived error **object** as a `data:` chunk, then
   `[DONE]`, then closes. No named `event: error` on the OpenAI surface — OpenAI clients only parse
   `data:` lines, so a named event reads as a silent hang.

---

## 3. G5 — worker supervision

`worker::spawn` was `std::thread::spawn(move || run(..))`. A panic inside unwound **that thread
only**: the process kept serving HTTP, `/health` stayed green forever, and every request blocked or
died on a closed channel.

The spawned thread is now a supervisor:

1. runs the scheduler inside `catch_unwind`;
2. on catch, `health.mark_dead(<quoted panic payload>)` — `/health` and `/readyz` flip in
   milliseconds, no staleness threshold to wait out;
3. attempts `MEMRA_WORKER_RESPAWN` (default **1**) respawns with backoff (`2 s × attempt`),
   bumping `health.generation` so a recovery is observable on `/health`;
4. otherwise prints a FATAL line and `std::process::exit(70)` (EX_SOFTWARE — distinguishable from
   the exit 1 of a bad config) so the supervisor restarts the unit whole.

One respawn attempt, deliberately: CUDA errors are sticky per process (after an OOM or an Xid the
context is poisoned), so an in-process retry is a long shot — worth one try because it saves a
~120 s weight reload when it works, and worth no more because a respawn loop against a poisoned
context is a box that looks alive and serves nothing.

The supervisor **owns the command `Receiver`** across restarts (`run()` now borrows it). Dropping
it would close the channel and make every subsequent HTTP handler's `send` fail permanently — the
exact invisible death this lane removes.

A clean `run()` return (channel closed = shutdown) is not a fault and never respawns.

### The stall threshold, derived

`MEMRA_HEALTH_STALL_S` default **120 s**. It must cover the longest *legitimate* single scheduler
iteration, which is a prefill tick, not a decode step: `prefill_tick` primes up to
`MEMRA_PREFILL_TICK` (1024) tokens per active session and loops over every active session inside
one iteration. At `MEMRA_MAX_SESSIONS` = 64 that is 64 × 1024 = 65,536 primed tokens in a single
pass; at this rig's measured 4k-prefill rate of ~1.2k tok/s (`research/memra-vs-llama-daily-
20260805/`) that is ≈ 55 s. 120 s is that worst case with ~2.2× margin, and it is the same number
`tools/serve-fleet.sh` already uses for `LOAD_GRACE` — so the app threshold, the bash supervisor's
grace and the systemd `StartLimitIntervalSec` sizing are one number instead of three.

`/health` publishes `tick_max_ms` (longest iteration this process actually observed) so raising the
threshold is ever a measured decision, not a guess.

The bound is a **per-instance field**, not a process-global `OnceLock`: with a global, observing the
stall branch at all would require a 120 s sleep, i.e. the branch that decides "restart this box"
would ship untested.

---

## 4. G24 — GPU-fault detection

Three independent detectors feeding one latch (`gpu_faulted`), on their own threads:

1. **Xid tail.** `/dev/kmsg` first (privileged deployments), falling back to
   `journalctl -k -n 0 -f`. Fatal set **48** (double-bit ECC), **64** (row-remap failure), **79**
   (GPU off the bus), **94/95** (contained/uncontained ECC), **119/120** (GSP RPC timeout) latch
   unhealthy; other Xids (13/31 app errors, 43/45 teardown, 62/63 remap pending) increment
   `xid_warnings` on `/health` so an operator can see a card degrading before it wedges.
2. **`nvidia-smi` value probe** every `MEMRA_GPU_WATCH_S` (60 s): uncorrectable volatile ECC,
   pending retired pages, row-remap failure. Faults **only on a definite value** — `[N/A]` is not
   evidence.
3. **Probe hang = the alarm.** Blackwell's worst wedge class (Xid 119/120) emits nothing to the
   process *and hangs `nvidia-smi` itself*. The probe therefore runs as a child killed at
   `MEMRA_GPU_PROBE_TIMEOUT_S` (10 s), and exceeding the deadline is treated as a fatal fault.
   Health reads only atomics, so a hung probe can never block a health answer.

Rig facts this was built against (measured on driver 595.84, not assumed):

* `nvidia-smi --query-gpu=xid.pending` → `Field "xid.pending" is not a valid field to query.` —
  so there is no supported "give me the Xid" query; the watcher degrades its field list (RICH →
  MIN) rather than dying when a field is rejected.
* ECC / retired-page / remap fields all return `[N/A]` on this RTX 5090 Laptop GPU — detector 2 is
  a no-op here and will only ever fire on a datacenter card. Stated, not hidden.
* `/dev/kmsg` is `Permission denied` and `dmesg` fails with
  `read kernel buffer failed: Operation not permitted` (`kernel.dmesg_restrict = 1`);
  `journalctl -k` works unprivileged. Hence the fallback order, and the systemd unit documents
  `CAP_SYSLOG` as the alternative.
* `nvidia-smi` missing entirely is "tool absent", not a fault — a container without the CLI must
  not report a wedged GPU.

A GPU fault **survives** `mark_ready()`: a respawned worker thread on a wedged card is not
recovery, and only a fresh process (new CUDA context) can be.

---

## 5. systemd contract

`deploy/systemd/memra-server.service`. What the server promises the unit:

* `READY=1` only after every model is resident **and** the socket is bound — so `systemctl start`
  returning means "can serve", not "process exists".
* `WATCHDOG=1` at half `WATCHDOG_USEC`, and **only while `live()` is Ok**. A wedged worker simply
  stops pinging and `Restart=` fires.
* `STOPPING=1` + `EXTEND_TIMEOUT_USEC` on SIGTERM so a legitimate drain is not SIGKILLed.
* exit **70** when the worker is unrecoverable (vs exit 1 = bad config).

Implemented with `std` only (a `UnixDatagram` to `$NOTIFY_SOCKET`), a complete no-op when the env
var is absent. Stated limitation: an abstract socket (`@`-prefixed) is not addressable from stable
std, so it disables the notifier with one warning instead of pretending to work — system units get
the path form.

Directive choices that are the actual content of that file:

* `StartLimitIntervalSec=3600` / `StartLimitBurst=4`. **The trap:** the defaults (10 s / 5) are
  sized for daemons that start in milliseconds. With a ~120 s load, five starts cannot fit in ten
  seconds, so the limiter can never trip and a crash loop restarts *forever* instead of failing and
  paging a human.
* `OOMPolicy=kill`. The default `stop` only reaps the offending process; with a multi-threaded
  server the kernel OOM killer can take out one **thread** — classically a worker — leaving a
  process that accepts connections and can never serve them. That is precisely this lane's failure
  mode. (Host memory only; CUDA OOM is HTTP 503, never a process kill.)
* `RestartSec=10` + `RestartSteps=4` + `RestartMaxDelaySec=160` (systemd ≥ 254): a card that just
  threw an Xid needs the driver to settle; hammering it makes recovery less likely.
* `WatchdogSec=180` > `MEMRA_HEALTH_STALL_S` (120) — the app-level bound must be the first to fire,
  so the supervisor sees an honest 503 before it starts killing.
* `TimeoutStartSec=600` (a cold NVMe load of a large bank is slower than the ~120 s page-cache
  case); `TimeoutStopSec=60` > `MEMRA_DRAIN_S` (30).
* Deliberately **not** `ProtectSystem=strict`: model paths, `/dev/nvidia*` and the CUDA cache all
  need real filesystem access, and a wrong strict sandbox fails at load time in ways that look like
  a model bug. Start loose, tighten with receipts.

---

## 6. Tests

`cargo test -p memra-server`: **91 pass** (82 before this lane). Raw log:
`logs/cargo-test-memra-server.txt`.

New, taxonomy:

* `taxonomy_maps_every_class_to_its_status_and_code` — class by class, plus an exhaustive sweep so
  a new class cannot be silently missing from the match.
* `a_cuda_oom_message_is_capacity_503_not_a_500` — the one quoted-text promotion.
* `retry_headers_follow_the_sdk_contract` — integer, ≤ 60, `retry-after-ms` agrees, no
  contradictory `x-should-retry`.
* `unfixable_client_errors_say_x_should_retry_false` — and carry no retry window.
* `a_closed_worker_channel_is_503_not_500`.
* `a_dark_lane_shed_is_429_with_an_openai_object_body` — pins the bare-string regression.
* `interactive_never_peeks_so_its_first_token_is_not_held`.
* `handler_layer_refusals_are_openai_objects_with_x_should_retry` — pins the *other* bare-string
  body (unknown `x-lane`) plus the header on both lane refusals, so the handler layer and the
  engine layer cannot drift apart again.

New, health (the integration arm the lane was asked for):

* `health_is_green_only_while_the_worker_is_alive` — asserts green, kills inference through the
  same handle a panic uses, asserts `/health` **and** `/readyz` flip to 503 with the quoted cause
  in `detail`, then asserts a successful respawn clears it.
* `a_wedged_gpu_flips_health_even_though_the_worker_thread_is_fine` — the G24 path, including that
  the GPU latch is not cleared by an in-process respawn.

In `health.rs` (7): Xid classification across both kernel line forms and every fatal id;
idle-healthy-at-any-age vs busy-stalls (at a 20 ms injected bound, so the branch is actually
reached); death and GPU-fault latching without a threshold wait; readiness-off-while-draining with
liveness-on; loading is neither live nor ready + generation bump; `tick_max_ms` records the longest
iteration; the `nvidia-smi` CSV scan faults only on definite values.

Updated: `error_bodies_use_the_openai_object_shape` and
`stream_worker_error_is_a_data_chunk_not_a_named_event` now assert `param`/`code` exactly rather
than mere presence; `draining_rejects_new_requests_with_503_and_retry_after` gained the
liveness-200 / readiness-503 split.

---

## 7. Live verification (on the wire, not in a test harness)

Unit tests pin this lane against a **fake** worker. That is not sufficient evidence for code
whose entire job is "notice that the real GPU worker died", and this lane already proved why:
the first supervisor deadlocked startup with a fully loaded model and an unbound socket, and no
unit test could have seen it. So both halves are also probed against a real CUDA worker, by two
committed scripts whose raw output is in `logs/`:

* **`probe-endpoints.sh`** → `logs/endpoints-live.txt` (+ `logs/endpoints-live-server.log`).
  Endpoint payloads during load / ready / drain, the reachable G6 arms (unknown model,
  over-context prompt, dark-lane shed on both the blocking and streaming surfaces), streaming
  still intact, the drain split, the exit code, and the G24 watcher's startup lines.
* **`probe-worker-death.sh`** → `logs/worker-death.txt` (+ two server logs). The G5 ladder end
  to end, via `MEMRA_PANIC_AFTER` (fault-injection door, `docs/FLAGS.md §Server`).

Each script states the knob it uses and why an arm is reachable, because two arms are only
reachable with help: `MEMRA_CTX=256` makes an over-context prompt possible at all (the model's
own 262,144 cap is not reachable with a test prompt), and `MEMRA_LANE_MAX_HARVEST=0` makes the
dark-lane shed deterministic instead of dependent on a live SLO breach — the same
`EngineError::rate_limit`, the same 429 path. Engine-fault (500) arms are **not** forced: faking
a CUDA fault would prove nothing about a real one, so those stay unit-pinned.

Measured results:

| arm | result |
|---|---|
| load window | connection refused on all three routes (bind follows load, §1) |
| ready | `/health` `/livez` 200 `ok`, `/readyz` 200 `ready`, worker block populated |
| unknown model | 400 `invalid_request_error` / `model_not_found` / `param:"model"` / `x-should-retry: false` |
| over-context prompt | 400 `context_length_exceeded` / `param:"messages"` / `x-should-retry: false`, message quotes both numbers (`prompt (410 tok) >= context cap (256)`) |
| bad `x-lane` | 400 `invalid_request_error` / `invalid_lane` / `param:"x-lane"` / `x-should-retry: false`, message lists the legal values |
| dark-lane shed | 429 `rate_limit_error` / `rate_limit_exceeded`, `Retry-After: 2` + `retry-after-ms: 2000`; **429 pre-header on the streaming surface too** |
| worker panic | `/health` `/livez` 503 `unhealthy`, `/readyz` 503 `not_ready`, all three carrying the quoted panic payload in `detail`, within ~200 ms of the panic |
| respawn | weights reloaded, `generation` 0 → 1, back to 200, and the respawned worker served a real completion (200) |
| request in the dead window | **served by the respawn** — the supervisor owns the command `Receiver` across restarts, so a queued request survived rather than erroring |
| `MEMRA_WORKER_RESPAWN=0` | process exit **70** (EX_SOFTWARE), port refused afterward — vs the pre-lane behavior, a permanently-200 `/health` in front of a box serving nothing |
| drain | `/health` `/livez` 200 `draining`, `/readyz` 503 `not_ready`, new completion 503 `draining` + `Retry-After: 30`, the in-flight 900-token generation completed, exit 0 |
| G24 startup | `Xid source: journalctl -k -f (/dev/kmsg unreadable — kernel.dmesg_restrict)`, `every 60s, probe deadline 10s, fatal Xid [48, 64, 79, 94, 95, 119, 120]` |

Two assumptions the wire corrected, recorded because both were wrong in the first draft:

1. **A drain is observable on fresh connections, not only pooled ones.** axum stops accepting
   when its shutdown *future* resolves, and memra's future **is** the drain loop — so the
   listener keeps accepting for the whole drain window. That is what makes the 503-on-a-new-
   completion path real rather than test-only.
2. **The panic injection had to be one-shot per process.** `n_completed` is per-`run()`, so a
   per-run trigger re-fired on the respawned worker's first request: the respawn reloaded, went
   green with `generation:1`, then panicked again and exited 70 — leaving "did the recovery
   actually serve traffic?" unanswerable. (Also: the very first probe run pointed at port 8181
   and diligently reported the owner's `llama-server` 404s as memra's. The script now refuses a
   port that is already in LISTEN.)

---

## 8. Gates

See `GATES.md` for verdicts and `logs/` for the raw output of each.
