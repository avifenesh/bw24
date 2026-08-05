# API-key management (lane/api-keys) — results, 2026-08-05

Rig: local RTX 5090 laptop, model Qwen3.5-9B NVFP4 (the serve gate default), bulk tier
(`MEMRA_SERVE_SPEC=0`), `MEMRA_PREFIX_CACHE_MB=1024`. Single interleaved run per gate
(these are behavioral pass/fail gates, not perf medians).

## What shipped

- `crates/memra-server/src/auth.rs` — keyring (TOML file / inline env), SHA-256-only
  storage, mtime-poll hot reload (<=2s; broken rewrite keeps the old ring), the auth
  composition law (`authenticate_with`), tenant namespace scoping
  (`t:<tenant>\x1f<salt>`), `--gen-key`/`--revoke-key` CLI.
- `main.rs` — `authenticate` (401 unknown / 403 disabled), `lane_for_tenant`
  (batch-class keys default harvest, 403 off interactive), per-tenant RAII in-flight
  gauge, `RateLimit::at_admit` min(override, global) law, `[meter] admit` seam,
  tenant-scoped `cache_ns` on both completion routes.
- Docs: SERVING.md "API keys" section, FLAGS.md `MEMRA_API_KEYS` entry.
- Gate: `tools/apikeys-gate.sh` -> `apikey_gate.py`.

## Live gate: 18/18 PASS (`apikey-gates.jsonl`, `gate-run-console.log`)

Two-tenant isolation proof (cache-hit oracle, the CacheProbe method):

| step | key | prompt | prompt_tokens | cached_tokens | law |
|---|---|---|---|---|---|
| seed | acme k1 | P | 94 | 0 | cold |
| hit | acme k2 | P | 94 | **94** | same tenant, different key -> SHARE |
| miss | blue | P | 94 | **0** | cross-tenant -> INVISIBLE |
| seed | blue | Q | 101 | 0 | cold |
| hit | blue | Q | 101 | **101** | own entry hits (cache alive, miss = isolation) |
| miss | acme k1 | Q | 101 | **0** | reverse direction also invisible |
| miss | acme k1 | P @salt=proj-x | 94 | 0 | salt sub-scopes WITHIN tenant |
| hit | acme k1 | P @salt=proj-x | 94 | 94 | salted namespace hits itself |

Auth refusals: no key -> 401, garbage key -> 401, pre-boot-revoked key -> 403
("api key is disabled"). Hot revoke (`--revoke-key` on a live server): 403 within the
2s poll; the sibling same-tenant key kept working.

Rate-limit headers: `rate_limit = 2` key reported limit=2 remaining=1;
uncapped key reported the global interactive cap (64). Batch-class key:
`x-lane: interactive` -> 403 with the actionable message; judge admitted.

Meter seam: `[meter] admit id=<x-request-id> tenant=<t> lane=<l> model=<m>` lines
present for default/acme/blue/bulk (sample in `meter-lines-sample.log`).

## Back-compat: PASS

- `MEMRA_API_KEY` single key alongside the keyring -> 200 as tenant `default`
  (gate `single-key-200`).
- `tools/serve-smoke.sh` (no keyring, the daily-driver path): **0 failed**, spec==plain
  greedy identity included (`serve-smoke-console.log`).
- `tools/serve-st-gate.sh`: see `serve-st-console.log`.
- Unit tests: 59/59 memra-server bin tests (9 new auth + 3 new handler-law tests).

## Contract notes

- No keyring configured -> `cache_ns` is the RAW salt, byte-identical to PC-ISO; the
  namespace only changes shape when `MEMRA_API_KEYS` is set (a keyring deploy is a new
  cache generation — old default-namespace entries are simply invisible, not corrupted).
- The global lane cap stays authoritative: per-key `rate_limit` can only narrow.
- Keyring + single key COMPOSE; the single key's `MEMRA_COMPAT=openai` default is
  untouched.
