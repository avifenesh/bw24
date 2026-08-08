# Research-preview API keys

memra already owns the key machinery. This trial uses the existing
`memra-server --gen-key` / `--revoke-key` CLI and
`MEMRA_API_KEYS=/etc/memra/keys.toml`; `deploy/glue/keyctl.sh` is only a
permission-safe convenience wrapper.

## Exact semantics

- The plaintext key is printed once. The ring stores only its SHA-256 hash,
  tenant, lane class, safe revoke prefix, enabled state, creation time, and
  optional `rate_limit`.
- `rate_limit` means **concurrent in-flight requests**, not requests/minute or
  tokens/minute. The effective cap is the lower of this per-key value and the
  global lane cap. A request that arrives while its tenant holds every configured
  slot gets an OpenAI-shaped `429 rate_limit_exceeded` before worker admission,
  with `Retry-After` and zero `x-ratelimit-remaining`.
- `interactive` keys use the protected interactive lane by default. `batch`
  keys default to harvest and cannot claim `x-lane: interactive`.
- The server polls the keyring mtime and applies a valid change within two
  seconds. A malformed rewrite keeps the previous ring; auth never fails open.
- Revoked keys return 403. Missing or unknown keys return 401.
- A tenant is also a cache and metering boundary. Give each directly issued
  preview user a distinct tenant rather than issuing many people keys under one
  shared tenant. Keys under the same tenant share its in-flight gauge.

The RunPod provisioner creates the ring as `root:memra 0640`: root mutates it,
and the unprivileged `memra-server` process can read it. `keyctl.sh` preserves
that mode and serializes owner mutations with `flock`.

## Trial key profiles

| Purpose | Tenant example | Lane | Concurrent cap |
|---|---|---:|---:|
| Direct preview recipient | `preview_001` | interactive | 1 |
| Poe shim | `poe` | interactive | 1 |
| AI Horde worker | `horde` | batch | 1 |
| Owner smoke/admin client | `owner` | interactive | 2-4 |

The SillyTavern-community pattern is long-context, always-on, and
rate-limit-seeking. Issue separate keys manually, keep the default cap at one,
and revoke a misbehaving recipient without disturbing other tenants. Do not
publish one shared bearer key in a public post.

## Mint

The wrapper defaults to interactive and one concurrent request:

```bash
sudo /opt/memra/deploy/glue/keyctl.sh mint preview_001
```

Explicit service examples:

```bash
sudo /opt/memra/deploy/glue/keyctl.sh \
  mint poe --lane interactive --rate-limit 1
sudo /opt/memra/deploy/glue/keyctl.sh \
  mint horde --lane batch --rate-limit 1
```

The only stdout line from a live mint is the plaintext `mk-...` key. Put it
directly into the recipient's secure channel or the relevant root-readable
service environment file. Do not paste it into tickets, chat logs, command
arguments, or the progress receipt.

Preview the exact mutation without needing a pod or binary:

```bash
deploy/glue/keyctl.sh --dry-run \
  mint preview_001 --lane interactive --rate-limit 1
```

## Revoke

Read the safe `prefix` field from `/etc/memra/keys.toml`, then revoke it:

```bash
sudo /opt/memra/deploy/glue/keyctl.sh \
  revoke mk-preview_001-012345abcdef
```

The wrapper intentionally refuses a full plaintext key on the command line so
it cannot enter shell history. The server CLI requires an unambiguous match and
disables exactly one entry. Allow two seconds for hot reload; the old bearer
should then receive 403 without a server restart.

## Direct CLI

For reference, the underlying commands are:

```bash
memra-server --gen-key TENANT \
  --lane interactive \
  --rate-limit 1 \
  --keys /etc/memra/keys.toml
memra-server --revoke-key SAFE_PREFIX \
  --keys /etc/memra/keys.toml
```

The implementation contract is documented in `docs/SERVING.md` under
**API keys - multi-key tenant auth** and implemented in
`crates/memra-server/src/auth.rs`.
