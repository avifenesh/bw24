# Step-3.7-Flash serving trial runbook

This runbook operates the research-preview trial after
`deploy/runpod/provision.sh` has prepared the two-card pod. It does not download
models, build memra, or replace the RunPod lane's provisioning and target-rig
correctness receipts.

Trial surfaces:

- direct OpenAI-compatible API: `https://api.tiyuvta.ai/v1`
- AI Horde text worker: `memra-research-preview-runpod`
- Poe server bot: `https://api.tiyuvta.ai/poe`
- local operator endpoints: `http://127.0.0.1:8002`
- durable fleet ledger: `/var/lib/memra/receipts/fleet.jsonl`

## 1. Preflight before opening traffic

Do not open any public surface until the RunPod lane has recorded the target-rig
`kernel-check`, `run-gen` argmax, and `run-spec` K=1..8 gates for the exact
runtime commit and staged model bytes.

Confirm the provisioner left these contracts in place:

```bash
test -x /usr/local/bin/memra-server
test -f /etc/memra/runpod.env
test -f /etc/memra/keys.toml
test -f /var/lib/memra/receipts/deployment.env
test -f /var/lib/memra/receipts/step-model-manifest.txt
systemctl cat memra-server.service >/dev/null
systemctl cat memra-fleet-meter.timer >/dev/null
test "$(findmnt -no TARGET --target /data)" = /data
test "$(findmnt -no TARGET --target /scratch)" = /scratch
```

The last two checks prevent a missing volume from silently placing durable
receipts on the pod root disk or model traffic on persistent EBS.

Complete the two Cloudflare dashboard routes before launch:

1. `api.tiyuvta.ai`, no path, to `http://127.0.0.1:8002`.
2. `api.tiyuvta.ai`, path `^/poe(/.*)?$`, to
   `http://127.0.0.1:8081`. Put this more-specific route ahead of the
   catch-all route.

The first route is the one manual DNS action: saving the published hostname
creates the proxied record. See `TLS.md` and `poe-bot/REGISTRATION.md`.

Create permanent AI Horde and Poe accounts/configuration:

- register a non-anonymous AI Horde account and retain its API key;
- create the Poe server bot, keep it private, and retain its 32-character
  access key;
- leave Poe monetization, attachments, tools, and parameter controls off.

Install the bridge templates, then mint dedicated backend keys:

```bash
cd /opt/memra
sudo deploy/glue/horde-worker/setup.sh
sudo deploy/glue/poe-bot/setup.sh

sudo deploy/glue/keyctl.sh mint horde --lane batch --rate-limit 1
sudo deploy/glue/keyctl.sh mint poe --lane interactive --rate-limit 1
sudo deploy/glue/keyctl.sh mint owner --lane interactive --rate-limit 2
```

Each live mint prints the plaintext key once. Put it directly into the
corresponding root-readable environment file with `sudoedit`; never put it in a
receipt, command argument, ticket, or chat:

- `/etc/memra/horde-worker.env`: `AI_HORDE_API_KEY` and
  `MEMRA_HORDE_BACKEND_KEY`
- `/etc/memra/poe-bot.env`: `POE_ACCESS_KEY` and
  `MEMRA_POE_BACKEND_KEY`
- `/etc/memra/cloudflared-token`: Cloudflare connector token, mode `0600`

Validate every offline path before touching services:

```bash
cd /opt/memra
deploy/glue/trial-up.sh --dry-run
sudo deploy/glue/horde-worker/run.sh --check
sudo deploy/glue/poe-bot/run.sh --check
```

## 2. Create the receipt root and launch

Receipts belong on durable `/data`, not only the pod root disk or `/scratch`.
Keep the directory private because journals and service metadata are
operationally sensitive even though the glue disables prompt logging.

```bash
cd /opt/memra
umask 027
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RECEIPT_ROOT="/data/memra-trial/${RUN_ID}"
TRIAL_START_UTC="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
sudo install -d -m 0750 \
  "${RECEIPT_ROOT}/raw" \
  "${RECEIPT_ROOT}/observations" \
  "${RECEIPT_ROOT}/derived"
sudo chown -R "$(id -u):$(id -g)" "$RECEIPT_ROOT"

{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'trial_start_utc=%s\n' "$TRIAL_START_UTC"
  printf 'repo_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'repo_branch=%s\n' "$(git branch --show-current)"
  printf 'hostname=api.tiyuvta.ai\n'
} >"${RECEIPT_ROOT}/raw/context.env"
git status --porcelain=v1 >"${RECEIPT_ROOT}/raw/git-status.txt"
sha256sum /usr/local/bin/memra-server \
  >"${RECEIPT_ROOT}/raw/memra-server.sha256"
```

Capture the launch output before parsing or summarizing it:

```bash
set -o pipefail
sudo deploy/glue/trial-up.sh \
  --hostname api.tiyuvta.ai \
  --token-file /etc/memra/cloudflared-token \
  2>&1 | tee "${RECEIPT_ROOT}/raw/trial-up.log"
```

Then rerun the ten-row health matrix and retain it:

```bash
set -o pipefail
sudo deploy/glue/trial-up.sh --check \
  2>&1 | tee "${RECEIPT_ROOT}/raw/health-matrix-launch.log"
```

The matrix must show all ten rows as `PASS`: local readiness, public TLS,
OpenRouter 2.4 metadata, cache metering, admission lanes, missing-key rejection,
fleet metering, Horde polling, local Poe liveness, and public Poe routing.

## 3. Smoke before making Poe public

Use an owner key without placing it in shell history:

```bash
read -rsp 'Owner API key: ' MEMRA_SMOKE_API_KEY
printf '\n'
curl --fail --silent --show-error \
  --header @- \
  --header 'Content-Type: application/json' \
  --data '{"model":"stepfun/step-3.7-flash","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8,"temperature":0}' \
  https://api.tiyuvta.ai/v1/chat/completions \
  >"${RECEIPT_ROOT}/raw/direct-smoke.json" \
  <<<"Authorization: Bearer ${MEMRA_SMOKE_API_KEY}"
unset MEMRA_SMOKE_API_KEY
jq -e '.choices[0].message.content | type == "string"' \
  "${RECEIPT_ROOT}/raw/direct-smoke.json"
```

While the Poe bot is private:

1. Send one short prompt and one multi-turn prompt.
2. Occupy the Poe tenant's only memra slot with one request and confirm a
   second concurrent request receives retryable busy behavior.
3. Confirm visible output excludes model reasoning.
4. Confirm the Horde worker record is `online`, is not in maintenance, and
   advertises only `memra-research-preview`.
5. Make the Poe bot public only after these checks pass.

The direct API remains key-only. Issue separate one-slot tenants to preview
recipients; do not publish a shared bearer key.

## 4. Observation snapshot every 2-4 hours

Take raw snapshots first. Do not pipe a live service through a parser without
retaining the unmodified response or log.

```bash
cd /opt/memra
: "${RECEIPT_ROOT:?set RECEIPT_ROOT from the launch shell}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OBS="${RECEIPT_ROOT}/observations/${STAMP}"
install -d -m 0750 "$OBS"

curl --fail --silent --show-error \
  --output "${OBS}/readyz.json" \
  http://127.0.0.1:8002/readyz
curl --fail --silent --show-error \
  --output "${OBS}/models-openrouter.json" \
  'http://127.0.0.1:8002/models?schema=openrouter'
curl --fail --silent --show-error \
  --output "${OBS}/metrics.json" \
  http://127.0.0.1:8002/metrics
curl --fail --silent --show-error \
  --output "${OBS}/yield-metrics.json" \
  http://127.0.0.1:8002/yield/metrics
curl --fail --silent --show-error \
  --output "${OBS}/horde-worker.json" \
  'https://aihorde.net/api/v2/workers/name/memra-research-preview-runpod'

for unit in \
  memra-server.service \
  memra-fleet-meter.timer \
  memra-horde-worker.service \
  memra-poe-bot.service \
  cloudflared.service
do
  sudo systemctl show "$unit" \
    --property=Id \
    --property=LoadState \
    --property=ActiveState \
    --property=SubState \
    --property=UnitFileState \
    --property=MainPID \
    --property=ExecMainCode \
    --property=ExecMainStatus \
    --property=NRestarts \
    --property=ActiveEnterTimestamp \
    >"${OBS}/${unit}.state"
done

for unit in \
  memra-server.service \
  memra-fleet-meter.service \
  memra-horde-worker.service \
  memra-poe-bot.service \
  cloudflared.service
do
  sudo journalctl -u "$unit" --since '4 hours ago' --no-pager \
    >"${OBS}/${unit}.journal"
done

nvidia-smi >"${OBS}/nvidia-smi.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader >"${OBS}/nvidia-compute-apps.csv"
df -h /scratch /data >"${OBS}/disk.txt"

jq -e '.status == "ready"' "${OBS}/readyz.json" >/dev/null
jq -e '.lanes.interactive and .lanes.judge and .lanes.harvest' \
  "${OBS}/yield-metrics.json" >/dev/null
jq -e '{
  id, name, online, maintenance_mode, requests_fulfilled,
  kudos_rewards, tokens_generated, models, uptime
}' "${OBS}/horde-worker.json" >"${OBS}/horde-summary.json"
sha256sum "${OBS}"/* >"${OBS}/SHA256SUMS"
```

Review these signals after every snapshot:

### Cache economics

```bash
python3 tools/cache_economics.py "${OBS}/metrics.json" \
  --cache-billing-factor 0.25 \
  >"${RECEIPT_ROOT}/derived/economics-${STAMP}.json" \
  2>"${RECEIPT_ROOT}/derived/economics-${STAMP}.txt"
python3 tools/fleet-report.py /var/lib/memra/receipts/fleet.jsonl \
  >"${RECEIPT_ROOT}/derived/fleet-${STAMP}.txt"
```

Check aggregate and per-tenant prompt, cached, and computed token deltas;
`cache_hit_token_ratio`; prefix-cache hit/miss/eviction growth; and the
`[64,512)` LCP share. If `cache_economics.py` reports no prompt tokens, record
`no traffic, no receipt`; do not turn zero counters into a multiplier.

### Admission and 429s

Compare consecutive `yield-metrics.json` snapshots:

- `lanes.*.admitted`, `shed`, `completed`, and `tokens_out` must be monotonic
  within a server process;
- judge/harvest `shed` deltas are dark-lane admission 429s;
- interactive queue pressure does not flip readiness and does not increment a
  shed counter;
- per-key concurrency 429s happen before worker admission, so they are not
  present in lane `admitted` or `shed`.

Because memra and the Poe shim intentionally have no HTTP access log, retain
status, response headers, and the OpenAI error body at the caller or edge for
direct/Poe per-key 429 counts. A 429 without a corresponding lane-shed delta is
expected for a tenant-cap rejection, not evidence that the admission meter is
wrong.

### Horde and Poe

- `online` must stay true and `maintenance_mode` false during the trial.
- `requests_fulfilled`, `tokens_generated`, and `kudos_rewards` should be
  non-decreasing. Report deltas between snapshots, not only totals.
- The model list must contain `memra-research-preview`.
- Watch the Horde journal for polling/backend failures and the Poe journal for
  protocol/backend failures.
- Re-run `sudo deploy/glue/trial-up.sh --check` after any restart or routing
  change.

### Error taxonomy and GPU state

Classify captured HTTP errors by status and OpenAI `error.code`:

| Status | Code | Meaning |
|---:|---|---|
| 400 | `context_length_exceeded`, `model_not_found`, or `invalid_lane` | client/configuration error |
| 401 / 403 | authentication error | missing, unknown, disabled, or lane-ineligible key |
| 429 | `rate_limit_exceeded` | tenant cap or dark-lane shed |
| 503 | `overloaded` | VRAM pressure, worker restart, or unavailable worker |
| 503 | `draining` | shutdown in progress |
| 500 | `engine_error` | step, prefill, graph, or constraint fault |

Search the raw memra journal for `[meter]`, `[abort]`, worker restart, GPU watch,
and admission messages. Call a failure `OOM` only when the raw log contains
`out of memory` or `CUDA_ERROR_OUT_OF_MEMORY`, and retain the concurrent
`nvidia-smi` compute-app state. If stderr was not captured, record `died, cause
unknown - repro needed`.

Every median or rate reported from the trial must state its sample count and
thermal regime. Do not combine restarted counter segments without using the
fleet report's restart handling.

## 5. Shutdown and receipt harvest

First set the Poe bot private in the Poe creator UI. This cannot be automated
from the pod.

Take one final observation snapshot using section 4, then stop the trial while
retaining the raw shutdown log:

```bash
cd /opt/memra
: "${RECEIPT_ROOT:?set RECEIPT_ROOT from the launch run id}"
set -o pipefail
sudo deploy/glue/trial-down.sh \
  2>&1 | tee "${RECEIPT_ROOT}/raw/trial-down.log"
```

`trial-down.sh` puts Horde into maintenance, disables Horde and Poe, disables
the fleet timer, takes one final fleet snapshot, gracefully drains and disables
memra, and disables cloudflared. Disabled units cannot reopen traffic on reboot.

Harvest raw receipts after the final meter snapshot:

```bash
install -d -m 0750 "${RECEIPT_ROOT}/raw/final"
TRIAL_START_UTC="${TRIAL_START_UTC:-$(
  sed -n 's/^trial_start_utc=//p' "${RECEIPT_ROOT}/raw/context.env"
)}"
test -n "$TRIAL_START_UTC"

sudo cp --preserve=mode,timestamps \
  /var/lib/memra/receipts/fleet.jsonl \
  /var/lib/memra/receipts/deployment.env \
  /var/lib/memra/receipts/step-model-manifest.txt \
  /var/lib/memra/receipts/step-model-manifest.sha256 \
  "${RECEIPT_ROOT}/raw/final/"
sudo chown -R "$(id -u):$(id -g)" "${RECEIPT_ROOT}/raw/final"

sudo sha256sum \
  /etc/memra/runpod.env \
  /etc/memra/keys.toml \
  /etc/memra/horde-worker.env \
  /etc/memra/poe-bot.env \
  /etc/memra/cloudflared-token \
  >"${RECEIPT_ROOT}/raw/final/secret-config.sha256"

for unit in \
  memra-server.service \
  memra-fleet-meter.timer \
  memra-horde-worker.service \
  memra-poe-bot.service \
  cloudflared.service
do
  sudo systemctl show "$unit" \
    --property=Id \
    --property=ActiveState \
    --property=SubState \
    --property=UnitFileState \
    --property=ExecMainCode \
    --property=ExecMainStatus \
    --property=NRestarts \
    >"${RECEIPT_ROOT}/raw/final/${unit}.state"
done

for unit in \
  memra-server.service \
  memra-fleet-meter.service \
  memra-horde-worker.service \
  memra-poe-bot.service \
  cloudflared.service
do
  sudo journalctl -u "$unit" --since "$TRIAL_START_UTC" --no-pager \
    >"${RECEIPT_ROOT}/raw/final/${unit}.journal"
done

nvidia-smi >"${RECEIPT_ROOT}/raw/final/nvidia-smi.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader \
  >"${RECEIPT_ROOT}/raw/final/nvidia-compute-apps.csv"
curl --fail --silent --show-error \
  --output "${RECEIPT_ROOT}/raw/final/horde-worker-after-down.json" \
  'https://aihorde.net/api/v2/workers/name/memra-research-preview-runpod'
sha256sum "${RECEIPT_ROOT}/raw/final/"* \
  >"${RECEIPT_ROOT}/raw/final/SHA256SUMS"
```

Do not copy the secret-bearing environment files, keyring, connector token, or
the cloudflared unit into the receipt bundle. The hashes bind their exact
configuration without disclosing credentials.

Only after raw harvest, generate final derived reports:

```bash
python3 tools/fleet-report.py \
  "${RECEIPT_ROOT}/raw/final/fleet.jsonl" \
  >"${RECEIPT_ROOT}/derived/fleet-final.txt"

LAST_METRICS="$(
  find "${RECEIPT_ROOT}/observations" -name metrics.json -print |
    sort | tail -n 1
)"
test -n "$LAST_METRICS"
python3 tools/cache_economics.py "$LAST_METRICS" \
  --cache-billing-factor 0.25 \
  >"${RECEIPT_ROOT}/derived/economics-final.json" \
  2>"${RECEIPT_ROOT}/derived/economics-final.txt"
```

Copy the complete `/data/memra-trial/${RUN_ID}` directory off the pod before
termination. Keep raw evidence and derived summaries together, but report spill
performance separately from model quality and trial traffic behavior.
