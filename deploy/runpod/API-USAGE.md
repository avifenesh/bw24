# RunPod Step API usage

This runbook is the engine-side handoff for a provisioned two-card RunPod pod serving
Step-3.7-Flash through memra's OpenAI-compatible API.

The deploy gate is an owner-approved memra commit or release after the Step performance
lanes finish. Do not substitute the current branch tip merely because it builds.

## Provision the pod

Use a two-card RTX PRO 6000 Blackwell pod whose template boots systemd as PID 1 and includes
the CUDA 13 runtime. A source build also requires the CUDA development toolkit and an existing
Cargo installation. From this worktree, put the provisioner on the fresh pod:

```bash
rsync -a deploy/runpod/provision.sh root@<pod-host>:/root/provision-memra-runpod.sh
ssh root@<pod-host>
chmod 0755 /root/provision-memra-runpod.sh
```

The script clones the approved memra revision itself. First inspect the mutation-free plan:

```bash
/root/provision-memra-runpod.sh --dry-run
```

For a source build with Hugging Face staging:

```bash
env \
  MEMRA_REF=<owner-approved-commit> \
  MEMRA_MODEL_SOURCE=hf \
  MEMRA_EXPOSURE=cloudflare \
  MEMRA_PUBLIC_URL=https://api.example.com \
  CLOUDFLARED_TOKEN=<remotely-managed-tunnel-token> \
  /root/provision-memra-runpod.sh
```

The Hugging Face path downloads exactly the three IQ4_XS trunk shards and the Q8_0 MTP
drafter from `stepfun-ai/Step-3.7-Flash-GGUF` revision
`0b69336d2fd2adfdef9c66e425f7778196c31482`, then verifies the four pinned SHA-256 values.
The default destination is `/scratch/models/step-3.7-flash`. Confirm that `/scratch` is the
pod's local NVMe mount before the live run, or set `MEMRA_MODEL_DIR` to the actual local-NVMe
path.

To pull a prepared directory with rsync:

```bash
env \
  MEMRA_REF=<owner-approved-commit> \
  MEMRA_MODEL_SOURCE=rsync \
  MEMRA_RSYNC_SOURCE=root@artifact-host:/data/step-3.7-flash \
  MEMRA_EXPOSURE=runpod-proxy \
  RUNPOD_POD_ID=<pod-id> \
  /root/provision-memra-runpod.sh
```

The rsync source root must contain `IQ4_XS/` and
`Step3.7-flash-mtp-Q8_0.gguf`. An operator-side push is also valid:

```bash
rsync -a --partial --append-verify \
  /data/step-3.7-flash/ \
  root@<pod-host>:/scratch/models/step-3.7-flash/
```

Then run the provisioner with `MEMRA_MODEL_SOURCE=existing`.

For an approved prebuilt release, set `MEMRA_INSTALL_MODE=release` and
`MEMRA_VERSION=<approved-tag>`. The script still checks out that tag so the systemd units and
fleet-meter tooling match the binary. It never installs Rust or invokes `rustup`.

## Launch contract

The provisioner installs `deploy/systemd/memra-server.service` plus a RunPod drop-in. Its
effective launch environment is:

```bash
CUDA_VISIBLE_DEVICES=0,1
LD_LIBRARY_PATH=<cuda-compat>:<cuda-lib64>
MEMRA_ADDR=127.0.0.1:8002
MEMRA_COMPAT=openai
MEMRA_MODELS=stepfun/step-3.7-flash=/scratch/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf+/scratch/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
MEMRA_API_KEYS=/etc/memra/keys.toml
MEMRA_PP_STAGES=2
MEMRA_PP_DEVICES=0,1
MEMRA_CTX=131072
```

`MEMRA_SERVE_SPEC`, `MEMRA_SPEC_K`, and `MEMRA_SERVE_BATCH` are deliberately absent. The
owner-approved train's current defaults govern those policies. In this worktree the documented
spec default is on with K=3, but the completed Step dance, not this runbook, owns the deployment
default. VRAM-aware admission remains on by default, and the keyring makes completion routes
authenticated.

The RunPod receipts require the CUDA compatibility directory to be first in
`LD_LIBRARY_PATH`; omitting it produced `CUBLAS_STATUS_NOT_INITIALIZED`. The provisioner refuses
to continue unless it finds a compatibility `libcuda.so.1`.

The script also installs and arms `memra-fleet-meter.timer`. It scrapes loopback `/metrics`
every 30 minutes and appends receipts to `/var/lib/memra/receipts/fleet.jsonl`.

On a rerun with an existing keyring, set `MEMRA_SMOKE_API_KEY` to one live plaintext key if
the provisioner should repeat its authorized local inference check.

## Endpoint URL

Preferred, with a remotely-managed Cloudflare Tunnel:

```text
https://api.example.com/v1
```

Create the tunnel and its public-hostname route in Cloudflare before provisioning. The route
must target `http://127.0.0.1:8002`. The token lets the script install the connector as a
systemd service; it does not create the dashboard-side hostname route.

Fallback, with RunPod's HTTP proxy:

```text
https://<pod-id>-8002.proxy.runpod.net/v1
```

Expose HTTP port `8002` in the pod template before launch. RunPod's proxy has a 100-second
time-to-first-byte limit, so use streaming for real requests. Memra emits stream data and SSE
keepalives; a long non-streaming request can still exceed the proxy limit.

Cloudflare Tunnel keeps memra loopback-bound. The RunPod proxy fallback requires
`0.0.0.0:8002`. `/metrics` is intentionally unauthenticated, so scrape it over loopback or SSH
and add an edge path restriction before treating either public origin as durable.

Set the machine-side client environment:

```bash
export MEMRA_BASE_URL=https://api.example.com/v1
export MEMRA_API_KEY=mk-owner-...
```

## Issue and revoke keys

The provisioner creates an initial `owner` key only when `/etc/memra/keys.toml` does not
exist. It prints the plaintext once; the keyring stores only SHA-256.

On the pod:

```bash
memra-server \
  --gen-key alice \
  --rate-limit 4 \
  --keys /etc/memra/keys.toml

memra-server \
  --revoke-key mk-alice-1a2b3c4d5e6f \
  --keys /etc/memra/keys.toml
```

The running server hot-reloads key changes within two seconds. Missing and unknown keys return
HTTP 401; revoked keys return HTTP 403.

## OpenAI Python SDK

Install the current `openai` package on the client machine, then:

```python
import os

from openai import OpenAI


client = OpenAI(
    api_key=os.environ["MEMRA_API_KEY"],
    base_url=os.environ["MEMRA_BASE_URL"],
)

response = client.chat.completions.create(
    model="stepfun/step-3.7-flash",
    messages=[{"role": "user", "content": "Give me one sentence about NVLink."}],
    max_tokens=64,
)

details = response.usage.prompt_tokens_details
cached = details.cached_tokens if details is not None else 0
print(response.choices[0].message.content)
print(
    response.usage.prompt_tokens,
    response.usage.completion_tokens,
    response.usage.total_tokens,
    cached,
)
```

Streaming keeps first-byte latency inside public-proxy limits:

```python
import os

from openai import OpenAI


client = OpenAI(
    api_key=os.environ["MEMRA_API_KEY"],
    base_url=os.environ["MEMRA_BASE_URL"],
)

final_usage = None
stream = client.chat.completions.create(
    model="stepfun/step-3.7-flash",
    messages=[{"role": "user", "content": "Explain pipeline parallelism briefly."}],
    max_tokens=128,
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if chunk.choices:
        delta = chunk.choices[0].delta
        text = delta.content or getattr(delta, "reasoning", None)
        if text:
            print(text, end="", flush=True)
    if chunk.usage is not None:
        final_usage = chunk.usage

print()
print(final_usage)
```

## curl

Non-streaming:

```bash
curl --fail --silent --show-error \
  -H "Authorization: Bearer $MEMRA_API_KEY" \
  -H "Content-Type: application/json" \
  "$MEMRA_BASE_URL/chat/completions" \
  -d '{
    "model": "stepfun/step-3.7-flash",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }' | jq
```

The response includes:

```json
{
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 12,
    "total_tokens": 30,
    "prompt_tokens_details": {
      "cached_tokens": 0
    }
  }
}
```

Streaming:

```bash
curl --no-buffer --fail --silent --show-error \
  -H "Authorization: Bearer $MEMRA_API_KEY" \
  -H "Content-Type: application/json" \
  "$MEMRA_BASE_URL/chat/completions" \
  -d '{
    "model": "stepfun/step-3.7-flash",
    "messages": [{"role": "user", "content": "Count from one to five."}],
    "max_tokens": 64,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

The final JSON event carries `usage`, followed by `data: [DONE]`.

## Public smoke

From a separate machine:

```bash
deploy/runpod/smoke.sh --dry-run

MEMRA_BASE_URL=https://api.example.com/v1 \
MEMRA_API_KEY=mk-owner-... \
deploy/runpod/smoke.sh --requests 5
```

Each request must return HTTP 200, emit a generated streaming delta, end with `[DONE]`, and
carry internally consistent usage fields including
`usage.prompt_tokens_details.cached_tokens`. The script prints request TTFT and a median. It
does not assert a latency target or a nonzero cache hit.

## Metrics after traffic

Read metrics on the pod:

```bash
curl --fail --silent --show-error http://127.0.0.1:8002/metrics |
  jq '{
    admitted,
    completed,
    tokens_out,
    step_p50_ms,
    step_p99_ms,
    prompt_tokens_in,
    cached_tokens_in,
    computed_tokens_in,
    cache_hit_token_ratio,
    tenants,
    spec
  }'
```

After successful calls:

- `admitted`, `completed`, `tokens_out`, and `prompt_tokens_in` increase.
- `cached_tokens_in` is never greater than `prompt_tokens_in`; it may remain zero.
- `computed_tokens_in` equals prompt tokens minus cached tokens.
- `tenants["t:<tenant>"]` appears after that tenant's first admitted request.
- `spec["stepfun/step-3.7-flash"]` appears after the first request that actually uses
  speculative decoding and reports rounds, drafted/accepted tokens, and per-position rates.
- Prefix-cache counters and the LCP histogram reflect only traffic eligible for that cache
  path; their remaining zero is not a failed deployment.

Operational checks:

```bash
systemctl status memra-server memra-fleet-meter.timer
systemctl status cloudflared  # Cloudflare Tunnel mode only
journalctl -u memra-server -f
tail -n 3 /var/lib/memra/receipts/fleet.jsonl
```

The 510 W cap observed on a prior RunPod community card is an absolute-throughput caveat.
Do not compare its raw token rate to the production board without a controlled interleaved
measurement. The API smoke is a correctness and reachability check, not a performance receipt.
