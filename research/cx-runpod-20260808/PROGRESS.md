# lane/cx-runpod-runbook - RunPod Step API deployment

Date: 2026-08-08
Task: owner gate sequence #53
Starting train tip: `9aebdb3e0b456868d7707d60d187ae4317ef4ae8`

## Mission

Make a fresh, systemd-capable RunPod 2x RTX PRO 6000 pod one script away from serving the
owner-approved Step-3.7-Flash train through an OpenAI-compatible public API.

This lane owns engine deployment tooling only. Product policy remains in `darklanes`.

## Deliverables

- `deploy/runpod/provision.sh`
  - source build or release-binary install from an explicit owner-approved ref;
  - RunPod CUDA compatibility-path preflight;
  - pinned HF, rsync, or pre-staged Step artifact flow with size and SHA-256 checks;
  - PP-2 systemd launch with the external MTP drafter, 128K context, keyring auth,
    current train defaults for spec/batching, live metrics, and fleet-meter timer;
  - Cloudflare Tunnel or RunPod HTTP proxy exposure;
  - local readiness, metrics, authentication, usage, and public-model checks.
- `deploy/runpod/API-USAGE.md`
  - endpoint shapes, model staging, key lifecycle, OpenAI SDK and curl examples,
    usage accounting, public smoke, and metrics interpretation.
- `deploy/runpod/smoke.sh`
  - machine-side N-request streaming smoke with HTTP 200, generated-delta, `[DONE]`,
    usage-shape, cached-token, and TTFT assertions.

## Frozen deployment facts

- Artifact source: `stepfun-ai/Step-3.7-Flash-GGUF` at revision
  `0b69336d2fd2adfdef9c66e425f7778196c31482`.
- Trunk: the three `IQ4_XS/Step-3.7-flash-IQ4_XS-*.gguf` shards.
- Drafter: `Step3.7-flash-mtp-Q8_0.gguf`.
- Total staged bytes: `108700839040`; all four SHA-256 values come from
  `research/step37-p2-20260806/raw/artifact-sha256-20260806.txt` and the pinned HF file
  receipt.
- RunPod receipts require `/usr/local/cuda-13.1/compat` ahead of the CUDA libraries;
  omitting it captured `CUBLAS_STATUS_NOT_INITIALIZED`. The script detects the installed
  CUDA 13 root and requires a compatibility `libcuda.so.1`.
- Pair placement is `MEMRA_PP_STAGES=2`, `MEMRA_PP_DEVICES=0,1`, with the drafter attached
  through `MEMRA_MODELS=alias=trunk+draft`.
- The script does not set `MEMRA_SERVE_SPEC`, `MEMRA_SPEC_K`, or `MEMRA_SERVE_BATCH`.
  Those are owned by the final approved train after the Step performance dance.
- Keyring auth is explicit. Admission remains the current default-on VRAM-aware path.
- `/metrics` stays a loopback fleet-meter source even when a public origin is configured.

## Gate posture

The starting tip is not automatically deployable. The pair-box v0.72 preparation receipt at
`research/v072-prep-20260808/PROGRESS.md` records two owner-call blockers: an inert tick
canary and the serving-layer spec+PP-2 regression from roughly 112.5 to 17.5 aggregate
tokens/s. `provision.sh` therefore requires `MEMRA_REF` or an explicit release tag for every
live run. Dry-run may use placeholders, but a live deployment cannot silently select this
worktree tip.

The older PP-2 example in `docs/SERVING.md` still pins `MEMRA_SERVE_SPEC=0`; the current
`docs/FLAGS.md` says the crash quarantine is lifted. This lane does not choose between the
remaining performance policies. It leaves the policy flags unset so the approved train owns
the answer.

## Steering

The requested steering file `~/.lanectl/inbox/cx-runpod.md` was absent when the lane began.
The only inbox file present was `cx-fleet.md`, which belongs to a different lane. The user
prompt and repository contracts were sufficient to implement this task; no fleet-lane
steering was imported.

## Verification

Local checks completed on 2026-08-08:

- PASS: `bash -n deploy/runpod/provision.sh`
- PASS: `bash -n deploy/runpod/smoke.sh`
- PASS: ShellCheck on both scripts
- PASS: source/HF/Cloudflare provision dry-run with no live inputs
- PASS: release/rsync/RunPod-proxy dry-run, including derived proxy URL
- PASS: smoke dry-run with a supplied secret; output contained only `<redacted>`
- PASS: the four embedded file hashes match
  `research/step37-p2-20260806/raw/artifact-sha256-20260806.txt`
- PASS: embedded model sizes sum to `108700839040`
- PASS: two-row `nvidia-smi` parser fixture preserves both GPU rows
- PASS: `systemd-analyze verify` for the three source units with only their not-yet-installed
  executable and working-directory paths substituted in a temporary copy
- PASS: three streaming requests through a local mock endpoint; every request asserted HTTP
  200, reasoning-first TTFT, `[DONE]`, final usage, and `cached_tokens`
- PASS: negative mock where `cached_tokens` exceeded `prompt_tokens`; the smoke script failed
  on the intended assertion
- PASS: curl header-from-stdin check used by the local authorized smoke
- PASS: forbidden-command/default-policy scans and `git diff --check`

No live provisioning command, GPU workload, model download, public tunnel, origin push, or tag
was run from this lane.

## Live-pod work still required

No RunPod pod existed during this lane. These facts cannot be claimed until the owner runs the
script on the actual two-card pod:

- the template really boots systemd and exposes two idle 96 GB sm_120 GPUs;
- the selected filesystem is local NVMe with sufficient free space;
- the approved source builds, or the approved release asset exists;
- all 108.7 GB of model bytes stage and verify on that pod;
- PP-2 peer access, full model load, `/readyz`, and the authorized inference smoke pass;
- the Cloudflare hostname route or RunPod proxy reaches the pod;
- the smoke script passes from a separate machine;
- any throughput, TTFT distribution, thermal regime, or 510 W impact measurement.

No performance result is produced by this paper-runnable deployment lane.
