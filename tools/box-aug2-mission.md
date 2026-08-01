# box-aug2 — GPU mission map (owner-approved plan)

**Window:** capacity block `cr-02251913b8f9ea0a6`, p5.48xlarge (8×H100 SXM), us-east-2a
(use2-az1), **2026-08-02T11:30Z → 2026-08-03T11:30Z**. Hard end: AWS terminates the box at
end-time — no grace. Reservation verified 2026-08-01: `state=scheduled` (payment cleared).

**Hour zero is one command** (from the local rig, at/after 11:30Z Aug 2):

```
bash tools/provision-aug2.sh preflight   # read-only, safe anytime
bash tools/provision-aug2.sh launch      # run-instances -> IP -> Mumbai SG -> onbox provisioning
bash tools/provision-aug2.sh sync-repo <box-ip>   # pin the intended commit over the Mumbai tree
```

`launch` installs the idle-safety on-box: login motd with a live countdown + `wall`
broadcasts at T-2h / T-30 / T-15 / T-5 (09:30, 11:00, 11:15, 11:25Z on Aug 3).

**GPU discipline (unchanged from the Jul-31 box):**

- **GPU 0 = bench-only** — validation battery and board cells, never a dev lane. A naked
  run grabs GPU 0 and poisons the bench regime; every lane exports
  `CUDA_VISIBLE_DEVICES=<n>` explicitly. Check `nvidia-smi` before benching: GPU 0 idle.
- GPUs 1-4 = dev lanes (one lane per GPU). GPUs 5-7 = fleet/serving work.

**Receipts convention (/tmp-free):** every harness writes under `~/receipts/<lane>/` on the
box — `tee` the raw log first, parse second; never `/tmp`, never rsync-only trees. One
rsync-back collects everything (phase 3). The Jul-31 wind-down had to rescue sk ncu/nsys
binaries from `/tmp` on a terminating box; this convention exists so that never repeats.

---

## Phase 1 — first ~2h (GPUs 0-7): weights, gates, comms re-confirm

### 1a. Weights pulls → `/opt/dlami/nvme/models` (start immediately; network-bound, zero GPU cost)

Two pulls, in this order (mission priority per `research/product-vision-20260801/ASSESSMENT.md`):

1. **Hy3 serving artifact** — the staging manifest comes from **lane/hy3-hopper's receipts**
   (`research/hy3-hopper-20260801/`): the exact artifact form already first-lit on the Mumbai
   H100 rsyncs from Mumbai's NVMe copy (box-to-box, the SG rule opens at launch). Record the
   staged manifest hash (CLAUDE.md staging rule).
2. **GLM-5.2** — exact commands in `research/mla-inc2-20260801/ARTIFACT.md` (MERGED:
   unsloth/GLM-5.2-GGUF @ abc55e72, UD-Q4_K_XL, 11 parts, 435.2 GiB, per-file sha256
   manifest; DevQuasar fallback). Network-bound fill — it must never displace Hy3 staging
   bandwidth or any GPU work.

### 1b. M1-finish: multi-device pp2 gates (`MEMRA_PP_DEVICES=0,1`)

**lane/m1-inc2 is MERGED** — the authoritative on-box M1-finish list lives in
`research/m1-inc2-20260801/` and runs verbatim:

```
mkdir -p ~/receipts/m1-pp2
cargo build --release                                  # arch auto-detects 90a
./target/release/pp-transport-smoke 2>&1 | tee ~/receipts/m1-pp2/transport-smoke.log
B=./target/release/pp2-gate; R=~/receipts/m1-pp2; M=~/models/Qwen3.5-9B-Q8_0.gguf
$B $M                                    2>&1 | tee $R/q9-singledev.log
MEMRA_PP_DEVICES=0,1 $B $M               2>&1 | tee $R/q9-dev01.log       # THE cross-device gate
MEMRA_PP_DEVICES=0,1 $B $M 16 32 5       2>&1 | tee $R/q9-dev01-split5.log
MEMRA_PP_DEVICES=0,1 MEMRA_PP_OVERLAP=1 $B $M 2>&1 | tee $R/q9-dev01-overlap.log
MEMRA_PP_DEVICES=0,7 $B $M               2>&1 | tee $R/q9-dev07.log       # non-adjacent NVLink pair
```

Gate contract: BIT-IDENTICAL logits at every step — any differing bit is a seam bug, FAIL.
Plus one receipt: per-tick boundary cost interleaved ×5 vs M0's 0.3-0.5% prediction
(weights are peer-read in this placement — correctness-mode, not the perf configuration).
M1 is DECLARED DONE when this list is green.

### 1c. M0 a2a re-confirm (new box, new NVSwitch fabric)

Re-run the all-to-all set from `research/m0-nccl-20260801/src/` (committed on lane/m0-nccl,
merged) — n=2/4/8 to match the existing `nccl_a2a_n{2,4,8}` receipts. On this box all
8 GPUs are ours; still verify idle before widening the set (the scripts do this).

```
mkdir -p ~/m0-nccl/receipts ~/receipts/m0-a2a && cd ~/m0-nccl
cp ~/memra/research/m0-nccl-20260801/src/{commbench.cu,commbench2.cu,runall.sh,runext.sh} .
nvcc -O3 commbench.cu  -o commbench  -I/opt/pytorch/cuda/include -L/opt/pytorch/cuda/lib -lnccl
nvcc -O3 commbench2.cu -o commbench2 -I/opt/pytorch/cuda/include -L/opt/pytorch/cuda/lib -lnccl
#   (NCCL location per the Jul-31 env receipt: /opt/pytorch/cuda/lib/libnccl.so.2 — re-verify)
./commbench a2a 0 1              > ~/receipts/m0-a2a/a2a_n2.jsonl 2> ~/receipts/m0-a2a/a2a_n2.err
./commbench a2a 0 1 2 3          > ~/receipts/m0-a2a/a2a_n4.jsonl 2> ~/receipts/m0-a2a/a2a_n4.err
./commbench a2a 0 1 2 3 4 5 6 7  > ~/receipts/m0-a2a/a2a_n8.jsonl 2> ~/receipts/m0-a2a/a2a_n8.err
nvidia-smi topo -m > ~/receipts/m0-a2a/topo.txt
```

Confirm target: the Jul-31 curve (19.7/30.8/55.8 µs at n=2/4/8; EP≤4 + graph-captured a2a
mandatory) reproduces on this fabric. A materially different curve re-opens the M0 verdict
before M2 leans on it.

### 1d. Validation battery (GPU 0)

```
mkdir -p ~/receipts/battery
export MEMRA_NVCC=$HOME/cuda-13.3.1/bin/nvcc     # provision adds this to ~/.bashrc
CUDA_VISIBLE_DEVICES=0 tools/validate-h100.sh ~/models/Qwen3.5-9B-Q8_0.gguf --quick \
  2>&1 | tee ~/receipts/battery/validate-q9-quick.log
# full battery on the models phase 2 touches:
CUDA_VISIBLE_DEVICES=0 tools/validate-h100.sh /opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf \
  2>&1 | tee ~/receipts/battery/validate-q27.log
```

Rebuild before the first battery (stale-binary lesson: rsync'd trees false-FAIL the
verify gate; validate-h100.sh touches the .cu sources for exactly this reason).

---

## Phase 2 — bulk of the window (assessment order: Hy3 spike is the centerpiece)

### 2a. Hy3 PP-2 serving spike — THE product deliverable of this window

Per `research/product-vision-20260801/ASSESSMENT.md`: Hy3 is the first darklanes SKU
(top-5 OpenRouter demand on five endpoints, zero new kernel classes, first-light GREEN on
the Mumbai H100 2026-08-01 — coherent generation, see `research/hy3-hopper-20260801/`).
Sequence:
1. Single-H100 replica gates on this box (argmax + short serve) — the Mumbai first-light
   reproduced on the box artifact copy.
2. **PP-2 Hy3**: `MEMRA_PP_DEVICES=<a>,<b>` on the Hy3 artifact — the M1 cross-device seam
   carrying the real SKU. Bit-gate first, then interleaved ×5 throughput.
3. 2×H100-per-replica fleet point: 4 replicas (GPUs 0-7 paired), serve-fleet + load-serve,
   in-window denominators.
4. **Spec K=1 wires in** per `research/hy3-accept-profile-20260802/` (all realistic
   classes clear S_est 1.22-1.38x; priority code-gen > code-review = agentic > chat).
   The spike measures φ (resident 2-position verify cost) in S=(1+r)/(1+φ) — r is
   already priced per class. K stays 1 (the nextn=1 head never chains). NEVER
   calibrate on the synthetic d1736 prompt (8.5% — understates real classes 5-9x);
   the $/Mtok floor row stays plain 2.49 tok/s (`research/hy3-spec-20260802/`).
5. Exit artifact: exactness receipts + a tok/s and $/Mtok table vs the 5 incumbent Hy3
   endpoints (pricing pinned in `research/model-demand-20260801/`).
Receipts → `~/receipts/hy3-spike/`.

### 2b. M2 PP8 bring-up (GPUs as free after 2a)

The M0 verdict (PP ~free at 0.3-0.5% tick, peerAsync > NCCL 2.8x at PP-activation sizes)
plus the merged M1 seam extend to N stages with microbatching. M2 needs (from
`research/m1-inc2-20260801/`): N-stage stage map, weight sharding (peer-read is
bring-up-only), deferred-readback pipelining, microbatch boundary slots + graph capture.
Gates before perf: the pp2-gate bit-identity contract generalizes — an 8-stage run must
match the unsplit reference exactly before any throughput number is quoted.
Receipts → `~/receipts/m2-pp8/`.

### 2d. MLA increment 3 on real weights (GLM-5.2 from 1a) — LOW-PRIORITY FILL

Order is fixed: **tensor audit → CPU-ref spot check → loader at scale.**

1. Tensor audit: every expected `blk.N.attn_{q_a,q_a_norm,q_b,kv_a_mqa,kv_a_norm,k_b,v_b,
   output}` (+ norms) present with the shapes pinned in
   `research/mla-bringup-20260801/DESIGN.md`; dump the inventory →
   `~/receipts/mla-inc3/tensor-audit.log`.
2. CPU-ref spot check: the increment-1 CPU f32 reference (absorbed form) against a handful
   of layers of the real checkpoint before any GPU path touches it.
3. Loader at scale: full-model load on one H100, memory accounting receipts.

### 2c. Fleet cap re-sweep (GPUs 5-7) — the flagged stale-verdict lever

Cap 8/replica was calibrated on the v0.59 core; the v0.61 binary moved the tick
(batched-tick inc2: 305→655 solo). Reference row: **1477.0 tok/s managed** (v0.60.0,
c=96×288, cap 8 — `research/fleet-v060-20260801/SUMMARY.md`). Per the clock-drift law,
that cross-day number is context, not the denominator: **the in-window cap-8 pass is the
denominator**, and cap arms interleave pass-wise.

```
mkdir -p ~/receipts/fleet-cap-resweep
for pass in 1 2; do
  for CAP in 8 12 16; do
    GPUS="5 6 7" REPLICAS_PER_GPU=2 CAP=$CAP MODEL=$HOME/models/Qwen3.5-9B-Q8_0.gguf \
      FLEET_RUN=~/receipts/fleet-cap-resweep/fleet-cap$CAP tools/serve-fleet.sh start
    sleep 45   # health settle
    python3 tools/load-serve.py --base http://127.0.0.1:8080 --concurrency 6 --requests 6 \
      --greedy --model qwen --out ~/receipts/fleet-cap-resweep/greedy.jsonl --label cap$CAP-greedy-p$pass
    python3 tools/load-serve.py --base http://127.0.0.1:8080 --concurrency 48 --requests 192 \
      --model qwen --out ~/receipts/fleet-cap-resweep/points.jsonl --label cap$CAP-c48-p$pass
    python3 tools/load-serve.py --base http://127.0.0.1:8080 --concurrency 96 --requests 288 \
      --model qwen --out ~/receipts/fleet-cap-resweep/points.jsonl --label cap$CAP-c96-p$pass
    GPUS="5 6 7" FLEET_RUN=~/receipts/fleet-cap-resweep/fleet-cap$CAP tools/serve-fleet.sh stop
  done
done
```

Verdict shape: cap X wins only if it beats the same-window cap-8 arm in both passes at
c=96 with zero 5xx and the greedy hash unchanged across caps. Publish medians with N and
thermal regime.

---

## Phase 3 — last 2h (the 09:30Z wall announces it)

1. **Receipts home:** from the local rig —
   `rsync -az -e "ssh -i ~/.ssh/darklanes-bench.pem" ubuntu@<box-ip>:~/receipts/
   research/box-aug2-20260802/receipts/` — then commit raw logs next to their summaries
   (raw sweep output is part of the deliverable; summary-only is not evidence).
2. **Board cells:** if any published number moved, update
   `research/tune-data/current-board.json`, run `python3 tools/update-perf-board.py`, and
   commit JSON + README + SVG together with the number-moving change. Never hand-edit the
   marker blocks; never `--no-verify`.
3. **Box-sweep** (the Jul-31 wind-down pattern from
   `research/f16g-permodel-20260801/ab-8x/SUMMARY.md`): nothing unreceipted may exist in
   `/tmp` or rsync-only trees —
   - `find /tmp -newer /var/lib/cloud/instance -type f` → anything evidence-like
     (scripts, stdout, `.ncu-rep`/`.nsys-rep` binaries) rescued into `~/receipts/box-sweep/`;
   - `ls ~` stray dirs: every rsync-only tree cross-checked file-for-file (md5) against
     the committed copies; only deltas rescued;
   - fleet `FLEET_RUN` logs are already under `~/receipts` by convention — verify;
   - final `nvidia-smi` state captured to `~/receipts/box-sweep/final-gpu-state.txt`.
4. No shutdown action needed — the block hard-terminates at 11:30Z. Everything must be
   synced by the T-30 wall.

---

## Gaps — status as of 2026-08-01 late

1. ~~lane/mla-inc2 ARTIFACT.md~~ **CLOSED** — merged (`research/mla-inc2-20260801/ARTIFACT.md`).
2. ~~lane/m1-inc2 command list~~ **CLOSED** — merged; 1b above is the verbatim list.
3. **Hy3 staging manifest:** lane/hy3-hopper still running on Mumbai (first-light GREEN);
   its receipts supply 1a's staging source. If it hasn't landed by box start, rsync the
   exact Mumbai NVMe artifact directory it first-lit (the copy IS the manifest).
4. **sync-repo pin:** the v0.62 release commit on restructure/public-split (cut before box
   start), which merges every box-relevant lane. The pin is the receipt.
5. **Root volume baked at 300 GiB gp3** (AMI default 20 GiB too small). ~\$1.3/window;
   veto/resize in `provision-aug2.sh` if wrong.
6. **Mumbai SG hygiene:** `launch` auto-revokes the dead Jul-31 box rule
   (18.117.231.105/32) and adds the new box IP. Flag if that stale rule is wanted.
