# accept-gate — an acceptance-delta assertion INSIDE the battery

Lane: `lane/accept-gate` off `origin/restructure/public-split`. Rig: local RTX 5090 Laptop.
Follow-up to `research/f8f4-flip-20260806/` (merged `c506317e`), which named this gate as the
open item: *"an acceptance-delta assertion belongs inside the battery."*

## 1. The blind spot, precisely

The f8f4 lane ran a kernel-arm A/B (`MEMRA_MMQ_F8F4=1`, a merged opt-in prefill route) against
the two production serve configs. Result: **served greedy text differed in 4 of 6 regime cells
at temperature 0**, and spec acceptance moved by up to **−9.5pp**. Meanwhile *every gate in the
battery stayed green in both arms*. Three independent structural reasons:

1. **The token goldens are 20 tokens.** Both of that lane's greedy divergences landed at
   generated index **22 and 38** — past the pins. A 20-token golden cannot see a change that
   starts at token 22, no matter how carefully it is checked.
2. **`--refresh-goldens` would have absorbed it.** Refresh after such a change silently re-pins
   the new arm's tokens; the gate then *defends* the regression. The failure mode is not "nobody
   ran the gate", it is "the gate was updated by the same hand that made the change".
3. **Acceptance is invisible to exactness gates by construction.** `run-spec` self-consistency
   asserts *spec output == plain output within one arm*. Both arms pass that. `run-gen` argmax
   MATCHes in both. Nothing anywhere compared **how many draft tokens were accepted** — which is
   spec throughput, i.e. the product the user feels.

A fourth reason it stayed hidden this long: acceptance had been measured, but with the GGUF's
**embedded MTP head**, not the drafter production actually attaches.

## 2. The law the gate encodes

The f8f4 lane's law-shaped finding: **acceptance sign follows (model × drafter × prompt), not
the model.** It *inverted* between the bare MTP head and the production regime drafter on the
same two models the same day — q27 −1.9pp bare / +0.45pp regime; q9 +1.6pp bare / −3.05pp
regime, worst cell −9.5pp.

Consequence for gate design, and the single biggest choice here: **a cell is a served config,
not a model.** Every cell attaches the artifact's real production drafter via `MEMRA_MODELS
"+draft"` (which replaces the embedded head at load, `worker.rs load_draft`) at its real serve
K, and drives it **through the server**, not through `run-spec`. A bare-head acceptance number
is not evidence about a served config, so this gate does not collect one.

## 3. What is asserted

Per cell, at the serve config, temperature 0:

| Assertion | Form | Why this form |
|---|---|---|
| **A. acceptance counts** | `(rounds, drafted, accepted)` **exact integer match** | temp 0 ⇒ drafting is deterministic ⇒ these are hard integers. A tolerance band would just relabel the blind spot at a coarser grain. |
| **B. generated text** | full completion **sha256**, `ngen=128` | 6.4× the 20-token golden window; covers the receipted divergence indices (22, 38). |
| **C. completion length** | exact | the arm shifted token counts (128↔129↔131) independently of content. |
| **D. config fingerprint** | sha256 of (model, draft, K, ngen, prompt file + its content hash) | a reference is only meaningful for the config it was minted under; a prompt edit must report "config moved", not a fake divergence. |

`accepted/drafted` is derived and printed for humans; it is never the assertion.

**Why A and B are separate assertions, empirically:** cell `q27-p3` under the arm moved
acceptance **+2.92pp with a byte-identical text sha**. A text-only gate — at *any* window
length — is blind to that cell. Conversely `q9-p1` moved text at char 252. Neither assertion
subsumes the other.

## 4. Why a single-shot read is legitimate

Greedy acceptance is a property of (build × model × drafter × prompt × K), not of the box's
mood. This is measured, not assumed, at three levels:

- **The seed harness's control** (`research/f8f4-flip-20260806/tools/regime_accept_ab.sh`, the
  OFF/ON/OFF2 pattern this lane reuses rather than rebuilds): the two independent OFF passes
  were byte-identical in all six cells.
- **`--control` in this gate**: re-measures every cell in a **second, independent server boot**
  and requires byte-identity. Verified green 6/6 (`logs/teeth-naked.log`). If it ever disagrees,
  no single-pass verdict from this gate — pass *or* fail — may be believed, and the gate says so
  in those words.
- **Cross-context reproduction**: the references minted here reproduce the f8f4 lane's OFF arm
  in **all 6 cells**, identical counts *and* identical text sha256, across a different worktree,
  a different ctx (8192 vs 16384) and a different time of day.
- **Across 18 commits of engine change**: merging `restructure/public-split` in (which had
  advanced past this lane's base with the batched PP-2 stage split, `pp.rs`, `run_gen.rs`'s argmax
  calibration, `worker.rs`, and two nvcc-resolution build fixes) and rebuilding, all 6 cells still
  reproduce byte-identically and `--teeth` still inverts — `logs/postmerge-full-matrix.log`,
  `logs/postmerge-teeth.log`. A pinned reference that has not been shown to reproduce on the
  branch it merges into is not a reference.

That last point is worth stating plainly: acceptance and greedy text at temp 0 are **not** drift
quantities. Unlike tok/s, they carry no thermal or clock dependence, so a FAIL here is never
"the machine's state" — the gate's FAIL banner says exactly that, to pre-empt the
settle-it-with-an-A/B protocol that a tok/s red correctly triggers.

## 5. The silent-re-pin trap, closed

`--pin` is the only way to write references, and it **refuses** to run when:

- **`crates/` is dirty** (staged or unstaged). References may only be minted from committed
  engine code, so every reference is attributable to a reviewable SHA. **There is deliberately
  no `--force`** on this check — an escape hatch here is the entire bug. (Dirt outside `crates/`
  is fine and is only reported.)
- **an arm env is set** (`MEMRA_MMQ_F8F4`, `..._PLAIN`, `MEMRA_FAST`, `MEMRA_PRIME_F32CHUNK0`).
  Pinning an opt-in arm's acceptance as the *default* reference is the same trap in a hat.

Both refusals verified before any GPU time was spent (exit 2, nothing launched).

## 6. Teeth — proven in both directions

A gate only ever observed passing proves nothing.

| Direction | Command | Required | Result | Receipt |
|---|---|---|---|---|
| naked build | `--full --control` | PASS | **GATE-RC=0**, 6/6 cells PASS, 6/6 control boots byte-identical | `logs/teeth-naked.log` |
| the arm | `MEMRA_MMQ_F8F4=1 --full` | FAIL | **GATE-RC=1**, detects it | `logs/teeth-f8f4.log` |
| the arm, q27 clean window | `MEMRA_MMQ_F8F4=1 --cells q27-*` | FAIL | **GATE-RC=1** | `logs/teeth-f8f4-q27.log` |
| `--teeth` flag path | `--teeth` | exit 0 on detection | **TEETH-FLAG-RC=0** | `logs/teeth-flag.log` |

Detected values match the f8f4 lane cell-for-cell — this gate reproduces the original finding
from a cold start:

```
q27-p1  rounds 42->43  drafted 126->129  sha 8fcb13e8->33d69474  -1.57pp   text@char 208 (~word 30)
q9-p1   acceptance 0.7236->0.6288 = -9.48pp  sha 27c6f8ab->648971af       text@char 252 (~word 28)
q27-p3  acceptance 0.7778->0.8070 = +2.92pp  text sha IDENTICAL  <- counts-only detection
q27-p2  PASS under the arm                                       <- specific, not a blanket red
```

`q27-p2` passing under the arm matters as much as the failures: the lane also found p2 unchanged.
The gate is **specific** — it is not simply red whenever an env var is set.

Every text divergence landed at char 208–411 (~word 28–65), i.e. **past** the 20-token window,
reproducing the blind spot on demand.

## 7. Two false-green classes found by building this

Both were found by the gate's own bring-up, and both are the kind that report success:

1. **Foreign responder on the port.** The rig's idle `llama-server` held the default port, so
   `/health` answered *instantly* from a process that does not speak this API ("up in 0s"), and
   all six cells failed with HTTP 500. The dangerous version is the one where the squatter
   answers 200 with a plausible body — the gate would have **measured someone else's model and
   pinned it**. Now: an occupied port is a hard abort (never a wait — we cannot prove the
   responder is ours), plus a post-health check that the listener's pid *is* our child, closing
   the race where something grabs the port after pre-flight. Default port moved 8181 → 8317.
2. **The control arm compared every cell against the last cell's json** (`MJ` leaked from the
   measure loop). It produced 4 spurious "BOOTS DISAGREE" FAILs on a run whose six cells were in
   fact byte-identical. A control arm that cries wolf is worse than none: it trains readers to
   discount the one signal that says *stop, nothing here is trustworthy*. The failure reporter
   underneath it was also a `SyntaxError`, i.e. that path had never once executed — a reminder
   that error paths need their own exercise.

## 8. Wiring and cost

- **`tools/local-ci.sh`**: named arm after serve-stress, `MEMRA_CI_ACCEPT=0` skips. Default =
  smoke tier (**q27-p1 only**, ~1 min including the 16 GB load) to hold the correctness stage
  near its ~3 min budget. Full matrix behind `--full`.
- **`tools/fast-gate/models.tsv`**: `accept` as `kind=cmd` (self-gating; emits the
  `accept-gate: SKIP` verdict word so a missing artifact cannot read as a pass).
- **`tools/fast-gate/map.tsv`**: routed from the four path classes that can move acceptance —
  the NVFP4 prefill tiles (where the receipted mover lives; a prefill-KV numeric change
  propagates into the draft head's read set), `spec_sample.cu`, the spec/draft host pipeline, and
  `crates/memra-server/`. Verified: a touch on `mmq_nvfp4_w4a8.cu` plans `q9,k27,accept`.
- **Cells**: `tools/fast-gate/accept-cells.tsv`. **References**:
  `tools/fast-gate/accept-refs/<cell>.{ref,text}`, mirroring the `goldens/<id>.{tokens,perf}`
  convention.

Cells use ctx 8192 rather than the 128k production window deliberately: the assertions are
ctx-independent at these prompt lengths (longest ~5.4k tokens) and a 128k session costs ~4.1 GB
of KV plus a slow boot for zero added assertion strength. The 128k window has its own gates
(serve-smoke, serve-stress).

## 9. Adopting a route on a new artifact

The f8f4 lane's shipping rule stands and this gate is how you satisfy it: any adoption of an
opt-in numeric route on a new artifact needs **its own acceptance receipt with its real
drafter** — it may not inherit another artifact's verdict. Add the cell to
`accept-cells.tsv`, `--pin` it on the naked build from committed `crates/`, then A/B the route
against that reference.

## 10. Files

```
tools/accept-gate.sh                      the gate
tools/fast-gate/accept-cells.tsv          cell registry (6 cells: 2 models x 3 prompt lengths)
tools/fast-gate/accept-refs/*.ref|.text   pinned references (counts, cfg fingerprint, sha, text)
research/accept-gate-20260806/logs/       teeth receipts (both directions) + battery log
```

Log index:

| Log | What it receipts |
|---|---|
| `teeth-naked.log` | `--full --control` on the naked build: GATE-RC=0, 6/6 PASS, 6/6 control boots byte-identical |
| `teeth-f8f4.log` | `MEMRA_MMQ_F8F4=1 --full`: GATE-RC=1 |
| `teeth-f8f4-q27.log` | the same, q27 re-run in a clean window after a neighbor lane's OOM |
| `teeth-flag.log` | the `--teeth` inverted-verdict path itself: TEETH-FLAG-RC=0 |
| `battery-local-ci.log` | the full battery with the arm wired: BATTERY-RC=0 |
| `postmerge-full-matrix.log` | 6/6 PASS after merging 18 upstream commits and rebuilding |
| `postmerge-teeth.log` | `--teeth` still inverts on the merged build |
| `reverify-after-loop-refactor.log` | the boot/shutdown wait-loop rewrite re-run, not just re-parsed |
| `<cell>[.ctl].measure.log` | per-cell raw measurement stderr (measure + control boot) |

Nothing in this lane changes published numbers, so the perf board is deliberately untouched.
