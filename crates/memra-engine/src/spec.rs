//! Qwen3.5 MTP (NextN) greedy speculative decode (research/mtp/MTP-PLAN.md §A/§B/§C/§D).
//!
//! Greedy spec decode is MATHEMATICALLY EXACT: the accepted+bonus token stream is token-for-token
//! identical to plain greedy `generate`. This module provides:
//!   - `mtp_head_forward`  (§A, T=1): one NextN draft-token forward.
//!   - `decode_step_t`     (§D.3, T=K+1): batched target verify forward, all-column logits.
//!   - `generate_spec`     (§B): the draft/verify/accept/rollback orchestrator.
//! Cache snapshot/rollback lives in cache.rs (§D.4). The MTP head uses its OWN scratch KV (§D.6),
//! PERSISTENT over the committed sequence (see `MtpScratch`).

use crate::cache::{Cache, KvLayer};
use crate::forward::argmax;
use crate::hybrid::{FullAttnLayer, HybridModel, LinearAttnLayer, Mixer, MtpHead};
use crate::Engine;
use cudarc::driver::CudaSlice;

/// H-SEED CONVENTION (MEMRA_SPEC_HPOST=1): feed the MTP head the POST-norm hidden — trunk rows
/// hand over `output_norm(x)` and the draft chain recurrence hands over `shared_head_norm(h_nextn)`
/// (= final_h) — matching the reference engines: llama.cpp #24025 ("qwen35: use post-norm hidden
/// state for MTP", t_h_nextn is taken AFTER the final norm in both trunk and MTP graphs) and
/// SGLang's qwen3_5_mtp (spec_info.hidden_states = the target model's post-norm output). memra's
/// historical convention (default, MTP-PLAN §A) is PRE-norm x. Draft-quality-only: exactness is
/// the verify's job either way; acceptance arbitrates. OnceLock: read once, hot-loop safe.
pub(crate) fn spec_hpost() -> bool {
    static H: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *H.get_or_init(|| {
        std::env::var("MEMRA_SPEC_HPOST")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// LEAN VERIFY (default ON since 2026-07-08; MEMRA_SPEC_LEAN=0 reverts — close35 lane): the verify m-scaling
/// probe + nsys diff showed the verify t-path pays ~1.0ms/call at m=1 over eager decode on the
/// 35B, and the kernels are NOT the cause (dev-MoE identical, kernel-time delta only +179us).
/// The overhead is (a) ~250 extra cuMemsetD8Async/call from `e.zeros()` on buffers every kernel
/// fully overwrites (~0.9ms host issue + ~0.35ms GPU) and (b) the t=1 FA rows dispatch (rows_v2 +
/// combine_rows, +50us vs the eager fa_decode pair). This flag switches (a) fully-overwritten
/// verify buffers to `e.uninit` (identical bytes: every element is written before read) and
/// (b) t==1 verify FA to the eager `fa_decode` entry (byte-identical: kernel-check pins the
/// rows-vs-loop identity and the per-row loop at t=1 IS fa_decode on the same q). Gates arbitrate.
pub(crate) fn spec_lean() -> bool {
    static L: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // DEFAULT ON since 2026-07-08 (MEMRA_SPEC_LEAN=0 reverts): bit-identical (buffers fully
    // overwritten; gates green incl maxdiff-identical run-gen) and measured +2.4% e2e p3 /
    // +1.5% p2 at the daily 35B config. m=1 verify now costs eager-decode parity.
    *L.get_or_init(|| {
        std::env::var("MEMRA_SPEC_LEAN")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// SMALL-M BATCHED VERIFY (default ON since 2026-07-09; MEMRA_SPEC_M2=0 reverts — lane/spec-m2): extend the
/// batched linear-attn verify arm down to t=2 and batch the MoE dev token loop over a
/// grid.z=token axis at every verify t. The close35 m-scaling probe put the m=2 verify tier at
/// x1.54 of m=1 (llama x1.14); the per-column linear chain (t<3) and the serial MoE dev token
/// loop are the two launch-structure causes. Both changes are LAUNCH-STRUCTURE ONLY:
/// (a) the batched conv's t<pad ring update is pure copies (ssm_conv_ring_rebuild from a cloned
///     ring — the ring stores raw input columns); every arithmetic kernel is the same one the
///     t>=3 arm already runs (matmul_decode_exact bit-identical at m=2-4, gdn_scan's internal
///     t-loop == chained T=1 steps);
/// (b) the MoE dev-rows twins run the serial loop's per-token warp program with tok-offset
///     pointers (same sel/w/aq/ad bytes, same dot order, same slot-ordered FMA chain).
/// Gates arbitrate: run-spec K=1..8 self-consistency (35B+9B), kernel-check, run-gen argmax.
pub(crate) fn spec_m2() -> bool {
    static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // DEFAULT ON since 2026-07-09 (MEMRA_SPEC_M2=0 reverts): launch-structure only — t=2
    // batched linear arm (ring-roll copies, zero new FP order) + MoE dev-rows kernels
    // (grid.z=token, 4 launches/layer at any verify t). Acceptance bit-identical at every K;
    // 35B p2 +3.4% / p3 +3.6%; the profitable-K plateau widens (new optimum K=3 at 223).
    *M.get_or_init(|| {
        std::env::var("MEMRA_SPEC_M2")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}
pub(crate) fn spec_stream() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_STREAM").as_deref() == Ok("1"))
}
pub(crate) fn spec_stream_m() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("MEMRA_SPEC_STREAM_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
    })
}
pub(crate) fn spec_devacc() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_DEVACC").as_deref() == Ok("1"))
}

/// GRAMMAR HOOK for constrained spec decode (lane/constrained-full, 2026-08-03). The engine
/// stays llguidance-agnostic: the server adapts its per-session grammar state behind this
/// trait. CONTRACT (the verify-side truncation rule — token-identical to constrained plain
/// greedy decode): the exactness walk runs UNMASKED first; the hook then (a) truncates
/// acceptance at the first grammar-illegal accepted token, and (b) when the truncation fired
/// or the bonus is illegal, the engine recomputes that slot as the MASKED argmax of the
/// target's own verify column (an unmasked argmax that is grammar-legal IS the masked argmax
/// — masking only removes tokens — so the common case pays nothing). `consume` advances the
/// state with each EMITTED token in order; EOS handling is the implementor's job (skip).
pub trait SpecConstraint {
    /// -inf the current state's banned ids on a HOST logits row (prompt-tail / init-feed
    /// masked argmax).
    fn mask_logits(&mut self, logits: &mut [f32]) -> Result<(), String>;
    /// Packed 32-bit bitset words of the CURRENT state's allowed set (device-mask form).
    fn mask_words(&mut self) -> Result<Vec<u32>, String>;
    /// Is `tok` consumable in the CURRENT state?
    fn is_allowed(&mut self, tok: u32) -> Result<bool, String>;
    /// Advance the state with an emitted token.
    fn consume(&mut self, tok: u32) -> Result<(), String>;

    // --- DRAFT-SIDE MASKING (lane/draft-mask, 2026-08-04) ---
    // The drafter proposed grammar-illegal tokens under tight schemas, so verify-side
    // truncation cut nearly every round (measured acceptance 0.467-0.513 tight vs 0.62-0.82
    // loose, research/constrained-full-20260803). These three methods let the engine mask the
    // DRAFT model's own sampling with the grammar's legal set, so proposals are legal by
    // construction. The state they walk is a SPECULATIVE CLONE of the session matcher — the
    // real state is advanced only by `consume` (emitted tokens), so verify-side truncation
    // stays the correctness backstop and the emitted stream is unchanged by construction
    // (an accepted draft is the target's unmasked argmax AND grammar-legal, hence the masked
    // argmax; a cut slot is recomputed as the masked argmax either way).
    // Default impls = feature OFF (pre-lane behaviour: unmasked drafts).

    /// Is draft-side masking available on this hook? Probed ONCE per burst, before the draft
    /// graph is captured (the mask is an in-graph node — its presence is a capture-time shape).
    fn draft_mask_enabled(&self) -> bool {
        false
    }
    /// Start a draft chain: clone the CURRENT (committed) grammar state into the speculative
    /// slot. Called once per spec round, before the first draft position.
    fn draft_begin(&mut self) -> Result<(), String> {
        Ok(())
    }
    /// Packed 32-bit bitset words of the SPECULATIVE state's allowed set (target-vocab ids),
    /// for the draft position about to be sampled. `None` = draft masking off (no-op).
    fn draft_mask_words(&mut self) -> Result<Option<Vec<u32>>, String> {
        Ok(None)
    }
    /// Advance the SPECULATIVE state with a PROPOSED draft token. `false` = the chain cannot
    /// continue (EOS proposed, or an unmasked position proposed something illegal) — the
    /// engine stops drafting; the token already pushed still goes through verify.
    fn draft_advance(&mut self, _tok: u32) -> Result<bool, String> {
        Ok(false)
    }
}

/// DRAFT-MASK UPLOAD (lane/draft-mask): pull the speculative state's allowed set (TARGET-id
/// space) from the hook, project it into the DRAFT head's vocab space, and upload it into the
/// stable device buffer the draft chain reads. Returns false when the chain must stop drafting:
/// the hook handed out no mask, or NO draft-vocab row is grammar-legal at this position (a
/// trimmed FR-Spec head genuinely cannot propose a legal token there — masking it would leave
/// a fully-banned row whose argmax is meaningless, so the round drafts fewer tokens and the
/// verify emits the masked argmax as usual).
fn upload_draft_mask(
    e: &Engine,
    c: &mut dyn SpecConstraint,
    dst: &mut CudaSlice<u32>,
    d2t: Option<&Vec<u32>>,
    d_vocab: usize,
    words: usize,
) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(tw) = c.draft_mask_words().map_err(|e2| format!("constraint: {e2}"))? else {
        return Ok(false);
    };
    let bit = |t: usize| -> bool {
        let w = t >> 5;
        w < tw.len() && (tw[w] >> (t & 31)) & 1 == 1
    };
    let mut buf = vec![0u32; words];
    match d2t {
        // TRIMMED draft head: row i proposes target id d2t[i] — permute the mask accordingly.
        Some(map) => {
            for (i, &t) in map.iter().enumerate().take(d_vocab) {
                if bit(t as usize) {
                    buf[i >> 5] |= 1u32 << (i & 31);
                }
            }
        }
        // UNTRIMMED: draft ids ARE target ids; the packed words transfer verbatim (a short
        // mask leaves the padded tail zeroed == banned, same rule as constrained::apply_mask).
        None => {
            let n = tw.len().min(words);
            buf[..n].copy_from_slice(&tw[..n]);
        }
    }
    if buf.iter().all(|w| *w == 0) {
        return Ok(false);
    }
    e.htod_u32_into(dst, &buf)?;
    Ok(true)
}

/// Keep the full token-embedding table in host memory and upload only the rows needed by each
/// MTP/verify step. This is an exact memory-capacity seam for very large BF16 vocab tables: host
/// gather expands the same source bits to f32, and only O(T*n_embd) bytes cross PCIe per step.
/// CUDA-graph/round-stream draft paths require device token ids and therefore stay disabled.
pub(crate) fn spec_host_embd() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_HOST_EMBD").as_deref() == Ok("1"))
}

/// VERIFY-TIER TRUNK LAUNCH-FUSION (default ON since 2026-07-09; MEMRA_SPEC_FUSED_T=0 reverts — lane/close35b): extend
/// the t=1 fused2/fused3 Q8_0 trunk launches to the batched verify tier (t=2-4, the K=1..3
/// verify shapes). At t>1 the trunk pairs/triples (35B wqkv+wqkv_gate, wq/wk/wv,
/// gate_shexp+up_shexp) each run a separate `matmul_decode_exact` — one q8_1 re-quantize of the
/// SAME activation plus one _b2/_b4 launch per tensor. The fused twins share ONE quantize and
/// ONE launch per group; per (tensor,token,row) the kernel body is q8_0_mmvq_batched verbatim
/// with the identical row mapping -> BIT-IDENTICAL by construction (kernel-check pins it,
/// run-spec K=1..8 + acceptance identity arbitrate e2e).
pub(crate) fn spec_fused_t() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // DEFAULT ON since 2026-07-09 (MEMRA_SPEC_FUSED_T=0 reverts): verify t=2-4 trunk launch-fusion
    // (fused2/fused3 Q8_0 batched twins, bit-identical by construction — m=1 block-offset split on
    // the batched body). m=2 marginal token 2117->1762us; 35B daily: p3 +3.7% (crosses llama), p2 +5%.
    *F.get_or_init(|| {
        std::env::var("MEMRA_SPEC_FUSED_T")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// zeros/uninit switch for verify-path buffers that are FULLY OVERWRITTEN before any read.
/// Only call this on such buffers — the lean contract is "identical bytes by construction".
fn vbuf(e: &Engine, n: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    if spec_lean() {
        e.uninit(n)
    } else {
        e.zeros(n)
    }
}

/// Scratch KV for the MTP block (one full-attn layer).
///
/// PERSISTENT MODE (default, 2026-07-03 — the acceptance lever): sized cap = max_ctx and kept in
/// sync with the COMMITTED sequence — slot p holds the MTP block's K/V for committed token p
/// (roped p+1, the chain's rope convention), so the draft chain's self-attention sees the FULL
/// committed history instead of only the current round's 1..K+1 chain tokens (the reference
/// engine's "mtp_update" design). Entries come from two sources:
///   - chain appends: accepted positions KEEP their chain-computed entries (embedding exact,
///     hidden chain-approximate — the reference engine accepts the same);
///   - `mtp_kv_fill` batches: prompt positions + the last-draft position on full accept, computed
///     from EXACT trunk hiddens (K/V-only MTP-block pass, no attention/FFN/lm_head).
/// Rejected drafts / p-min extras / pseudo-seed appends are all discarded by the round-start
/// `set_len` truncation (the KvLayer len mechanism — §C rollback for the draft side).
/// Multi-turn spec-decode session (2026-07-05): trunk Cache + persistent MTP draft scratch +
/// the committed token list, alive across generate_spec_session calls. Turn N+1 primes ONLY its
/// suffix (chunked continuation prime over the quantized past) and mtp_kv_fill's its suffix rows,
/// then the round loop runs unchanged. `last_h` carries the pre-output_norm hidden of the last
/// committed row across turns (the predecessor-pairing seed + fill anchor).
/// Per-request sampling config for the sampled-spec serve path.
#[derive(Clone, Copy, Debug)]
pub struct SpecSampling {
    pub temp: f32,
    pub seed: u64,
    pub top_k: i32,            // 0 = off
    pub top_p: f32,            // 1.0 = off
    pub min_p: f32,            // 0.0 = off
    pub penalty_last_n: usize, // 0 = penalties off
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
}

/// Tracked draft positions for [`SpecTelemetry`] (serve K defaults to 3; the run-spec gate
/// sweeps K=1..8, and MEMRA_SPEC_CAPMAX defaults to 7 — 8 covers every tuned config).
pub const SPEC_TELEM_POS: usize = 8;

/// Always-on per-draft-position acceptance telemetry (lane/accept-telemetry, 2026-08-05 —
/// the llama.cpp #26389 / vLLM spec-decode counter schema, upstream-sweeps 2026-08-05).
/// Lives on the [`SpecSession`] and accumulates across bursts; the serve worker diffs a
/// stashed copy per burst for its per-model /metrics aggregation and per-request usage.
/// Same normalization as the `[spec-stats]` line: p-min-discarded chain tokens are counted
/// in NEITHER drafted nor accepted.
#[derive(Clone, Copy, Default, Debug)]
pub struct SpecTelemetry {
    /// verify rounds completed (a round-stream burst counts each of its M rounds).
    pub rounds: u64,
    /// tokens drafted / accepted across all rounds.
    pub drafted: u64,
    pub accepted: u64,
    /// how often draft position j (0-based within a round's chain) was offered / accepted.
    /// Positions >= SPEC_TELEM_POS are untracked (totals still count them). The opt-in
    /// round-stream arm (MEMRA_SPEC_STREAM=1) reads back only totals, so under it these
    /// arrays cover the standard-path rounds only and their sums may undercount the totals.
    pub pos_drafted: [u64; SPEC_TELEM_POS],
    pub pos_accepted: [u64; SPEC_TELEM_POS],
}

impl SpecTelemetry {
    /// Fieldwise `self - prev` — the worker's per-burst delta off a copy stashed before the
    /// burst call. Saturating: a caller diffing against the wrong snapshot gets zeros, not
    /// a wrapped counter.
    pub fn delta_since(&self, prev: &SpecTelemetry) -> SpecTelemetry {
        let mut d = SpecTelemetry {
            rounds: self.rounds.saturating_sub(prev.rounds),
            drafted: self.drafted.saturating_sub(prev.drafted),
            accepted: self.accepted.saturating_sub(prev.accepted),
            ..Default::default()
        };
        for j in 0..SPEC_TELEM_POS {
            d.pos_drafted[j] = self.pos_drafted[j].saturating_sub(prev.pos_drafted[j]);
            d.pos_accepted[j] = self.pos_accepted[j].saturating_sub(prev.pos_accepted[j]);
        }
        d
    }
    /// Fieldwise `self += d` — the worker's per-model aggregation.
    pub fn merge(&mut self, d: &SpecTelemetry) {
        self.rounds += d.rounds;
        self.drafted += d.drafted;
        self.accepted += d.accepted;
        for j in 0..SPEC_TELEM_POS {
            self.pos_drafted[j] += d.pos_drafted[j];
            self.pos_accepted[j] += d.pos_accepted[j];
        }
    }
}

pub struct SpecSession {
    pub(crate) cache: Cache,
    pub(crate) scratch: MtpScratch,
    /// Every token whose state the caches hold, in order (prompt turns + generated), INCLUDING
    /// overshoot: spec commits accepted drafts past max_new; those rows are in the caches, so the
    /// session must count them. Callers render output from this, not from their own echo.
    pub committed: Vec<u32>,
    /// Pre-output_norm hidden of the LAST committed row (device). None before the first turn.
    pub(crate) last_h: Option<CudaSlice<f32>>,
    /// Greedy argmax predicting the token AFTER committed.last() (from the last turn's final
    /// logits). Fuels empty-suffix continuation bursts (serve): the next turn emits this token
    /// first, feeds it, and the round loop resumes without any prime. None before the first turn.
    pub next_pred: Option<u32>,
    /// SAMPLED-SPEC stream continuity across bursts: Philox event counters persist here so a
    /// session's randomness never repeats between generate_spec_session calls. (0,0) at admit.
    pub sctr: u32,
    pub uctr: u32,
    /// PERSISTENT DRAFT-GRAPH CONTEXT (2026-08-01, the serve-burst fixed-cost fix): the captured
    /// draft graph(s) + every device I/O buffer they bake, carried ACROSS generate_spec_session
    /// calls. Before this, every serve burst re-captured the draft graph (2 warmup forwards +
    /// instantiate) — measured ~16ms/burst on H100 q27 (MEMRA_SPEC_BURST sweep,
    /// research/spec-serving-20260801). None before the first turn; error paths drop it
    /// (next burst recaptures — serve retires errored sessions anyway).
    pub(crate) draft_ctx: Option<DraftGraphCtx>,
    /// PENDING-CARRY across bursts (2026-08-01, the serve burst-boundary fix): the bonus token
    /// emitted by the last round but NOT committed to the caches. The old tail committed it with
    /// a solo T=1 trunk pass (+ draft fill), and the next burst's setup fed the stashed next_pred
    /// with ANOTHER solo pass — 2x ~11.5ms/burst measured on H100 q27 ([spec-setup] trace).
    /// Carrying it lets the next empty-suffix greedy burst consume it as round-0 verify col 0,
    /// exactly like a mid-burst full-accept boundary (no solo passes). INVARIANT: when set,
    /// `committed` (== cache rows) EXCLUDES this token although it was already emitted in the
    /// last burst's output, and `last_h` holds the hidden of the last COMMITTED row (its
    /// predecessor — the chain-seed/fill anchor). `next_pred` is None (unknown without the
    /// commit pass). Non-empty-suffix or sampled turns must flush first (spec_flush_pending);
    /// generate_spec_session_sampled does this at entry, and serve parks only flushed sessions.
    pub pending_tok: Option<u32>,
    /// SESSION-AFFINITY TURN CHECKPOINT (lane/session-affinity, 2026-08-05): the state at this
    /// turn's PROMPT-END boundary, retained so a later turn can REWIND here. See
    /// [`SpecCheckpoint`]. Refreshed by every non-empty prime; None until the first one, and on
    /// a rig too tight to hold it (a failed capture is silent — resume just isn't available).
    pub(crate) turn_ckpt: Option<SpecCheckpoint>,
    /// Session-lifetime acceptance telemetry (lane/accept-telemetry). Host-side u64 adds at
    /// the round accounting the loop already does — no syncs, no allocation. NOTE a
    /// pool-resumed session carries the PREVIOUS requests' counts; per-request consumers
    /// diff with [`SpecTelemetry::delta_since`] around each burst.
    pub telem: SpecTelemetry,
}
impl SpecSession {
    /// Context capacity of the session's caches (the server's ContextFull guard).
    pub fn cache_max_ctx(&self) -> usize {
        self.cache.max_ctx
    }
    /// Committed position this session can REWIND to (its retained prompt-end boundary), if any.
    /// A request whose prompt matches `committed[..pos]` exactly can resume from here — see
    /// `spec_rewind_to_checkpoint`.
    pub fn rewind_pos(&self) -> Option<usize> {
        self.turn_ckpt.as_ref().map(|c| c.pos)
    }
    /// Is this session in the DEMOTION-READY shape (see [`SpecSession::into_demoted`])?
    /// `false` means a carried pending must be flushed first (`spec_flush_pending`), or the
    /// session has never run a turn and has no prediction to hand over.
    pub fn demote_ready(&self) -> bool {
        self.pending_tok.is_none() && self.next_pred.is_some()
    }
    /// Does this session hold a carried pending bonus (flush required before a handoff/park)?
    pub fn has_pending(&self) -> bool {
        self.pending_tok.is_some()
    }
    /// Committed row count == cache rows (the session invariant), for the caller's own
    /// `fed`-length cross-check at a handoff boundary.
    pub fn committed_len(&self) -> usize {
        self.committed.len()
    }
    /// DEMOTION HANDOFF (lane/spec-gate, 2026-08-07): consume this session and hand its trunk
    /// cache + next-token prediction to the plain batched-decode path.
    ///
    /// WHY THIS IS EXACT (greedy). The invariant at a burst boundary is `cache.pos ==
    /// committed.len()`: every committed row has trunk KV + recurrent state, exactly as a plain
    /// tokenwise prime of the same `committed` sequence would have left it (that is the
    /// session-tail contract, and the same property `spec_rewind_to_checkpoint` and the reuse
    /// pool already rely on). `next_pred` is the argmax of the verify's logits for the LAST
    /// committed row — and verify-column logits are bit-identical to plain decode's logits at
    /// that position, because `matmul_decode_exact` bit-identity IS the basis of the greedy
    /// accept walk. So handing (cache, next_pred) to the batched path continues the stream from
    /// a state indistinguishable from one the batched path produced itself: the batched tick
    /// emits `next_pred`, feeds it into this same cache, and decodes on.
    ///
    /// `None` when the session is not in the handoff shape — a carried pending (its bonus row is
    /// NOT in the cache, so `spec_flush_pending` must commit it first) or no `next_pred` yet
    /// (never bursted). Callers must not force it: a half-committed cache handed to the batched
    /// path would silently skip a token.
    ///
    /// The MTP draft scratch, the persistent draft-graph context and the turn checkpoint are
    /// DROPPED here (freeing their VRAM): the batched path never drafts, and this handoff is
    /// one-way by design — there is no cheap symmetric re-promotion (rebuilding the draft KV
    /// would mean an `mtp_kv_fill` over the whole committed history).
    pub fn into_demoted(self) -> Option<(Cache, u32)> {
        if self.pending_tok.is_some() {
            return None;
        }
        let np = self.next_pred?;
        debug_assert_eq!(
            self.cache.pos,
            self.committed.len(),
            "demotion handoff: cache rows != committed tokens"
        );
        Some((self.cache, np))
    }
    /// Pool-resume hook (audit Q2): clear the parked draft-graph failure memoization so a
    /// NEW request resuming this session gets one fresh capture chance — a transient-pressure
    /// capture failure must not persist for the pool's whole lifetime (the TRT #16072 class).
    /// Logs once iff a flag was actually set; a no-fallback resume is silent and free.
    pub fn reset_graph_fallback_on_resume(&mut self) {
        if let Some(line) = self
            .draft_ctx
            .as_mut()
            .and_then(|c| c.failed.reset_on_resume())
        {
            eprintln!("{line}");
        }
    }
}

/// A session's PROMPT-END boundary state, the rewind target for session-affinity resume.
///
/// WHY THIS BOUNDARY, AND WHY IT IS THE ONLY ONE WORTH KEEPING. The rewrite class this lane
/// exists for (a client that strips `<think>` blocks out of prior assistant turns) mutates the
/// text the session GENERATED, never the prompt it was given. So turn N's prompt agrees with
/// turn N-1's committed tokens up to almost exactly where turn N-1's generation began — the
/// prompt-end boundary. Keeping a checkpoint there means the next turn re-primes only its own
/// delta (the rewritten answer + the new user turn) instead of the whole conversation.
///
/// WHAT IT MUST HOLD. Full-attn KV is append-only and position-addressed, so rewinding it is a
/// `len` truncation (no data). Linear-attn (GDN) conv/ssm state is mutated IN PLACE with no
/// position index, so it must be a real device COPY — that copy is the entire reason a spec
/// session could not previously rewind. The MTP draft scratch needs no copy either: its rows
/// below the boundary were written by this turn's fill and are never revisited (the per-round
/// true-hidden refresh only rewrites the CURRENT burst's committed positions), so rewinding it
/// is also just a `len` reset. `last_h` is the hidden of the last row below the boundary — the
/// predecessor-pairing anchor the next prime's fill reads for its first row.
///
/// COST: one `Cache::snapshot` per TURN, on a code path that already takes one per ROUND.
pub(crate) struct SpecCheckpoint {
    snap: crate::cache::CacheSnapshot,
    /// Committed length at the boundary (== cache.pos there, the session invariant).
    pos: usize,
    /// Pre-output_norm hidden of row `pos - 1`.
    last_h: CudaSlice<f32>,
}

/// Per-session persistent draft-graph context: the captured CUDA graph(s) plus the device
/// buffers whose POINTERS the capture bakes. Reuse legality: the greedy capture bakes only
/// session-stable pointers (the session's own MtpScratch KV — allocated once, never realloc'd;
/// the model's resident embedding; the process-wide OnceLock p_min) and the g_* buffers held
/// HERE — so one capture serves the session's whole lifetime. The sampled capture additionally
/// bakes (seed, temp) as capture-time constants and needs k q-slots — keyed by `s_key`, dropped
/// and recaptured when a pool-resumed request changes them. `*_failed` memoizes a failed capture
/// so the eager fallback doesn't pay a doomed capture attempt every burst.
pub(crate) struct DraftGraphCtx {
    g_tok: CudaSlice<u32>,
    g_pos: CudaSlice<i32>,
    g_seed: CudaSlice<f32>,
    g_p: CudaSlice<f32>,
    g_ctr: CudaSlice<u32>,
    g_q: CudaSlice<f32>,
    g_perturb: CudaSlice<f32>,
    q_slots: Vec<CudaSlice<f32>>,
    /// DRAFT-SIDE GRAMMAR MASK (lane/draft-mask): packed allowed-set words over the DRAFT
    /// head's vocab, at a STABLE address so the captured draft graph's mask node reads the
    /// per-position contents the host re-uploads before each replay (the graph-promote
    /// pattern from decode.rs). Empty unless the session drafts under a grammar.
    g_dmask: CudaSlice<u32>,
    /// was `graph` captured WITH the mask node? A parked graph of the wrong shape is dropped.
    graph_masked: bool,
    graph: Option<cudarc::driver::CudaGraph>,
    graph_s: Option<cudarc::driver::CudaGraph>,
    /// Failed-capture memoization for both graphs — LOUD on flip, cleared on pool resume
    /// (audit Q2, the TRT #16072 silent-permanent-coverage-loss class).
    failed: DraftGraphFallback,
    /// (seed, temp.to_bits(), k) baked into graph_s at its capture.
    s_key: Option<(u64, u32, usize)>,
    /// CAPTURE-RETAIN keepers (#68 root cause, 2026-08-04): the warmup-run transients whose
    /// pool addresses the captured graph(s) bake. Without these, the transients return to the
    /// pool at capture-body exit and later work (burst-boundary prime/fill/commit passes, or a
    /// co-served session in the worker) reuses those addresses — the persisted graph's replay
    /// then reads/writes live unrelated buffers (exactness corruption, first seen as the ST
    /// serve-spec 4B graph-arm corruption; one-shot CLI calls never re-shuffled the pool, which
    /// is why run-spec K=1..8 passed on the same checkpoint). Same fix class as
    /// capture_graph_retained's gemma/decode.rs sites — hold as long as the graph replays.
    keeper: Vec<Box<dyn std::any::Any + Send>>,
    keeper_s: Vec<Box<dyn std::any::Any + Send>>,
}

/// Failed-capture memoization for the two draft graphs (audit Q2, 2026-08-05 — the
/// TRT #16072 trap class: pressure-triggered, silent, long-lived coverage loss).
///
/// Three contracts:
/// - LOUD FLIP: `mark_*` returns the warn line exactly on the false→true transition
///   (returned, not printed, so the once-per-flip contract is unit-testable); the caller
///   `eprintln!`s it UNCONDITIONALLY — a dropped draft graph is never silent. Re-marking
///   an already-failed graph returns None (the per-burst memoization that keeps the eager
///   fallback from paying a doomed capture attempt every burst).
/// - RESET ON RESUME: `reset_on_resume` clears both flags — a parked session resumed by a
///   NEW request gets one fresh capture chance instead of carrying a transient-pressure
///   failure for the pool's whole lifetime. Returns the note line only when a flag was
///   actually set (quiet on the common clean-resume path).
/// - Shape-change clears (`clear_*`) stay silent, exactly as before: they precede a fresh
///   capture attempt whose own failure would re-flip loudly.
#[derive(Default)]
pub(crate) struct DraftGraphFallback {
    greedy: bool,
    sampled: bool,
}
impl DraftGraphFallback {
    fn mark_greedy(&mut self, reason: &str) -> Option<String> {
        if self.greedy {
            return None;
        }
        self.greedy = true;
        Some(format!(
            "[spec] WARN: draft-graph capture failed ({reason}); eager fallback until session resume"
        ))
    }
    fn mark_sampled(&mut self, reason: &str) -> Option<String> {
        if self.sampled {
            return None;
        }
        self.sampled = true;
        Some(format!(
            "[spec] WARN: sampled draft-graph capture failed ({reason}); eager fallback until session resume"
        ))
    }
    fn greedy_failed(&self) -> bool {
        self.greedy
    }
    fn sampled_failed(&self) -> bool {
        self.sampled
    }
    fn clear_greedy(&mut self) {
        self.greedy = false;
    }
    fn clear_sampled(&mut self) {
        self.sampled = false;
    }
    /// Pool-resume reset: both graphs get a fresh capture chance. Some(note) iff any flag
    /// was set (so clean resumes stay quiet).
    pub(crate) fn reset_on_resume(&mut self) -> Option<String> {
        if !self.greedy && !self.sampled {
            return None;
        }
        let which = match (self.greedy, self.sampled) {
            (true, true) => "greedy+sampled",
            (true, false) => "greedy",
            _ => "sampled",
        };
        self.greedy = false;
        self.sampled = false;
        Some(format!(
            "[spec] draft-graph fallback reset on session resume ({which}); recapture eligible"
        ))
    }
}

impl DraftGraphCtx {
    fn new(e: &Engine, n_embd: usize, qlen: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(DraftGraphCtx {
            g_tok: e.alloc_u32_zeroed(1)?,
            g_pos: e.htod_i32(&[0])?,
            g_seed: e.zeros(n_embd)?,
            g_p: e.zeros(1)?,
            g_ctr: e.alloc_u32_zeroed(1)?,
            g_q: e.zeros(qlen)?,
            g_perturb: e.zeros(qlen)?,
            q_slots: Vec::new(),
            g_dmask: e.alloc_u32_zeroed(1)?,
            graph_masked: false,
            graph: None,
            graph_s: None,
            failed: DraftGraphFallback::default(),
            s_key: None,
            keeper: Vec::new(),
            keeper_s: Vec::new(),
        })
    }
}

pub(crate) struct MtpScratch {
    kv: KvLayer,
    /// Row capacity. Doubles as the fa_decode_dc bucket_max for BOTH draft paths (graph + eager):
    /// n_splits is sized from it ONCE, so the graph captured at round 0 stays valid for every
    /// later t_kv (splits beyond the device len_d exit empty; the shared combine skips them) —
    /// KV growth without recapture. Eager uses the SAME bucket_max -> identical dispatch ->
    /// bit-identical drafts (the graph-vs-eager parity gate).
    cap: usize,
}
impl MtpScratch {
    fn new(
        e: &Engine,
        cfg: &memra_gguf::config::ModelConfig,
        cap: usize,
        geom: Option<&crate::hybrid::DraftGeom>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // student draft heads carry fewer KV heads (head_dim unchanged) -> smaller scratch rows.
        let n_head_kv = geom.map(|g| g.n_head_kv).unwrap_or(cfg.n_head_kv as usize);
        let head_dim_k = cfg.head_dim_k as usize;
        let head_dim_v = cfg.head_dim_v as usize;
        assert!(
            head_dim_k % 32 == 0 && head_dim_v % 32 == 0,
            "KVQUANT requires head_dim%32==0 (MTP scratch)"
        );
        let kv_dim_k = head_dim_k * n_head_kv;
        let kv_dim_v = head_dim_v * n_head_kv;
        // env-selected KV formats (default 34/24). The fp8-KV arm (MEMRA_KV_FP8) deliberately
        // does NOT reach the draft scratch: fp8 drafts drifted acceptance 69-88% -> 46%
        // (2026-07-12 A/B); the scratch is tiny, so it keeps baseline q8_0/q5_1 numerics
        // while the TRUNK cache carries the fp8 depth win. Scratch append/fa pass g=false.
        let (kbb, vbb) = crate::kv_blk_bytes();
        let k_tok_bytes = (kv_dim_k / 32) * kbb;
        let v_tok_bytes = (kv_dim_v / 32) * vbb;
        Ok(MtpScratch {
            kv: KvLayer {
                k: e.alloc_u8(cap * k_tok_bytes)?,
                v: e.alloc_u8(cap * v_tok_bytes)?,
                kv_dim_k,
                kv_dim_v,
                k_tok_bytes,
                v_tok_bytes,
                len: 0,
                len_d: e.htod_i32(&[0])?,
            },
            cap,
        })
    }
    /// Set BOTH length counters: the host mirror AND the device len_d the captured append/fa read
    /// (a 4-byte in-place htod — the counter pointer is baked into the graph, never realloc'd).
    /// This is the ONLY truncation/rollback mechanism the persistent draft KV needs.
    fn set_len(&mut self, e: &Engine, n: usize) -> Result<(), Box<dyn std::error::Error>> {
        self.kv.len = n;
        e.set_i32_one(&mut self.kv.len_d, n as i32)
    }
}

/// Retained verify intermediates for the REPLAY-FREE partial accept (2026-07-03, the profiled
/// #1 spec cost at long ctx: the partial-accept replay was a DUPLICATE trunk pass — ~0.54 extra
/// full weight reads per round — recomputing columns the verify had already produced
/// bit-identically). Holds, per linear layer, everything needed to rebuild its recurrent state
/// to "after the first j verify columns" WITHOUT re-running the trunk:
/// - BATCHED-path layers (`gdn`): the exact token-major inputs the round's ONE gdn_scan
///   consumed. A prefix re-run of the SAME kernel (t=j) from the snapshot state is bit-identical
///   to the first j iterations of the verify's scan — the kernel's t-loop carries state in
///   registers and iteration t never depends on T. `qkv_mixed` (the conv input) feeds the
///   pure-copy ring rebuild.
/// - PER-COLUMN-path layers (`cols`): dtod clones of (conv_state, ssm_state) taken after each
///   column 0..t-2 — pure copies of the actual chain states (the last column is never a rebuild
///   target: j <= t-1).
/// Full-attn layers need nothing: their verify KV rows are bit-identical to eager's (the
/// decode-exact contract; verify-probe pins it), so rollback = len truncation.
struct GdnStash {
    qkv_mixed: CudaSlice<f32>, // [t, conv_dim] token-major (conv input)
    q_l2: CudaSlice<f32>,
    k_l2: CudaSlice<f32>,
    v_g: CudaSlice<f32>, // [t, num_v, d_state]
    g_log: CudaSlice<f32>,
    beta: CudaSlice<f32>, // [t, num_v]
}
struct VerifyCkpt {
    gdn: Vec<Option<GdnStash>>, // [n_layer], Some iff batched linear path ran
    cols: Vec<Option<Vec<(CudaSlice<f32>, CudaSlice<f32>)>>>, // [n_layer][col] = (conv, ssm) after col
}
impl VerifyCkpt {
    fn new(n_layer: usize) -> Self {
        VerifyCkpt {
            gdn: (0..n_layer).map(|_| None).collect(),
            cols: (0..n_layer).map(|_| None).collect(),
        }
    }
}

impl HybridModel {
    /// NextN head forward for ONE draft token (§A ops 1-13, T=1).
    /// Inputs: `e_tok` = the token to predict FROM (last committed / previous draft); `h_seed` =
    /// the trunk's pre-output_norm hidden of that token (§A op 2 input). `mtp_pos` = absolute
    /// position of the token being predicted from. Returns (draft_logits[n_vocab] host, h_nextn dev).
    /// `h_nextn` (§A op 10) becomes `h_seed` for the next autoregressive draft step.
    /// Device-resident: returns draft logits ON DEVICE (no [n_vocab] dtoh). The greedy draft
    /// loop only needs argmax — paired with `argmax_token_device` this cuts the ~600KB logits
    /// transfer + host argmax per draft token from the K-token draft chain.
    #[allow(clippy::too_many_arguments)]
    fn mtp_head_forward_dev(
        &self,
        e: &Engine,
        mtp: &MtpHead,
        e_tok: u32,
        h_seed: &CudaSlice<f32>,
        scratch: &mut MtpScratch,
        mtp_pos: usize,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        // DRAFT-SIDE GRAMMAR MASK (lane/draft-mask): (packed draft-vocab allowed set, words).
        // Applied to the head logits BEFORE they are returned, so every consumer (argmax,
        // gumbel draw, p-min prob) sees the grammar-legal row. None = unmasked (pre-lane).
        mask: Option<(&CudaSlice<u32>, usize)>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        // Distilled-student geometry: the block runs at the INNER width `di` (eh_proj out /
        // attn / ffn); the n_embd interface (embed, norms in, carrier out, head in) is unchanged.
        let di = mtp.geom.as_ref().map(|g| g.d_inner).unwrap_or(n_embd);
        let eps = cfg.rms_eps;
        let pos_d = e.htod_i32(&[mtp_pos as i32])?;

        // op A: a resident table transfers one 4B token id. The exact host-row capacity path
        // expands this one row on CPU and transfers n_embd f32 values instead.
        let e_emb = match embd_dev {
            Some((g, qt, rb)) => e.embed_gather_device_t(g, &[e_tok], n_embd, qt, rb)?,
            None => e.htod(&self.embd.gather(n_embd, &[e_tok]))?,
        };

        // op 1/2: e_norm = RMSNorm(e, enorm); h_norm = RMSNorm(h_seed, hnorm)
        let mut e_norm = e.zeros(n_embd)?;
        e.rms_norm(&e_emb, mtp.enorm.float_data(), &mut e_norm, n_embd, 1, eps)?;
        let mut h_norm = e.zeros(n_embd)?;
        e.rms_norm(h_seed, mtp.hnorm.float_data(), &mut h_norm, n_embd, 1, eps)?;

        // op 3: concat = [e_norm ; h_norm] -> [2*n_embd], e_norm in [0,n_embd), h_norm in [n_embd,2n_embd)
        let mut concat = e.zeros(2 * n_embd)?;
        e.copy_into(&mut concat, 0, &e_norm, n_embd)?;
        e.copy_into(&mut concat, n_embd, &h_norm, n_embd)?;

        // op 4: inpSA = eh_proj @ concat  (eh_proj [2*n_embd, n_embd]) -> [n_embd]
        let inp_sa = e.matmul(&mtp.eh_proj, &concat, 1)?;

        // op 5: a_norm = RMSNorm(inpSA, attn_norm)
        let mut a_norm = e.zeros(di)?;
        e.rms_norm(&inp_sa, mtp.attn_norm.float_data(), &mut a_norm, di, 1, eps)?;

        // op 6: attention on the scratch KV. SAME dc launcher as the graph path (bucket_max =
        // scratch.cap, length from the device len_d) so eager drafts match graph drafts
        // bit-for-bit at any t_kv (the parity gate). Host len mirrored here (the dc append
        // advances only the device counter).
        let attn_out = match &mtp.mixer {
            Mixer::Full(fa) => {
                let out =
                    self.mtp_full_attn_dc(e, fa, &a_norm, &pos_d, scratch, mtp.geom.as_ref())?;
                scratch.kv.len += 1;
                out
            }
            Mixer::Linear(_) => {
                panic!("MTP block is full-attn in qwen35; linear MTP not supported")
            }
            Mixer::Mla(_) => crate::hybrid::mla_forward_unimplemented(),
        };

        // op 7: x1 = inpSA + attn_out
        let mut x1 = e.zeros(di)?;
        e.add(&inp_sa, &attn_out, &mut x1, di)?;

        // op 8: z = RMSNorm(x1, post_attn_norm)  (pre-FFN norm)
        let mut z = e.zeros(di)?;
        e.rms_norm(&x1, mtp.post_attn_norm.float_data(), &mut z, di, 1, eps)?;

        // op 9: FFN (Dense or MoE) — same as the trunk decode FFN
        let ffn_out = match &mtp.ffn {
            crate::hybrid::Ffn::Dense {
                ffn_gate,
                ffn_up,
                ffn_down,
            } => {
                let n_ff = ffn_gate.out_features();
                let (gate, up) = if e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up) {
                    let (zq, zd) = e.quantize_q8_1(&z, 1, di)?;
                    (
                        e.matmul_pre(ffn_gate, &zq, &zd, &z, 1)?,
                        e.matmul_pre(ffn_up, &zq, &zd, &z, 1)?,
                    )
                } else {
                    (e.matmul(ffn_gate, &z, 1)?, e.matmul(ffn_up, &z, 1)?)
                };
                let mut act = e.zeros(n_ff)?;
                Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, n_ff)?;
                e.matmul(ffn_down, &act, 1)?
            }
            // MTP head is a distinct block — key its experts under a separate layer index (u16::MAX)
            // so they never alias trunk layer 0's cache keys.
            crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, 1, u16::MAX)?,
        };

        // op 10: h_nextn = x1 + ffn_out (at di)
        let mut h_inner = e.zeros(di)?;
        e.add(&x1, &ffn_out, &mut h_inner, di)?;

        // op 10.5 (student): up-project the inner hidden back to n_embd — training semantics:
        // the chain carrier AND the head input are out_up(h_inner) (pre-final-norm).
        let h_nextn = match mtp.geom.as_ref() {
            Some(g) => e.matmul(&g.out_up, &h_inner, 1)?,
            None => h_inner,
        };

        // op 11: final = RMSNorm(h_nextn, shared_head_norm OR output_norm)
        let final_norm = mtp.shared_head_norm.as_ref().unwrap_or(&self.output_norm);
        let mut final_h = e.zeros(n_embd)?;
        e.rms_norm(
            &h_nextn,
            final_norm.float_data(),
            &mut final_h,
            n_embd,
            1,
            eps,
        )?;

        // op 12: draft_logits = (shared_head_head OR output) @ final — stays ON DEVICE.
        let head = mtp.shared_head_head.as_ref().unwrap_or(&self.output);
        let mut logits = e.matmul(head, &final_h, 1)?;
        // op 12b (lane/draft-mask): grammar mask over the DRAFT vocab, applied here so the
        // caller's argmax / gumbel draw / p-min prob all read the grammar-legal row.
        if let Some((mask_d, mw)) = mask {
            let d_vocab = head.out_features();
            e.mask_logits_col(&mut logits, mask_d, 0, d_vocab, mw)?;
        }
        // Chain recurrence hand-over: pre-norm h_nextn (default) or post-norm final_h
        // (MEMRA_SPEC_HPOST — llama.cpp #24025's t_h_nextn is taken AFTER the head norm).
        Ok((logits, if spec_hpost() { final_h } else { h_nextn }))
    }

    /// MTP-block full attention, T=1, on the scratch KV (BOTH draft paths — eager and graph):
    /// the scratch write slot and the attention bound come from `scratch.kv.len_d` (device i32[1])
    /// so the launch args are FIXED across draft steps — ONE captured graph serves the whole
    /// chain, and replays keep seeing KV growth through the device counter (no recapture).
    /// Geometry contract: n_splits is sized from `scratch.cap` (the persistent capacity); splits
    /// whose key range lies beyond the device t_kv exit empty and the shared combine skips them
    /// (fa_decode_dc bit-correct-for-any-t_kv<=bucket_max contract). The eager path uses the SAME
    /// launcher with the SAME bucket_max -> identical dispatch -> bit-identical draft tokens (the
    /// graph-vs-eager parity gate). Host len is NOT advanced here (graph contract); callers mirror.
    fn mtp_full_attn_dc(
        &self,
        e: &Engine,
        fa: &FullAttnLayer,
        h: &CudaSlice<f32>,
        pos_d: &CudaSlice<i32>,
        scratch: &mut MtpScratch,
        geom: Option<&crate::hybrid::DraftGeom>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_head = geom.map(|g| g.n_head).unwrap_or(cfg.n_head as usize);
        let n_head_kv = geom.map(|g| g.n_head_kv).unwrap_or(cfg.n_head_kv as usize);
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_embd = geom.map(|g| g.d_inner).unwrap_or(cfg.n_embd as usize);
        let bucket_max = scratch.cap; // < 96 guaranteed by the graph_draft eligibility gate

        let (qf, mut k, v) =
            if e.uses_q8_1_fast(&fa.wq) && e.uses_q8_1_fast(&fa.wk) && e.uses_q8_1_fast(&fa.wv) {
                let (hq, hd) = e.quantize_q8_1(h, 1, n_embd)?;
                (
                    e.matmul_pre(&fa.wq, &hq, &hd, h, 1)?,
                    e.matmul_pre(&fa.wk, &hq, &hd, h, 1)?,
                    e.matmul_pre(&fa.wv, &hq, &hd, h, 1)?,
                )
            } else {
                (
                    e.matmul(&fa.wq, h, 1)?,
                    e.matmul(&fa.wk, h, 1)?,
                    e.matmul(&fa.wv, h, 1)?,
                )
            };
        // M3/Hy3 have no attention output gate — wq out is exactly q; skip the split.
        let gated = self.cfg.attn_out_gate();
        let (mut q, gate) = if gated {
            let mut q = e.zeros(n_head * head_dim)?;
            let mut gate = e.zeros(n_head * head_dim)?;
            e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, 1)?;
            (q, Some(gate))
        } else {
            (qf, None)
        };

        let mut qn = e.zeros(n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head, eps)?;
        q = qn;
        let mut kn = e.zeros(n_head_kv * head_dim)?;
        e.rms_norm(
            &k,
            fa.k_norm.float_data(),
            &mut kn,
            head_dim,
            n_head_kv,
            eps,
        )?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        e.rope_neox(
            &mut q,
            pos_d,
            head_dim,
            rope_dims,
            n_head,
            1,
            cfg.rope_freq_base,
            1.0,
        )?;
        e.rope_neox(
            &mut k,
            pos_d,
            head_dim,
            rope_dims,
            n_head_kv,
            1,
            cfg.rope_freq_base,
            1.0,
        )?;

        let kv = &mut scratch.kv;
        // append at the DEVICE slot (kv.len_d == old len), then advance the counter in-graph.
        e.append_kv_quantized_dc(
            &k,
            &v,
            &mut kv.k,
            &mut kv.v,
            &kv.len_d,
            kv.kv_dim_k,
            kv.kv_dim_v,
            kv.k_tok_bytes,
            kv.v_tok_bytes,
            false,
        )?;
        e.inc_seqlen(&mut kv.len_d)?;
        // full-buffer views (any in-round t_kv stays in range on replay); the kernel bounds the
        // key range from the device counter.
        let k_view = e.view_u8(&kv.k, kv.k.len());
        let v_view = e.view_u8(&kv.v, kv.v.len());
        let (ktb, vtb) = (kv.k_tok_bytes, kv.v_tok_bytes);
        let mut attn = e.zeros(n_head * head_dim)?;
        e.fa_decode_dc(
            &q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv, &kv.len_d, bucket_max,
            scale, ktb, vtb, false,
        )?;

        let attn_g = match &gate {
            Some(gate) => {
                let mut gsig = e.zeros(n_head * head_dim)?;
                e.sigmoid(gate, &mut gsig, n_head * head_dim)?;
                let mut ag = e.zeros(n_head * head_dim)?;
                e.mul(&attn, &gsig, &mut ag, n_head * head_dim)?;
                ag
            }
            None => attn,
        };
        Ok(e.matmul(&fa.wo, &attn_g, 1)?)
    }

    /// PERSISTENT-DRAFT-KV fill (the reference engine's "mtp_update" analogue): compute the MTP
    /// block's K/V for `tokens` (committed tokens at positions pos0..pos0+T) from their EXACT
    /// trunk hiddens `h` ([T, n_embd] token-major, pre-output_norm) and append at slots pos0.. of
    /// the scratch KV. K/V-ONLY — ops A/1-5 plus the K-side of op 6 (wk/wv + k_norm + rope +
    /// quantized append); no wq/attention/FFN/lm_head, so per-token cost ~= eh_proj + wk/wv (a
    /// small fraction of one trunk layer), T-batched. Rope follows the chain convention
    /// rope(token@p) = p+1. Runs at round boundaries OUTSIDE the captured graph in BOTH draft
    /// modes -> draft parity by construction. Caller must have scratch.kv.len == pos0.
    #[allow(clippy::too_many_arguments)]
    fn mtp_kv_fill(
        &self,
        e: &Engine,
        mtp: &MtpHead,
        tokens: &[u32],
        h: &CudaSlice<f32>,
        pos0: usize,
        scratch: &mut MtpScratch,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let t = tokens.len();
        assert_eq!(scratch.kv.len, pos0, "mtp_kv_fill: append slot mismatch");
        assert!(pos0 + t <= scratch.cap, "mtp_kv_fill: scratch overflow");
        let Mixer::Full(fa) = &mtp.mixer else {
            panic!("MTP block is full-attn in qwen35; linear MTP not supported")
        };
        let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i + 1) as i32).collect();
        let pos_d = e.htod_i32(&pos_vec)?;

        // ops A/1/2: embed + the two input norms, T-wide.
        let e_emb = match embd_dev {
            Some((g, qt, rb)) => e.embed_gather_device_t(g, tokens, n_embd, qt, rb)?,
            None => e.htod(&self.embd.gather(n_embd, tokens))?,
        };
        let mut e_norm = e.zeros(t * n_embd)?;
        e.rms_norm(&e_emb, mtp.enorm.float_data(), &mut e_norm, n_embd, t, eps)?;
        let mut h_norm = e.zeros(t * n_embd)?;
        e.rms_norm(h, mtp.hnorm.float_data(), &mut h_norm, n_embd, t, eps)?;

        // op 3: per-row [e_norm ; h_norm] concat, token-major [T, 2*n_embd].
        let mut concat = e.zeros(t * 2 * n_embd)?;
        for i in 0..t {
            e.copy_view_into(
                &mut concat,
                i * 2 * n_embd,
                &e_norm.slice(i * n_embd..(i + 1) * n_embd),
                n_embd,
            )?;
            e.copy_view_into(
                &mut concat,
                i * 2 * n_embd + n_embd,
                &h_norm.slice(i * n_embd..(i + 1) * n_embd),
                n_embd,
            )?;
        }

        // ops 4/5: eh_proj + attn_norm, T-wide (at the student inner width when geom is set).
        let di = mtp.geom.as_ref().map(|g| g.d_inner).unwrap_or(n_embd);
        let inp_sa = e.matmul(&mtp.eh_proj, &concat, t)?;
        let mut a_norm = e.zeros(t * di)?;
        e.rms_norm(&inp_sa, mtp.attn_norm.float_data(), &mut a_norm, di, t, eps)?;

        // op 6 (K/V half): wk/wv + k_norm + rope + per-row quantized append. No wq/attention —
        // the fill only has to leave correct K/V rows behind for later chains to attend over.
        let n_head_kv = mtp
            .geom
            .as_ref()
            .map(|g| g.n_head_kv)
            .unwrap_or(cfg.n_head_kv as usize);
        let head_dim = cfg.head_dim_k as usize;
        let mut k = e.matmul(&fa.wk, &a_norm, t)?;
        let v = e.matmul(&fa.wv, &a_norm, t)?;
        let mut kn = e.zeros(t * n_head_kv * head_dim)?;
        e.rms_norm(
            &k,
            fa.k_norm.float_data(),
            &mut kn,
            head_dim,
            n_head_kv * t,
            eps,
        )?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        e.rope_neox(
            &mut k,
            &pos_d,
            head_dim,
            rope_dims,
            n_head_kv,
            t,
            cfg.rope_freq_base,
            1.0,
        )?;

        let kv = &mut scratch.kv;
        for i in 0..t {
            let k_row = k.slice(i * kv.kv_dim_k..(i + 1) * kv.kv_dim_k);
            let v_row = v.slice(i * kv.kv_dim_v..(i + 1) * kv.kv_dim_v);
            e.append_kv_quantized_view(
                &k_row,
                &v_row,
                &mut kv.k,
                &mut kv.v,
                kv.len + i,
                kv.kv_dim_k,
                kv.kv_dim_v,
                kv.k_tok_bytes,
                kv.v_tok_bytes,
                false,
            )?;
        }
        kv.len += t;
        e.set_i32_one(&mut kv.len_d, kv.len as i32)?;
        Ok(())
    }

    /// CAPTURE body for the GRAPH DRAFT (stage 2 of graph-grade spec): ONE MTP head forward with
    /// every varying input device-resident —
    ///   - token id from the persistent `tok_d` (the previous replay's in-graph argmax wrote it,
    ///     so the chain feeds itself; the host reads the same 4 bytes for the draft list),
    ///   - h_seed from the persistent `h_seed_d` (h_nextn is copied BACK into it at the end),
    ///   - rope pos from the persistent `pos_d` counter (inc'd in-graph),
    ///   - scratch KV slot/bound from `scratch.kv.len_d` (see mtp_full_attn_dc).
    /// The p-min confidence lands in the persistent `p_d` iff `with_prob` (env is fixed per run).
    /// Same kernels, same dispatch as the eager mtp_head_forward_dev chain -> same draft tokens
    /// (exactness never depends on drafts — the verify arbitrates — but acceptance parity does).
    /// `with_head=false` captures the HEAD-LESS twin for the pseudo-seed replay (2026-07-03):
    /// the pseudo pass only needs h_nextn (op 10) + the scratch append — the lm_head read
    /// (~1.06ms q6_K on the 9B), argmax and prob are dead weight there. h_nextn's inputs are
    /// untouched, so the seed value is identical; round-start resets overwrite tok_d/p_d anyway.
    /// `sampled_cap` = Some((ctr_d, perturb_d, q_out_d, seed, temp)) captures the SAMPLED twin
    /// (step 3 of the sampled-spec arc): head logits are retained in the persistent `q_out_d`
    /// (host D2Ds them to the round's q slot after each replay), the DEVICE event counter is
    /// bumped in-graph, and the argmax reads GUMBEL-PERTURBED logits — one categorical draw per
    /// replay, bit-identical to the eager arm's gumbel_perturb at the same (seed, sctr, temp).
    /// seed/temp are capture-time constants (fixed per generate call, like p_min).
    #[allow(clippy::too_many_arguments)]
    fn mtp_head_forward_cap(
        &self,
        e: &Engine,
        mtp: &MtpHead,
        tok_d: &mut CudaSlice<u32>,
        pos_d: &mut CudaSlice<i32>,
        h_seed_d: &mut CudaSlice<f32>,
        p_d: &mut CudaSlice<f32>,
        scratch: &mut MtpScratch,
        with_prob: bool,
        with_head: bool,
        embd_gpu: &CudaSlice<u8>,
        embd_qt: i32,
        embd_rb: usize,
        d_vocab: usize,
        sampled_cap: Option<(
            &mut CudaSlice<u32>,
            &mut CudaSlice<f32>,
            &mut CudaSlice<f32>,
            u64,
            f32,
        )>,
        stream_pack: Option<(&mut CudaSlice<u32>, usize, Option<&CudaSlice<u32>>)>,
        // DRAFT-SIDE GRAMMAR MASK (lane/draft-mask): (packed draft-vocab allowed-set buffer,
        // word count). Captured as ONE mask_logits_f32 node between the head matmul and the
        // in-graph argmax; the buffer address is baked, its CONTENTS are re-uploaded by the
        // host before every replay (the decode.rs graph-mask pattern). All-ones contents = a
        // no-op ban, so a position the grammar cannot constrain costs one pass over the row.
        mask_cap: Option<(&CudaSlice<u32>, usize)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        // student inner width (see mtp_head_forward_dev) — interface dims stay n_embd.
        let di = mtp.geom.as_ref().map(|g| g.d_inner).unwrap_or(n_embd);
        let eps = cfg.rms_eps;
        let e_emb = e.embed_gather_device(embd_gpu, tok_d, n_embd, embd_qt, embd_rb)?;
        let mut e_norm = e.zeros(n_embd)?;
        e.rms_norm(&e_emb, mtp.enorm.float_data(), &mut e_norm, n_embd, 1, eps)?;
        let mut h_norm = e.zeros(n_embd)?;
        e.rms_norm(
            &*h_seed_d,
            mtp.hnorm.float_data(),
            &mut h_norm,
            n_embd,
            1,
            eps,
        )?;
        let mut concat = e.zeros(2 * n_embd)?;
        e.copy_into(&mut concat, 0, &e_norm, n_embd)?;
        e.copy_into(&mut concat, n_embd, &h_norm, n_embd)?;
        let inp_sa = e.matmul(&mtp.eh_proj, &concat, 1)?;
        let mut a_norm = e.zeros(di)?;
        e.rms_norm(&inp_sa, mtp.attn_norm.float_data(), &mut a_norm, di, 1, eps)?;
        let attn_out = match &mtp.mixer {
            Mixer::Full(fa) => {
                self.mtp_full_attn_dc(e, fa, &a_norm, pos_d, scratch, mtp.geom.as_ref())?
            }
            Mixer::Linear(_) => {
                panic!("MTP block is full-attn in qwen35; linear MTP not supported")
            }
            Mixer::Mla(_) => crate::hybrid::mla_forward_unimplemented(),
        };
        let mut x1 = e.zeros(di)?;
        e.add(&inp_sa, &attn_out, &mut x1, di)?;
        let mut z = e.zeros(di)?;
        e.rms_norm(&x1, mtp.post_attn_norm.float_data(), &mut z, di, 1, eps)?;
        let ffn_out = match &mtp.ffn {
            crate::hybrid::Ffn::Dense {
                ffn_gate,
                ffn_up,
                ffn_down,
            } => {
                let n_ff = ffn_gate.out_features();
                let (gate, up) = if e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up) {
                    let (zq, zd) = e.quantize_q8_1(&z, 1, di)?;
                    (
                        e.matmul_pre(ffn_gate, &zq, &zd, &z, 1)?,
                        e.matmul_pre(ffn_up, &zq, &zd, &z, 1)?,
                    )
                } else {
                    (e.matmul(ffn_gate, &z, 1)?, e.matmul(ffn_up, &z, 1)?)
                };
                let mut act = e.zeros(n_ff)?;
                Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, n_ff)?;
                e.matmul(ffn_down, &act, 1)?
            }
            // ROUND-STREAM: the 35B NextN block carries a MoE FFN. With RESIDENT experts the
            // dev path is pure device launches (device top-k + rows kernels, ZERO-DtoH by
            // design) — capture-legal. Non-resident (SLRU-lock) stays rejected: the capture
            // error arm degrades the caller to eager/stream-off.
            crate::hybrid::Ffn::Moe(m) if m.dev_exps.is_some() => {
                self.moe_ffn_il(e, m, &z, 1, u16::MAX)?
            }
            crate::hybrid::Ffn::Moe(_) => {
                return Err("graph draft requires a Dense (or resident-MoE) MTP FFN".into())
            }
        };
        let mut h_inner = e.zeros(di)?;
        e.add(&x1, &ffn_out, &mut h_inner, di)?;
        // student: up-project back to n_embd (carrier + head input; see mtp_head_forward_dev).
        let h_nextn = match mtp.geom.as_ref() {
            Some(g) => e.matmul(&g.out_up, &h_inner, 1)?,
            None => h_inner,
        };
        // MEMRA_SPEC_HPOST needs final_h even head-less (it IS the next seed under that convention).
        let final_h = if with_head || spec_hpost() {
            let final_norm = mtp.shared_head_norm.as_ref().unwrap_or(&self.output_norm);
            let mut fh = e.zeros(n_embd)?;
            e.rms_norm(&h_nextn, final_norm.float_data(), &mut fh, n_embd, 1, eps)?;
            Some(fh)
        } else {
            None
        };
        if with_head {
            let head = mtp.shared_head_head.as_ref().unwrap_or(&self.output);
            let mut logits = e.matmul(head, final_h.as_ref().unwrap(), 1)?;
            // DRAFT-SIDE GRAMMAR MASK: ban the grammar-illegal draft ids IN the captured chain,
            // before the argmax — proposals become legal by construction. Contents-only
            // per-replay upload keeps the capture valid.
            if let Some((mask_d, mw)) = mask_cap {
                e.mask_logits_col(&mut logits, mask_d, 0, d_vocab, mw)?;
            }
            if let Some((ctr_d, perturb_d, q_out_d, seed, temp)) = sampled_cap {
                // SAMPLED chain: retain q (raw head logits -> persistent q_out_d; the matmul's
                // own buffer is pool-recycled after the capture body returns, so it can't be the
                // retention target), bump the device event counter, gumbel-perturb reading it,
                // and argmax the PERTURBED logits into tok_d — the in-graph categorical draw.
                e.copy_into(q_out_d, 0, &logits, d_vocab)?;
                e.sctr_inc(ctr_d)?;
                e.gumbel_perturb_ctr(&logits, perturb_d, d_vocab, seed, ctr_d, temp)?;
                e.argmax_token_device_into(perturb_d, tok_d, d_vocab)?;
                // p-min prob = the head's RAW softmax confidence in the SAMPLED pick — same
                // semantics as the eager sampled arm's prob_of_token_device(dl_d, tok_d).
                if with_prob {
                    e.prob_of_token_device_into(&logits, tok_d, p_d, d_vocab)?;
                }
            } else {
                // draft token -> persistent tok_d (next replay's embed reads it; host reads the 4 bytes).
                e.argmax_token_device_into(&logits, tok_d, d_vocab)?;
                // p-min under a draft mask reads the MASKED row: confidence relative to the
                // grammar-LEGAL alternatives (illegal ids leave the softmax denominator), which
                // is the right semantics for "does the drafter know what comes next here" and
                // the same row the pick came from. Draft-quality only — verify arbitrates.
                if with_prob {
                    e.prob_of_token_device_into(&logits, tok_d, p_d, d_vocab)?;
                }
            }
        }
        // ROUND-STREAM K-chain: pack (tok, p) into slot j, then remap tok through d2t so the
        // NEXT chained body's embed reads the TARGET id — zero host involvement per step.
        if let Some((out, slot, d2t)) = stream_pack {
            e.pack_tok_p(tok_d, p_d, out, slot)?;
            if let Some(map) = d2t {
                e.tok_map_u32(tok_d, map)?;
            }
        }
        // Next draft step's h_seed: pre-norm h_nextn (default) or post-norm final_h (HPOST).
        if spec_hpost() {
            e.copy_into(h_seed_d, 0, final_h.as_ref().unwrap(), n_embd)?;
        } else {
            e.copy_into(h_seed_d, 0, &h_nextn, n_embd)?;
        }
        // advance the draft rope position in-graph.
        e.inc_seqlen(pos_d)?;
        Ok(())
    }

    /// Batched target verify forward over `tokens` at positions `pos0..pos0+T` (§D.3, T=K+1).
    /// Returns ALL T logit columns (host f32, [T*n_vocab]); appends T cols to every full-attn KV
    /// and advances every linear-attn recur state by T steps (the recur steps are SEQUENTIAL T=1).
    /// Advances `cache.pos` by T.
    pub fn decode_step_t(&self, e: &Engine, tokens: &[u32], pos0: usize, cache: &mut Cache)
                         -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if self.is_gemma4_e4b() {
            return Ok(self.gemma4_e4b_decode_step_t_h(e, tokens, pos0, cache)?.0);
        }
        if self.cfg.gemma4.is_some() {
            return self.gemma4_decode_step_t(e, tokens, pos0, cache);
        }
        Ok(self.decode_step_t_h(e, tokens, pos0, cache)?.0)
    }

    /// Like `decode_step_t` but ALSO returns the LAST column's pre-output_norm hidden (h_seed for
    /// the next draft round). This lets partial-accept replay run as ONE batched T=(n_acc+1) forward
    /// (single weight read) instead of n_acc+1 separate T=1 decode_steps (n_acc+1 weight reads).
    /// At batch=1 decode is bandwidth-bound, so batching the replay is THE MTP profitability lever.
    pub fn decode_step_t_h(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
    ) -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        self.decode_step_t_h_emb(e, tokens, pos0, cache, None)
    }

    /// Like `decode_step_t_h` with an optional RESIDENT embed table (spec hot loop): device
    /// gather instead of host dequant + [T, n_embd] f32 htod. Bit-identical rows.
    pub fn decode_step_t_h_emb(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
    ) -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (logits_d, h_seed) = self.decode_step_t_h_emb_dev(e, tokens, pos0, cache, embd_dev)?;
        Ok((e.dtoh(&logits_d)?, h_seed))
    }

    /// DEVICE-LOGITS verify forward (spec device-argmax lever): identical kernel chain to
    /// `decode_step_t_h_emb` but returns the [T, n_vocab] logits ON DEVICE — the accept walk
    /// argmaxes each column on-device and reads back ONE [T] u32 instead of dtoh'ing the full
    /// T x n_vocab f32 block (~1-4 MB + T host argmaxes, every round). Kernel dispatch is
    /// UNCHANGED (same decode-exact kernels); only the post-logits transfer moves.
    pub fn decode_step_t_h_emb_dev(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let t = tokens.len();
        let (logits, x) = self.decode_step_t_core(e, tokens, pos0, cache, embd_dev, None)?;
        // h_seed for the next round = LAST column's pre-output_norm hidden ([n_embd]).
        let mut hs = vbuf(e, n_embd)?; // fully written by copy_view_into below
        e.copy_view_into(&mut hs, 0, &x.slice((t - 1) * n_embd..t * n_embd), n_embd)?;
        Ok((logits, hs))
    }

    /// CORE verify forward: the `decode_step_t_h_emb_dev` kernel chain, returning the FULL
    /// pre-output_norm hidden stack x ([T, n_embd], any column extractable) and optionally
    /// filling a `VerifyCkpt` (retained per-layer state-rebuild inputs) for the REPLAY-FREE
    /// partial accept. `ckpt: None` => byte-for-byte the old behavior (the ckpt writes are pure
    /// retains/copies — they never change what any kernel computes).
    fn decode_step_t_core(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        mut ckpt: Option<&mut VerifyCkpt>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        self.decode_step_t_core_stream(e, tokens, pos0, cache, embd_dev, ckpt.take(), None)
    }

    /// ROUND-STREAM stage (c) 4: `stream` = (device verify tokens [t], device pos counter) —
    /// when Some, rope positions come from pos_iota over the counter, the embed gathers the
    /// device tokens, and full_attn_verify routes appends/FA through the _dc twins reading the
    /// SAME counter (every layer's kvl.len == cache.pos, one counter drives all three). The
    /// host `tokens`/`pos0` args still size buffers (t is FIXED K+1 in stream mode).
    #[allow(clippy::too_many_arguments)]
    fn decode_step_t_core_stream(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        mut ckpt: Option<&mut VerifyCkpt>,
        stream: Option<(&CudaSlice<u32>, &CudaSlice<i32>)>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        // PP DOOR (lane/pp2-spec 2026-08-06): the verify trunk now takes its OWN stage split,
        // exactly as the eager and batched steps do. This is the single funnel every verify
        // forward reaches (decode_step_t / _h / _h_emb / _h_emb_dev / _core all land here), so
        // wiring it here wires the whole spec surface — the draft/accept/commit machinery above
        // is untouched.
        //
        // History: pp2-hardening (2026-08-06) made this funnel FAIL CLOSED, because its trunk
        // walk was unsplit on one stream and a sharded cross-device placement peer-read every
        // remote layer's weights on every spec round (measured 13.9-28x on the batched twin).
        // The refusal below survives to cover the residue — MEMRA_SPEC_PP=0, MEMRA_PP_STREAMS=0,
        // or a placement whose PpNRt fails to build — so a config that would still walk the
        // whole trunk on one stream refuses instead of regressing 28x.
        if let Some(fence) = crate::pp::pp_cuts(self.layers.len()) {
            if !crate::pp::pp2_streams_off() && crate::pp::spec_pp_on() {
                return self.decode_step_t_core_ppn(
                    e, tokens, pos0, cache, embd_dev, ckpt.take(), stream, &fence,
                );
            }
        }
        crate::pp::refuse_unsplit_if_remote(
            "decode_step_t (spec verify)",
            "drop MEMRA_SPEC_PP=0 / MEMRA_PP_STREAMS=0 so the verify trunk takes its OWN stage \
             split (decode_step_t_core_ppn); or run spec on one device",
        )?;
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let t = tokens.len();
        let pos_d = match stream {
            Some((_, ctr)) => {
                let mut p = e.alloc_uninit::<i32>(t)?;
                e.pos_iota(ctr, &mut p, t)?;
                p
            }
            None => {
                let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
                e.htod_i32(&pos_vec)?
            }
        };

        // embed T tokens -> [T, n_embd] token-major (device gather on the spec hot loop)
        let x = match (stream, embd_dev) {
            (Some((vtok, _)), Some((g, qt, rb))) => {
                e.embed_gather_device_td(g, vtok, t, n_embd, qt, rb)?
            }
            (None, Some((g, qt, rb))) => e.embed_gather_device_t(g, tokens, n_embd, qt, rb)?,
            _ => e.htod(&self.embd.gather(n_embd, tokens))?,
        };

        // TRUNK WALK: layers [0, n_layers) through the SAME range-scoped subgraph the PP-N
        // stage split calls per stage (`verify_layers`) — one code path, so the split cannot
        // drift from the unsplit dispatch mirroring. lane/pp2-spec 2026-08-06.
        let x = self.verify_layers(
            e, x, 0, self.layers.len(), &pos_d, t, cache, ckpt.take(), stream,
        )?;

        let mut hn = vbuf(e, t * n_embd)?; // fully written by rms_norm_decode
        e.rms_norm_decode(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let logits = e.matmul_decode_exact(&self.output, &hn, t)?;
        // stream: the device pos counter owns position; host mirror reconciles at drain.
        if stream.is_none() {
            cache.pos += t;
        }
        // Hidden stack for seeds/refresh-fills: pre-norm x (default) or post-norm hn (HPOST).
        Ok((logits, if spec_hpost() { hn } else { x }))
    }

    /// THE VERIFY TRUNK OVER PP-N (lane/pp2-spec 2026-08-06): `decode_step_t_core_stream`'s walk
    /// as N stage subgraphs, each on its own engine/stream (and, under `MEMRA_PP_DEVICES`, its own
    /// device), with a `[T, n_embd]` boundary transfer between them. T = K+1 (the verify batch),
    /// so this is the batched-boundary shape the pp2-batch lane's grow-only slots already handle
    /// (`tx(b, x, t*n_embd)`; the slot grows to the high-water T and the transport moves exactly
    /// the payload).
    ///
    /// Structure is `decode_step_batch_ppn`'s, which is `decode_step_h_ppn`'s. FOUR THINGS ARE
    /// PER-STAGE and each for a measured reason (see `decode_step_batch_ppn`'s header for the
    /// receipts):
    ///
    /// 1. THE ENGINE (`rt.engine(s, e)`) — `Engine` owns lazily-grown stable-pointer scratch
    ///    (`fa_part_pool`, `fa_vf16_scratch`, `argmax_partials`) that is single-stream-safe BY
    ///    DESIGN. Two stage streams through one Engine is the 2026-08-02 shared-scratch race
    ///    (35% flake, nondeterministic all-logits divergence). `PpNRt::build` gives every stage
    ///    s>0 its own Engine even on the primary device; honouring it here is what scopes the
    ///    pools. The verify path allocates MORE of that scratch than eager decode does (FA at
    ///    m=T, and the per-layer `GdnStash` retains), so this is load-bearing, not inherited.
    ///
    /// 2. `pos_d` — each stage uploads its OWN copy of the T rope positions on ITS stream, so the
    ///    buffer is allocated, consumed and freed on one stream. In `stream` mode that means each
    ///    stage runs its own `pos_iota` over the SHARED device counter (`pos_ctr`): the counter is
    ///    read-only during the forward (the round's `inc`/`copy_add` happen outside it), so every
    ///    stage derives the identical iota, and each stage's own output buffer is stream-local.
    ///
    /// 3. THE EMBED lives with stage 0 (`self.embd` / `embd_gpu` are host/primary-side; the
    ///    sharded loader leaves the table with stage 0 by construction).
    ///
    /// 4. THE HEAD (`output_norm` + `output`) runs on the LAST stage — the sharded loader uploaded
    ///    both through that stage's engine (`hybrid.rs`: `e_head = layer_engine(e, n_trunk,
    ///    n_trunk-1)`), so reading them anywhere else is a peer read of the biggest tensor in the
    ///    model, every round.
    ///
    /// WHAT STAYS ON THE PRIMARY, deliberately: the returned logits and hidden stack `x`. Both are
    /// last-stage-allocated device buffers, and every consumer (the device argmax walk, the accept
    /// kernels, `spec_seed_gather`, the ckpt rebuild in `commit_verified_prefix`) reads them
    /// through the primary context by UVA — the same read the batched serving epilogue's
    /// `last_logits_dev` park does. Those consumers are per-round O(T x n_vocab) and O(n_embd),
    /// not per-layer, so they are not the 28x class; splitting them is a separate lane.
    ///
    /// The MTP HEAD (draft side) is NOT split: it is one block, it lives wherever the loader put
    /// it (`load_mtp` uses the primary engine), and it is ~1-2 GB against the trunk's tens. Draft
    /// placement is measured, not assumed — see `research/pp2-spec-20260806`.
    ///
    /// EXACTNESS: PP-N adds ZERO deviation. Each stage runs the SAME kernels on the SAME bytes in
    /// the same order via the SAME `verify_layers` the unsplit body calls; the only change is
    /// where the residual is materialized, and the boundary is a straight f32 copy (dtod
    /// same-device / `cudaMemcpyPeerAsync` cross-device, no conversion). So the split MUST be
    /// BIT-IDENTICAL to the unsplit verify at the same T, in both placement orders. Gate:
    /// `decode-batch-gate --mode ppspec`. Acceptance counts are a DERIVED consequence — greedy
    /// accept argmaxes these logits, so bit-identical logits force identical accept walks; the
    /// `run-spec` K=1..8 arm checks that end-to-end rather than trusting the implication.
    #[allow(clippy::too_many_arguments)]
    fn decode_step_t_core_ppn(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        mut ckpt: Option<&mut VerifyCkpt>,
        stream: Option<(&CudaSlice<u32>, &CudaSlice<i32>)>,
        fence: &[usize],
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        assert!(
            !self.is_gemma4_e4b() && self.cfg.gemma4.is_none(),
            "decode_step_t_core_ppn covers the hybrid non-gemma4 verify trunk only \
             (the gemma4 arms have their own decode_step_t twins)"
        );
        let rt = crate::pp::PpNRt::get(e)?;
        let n_st = fence.len() - 1;
        assert_eq!(
            rt.n_stages(), n_st,
            "PpNRt stage count {} != fence stages {n_st}", rt.n_stages()
        );
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let t = tokens.len();
        let payload = t * n_embd;
        // The CALLER's ambient stream, captured BEFORE any `rt.enter()` pushes a stage stream:
        // this body returns DEVICE-RESIDENT buffers (the device-argmax accept walk's contract),
        // so the exit needs the same publication the boundaries get — see `PpNRt::publish_to`.
        // Taken here, not at the end, because inside the last-stage scope `e.stream()` IS the
        // stage stream and the wait would self-order into a no-op.
        let caller_stream = e.stream();

        // Per-stage rope positions: in host mode the same [T] iota each stage uploads itself; in
        // stream mode each stage's own `pos_iota` over the shared read-only device counter.
        let stage_pos = |es: &Engine| -> Result<CudaSlice<i32>, Box<dyn std::error::Error>> {
            match stream {
                Some((_, ctr)) => {
                    let mut p = es.alloc_uninit::<i32>(t)?;
                    es.pos_iota(ctr, &mut p, t)?;
                    Ok(p)
                }
                None => {
                    let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
                    es.htod_i32(&pos_vec)
                }
            }
        };

        // ---- STAGE 0: embed (the table lives with stage 0) + layers [0, fence[1]) + TX ----
        let mut slot = {
            let _st0 = rt.enter(0);
            let e0 = rt.engine(0, e);
            let pos_d = stage_pos(e0)?;
            let x = match (stream, embd_dev) {
                (Some((vtok, _)), Some((g, qt, rb))) => {
                    e0.embed_gather_device_td(g, vtok, t, n_embd, qt, rb)?
                }
                (None, Some((g, qt, rb))) => e0.embed_gather_device_t(g, tokens, n_embd, qt, rb)?,
                _ => e0.htod(&self.embd.gather(n_embd, tokens))?,
            };
            let x = self.verify_layers(
                e0, x, fence[0], fence[1], &pos_d, t, cache, ckpt.as_deref_mut(), stream,
            )?;
            rt.tx(0, &x, payload)?
            // x + pos_d drop here: freed stream-ordered on stage-0's stream after use.
        };

        // ---- MIDDLE STAGES: RX boundary s-1 -> range -> TX boundary s ----
        for s in 1..n_st - 1 {
            let _st = rt.enter(s);
            let es = rt.engine(s, e);
            let pos_d = stage_pos(es)?;
            let x = rt.rx(s - 1, slot, payload)?;
            let x = self.verify_layers(
                es, x, fence[s], fence[s + 1], &pos_d, t, cache, ckpt.as_deref_mut(), stream,
            )?;
            slot = rt.tx(s, &x, payload)?;
        }

        // ---- LAST STAGE: RX + final range + output_norm + lm head ----
        let _stl = rt.enter(n_st - 1);
        let el = rt.engine(n_st - 1, e);
        let pos_d = stage_pos(el)?;
        let x = rt.rx(n_st - 2, slot, payload)?;
        let x = self.verify_layers(
            el, x, fence[n_st - 1], fence[n_st], &pos_d, t, cache, ckpt.as_deref_mut(), stream,
        )?;

        let mut hn = vbuf(el, payload)?; // fully written by rms_norm_decode
        el.rms_norm_decode(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let logits = el.matmul_decode_exact(&self.output, &hn, t)?;
        // EXIT PUBLICATION: both returned buffers are still being produced on the last stage's
        // stream. Order the caller's stream behind that work before the buffers escape this
        // scope (the 2026-08-06 same-device ppspec find: without it the caller's primary-stream
        // consumer read unwritten logits — nondeterministic, one-device-only, and it poisoned
        // the following arm's KV in the same process).
        rt.publish_to(n_st - 1, &caller_stream)?;
        // stream: the device pos counter owns position; host mirror reconciles at drain.
        if stream.is_none() {
            cache.pos += t;
        }
        Ok((logits, if spec_hpost() { hn } else { x }))
    }

    /// PP-N STAGE SUBGRAPH of the verify trunk: layers `[lo, hi)` of `decode_step_t_core_stream`'s
    /// walk, verbatim. Enters with a MATERIALIZED `[T, n_embd]` residual (no pending fusion pair
    /// carried in from outside the range) and exits with the range's final residual materialized
    /// (the trailing add executed) — exactly the `decode_layers_eager(lo, hi)` contract, T rows
    /// instead of one.
    ///
    /// EXTRACTED (lane/pp2-spec 2026-08-06) rather than duplicated: `decode_step_t_core_stream` IS
    /// the single funnel every verify forward reaches, and its per-layer dispatch MIRRORING (norm
    /// fusion per layer, the t>=3/spec_m2 batched-linear window, the fused-q8 FFN chain, the
    /// decode-exact projections) is what makes verify bit-identical to eager decode. A second copy
    /// for the split arm is how those mirrors drift apart on the next lever. The unsplit body now
    /// calls this with `(0, n_layers)`, so the whole-trunk path and every stage range run the SAME
    /// code — there is no "split version" of the verify math.
    ///
    /// Bit-identity of a cut rests on the same kernel-check-pinned identity the eager arm's cut
    /// does — `add_rms_norm_q8_1 == add then rms_norm_q8_1` at nrows=T — because the ONLY thing a
    /// fence changes is that the cross-layer fusion carry breaks at `hi-1` and is re-materialized
    /// as an explicit `add`. `decode-batch-gate --mode ppspec` verifies end-to-end on real weights.
    #[allow(clippy::too_many_arguments)]
    fn verify_layers(
        &self,
        e: &Engine,
        mut x: CudaSlice<f32>,
        lo: usize,
        hi: usize,
        pos_d: &CudaSlice<i32>,
        t: usize,
        cache: &mut Cache,
        mut ckpt: Option<&mut VerifyCkpt>,
        stream: Option<(&CudaSlice<u32>, &CudaSlice<i32>)>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        // CROSS-LAYER ADD+NORM FUSION (lane/vt-fixes fix 2, mirroring decode_step_h's
        // launch-arc form): layer il's post-FFN residual add (x2 = x1 + ffn_out) and layer
        // il+1's attn_norm(+quantize) are consecutive row-wise ops — ONE add_rms_norm_q8_1
        // launch at nrows=t does all three (bit-identity pinned by the T-row kernel-check
        // arms). Carry the un-added (x1, ffn_out) pair; the fused launch materializes x2 (the
        // residual the next layer needs) as its `res` output. Falls back to the separate add
        // when the next layer is off the fused-q8 path.
        let mut pending: Option<(CudaSlice<f32>, CudaSlice<f32>)> = None;
        for il in lo..hi {
            let layer = &self.layers[il];
            // DISPATCH-MIRRORED attn-input RMSNorm (FP-order lesson #8): eager decode fuses the
            // 1024-thread rms_norm_q8_1 ONLY when every mixer projection is q8_1-fast; layers with
            // Float projections (ssm_beta/ssm_alpha on layers 1/2/4 of the 9B NVFP4 GGUF) take the
            // UNFUSED 256-thread rms_norm. The verify norm must mirror that PER-LAYER choice —
            // blockDim changes the sum-of-squares reduce order, and the ULP shift amplifies through
            // the GDN recurrence into argmax flips (measured: 9B text prompt, 1 ULP at layer 2 ->
            // 2.3e-1 logit maxdiff at the head -> K=1..8 divergence at a 0.03-margin token).
            let mixer_fast = self.mixer_in_q8_1_fast(e, &layer.mixer);
            let norm_fused = std::env::var("MEMRA_NO_FUSE_NORMQ").is_err() && mixer_fast;
            // BATCHED EPILOGUE RE-FUSE (lane/vt-fixes fix 2, 2026-08-03): when the norm is
            // dispatch-fused AND every consumer of `h` reads only its q8_1 form (Full mixer:
            // projections only; Linear mixer: the batched arm — the per-column fallback needs
            // f32 h), emit the attn-input norm DIRECTLY as q8_1 via `rms_norm_q8_1` at nrows=t
            // (row-indexed kernel — the T-row launch is the per-row m=1 program, kernel-check
            // pins bit-identity vs rms_norm_decode -> quantize_q8_1). Kills the standalone
            // quantize launch(es) + the f32 h HBM round-trip that decode never pays.
            let lin_q8_only = match &layer.mixer {
                Mixer::Linear(la) => {
                    (t >= 3 || (t == 2 && spec_m2())) && e.uses_q8_1_fast(&la.ssm_out)
                }
                _ => true,
            };
            // NOTE decode.rs's take()-first lesson: take the pending pair BEFORE branching so
            // a non-fused layer still performs the residual add.
            let taken = pending.take();
            let (h, h_q8) = if norm_fused && lin_q8_only {
                let pair = match taken {
                    // fused add + attn_norm + q8_1: ONE launch resolves the carried residual
                    // AND emits this layer's mixer input pre-quantized. res -> x2 (= new x).
                    Some((x1p, f1p)) => {
                        let mut x2 = vbuf(e, t * n_embd)?; // fully written (res output)
                        let p = e.add_rms_norm_q8_1(
                            &x1p, &f1p, layer.attn_norm.float_data(), &mut x2, n_embd, t, eps,
                        )?;
                        x = x2;
                        p
                    }
                    None => e.rms_norm_q8_1(&x, layer.attn_norm.float_data(), n_embd, t, eps)?,
                };
                (e.zeros(0)?, Some(pair)) // h unused on this path (q8-only consumers)
            } else {
                if let Some((x1p, f1p)) = taken {
                    let mut x2 = vbuf(e, t * n_embd)?; // fully written by add
                    e.add(&x1p, &f1p, &mut x2, t * n_embd)?;
                    x = x2;
                }
                let mut h = vbuf(e, t * n_embd)?; // fully written by either rms_norm arm
                if norm_fused {
                    e.rms_norm_decode(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
                } else {
                    e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
                }
                (h, None)
            };
            let h_q8_ref = h_q8.as_ref().map(|(q, d)| (q, d));

            let mixed = match &layer.mixer {
                Mixer::Full(fa) => {
                    self.full_attn_verify(e, fa, &h, h_q8_ref, pos_d, t, cache, il,
                                          stream.map(|(_, c)| c))?
                }
                Mixer::Mla(_) => crate::hybrid::mla_forward_unimplemented(),
                Mixer::Linear(la) => {
                    // BATCHED linear verify (2026-07-03, the MTP-profit lever): one T-token pass —
                    // batched projections (weight read ONCE, hits the m=2-4 weight-resident matvec),
                    // carried-state conv (ssm_conv1d_tm_state), GDN prep on the prefill kernels, and
                    // ONE gdn_scan whose internal sequential t-loop is the SAME recurrence as T
                    // chained T=1 steps (bit-identical). Falls back to the sequential per-column
                    // chain when T < d_conv-1 (conv ring update needs T >= pad) — or when ANY
                    // projection is off the q8_1 fast path: matmul_decode_exact would route a Float
                    // tensor to cuBLAS at m=t (different FP accumulation than eager's per-token
                    // GEMV), so mixed-dtype layers stay on the eager-identical per-column chain.
                    // MEMRA_SPEC_M2 (lane/spec-m2): the t==2 batch rides the same arm — the conv
                    // wrapper handles t<pad with a pure-copy ring rebuild; see spec_m2() header.
                    if (t >= 3 || (t == 2 && spec_m2()))
                        && mixer_fast
                        && e.uses_q8_1_fast(&la.ssm_out)
                    {
                        let want = ckpt.is_some();
                        let (out, stash) =
                            self.linear_attn_verify_t(e, la, &h, h_q8_ref, t, cache, il, want)?;
                        if let (Some(ck), Some(st)) = (ckpt.as_deref_mut(), stash) {
                            ck.gdn[il] = Some(st);
                        }
                        out
                    } else {
                        let mut out = vbuf(e, t * n_embd)?; // every col written by copy_into
                        let mut col_states: Option<Vec<(CudaSlice<f32>, CudaSlice<f32>)>> =
                            if ckpt.is_some() && t >= 2 {
                                Some(Vec::with_capacity(t - 1))
                            } else {
                                None
                            };
                        for col in 0..t {
                            let mut h_col = vbuf(e, n_embd)?; // fully written by copy_view_into
                            let src = h.slice(col * n_embd..(col + 1) * n_embd);
                            e.copy_view_into(&mut h_col, 0, &src, n_embd)?;
                            let m_col = self.linear_attn_decode(e, la, &h_col, cache, il)?;
                            e.copy_into(&mut out, col * n_embd, &m_col, n_embd)?;
                            // REPLAY-FREE ckpt: clone the chain's ACTUAL state after this column
                            // (pure dtod — cannot change any computed value). Last column skipped:
                            // rebuild targets are j <= t-1 columns.
                            if let Some(cs) = col_states.as_mut() {
                                if col + 1 < t {
                                    let rl = cache.recur[il].as_ref().unwrap();
                                    cs.push((
                                        e.clone_dtod(&rl.conv_state)?,
                                        e.clone_dtod(&rl.ssm_state)?,
                                    ));
                                }
                            }
                        }
                        if let (Some(ck), Some(cs)) = (ckpt.as_deref_mut(), col_states) {
                            // ReplaySSM-assessment instrumentation (2026-07-30): the
                            // per-column clones are the only true state snapshots left in
                            // the verify (the batched path stashes INPUTS and replays).
                            if std::env::var("MEMRA_SPEC_STATS").as_deref() == Ok("1") {
                                static ONCE: std::sync::Once = std::sync::Once::new();
                                let bytes: usize = cs.iter()
                                    .map(|(c, s)| (c.len() + s.len()) * 4).sum();
                                ONCE.call_once(|| eprintln!(
                                    "[verify-ckpt] per-column layer il={il}: {} clones, {:.2} MB/layer/round",
                                    cs.len(), bytes as f64 / 1e6));
                            }
                            ck.cols[il] = Some(cs);
                        }
                        out
                    }
                }
            };

            // DISPATCH-MIRRORED post-attn norm: eager residual_norm_ffn fuses add+norm+quant
            // (1024-thread add_rms_norm_q8_1) only for Dense FFNs whose gate+up are q8_1-fast;
            // otherwise (and for MoE) it runs the 256-thread fused add_rms_norm. Mirror per layer.
            let ffn_fuse = match &layer.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate, ffn_up, ..
                } => {
                    std::env::var("MEMRA_NO_FUSE_NORMQ").is_err()
                        && e.uses_q8_1_fast(ffn_gate)
                        && e.uses_q8_1_fast(ffn_up)
                }
                crate::hybrid::Ffn::Moe(_) => false,
            };
            // BATCHED EPILOGUE RE-FUSE (lane/vt-fixes fix 2): on the ffn_fuse path (Dense,
            // gate+up q8_1-fast, non-M3) the FFN input is emitted DIRECTLY as q8_1 by ONE
            // add_rms_norm_q8_1 launch at nrows=t (row-indexed kernel: the T-row launch is the
            // per-row m=1 program; kernel-check pins bit-identity vs the unfused
            // add_f32 -> rms_norm_decode -> quantize_q8_1 chain at T=2/4/5/8) — replacing the
            // add + rms_norm_decode launches AND the dual/singles' internal re-quantize.
            // M3's swigluoai must keep the f32 chain (the fused SwiGLU epilogue encodes plain
            // SiLU), mirroring residual_norm_ffn's m3 guard on the decode path.
            let fuse_q8 = ffn_fuse && self.cfg.m3.is_none();
            let mut x1 = vbuf(e, t * n_embd)?; // fully written by add / add_rms_norm*
            let mut z = e.zeros(0)?; // replaced below on the unfused arms
            let z_q8 = if fuse_q8 {
                Some(e.add_rms_norm_q8_1(
                    &x,
                    &mixed,
                    layer.post_attn_norm.float_data(),
                    &mut x1,
                    n_embd,
                    t,
                    eps,
                )?)
            } else {
                let mut zf = vbuf(e, t * n_embd)?; // fully written by rms_norm_decode / add_rms_norm
                if ffn_fuse {
                    e.add(&x, &mixed, &mut x1, t * n_embd)?;
                    e.rms_norm_decode(
                        &x1,
                        layer.post_attn_norm.float_data(),
                        &mut zf,
                        n_embd,
                        t,
                        eps,
                    )?;
                } else {
                    e.add_rms_norm(
                        &x,
                        &mixed,
                        layer.post_attn_norm.float_data(),
                        &mut x1,
                        &mut zf,
                        n_embd,
                        t,
                        eps,
                    )?;
                }
                z = zf;
                None
            };
            // DECODE-EXACT FFN projections: force MMVQ for gate/up/down at any T to match the
            // T=1 decode FP accumulation order. At T>=5 the generic matmul/matmul_pre falls to dp4a
            // (128-thread, different FP sum order). At T=2-4 the batched MMVQ is already bit-identical.
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                } => {
                    let n_ff = ffn_gate.out_features();
                    if let Some((zq, zd)) = z_q8.as_ref() {
                        // FUSED CHAIN (fix 2): pre-quantized z feeds the projections; the SwiGLU
                        // epilogue emits act pre-quantized for ffn_down (silu_mul_scaled_q8_1,
                        // bit-identical to silu_mul + quantize — kernel-check-pinned) with the
                        // NVFP4 macro-scales folded (deferred-scale dual: y*s inline == the
                        // scale_inplace store, value-exact) — the exact m=1 decode epilogue
                        // structure at nrows=t.
                        let pair = match e.matmul_decode_exact_dual_pre(ffn_gate, ffn_up, zq, zd, t)? {
                            Some(((g, gs), (u, us))) => Some((g, gs, u, us)),
                            None => None,
                        };
                        let (gate, gs, up, us) = match pair {
                            Some(x4) => x4,
                            None => (
                                e.matmul_decode_exact_pre(ffn_gate, zq, zd, t)?,
                                1.0, // scale already applied inside _pre
                                e.matmul_decode_exact_pre(ffn_up, zq, zd, t)?,
                                1.0,
                            ),
                        };
                        if e.uses_q8_1_fast(ffn_down) {
                            let (aq, ad) = e.silu_mul_scaled_q8_1(&gate, &up, gs, us, t * n_ff)?;
                            e.matmul_decode_exact_pre(ffn_down, &aq, &ad, t)?
                        } else {
                            let mut act = vbuf(e, t * n_ff)?;
                            e.silu_mul_scaled(&gate, &up, gs, us, &mut act, t * n_ff)?;
                            e.matmul_decode_exact(ffn_down, &act, t)?
                        }
                    } else {
                        // UNFUSED (pre-fix) chain — MoE-adjacent/M3/off-fast layers, unchanged.
                        // DUAL gate+up batched twin (lane/verify-economics, 2026-08-02): one launch
                        // for the pair at t=2..8 — bit-identical per (tensor,token,row) to the two
                        // singles (kernel-check pins bitwise; MEMRA_SPEC_DUAL_T=0 reverts). None
                        // (non-NVFP4 / t outside the tier / seam off) -> the two singles, unchanged.
                        let (gate, up) = match e.matmul_decode_exact_dual(ffn_gate, ffn_up, &z, t)? {
                            Some(pair) => pair,
                            None => (
                                e.matmul_decode_exact(ffn_gate, &z, t)?,
                                e.matmul_decode_exact(ffn_up, &z, t)?,
                            ),
                        };
                        let mut act = vbuf(e, t * n_ff)?; // fully written by ffn_act
                        Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, t * n_ff)?;
                        e.matmul_decode_exact(ffn_down, &act, t)?
                    }
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, t, il as u16)?,
            };
            // CROSS-LAYER fusion: defer this layer's post-FFN residual add — the next layer's
            // fused-q8 attn norm folds it in (add_rms_norm_q8_1 == add; rms_norm; quantize,
            // kernel-check-pinned at nrows=T). Non-fused next layers add explicitly above.
            pending = Some((x1, ffn_out));
        }
        // RANGE's final add (no next norm INSIDE the range to fuse with; for the
        // whole-trunk call that is the last layer, whose next norm is output_norm — f32-out).
        if let Some((x1p, f1p)) = pending.take() {
            let mut x2 = vbuf(e, t * n_embd)?; // fully written by add
            e.add(&x1p, &f1p, &mut x2, t * n_embd)?;
            x = x2;
        }
        Ok(x)
    }
    /// BATCHED linear-attn verify (T=K+1): the whole layer in ~10 launches instead of T x the
    /// T=1 decode chain (T x ~12 launches + T weight reads of the four projections). The GDN
    /// recurrence itself is inherently sequential — gdn_scan_s128 runs its internal t-loop with
    /// the SAME per-token math as chained T=1 calls (bit-identical state evolution); everything
    /// around it (projections, conv, prep, gated norm, out-proj) batches. Advances conv ring +
    /// ssm state exactly like T sequential decode steps.
    /// `want_stash`: additionally RETAIN the gdn-scan inputs (pure buffer keep-alives, zero extra
    /// kernels) so a partial accept can rebuild the state after any column prefix (REPLAY-FREE).
    #[allow(clippy::too_many_arguments)]
    fn linear_attn_verify_t(
        &self,
        e: &Engine,
        la: &LinearAttnLayer,
        h: &CudaSlice<f32>,
        h_q8: Option<(&CudaSlice<i8>, &CudaSlice<f32>)>,
        t: usize,
        cache: &mut Cache,
        il: usize,
        want_stash: bool,
    ) -> Result<(CudaSlice<f32>, Option<GdnStash>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let ssm = cfg.ssm.as_ref().unwrap();
        let d_state = ssm.state_size as usize;
        let num_k = ssm.group_count as usize;
        let num_v = ssm.time_step_rank as usize;
        let d_conv = ssm.conv_kernel as usize;
        let key_dim = d_state * num_k;
        let conv_dim = key_dim * 2 + d_state * num_v;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (d_state as f32).sqrt();

        // DECODE-EXACT projections: matmul_decode_exact forces the MMVQ (warp-per-row, 32-thread)
        // accumulation order for EVERY m, matching the T=1 decode path bit-for-bit. The generic
        // `matmul` at m>=5 falls to dp4a (128-thread, two-level reduce) which has a different FP
        // sum order — ULP differences propagate through gdn_scan and flip argmax on the 27B.
        // Q8 TRUNK-FUSION at T=1 (35B: wqkv+wqkv_gate both Q8_0): one fused2 launch, bit-identical
        // per (tensor,row) to the two m=1 MMVQ dispatches below — decode-exact contract holds.
        // VERIFY-TIER TRUNK FUSION (MEMRA_SPEC_FUSED_T, t=2-4): quantize h ONCE for every
        // fused-eligible same-input Q8_0 pair of this layer (35B wqkv+wqkv_gate; 9B
        // ssm_beta+ssm_alpha) — each fused2 batched launch then replaces two decode-exact
        // calls (each of which re-quantizes the same h + runs its own _b2/_b4 launch).
        // Bit-identical per (tensor,token,row) — see spec_fused_t().
        // BATCHED EPILOGUE RE-FUSE (lane/vt-fixes fix 2): `h_q8` = the attn-input norm emitted
        // directly as q8_1 by the caller's fused rms_norm_q8_1 (bit-identical to the unfused
        // chain, kernel-check-pinned). When present it REPLACES the standalone quantize below
        // and feeds every projection; the caller guaranteed all four input projections are
        // q8_1-fast. When absent, the old shared-quantize (fused-t window) stands.
        let h_q8_t = if h_q8.is_none()
            && spec_fused_t()
            && (2..=4).contains(&t)
            && ((e.uses_q8_1_fast(&la.wqkv) && e.uses_q8_1_fast(&la.wqkv_gate))
                || (e.uses_q8_1_fast(&la.ssm_beta) && e.uses_q8_1_fast(&la.ssm_alpha)))
        {
            Some(e.quantize_q8_1(h, t, cfg.n_embd as usize)?)
        } else {
            None
        };
        // one view: the caller's fused-norm q8 or this fn's own shared quantize.
        let hq8_any: Option<(&CudaSlice<i8>, &CudaSlice<f32>)> =
            h_q8.or(h_q8_t.as_ref().map(|(q, d)| (q, d)));
        let (qkv_mixed, z) = {
            let mut fused = None;
            if t == 1 && e.uses_q8_1_fast(&la.wqkv) && e.uses_q8_1_fast(&la.wqkv_gate) {
                let (hq, hd) = e.quantize_q8_1(h, 1, cfg.n_embd as usize)?;
                fused = e.matmul_q8_fused2(&la.wqkv, &la.wqkv_gate, &hq, &hd)?;
            } else if let Some((hq, hd)) = hq8_any {
                if spec_fused_t() && (2..=4).contains(&t) {
                    fused = e.matmul_q8_fused2_t(&la.wqkv, &la.wqkv_gate, hq, hd, t)?;
                }
            }
            match (fused, hq8_any) {
                (Some(pair), _) => pair,
                (None, Some((hq, hd))) if h_q8.is_some() => (
                    e.matmul_decode_exact_pre(&la.wqkv, hq, hd, t)?,
                    e.matmul_decode_exact_pre(&la.wqkv_gate, hq, hd, t)?,
                ),
                (None, _) => (
                    e.matmul_decode_exact(&la.wqkv, h, t)?,
                    e.matmul_decode_exact(&la.wqkv_gate, h, t)?,
                ),
            }
        };
        // beta+alpha DUAL at T=1 (75% of p3 rounds run T=1 verify — p-min chain cuts): the dual
        // mr2 kernel is bit-identical per element to the m=1 MMVQ matmul_decode_exact dispatches
        // (same warp-per-row body, blockIdx.y picks the weight), so the decode-exact contract
        // holds; the run-spec battery is the arbiter. T>1 keeps the per-tensor decode-exact path.
        let (beta_raw, alpha) = if t == 1 {
            let (hq, hd) = e.quantize_q8_1(h, 1, cfg.n_embd as usize)?;
            match e.matmul_pre_dual_noscale(&la.ssm_beta, &la.ssm_alpha, &hq, &hd, 1)? {
                Some(((mut b, bs), (mut a, as_))) => {
                    if bs != 1.0 {
                        e.scale_inplace(&mut b, bs, la.ssm_beta.out_features())?;
                    }
                    if as_ != 1.0 {
                        e.scale_inplace(&mut a, as_, la.ssm_alpha.out_features())?;
                    }
                    (b, a)
                }
                // Q8_0 fused2 twin (9B stores beta/alpha as Q8_0): DISPATCH-MIRRORS the eager
                // decode's beta_alpha closure — the fused body is qmatvec_q8_0_mmvq verbatim,
                // bit-identical per row (kernel-check rel=0.00e0 gate), so decode==verify holds.
                None => match e.matmul_q8_fused2(&la.ssm_beta, &la.ssm_alpha, &hq, &hd)? {
                    Some((b, a)) => (b, a),
                    None => (
                        e.matmul_decode_exact(&la.ssm_beta, h, 1)?,
                        e.matmul_decode_exact(&la.ssm_alpha, h, 1)?,
                    ),
                },
            }
        } else {
            // fused-t twin (9B stores beta/alpha as Q8_0): same shared-quantize + one launch
            // contract as the wqkv pair above; 35B beta/alpha are Float -> None -> fallback.
            let mut fused = None;
            if let Some((hq, hd)) = hq8_any {
                if spec_fused_t() && (2..=4).contains(&t) {
                    fused = e.matmul_q8_fused2_t(&la.ssm_beta, &la.ssm_alpha, hq, hd, t)?;
                }
            }
            match (fused, hq8_any) {
                (Some(pair), _) => pair,
                (None, Some((hq, hd))) if h_q8.is_some() => (
                    e.matmul_decode_exact_pre(&la.ssm_beta, hq, hd, t)?,
                    e.matmul_decode_exact_pre(&la.ssm_alpha, hq, hd, t)?,
                ),
                (None, _) => (
                    e.matmul_decode_exact(&la.ssm_beta, h, t)?,
                    e.matmul_decode_exact(&la.ssm_alpha, h, t)?,
                ),
            }
        };

        // conv with CARRIED state + ring roll (T >= pad rides the input-column update kernel;
        // T < pad — the MEMRA_SPEC_M2 t=2 arm — rolls via the pure-copy ring rebuild).
        let rl = cache.recur[il].as_mut().unwrap();
        let mut conv_out = e.uninit(conv_dim * t)?;
        e.ssm_conv1d_tm_state(
            &qkv_mixed,
            &mut rl.conv_state,
            la.ssm_conv1d.float_data(),
            &mut conv_out,
            conv_dim,
            t,
            d_conv,
        )?;

        // GDN prep via the prefill kernels (repack + L2 + sigmoid + glog), T-wide.
        let mut q_g = e.uninit(d_state * num_v * t)?;
        let mut k_g = e.uninit(d_state * num_v * t)?;
        let mut v_g = e.uninit(d_state * num_v * t)?;
        e.qkv_to_gdn_repack(
            &conv_out, &mut q_g, &mut k_g, &mut v_g, d_state, num_v, num_k, key_dim, t,
        )?;
        let mut q_l2 = e.uninit(d_state * num_v * t)?;
        e.l2_norm_decode(&q_g, &mut q_l2, d_state, num_v * t, eps)?;
        let mut k_l2 = e.uninit(d_state * num_v * t)?;
        e.l2_norm_decode(&k_g, &mut k_l2, d_state, num_v * t, eps)?;
        let mut beta = e.uninit(t * num_v)?;
        e.sigmoid(&beta_raw, &mut beta, t * num_v)?;
        let mut g_log = e.uninit(t * num_v)?;
        e.gdn_glog(
            &alpha,
            la.ssm_dt.float_data(),
            la.ssm_a.float_data(),
            &mut g_log,
            num_v,
            t,
        )?;

        // ONE gdn_scan over T tokens from the carried state (internal sequential loop ==
        // T chained T=1 steps). Ping-pong the resident buffers like eager decode.
        let mut o = e.uninit(d_state * num_v * t)?;
        {
            let crate::cache::RecurLayer {
                ssm_state,
                ssm_state_alt,
                ..
            } = rl;
            e.gdn_scan_s128(
                &q_l2,
                &k_l2,
                &v_g,
                &g_log,
                &beta,
                ssm_state,
                ssm_state_alt,
                &mut o,
                num_v,
                t,
                scale,
            )?;
        }
        std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);

        // gated RMSNorm + out projection, T-wide. FUSED-QUANTIZE ARM (lane/vt-fixes fix 2,
        // mirroring the T=1 decode's launch-arc form): when ssm_out rides the q8_1 fast path,
        // emit q8_1 straight from the gated norm at nrows=num_v*t (row-indexed kernel, the
        // T-wide launch is the per-row program; kernel-check pins bit-identity vs
        // gated_rmsnorm -> quantize_q8_1 at T=1 and T=5) and feed the decode-exact dispatch
        // pre-quantized — one launch replaces norm + quantize. Fallback = the f32 chain.
        let out = if e.uses_q8_1_fast(&la.ssm_out) {
            let (gq, gd) =
                e.gated_rmsnorm_q8_1(&o, la.ssm_norm.float_data(), &z, d_state, num_v * t, eps)?;
            e.matmul_decode_exact_pre(&la.ssm_out, &gq, &gd, t)?
        } else {
            let mut gn = e.uninit(d_state * num_v * t)?;
            e.gated_rmsnorm(
                &o,
                la.ssm_norm.float_data(),
                &z,
                &mut gn,
                d_state,
                num_v * t,
                eps,
            )?;
            // DECODE-EXACT out-projection: same MMVQ path as the T=1 decode (ssm_out at m>=5
            // would fall to dp4a with a different FP reduction order — same class of bug as
            // the input projs).
            e.matmul_decode_exact(&la.ssm_out, &gn, t)?
        };
        let stash = if want_stash {
            Some(GdnStash {
                qkv_mixed,
                q_l2,
                k_l2,
                v_g,
                g_log,
                beta,
            })
        } else {
            None
        };
        Ok((out, stash))
    }

    /// REPLAY-FREE partial-accept commit (2026-07-03): make the cache state == "committed through
    /// the first `j` verify columns" WITHOUT the legacy rollback + duplicate trunk replay.
    /// - Full-attn KV: truncate len to snapshot + j. The verify's appended rows for those columns
    ///   are bit-identical to what an eager T=1 chain writes (the decode-exact contract the
    ///   verify-probe gates), so keeping them == replaying them.
    /// - Linear layers, batched path: rebuild the conv ring by PURE COPIES (ring holds raw input
    ///   columns) and the ssm state by a prefix re-run of the SAME gdn_scan kernel (t=j) from the
    ///   snapshot state over the stash's identical inputs — the kernel's t-loop carries state in
    ///   registers and writes it once at the end, so iterations 0..j-1 are independent of T:
    ///   bit-identical to the verify's own state after j tokens == the eager chain state.
    /// - Linear layers, per-column path: restore the cloned actual state after column j-1.
    /// Caller guarantees 1 <= j <= t-1 (j==0 rounds take the legacy rollback; j==t is full accept).
    fn commit_verified_prefix(
        &self,
        e: &Engine,
        cache: &mut Cache,
        snap: &crate::cache::CacheSnapshot,
        ckpt: &VerifyCkpt,
        j: usize,
        kv_lens_done: bool,
        dev_j: Option<(&CudaSlice<u32>, usize, usize)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let ssm = cfg.ssm.as_ref().unwrap();
        let d_state = ssm.state_size as usize;
        let num_k = ssm.group_count as usize;
        let num_v = ssm.time_step_rank as usize;
        let d_conv = ssm.conv_kernel as usize;
        let conv_dim = d_state * num_k * 2 + d_state * num_v;
        let scale = 1.0 / (d_state as f32).sqrt();
        for il in 0..self.layers.len() {
            if let (Some(kvl), Some(saved)) = (cache.kv[il].as_mut(), snap.kv_len[il]) {
                kvl.len = saved + j;
                // devacc 3a: spec_rollback_kv already wrote len_d on-device (same value).
                if !kv_lens_done {
                    e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
                }
            }
            if let Some(rl) = cache.recur[il].as_mut() {
                if let Some(st) = &ckpt.gdn[il] {
                    let ring_old = snap.conv[il].as_ref().expect("snapshot missing conv");
                    let state_in = snap.ssm[il].as_ref().expect("snapshot missing ssm");
                    if let Some((acc, base, t_v)) = dev_j {
                        // 3b: j read on-device (_dc twins, same bodies; full accept early-exits).
                        e.ssm_conv_ring_rebuild_dc(
                            &st.qkv_mixed,
                            ring_old,
                            &mut rl.conv_state,
                            conv_dim,
                            acc,
                            base,
                            t_v,
                            d_conv,
                        )?;
                        let mut o = e.uninit(d_state * num_v * j.max(1))?;
                        e.gdn_scan_s128_dc(
                            &st.q_l2,
                            &st.k_l2,
                            &st.v_g,
                            &st.g_log,
                            &st.beta,
                            state_in,
                            &mut rl.ssm_state,
                            &mut o,
                            num_v,
                            acc,
                            base,
                            t_v,
                            scale,
                        )?;
                    } else {
                        e.ssm_conv_ring_rebuild(
                            &st.qkv_mixed,
                            ring_old,
                            &mut rl.conv_state,
                            conv_dim,
                            j,
                            d_conv,
                        )?;
                        let mut o = e.uninit(d_state * num_v * j)?; // scan output, discarded
                        e.gdn_scan_s128(
                            &st.q_l2,
                            &st.k_l2,
                            &st.v_g,
                            &st.g_log,
                            &st.beta,
                            state_in,
                            &mut rl.ssm_state,
                            &mut o,
                            num_v,
                            j,
                            scale,
                        )?;
                    }
                } else if let Some(cols) = &ckpt.cols[il] {
                    let (c, s) = &cols[j - 1];
                    e.copy_into(&mut rl.conv_state, 0, c, c.len())?;
                    e.copy_into(&mut rl.ssm_state, 0, s, s.len())?;
                } else {
                    return Err(
                        "commit_verified_prefix: verify ckpt missing for linear layer".into(),
                    );
                }
            }
        }
        cache.pos = snap.pos + j;
        Ok(())
    }

    /// ROUND-STREAM: recur restore with device-j (the _dc twins; full accept early-exits
    /// in-kernel). Requires the batched-linear stash on every linear layer (stream gate).
    fn commit_verified_prefix_stream(
        &self,
        e: &Engine,
        cache: &mut Cache,
        snap: &crate::cache::CacheSnapshot,
        ckpt: &VerifyCkpt,
        acc: &CudaSlice<u32>,
        base: usize,
        t_v: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let ssm = cfg.ssm.as_ref().unwrap();
        let d_state = ssm.state_size as usize;
        let num_k = ssm.group_count as usize;
        let num_v = ssm.time_step_rank as usize;
        let d_conv = ssm.conv_kernel as usize;
        let conv_dim = d_state * num_k * 2 + d_state * num_v;
        let scale = 1.0 / (d_state as f32).sqrt();
        for il in 0..self.layers.len() {
            if let Some(rl) = cache.recur[il].as_mut() {
                let st = ckpt.gdn[il]
                    .as_ref()
                    .ok_or("stream restore: batched-linear stash missing")?;
                let ring_old = snap.conv[il].as_ref().expect("snapshot missing conv");
                let state_in = snap.ssm[il].as_ref().expect("snapshot missing ssm");
                e.ssm_conv_ring_rebuild_dc(
                    &st.qkv_mixed,
                    ring_old,
                    &mut rl.conv_state,
                    conv_dim,
                    acc,
                    base,
                    t_v,
                    d_conv,
                )?;
                let mut o = e.uninit(d_state * num_v * t_v)?;
                e.gdn_scan_s128_dc(
                    &st.q_l2,
                    &st.k_l2,
                    &st.v_g,
                    &st.g_log,
                    &st.beta,
                    state_in,
                    &mut rl.ssm_state,
                    &mut o,
                    num_v,
                    acc,
                    base,
                    t_v,
                    scale,
                )?;
            }
        }
        Ok(())
    }

    /// EAGLE3 aux-capturing verify forward over `tokens` (T) — mirrors `decode_step_t_h` exactly
    /// (same KV append, same causal verify, same recur advance) but ALSO clones the aux residual-
    /// stream hiddens (blocks in `aux_layers`) for TWO columns: the LAST column (always) and the
    /// optional `pred_col` (the EAGLE seed = bonus's predecessor). Returns
    /// (all_T_logits host, last_col_aux, pred_col_aux?). Used by the EAGLE3 orchestrator's commit.
    pub fn decode_step_t_aux2(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        aux_layers: &[usize],
        pred_col: Option<usize>,
    ) -> Result<
        (Vec<f32>, Vec<CudaSlice<f32>>, Option<Vec<CudaSlice<f32>>>),
        Box<dyn std::error::Error>,
    > {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let t = tokens.len();
        let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
        let pos_d = e.htod_i32(&pos_vec)?;
        let mut x = e.htod(&self.embd.gather(n_embd, tokens))?;
        let mut aux_last: Vec<CudaSlice<f32>> = Vec::with_capacity(aux_layers.len());
        let mut aux_pred: Vec<CudaSlice<f32>> = Vec::new();
        let want_pred = pred_col.is_some();

        for (il, layer) in self.layers.iter().enumerate() {
            // DISPATCH-MIRRORED norms (FP-order lesson #8) — see decode_step_t_h_emb.
            let mixer_fast = self.mixer_in_q8_1_fast(e, &layer.mixer);
            let norm_fused = std::env::var("MEMRA_NO_FUSE_NORMQ").is_err() && mixer_fast;
            let mut h = vbuf(e, t * n_embd)?; // fully written by either rms_norm arm
            if norm_fused {
                e.rms_norm_decode(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            } else {
                e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            }
            let mixed = match &layer.mixer {
                Mixer::Full(fa) => {
                    self.full_attn_verify(e, fa, &h, None, &pos_d, t, cache, il, None)?
                }
                Mixer::Mla(_) => crate::hybrid::mla_forward_unimplemented(),
                Mixer::Linear(la) => {
                    let mut out = e.zeros(t * n_embd)?;
                    for col in 0..t {
                        let mut h_col = e.zeros(n_embd)?;
                        let src = h.slice(col * n_embd..(col + 1) * n_embd);
                        e.copy_view_into(&mut h_col, 0, &src, n_embd)?;
                        let m_col = self.linear_attn_decode(e, la, &h_col, cache, il)?;
                        e.copy_into(&mut out, col * n_embd, &m_col, n_embd)?;
                    }
                    out
                }
            };
            let ffn_fuse = match &layer.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate, ffn_up, ..
                } => {
                    std::env::var("MEMRA_NO_FUSE_NORMQ").is_err()
                        && e.uses_q8_1_fast(ffn_gate)
                        && e.uses_q8_1_fast(ffn_up)
                }
                crate::hybrid::Ffn::Moe(_) => false,
            };
            let mut x1 = vbuf(e, t * n_embd)?; // fully written by add / add_rms_norm
            let mut z = vbuf(e, t * n_embd)?; // fully written by rms_norm_decode / add_rms_norm
            if ffn_fuse {
                e.add(&x, &mixed, &mut x1, t * n_embd)?;
                e.rms_norm_decode(
                    &x1,
                    layer.post_attn_norm.float_data(),
                    &mut z,
                    n_embd,
                    t,
                    eps,
                )?;
            } else {
                e.add_rms_norm(
                    &x,
                    &mixed,
                    layer.post_attn_norm.float_data(),
                    &mut x1,
                    &mut z,
                    n_embd,
                    t,
                    eps,
                )?;
            }
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                } => {
                    let n_ff = ffn_gate.out_features();
                    let gate = e.matmul_decode_exact(ffn_gate, &z, t)?;
                    let up = e.matmul_decode_exact(ffn_up, &z, t)?;
                    let mut act = vbuf(e, t * n_ff)?; // fully written by ffn_act
                    Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, t * n_ff)?;
                    e.matmul_decode_exact(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, t, il as u16)?,
            };
            let mut x2 = vbuf(e, t * n_embd)?; // fully written by add
            e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
            if aux_layers.contains(&il) {
                let mut a = e.zeros(n_embd)?;
                e.copy_view_into(&mut a, 0, &x2.slice((t - 1) * n_embd..t * n_embd), n_embd)?;
                aux_last.push(a);
                if let Some(pc) = pred_col {
                    let mut ap = e.zeros(n_embd)?;
                    e.copy_view_into(
                        &mut ap,
                        0,
                        &x2.slice(pc * n_embd..(pc + 1) * n_embd),
                        n_embd,
                    )?;
                    aux_pred.push(ap);
                }
            }
            x = x2;
        }
        let mut hn = vbuf(e, t * n_embd)?; // fully written by rms_norm_decode
        e.rms_norm_decode(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let logits = e.matmul_decode_exact(&self.output, &hn, t)?;
        let host = e.dtoh(&logits)?;
        cache.pos += t;
        Ok((
            host,
            aux_last,
            if want_pred { Some(aux_pred) } else { None },
        ))
    }

    /// Full-attention mixer over T query tokens with a GROWING resident KV (verify path, §D.3).
    /// Appends the T new K/V columns to cache.kv[il] then attends causally over [0..len) via
    /// fa_prefill. Token-major [T, kv_dim] projection layout == cache row layout (single copy).
    #[allow(clippy::too_many_arguments)]
    fn full_attn_verify(
        &self,
        e: &Engine,
        fa: &FullAttnLayer,
        h: &CudaSlice<f32>,
        h_q8: Option<(&CudaSlice<i8>, &CudaSlice<f32>)>,
        pos_d: &CudaSlice<i32>,
        t: usize,
        cache: &mut Cache,
        il: usize,
        stream_ctr: Option<&CudaSlice<i32>>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_embd = cfg.n_embd as usize;

        // DECODE-EXACT Q/K/V projections: matmul_decode_exact forces the MMVQ (warp-per-row) path
        // for every m, matching the T=1 decode's FP accumulation order. matmul_pre at m>=5 would
        // fall to dp4a (128-thread, two-level reduce) with a different FP sum order.
        // Q8 TRUNK-FUSION at T=1: DISPATCH-MIRRORS the eager decode's fused3 (bit-identical body).
        // BATCHED EPILOGUE RE-FUSE (lane/vt-fixes fix 2): `h_q8` = the attn-input norm's q8_1
        // form emitted by the fused rms_norm_q8_1 (bit-identical to rms_norm_decode ->
        // quantize_q8_1, kernel-check-pinned). When present (caller checked mixer q8_1-fast),
        // every projection consumes it — `h` may be a zero-len placeholder and must not be read.
        let (qf, mut k, v) = {
            let mut fused = None;
            let qkv_fast = e.uses_q8_1_fast(&fa.wq)
                && e.uses_q8_1_fast(&fa.wk)
                && e.uses_q8_1_fast(&fa.wv);
            if t == 1 && qkv_fast {
                let (hq_o, hd_o);
                let (hq, hd): (&CudaSlice<i8>, &CudaSlice<f32>) = match h_q8 {
                    Some(p) => p,
                    None => {
                        (hq_o, hd_o) = e.quantize_q8_1(h, 1, n_embd)?;
                        (&hq_o, &hd_o)
                    }
                };
                fused = e.matmul_q8_fused3(&fa.wq, &fa.wk, &fa.wv, hq, hd)?;
            } else if spec_fused_t() && (2..=4).contains(&t) && qkv_fast {
                // VERIFY-TIER TRUNK FUSION (MEMRA_SPEC_FUSED_T): one shared quantize + one
                // fused3 batched launch replaces three decode-exact calls (3 re-quantizes of
                // the same h + 3 _b2/_b4 launches). Bit-identical per (tensor,token,row).
                let (hq_o, hd_o);
                let (hq, hd): (&CudaSlice<i8>, &CudaSlice<f32>) = match h_q8 {
                    Some(p) => p,
                    None => {
                        (hq_o, hd_o) = e.quantize_q8_1(h, t, n_embd)?;
                        (&hq_o, &hd_o)
                    }
                };
                fused = e.matmul_q8_fused3_t(&fa.wq, &fa.wk, &fa.wv, hq, hd, t)?;
            }
            match (fused, h_q8) {
                (Some(triple), _) => triple,
                // shared pre-quantized activation (q8_1-fast guaranteed by the caller): the
                // decode-exact dispatch consumes (hq, hd) instead of re-quantizing 3x.
                (None, Some((hq, hd))) if qkv_fast => (
                    e.matmul_decode_exact_pre(&fa.wq, hq, hd, t)?,
                    e.matmul_decode_exact_pre(&fa.wk, hq, hd, t)?,
                    e.matmul_decode_exact_pre(&fa.wv, hq, hd, t)?,
                ),
                (None, _) => (
                    e.matmul_decode_exact(&fa.wq, h, t)?,
                    e.matmul_decode_exact(&fa.wk, h, t)?,
                    e.matmul_decode_exact(&fa.wv, h, t)?,
                ),
            }
        };
        // M3/Hy3 have no attention output gate — wq out is exactly q; skip the split.
        let gated = self.cfg.attn_out_gate();
        let (mut q, gate) = if gated {
            let mut q = vbuf(e, t * n_head * head_dim)?; // fully written by q_gate_split
            let mut gate = vbuf(e, t * n_head * head_dim)?; // fully written by q_gate_split
            e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, t)?;
            (q, Some(gate))
        } else {
            (qf, None)
        };

        let mut qn = vbuf(e, t * n_head * head_dim)?; // fully written by rms_norm
        e.rms_norm(
            &q,
            fa.q_norm.float_data(),
            &mut qn,
            head_dim,
            n_head * t,
            eps,
        )?;
        q = qn;
        let mut kn = vbuf(e, t * n_head_kv * head_dim)?; // fully written by rms_norm
        e.rms_norm(
            &k,
            fa.k_norm.float_data(),
            &mut kn,
            head_dim,
            n_head_kv * t,
            eps,
        )?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        e.rope_neox(
            &mut q,
            pos_d,
            head_dim,
            rope_dims,
            n_head,
            t,
            cfg.rope_freq_base,
            1.0,
        )?;
        e.rope_neox(
            &mut k,
            pos_d,
            head_dim,
            rope_dims,
            n_head_kv,
            t,
            cfg.rope_freq_base,
            1.0,
        )?;

        // append T new K/V columns to the resident QUANTIZED cache. k/v are token-major [T, kv_dim]
        // f32; append-quantize each of the T token rows into the byte cache (q8_0 K / q5_1 V).
        let kvl = cache.kv[il].as_mut().unwrap();
        let (kv_dim_k, kv_dim_v, ktb, vtb) =
            (kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes);
        if let Some(ctr) = stream_ctr {
            // stream: ONE batched append at the device counter (rows kernel = the per-view warp
            // math on a (block, token) grid, documented byte-identical); host len is a stale
            // LOWER BOUND under pre-issue (drain reconciles it).
            e.append_kv_quantized_rows_dc(
                &k,
                &v,
                &mut kvl.k,
                &mut kvl.v,
                ctr,
                t,
                kv_dim_k,
                kv_dim_v,
                ktb,
                vtb,
                crate::Engine::kv_fp8_on(),
            )?;
        } else {
            for i in 0..t {
                let k_row = k.slice(i * kv_dim_k..(i + 1) * kv_dim_k);
                let v_row = v.slice(i * kv_dim_v..(i + 1) * kv_dim_v);
                e.append_kv_quantized_view(
                    &k_row,
                    &v_row,
                    &mut kvl.k,
                    &mut kvl.v,
                    kvl.len + i,
                    kv_dim_k,
                    kv_dim_v,
                    ktb,
                    vtb,
                    crate::Engine::kv_fp8_on(),
                )?;
            }
            kvl.len += t;
        }

        // BIT-IDENTICAL VERIFY ATTENTION (spec-exactness fix): the FP accumulation order must be
        // byte-for-byte identical to the eager decode path. fa_prefill uses a different tile size
        // (BLOCK_Q=64, BK=32) and online-softmax structure than fa_decode's split-K + combine,
        // which changes FP summation order and can flip argmax at tight logit margins. Query row r
        // attends to keys [0..base_len+r+1) — each successive row sees one more key (the causal
        // property). This matches eager: decode appends k at len, then fa_decode sees t_kv = len+1
        // keys. The verify appends all T tokens first but bounds the key range per row.
        //
        // MULTI-ROW FUSED PATH (the long-ctx spec fix, 2026-07-03): when every row takes the vec
        // kernel (base_len+1 >= FA_VEC_MIN_TKV), ONE fa_decode_rows launch executes the exact
        // per-row program for all T rows (grid.z = row, per-row n_splits from the same
        // fa_split_keys formula) — replacing T x (2 launches + 2 dtod copies + 5 partial allocs)
        // and multiplying resident CTAs by T on a latency-bound kernel. Bit-identical per row by
        // construction; kernel-check pins rows-vs-loop byte identity, run-spec is the end gate.
        // Short ctx (any row below the vec crossover) and MEMRA_NO_FA_VEC/MEMRA_FA_ROWS_OFF keep the
        // per-row loop (whose fa_decode picks scalar/vec per row exactly like eager decode).
        let mut attn = vbuf(e, t * n_head * head_dim)?; // fully written by every FA arm below
        let base_len = kvl.len - t; // KV len BEFORE this round's T tokens were appended
                                    // T=1 INCLUDED (2026-07-05): p-min cuts the draft to 1 in ~75% of rounds on hard
                                    // (agentic) content — the old t>1 gate sent those rounds to the per-row loop (262us/row
                                    // + q-row copy + per-row allocs vs 93us/row through the fused kernel at grid.z=1, same
                                    // program). nsys accounting: 1088 of 1456 verify FA launches were T=1 escapees.
                                    // LEAN T=1 ARM (MEMRA_SPEC_LEAN, close35): at t==1, q IS one row and fa_decode on it is
                                    // the EXACT eager decode dispatch (vec_q_v2 + combine_f32; the rows pair measured +50us
                                    // at m=1). Byte-identical: kernel-check pins rows-vs-loop identity, and the per-row loop
                                    // at t=1 is fa_decode on the same q with zero-offset copies. Gates arbitrate.
        if let Some(ctr) = stream_ctr {
            // STREAM ARM: causal base from the device counter; host kvl.len is a stale lower
            // bound used only for the split-sizing upper bound (+64 slack covers M pre-issued
            // rounds at K<=8). Views span the bound; per-row limits derive in-kernel.
            let upper = kvl.len + t + 64;
            let k_view = e.view_u8(&kvl.k, (upper.min(cache.max_ctx)) * ktb);
            let v_view = e.view_u8(&kvl.v, (upper.min(cache.max_ctx)) * vtb);
            e.fa_decode_rows_dc(
                &q,
                &k_view,
                &v_view,
                &mut attn,
                head_dim,
                n_head,
                n_head_kv,
                ctr,
                upper.min(cache.max_ctx),
                t,
                scale,
                ktb,
                vtb,
                0,
                false,
            )?;
        } else if spec_lean() && t == 1 {
            let t_kv = base_len + 1;
            let k_view = e.view_u8(&kvl.k, t_kv * ktb);
            let v_view = e.view_u8(&kvl.v, t_kv * vtb);
            e.fa_decode_kvmod(
                &q,
                &k_view,
                &v_view,
                &mut attn,
                head_dim,
                n_head,
                n_head_kv,
                t_kv,
                scale,
                ktb,
                vtb,
                crate::Engine::kv_fp8_on(),
            )?;
        } else if e.fa_rows_eligible(base_len, head_dim) {
            let k_view = e.view_u8(&kvl.k, (base_len + t) * ktb);
            let v_view = e.view_u8(&kvl.v, (base_len + t) * vtb);
            e.fa_decode_rows(
                &q,
                &k_view,
                &v_view,
                &mut attn,
                head_dim,
                n_head,
                n_head_kv,
                base_len,
                t,
                scale,
                ktb,
                vtb,
                None,
                false,
                crate::Engine::kv_fp8_on(),
                None,
            )?;
        } else {
            for r in 0..t {
                let t_kv_r = base_len + r + 1; // this row sees keys [0..t_kv_r)
                let k_view_r = e.view_u8(&kvl.k, t_kv_r * ktb);
                let v_view_r = e.view_u8(&kvl.v, t_kv_r * vtb);
                // copy q row into an owned buffer (fa_decode takes &CudaSlice, not CudaView)
                let mut q_row = vbuf(e, n_head * head_dim)?; // fully written by copy_view_into
                let q_src = q.slice(r * n_head * head_dim..(r + 1) * n_head * head_dim);
                e.copy_view_into(&mut q_row, 0, &q_src, n_head * head_dim)?;
                let mut attn_row = vbuf(e, n_head * head_dim)?; // fully written by fa_decode
                e.fa_decode_kvmod(
                    &q_row,
                    &k_view_r,
                    &v_view_r,
                    &mut attn_row,
                    head_dim,
                    n_head,
                    n_head_kv,
                    t_kv_r,
                    scale,
                    ktb,
                    vtb,
                    crate::Engine::kv_fp8_on(),
                )?;
                e.copy_into(
                    &mut attn,
                    r * n_head * head_dim,
                    &attn_row,
                    n_head * head_dim,
                )?;
            }
        }

        let attn_g = match &gate {
            Some(gate) => {
                let mut gsig = vbuf(e, t * n_head * head_dim)?; // fully written by sigmoid
                e.sigmoid(gate, &mut gsig, t * n_head * head_dim)?;
                let mut ag = vbuf(e, t * n_head * head_dim)?; // fully written by mul
                e.mul(&attn, &gsig, &mut ag, t * n_head * head_dim)?;
                ag
            }
            None => attn,
        };
        // DECODE-EXACT wo projection: at m>=5 (K=4+ with pending) the generic matmul would use dp4a
        // (128-thread, different FP sum order than MMVQ). Force MMVQ for bit-identity with decode.
        Ok(e.matmul_decode_exact(&fa.wo, &attn_g, t)?)
    }

    /// Greedy MTP speculative decode (§B). Token-identical to `generate(prompt, max_new)` but uses
    /// the NextN head to draft K tokens then verifies them in one batched target forward.
    /// Returns (generated tokens, total_drafted, total_accepted) so the caller can report
    /// acceptance rate. `k` = draft length per round.
    ///
    /// GRAPH DRAFT (stage 2 of graph-grade spec): when the model is all-Dense and the MTP head is
    /// Dense (no MoE host readbacks), the fixed-shape T=1 MTP forward is CUDA-graph-captured ONCE
    /// and replayed per draft step — the ~40 eager launches per drafted token collapse into one
    /// graph dispatch; only the 4-byte token id (and 4-byte p-min confidence) round-trip per step.
    /// Event tracking is disabled for the whole call (generate_graph pattern) so every buffer the
    /// captured graph references is event-free; the spec loop is strictly single-stream.
    /// MEMRA_SPEC_NOGRAPH=1 forces the eager draft chain.
    /// SAMPLED mode (MEMRA_SPEC_TEMP>0) has its OWN capture (gumbel-perturbed in-graph argmax,
    /// device Philox event counter, persistent q retention) — graph-vs-eager sampled streams are
    /// bit-identical for the same (seed, prompt, K, temp); see the sampled-graph setup in
    /// generate_spec_inner2.
    /// Multi-turn session: trunk cache + MTP draft scratch persist across generate calls, so
    /// turn N+1 primes ONLY its new suffix (the 124k-conversation daily pattern — re-priming a
    /// 32k history costs ~54s; a suffix prime costs seconds). APPEND-ONLY by construction: the
    /// hybrid linear-attn states are in-place (no position index), so a session can extend but
    /// never rewind — `committed` is the exact token list whose state the caches hold (includes
    /// any overshoot tokens past max_new; the caller renders from `committed`, not its own echo).
    pub fn new_session(
        &self,
        e: &Engine,
        max_ctx: usize,
    ) -> Result<SpecSession, Box<dyn std::error::Error>> {
        Ok(SpecSession {
            // STAGE-OWNED KV (lane/pp2-spec 2026-08-06): `pp::new_cache`, not `Cache::new`. This
            // is the SERVING spec-session path, and with the ppN door open across two cards a
            // primary-homed cache makes every remote stage peer-read its OWN KV on every verify
            // round — the wrong-card class already fixed on the two batched serving paths
            // (worker.rs 2483 / 2837). With the door shut `new_cache` IS `Cache::new` (same
            // branch, same allocations), so single-device behavior is byte-unchanged.
            cache: crate::pp::new_cache(e, &self.cfg, max_ctx)?,
            scratch: MtpScratch::new(
                e,
                &self.cfg,
                max_ctx,
                self.mtp.as_ref().and_then(|m| m.geom.as_ref()),
            )?,
            committed: Vec::new(),
            last_h: None,
            next_pred: None,
            sctr: 0,
            uctr: 0,
            draft_ctx: None,
            pending_tok: None,
            turn_ckpt: None,
            telem: SpecTelemetry::default(),
        })
    }

    /// SESSION-AFFINITY REWIND (lane/session-affinity, 2026-08-05): roll `sess` back to its
    /// retained prompt-end checkpoint, so a request whose prompt matches
    /// `committed[..rewind_pos()]` exactly can resume there and prime only its own delta.
    ///
    /// EXACTNESS. After this returns, the session is byte-for-byte the state it was in AT that
    /// boundary: full-attn KV truncated to it (append-only, position-addressed), GDN conv/ssm
    /// restored from the device copy taken there, draft scratch length reset, `committed`
    /// truncated, `last_h` = the boundary's predecessor anchor. That is precisely the state a
    /// fresh prime of `committed[..pos]` would have produced, so the following suffix prime and
    /// every burst after it are identical to a cold run of the same token stream — the
    /// committed-tokens-authoritative contract.
    ///
    /// `next_pred` and `pending_tok` are CLEARED: both describe generation past the boundary,
    /// which the rewind discards. The caller therefore must supply a non-empty suffix (a
    /// rewound session cannot serve an empty-suffix continuation burst — there is nothing to
    /// continue). The persistent draft graph survives: it bakes only session-stable pointers
    /// (the scratch KV, the resident embedding), none of which the rewind moves.
    ///
    /// The checkpoint is CONSUMED (`turn_ckpt` taken): its snapshot buffers are freed here, and
    /// this turn's own prime installs a fresh one at the new prompt end. Returns the position
    /// rewound to, or `None` when the session holds no checkpoint (caller: full re-prime).
    pub fn spec_rewind_to_checkpoint(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
    ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        let Some(ckpt) = sess.turn_ckpt.take() else {
            return Ok(None);
        };
        assert!(
            ckpt.pos <= sess.committed.len(),
            "checkpoint past committed ({} > {})",
            ckpt.pos,
            sess.committed.len()
        );
        // accept_len 0: roll all the way back to the snapshot's own boundary. `rollback` sets
        // each full-attn len to its saved value, restores conv/ssm by D2D copy, and sets
        // cache.pos = snap.pos.
        sess.cache.rollback(e, &ckpt.snap, 0)?;
        debug_assert_eq!(sess.cache.pos, ckpt.pos, "rollback landed off the checkpoint");
        sess.scratch.set_len(e, ckpt.pos)?;
        sess.committed.truncate(ckpt.pos);
        sess.last_h = Some(ckpt.last_h);
        sess.next_pred = None;
        sess.pending_tok = None;
        Ok(Some(ckpt.pos))
    }

    /// Commit a carried pending bonus (see SpecSession::pending_tok): one T=1 trunk pass
    /// (its logits' argmax becomes next_pred) + the draft-KV fill at the carried anchor —
    /// byte-identical to the pre-carry session tail. Required before a non-empty-suffix
    /// prime, a sampled turn, or parking a session for pool reuse. No-op without a pending.
    pub fn spec_flush_pending(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(b) = sess.pending_tok.take() else {
            return Ok(());
        };
        let mtp = self.mtp.as_ref().expect("pending carry requires an MTP head");
        let n_embd = self.cfg.n_embd as usize;
        let (embd_qt, embd_rb) = self.embd.qt_and_row_bytes(n_embd);
        let embd_gpu = if spec_host_embd() {
            None
        } else {
            Some(
                self.embd_gpu
                    .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload")),
            )
        };
        let embd_dev = embd_gpu.map(|g| (g, embd_qt, embd_rb));
        let pos_b = sess.cache.pos;
        sess.scratch.set_len(e, pos_b)?;
        let (lg_b, hb) = self.decode_step_h(e, b, &mut sess.cache)?;
        sess.next_pred = Some(argmax(&lg_b) as u32);
        let anchor = sess
            .last_h
            .as_ref()
            .expect("pending carry requires last_h (the predecessor-row anchor)");
        self.mtp_kv_fill(e, mtp, &[b], anchor, pos_b, &mut sess.scratch, embd_dev)?;
        sess.last_h = Some(hb);
        sess.committed.push(b);
        Ok(())
    }

    /// One spec-decode turn on a live session. `suffix` = the NEW tokens only (turn N+1's user
    /// message rendered through the chat template continuation). Returns (new tokens emitted,
    /// drafted, accepted); session.committed grows by suffix + emitted.
    pub fn generate_spec_session(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
        suffix: &[u32],
        max_new: usize,
        k: usize,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        self.generate_spec_session_sampled(e, sess, suffix, max_new, k, None, None)
    }

    /// Serve-path sampled spec: routes the burst through the rejection-sampling verify with
    /// per-SESSION Philox continuity (sess.sctr/uctr). None = env-driven (CLI) or greedy.
    /// Filters (top-k/p/min-p) apply SYMMETRICALLY to draft q and verify p — distribution-exact
    /// for the filtered target (feat/filtered-spec).
    ///
    /// `on_commit` (sse-cadence, 2026-08-05): called with each newly-emitted slice of the
    /// output — once right after the prime's first token, then once per round commit — so a
    /// streaming caller can flush text at round cadence instead of once per burst. The slices
    /// are disjoint, in order, and concatenate to exactly the returned token vec. Emission-
    /// timing only: token bytes, session state, and exactness are untouched.
    ///
    /// The returned bool is a CONTINUE-VERDICT (admission yield, 2026-08-06): `false` ends
    /// the burst at the current round boundary, exactly as if `max_new` had been reached —
    /// the caller's scheduler regains control without waiting the burst out. Burst size is
    /// content-neutral (spec-levers battery), so an early exit moves WHEN the burst returns,
    /// never what tokens say. The slice may be EMPTY (a poll-only boundary — round-stream
    /// drains and the defensive tail flush can land with nothing new committed).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_spec_session_sampled(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
        suffix: &[u32],
        max_new: usize,
        k: usize,
        sampling: Option<SpecSampling>,
        on_commit: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        self.generate_spec_session_constrained(e, sess, suffix, max_new, k, sampling, None, on_commit)
    }

    /// `generate_spec_session_sampled` + GRAMMAR (constrained decoding, 2026-08-03): the
    /// hook truncates acceptance at the first grammar-illegal token AFTER the exactness
    /// verify (grammar is an extra rejection rule, ordering like the batched-verify twins)
    /// and replaces an illegal bonus with the MASKED argmax of the target's own verify
    /// column — token-identical to constrained plain greedy decode. GREEDY only (the
    /// worker routes sampled constrained to plain decode). Acceptance under tight grammars
    /// may drop (drafter is unconstrained); that is measured, not hidden.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_spec_session_constrained(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
        suffix: &[u32],
        max_new: usize,
        k: usize,
        sampling: Option<SpecSampling>,
        constraint: Option<&mut dyn SpecConstraint>,
        on_commit: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        if constraint.is_some() && sampling.is_some_and(|s| s.temp > 0.0) {
            return Err("constrained spec decode is greedy-only (worker routes sampled \
                        constrained to plain decode)".into());
        }
        // PENDING-CARRY entry flush: a carried bonus precedes any new suffix in the sequence,
        // so it must commit BEFORE the suffix primes; the sampled path doesn't carry (its
        // round-0 accept needs the commit pass's logits). Empty-suffix greedy bursts — the
        // serve continuation case — consume the carry in-loop with zero solo passes.
        if sess.pending_tok.is_some()
            && (!suffix.is_empty() || sampling.map_or(false, |s| s.temp > 0.0))
        {
            self.spec_flush_pending(e, sess)?;
        }
        let mtp_dense = self
            .mtp
            .as_ref()
            .map(|m| matches!(m.ffn, crate::hybrid::Ffn::Dense { .. }))
            .unwrap_or(false);
        let trunk_dense = self
            .layers
            .iter()
            .all(|l| matches!(l.ffn, crate::hybrid::Ffn::Dense { .. }));
        // FULL_PREC forces the EAGER draft: the graph capture would enclose cuBLASLt f32 GEMV
        // (the FloatBf16 else-branches) and a bf16_to_f32 dequant alloc — neither is stream-capture
        // safe. Eager rides matmul/matmul_decode_exact, which dequant FloatBf16 on use. (§item 2.)
        let graph_draft = std::env::var("MEMRA_SPEC_NOGRAPH").is_err()
            && !spec_host_embd()
            && mtp_dense
            && trunk_dense
            && k + 2 < 96
            && !crate::model::full_prec_enabled();
        let was_tracking = e.ctx().is_event_tracking();
        if graph_draft && was_tracking {
            unsafe {
                e.ctx().disable_event_tracking();
            }
        }
        let r = self.generate_spec_inner2(e, suffix, max_new, k, graph_draft, Some(sess), sampling, constraint, on_commit);
        if graph_draft && was_tracking {
            unsafe {
                e.ctx().enable_event_tracking();
            }
        }
        let (out, d, a) = r?;
        Ok((out, d, a))
    }

    pub fn generate_spec(
        &self,
        e: &Engine,
        prompt: &[u32],
        max_new: usize,
        k: usize,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        let mtp_dense = self
            .mtp
            .as_ref()
            .map(|m| matches!(m.ffn, crate::hybrid::Ffn::Dense { .. }))
            .unwrap_or(false);
        let trunk_dense = self
            .layers
            .iter()
            .all(|l| matches!(l.ffn, crate::hybrid::Ffn::Dense { .. }));
        // FULL_PREC forces eager (see generate_spec_session note): CUDA graph capture cannot
        // enclose cuBLASLt f32 GEMV or the bf16_to_f32 dequant alloc the FloatBf16 path needs.
        let graph_draft = std::env::var("MEMRA_SPEC_NOGRAPH").is_err()
            && !spec_host_embd()
            && mtp_dense
            && trunk_dense
            && k + 2 < 96
            && !crate::model::full_prec_enabled();
        if !graph_draft {
            return self.generate_spec_inner2(e, prompt, max_new, k, false, None, None, None, None);
        }
        let was_tracking = e.ctx().is_event_tracking();
        if was_tracking {
            unsafe {
                e.ctx().disable_event_tracking();
            }
        }
        let r = self.generate_spec_inner2(e, prompt, max_new, k, true, None, None, None, None);
        if was_tracking {
            unsafe {
                e.ctx().enable_event_tracking();
            }
        }
        r
    }

    fn generate_spec_inner2(
        &self,
        e: &Engine,
        prompt: &[u32],
        max_new: usize,
        k: usize,
        graph_draft: bool,
        mut sess: Option<&mut SpecSession>,
        sampling: Option<SpecSampling>,
        mut constraint: Option<&mut dyn SpecConstraint>,
        mut on_commit: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        assert!(k >= 1, "k must be >= 1");
        // sse-cadence flush cursor: everything in out[..flushed] has been handed to on_commit.
        let mut flushed = 0usize;
        // admission yield (2026-08-06): on_commit's continue-verdict; false = end the burst
        // at the next round boundary (same exit as max_new reached — the session tail runs).
        // Initialized by the unconditional post-prime flush below.
        let mut keep_going;
        let mtp = self
            .mtp
            .as_ref()
            .expect("generate_spec requires an MTP head (nextn_predict_layers>0)");
        let n_vocab = self.output.out_features();
        // FR-Spec: the draft head may be TRIMMED (fewer rows than n_vocab); the draft argmax runs
        // over the draft vocab and the winning index maps through d2t to a TARGET token id.
        // Everything downstream (verify/accept/commit) sees target ids only — exactness unchanged.
        let d_vocab = mtp
            .shared_head_head
            .as_ref()
            .unwrap_or(&self.output)
            .out_features();
        let n_embd = self.cfg.n_embd as usize;
        // SESSION MODE: reuse the live cache/scratch, prime only the suffix. `base` = tokens
        // already committed (their state is in the caches); 0 = fresh single-shot call.
        let session_mode = sess.is_some();
        let max_ctx = match sess.as_ref() {
            Some(s) => s.cache.max_ctx,
            None => prompt.len() + max_new + k + 8,
        };
        let mut own_cache;
        let mut own_scratch;
        let (
            cache,
            scratch,
            mut sess_tail,
            mut sess_draft_slot,
            mut sess_pending_slot,
            sess_ckpt_slot,
            mut sess_telem,
        ): (
            &mut Cache,
            &mut MtpScratch,
            Option<(
                &mut Vec<u32>,
                &mut Option<CudaSlice<f32>>,
                &mut Option<u32>,
                &mut u32,
                &mut u32,
            )>,
            Option<&mut Option<DraftGraphCtx>>,
            Option<&mut Option<u32>>,
            Option<&mut Option<SpecCheckpoint>>,
            Option<&mut SpecTelemetry>,
        ) = match sess.take() {
            Some(sr) => {
                let SpecSession {
                    cache,
                    scratch,
                    committed,
                    last_h,
                    next_pred,
                    sctr: s_sctr,
                    uctr: s_uctr,
                    draft_ctx,
                    pending_tok,
                    turn_ckpt,
                    telem,
                } = sr;
                (
                    cache,
                    scratch,
                    Some((committed, last_h, next_pred, s_sctr, s_uctr)),
                    Some(draft_ctx),
                    Some(pending_tok),
                    Some(turn_ckpt),
                    Some(telem),
                )
            }
            None => {
                // STAGE-OWNED KV (lane/pp2-spec 2026-08-06) — see `new_session`. Door shut =
                // `Cache::new` verbatim.
                own_cache = crate::pp::new_cache(e, &self.cfg, max_ctx)?;
                // Persistent scratch = max_ctx rows (~2KB/token quantized).
                own_scratch = MtpScratch::new(
                    e,
                    &self.cfg,
                    max_ctx,
                    self.mtp.as_ref().and_then(|m| m.geom.as_ref()),
                )?;
                (&mut own_cache, &mut own_scratch, None, None, None, None, None)
            }
        };
        let base = cache.pos;
        // PENDING-CARRY consume (2026-08-01): a carried bonus reaches here only on the
        // empty-suffix GREEDY continuation path (generate_spec_session_sampled flushed every
        // other case). It enters the round loop as round-0's pending — verify col 0 — exactly
        // like a mid-burst full-accept boundary: no init feed, no tail commit pass.
        let carried_pending: Option<u32> = sess_pending_slot.as_mut().and_then(|s| s.take());
        // PERSISTENT DRAFT KV (the only mode since 2026-07-08 — the legacy round-local scratch,
        // MEMRA_SPEC_KVLOCAL, measured -35 acceptance pts on the 27B p3 sweep and was removed;
        // acceptance-only — exactness is verify's job either way).
        // HIDDEN-PAIRING CONVENTION (DEFAULT = predecessor-row, 2026-07-04 — the 27B acceptance
        // unlock, +16pts): the MTP head is TRAINED on rows pairing token x_p with the trunk
        // hidden of its PREDECESSOR h_{p-1} (the reference engine's mtp_update shifts the target
        // hiddens right by one; its draft step 0 feeds (id_last, TRUE hidden of the row id_last
        // was sampled from)). memra's historical convention paired SAME-ROW (x_p, h_p) in the fill
        // and seeded chain step 0 through an extra MTP pass on a duplicated token (the
        // pseudo-seed) — measured 27B p2 K=3 acceptance 0.569 vs 0.731, p3 0.445 vs 0.63+, and
        // the chain steps j>=1 were already predecessor-shaped, so ONLY the fill + step-0 seed
        // move. The fill shifts by one and the chain seeds from the predecessor's true hidden
        // DIRECTLY (vh_seed / vx[j-1]) — the pseudo pass disappears (one MTP-block pass saved
        // per round on top of the acceptance win). Draft-quality-only: exactness stays the
        // verify's job either way. (The legacy same-row pairing seam, MEMRA_SPEC_HSAME, and its
        // pseudo-seed passes were removed 2026-07-08 — predecessor pairing won by +16 acc pts;
        // the legacy round-local scratch, MEMRA_SPEC_KVLOCAL, went with it.)
        // REPLAY-FREE PARTIAL ACCEPT (default, 2026-07-03): partial rounds keep the verify's own
        // bit-identical committed-prefix state (KV truncate + recur rebuild from the VerifyCkpt)
        // and leave the bonus PENDING — no duplicate trunk pass (profiled ~0.54 extra full weight
        // reads/round at long ctx). MEMRA_SPEC_REPLAY=1 restores the legacy rollback+replay (A/B
        // + fallback seam).
        let spec_replay = std::env::var("MEMRA_SPEC_REPLAY").is_ok();
        if constraint.is_some() && spec_replay {
            return Err("constrained spec decode does not support MEMRA_SPEC_REPLAY=1 \
                        (legacy replay commits an unmasked bonus)".into());
        }
        // TRUE-HIDDEN REFRESH (default in persistent-draft-KV mode): every round overwrites the
        // committed positions' scratch entries from the verify's exact hiddens (mtp_kv_fill batch)
        // instead of keeping chain-approximate entries. MEMRA_SPEC_NOREFRESH=1 = legacy (A/B seam).
        let refresh = std::env::var("MEMRA_SPEC_NOREFRESH").is_err();

        // prime: BATCHED cache prime (prime_cache — the measured #1 e2e gap: tokenwise primed at
        // ~102/38 tok/s vs the engine's ~2000-5900 tok/s batched prefill). prime_cache returns the
        // full pre-output_norm hidden stack [T, n_embd], which IS prompt_h (the persistent-draft-KV
        // mtp_kv_fill input) — no per-token collection needed. Prompts below PRIME_MIN_T, and
        // MEMRA_PRIME_TOKENWISE=1, and frozen Hy3 CPU/GPU expert splits take the tokenwise
        // decode_step_h loop. The latter avoids transient GPU staging of the spilled expert bank.
        // EMPTY-SUFFIX CONTINUATION (serve bursts): a session turn with NO new tokens resumes
        // generation exactly where the last turn stopped — no prime at all. The stashed
        // `next_pred` plays prime_logits' argmax role (it IS the argmax of the logits after
        // committed.last()); `last_h` seeds the predecessor pairing below. Fresh calls and
        // non-empty suffixes take the normal path.
        let continuation = prompt.is_empty();
        if continuation {
            assert!(session_mode, "empty prompt requires a session");
            assert!(
                sess_tail
                    .as_ref()
                    .map_or(false, |(c, lh, np, _, _)| !c.is_empty()
                        && lh.is_some()
                        && (np.is_some() || carried_pending.is_some())),
                "empty-suffix continuation needs a primed session (committed + last_h + next_pred|pending)"
            );
        }
        let mut prime_logits;
        let mut prompt_h: Option<CudaSlice<f32>> = None;
        let t_prime = std::time::Instant::now();
        let batched_prime = !continuation
            && prompt.len() >= crate::hybrid_forward::PRIME_MIN_T
            && std::env::var("MEMRA_PRIME_TOKENWISE").is_err()
            && !e.frozen_cpu_experts_prefer_tokenwise_prime();
        if continuation {
            prime_logits = Vec::new();
        } else if batched_prime {
            let (l, _h_seed, hiddens) = self.prime_cache(e, prompt, &mut *cache)?;
            prime_logits = l;
            prompt_h = Some(hiddens);
        } else {
            prime_logits = Vec::new();
            prompt_h = Some(e.uninit(prompt.len() * n_embd)?);
            for (i, &tok) in prompt.iter().enumerate() {
                let (l, h) = self.decode_step_h(e, tok, &mut *cache)?;
                if let Some(ph) = prompt_h.as_mut() {
                    e.copy_into(ph, i * n_embd, &h, n_embd)?;
                }
                prime_logits = l;
            }
        }
        e.stream().synchronize()?;
        // Harness timing contract (see crate::PRIME_NANOS): gen-only throughput without the
        // prime-subtraction hack.
        crate::PRIME_NANOS.store(
            t_prime.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        let (embd_qt, embd_rb) = self.embd.qt_and_row_bytes(n_embd);
        // Resident table is fastest when it fits. Large spill deployments can preserve that HBM
        // for expert-cache slots and gather only the exact rows needed by MTP/verify from host.
        let host_embd = spec_host_embd();
        let embd_gpu = if host_embd {
            None
        } else {
            Some(
                self.embd_gpu
                    .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload")),
            )
        };
        let embd_dev = embd_gpu.map(|g| (g, embd_qt, embd_rb));
        if host_embd {
            eprintln!(
                "[spec] host-row embedding: {} bytes kept off HBM",
                self.embd.raw.len()
            );
        }
        let mut out: Vec<u32> = Vec::with_capacity(max_new);
        let mut total_drafted = 0usize;
        let mut total_accepted = 0usize;

        // First generated token = argmax of the prompt's last logits (== greedy's first token).
        // Emit it, then FEED it to establish the loop invariant below.
        // PENDING-CARRY: the carried bonus was already emitted by the LAST burst — it becomes
        // last_token WITHOUT re-emission, and round 0 consumes it as pending (no init feed).
        // CONSTRAINED entry rules: the first emitted token is the MASKED argmax of the
        // prompt's last logits (plain constrained-greedy identity); a continuation without
        // a carried pending would emit an UNMASKED stashed next_pred — refused loudly (the
        // worker never resumes constrained sessions from the pool, so this cannot fire).
        if let Some(c) = constraint.as_deref_mut() {
            if continuation && carried_pending.is_none() {
                return Err("constrained spec continuation requires a carried pending \
                            (pool resume is unconstrained-only)".into());
            }
            if !continuation {
                c.mask_logits(&mut prime_logits)
                    .map_err(|e2| format!("constraint: {e2}"))?;
            }
        }
        let mut last_token = if let Some(b) = carried_pending {
            b
        } else if continuation {
            sess_tail.as_ref().unwrap().2.unwrap()
        } else {
            argmax(&prime_logits) as u32
        };
        if carried_pending.is_none() {
            out.push(last_token);
            // grammar advances with every emitted token (carried pendings were consumed
            // by the burst that emitted them).
            if let Some(c) = constraint.as_deref_mut() {
                c.consume(last_token).map_err(|e2| format!("constraint: {e2}"))?;
            }
        }
        if continuation {
            // draft-KV invariant: entries [0..base) are the session's exact fills; truncate any
            // overhang so the chain's first append lands at slot base (== committed.len()).
            scratch.set_len(e, base)?;
        }
        // sse-cadence: hand the caller every not-yet-flushed token (disjoint in-order slices
        // concatenating to the full `out`). Called after the prime's first token and after each
        // round commit — emission timing only, token bytes untouched. The slice may be EMPTY
        // (poll-only boundary: zero-round folds commit nothing new); returns the caller's
        // continue-verdict (admission yield, 2026-08-06) — false ends the burst at this round.
        fn flush_commit(
            cb: &mut Option<&mut dyn FnMut(&[u32]) -> bool>,
            out: &[u32],
            flushed: &mut usize,
        ) -> bool {
            if let Some(f) = cb.as_mut() {
                let keep = f(&out[*flushed..]);
                *flushed = out.len();
                keep
            } else {
                true
            }
        }
        keep_going = flush_commit(&mut on_commit, &out, &mut flushed);
        // INVARIANT at loop top: `last_token` is the most-recently-committed/emitted token, its
        // KV+recur state IS in `cache` (cache.pos = position right AFTER last_token), `last_pred`
        // is the greedy ARGMAX of the logits that predict the token FOLLOWING last_token, and
        // `h_seed` = last_token's pre-output_norm hidden. Establish it by feeding last_token once
        // (mirrors plain greedy). DEVICE-ARGMAX lever: the accept walk only ever consumes the
        // argmax of those logits — never the full vector — so a host u32 replaces the Vec<f32>.
        // --- SAMPLED SPEC (MEMRA_SPEC_TEMP>0, research/sampled-spec-impl-map.md): rejection-
        // sampling verify (Leviathan/Chen) — accept draft x at u < p(x)/q(x), resample from
        // norm(max(0,p-q)) on reject, bonus sampled from p on full accept. Counter-based Philox
        // everywhere (seed, event) -> reproducible. temp==0/unset = the greedy path, untouched.
        let sp = sampling.unwrap_or_else(|| SpecSampling {
            temp: std::env::var("MEMRA_SPEC_TEMP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            seed: std::env::var("MEMRA_SEED")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(42),
            top_k: std::env::var("MEMRA_TOP_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            top_p: std::env::var("MEMRA_TOP_P")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            min_p: std::env::var("MEMRA_MIN_P")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            penalty_last_n: std::env::var("MEMRA_PENALTY_LAST_N")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            penalty_repeat: std::env::var("MEMRA_PENALTY_REPEAT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            penalty_freq: std::env::var("MEMRA_PENALTY_FREQ")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            penalty_present: std::env::var("MEMRA_PENALTY_PRESENT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
        });
        let (sp_temp, sp_seed) = (sp.temp, sp.seed);
        let sampled = sp_temp > 0.0;
        // Trimmed heads: q lives on the trimmed vocab; accept gathers use the TRIMMED index and
        // the residual scatters q into target-id space (q=-inf off-trim — the head cannot propose
        // those, so their residual mass is p(x), correct by construction).
        let d2t_dev: Option<CudaSlice<u32>> = if sampled || crate::spec::spec_stream() {
            match &mtp.d2t {
                Some(map) => Some(e.htod_u32_v(map)?),
                None => None,
            }
        } else {
            None
        };
        let mut q_full_buf: Option<CudaSlice<f32>> = None;
        // Counters resume from the session (burst continuity: randomness must never repeat
        // across generate_spec_session calls); one-shot callers start at (0,0). Read through
        // sess_tail — `sess` was take()n into it above, so sess.as_ref() here is always None.
        let mut sctr: u32 = sess_tail.as_ref().map(|(_, _, _, s, _)| **s).unwrap_or(0);
        let mut uctr: u32 = sess_tail.as_ref().map(|(_, _, _, _, u)| **u).unwrap_or(0);
        // host Philox4x32-10 (mirrors spec_sample.cu; independent stream via ctr_lo tag)
        let host_u01 = |seed: u64, ctr: u32| -> f32 {
            let (m0, m1) = (0xD2511F53u32, 0xCD9E8D57u32);
            let (mut c0, mut c1, mut c2, mut c3) = (0xFFFF_FFFEu32, ctr, 0u32, 0u32);
            let (mut k0, mut k1) = ((seed & 0xFFFF_FFFF) as u32, (seed >> 32) as u32);
            for _ in 0..10 {
                let (h0, l0) = (((m0 as u64 * c0 as u64) >> 32) as u32, m0.wrapping_mul(c0));
                let (h1, l1) = (((m1 as u64 * c2 as u64) >> 32) as u32, m1.wrapping_mul(c2));
                let (n0, n1, n2, n3) = (h1 ^ c1 ^ k0, l1, h0 ^ c3 ^ k1, l0);
                c0 = n0;
                c1 = n1;
                c2 = n2;
                c3 = n3;
                k0 = k0.wrapping_add(0x9E3779B9);
                k1 = k1.wrapping_add(0xBB67AE85);
            }
            (c0 as f32 + 1.0) * (1.0 / 4294967296.0)
        };
        let mut draft_logits: Vec<CudaSlice<f32>> = Vec::new(); // retained head logits (q), per slot
        let mut draft_stats: Vec<(f32, f32, f32)> = Vec::new(); // (row_max, th_e, z_e) per slot
        let mut perturb_buf: Option<CudaSlice<f32>> = None; // gumbel scratch (max(n_vocab,d_vocab))
        let mut sample_tok = e.alloc_u32_zeroed(1)?; // residual/bonus sample out
        let mut col_buf: Option<CudaSlice<f32>> = None; // materialized verify column
                                                        // Penalties (v2.1): applied to COPIES of q rows and p columns symmetrically (exactness
                                                        // for the penalized+filtered target). History = generated tokens, host-tracked window.
        let pen_on = sampled
            && sp.penalty_last_n > 0
            && (sp.penalty_repeat != 1.0 || sp.penalty_freq != 0.0 || sp.penalty_present != 0.0);
        let mut pen_hist: Vec<u32> = if pen_on {
            prompt.iter().rev().take(64).rev().cloned().collect() // llama-parity: history spans prompt tail too
        } else {
            Vec::new()
        };
        let mut pen_hist_d: Option<CudaSlice<u32>> = None;
        let mut pcol_buf: Option<CudaSlice<f32>> = None; // penalized p-column scratch
        // MEMRA_SPEC_SETUP_TRACE=1 (diagnostics): per-call wall decomposition of the burst
        // SETUP + TAIL segments (the round loop's internals are MEMRA_SPEC_PHASE's job) —
        // built to pin the serve per-burst fixed cost (research/spec-serving-20260801).
        let setup_trace = std::env::var("MEMRA_SPEC_SETUP_TRACE").as_deref() == Ok("1");
        let t_ent = std::time::Instant::now();

        // SESSION-AFFINITY TURN CHECKPOINT (lane/session-affinity, 2026-08-05): capture the
        // PROMPT-END boundary state so a LATER turn can rewind here and re-prime only its own
        // delta instead of the whole conversation. See `SpecCheckpoint` for why this boundary is
        // the one that matters (a history-rewriting client mutates what the session GENERATED,
        // so the next turn's prompt agrees with this one up to exactly here).
        //
        // WHERE — AND WHY THIS EXACT LINE. Right after the trunk prime, BEFORE the init feed
        // (`decode_step_h(last_token)`) and before round 0: the last instant at which the caches
        // hold exactly `base + prompt.len()` rows and nothing generated.
        //
        // This was WRONG in the first cut of this lane: the capture sat after the draft-KV fill,
        // which is also after the init feed, so `cache.pos` was `base + prompt.len() + 1` — the
        // boundary included the FIRST GENERATED TOKEN. That token is the first thing inside the
        // `<think>` block the client strips, so every later turn's diff diverged exactly one
        // token below the checkpoint and affinity declined 100% of the time. Measured on the
        // owner regime: "history diverged at 12233 of checkpoint 12234". The off-by-one made the
        // whole mechanism inert while looking, from the outside, like a working
        // correctness-declines-safely path — hence the decline log carries the offsets.
        //
        // The full-attn planes are `len`-truncatable so the snapshot copies only the GDN conv/ssm
        // state (the reason a spec session could not rewind before). The draft scratch needs no
        // copy: rows below the boundary are rewritten by the next turn's own fill.
        //
        // WHEN: non-empty prime only. An empty-suffix continuation burst adds no prompt boundary
        // (its "prompt end" IS the previous checkpoint's, already held), so it keeps the existing
        // checkpoint rather than replacing it with a strictly worse one.
        //
        // FAILURE IS SILENT BY DESIGN: on a VRAM-tight rig the snapshot alloc can fail. That
        // costs the NEXT turn its rewind (it re-primes fully, today's behavior) and must never
        // fail the burst that is already running — so the error is swallowed, loud only under
        // MEMRA_DEBUG_SPEC.
        if let Some(slot) = sess_ckpt_slot {
            if !continuation {
                let pos = cache.pos;
                debug_assert_eq!(
                    pos,
                    base + prompt.len(),
                    "turn checkpoint must sit at the prompt end, before the init feed"
                );
                let anchor: Result<CudaSlice<f32>, Box<dyn std::error::Error>> =
                    if let Some(ph) = &prompt_h {
                        // hidden of the LAST primed row = the predecessor anchor at this
                        // boundary (exactly what a fresh prime of committed[..pos] leaves in
                        // last_h, and what the next prime's fill reads for its first row).
                        let np = prompt.len();
                        e.uninit(n_embd).and_then(|mut a| {
                            e.copy_view_into(
                                &mut a,
                                0,
                                &ph.slice((np - 1) * n_embd..np * n_embd),
                                n_embd,
                            )?;
                            Ok(a)
                        })
                    } else {
                        Err("no prompt hiddens".into())
                    };
                match (cache.snapshot(e), anchor) {
                    (Ok(snap), Ok(last_h)) => {
                        *slot = Some(SpecCheckpoint { snap, pos, last_h });
                    }
                    (s, a) => {
                        *slot = None; // a stale checkpoint would rewind to the WRONG boundary
                        if std::env::var("MEMRA_DEBUG_SPEC").is_ok() {
                            let err = s.err().map(|e| e.to_string())
                                .or_else(|| a.err().map(|e| e.to_string()))
                                .unwrap_or_default();
                            eprintln!("[spec] turn checkpoint skipped ({err}); \
                                       next turn re-primes in full");
                        }
                    }
                }
            }
        }
        // INIT FEED — skipped on a pending carry: last_token (the carried bonus) is NOT in the
        // caches and must NOT be fed solo; round 0's batched verify commits it as col 0. Its
        // seed/anchor hidden is the carried last_h (copied below); last_pred is dead in the
        // pending path (t_pred reads verify col 0 — the accept walk overwrites it).
        let mut last_pred = 0u32;
        let mut last_col_logits: Option<CudaSlice<f32>> = None;
        // CONSTRAINED: the init feed's logits back the (n_acc==0, base==0) masked-argmax
        // recompute in the grammar-truncation walk — retained host-side, round 0 only.
        let mut init_logits_host: Option<Vec<f32>> = None;
        let h_seed0: CudaSlice<f32> = if carried_pending.is_none() {
            let (init_logits, h) = self.decode_step_h(e, last_token, &mut *cache)?;
            last_pred = argmax(&init_logits) as u32;
            if constraint.is_some() {
                init_logits_host = Some(init_logits.clone());
            }
            // sampled mode: p-distribution after last_token, for the j==0/base==0 accept test.
            if sampled {
                last_col_logits = Some(e.htod(&init_logits)?);
            }
            h
        } else {
            // predecessor-row anchor: hidden of the last COMMITTED row (the carry contract).
            let lh = sess_tail
                .as_ref()
                .unwrap()
                .1
                .as_ref()
                .expect("pending carry requires last_h");
            e.clone_dtod(lh)?
        };
        let t_init = t_ent.elapsed();
        let mut last_col_stats: Option<(f32, f32, f32)> = None;
        // PERSISTENT h_seed buffer (allocated BEFORE any graph capture so no captured scratch can
        // alias it): every path that updates the round seed copies INTO it — no per-round allocs,
        // stable pointer for the graph-draft round-start copy.
        let mut h_seed_buf = e.clone_dtod(&h_seed0)?;
        // Predecessor-pairing trackers: `fill_prev` = trunk hidden AT the last COMMITTED row (the
        // predecessor of the next verify's col 0 — the reference's carried pending-h analogue;
        // also the predecessor-row hidden for the round-0 legacy-replay seed). At round 0 that
        // row is last_token's own (h_seed0). The chain step-0 seed under the pairing default =
        // hidden of the row BEFORE last_token = the prompt's last row at round 0 (h_seed_buf
        // overwritten below).
        let mut fill_prev = e.clone_dtod(&h_seed0)?;
        {
            if let Some(ph) = &prompt_h {
                let np = prompt.len();
                e.copy_view_into(
                    &mut h_seed_buf,
                    0,
                    &ph.slice((np - 1) * n_embd..np * n_embd),
                    n_embd,
                )?;
            } else if continuation {
                if let Some((_, lh, _, _, _)) = sess_tail.as_ref() {
                    if let Some(lh) = lh.as_ref() {
                        e.copy_into(&mut h_seed_buf, 0, lh, n_embd)?;
                    }
                }
            }
        }
        // Persistent device prediction slots for the accept walk (max k+1 verify columns).
        let mut preds_d = e.alloc_u32_zeroed(k + 2)?;

        let debug_spec = std::env::var("MEMRA_DEBUG_SPEC").is_ok();
        // MEMRA_SPEC_STATS=1: per-slot accept histogram + draft-length histogram, printed once at
        // the end. Metric normalization vs the reference engine: BOTH engines count
        // accepted/drafted where the chain stopped at p-min and the sub-threshold token is
        // discarded uncounted — per-slot decay + chain-length mix are the extra dimensions.
        let spec_stats = std::env::var("MEMRA_SPEC_STATS").is_ok();
        let mut st_drafted = vec![0usize; k];
        let mut st_accepted = vec![0usize; k];
        let mut st_len_hist = vec![0usize; k + 1];
        let mut st_full = 0usize;
        // P-MIN CONFIDENCE GATE (MEMRA_SPEC_PMIN, the serve script's --spec-draft-p-min mechanism):
        // stop the draft chain early when the head's softmax confidence in its own pick drops
        // below p_min. Hoisted above the loop: the graph capture bakes the prob kernels iff on.
        static PMIN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        let p_min = *PMIN.get_or_init(|| {
            std::env::var("MEMRA_SPEC_PMIN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0)
        });
        // ZERO-DRAFT ROUNDS (MEMRA_SPEC_PMIN0=1, vendored from llama.cpp's draft gating): let the
        // p-min gate apply at j==0 too, so a low-confidence round drafts NOTHING and the verify
        // batch is just the pending bonus (m=1 = a plain decode step). llama's 35B win rides
        // exactly this — draft acceptance 76% at mean len 2.5 because unpredictable stretches
        // never pay draft+verify overhead. Only legal when a pending bonus exists (an empty
        // verify batch is not); the j==0 exemption stays for pending-less rounds.
        let pmin0 = std::env::var("MEMRA_SPEC_PMIN0")
            .map(|v| v == "1")
            .unwrap_or(false);

        // --- GRAPH DRAFT setup: persistent I/O buffers + ONE capture (2 warmups inside). The
        // warmups mutate scratch len_d / pos / tok / seed — all reset at every round start, so the
        // only restore needed is the scratch counter. Capture failure (e.g. a non-capturable
        // cuBLAS path in an exotic head) falls back to the eager draft chain.
        // PER-SESSION PERSISTENCE (2026-08-01): session calls reuse the DraftGraphCtx parked on
        // the SpecSession — the capture (2 warmup head forwards + instantiate) ran ONCE at the
        // session's first burst, not per burst (measured ~16ms/burst fixed cost on H100 q27,
        // research/spec-serving-20260801). Reuse is pointer-exact: the graph bakes the session's
        // own scratch KV (never realloc'd), the model's resident embedding, the OnceLock p_min,
        // and the g_* buffers carried in the ctx — replay dispatch is identical to a fresh
        // capture, so draft tokens are bit-identical (drafts never decide exactness anyway; the
        // verify arbitrates). Single-shot calls (sess=None) build a fresh ctx and drop it.
        let mut dctx: DraftGraphCtx = match sess_draft_slot.as_mut().and_then(|s| s.take()) {
            Some(c) => c,
            None => DraftGraphCtx::new(e, n_embd, if sampled { d_vocab } else { 1 })?,
        };
        // A session that ran greedy bursts first sized g_q/g_perturb at 1; a sampled resume
        // needs d_vocab. Realloc is legal exactly while graph_s is None (nothing baked them).
        if sampled && dctx.g_q.len() < d_vocab {
            dctx.g_q = e.zeros(d_vocab)?;
            dctx.g_perturb = e.zeros(d_vocab)?;
        }
        // DRAFT-SIDE GRAMMAR MASK (lane/draft-mask, 2026-08-04): the drafter samples the
        // grammar's legal set, so proposals are legal BY CONSTRUCTION and the verify-side
        // truncation (the correctness backstop) stops cutting every tight-schema round.
        // The mask is one node inside the captured draft chain — presence is a CAPTURE-TIME
        // shape, so a parked graph of the other shape is dropped and recaptured.
        let dmask_on = constraint.as_deref().is_some_and(|c| c.draft_mask_enabled());
        let dmask_words = if dmask_on { d_vocab.div_ceil(32) } else { 0 };
        if dmask_on && dctx.g_dmask.len() < dmask_words {
            dctx.g_dmask = e.alloc_u32_zeroed(dmask_words)?;
            dctx.graph = None; // the old capture baked the old (or no) mask pointer
            dctx.failed.clear_greedy();
            dctx.keeper.clear();
        }
        if dctx.graph.is_some() && dctx.graph_masked != dmask_on {
            dctx.graph = None;
            dctx.failed.clear_greedy();
            dctx.keeper.clear();
        }
        if graph_draft && !sampled && dctx.graph.is_none() && !dctx.failed.greedy_failed() {
            let DraftGraphCtx { g_tok, g_pos, g_seed, g_p, g_dmask, .. } = &mut dctx;
            // capture-time contents: ALL-ONES (ban nothing). A replay only ever runs after the
            // host uploads the position's real words, so the warmups stay grammar-free.
            if dmask_on {
                e.htod_u32_into(g_dmask, &vec![u32::MAX; dmask_words])?;
            }
            let g_dmask_ro: &CudaSlice<u32> = &*g_dmask;
            // CAPTURE-RETAIN (#68 fix): the warmup transients' pool addresses are baked into the
            // captured graph; the keeper pins them for the graph's lifetime. capture_graph (non-
            // retained) freed them at exit — safe for one-shot generate_spec (nothing else touches
            // the pool between replays) but WRONG for sessions: burst-boundary prime/fill/commit
            // passes (and, in serve, other sessions) recycle those addresses and the replay then
            // clobbers live buffers — the ST serve-spec corruption (research/serve-st-20260803).
            let cap_res = e.capture_graph_retained(|e| {
                self.mtp_head_forward_cap(
                    e,
                    mtp,
                    g_tok,
                    g_pos,
                    g_seed,
                    g_p,
                    &mut *scratch,
                    p_min > 0.0,
                    true,
                    embd_gpu.expect("graph draft requires resident embedding"),
                    embd_qt,
                    embd_rb,
                    d_vocab,
                    None,
                    None,
                    if dmask_on { Some((g_dmask_ro, dmask_words)) } else { None },
                )
            });
            match cap_res {
                Ok((g, keep)) => {
                    scratch.set_len(e, base)?;
                    dctx.graph = Some(g);
                    dctx.graph_masked = dmask_on;
                    dctx.keeper = keep;
                }
                Err(err) => {
                    scratch.set_len(e, base)?;
                    // LOUD flip (audit Q2): a dropped draft graph is a coverage loss, never
                    // silent. Once per flip — mark returns None on an already-failed ctx.
                    if let Some(line) = dctx.failed.mark_greedy(&err.to_string()) {
                        eprintln!("{line}");
                    }
                }
            }
        }
        // --- SAMPLED GRAPH DRAFT setup (step 3 of the sampled-spec arc): a SECOND capture, own
        // graph object, built only when sampled && graph-eligible — the greedy capture above is
        // untouched (and skipped when sampled: its graph would never be launched). Same head
        // forward, but the in-graph argmax reads GUMBEL-PERTURBED logits; the Philox event
        // counter lives in the persistent device g_ctr (bumped in-graph, host-seeded from sctr
        // once per round); the raw head logits land in the persistent g_q for the host's
        // per-replay async D2D into the round's q slot (q_slots, K x d_vocab, allocated once).
        // seed/temp are capture-time constants — baked into graph_s, so a pool-resumed request
        // with a different (seed, temp, k) drops the parked sampled graph and recaptures.
        // COST OF THE FRESH-SEED SERVE DEFAULT (dogfood F4, 2026-08-04): omitting `seed` on a
        // serve request now draws fresh per-request entropy (it used to default to a pinned 0),
        // so a seed-omitting request that RESUMES a parked spec session finds an s_key baked
        // with the PREVIOUS request's seed and pays one recapture. Bounded, and it does not
        // reopen the ~16ms/burst regression the persistent ctx exists to fix: a session's seed
        // is fixed for its whole lifetime (worker.rs reads s.sampler.seed() per burst), so
        // this compare misses at most ONCE per resumed request — the first burst recaptures
        // and every later burst in that request replays. A client that wants the parked graph
        // AND reproducibility supplies an explicit `seed`, honored exactly, which keeps s_key
        // stable across its whole conversation.
        // COMPOSITION RULE (fspec x gsd merge): the in-graph chain samples from the RAW
        // softmax — it can hold neither per-row filter stats nor the varying penalty history.
        // The sampled graph therefore engages only in the PURE-TEMP regime; filters/penalties
        // force the eager draft (which computes stats/penalties per row).
        let pure_temp = sp.top_k == 0 && sp.top_p >= 1.0 && sp.min_p <= 0.0 && !pen_on;
        let s_key = (sp_seed, sp_temp.to_bits(), k);
        if sampled && dctx.s_key.is_some_and(|old| old != s_key) {
            dctx.graph_s = None;
            dctx.failed.clear_sampled();
            dctx.s_key = None;
            dctx.q_slots.clear();
            dctx.keeper_s.clear();
        }
        if graph_draft && sampled && pure_temp && dctx.graph_s.is_none()
            && !dctx.failed.sampled_failed()
        {
            let DraftGraphCtx { g_tok, g_pos, g_seed, g_p, g_ctr, g_perturb, g_q, .. } = &mut dctx;
            // CAPTURE-RETAIN (#68 fix): same keeper contract as the greedy capture above.
            let cap_res = e.capture_graph_retained(|e| {
                self.mtp_head_forward_cap(
                    e,
                    mtp,
                    g_tok,
                    g_pos,
                    g_seed,
                    g_p,
                    &mut *scratch,
                    p_min > 0.0,
                    true,
                    embd_gpu.expect("graph draft requires resident embedding"),
                    embd_qt,
                    embd_rb,
                    d_vocab,
                    Some((g_ctr, g_perturb, g_q, sp_seed, sp_temp)),
                    None,
                    None, // constrained spec is greedy-only — sampled never carries a hook
                )
            });
            match cap_res {
                Ok((g, keep)) => {
                    scratch.set_len(e, base)?;
                    for _ in 0..k {
                        dctx.q_slots.push(e.zeros(d_vocab)?);
                    }
                    dctx.graph_s = Some(g);
                    dctx.s_key = Some(s_key);
                    dctx.keeper_s = keep;
                }
                Err(err) => {
                    scratch.set_len(e, base)?;
                    // LOUD flip (audit Q2): same contract as the greedy capture above.
                    if let Some(line) = dctx.failed.mark_sampled(&err.to_string()) {
                        eprintln!("{line}");
                    }
                }
            }
        }
        let t_cap = t_ent.elapsed();
        // PERSISTENT DRAFT KV: fill the MTP block's K/V for every prompt position from the exact
        // trunk hiddens collected during prime — ONE batched K/V-only pass (overwrites any
        // capture-warmup garbage; capture left len at 0). last_token (the init feed) needs no
        // fill: the first chain step processes it and appends its entry at slot prompt.len().
        if let Some(ph) = &prompt_h {
            // SESSION: rows [0..base) are the previous turns' exact fills (refresh overwrote them
            // with true verify hiddens) — truncate any draft overhang, fill ONLY the suffix at
            // global positions [base..base+tp). Fresh call: base==0, identical to before.
            scratch.set_len(e, base)?;
            // CHUNKED FILL (long-ctx OOM fix, 2026-07-05): mtp_kv_fill's transients scale with its
            // T (concat = T*2*n_embd*4B — 1.5GB at 40k) and its concat loop is 2*T launches. The
            // fill is a pure sequential append, so chunking is exact: each chunk appends its rows
            // at pos0=base+start with the identical per-row math. Same knob as the trunk prime.
            let fill_chunk: usize = std::env::var("MEMRA_PRIME_CHUNK")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096);
            let tp = prompt.len();
            let fill_chunk = if fill_chunk == 0 { tp } else { fill_chunk };
            let mut start = 0usize;
            while start < tp {
                let end = (start + fill_chunk).min(tp);
                let tc = end - start;
                {
                    // PREDECESSOR pairing: row i gets h[i-1]; global row 0 a zeros row (the
                    // reference engine's initial pending-h is zeroed too); a session turn's row 0
                    // gets the PREVIOUS turn's last committed hidden (sess.last_h). Per chunk:
                    // rows start..end read h[start-1..end-1] — one dtod into a chunk buffer.
                    let mut phs = e.zeros(tc * n_embd)?;
                    let (src_lo, dst_off) = if start == 0 {
                        (0, n_embd)
                    } else {
                        ((start - 1) * n_embd, 0)
                    };
                    let n_copy = if start == 0 {
                        (tc - 1) * n_embd
                    } else {
                        tc * n_embd
                    };
                    if start == 0 {
                        if let Some((_, lh, _, _, _)) = sess_tail.as_ref() {
                            if let Some(lh) = lh.as_ref() {
                                e.copy_into(&mut phs, 0, lh, n_embd)?;
                            }
                        }
                    }
                    if n_copy > 0 {
                        e.copy_view_into(
                            &mut phs,
                            dst_off,
                            &ph.slice(src_lo..src_lo + n_copy),
                            n_copy,
                        )?;
                    }
                    self.mtp_kv_fill(
                        e,
                        mtp,
                        &prompt[start..end],
                        &phs,
                        base + start,
                        &mut *scratch,
                        embd_dev,
                    )?;
                }
                start = end;
            }
        }
        // MEMRA_PROFILE_SPEC=2: profiler capture starts HERE — after the prime, so an
        // `nsys -c cudaProfilerApi` capture contains ONLY the round loop (draft/verify/commit).
        // (=1 brackets the whole call in run_spec.rs, prime included.)
        if std::env::var("MEMRA_PROFILE_SPEC").as_deref() == Ok("2") {
            unsafe extern "C" {
                fn cudaProfilerStart() -> i32;
            }
            unsafe {
                cudaProfilerStart();
            }
        }
        // ROUND-STREAM stage (c) 4 (MEMRA_SPEC_STREAM=1, experimental): pre-issued M-round
        // bursts with ZERO per-round host readbacks — the accept/seed/rollback/ring kernels
        // consume each other's device outputs; the host drains the ring every M rounds. v1
        // constraints: greedy, !spec_replay, single-shot, batched-linear layers, no refresh
        // fills (acceptance effect A/B-arbitrated), enters from round 1 (pending guaranteed).
        // NOTE: not gated on the caller's graph_draft (its trunk_dense conjunct turns the 35B
        // MoE off) — the stream capture encloses ONLY the dense MTP head; the head-dense /
        // full-prec / k gates are re-derived here and a failed capture degrades to stream-off.
        let stream_on = crate::spec::spec_stream()
            && !sampled
            && !spec_replay
            && constraint.is_none()
            && !session_mode
            && embd_gpu.is_some()
            && !crate::model::full_prec_enabled()
            && k + 2 < 96;
        let mut stream_graph: Option<cudarc::driver::CudaGraph> = None;
        let mut g_tokp2k = e.alloc_u32_zeroed(2 * k.max(1))?;
        if stream_on {
            let cap = e.capture_graph(|e| {
                for j in 0..k.max(1) {
                    self.mtp_head_forward_cap(
                        e,
                        mtp,
                        &mut dctx.g_tok,
                        &mut dctx.g_pos,
                        &mut dctx.g_seed,
                        &mut dctx.g_p,
                        &mut *scratch,
                        true,
                        true,
                        embd_gpu.expect("round stream requires resident embedding"),
                        embd_qt,
                        embd_rb,
                        d_vocab,
                        None,
                        Some((&mut g_tokp2k, j, d2t_dev.as_ref())),
                        None, // round-stream requires constraint.is_none() (see stream_on)
                    )?;
                }
                Ok(())
            });
            match cap {
                Ok(g) => {
                    scratch.set_len(e, 0)?;
                    stream_graph = Some(g);
                }
                Err(err) => {
                    scratch.set_len(e, 0)?;
                    if debug_spec {
                        eprintln!("[spec] stream-graph capture failed ({err}); stream off");
                    }
                }
            }
        }
        let stream_active = stream_on && stream_graph.is_some();
        if debug_spec {
            eprintln!("[spec] stream_on={stream_on} env={} samp={sampled} dg={} captured={} active={stream_active} session={session_mode} replay={spec_replay}",
                      crate::spec::spec_stream(), dctx.graph.is_some(), stream_graph.is_some());
        }
        let t_v_s = k + 1;
        // ROUND-STREAM buffers + ptr tables now live in the model-generic round_stream
        // module (extracted 2026-07-12; the gemma burst reuses them).
        let sb = crate::round_stream::StreamBufs::new(e, k, crate::spec::spec_stream_m())?;
        let crate::round_stream::StreamBufs {
            mut vtok_d,
            mut brk_d,
            mut pend_d,
            last_pred_d,
            mut pos_ctr,
            mut pos_start_d,
            mut ring_d,
            acc_d: mut stream_acc,
            m_rounds,
            k: _,
        } = sb;
        let stream_ptrs: Option<CudaSlice<u64>> = if stream_active {
            Some(crate::round_stream::kv_len_ptr_table(
                e,
                cache,
                Some(&pos_ctr),
            )?)
        } else {
            None
        };

        let t_fill = t_ent.elapsed();
        let mut round = 0usize;
        // ADAPTIVE DRAFT LENGTH (MEMRA_SPEC_ADAPT=1, opt-in — the gemma_spec accepted-run law,
        // ported 2026-08-01): next round's draft depth = last round's accepted run + 1, clamped
        // to [floor(pos), k_cap] — a miss shrinks the next draft to the miss point + 1,
        // full-accept streaks re-deepen one step per round. NOT the 2026-07-07 acceptance-EMA
        // (that arm measured an HONEST LOSS to static per-class optima — 115.0/85.8/73.4 vs
        // 121.6/92.7/75.6, EMA lag — and was removed 2026-07-08; rig5090.jsonl has the record).
        // The gemma law has no lag class: it reacts within one round, and was worth +7-20% on
        // the gemma cells at unchanged exactness (2026-07-10 flip; floor sweep 2026-07-25;
        // position key 2026-07-26). Signal = n_acc from the round's EXISTING accept readback —
        // zero new syncs; the draft graph is a SINGLE-STEP capture replayed per drafted token,
        // so a per-round depth needs no re-capture (unlike gemma's whole-chain graphs). qwen's
        // in-round p-min cut already shortens chains mid-round, so gemma's one-round-late p-min
        // fold into kc is unnecessary here — the accepted-run law sees the cut via n_acc.
        // Exactness is the verify's job at ANY depth (same contract as p-min variable rounds).
        // DEFAULT OFF on the qwen path until its cells gate a flip (gemma's is default-on).
        // MEASURED 2026-08-01 (H100 GPU-3, interleaved x3, NGEN=256, same-invocation plain
        // denominators; research/qwen-adaptive-k-20260801/): REFUTED on the tuned qwen configs.
        // q27 K=3+HPOST+PMIN=0.3: short +0.8% (noise; law ~idles, len_hist identical), board
        // -1.9%, agentic -0.5%; board PMIN=0 -2.1% (not p-min shadowing — the law itself);
        // floor=1 -2.8% (gemma's floor-collapse, reproduced). q35 K=2 board: -6.4% (52/136
        // rounds shrink to depth 1; no depth to reclaim at K=2). The gemma direction DOES
        // appear at untuned depth-K — q27 K=6 floor=4 +1.5% over fixed K=6 — but stays -3.7%
        // below fixed K=3: same verdict class as the retired EMA arm (honest loss to static
        // per-class optima). Acceptance-rate rises under the law while tokens/round falls —
        // it buys accept-% by adding rounds, and a round's fixed draft+verify cost wins.
        // K=1..8 self-consistency PASS both models with the law ON (exactness held).
        let adapt = std::env::var("MEMRA_SPEC_ADAPT").as_deref() == Ok("1");
        // floor: per-model default keyed on n_embd (gemma's tiering — models with an expensive
        // verify keep deep drafts after a miss); MEMRA_SPEC_ADAPT_FLOOR pins it everywhere.
        let adapt_floor_env: Option<usize> = std::env::var("MEMRA_SPEC_ADAPT_FLOOR")
            .ok()
            .and_then(|v| v.parse().ok());
        let adapt_floor_default: usize = if self.cfg.n_embd as usize >= 3500 {
            4
        } else if self.cfg.n_embd as usize >= 2500 {
            2
        } else {
            1
        };
        let adapt_floor: usize = adapt_floor_env.unwrap_or(adapt_floor_default);
        // position key: past floor_ctx a HIGH floor (>=4) relaxes to 1 — forced-deep drafts
        // turn net-negative at depth (gemma 31B d1736 evidence); MEMRA_SPEC_FLOOR_CTX moves
        // the boundary, an explicit MEMRA_SPEC_ADAPT_FLOOR pins the floor everywhere.
        let floor_ctx: usize = std::env::var("MEMRA_SPEC_FLOOR_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        let floor_at = |pos: usize| -> usize {
            if adapt_floor_env.is_some() || pos < floor_ctx {
                adapt_floor
            } else if adapt_floor >= 4 {
                1
            } else {
                adapt_floor
            }
        };
        // cap: MEMRA_SPEC_CAPMAX (gemma semantics, default 7). Binds only under adapt — the
        // fixed-K default path is untouched by this whole block.
        let cap_max: usize = std::env::var("MEMRA_SPEC_CAPMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7);
        let k_cap = k.min(cap_max).max(1);
        let mut kc = k_cap;
        // PERSISTENT snapshot buffers: allocate ONCE, refresh in place each round (was 2 fresh
        // D2D clones per linear layer per round = 48 allocs + ~50MB of pool churn per round).
        let mut snap = cache.snapshot(e)?;
        // ROUND-STREAM stage (b) 3a: device table of per-layer kvl.len_d pointers (stable — the
        // cache never reallocates len_d; see cache.rs "stable pointer" note). 0 = no KV layer.
        let kv_len_ptrs: Option<CudaSlice<u64>> = if spec_devacc() && !spec_replay {
            Some(crate::round_stream::kv_len_ptr_table(e, cache, None)?)
        } else {
            None
        };
        // BONUS FOLD (2026-07-04): after a FULL accept the bonus token is NOT committed with a
        // separate T=1 trunk pass (a full weight read per round). It stays PENDING and rides as
        // column 0 of the NEXT round's verify batch. Under predecessor pairing the next chain
        // seeds from the bonus's predecessor's TRUE verify hidden (free — no extra
        // pass of any kind). Verify still
        // checks every emitted token against the target -> exactness holds by construction; only
        // DRAFT QUALITY can shift, which the acceptance numbers arbitrate.
        // bonus emitted but not yet committed to cache. A carried pending (see SpecSession::
        // pending_tok) enters round 0 directly — the burst boundary becomes a plain round edge.
        let mut pending: Option<u32> = carried_pending;
                                             // MEMRA_SPEC_PHASE=1: per-round wall decomposition (draft / verify / accept+commit) —
                                             // no tracing, no extra syncs (each phase is naturally sync-bounded: draft readbacks,
                                             // the verify accept readback). Printed once at loop end via spec-stats.
        let phase_on = std::env::var("MEMRA_SPEC_PHASE").as_deref() == Ok("1");
        // DRAFT-MASK receipt (lane/draft-mask): speculative-clone wall + rounds, printed with
        // spec-stats. The clone is the one cost the design adds per round — measured, not assumed.
        let (mut dm_clone_ns, mut dm_rounds) = (0u128, 0usize);
        // grammar-truncation counters: how many rounds the verify-side cut fired and how many
        // already-verified tokens it threw away. THIS is the quantity draft masking targets.
        let (mut dm_cuts, mut dm_cut_tokens) = (0usize, 0usize);
        let (mut ph_draft, mut ph_verify, mut ph_rest) = (0f64, 0f64, 0f64);
        let mut ph_wait = 0f64;
        let mut ph_t = std::time::Instant::now();
        let mut ph_mark = |acc: &mut f64, on: bool| {
            if on {
                let now = std::time::Instant::now();
                *acc += (now - ph_t).as_secs_f64();
                ph_t = now;
            }
        };
        while keep_going && out.len() < max_new {
            // ROUND-STREAM BURST: from round 1 (pending guaranteed by every non-replay arm),
            // issue M rounds with zero readbacks, then drain the ring + reconcile mirrors.
            if let (true, Some(sg), Some(ptrs)) = (
                stream_active && round >= 1 && pending.is_some(),
                &stream_graph,
                &stream_ptrs,
            ) {
                if debug_spec {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        eprintln!("[memra] ROUND-STREAM burst engaged (M={m_rounds} k={k})")
                    });
                }
                e.set_i32_one(&mut pos_ctr, cache.pos as i32)?;
                e.set_u32_one(&mut pend_d, pending.unwrap())?;
                e.set_u32_one(&mut ring_d, 0)?; // ring count = 0 (writes element 0)
                for _mi in 0..m_rounds {
                    e.i32_copy_add(&pos_ctr, &mut pos_start_d, 0)?;
                    cache.snapshot_into(e, &mut snap)?; // device D2Ds, stream-ordered
                    e.i32_copy_add(&pos_ctr, &mut scratch.kv.len_d, 0)?; // draft-KV rollback
                    e.i32_copy_add(&pos_ctr, &mut dctx.g_pos, 1)?; // rope pos = pos + base
                    e.u32_copy(&pend_d, &mut dctx.g_tok)?;
                    e.copy_into(&mut dctx.g_seed, 0, &h_seed_buf, n_embd)?;
                    sg.launch()?;
                    e.spec_assemble_verify(
                        &g_tokp2k,
                        &pend_d,
                        d2t_dev.as_ref(),
                        &mut vtok_d,
                        &mut brk_d,
                        p_min,
                        k,
                        pmin0,
                    )?;
                    let mut ck = VerifyCkpt::new(self.layers.len());
                    let dummy = vec![0u32; t_v_s];
                    let (tl_d, vx) = self.decode_step_t_core_stream(
                        e,
                        &dummy,
                        0,
                        &mut *cache,
                        embd_dev,
                        Some(&mut ck),
                        Some((&vtok_d, &pos_ctr)),
                    )?;
                    for j in 0..t_v_s {
                        e.argmax_token_device_col(&tl_d, j, n_vocab, &mut preds_d, j)?;
                    }
                    e.spec_accept_greedy_dc(
                        &preds_d,
                        &vtok_d,
                        &last_pred_d,
                        &brk_d,
                        &mut stream_acc,
                    )?;
                    e.spec_seed_gather(&vx, &fill_prev, &stream_acc, &mut h_seed_buf, 1, n_embd)?;
                    e.copy_into(&mut fill_prev, 0, &h_seed_buf, n_embd)?;
                    self.commit_verified_prefix_stream(
                        e,
                        &mut *cache,
                        &snap,
                        &ck,
                        &stream_acc,
                        1,
                        t_v_s,
                    )?;
                    e.spec_rollback_stream(
                        ptrs,
                        &pos_start_d,
                        &stream_acc,
                        1,
                        self.layers.len() + 1,
                    )?;
                    e.spec_ring_commit(&vtok_d, &stream_acc, &brk_d, &mut ring_d, &mut pend_d)?;
                }
                e.stream().synchronize()?;
                let ring_h = e.dtoh_u32(&ring_d)?;
                let cnt = ring_h[0] as usize;
                for i in 0..cnt {
                    if out.len() < max_new {
                        out.push(ring_h[1 + i]);
                    }
                }
                let pos_h = e.dtoh_i32(&pos_ctr)?[0] as usize;
                for il in 0..self.layers.len() {
                    if let Some(kvl) = cache.kv[il].as_mut() {
                        kvl.len = pos_h;
                    }
                }
                cache.pos = pos_h;
                scratch.kv.len = pos_h;
                pending = Some(ring_h[cnt]); // last drained token = the live bonus
                last_token = ring_h[cnt];
                total_drafted += k * m_rounds; // upper bound (p-min breaks uncounted)
                total_accepted += cnt.saturating_sub(m_rounds);
                if let Some(t) = sess_telem.as_deref_mut() {
                    // totals only — the burst's per-round accept counts stayed on device
                    // (that is the point of the round-stream arm). pos_* untouched.
                    t.rounds += m_rounds as u64;
                    t.drafted += (k * m_rounds) as u64;
                    t.accepted += cnt.saturating_sub(m_rounds) as u64;
                }
                round += m_rounds;
                // sse-cadence: the drained ring is committed — flush it at burst-drain cadence.
                keep_going = flush_commit(&mut on_commit, &out, &mut flushed);
                continue;
            }
            let pos = cache.pos; // #tokens committed (EXCLUDES a pending bonus)
            cache.snapshot_into(e, &mut snap)?; // §C: snapshot BEFORE draft+verify
            ph_mark(&mut ph_rest, phase_on);

            // --- 1. DRAFT k tokens with the NextN head (autoregressive, T=1 each) ---
            // p-min semantics (both paths): stop the chain early when the head's confidence in
            // its own pick drops below p_min — the just-drafted token is DISCARDED, but its
            // scratch append stands (identical to the eager chain's ordering). j==0 always drafts.
            let base0 = if pending.is_some() { 1usize } else { 0usize };
            // Round-start draft-KV sync (BOTH paths). Persistent: truncate/align to the committed
            // history — slots 0..P hold entries for the tokens before last_token@P (P = pos +
            // base0 - 1); this single set_len IS the draft-side rollback (drops last round's
            // rejected drafts and p-min extras via the len mechanism).
            scratch.set_len(e, pos + base0 - 1)?;
            if pen_on {
                let w0 = pen_hist.len().saturating_sub(sp.penalty_last_n);
                pen_hist_d = Some(e.htod_u32_v(&pen_hist[w0..])?);
            }
            // fixed draft length by default; MEMRA_SPEC_ADAPT=1 drafts at last round's
            // accepted run + 1 (the gemma law — see the setup block above the loop).
            let k_this = if adapt { kc } else { k };
            let mut draft: Vec<u32> = Vec::with_capacity(k);
            let mut draft_idx: Vec<u32> = Vec::with_capacity(k); // trimmed-vocab ids (== draft when untrimmed)
            if sampled {
                draft_logits.clear();
                draft_stats.clear();
            }
            // DRAFT-SIDE GRAMMAR MASK: clone the committed grammar state ONCE per round; each
            // position's mask is computed on that clone and advanced by the PROPOSED token. The
            // real state moves only on emission (verify's job), so the emitted stream is
            // unchanged — the mask only removes tokens the verify would have truncated anyway.
            let mut dmask_live = dmask_on;
            if dmask_live {
                let t_c = std::time::Instant::now();
                constraint
                    .as_deref_mut()
                    .unwrap()
                    .draft_begin()
                    .map_err(|e2| format!("constraint: {e2}"))?;
                dm_clone_ns += t_c.elapsed().as_nanos();
                dm_rounds += 1;
            }
            if let (false, Some(gr)) = (sampled || pen_on, &dctx.graph) {
                // GRAPH DRAFT: one dispatch per drafted token. The chain feeds itself on-device
                // (in-graph argmax -> tok_d -> next replay's embed; h_nextn -> h_seed_d; pos_d
                // inc'd in-graph); the host only reads 4B token (+4B p) and decides the break.
                e.set_i32_one(&mut dctx.g_pos, (pos + base0) as i32)?;
                e.set_u32_one(&mut dctx.g_tok, last_token)?;
                e.copy_into(&mut dctx.g_seed, 0, &h_seed_buf, n_embd)?;
                for j in 0..k_this {
                    // per-position mask upload (contents only — the graph's baked pointer is
                    // dctx.g_dmask). All-ones once masking goes dead mid-chain, so the captured
                    // mask node degrades to a no-op ban instead of needing a second graph.
                    if dmask_live
                        && !upload_draft_mask(
                            e,
                            constraint.as_deref_mut().unwrap(),
                            &mut dctx.g_dmask,
                            mtp.d2t.as_ref(),
                            d_vocab,
                            dmask_words,
                        )?
                    {
                        // no draft-vocab row is grammar-legal here (a trimmed FR-Spec head can
                        // genuinely miss the legal set): neutralize the captured mask node and
                        // finish the chain UNMASKED — exactly pre-lane behaviour, never worse.
                        e.htod_u32_into(&mut dctx.g_dmask, &vec![u32::MAX; dmask_words])?;
                        dmask_live = false;
                    }
                    gr.launch()?;
                    scratch.kv.len += 1; // host mirror (len_d advanced in-graph)
                    let idx = e.dtoh_u32_one(&dctx.g_tok)?;
                    // trimmed draft vocab -> target token id (identity when no d2t map)
                    let d = match &mtp.d2t {
                        Some(map) => map[idx as usize],
                        None => idx,
                    };
                    if p_min > 0.0 {
                        let p = e.dtoh(&dctx.g_p)?[0];
                        if p < p_min && (j > 0 || (pmin0 && base0 == 1)) {
                            break;
                        }
                    }
                    draft.push(d);
                    // with a trimmed head the NEXT embed must read the TARGET id, not the draft
                    // index the argmax wrote — patch the persistent token buffer (4B htod).
                    if d != idx {
                        e.set_u32_one(&mut dctx.g_tok, d)?;
                    }
                    // advance the SPECULATIVE state with the proposal; a dead chain drops to
                    // unmasked drafting for the remaining positions (verify still arbitrates).
                    // speculative advance; a chain the grammar can no longer follow (EOS
                    // proposed) ends here. The captured mask node always runs, so a dead chain
                    // leaves the buffer NEUTRAL (all-ones = ban nothing) before it exits.
                    if dmask_live
                        && !constraint
                            .as_deref_mut()
                            .unwrap()
                            .draft_advance(d)
                            .map_err(|e2| format!("constraint: {e2}"))?
                    {
                        e.htod_u32_into(&mut dctx.g_dmask, &vec![u32::MAX; dmask_words])?;
                        break;
                    }
                }
            } else if let (true, Some(gr)) = (sampled, &dctx.graph_s) {
                // SAMPLED GRAPH DRAFT: one replay per drafted token — head forward + gumbel +
                // argmax in ONE dispatch; the host reads 4B token (+4B p), D2Ds q into slot j,
                // and decides the break. Event-counter continuity: g_ctr is host-seeded to
                // sctr-1 ONCE per round (outside the graph); the in-graph bump runs BEFORE the
                // perturb, so replay j consumes counter sctr+j — exactly the eager arm's Philox
                // stream. Host sctr advances in lockstep (computed, no readback needed).
                e.set_i32_one(&mut dctx.g_pos, (pos + base0) as i32)?;
                e.set_u32_one(&mut dctx.g_tok, last_token)?;
                e.copy_into(&mut dctx.g_seed, 0, &h_seed_buf, n_embd)?;
                e.set_u32_one(&mut dctx.g_ctr, sctr.wrapping_sub(1))?;
                for j in 0..k_this {
                    gr.launch()?;
                    scratch.kv.len += 1; // host mirror (len_d advanced in-graph)
                    sctr += 1; // mirrors the in-graph g_ctr bump (eager parity:
                               // counts the p-min-discarded token too)
                               // q retention: ONE async D2D of the persistent head-logits buffer into this
                               // round's slot j (stream-ordered after the replay, before the next one).
                    e.copy_into(&mut dctx.q_slots[j], 0, &dctx.g_q, d_vocab)?;
                    let idx = e.dtoh_u32_one(&dctx.g_tok)?;
                    let d = match &mtp.d2t {
                        Some(map) => map[idx as usize],
                        None => idx,
                    };
                    draft_idx.push(idx);
                    if p_min > 0.0 {
                        let p = e.dtoh(&dctx.g_p)?[0];
                        if p < p_min && (j > 0 || (pmin0 && base0 == 1)) {
                            break;
                        }
                    }
                    draft.push(d);
                    // trimmed head: the NEXT embed must read the TARGET id (see the greedy arm).
                    if d != idx {
                        e.set_u32_one(&mut dctx.g_tok, d)?;
                    }
                }
                // uniform accept path: fill draft_stats per used slot (pure-temp regime — the
                // stats degenerate to th=0 / full-Z; one filter_stats launch per slot, tiny).
                for j in 0..draft.len().max(draft_idx.len()) {
                    let rows0 = e.htod_i32(&[0])?;
                    let (mut th_d, mut z_d, mut mx_d) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                    e.filter_stats(
                        &dctx.q_slots[j],
                        d_vocab,
                        &rows0,
                        &mut th_d,
                        &mut z_d,
                        &mut mx_d,
                        d_vocab,
                        1,
                        sp_temp,
                        sp.top_k,
                        sp.top_p,
                        sp.min_p,
                    )?;
                    draft_stats.push((e.dtoh(&mx_d)?[0], e.dtoh(&th_d)?[0], e.dtoh(&z_d)?[0]));
                }
            } else {
                // EAGER DRAFT (fallback: MoE head/trunk, huge k, MEMRA_SPEC_NOGRAPH, capture fail).
                let mut e_tok = last_token;
                let mut d_seed = e.clone_dtod(&h_seed_buf)?;
                for j in 0..k_this {
                    // GPU-ARGMAX DRAFT (2026-07-03): device logits + device argmax + 4-byte token
                    // read instead of the ~600KB full-vocab dtoh + host argmax per draft token.
                    let mtp_pos = pos + base0 + j;
                    // draft-side grammar mask (eager twin of the graph arm's in-graph node).
                    // A position with no legal draft-vocab row drops to unmasked drafting for
                    // the rest of the chain (pre-lane behaviour; verify still arbitrates).
                    if dmask_live {
                        dmask_live = upload_draft_mask(
                            e,
                            constraint.as_deref_mut().unwrap(),
                            &mut dctx.g_dmask,
                            mtp.d2t.as_ref(),
                            d_vocab,
                            dmask_words,
                        )?;
                    }
                    let (dl_d, h_nextn) = self.mtp_head_forward_dev(
                        e,
                        mtp,
                        e_tok,
                        &d_seed,
                        &mut *scratch,
                        mtp_pos,
                        embd_dev,
                        if dmask_live { Some((&dctx.g_dmask, dmask_words)) } else { None },
                    )?;
                    let tok_d = if sampled {
                        // FILTERED Gumbel-max: stats -> masked perturb -> argmax = one draw from
                        // the filtered softmax (filters off => th=0, exact v1 semantics).
                        if perturb_buf.is_none() {
                            perturb_buf = Some(e.zeros(d_vocab.max(n_vocab))?);
                        }
                        let mut q_row = e.clone_dtod(&dl_d)?; // retained q (penalized when on)
                        if pen_on {
                            let h = pen_hist_d.as_ref().unwrap();
                            let nh = h.len();
                            e.penalize_logits(
                                &mut q_row,
                                h,
                                nh,
                                sp.penalty_repeat,
                                sp.penalty_freq,
                                sp.penalty_present,
                                d_vocab,
                            )?;
                        }
                        let rows0 = e.htod_i32(&[0])?;
                        let (mut th_d, mut z_d, mut mx_d) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                        e.filter_stats(
                            &q_row, d_vocab, &rows0, &mut th_d, &mut z_d, &mut mx_d, d_vocab, 1,
                            sp_temp, sp.top_k, sp.top_p, sp.min_p,
                        )?;
                        let (th, z, mx) = (e.dtoh(&th_d)?[0], e.dtoh(&z_d)?[0], e.dtoh(&mx_d)?[0]);
                        let pb = perturb_buf.as_mut().unwrap();
                        e.gumbel_perturb_filtered(
                            &q_row, pb, d_vocab, sp_seed, sctr, sp_temp, mx, th,
                        )?;
                        sctr += 1;
                        draft_logits.push(q_row);
                        draft_stats.push((mx, th, z));
                        e.argmax_token_device(pb, d_vocab)?
                    } else {
                        e.argmax_token_device(&dl_d, d_vocab)?
                    };
                    let idx = e.dtoh_u32_one(&tok_d)?;
                    let d = match &mtp.d2t {
                        Some(map) => map[idx as usize],
                        None => idx,
                    };
                    if sampled {
                        draft_idx.push(idx);
                    }
                    if p_min > 0.0 {
                        let p_d = e.prob_of_token_device(&dl_d, &tok_d, d_vocab)?;
                        let p = e.dtoh(&p_d)?[0];
                        if p < p_min && (j > 0 || (pmin0 && base0 == 1)) {
                            break;
                        }
                    }
                    draft.push(d);
                    e_tok = d;
                    d_seed = h_nextn;
                    // speculative advance; a chain the grammar can no longer follow (EOS
                    // proposed) ends here — the prefix already proposed still rides verify.
                    if dmask_live
                        && !constraint
                            .as_deref_mut()
                            .unwrap()
                            .draft_advance(d)
                            .map_err(|e2| format!("constraint: {e2}"))?
                    {
                        break;
                    }
                }
            }
            let k_round = draft.len();

            ph_mark(&mut ph_draft, phase_on);
            // --- 2. VERIFY: one batched target forward. With a pending bonus, it rides as col 0
            //         (committing its KV/recur inside the SAME weight read); drafts follow. ---
            let verify_tokens: Vec<u32> = match pending {
                Some(b) => {
                    let mut v = Vec::with_capacity(k_round + 1);
                    v.push(b);
                    v.extend_from_slice(&draft);
                    v
                }
                None => draft.clone(),
            };
            let base = if pending.is_some() { 1 } else { 0 };
            // ckpt (REPLAY-FREE partial accept): retain per-layer state-rebuild inputs alongside
            // the verify. Pure buffer keep-alives + dtod clones — kernel work is unchanged.
            let mut ckpt = if spec_replay {
                None
            } else {
                Some(VerifyCkpt::new(self.layers.len()))
            };
            let (tlogits_d, vx) = self.decode_step_t_core(
                e,
                &verify_tokens,
                pos,
                &mut *cache,
                embd_dev,
                ckpt.as_mut(),
            )?;

            ph_mark(&mut ph_verify, phase_on);
            // --- 3. GREEDY ACCEPT (walk prefix, stop at first mismatch) ---
            // DEVICE-ARGMAX ACCEPT: argmax every verify column ON DEVICE (same 2-pass kernels +
            // smallest-index tie-break as host argmax, argmax_gate-validated) and read back ONE
            // [T] u32 — replaces the T x n_vocab f32 dtoh + T host argmaxes per round.
            // t_pred[j] = target's greedy prediction for the slot after draft[j-1] (j>=1) or after
            // last_token (j==0). With a pending bonus, col 0 IS the prediction after last_token
            // (== the bonus), so every index shifts by `base` and last_pred is unused.
            let t_v = verify_tokens.len();
            let mut preds: Vec<u32> = Vec::new();
            if !sampled {
                for j in 0..t_v {
                    e.argmax_token_device_col(&tlogits_d, j, n_vocab, &mut preds_d, j)?;
                }
                preds = e.dtoh_u32(&preds_d)?; // <- the verify-GPU wait lands here
            }
            ph_mark(&mut ph_wait, phase_on);
            let t_pred = |j: usize| -> u32 {
                if j == 0 && base == 0 {
                    last_pred
                } else {
                    preds[base + j - 1]
                }
            };
            let mut devacc_seeded = false;
            let mut devacc_acc: Option<CudaSlice<u32>> = None;
            let (n_acc, bonus) = if !sampled {
                // ROUND-STREAM stage (a) (MEMRA_SPEC_DEVACC=1 opt-in): the walk runs ON DEVICE
                // (spec_accept_greedy, verbatim rule) and the host reads back 8B (n_acc, bonus)
                // instead of the [T] preds. Same sync count — machinery for stages (b)/(c),
                // gated on token identity vs the host walk (the arms below are bit-equal rules).
                if crate::spec::spec_devacc() && k_round > 0 && !spec_replay
                    && constraint.is_none() {
                    let draft_d = e.htod_u32_v(&draft)?;
                    let mut acc_out = e.alloc_u32_zeroed(2)?;
                    e.spec_accept_greedy(
                        &preds_d,
                        &draft_d,
                        last_pred,
                        base,
                        k_round,
                        &mut acc_out,
                    )?;
                    devacc_acc = Some(acc_out.clone());
                    // stage (b): next-round seed gathered ON DEVICE from acc_out before the host
                    // ever reads n_acc (j=base+n_acc -> vx col j-1; j==0 -> fill_prev). The three
                    // non-replay commit arms skip their host-offset seed copies (guarded below);
                    // the legacy spec_replay arm keeps its own rx-based seeding (excluded here).
                    // NOTE: fill_prev is NOT updated here — the commit arms' TRUE-HIDDEN
                    // REFRESH reads the OLD fill_prev (predecessor of this round's verify batch);
                    // the update lands after the arms (devacc_seeded guard below).
                    e.spec_seed_gather(&vx, &fill_prev, &acc_out, &mut h_seed_buf, base, n_embd)?;
                    // 3a: KV lens roll back on device (len = saved + base + n_acc, all arms'
                    // unified rule; full accept rewrites the verify-left value). Host mirrors
                    // update after the readback; commit_verified_prefix skips its len_d writes.
                    if let Some(ptrs) = &kv_len_ptrs {
                        let saved: Vec<i32> = (0..self.layers.len())
                            .map(|il| snap.kv_len[il].map(|v| v as i32).unwrap_or(0))
                            .collect();
                        let saved_d = e.htod_i32(&saved)?;
                        e.spec_rollback_kv(ptrs, &saved_d, &acc_out, base, self.layers.len())?;
                    }
                    devacc_seeded = true;
                    let ab = e.dtoh_u32(&acc_out)?;
                    (ab[0] as usize, ab[1])
                } else {
                    let mut n_acc = 0usize;
                    for j in 0..k_round {
                        if t_pred(j) == draft[j] {
                            n_acc += 1;
                        } else {
                            break;
                        }
                    }
                    // bonus = target's own token at the first non-accepted slot. n_acc in 0..=k; t_pred
                    // is defined for j in 0..=k (j==0 -> last_logits, j>=1 -> col j-1, last col = k-1).
                    (n_acc, t_pred(n_acc))
                }
            } else {
                // --- SAMPLED ACCEPT (rejection sampling): u_j < p_j(x_j)/q_j(x_j) walk ---
                if col_buf.is_none() {
                    col_buf = Some(e.zeros(n_vocab)?);
                }
                // FILTERED p_j: per-verify-col stats (one batched filter_stats call), then the
                // filtered gather. j==0&&base==0 reads last_col (its own stats row appended).
                let mut pj = vec![0f32; k_round.max(1)];
                let mut col_stats: Vec<(f32, f32, f32)> = Vec::new(); // (max, th, z) per verify col used
                if k_round > 0 {
                    let mut ids: Vec<u32> = Vec::new();
                    let mut rows: Vec<i32> = Vec::new();
                    for j in 0..k_round {
                        if j > 0 || base == 1 {
                            ids.push(draft[j]);
                            rows.push((base + j) as i32 - 1);
                        }
                    }
                    if !ids.is_empty() {
                        let nr = rows.len();
                        // penalties: materialize the used columns into one contiguous penalized
                        // buffer (rows remapped 0..nr) so stats+gathers see the penalized p.
                        // penalties: materialize used columns contiguously, penalize all rows in
                        // one launch, and point stats+gathers at the penalized buffer (rows 0..nr).
                        let p_rows: Vec<i32> = if pen_on {
                            (0..nr as i32).collect()
                        } else {
                            rows.clone()
                        };
                        if pen_on {
                            if pcol_buf.as_ref().map(|b| b.len()).unwrap_or(0) < nr * n_vocab {
                                pcol_buf = Some(e.zeros(nr * n_vocab)?);
                            }
                            let pc = pcol_buf.as_mut().unwrap();
                            for (i2, &r) in rows.iter().enumerate() {
                                let c = r as usize;
                                e.copy_view_into(
                                    pc,
                                    i2 * n_vocab,
                                    &tlogits_d.slice(c * n_vocab..(c + 1) * n_vocab),
                                    n_vocab,
                                )?;
                            }
                            let h = pen_hist_d.as_ref().unwrap();
                            let nh = h.len();
                            e.penalize_logits_rows(
                                pc,
                                h,
                                nh,
                                sp.penalty_repeat,
                                sp.penalty_freq,
                                sp.penalty_present,
                                n_vocab,
                                nr,
                            )?;
                        }
                        let p_src: &CudaSlice<f32> = if pen_on {
                            pcol_buf.as_ref().unwrap()
                        } else {
                            &tlogits_d
                        };
                        let rowsd = e.htod_i32(&p_rows)?;
                        let (mut th_d, mut z_d, mut mx_d) =
                            (e.zeros(nr)?, e.zeros(nr)?, e.zeros(nr)?);
                        e.filter_stats(
                            p_src, n_vocab, &rowsd, &mut th_d, &mut z_d, &mut mx_d, n_vocab, nr,
                            sp_temp, sp.top_k, sp.top_p, sp.min_p,
                        )?;
                        let idsd = e.htod_u32_v(&ids)?;
                        let mut outd = e.zeros(nr)?;
                        e.softmax_gather_filtered(
                            p_src, n_vocab, &idsd, &rowsd, &th_d, &z_d, &mut outd, n_vocab, nr,
                            sp_temp,
                        )?;
                        let outv = e.dtoh(&outd)?;
                        let (thv, zv, mxv) = (e.dtoh(&th_d)?, e.dtoh(&z_d)?, e.dtoh(&mx_d)?);
                        let mut oi = 0usize;
                        for j in 0..k_round {
                            if j > 0 || base == 1 {
                                pj[j] = outv[oi];
                                oi += 1;
                            }
                        }
                        col_stats = (0..nr).map(|i| (mxv[i], thv[i], zv[i])).collect();
                    }
                    if base == 0 {
                        let lc: &CudaSlice<f32> = if pen_on {
                            if col_buf.is_none() {
                                col_buf = Some(e.zeros(n_vocab)?);
                            }
                            let cb = col_buf.as_mut().unwrap();
                            e.copy_into(
                                cb,
                                0,
                                last_col_logits
                                    .as_ref()
                                    .expect("sampled: last_col_logits unset"),
                                n_vocab,
                            )?;
                            let h = pen_hist_d.as_ref().unwrap();
                            let nh = h.len();
                            e.penalize_logits(
                                cb,
                                h,
                                nh,
                                sp.penalty_repeat,
                                sp.penalty_freq,
                                sp.penalty_present,
                                n_vocab,
                            )?;
                            col_buf.as_ref().unwrap()
                        } else {
                            last_col_logits
                                .as_ref()
                                .expect("sampled: last_col_logits unset")
                        };
                        let rows0 = e.htod_i32(&[0])?;
                        let (mut th_d, mut z_d, mut mx_d) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                        e.filter_stats(
                            lc, n_vocab, &rows0, &mut th_d, &mut z_d, &mut mx_d, n_vocab, 1,
                            sp_temp, sp.top_k, sp.top_p, sp.min_p,
                        )?;
                        let idsd = e.htod_u32_v(&[draft[0]])?;
                        let mut outd = e.zeros(1)?;
                        e.softmax_gather_filtered(
                            lc, n_vocab, &idsd, &rows0, &th_d, &z_d, &mut outd, n_vocab, 1, sp_temp,
                        )?;
                        pj[0] = e.dtoh(&outd)?[0];
                        last_col_stats =
                            Some((e.dtoh(&mx_d)?[0], e.dtoh(&th_d)?[0], e.dtoh(&z_d)?[0]));
                    }
                }
                // q source: the graph arm retained the head logits in the persistent q_slots;
                // the eager arm in per-round draft_logits clones. Same raw-logit values either way.
                // FILTERED q_j: stats from draft_stats (eager pushes in-chain; the graph arm
                // computes them post-replay — graph engages only filter/penalty-free, so the
                // stats degenerate to th=0/full-Z there, keeping ONE accept path).
                let q_bufs: &[CudaSlice<f32>] = if dctx.graph_s.is_some() {
                    &dctx.q_slots
                } else {
                    &draft_logits
                };
                let mut n_acc = 0usize;
                for j in 0..k_round {
                    let (qmx, qth, qz) = draft_stats[j];
                    let idsd = e.htod_u32_v(&[draft_idx[j]])?;
                    let rowsd = e.htod_i32(&[0])?;
                    let thd = e.htod(&[qth])?;
                    let zd = e.htod(&[qz])?;
                    let _ = qmx;
                    let mut outd = e.zeros(1)?;
                    e.softmax_gather_filtered(
                        &q_bufs[j], d_vocab, &idsd, &rowsd, &thd, &zd, &mut outd, d_vocab, 1,
                        sp_temp,
                    )?;
                    let qj = e.dtoh(&outd)?[0];
                    let u = host_u01(sp_seed, uctr);
                    uctr += 1;
                    if (u as f64) * (qj as f64) < pj[j] as f64 {
                        n_acc += 1;
                    } else {
                        break;
                    }
                }
                let bonus = if n_acc == k_round {
                    // FULL ACCEPT: bonus ~ FILTERED softmax at the last verify column.
                    let col = base + k_round - 1;
                    let cb = col_buf.as_mut().unwrap();
                    e.copy_view_into(
                        cb,
                        0,
                        &tlogits_d.slice(col * n_vocab..(col + 1) * n_vocab),
                        n_vocab,
                    )?;
                    if pen_on {
                        let h = pen_hist_d.as_ref().unwrap();
                        let nh = h.len();
                        e.penalize_logits(
                            cb,
                            h,
                            nh,
                            sp.penalty_repeat,
                            sp.penalty_freq,
                            sp.penalty_present,
                            n_vocab,
                        )?;
                    }
                    if perturb_buf.is_none() {
                        perturb_buf = Some(e.zeros(d_vocab.max(n_vocab))?);
                    }
                    // STATS MUST COME FROM THIS COLUMN (bug fix 2026-08-05, lane/sampler-
                    // truncation-fix; receipts research/sampfix-20260805/). The old code reused
                    // `col_stats.last()` here, which is ALWAYS the wrong row: the gathered set
                    // covers verify columns 0..=(base+k_round-2) (rows pushed as base+j-1), while
                    // the full-accept bonus samples column base+k_round-1 — exactly ONE PAST the
                    // last gathered column, in both base arms. `th` is a threshold in e-units of
                    // its OWN row's max, so feeding a neighbour's (row_max, th) into
                    // gumbel_perturb_filtered mis-scales every e0 = exp((x-row_max)/T). When the
                    // donor column's peak is higher by more than T*ln(1/th), EVERY id fails
                    // `e0 >= th`, the whole perturbed row becomes -3.4e38, and the 2-pass argmax
                    // falls through to its smallest-index tie-break => token id 0 ("!") spliced
                    // mid-word. Fragility is ordered by how large th is: min_p pins th = min_p
                    // (0.05 => trigger at delta > 2.4 at T=0.8, fires constantly), top_p's
                    // mass-boundary th is smaller, top_k's k-th-largest th smaller still — which
                    // is why the head-to-head matrix saw min_p and top_p corrupt while top_k-only
                    // stayed clean. The pure-temp default regime is immune (th == 0 masks nothing,
                    // and row_max is unused once nothing is masked), so this fix is a byte-level
                    // no-op for the untruncated serve default. One extra one-block filter_stats
                    // per full-accept round is the whole cost.
                    let (mx, th) = {
                        let rows0 = e.htod_i32(&[0])?;
                        let (mut th_d, mut z_d, mut mx_d) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                        let cb0 = col_buf.as_ref().unwrap();
                        e.filter_stats(
                            cb0, n_vocab, &rows0, &mut th_d, &mut z_d, &mut mx_d, n_vocab, 1,
                            sp_temp, sp.top_k, sp.top_p, sp.min_p,
                        )?;
                        (e.dtoh(&mx_d)?[0], e.dtoh(&th_d)?[0])
                    };
                    let pb = perturb_buf.as_mut().unwrap();
                    let cb2 = col_buf.as_ref().unwrap();
                    e.gumbel_perturb_filtered(cb2, pb, n_vocab, sp_seed, sctr, sp_temp, mx, th)?;
                    sctr += 1;
                    let td = e.argmax_token_device(pb, n_vocab)?;
                    e.dtoh_u32_one(&td)?
                } else {
                    // REJECT at n_acc: bonus ~ norm(max(0, softmax_T(p) - softmax_T(q))).
                    let cb = col_buf.as_mut().unwrap();
                    if n_acc > 0 || base == 1 {
                        let col = base + n_acc - 1;
                        e.copy_view_into(
                            cb,
                            0,
                            &tlogits_d.slice(col * n_vocab..(col + 1) * n_vocab),
                            n_vocab,
                        )?;
                    } else {
                        let lc = last_col_logits.as_ref().unwrap();
                        e.copy_into(cb, 0, lc, n_vocab)?;
                    }
                    if pen_on {
                        let h = pen_hist_d.as_ref().unwrap();
                        let nh = h.len();
                        e.penalize_logits(
                            cb,
                            h,
                            nh,
                            sp.penalty_repeat,
                            sp.penalty_freq,
                            sp.penalty_present,
                            n_vocab,
                        )?;
                    }
                    let cb2 = col_buf.as_ref().unwrap();
                    let sc = sctr;
                    sctr += 1;
                    // p-stats for the reject column: from col_stats when the col was gathered,
                    // else (j==0&&base==0) from last_col_stats.
                    let p_stats = if n_acc > 0 || base == 1 {
                        // col index within the gathered set == number of gathered cols before n_acc
                        let gi = if base == 1 { n_acc } else { n_acc - 1 };
                        col_stats.get(gi).copied().unwrap_or_else(|| {
                            (0.0, 0.0, 1.0) // unreachable: gathered cols always cover the reject slot
                        })
                    } else {
                        last_col_stats.expect("sampled: last_col_stats unset at reject")
                    };
                    let q_stats = draft_stats[n_acc];
                    if let Some(map) = &d2t_dev {
                        if q_full_buf.is_none() {
                            q_full_buf = Some(e.zeros(n_vocab)?);
                        }
                        let qf = q_full_buf.as_mut().unwrap();
                        e.scatter_trim_logits(&q_bufs[n_acc], map, qf, d_vocab, n_vocab)?;
                        let qf2 = q_full_buf.as_ref().unwrap();
                        e.residual_sample_filtered(
                            cb2,
                            Some(qf2),
                            n_vocab,
                            sp_temp,
                            sp_seed,
                            sc,
                            p_stats,
                            q_stats,
                            &mut sample_tok,
                        )?;
                    } else {
                        e.residual_sample_filtered(
                            cb2,
                            Some(&q_bufs[n_acc]),
                            n_vocab,
                            sp_temp,
                            sp_seed,
                            sc,
                            p_stats,
                            q_stats,
                            &mut sample_tok,
                        )?;
                    }
                    e.dtoh_u32(&sample_tok)?[0]
                };
                (n_acc, bonus)
            };
            // --- 3b. GRAMMAR TRUNCATION (constrained spec, 2026-08-03): the grammar is
            // an extra rejection rule AFTER the exactness verify (the batched-verify-twins
            // ordering). Walk the accepted drafts through the grammar in commit order; the
            // first illegal token truncates acceptance at its slot, and that slot's emission
            // is recomputed as the MASKED argmax of the target's own verify column — token-
            // identical to constrained plain greedy decode (an unmasked argmax that is
            // grammar-legal IS the masked argmax: masking only removes competitors). The
            // column D2H (~1MB) is paid only when a cut fires — the tight-grammar cost,
            // measured in acceptance numbers, never hidden.
            let (n_acc, bonus) = match constraint.as_deref_mut() {
                None => (n_acc, bonus),
                Some(c) => {
                    fn ce(e2: String) -> Box<dyn std::error::Error> {
                        format!("constraint: {e2}").into()
                    }
                    let mut na = n_acc;
                    let mut cut = false;
                    for (j, &d) in draft.iter().enumerate().take(n_acc) {
                        if c.is_allowed(d).map_err(ce)? {
                            c.consume(d).map_err(ce)?;
                        } else {
                            na = j;
                            cut = true;
                            dm_cut_tokens += n_acc - j;
                            break;
                        }
                    }
                    if cut {
                        dm_cuts += 1;
                    }
                    let mut bo = bonus;
                    if cut || !c.is_allowed(bo).map_err(ce)? {
                        let mut row = if na == 0 && base == 0 {
                            init_logits_host.clone()
                                .ok_or("constraint: init logits missing (round-0 cut)")?
                        } else {
                            e.dtoh_view(&tlogits_d.slice(
                                (base + na - 1) * n_vocab..(base + na) * n_vocab))?
                        };
                        c.mask_logits(&mut row).map_err(ce)?;
                        bo = argmax(&row) as u32;
                    }
                    c.consume(bo).map_err(ce)?;
                    (na, bo)
                }
            };
            total_drafted += k_round;
            total_accepted += n_acc;
            if let Some(t) = sess_telem.as_deref_mut() {
                // per-position accept walk (lane/accept-telemetry): host u64 adds on counts
                // the round already read back — zero syncs, zero allocation.
                t.rounds += 1;
                t.drafted += k_round as u64;
                t.accepted += n_acc as u64;
                for j in 0..k_round.min(SPEC_TELEM_POS) {
                    t.pos_drafted[j] += 1;
                }
                for j in 0..n_acc.min(SPEC_TELEM_POS) {
                    t.pos_accepted[j] += 1;
                }
            }
            if spec_stats {
                st_len_hist[k_round] += 1;
                for j in 0..k_round {
                    st_drafted[j] += 1;
                }
                for j in 0..n_acc {
                    st_accepted[j] += 1;
                }
                if n_acc == k_round {
                    st_full += 1;
                }
            }

            if debug_spec {
                eprintln!("[R{round}] pos={pos} out_len={} last_tok={last_token} draft={draft:?} n_acc={n_acc} bonus={bonus} t_pred0={}", out.len(), t_pred(0));
            }

            // --- 4. COMMIT: draft[0..n_acc] then bonus (n_acc + 1 tokens) ---
            // SESSION MODE: every accepted column is already in the CACHE — `out` must carry all
            // of them (overshoot past max_new included) or `committed` under-counts the cache rows
            // and the next turn's continuation seeds one token off (gate-caught 2026-07-05). The
            // single-shot path keeps the cap (its caller truncates + drops the cache anyway).
            for j in 0..n_acc {
                if !session_mode && out.len() >= max_new {
                    break;
                }
                out.push(draft[j]);
            }
            if pen_on {
                pen_hist.extend_from_slice(&draft[0..n_acc]);
                pen_hist.push(bonus);
            }
            let bonus_emitted = session_mode || out.len() < max_new;
            if bonus_emitted {
                out.push(bonus);
            }
            last_token = bonus;

            // --- 5. ROLLBACK + advance (§C) ---
            if n_acc == k_round {
                // FULL ACCEPT, BONUS FOLD: all verify columns (pending? + drafts) are committed in
                // cache; the NEW bonus stays PENDING for the next round's verify batch — NO extra
                // T=1 trunk pass. The next draft chain seeds from the MTP block's h_nextn at the
                // bonus position: one MTP-block pass (~1/33 trunk cost) replaces the trunk read.
                // last_pred is dead in the pending path (t_pred reads verify col 0).
                //
                // PERSISTENT DRAFT KV, full-accept fill: the chain covered last_token +
                // draft[0..k_round-2] as INPUTS (slots P..P'-2); draft[k_round-1] (slot P'-1) was
                // only ever an output, so its entry is MISSING. Fill it from vh_seed — its EXACT
                // trunk hidden (the last verify column). set_len first: a p-min break may have
                // left one extra chain append at that slot. Partial accepts need NO fill (the
                // chain already covered every accepted position; round-start set_len truncates).
                let mut vh_seed = e.zeros(n_embd)?;
                e.copy_view_into(
                    &mut vh_seed,
                    0,
                    &vx.slice((t_v - 1) * n_embd..t_v * n_embd),
                    n_embd,
                )?;
                if refresh {
                    // TRUE-HIDDEN REFRESH (2026-07-03, the HANDOVER-listed acceptance lever):
                    // overwrite ALL committed positions' scratch entries with K/V from their EXACT
                    // verify hiddens — the reference engine's mtp_update fills from true hiddens;
                    // the full stack (vx) is already resident from the verify. Replaces both the
                    // chain-approximate entries AND the old last-token-only fill. Acceptance-only
                    // (draft attention quality); exactness stays the verify's job.
                    scratch.set_len(e, pos)?;
                    // PREDECESSOR pairing: row i gets vx[i-1]; row 0 the carried fill_prev
                    // (hidden of the last committed row before this verify batch).
                    let mut vxs = e.zeros(t_v * n_embd)?;
                    e.copy_into(&mut vxs, 0, &fill_prev, n_embd)?;
                    if t_v > 1 {
                        e.copy_view_into(
                            &mut vxs,
                            n_embd,
                            &vx.slice(0..(t_v - 1) * n_embd),
                            (t_v - 1) * n_embd,
                        )?;
                    }
                    self.mtp_kv_fill(e, mtp, &verify_tokens, &vxs, pos, &mut *scratch, embd_dev)?;
                } else {
                    scratch.set_len(e, pos + base + k_round - 1)?;
                    // predecessor of the last draft = verify col t_v-2 (or fill_prev at t_v==1)
                    let mut hp = e.zeros(n_embd)?;
                    if t_v >= 2 {
                        e.copy_view_into(
                            &mut hp,
                            0,
                            &vx.slice((t_v - 2) * n_embd..(t_v - 1) * n_embd),
                            n_embd,
                        )?;
                    } else {
                        e.copy_into(&mut hp, 0, &fill_prev, n_embd)?;
                    }
                    self.mtp_kv_fill(
                        e,
                        mtp,
                        &[draft[k_round - 1]],
                        &hp,
                        pos + base + k_round - 1,
                        &mut *scratch,
                        embd_dev,
                    )?;
                }
                // REFERENCE SEEDING: no pseudo pass — the next chain's step 0 IS the
                // reference's (id_last, h_prev) draft row; it appends the bonus's scratch
                // entry itself. Seed = TRUE hidden of the bonus's predecessor (last verify
                // col). Saves one MTP-block pass per round on top of the pairing fix.
                if !devacc_seeded {
                    e.copy_into(&mut h_seed_buf, 0, &vh_seed, n_embd)?;
                    e.copy_into(&mut fill_prev, 0, &vh_seed, n_embd)?;
                }
                pending = Some(bonus);
                if debug_spec {
                    eprintln!("  -> FULL ACCEPT (bonus pending, prev-h seed)");
                }
            } else if !spec_replay && base + n_acc >= 1 {
                // PARTIAL ACCEPT, REPLAY-FREE (2026-07-03 — the profiled #1 long-ctx spec cost):
                // the verify's first j = base+n_acc columns ARE the committed sequence, computed
                // bit-identically to eager (decode-exact contract) — so KEEP them: KV truncates to
                // pos+j, recurrent state rebuilds from the VerifyCkpt (same-kernel gdn prefix
                // re-run / pure state-clone restore), and the bonus stays PENDING exactly like the
                // full-accept path — the legacy duplicate trunk replay is gone. The next chain
                // seeds from the MTP pseudo-hidden of the bonus, whose seed = the TRUE verify
                // hidden of its predecessor (col j-1) — same one-hop pseudo structure as full
                // accept (never compounds: the next verify recomputes true hiddens for all
                // committed columns).
                let j = base + n_acc;
                self.commit_verified_prefix(
                    e,
                    &mut *cache,
                    &snap,
                    ckpt.as_ref().unwrap(),
                    j,
                    devacc_seeded,
                    if devacc_seeded {
                        devacc_acc.as_ref().map(|a| (a, base, t_v))
                    } else {
                        None
                    },
                )?;
                let mut seed = e.zeros(n_embd)?;
                e.copy_view_into(
                    &mut seed,
                    0,
                    &vx.slice((j - 1) * n_embd..j * n_embd),
                    n_embd,
                )?;
                // Draft scratch: TRUE-HIDDEN REFRESH of the committed prefix (see the full-accept
                // branch); without it the chain entries stand and only the tail truncates. Either
                // way len ends at pos+j so the pseudo append lands at the bonus's slot pos+j
                // (persistent mode), rope pos+j+1 (chain convention).
                if refresh {
                    scratch.set_len(e, pos)?;
                    let mut vxs = e.zeros(j * n_embd)?;
                    e.copy_into(&mut vxs, 0, &fill_prev, n_embd)?;
                    if j > 1 {
                        e.copy_view_into(
                            &mut vxs,
                            n_embd,
                            &vx.slice(0..(j - 1) * n_embd),
                            (j - 1) * n_embd,
                        )?;
                    }
                    self.mtp_kv_fill(
                        e,
                        mtp,
                        &verify_tokens[0..j],
                        &vxs,
                        pos,
                        &mut *scratch,
                        embd_dev,
                    )?;
                } else {
                    scratch.set_len(e, pos + j)?;
                }
                // REFERENCE SEEDING (see the full-accept branch): seed = TRUE hidden of the
                // bonus's predecessor (verify col j-1); no pseudo pass.
                if !devacc_seeded {
                    e.copy_into(&mut h_seed_buf, 0, &seed, n_embd)?;
                    e.copy_into(&mut fill_prev, 0, &seed, n_embd)?;
                }
                pending = Some(bonus);
                if debug_spec {
                    eprintln!("  -> PARTIAL(replay-free j={j}, bonus pending, prev-h seed)");
                }
            } else if !spec_replay {
                // ZERO ROUND FOLD (2026-07-10, verify-cost target #3): base+n_acc == 0 — a
                // pending-less round where nothing was accepted (PMIN0 zero-draft chains after a
                // replay/commit, or plain 0-accept rounds at round 0). The old path replayed
                // [bonus] through a FULL m=1 trunk+head forward (the 489us full-vocab head pass
                // measured at ~0.75/round on PMIN0 configs). Instead: restore the pre-round
                // snapshot and let the bonus ride the NEXT round's verify as col 0 — the existing
                // base=1 pending machinery, bit-identical by the decode-exact verify contract.
                // Seed: the bonus's predecessor is the last COMMITTED token, whose hidden
                // fill_prev already carries (same seeding as the 1-token-replay case it replaces).
                cache.rollback(e, &snap, 0)?;
                scratch.set_len(e, pos)?;
                e.copy_into(&mut h_seed_buf, 0, &fill_prev, n_embd)?;
                pending = Some(bonus);
                if debug_spec {
                    eprintln!("  -> ZERO-ROUND FOLD (bonus pending, fill_prev seed)");
                }
            } else {
                // PARTIAL ACCEPT, LEGACY REPLAY (seam MEMRA_SPEC_REPLAY=1 — or j==0: nothing of
                // this round survives, only possible before the first pending exists, ~round 0):
                // restore EVERYTHING to the pre-round snapshot (KV truncate to pos + recur
                // restore), then replay the committed prefix pending? ++ draft[0..n_acc] ++
                // [bonus] as ONE batched T forward — single weight read, bit-identical to greedy
                // (the verify-all-columns path is the same math). Commits the bonus with a TRUE
                // trunk hidden.
                cache.rollback(e, &snap, 0)?; // accept_len=0: KV len = pos, recur = snapshot
                let mut replay: Vec<u32> = Vec::with_capacity(base + n_acc + 1);
                if let Some(b) = pending.take() {
                    replay.push(b);
                }
                replay.extend_from_slice(&draft[0..n_acc]);
                replay.push(bonus);
                // Full-stack forward (decode_step_t_core = decode_step_t_h_emb_dev's body):
                // Predecessor pairing seeds from the PREDECESSOR row (col len-2) — the same-row path takes the
                // last col exactly as before (byte-identical to the old _h_emb_dev call).
                let (rl_d, rx) =
                    self.decode_step_t_core(e, &replay, pos, &mut *cache, embd_dev, None)?;
                // last_pred = argmax of the LAST column's logits (predicts the token after `bonus`)
                // — device argmax + one 4-byte read instead of the full-vocab column dtoh.
                e.argmax_token_device_col(&rl_d, replay.len() - 1, n_vocab, &mut preds_d, 0)?;
                last_pred = e.dtoh_u32(&preds_d)?[0];
                if sampled {
                    let lr0 = replay.len();
                    let lc = last_col_logits
                        .as_mut()
                        .expect("sampled: last_col_logits unset");
                    e.copy_view_into(
                        lc,
                        0,
                        &rl_d.slice((lr0 - 1) * n_vocab..lr0 * n_vocab),
                        n_vocab,
                    )?;
                }
                let lr = replay.len();
                if lr >= 2 {
                    e.copy_view_into(
                        &mut h_seed_buf,
                        0,
                        &rx.slice((lr - 2) * n_embd..(lr - 1) * n_embd),
                        n_embd,
                    )?;
                } else {
                    // 1-token replay (round-0 miss): the bonus's predecessor is the OLD
                    // last_token, whose own-row hidden fill_prev still holds.
                    e.copy_into(&mut h_seed_buf, 0, &fill_prev, n_embd)?;
                }
                // the bonus is COMMITTED here — it becomes the last committed row.
                let mut rh_last = e.zeros(n_embd)?;
                e.copy_view_into(
                    &mut rh_last,
                    0,
                    &rx.slice((lr - 1) * n_embd..lr * n_embd),
                    n_embd,
                )?;
                e.copy_into(&mut fill_prev, 0, &rh_last, n_embd)?;
                if debug_spec {
                    eprintln!("  -> PARTIAL(replay={replay:?}), next_pred={last_pred}");
                }
            }
            if devacc_seeded {
                // stage (b) epilogue: fill_prev takes the gathered seed AFTER the refresh fills
                // consumed the old value (both slots carry the same value in every non-replay arm).
                e.copy_into(&mut fill_prev, 0, &h_seed_buf, n_embd)?;
            }
            // adaptive-K update (host math, zero syncs): next round drafts accepted-run + 1,
            // clamped to [floor(pos), k_cap]. cache.pos is post-rollback here (the round's
            // final position — the floor's position key reads the committed depth). Burst
            // rounds (`continue` above) draft the captured fixed depth and skip this, exactly
            // like gemma's burst arm.
            if adapt {
                let fl_now = floor_at(cache.pos);
                kc = (n_acc + 1).clamp(fl_now.min(k_cap), k_cap);
            }
            ph_mark(&mut ph_rest, phase_on);
            round += 1;
            // sse-cadence: this round's accepted drafts + bonus are committed (out is
            // append-only past step 4) — flush at round cadence.
            keep_going = flush_commit(&mut on_commit, &out, &mut flushed);
        }
        // sse-cadence: nothing below appends to `out`; flush any remainder (defensive).
        // (verdict ignored — the burst is over either way; the session tail runs unchanged.)
        let _ = flush_commit(&mut on_commit, &out, &mut flushed);

        if spec_stats {
            let per_slot: Vec<String> = (0..k)
                .map(|j| {
                    if st_drafted[j] > 0 {
                        format!(
                            "{}/{}={:.3}",
                            st_accepted[j],
                            st_drafted[j],
                            st_accepted[j] as f64 / st_drafted[j] as f64
                        )
                    } else {
                        "0/0".into()
                    }
                })
                .collect();
            let acc = if total_drafted > 0 {
                total_accepted as f64 / total_drafted as f64
            } else {
                0.0
            };
            eprintln!(
                "[spec-stats] rounds={round} full_accept={st_full} len_hist={st_len_hist:?} \
                       per_slot=[{}] total={total_accepted}/{total_drafted}={acc:.3} \
                       tok_per_round={:.3}",
                per_slot.join(" "),
                (total_accepted + round) as f64 / round.max(1) as f64
            );
        }
        if constraint.is_some() {
            eprintln!(
                "[draft-mask] mask_rounds={dm_rounds} clone_total={:.3}ms \
                 clone_per_round={:.4}ms gram_cuts={dm_cuts}/{round} cut_tokens={dm_cut_tokens}",
                dm_clone_ns as f64 / 1e6,
                dm_clone_ns as f64 / 1e6 / dm_rounds.max(1) as f64
            );
        }
        if phase_on {
            let tot = ph_draft + ph_verify + ph_wait + ph_rest;
            eprintln!("[spec-phase] draft={:.1}ms ({:.1}%) verify-issue={:.1}ms ({:.1}%) verify-wait={:.1}ms ({:.1}%) commit-host={:.1}ms ({:.1}%) rounds={round}",
                      ph_draft * 1e3, ph_draft / tot * 100.0,
                      ph_verify * 1e3, ph_verify / tot * 100.0,
                      ph_wait * 1e3, ph_wait / tot * 100.0,
                      ph_rest * 1e3, ph_rest / tot * 100.0);
        }
        // SESSION TAIL: leave the session in the exact invariant the next turn's suffix prime
        // expects — every row in `committed` has trunk KV/recur state AND an exact draft-KV row.
        // Park the draft-graph ctx back on the session (the serve-burst fixed-cost fix): the next
        // burst replays instead of recapturing. Error paths (`?` above) drop it — recaptured then.
        if let Some(slot) = sess_draft_slot.take() {
            *slot = Some(dctx);
        }
        let t_rounds = t_ent.elapsed();
        if let Some((committed, last_h, next_pred_slot, sctr_slot, uctr_slot)) = sess_tail.take() {
            *sctr_slot = sctr;
            *uctr_slot = uctr;
            *next_pred_slot = Some(last_pred);
            let mut stashed_pending = false;
            if let Some(b) = pending.take() {
                if !sampled {
                    // PENDING-CARRY (2026-08-01): stash the bonus on the session instead of
                    // committing it with a solo T=1 pass — the next empty-suffix greedy burst
                    // consumes it as round-0 verify col 0 (a plain round edge; the old tail
                    // commit + next burst's init feed were 11.6+11.5ms solo trunk passes per
                    // burst on H100 q27, [spec-setup] trace). b stays in `out` (emitted) but
                    // OUT of `committed` (cache rows == committed); the consuming call
                    // prepends it once its verify commits the row. next_pred is unknowable
                    // without the commit pass — None; callers gate on pending_tok too.
                    debug_assert_eq!(out.last(), Some(&b), "pending must be the last emitted");
                    if let Some(slot) = sess_pending_slot.take() {
                        *slot = Some(b);
                    }
                    *next_pred_slot = None;
                    // fill_prev = hidden of the last COMMITTED row (b's predecessor) — the
                    // exact chain-seed/fill anchor the consuming burst (or a flush) needs.
                    *last_h = Some(e.clone_dtod(&fill_prev)?);
                    stashed_pending = true;
                } else {
                    // SAMPLED tail (unchanged): commit the bonus (one T=1 pass) + draft fill —
                    // the sampled round-0 accept needs this pass's logits (last_col_logits).
                    let pos_b = cache.pos;
                    scratch.set_len(e, pos_b)?;
                    let (lg_b, hb) = self.decode_step_h(e, b, &mut *cache)?;
                    // after a FULL-accept exit `last_pred` is STALE (it predicted the bonus
                    // itself — the prediction AFTER the bonus never materialized; it would have
                    // been the next round's verify col 0). The commit's logits ARE that
                    // prediction.
                    *next_pred_slot = Some(argmax(&lg_b) as u32);
                    self.mtp_kv_fill(e, mtp, &[b], &fill_prev, pos_b, &mut *scratch, embd_dev)?;
                    *last_h = Some(hb);
                }
            } else {
                // fill_prev tracks the hidden of the last COMMITTED row throughout the loop.
                *last_h = Some(e.clone_dtod(&fill_prev)?);
            }
            committed.extend_from_slice(prompt);
            if let Some(cb) = carried_pending {
                // the consumed carry's cache row landed in round 0's verify (every pending
                // round commits col 0) — it joins `committed` here, in sequence order.
                committed.push(cb);
            }
            if stashed_pending {
                // ZERO-EMIT BURST (2026-08-06 c=8 serve panic, pre-existing since b4aea184):
                // `out.len() - 1` underflowed on an EMPTY `out` — "range end index
                // 18446744073709551615 out of range for slice of length 0", killing the
                // memra-gpu-worker and failing 31 of 32 concurrent requests with "worker closed
                // stream". Reachable because `pending` starts as `carried_pending` (a bonus
                // stashed by the PREVIOUS burst) while `out` starts empty, and the carry is
                // deliberately NOT pushed to `out` (line ~3239: the burst that emitted it already
                // did). So a burst that stashes a pending without emitting anything of its own —
                // the round loop exits before a push, e.g. the ring drain's `out.len() < max_new`
                // guard skipping every token under a tight budget — arrives here with
                // out.len() == 0 and stashed_pending == true.
                //
                // The invariant is unchanged: `committed` gets every emitted token EXCEPT the
                // stashed bonus. With nothing emitted, that is nothing — and the carry pushed
                // just above is already accounted. Saturating, not a min/assert: an empty `out`
                // here is a legitimate burst shape, not a corrupt state.
                let emitted = out.len().saturating_sub(1);
                committed.extend_from_slice(&out[..emitted]);
            } else {
                committed.extend_from_slice(&out); // FULL out incl. overshoot — all committed
            }
            debug_assert_eq!(
                cache.pos,
                committed.len(),
                "session invariant: cache rows == committed tokens"
            );
            if setup_trace {
                e.stream().synchronize()?; // bound the async tail fill in the trace
                let t_tail = t_ent.elapsed();
                eprintln!(
                    "[spec-setup] init={:.2}ms cap={:.2}ms fill={:.2}ms rounds={:.2}ms tail={:.2}ms total={:.2}ms out={} cont={}",
                    t_init.as_secs_f64() * 1e3,
                    (t_cap - t_init).as_secs_f64() * 1e3,
                    (t_fill - t_cap).as_secs_f64() * 1e3,
                    (t_rounds - t_fill).as_secs_f64() * 1e3,
                    (t_tail - t_rounds).as_secs_f64() * 1e3,
                    t_tail.as_secs_f64() * 1e3,
                    out.len(),
                    continuation
                );
            }
            return Ok((out, total_drafted, total_accepted));
        }
        out.truncate(max_new);
        Ok((out, total_drafted, total_accepted))
    }

    /// TEACHER-FORCED REPLAY ACCEPTANCE (hqmtp MTP-heal protocol): walk a FIXED token
    /// sequence and, at sampled positions, compare the MTP head's K-token draft chain against
    /// the trunk's own teacher-forced greedy predictions. Nothing is generated — the context is
    /// the corpus text itself, so (a) degenerate self-generated loops cannot inflate acceptance
    /// and (b) two arms (bf16 ceiling vs NVFP4) score on IDENTICAL contexts, isolating the
    /// quant-induced head/hidden-state mismatch from text drift.
    ///
    /// Per eval position p (context = tokens[0..=p], predecessor pairing as in spec decode):
    ///   draft_j  = chain token j from (tokens[p], h_{p-1}), then its own drafts — the exact
    ///              eager spec-decode chain (same mtp_head_forward_dev, same rope positions).
    ///   target_j = teacher-forced greedy pick for position p+1+j (argmax of the trunk logits
    ///              at forced context tokens[0..p+j]). For j==0 this equals live spec
    ///              acceptance; for j>=1 live verify would condition on the drafts, here it
    ///              conditions on the corpus — deterministic and arm-comparable by design.
    ///
    /// Returns (rows, bg): one (p, drafts[k], targets[k]) row per eval position (ascending p),
    /// plus the full teacher-forced greedy track bg (bg[i] = greedy pick for position i, i>=1)
    /// so harnesses can cross-check runs (e.g. different chunk sizes must give identical bg).
    ///
    /// `hdump`: when Some, every position's pre-output_norm trunk hidden (the exact rows the
    /// draft-KV fill pairs from) streams to the file as little-endian f32 [t_total, n_embd] —
    /// the head-distillation extraction (hqmtp): the ENGINE is the source of truth for trunk
    /// hiddens (HF torch reproductions of the hybrid trunk measured only ~0.5 greedy
    /// agreement vs this path — not usable as a training-data source).
    pub fn replay_acceptance(
        &self,
        e: &Engine,
        tokens: &[u32],
        k: usize,
        stride: usize,
        chunk: usize,
        mut hdump: Option<&mut std::fs::File>,
    ) -> Result<(Vec<(usize, Vec<u32>, Vec<u32>)>, Vec<u32>), Box<dyn std::error::Error>> {
        assert!(k >= 1 && stride >= 1 && chunk >= 2);
        let mtp = self
            .mtp
            .as_ref()
            .expect("replay_acceptance requires an MTP head");
        let n_vocab = self.output.out_features();
        let d_vocab = mtp
            .shared_head_head
            .as_ref()
            .unwrap_or(&self.output)
            .out_features();
        let n_embd = self.cfg.n_embd as usize;
        let t_total = tokens.len();
        assert!(t_total >= 8, "corpus too short ({t_total} tokens)");
        // STAGE-OWNED KV (lane/pp2-spec 2026-08-06) — see `new_session`. Door shut = `Cache::new`.
        let mut cache = crate::pp::new_cache(e, &self.cfg, t_total + k + 8)?;
        let mut scratch = MtpScratch::new(
            e,
            &self.cfg,
            t_total + k + 8,
            self.mtp.as_ref().and_then(|m| m.geom.as_ref()),
        )?;
        let (embd_qt, embd_rb) = self.embd.qt_and_row_bytes(n_embd);
        let embd_gpu = if spec_host_embd() {
            None
        } else {
            Some(
                self.embd_gpu
                    .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload")),
            )
        };
        let embd_dev = embd_gpu.map(|g| (g, embd_qt, embd_rb));

        // bg[i] = the trunk's greedy pick for position i under the forced context (i >= 1).
        let mut bg: Vec<u32> = vec![0; t_total + 1];
        let mut rows: Vec<(usize, Vec<u32>, Vec<u32>)> = Vec::new();
        let mut prev_last_h = e.zeros(n_embd)?; // predecessor hidden entering the chunk
        let mut seed_buf = e.zeros(n_embd)?;
        let mut preds_d = e.alloc_u32_zeroed(chunk)?;
        let nll_on = std::env::var("MEMRA_REPLAY_NLL").as_deref() == Ok("1");
        let (mut nll_sum, mut nll_cnt) = (0f64, 0u64);
        let mut s = 0usize;
        while s < t_total {
            let cend = (s + chunk).min(t_total);
            let tc = cend - s;
            let ch = &tokens[s..cend];
            // 1. forced trunk pass — verify path (decode-exact contract): all-column logits +
            //    the chunk's true hiddens.
            let (tl_d, vx) = self.decode_step_t_core(e, ch, s, &mut cache, embd_dev, None)?;
            for j in 0..tc {
                e.argmax_token_device_col(&tl_d, j, n_vocab, &mut preds_d, j)?;
            }
            let preds = e.dtoh_u32(&preds_d)?;
            for j in 0..tc {
                bg[s + j + 1] = preds[j];
            }
            // MEMRA_REPLAY_NLL=1: teacher-forced NLL/perplexity over the same forced pass — the
            // checkpoint-quality metric (position j's logits score the GOLD next token).
            if nll_on {
                let jmax = if cend < t_total { tc } else { tc - 1 }; // last pos has no gold next
                if jmax > 0 {
                    let ids: Vec<u32> = (0..jmax).map(|j| tokens[s + j + 1]).collect();
                    let rows: Vec<i32> = (0..jmax as i32).collect();
                    let idsd = e.htod_u32_v(&ids)?;
                    let rowsd = e.htod_i32(&rows)?;
                    let mut outd = e.zeros(jmax)?;
                    e.softmax_gather(&tl_d, n_vocab, &idsd, &rowsd, &mut outd, n_vocab, jmax, 1.0)?;
                    for pr in e.dtoh(&outd)? {
                        nll_sum += -((pr.max(1e-30)) as f64).ln();
                        nll_cnt += 1;
                    }
                }
            }
            if let Some(f) = hdump.as_deref_mut() {
                use std::io::Write;
                let host: Vec<f32> = e.dtoh(&vx)?;
                // bf16 round-to-nearest-even — f32 doubled the disk bill at bulk
                // extraction scale (20M tokens x 4096 = 320GB f32 vs 160GB bf16).
                let mut bytes = Vec::with_capacity(tc * n_embd * 2);
                for v in &host[..tc * n_embd] {
                    let b = v.to_bits();
                    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
                    bytes.extend_from_slice(&((r >> 16) as u16).to_le_bytes());
                }
                f.write_all(&bytes)?;
            }
            // CHAINLESS extraction (stride > corpus, the bulk-hdump mode): no chunk ever
            // drafts, so the draft-KV fills are pure waste — skip them (2 MTP-block passes
            // per token saved; the forced trunk pass + hdump is all the mode needs).
            let chainless = stride > t_total;
            if chainless {
                e.copy_view_into(
                    &mut prev_last_h,
                    0,
                    &vx.slice((tc - 1) * n_embd..tc * n_embd),
                    n_embd,
                )?;
                s = cend;
                continue;
            }
            // 2. TRUE predecessor-paired draft-KV fill for the chunk (row i carries h_{i-1};
            //    row s reads the previous chunk's last true hidden, zeros at corpus start).
            let mut vxs = e.zeros(tc * n_embd)?;
            e.copy_into(&mut vxs, 0, &prev_last_h, n_embd)?;
            if tc > 1 {
                e.copy_view_into(
                    &mut vxs,
                    n_embd,
                    &vx.slice(0..(tc - 1) * n_embd),
                    (tc - 1) * n_embd,
                )?;
            }
            scratch.set_len(e, s)?;
            self.mtp_kv_fill(e, mtp, ch, &vxs, s, &mut scratch, embd_dev)?;
            // 3. draft chains at sampled positions, DESCENDING: a chain reads only slots
            //    [0..p) (true fills) and appends at >= p; the next (smaller-p) chain's set_len
            //    truncates those approximate appends before they can ever be read.
            let ps: Vec<usize> = (s..cend)
                .filter(|p| *p >= 1 && *p % stride == 0 && *p + k <= t_total)
                .collect();
            for &p in ps.iter().rev() {
                scratch.set_len(e, p)?;
                if p == s {
                    e.copy_into(&mut seed_buf, 0, &prev_last_h, n_embd)?;
                } else {
                    e.copy_view_into(
                        &mut seed_buf,
                        0,
                        &vx.slice((p - 1 - s) * n_embd..(p - s) * n_embd),
                        n_embd,
                    )?;
                }
                let mut e_tok = tokens[p];
                let mut d_seed = e.clone_dtod(&seed_buf)?;
                let mut drafts: Vec<u32> = Vec::with_capacity(k);
                for j in 0..k {
                    let (dl_d, h_nextn) = self.mtp_head_forward_dev(
                        e,
                        mtp,
                        e_tok,
                        &d_seed,
                        &mut scratch,
                        p + 1 + j,
                        embd_dev,
                        None, // acceptance-oracle walk: no grammar
                    )?;
                    let tok_d = e.argmax_token_device(&dl_d, d_vocab)?;
                    let idx = e.dtoh_u32_one(&tok_d)?;
                    let d = match &mtp.d2t {
                        Some(map) => map[idx as usize],
                        None => idx,
                    };
                    drafts.push(d);
                    e_tok = d;
                    d_seed = h_nextn;
                }
                // targets may live in a LATER chunk's bg — resolved after the walk.
                rows.push((p, drafts, Vec::new()));
            }
            // 4. restore TRUE entries for the whole chunk (the next chunk's chains and fills
            //    expect scratch.len == cend with exact rows).
            scratch.set_len(e, s)?;
            self.mtp_kv_fill(e, mtp, ch, &vxs, s, &mut scratch, embd_dev)?;
            e.copy_view_into(
                &mut prev_last_h,
                0,
                &vx.slice((tc - 1) * n_embd..tc * n_embd),
                n_embd,
            )?;
            s = cend;
        }
        for (p, drafts, targets) in rows.iter_mut() {
            for j in 0..drafts.len() {
                targets.push(bg[*p + 1 + j]);
            }
        }
        rows.sort_by_key(|r| r.0);
        if nll_cnt > 0 {
            let mean = nll_sum / nll_cnt as f64;
            println!(
                "[replay-nll] tokens={nll_cnt} nll/token={mean:.5} ppl={:.4}",
                mean.exp()
            );
        }
        Ok((rows, bg))
    }
}

#[cfg(test)]
mod telem_tests {
    use super::{SpecTelemetry, SPEC_TELEM_POS};

    /// The worker's per-burst pattern: stash, accumulate, diff — the delta must isolate
    /// exactly the burst's contribution (pool-resumed sessions carry prior requests' counts).
    #[test]
    fn delta_isolates_burst_contribution() {
        let mut t = SpecTelemetry::default();
        // "previous request": 2 rounds of k=3, accepts 3 then 1.
        for (kr, na) in [(3usize, 3usize), (3, 1)] {
            t.rounds += 1;
            t.drafted += kr as u64;
            t.accepted += na as u64;
            for j in 0..kr { t.pos_drafted[j] += 1; }
            for j in 0..na { t.pos_accepted[j] += 1; }
        }
        let before = t;
        // "this burst": 1 round k=3, accepts 2.
        t.rounds += 1;
        t.drafted += 3;
        t.accepted += 2;
        for j in 0..3 { t.pos_drafted[j] += 1; }
        for j in 0..2 { t.pos_accepted[j] += 1; }
        let d = t.delta_since(&before);
        assert_eq!((d.rounds, d.drafted, d.accepted), (1, 3, 2));
        assert_eq!(&d.pos_drafted[..3], &[1, 1, 1]);
        assert_eq!(&d.pos_accepted[..3], &[1, 1, 0]);
        assert_eq!(d.pos_drafted[3..], [0; SPEC_TELEM_POS - 3]);
    }

    /// merge(delta) then merge(delta2) equals accumulating both — the per-model /metrics
    /// aggregation invariant.
    #[test]
    fn merge_accumulates_fieldwise() {
        let mut agg = SpecTelemetry::default();
        let mut d1 = SpecTelemetry { rounds: 2, drafted: 6, accepted: 4, ..Default::default() };
        d1.pos_drafted[0] = 2;
        d1.pos_accepted[0] = 2;
        let mut d2 = SpecTelemetry { rounds: 1, drafted: 3, accepted: 1, ..Default::default() };
        d2.pos_drafted[0] = 1;
        d2.pos_accepted[0] = 1;
        d2.pos_drafted[1] = 1;
        agg.merge(&d1);
        agg.merge(&d2);
        assert_eq!((agg.rounds, agg.drafted, agg.accepted), (3, 9, 5));
        assert_eq!(agg.pos_drafted[0], 3);
        assert_eq!(agg.pos_accepted[0], 3);
        assert_eq!(agg.pos_drafted[1], 1);
        assert_eq!(agg.pos_accepted[1], 0);
    }

    /// Wrong-snapshot diff saturates to zero instead of wrapping — the counters feed a
    /// public metrics surface and must never publish a u64-wrapped garbage value.
    #[test]
    fn delta_saturates_never_wraps() {
        let small = SpecTelemetry { rounds: 1, drafted: 2, accepted: 1, ..Default::default() };
        let big = SpecTelemetry { rounds: 5, drafted: 15, accepted: 9, ..Default::default() };
        let d = small.delta_since(&big);
        assert_eq!((d.rounds, d.drafted, d.accepted), (0, 0, 0));
    }
}

#[cfg(test)]
mod draft_graph_fallback_tests {
    use super::DraftGraphFallback;

    /// The Q2 contract, part (a): a fallback flip is LOUD — exactly once per flip.
    #[test]
    fn flip_is_loud_once_and_memoized_after() {
        let mut f = DraftGraphFallback::default();
        let line = f.mark_greedy("out of memory").expect("first flip must return the warn line");
        assert!(line.contains("WARN"), "flip line must be warn-level: {line}");
        assert!(line.contains("out of memory"), "flip line must carry the reason: {line}");
        assert!(f.greedy_failed());
        // re-marking an already-failed graph is the memoization: quiet, still failed.
        assert!(f.mark_greedy("out of memory").is_none());
        assert!(f.greedy_failed());
        // the two graphs' flags are independent (greedy flip leaves sampled capturable).
        assert!(!f.sampled_failed());
        let line_s = f.mark_sampled("capture unsupported").expect("sampled flip is its own flip");
        assert!(line_s.contains("sampled"), "sampled flip names itself: {line_s}");
        assert!(f.mark_sampled("capture unsupported").is_none());
    }

    /// The Q2 contract, part (b): resume-from-pool RESETS both flags (fresh capture chance),
    /// and says so exactly when there was something to reset.
    #[test]
    fn reset_on_resume_clears_flags_and_logs_once() {
        let mut f = DraftGraphFallback::default();
        // clean session: resume is silent, nothing to reset.
        assert!(f.reset_on_resume().is_none());
        f.mark_greedy("oom").unwrap();
        f.mark_sampled("oom").unwrap();
        let note = f.reset_on_resume().expect("a set flag must produce the reset note");
        assert!(note.contains("greedy+sampled"), "note names what was reset: {note}");
        assert!(!f.greedy_failed() && !f.sampled_failed(), "both flags cleared");
        // and the NEXT failure after a reset is a fresh flip — loud again.
        assert!(f.mark_greedy("oom again").is_some());
        let note2 = f.reset_on_resume().expect("greedy-only reset");
        assert!(note2.contains("(greedy)"), "single-flag note: {note2}");
    }

    /// Shape-change clears (dmask realloc / mask-shape mismatch / s_key change) stay silent —
    /// they precede a fresh capture attempt whose own failure re-flips loudly.
    #[test]
    fn shape_change_clears_are_silent() {
        let mut f = DraftGraphFallback::default();
        f.mark_greedy("oom").unwrap();
        f.clear_greedy();
        assert!(!f.greedy_failed());
        f.mark_sampled("oom").unwrap();
        f.clear_sampled();
        assert!(!f.sampled_failed());
        // after a silent clear there is nothing left for resume to report.
        assert!(f.reset_on_resume().is_none());
    }
}
