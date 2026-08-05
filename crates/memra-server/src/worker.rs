//! The single GPU worker thread + step-interleave scheduler (BASE-4, MEMRA-BUILD-MAP §4e).
//!
//! WHY a dedicated thread: the CUDA context is THREAD-AFFINE. `Engine` (and every `CudaStream` /
//! `CudaSlice` it owns) must only ever be touched from the one thread that created the context.
//! So we spawn ONE OS thread, build `Engine::new(0)` on it, load every `HybridModel` on it, and
//! never let an `Engine`/`Cache`/`CudaSlice` cross a thread boundary. Async HTTP handlers run on a
//! separate tokio runtime and submit work over an `mpsc` channel; each request carries a `tokio`
//! mpsc Sender back which the worker uses to stream tokens (and a final Done) to that one request.
//!
//! SCHEDULER LOOP: the worker holds a `Vec<Session>` of active generations. Each iteration it
//! round-robin steps EVERY active session by exactly ONE `decode_step` (one token of prefill OR
//! one decode token), samples, checks stop, streams the token text back on that session's channel,
//! and retires finished sessions. Queued admits fill empty slots up to `MAX_ACTIVE`. This is the
//! interleave: a long generation and a freshly-admitted one make forward progress in the same loop,
//! so the second produces tokens before the first finishes (not serialized end-to-end).

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

use cudarc::driver::CudaSlice;
use memra_engine::Engine;
use memra_engine::cache::Cache;
use memra_engine::decode::{GenParams, StopReason};
use memra_engine::hybrid::HybridModel;
use memra_engine::sampler::{Sampler, SamplerConfig};
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;

/// Max concurrently-active sessions in legacy round-robin mode (MEMRA_SERVE_BATCH=0).
/// Batched scheduling caps at MEMRA_MAX_SESSIONS (default 64). Admits beyond the cap queue (FIFO).
pub const MAX_ACTIVE: usize = 4;

/// `GenParams.max_new` sentinel: the request OMITTED `max_tokens` (gap-scan F2), so the
/// generation budget is CONTEXT-BOUNDED — session ctx minus prompt tokens, model-capped —
/// the OpenAI default-when-omitted semantics, never a silent 128-token truncation.
/// (`budget = max_new.min(room)` makes the sentinel safe everywhere downstream.)
pub const MAX_NEW_CTX_BOUNDED: usize = usize::MAX;

/// Per-tick prefill chunk cap: tokens primed per scheduler tick per session. Priming runs at
/// prefill throughput instead of tokenwise decode, while the per-tick cap keeps round-robin
/// latency for concurrent sessions bounded.
const PREFILL_TICK_T: usize = 1024;

/// A model loaded resident on the worker thread: weights + its own tokenizer + config snapshot.
struct LoadedModel {
    model: HybridModel,
    tok: Tokenizer,
    eos_id: u32,
    /// Loaded from a checkpoint DIRECTORY (safetensors/repack) rather than a GGUF file.
    /// Feeds ModelCaps::chat_ok: a dir checkpoint with no chat template 400s on chat
    /// requests (serve-st v1 honesty gate) instead of silently rendering fallback ChatML;
    /// GGUF models keep the historical ChatML fallback.
    from_dir: bool,
    /// Constrained-decoding grammar factory (llguidance TokTrie over this vocab). Built
    /// LAZILY on the first `response_format` request against this model — unconstrained
    /// serving never pays the vocab-trie build. `Err` = vocab unusable (kept, so every
    /// constrained request fails with the same clean message instead of rebuilding).
    constraints: std::cell::OnceCell<Result<crate::constrained::ConstraintFactory, String>>,
}

/// What the worker streams back to one request, over its per-request tokio mpsc channel.
#[derive(Debug, Clone)]
pub enum Event {
    /// One decoded token: the raw id + the incremental text delta (detokenized tail minus prefix).
    Token { id: u32, text: String },
    /// Terminal event: why we stopped + final token count + timing. `n_prompt` / `n_cached`
    /// are WORKER-TRUTH prompt accounting: total prompt tokens this session fed or resumed —
    /// the tokenized RENDERED prompt (tools block included when one was rendered; the
    /// text-prefix spec resume counts the actually-fed remainder) — and how many of those
    /// came from a cache (continuation pool, spec resume, or the cross-request prefix cache)
    /// instead of being computed — the OpenAI `usage.prompt_tokens_details.cached_tokens`
    /// source. ONE source of truth: both counts come off the same rendered-prompt token ids.
    Done { stop_reason: String, n_tokens: usize, n_prompt: usize, n_cached: usize, elapsed_s: f64 },
    /// The request could not start (bad model name, ctx full at admit, etc).
    Error(String),
}

/// A generation request submitted by an HTTP handler to the worker.
pub struct Request {
    pub model: String,
    pub prompt_ids: Vec<u32>,   // already tokenized? no — worker tokenizes (it owns the Tokenizer)
    pub prompt_text: String,
    pub chat: bool,
    pub chat_turns: Vec<memra_tokenizer::chat::Turn>,
    /// Tool schemas pre-serialized (client key order preserved) for the template's <tools> block.
    pub tools_json: Vec<String>,
    pub think: memra_tokenizer::chat::ThinkMode,
    pub params: GenParams,
    pub sampler_cfg: SamplerConfig,
    pub stop_strings: Vec<String>,
    pub trace_id: Option<String>,
    /// PC-ISO cache namespace (lane/pc-iso, 2026-08-02): the tenant isolation salt for
    /// EVERY cross-request KV reuse tier (prefix cache, continuation pool, spec pool) —
    /// the vLLM `cache_salt` design. Derived by the HTTP layer (request `cache_salt`
    /// field; "" = the default single-tenant namespace, byte-identical to pre-PC-ISO).
    pub cache_ns: String,
    /// SESSION AFFINITY explicit tier (lane/session-affinity, 2026-08-05): the client's own
    /// name for this conversation (`session_id`/`user` body field, or the `x-session-id`
    /// header — see `crate::affinity_key`). Some(id) nominates that conversation's parked
    /// session directly; None falls back to the implicit structural fingerprint. Either way
    /// the resume decision is the exact token diff (`affinity_match`), scoped to this
    /// request's own (model, cache_ns) pool.
    pub affinity: Option<String>,
    /// yield lane (x-lane header; default interactive). Drives admission + prefill budgets
    /// (lane/dl-metering QoS gate, ported 2026-08-02 — the metering half stayed behind).
    pub lane: crate::lanes::Lane,
    /// Constrained decoding (`response_format` json_object/json_schema): the parsed
    /// grammar spec. None = unconstrained — the request takes the exact legacy path
    /// (no factory, no matcher, no masking branch).
    pub grammar: Option<crate::constrained::GrammarSpec>,
    /// per-request stream back to the handler. tokio mpsc so the async side can await it.
    pub tx: tokio::sync::mpsc::UnboundedSender<Event>,
}

/// Chat-template capabilities probed from a loaded model's template at spawn time — the
/// HTTP layer rejects `tools` on models whose template has no tools branch BEFORE the
/// request reaches the worker, and arms the tool-call parser's think gate.
/// Plus the /v1/models metadata surface (serve-tail lane, 2026-08-04): trained context,
/// tokenizer family, chat-template family — worker truth captured once at spawn so the
/// HTTP layer never invents values (unknown = 0/""/None -> honest nulls in the route).
#[derive(Debug, Clone, Default)]
pub struct ModelCaps {
    /// template carries the qwen-class `<tools>` branch (tools + tool_response rendering).
    pub tools_branch: bool,
    /// template appends a `<think>` tail on the generation prompt (qwen think class).
    pub qwen_think: bool,
    /// template has the `enable_thinking` switch (ThinkMode::NoThink is honored).
    pub think_switch: bool,
    /// chat requests are honest against this model: it has a chat template, OR it is a
    /// GGUF (which keeps the historical plain-ChatML fallback). A safetensors/repack DIR
    /// checkpoint without a template 400s on /v1/chat/completions (serve-st v1 honesty
    /// gate) instead of silently rendering a format the model was never trained on.
    pub chat_ok: bool,
    /// model's trained context length (config; 0 = unknown) — /v1/models `context_length`.
    pub context_length: usize,
    /// tokenizer family (the GGUF/HF pre-tokenizer name, e.g. "qwen2"; "" = unknown).
    pub tokenizer: String,
    /// chat-template family ("chatml" / "gemma"); None = no template or unrecognized.
    pub instruct_type: Option<String>,
}

/// Control messages into the worker. Currently just generation requests; /models and /health are
/// served from the cached model-name list captured at spawn (no need to round-trip the worker).
pub enum Cmd {
    Generate(Box<Request>),
}

/// Serving counters + engine-truth step latency, published every 32nd tick.
#[derive(Clone, Default)]
pub struct Metrics {
    pub admitted: u64,
    pub completed: u64,
    pub tokens_out: u64,
    pub step_p50_ms: f32,
    pub step_p99_ms: f32,
    /// worker-truth prompt accounting: total prompt tokens admitted, and how many of
    /// those were served from a cache (continuation pool / spec resume / prefix cache).
    pub prompt_tokens_in: u64,
    pub cached_tokens_in: u64,
    /// cross-request prefix cache state (hits/entries/resident bytes).
    pub prefix_hits: u64,
    pub prefix_entries: u64,
    pub prefix_bytes: u64,
    /// per-lane QoS counters [interactive, judge, harvest] — the x-lane yield gate
    /// (/yield/metrics, sidecar-compatible shape; lane/dl-metering QoS extraction).
    pub lane_admitted: [u64; 3],
    pub lane_shed: [u64; 3],
    pub lane_completed: [u64; 3],
    pub lane_tokens: [u64; 3],
    pub batch_size_last: usize,
}
pub type SharedMetrics = std::sync::Arc<std::sync::Mutex<Metrics>>;

/// Windowed percentile over decode-step latencies (ms) — the interactive SLO sensor.
/// Engine ground truth: the worker records the wall time of each batched decode tick that
/// advanced at least one interactive session — that IS the client-visible TPOT for that
/// tick. (Shared with out-of-process controllers via the memra-lanes crate.)
use crate::lanes::StepStats;

/// Live per-session state on the worker thread. One `Session` per in-flight generation.
/// Holds the per-session `Cache` (model-specific dims — NO sharing between sessions, which is what
/// makes the concurrent streams byte-identical to isolated runs) and per-session `Sampler`.
/// KV PREFIX REUSE (append-only continuation): retired sessions park (fed tokens, Cache,
/// last_logits) here; a new request whose prompt EXACTLY EXTENDS a parked `fed` sequence takes
/// the Cache and primes only the suffix. Correct by construction for hybrid models: the
/// recurrent (conv/ssm) state in the Cache is the state AFTER the last fed token — the exact
/// resume point for an append-only continuation. NO arbitrary-prefix truncation is attempted
/// (GDN state cannot roll back without checkpoints); a non-extending prompt takes the cold path.
/// NOTE chat-template callers: templates that rewrite history (e.g. stripping think blocks from
/// prior assistant turns) break exact-extension and simply miss the pool — raw `prompt_ids`
/// callers (agent loops) always hit. Pool: at most MEMRA_REUSE_POOL entries per model, LRU.
struct ReuseEntry {
    fed: Vec<u32>,
    cache: Cache,
    last_logits: Vec<f32>,
    cap: usize,
}
/// SPEC-session reuse (2026-07-05): a retired spec session parks WHOLE (trunk cache + draft
/// scratch + committed + next_pred). A new greedy request whose prompt exactly extends
/// `committed` resumes it — turn N+1 primes only the suffix (or nothing, the continuation
/// burst). Same exact-extension rule as ReuseEntry; the session-gate oracle covers this path.
struct SpecReuseEntry {
    sess: memra_engine::spec::SpecSession,
    /// detok(committed) — TEXT-level prefix matching (2026-07-06). Token-level starts_with
    /// missed ~50% of chat turn boundaries (detok->retok BPE merges differ at the seam). Text
    /// matching resumes whenever the new prompt string literally extends the parked
    /// conversation; only the remainder is tokenized (no BOS). Same acceptable-divergence class
    /// as llama serve's cache_prompt: the suffix's boundary tokenization may differ from a cold
    /// full-retok — committed tokens stay authoritative, spec==greedy exactness is untouched.
    committed_text: String,
}
/// Parked-cache pool cap per model (MEMRA_REUSE_POOL, default 2 — each parked cache holds
/// its full KV, ~119MB at ctx 8192 on the 9B, so the cap is a VRAM budget knob).
/// KNOWN CASCADE at the default (pinned 2026-07-31): park-on-retire evicts LRU
/// unconditionally, so a SEQUENTIAL multi-turn workload of N>cap sessions evicts each
/// next request's entry one step ahead of its arrival (0/N resumes), while the same
/// requests arriving CONCURRENTLY all probe the intact pool first (hit). Raise the cap
/// to >= the expected concurrent-session count when VRAM allows.
fn reuse_pool_per_model() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("MEMRA_REUSE_POOL").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(2))
}
/// Minimum parked prefix worth reusing (below this, cold prime is cheaper than bookkeeping).
const REUSE_MIN_PREFIX: usize = 16;

/// F5 (spec-pool thrash, 2026-08-05 — research/specpool-20260804): learned per-model
/// spec-session sizing, process-lifetime (VRAM geometry is static per server run;
/// a restart re-probes). Two lessons the worker remembers so pool misses stop
/// re-paying doomed multi-GB cudaMalloc walks every turn:
#[derive(Default)]
struct SpecSizing {
    /// Models OBSERVED VRAM-tight: a spec-session alloc failed while a parked pool
    /// entry existed. Later misses on these models evict the (dead-weight) pool
    /// BEFORE allocating — the same eviction the failure path forced anyway, minus
    /// the failed alloc + full realloc churn (the owner's live thrash: "spec pool
    /// evicted (1) after alloc failure" once per request, every turn a miss).
    /// Rigs where ghost + new session both fit never set the flag and keep the
    /// parked entry's resume value.
    evict_first: std::collections::HashSet<String>,
    /// model -> largest ctx ask known to fit after a GENUINE (empty-pool) alloc
    /// failure — the right-size ladder's landing point. Later asks start here
    /// instead of re-laddering; never exceeds the request's own ctx_cap.
    learned_ctx: HashMap<String, usize>,
}
/// Right-size ladder slack: a shrunken spec session's cap must cover
/// prompt + budget + this, so the burst-loop ContextFull guard
/// (`committed + k + 3 >= cache_max_ctx`) can NEVER fire before MaxNew — a
/// shrunken session emits exactly the tokens a full-size one would (the F5
/// exactness contract: pool sizing is pure perf). Worst case per burst:
/// committed reaches prompt + budget + overshoot (up to k accepted drafts past
/// max_new) + a carried pending, and the guard adds k + 3; k = MEMRA_SPEC_K <= 8
/// in practice — 64 covers it with margin at ~2MB of KV. Requests whose budget
/// already spans the whole ctx_cap (max_tokens omitted) cannot shrink and keep
/// the legacy tokenwise fallback on failure.
const SPEC_SHRINK_SLACK: usize = 64;
/// Ladder transient reserve: after a landing, this much VRAM must still be
/// PROBE-allocatable for forward-pass transients — prime chunk slabs (~140MB
/// apiece at MEMRA_PRIME_CHUNK=2048; the serve-script's measured 36.5k probe
/// passed with 1.3GB free) and the FA dequant workspace. Some transients live
/// on PANICKING lazy paths (expect()), so a session that "fits" with zero
/// headroom kills the worker on its first prefill (observed: ladder landed
/// 65536, embed-table upload OOM panic — research/specpool-20260804/
/// server-ladder-miss.log; the embed table is made resident fallibly at landing
/// for that reason). The probe is alloc-and-drop, not a mem_get_info read: the
/// async pool's pinned release threshold keeps freed blocks cached and
/// invisible to free-VRAM queries. A landing that can't clear the probe is
/// DROPPED and the ladder keeps shrinking; if even `need` can't, the request
/// takes the legacy tokenwise fallback (whose own alloc failure is a clean
/// quoted error — the pre-fix behavior, never a panic).
const SPEC_SHRINK_RESERVE: usize = 1536 << 20; // 1.5 GiB

/// PC-ISO pool key (lane/pc-iso, 2026-08-02): every cross-request reuse pool — the prefix
/// cache, the continuation pool, and the spec pool — keys on (model, cache namespace), not
/// model alone. The namespace is the request's `cache_salt` (vLLM cache_salt design, PR
/// #17045): a lookup only ever scans its own (model, ns) pool, so no token-prefix match can
/// cross a trust boundary, and the `cached_tokens` billing field can only reveal the
/// caller's own namespace's history (the CacheProbe/PROMPTPEEK mitigation —
/// research/cache-tools-20260802/REPORT.md §1.4/§4). "" is the default single-tenant
/// namespace: no salt supplied = today's behavior, byte-for-byte.
type PoolKey = (String, String);

/// Log suffix for a pool key's namespace: silent for the default "" namespace (default-path
/// log lines stay byte-identical to pre-PC-ISO), quoted otherwise.
fn ns_suffix(ns: &str) -> String {
    if ns.is_empty() { String::new() } else { format!(", ns {ns:?}") }
}

// ---------------- SESSION AFFINITY (lane/session-affinity, 2026-08-05) ----------------
//
// THE PROBLEM (receipts: research/specpool-20260804/RESULTS.md). The spec pool resumes a
// parked session only when the new prompt EXACTLY EXTENDS it — token-prefix, or (since
// 2026-07-06) text-prefix. Real agent clients rewrite conversation history between turns:
// the owner's client strips `<think>` blocks out of PRIOR assistant turns before re-sending,
// so turn N's prompt is NOT a prefix-extension of turn N-1's committed text. Both probes
// miss, the parked ~4GB session is discarded as dead weight, and every turn re-primes the
// whole growing conversation (11k-14k tokens ~= 3s TTFT vs llama's 0.19s).
//
// Affinity closes that gap by answering a DIFFERENT question than the prefix probes: not
// "does this prompt extend that session's bytes?" but "is this the SAME CONVERSATION as
// that session?". Once a candidate is nominated by identity, the resume decision is made by
// an EXACT token-level diff (see `AffinityMatch`) — identity nominates, bytes decide. That
// split is the whole safety argument: a fingerprint collision can only ever nominate a
// candidate whose committed tokens are then compared exactly, so it can cost a wasted probe,
// never a wrong resume.
//
// TWO TIERS.
//   (a) EXPLICIT (`AffinityKey::Explicit`) — the client names its conversation. Accepted from
//       two conventions, both documented in docs/API.md:
//         * `session_id` / `user` request-body fields (OpenAI's `user` is the field real
//           clients already send; `session_id` is the explicit spelling),
//         * the `x-session-id` request header (the convention vLLM/TGI-adjacent proxies use).
//       Body beats header when both appear (the body is the caller's own statement of
//       identity; a header can be injected by an intermediary).
//   (b) IMPLICIT (`AffinityKey::Fingerprint`) — nothing named, so identity is STRUCTURAL: a
//       hash of the conversation's SHAPE that is invariant under exactly the rewrite class we
//       need to survive. See `conversation_fingerprint`.
//
// TENANT SCOPE. Affinity is stored per `PoolKey = (model, cache_ns)`, so an affinity key can
// only ever nominate a session inside its own PC-ISO namespace — affinity adds NO new
// cross-tenant reach beyond what the existing pools already have. (The api-keys lane's
// TenantCtx is not on this branch; when it merges, its per-key namespace derivation flows
// into `cache_ns` and affinity inherits the boundary for free.)

/// Prefix/suffix token window hashed per conversation segment (see `conversation_fingerprint`).
/// Small enough that a rewritten segment BODY doesn't perturb the hash, large enough that
/// distinct segments don't collide: the head pins "which turn is this" (role marker + opening
/// words) and the tail pins the segment's end boundary.
const FP_WINDOW: usize = 8;
/// Minimum segments before an implicit fingerprint is trusted to name a conversation. A
/// one-or-two-segment prompt is a generic opener (a bare system prompt shared by every fresh
/// conversation); nominating on it would cross-link unrelated conversations into one session.
const FP_MIN_SEGMENTS: usize = 3;

/// FNV-1a over a token stream — a stable, allocation-free 64-bit mix. (Not a cryptographic
/// hash and does not need to be: a collision costs one wasted exact-diff probe, never a
/// wrong resume, and the pool it indexes is already tenant-scoped.)
fn fnv1a(seed: u64, toks: &[u32]) -> u64 {
    let mut h = seed;
    for &t in toks {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Structural fingerprint of a conversation: the CHAIN of per-segment boundary hashes, one
/// entry per conversation segment in order, each hashing only that segment's (head window,
/// tail window) — never its interior.
///
/// WHY A CHAIN, NOT ONE HASH. Turn N+1 of a conversation has strictly MORE segments than turn
/// N (the previous answer plus a new user turn were appended), so a single whole-conversation
/// digest can never match across turns. Identity is therefore a PREFIX relation over the
/// chain (`fingerprint_affinity`): the parked session's conversation is an ancestor of this
/// request's when their chains share a long-enough leading run.
///
/// WHY BOUNDARY WINDOWS. The rewrite class we must tolerate mutates the INTERIOR of prior
/// assistant segments (a stripped `<think>` block is deleted text inside a turn). Hashing only
/// each segment's first and last few tokens leaves those edits invisible while still separating
/// genuinely different segments. Where a rewrite reaches into a head window too (a `<think>`
/// tag can sit right after the role marker), the chain degrades GRACEFULLY instead of failing:
/// that one segment's hash changes, the shared leading run simply ends earlier, and the
/// candidate is still nominated on the stable prefix (system prompt + early turns, which no
/// client rewrites). Nomination only has to be a good guess — `affinity_match` decides on bytes.
///
/// SEGMENTATION on the raw-prompt path. The owner's client renders the chat template
/// CLIENT-side and posts raw `/v1/completions`, so there is no `chat_turns` structure to walk:
/// the worker sees one flat token stream. Segments are recovered from the stream itself by
/// splitting at the template's own turn-marker tokens (the tokenizer's control tokens — exactly
/// what a chat template emits at every turn boundary: `<|im_start|>`/`<|im_end|>` and friends).
/// The implicit tier therefore works identically for client-rendered raw prompts and for
/// server-rendered `/v1/chat/completions` traffic.
///
/// `is_boundary(tok)` reports whether a token is a template turn marker. `drop_live` excludes
/// the trailing segment (a REQUEST's last segment is the turn being generated — new every turn
/// by construction, so it must not contribute to identity; a PARKED session's committed stream
/// has no such live tail and keeps every segment).
fn conversation_fingerprint(
    toks: &[u32],
    is_boundary: &dyn Fn(u32) -> bool,
    drop_live: bool,
) -> Vec<u64> {
    // Split into segments at boundary tokens. The boundary token itself joins the segment it
    // opens, so a segment's head window carries its own role marker.
    let mut segs: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    for (i, &t) in toks.iter().enumerate() {
        if is_boundary(t) && i > start {
            segs.push((start, i));
            start = i;
        }
    }
    if start < toks.len() {
        segs.push((start, toks.len()));
    }
    if drop_live && !segs.is_empty() {
        segs.pop();
    }
    segs.iter()
        .map(|&(lo, hi)| {
            let seg = &toks[lo..hi];
            let head = &seg[..FP_WINDOW.min(seg.len())];
            let tail = &seg[seg.len().saturating_sub(FP_WINDOW)..];
            fnv1a(fnv1a(0xcbf29ce484222325, head), tail)
        })
        .collect()
}

/// Length of the leading run two fingerprint chains share. `>= FP_MIN_SEGMENTS` is the
/// nomination bar: below it the shared run is a generic opener (a bare system prompt is
/// byte-identical across every fresh conversation with the same client), and nominating on it
/// would cross-link unrelated conversations. Markerless raw prompts produce a 1-segment chain
/// and so can never clear the bar — non-chat callers keep the plain prefix probes untouched.
fn fingerprint_affinity(a: &[u64], b: &[u64]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Verdict of the EXACT token diff run against an affinity-nominated parked session. Identity
/// nominated the candidate; this decides — on bytes — whether resuming it is EXACT.
///
/// THE EXACTNESS CONTRACT. A resumed session must emit BYTE-IDENTICAL output to a fresh full
/// prime of the same request. The committed tokens in the parked caches are authoritative
/// state: whatever they are, the caches hold exactly their KV/recurrent state. So resuming is
/// exact iff the new prompt begins with the session's ENTIRE committed sequence — then the
/// caches are precisely "the state after the prompt's first `committed.len()` tokens" and only
/// the remaining suffix needs priming. Any DIVERGENCE inside the committed range means the
/// caches hold state for tokens this request does not have, and no amount of suffix priming
/// can repair that (hybrid GDN recurrent state is mutated in place and has no per-position
/// index to truncate). There is one legal repair — roll the session back to the divergence
/// point — and it requires a checkpoint AT that boundary, which a parked session does not
/// carry. So divergence inside the committed range is a full re-prime, always. Correctness
/// first; the affinity win comes from the (dominant) case where the rewrite touches only text
/// the session has not committed yet.
#[derive(PartialEq, Eq, Debug)]
enum AffinityMatch {
    /// The prompt begins with the session's entire committed sequence: resume, prime the
    /// `suffix_from` tail only. (`suffix_from == prompt.len()` = pure continuation burst.)
    Exact { suffix_from: usize },
    /// The prompt diverges from the committed tokens at this index: the parked caches hold
    /// state for tokens this request does not have. Full re-prime.
    Diverged { at: usize },
}

/// Exact token-level diff of a request's prompt against a parked session's committed tokens.
/// The ONLY authority on whether an affinity-nominated session may be resumed.
fn affinity_match(prompt: &[u32], committed: &[u32]) -> AffinityMatch {
    let n = committed.len().min(prompt.len());
    for i in 0..n {
        if prompt[i] != committed[i] {
            return AffinityMatch::Diverged { at: i };
        }
    }
    if prompt.len() < committed.len() {
        // The prompt is a strict PREFIX of committed: the session has generated past what
        // this request contains (a client that dropped its own tail, or a re-issued earlier
        // turn). The caches hold extra committed rows with no boundary checkpoint to trim
        // them at — treat as divergence at the prompt's end.
        return AffinityMatch::Diverged { at: prompt.len() };
    }
    AffinityMatch::Exact { suffix_from: committed.len() }
}

// ---------------- CROSS-REQUEST PREFIX CACHE (lane/prompt-cache, 2026-08-02) ----------------
//
// The continuation pool above only serves a prompt that EXACTLY EXTENDS a retired session's
// whole fed sequence (prompt + generation) — a NEW session that merely shares a system-prompt
// prefix with earlier traffic always misses. The prefix cache closes that gap: entries are
// compact device copies of primed state at a TOKEN boundary, keyed by the exact token-id
// prefix, and are REUSABLE (a hit deep-copies the entry into the new session's cache — one
// marketplace system prompt serves any number of sessions, unlike the move-out pool).
//
// WHY snapshots, not truncation: hybrid models (qwen35-class GDN) carry recurrent conv/ssm
// state that cannot roll back to an arbitrary shorter prefix, so a longer cache can never be
// trimmed to the shared prefix. Instead the state is captured AT the boundary while a fresh
// session primes:
//   - SEED: a cold session's full prompt is inserted at prefill-done (before any decode).
//   - LCP SPLIT (the learning step): a cold miss whose prompt shares >= PREFIX_CACHE_MIN_TOKENS
//     tokens with an existing entry splits its own prime at the longest-common-prefix point,
//     snapshots there, then continues — request 3+ of a shared-system-prompt pattern hits.
//
// EXACTNESS CONTRACT (docs/SERVING.md "Prompt caching"): an entry stores the KV/recurrent
// bytes from WHATEVER prime config ran (single, chunked, or concat batch-prime); decode from
// those bytes is deterministic, so serving a hit is bit-identical to the run that computed the
// prefix. Cross-config comparisons (a cached-hit stream vs a whole-prompt fresh prime) inherit
// the documented batched-prime near-tie first-token law — same class, not a new one.
//
// VRAM: entries compete with session KV under MEMRA_PREFIX_CACHE_MB (default 256; 0 disables),
// LRU-evicted, per-model keyed; a failed session-cache alloc evicts the whole cache and
// retries (headroom discipline — sessions always win over the cache).
//
// POLICY: spec sessions bypass the prefix cache entirely (SpecSession owns trunk + draft
// caches; restoring a trunk-only prefix would leave draft state unprimed). The spec tier keeps
// its own continuation pool; the prefix cache serves the batched bulk tier. Legacy round-robin
// mode (MEMRA_SERVE_BATCH=0) also bypasses.
//
// ISOLATION (PC-ISO, lane/pc-iso 2026-08-02): pools key on (model, cache namespace) — see
// PoolKey. Same-namespace traffic shares entries exactly as before; requests carrying
// different `cache_salt` values never see each other's prefixes, in either direction.

/// Prefixes shorter than this are not worth VRAM + copy bookkeeping (also keeps the bare
/// chat-template header — common to every request of a model — out of the cache).
const PREFIX_CACHE_MIN_TOKENS: usize = 64;

/// MEMRA_PREFIX_CACHE_MB (default 256): resident byte budget for the prefix cache. 0 = off.
fn prefix_cache_budget_bytes() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("MEMRA_PREFIX_CACHE_MB").ok()
            .and_then(|v| v.parse::<usize>().ok()).unwrap_or(256)
            .saturating_mul(1024 * 1024)
    })
}

/// Batched scheduling on? (read once — mirrors the run-loop static; the prefix cache only
/// engages in batched mode, the default.)
fn serve_batching() -> bool {
    static B: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("MEMRA_SERVE_BATCH").map(|v| v != "0").unwrap_or(true))
}

/// One full-attn layer's cached prefix bytes: exactly `len` tokens of quantized K/V.
struct PrefixPlane {
    k: CudaSlice<u8>,
    v: CudaSlice<u8>,
    len: usize,
}

/// A cached token-prefix: per-layer KV byte copies + recurrent conv/ssm copies + the logits
/// row AT the boundary (empty-suffix resumes sample from it, same as the continuation pool).
struct PrefixEntry {
    toks: Vec<u32>,
    kv: Vec<Option<PrefixPlane>>,
    conv: Vec<Option<CudaSlice<f32>>>,
    ssm: Vec<Option<CudaSlice<f32>>>,
    pos: usize,
    last_logits: Vec<f32>,
    bytes: usize,
    last_use: Instant,
}

#[derive(Default)]
struct PrefixCache {
    /// per-(model, namespace) entry pools (KV geometry/format is per model; the namespace
    /// is the PC-ISO trust boundary — the equality check is part of the map key, so the
    /// default path pays nothing beyond hashing "" alongside the model id).
    entries: HashMap<PoolKey, Vec<PrefixEntry>>,
    total_bytes: usize,
    hits: u64,
    misses: u64,
    inserts: u64,
    evictions: u64,
    hit_tokens: u64,
}

impl PrefixCache {
    fn lcp(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    fn n_entries(&self) -> usize {
        self.entries.values().map(|p| p.len()).sum()
    }

    /// Longest entry whose token key exactly prefixes `prompt` (floor PREFIX_CACHE_MIN_TOKENS).
    /// Only the caller's own (model, namespace) pool is scanned — cross-namespace entries
    /// are structurally unreachable (PC-ISO).
    fn lookup(&self, key: &PoolKey, prompt: &[u32]) -> Option<usize> {
        let pool = self.entries.get(key)?;
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in pool.iter().enumerate() {
            let n = e.toks.len();
            if n >= PREFIX_CACHE_MIN_TOKENS && n <= prompt.len() && prompt[..n] == e.toks[..]
                && best.is_none_or(|(_, bn)| n > bn)
            {
                best = Some((i, n));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Longest common prefix of `prompt` with ANY entry (the LCP-split learning signal).
    fn best_lcp(&self, key: &PoolKey, prompt: &[u32]) -> usize {
        self.entries.get(key)
            .map(|pool| pool.iter().map(|e| Self::lcp(&e.toks, prompt)).max().unwrap_or(0))
            .unwrap_or(0)
    }

    /// Is any entry (>= min tokens) already a full prefix of `prompt`? (seed dedupe)
    fn has_covering(&self, key: &PoolKey, prompt: &[u32]) -> bool {
        self.entries.get(key).is_some_and(|pool| pool.iter().any(|e| {
            let n = e.toks.len();
            n >= PREFIX_CACHE_MIN_TOKENS && n <= prompt.len() && prompt[..n] == e.toks[..]
        }))
    }

    fn has_key(&self, key: &PoolKey, toks: &[u32]) -> bool {
        self.entries.get(key).is_some_and(|pool| pool.iter().any(|e| e.toks[..] == *toks))
    }

    /// Insert (exact-key deduped per namespace) + LRU-evict back under MEMRA_PREFIX_CACHE_MB.
    /// The LRU budget stays GLOBAL across namespaces (VRAM is one resource); only visibility
    /// is namespaced.
    fn insert(&mut self, key: &PoolKey, e: PrefixEntry, why: &str) {
        let budget = prefix_cache_budget_bytes();
        if e.bytes > budget {
            eprintln!("[prefix-cache] skip {why} insert: entry {:.1}MB > budget {:.0}MB",
                      e.bytes as f64 / 1e6, budget as f64 / 1e6);
            return;
        }
        if self.has_key(key, &e.toks) {
            return; // identical key raced in (concurrent learners) — first one wins
        }
        self.total_bytes += e.bytes;
        self.inserts += 1;
        eprintln!("[prefix-cache] insert ({why}): {} tokens, {:.1}MB (resident {:.1}MB / {:.0}MB, model {}{})",
                  e.toks.len(), e.bytes as f64 / 1e6,
                  self.total_bytes as f64 / 1e6, budget as f64 / 1e6,
                  key.0, ns_suffix(&key.1));
        self.entries.entry(key.clone()).or_default().push(e);
        while self.total_bytes > budget {
            let mut victim: Option<(PoolKey, usize, Instant)> = None;
            for (k, pool) in &self.entries {
                for (i, e) in pool.iter().enumerate() {
                    if victim.as_ref().is_none_or(|&(_, _, t)| e.last_use < t) {
                        victim = Some((k.clone(), i, e.last_use));
                    }
                }
            }
            let Some((k, i, _)) = victim else { break };
            let dead = self.entries.get_mut(&k).map(|p| p.remove(i));
            if let Some(dead) = dead {
                self.total_bytes = self.total_bytes.saturating_sub(dead.bytes);
                self.evictions += 1;
                eprintln!("[prefix-cache] evict (LRU): {} tokens, {:.1}MB (model {}{})",
                          dead.toks.len(), dead.bytes as f64 / 1e6, k.0, ns_suffix(&k.1));
            }
        }
    }

    /// Drop everything (session cache alloc failed — sessions win over the cache).
    fn evict_all(&mut self) -> usize {
        let n = self.n_entries();
        self.entries.clear();
        self.total_bytes = 0;
        self.evictions += n as u64;
        n
    }
}

/// Deep-copy the primed prefix state OUT of a live session cache into a compact entry.
/// All copies are stream-ordered on the engine worker stream (the CUDA owner thread), so no
/// sync is needed against the prime that produced the bytes or the decode that follows.
fn prefix_snapshot(
    engine: &Engine,
    cache: &Cache,
    toks: &[u32],
    last_logits: &[f32],
) -> Result<PrefixEntry, Box<dyn std::error::Error>> {
    let n = cache.kv.len();
    let mut kv = Vec::with_capacity(n);
    let mut conv = Vec::with_capacity(n);
    let mut ssm = Vec::with_capacity(n);
    let mut bytes = 0usize;
    for il in 0..n {
        match &cache.kv[il] {
            Some(l) => {
                let kb = l.len * l.k_tok_bytes;
                let vb = l.len * l.v_tok_bytes;
                let mut k = engine.alloc_u8(kb.max(1))?;
                let mut v = engine.alloc_u8(vb.max(1))?;
                if kb > 0 { engine.copy_u8_into(&mut k, 0, &l.k, kb)?; }
                if vb > 0 { engine.copy_u8_into(&mut v, 0, &l.v, vb)?; }
                bytes += kb + vb;
                kv.push(Some(PrefixPlane { k, v, len: l.len }));
            }
            None => kv.push(None),
        }
        match &cache.recur[il] {
            Some(r) => {
                conv.push(Some(engine.clone_dtod(&r.conv_state)?));
                ssm.push(Some(engine.clone_dtod(&r.ssm_state)?));
                bytes += (r.conv_state.len() + r.ssm_state.len()) * 4;
            }
            None => {
                conv.push(None);
                ssm.push(None);
            }
        }
    }
    Ok(PrefixEntry {
        toks: toks.to_vec(),
        kv,
        conv,
        ssm,
        pos: cache.pos,
        last_logits: last_logits.to_vec(),
        bytes,
        last_use: Instant::now(),
    })
}

/// Deep-copy an entry INTO a freshly allocated session cache: KV bytes at [0..len), per-layer
/// len + device len mirror, recurrent conv/ssm state, pos. The ssm ping-pong spare and
/// last_logits_dev stay as allocated (scratch — overwritten before any read).
fn prefix_restore(
    engine: &Engine,
    cache: &mut Cache,
    e: &PrefixEntry,
) -> Result<(), Box<dyn std::error::Error>> {
    if cache.kv.len() != e.kv.len() {
        return Err(format!("prefix entry layer count {} != cache {}", e.kv.len(), cache.kv.len()).into());
    }
    for il in 0..cache.kv.len() {
        match (cache.kv[il].as_mut(), &e.kv[il]) {
            (Some(dst), Some(src)) => {
                let kb = src.len * dst.k_tok_bytes;
                let vb = src.len * dst.v_tok_bytes;
                if kb > 0 { engine.copy_u8_into(&mut dst.k, 0, &src.k, kb)?; }
                if vb > 0 { engine.copy_u8_into(&mut dst.v, 0, &src.v, vb)?; }
                dst.len = src.len;
                engine.set_i32_one(&mut dst.len_d, src.len as i32)?;
            }
            (None, None) => {}
            _ => return Err(format!("prefix entry layer {il} kind mismatch").into()),
        }
        match (cache.recur[il].as_mut(), &e.conv[il], &e.ssm[il]) {
            (Some(dst), Some(c), Some(s)) => {
                engine.copy_into(&mut dst.conv_state, 0, c, c.len())?;
                engine.copy_into(&mut dst.ssm_state, 0, s, s.len())?;
            }
            (None, None, None) => {}
            _ => return Err(format!("prefix entry recur {il} mismatch").into()),
        }
    }
    cache.pos = e.pos;
    Ok(())
}

/// Snapshot + insert the session's CURRENT primed state (fed tokens, boundary logits).
/// No-op when the session cannot serve an empty-suffix resume (no host logits yet).
fn prefix_insert_from_session(engine: &Engine, px: &mut PrefixCache, s: &Session, why: &str) {
    let Some(cache) = s.cache.as_ref() else { return };
    if s.last_logits.is_empty() {
        return;
    }
    match prefix_snapshot(engine, cache, &s.fed, &s.last_logits) {
        Ok(e) => px.insert(&s.pool_key(), e, why),
        Err(err) => eprintln!("[prefix-cache] snapshot failed ({err}); prefix not cached"),
    }
}

/// SEED insert at prefill-done: a cold session (nothing resumed) whose full prompt is long
/// enough and not already covered by an entry parks its primed prompt state for future
/// same-prefix traffic. `s.fed` == the prompt exactly at this point (no generation yet).
fn maybe_prefix_seed(engine: &Engine, px: &mut PrefixCache, s: &mut Session) {
    if !s.seed_prefix {
        return;
    }
    s.seed_prefix = false;
    if s.n_cached > 0 || s.cache.is_none() || s.fed.len() < PREFIX_CACHE_MIN_TOKENS {
        return;
    }
    if px.has_covering(&s.pool_key(), &s.fed) {
        return; // an entry already serves this prefix class
    }
    prefix_insert_from_session(engine, px, s, "seed");
}

struct Session {
    model: String,
    /// PC-ISO cache namespace this session admits, hits, and parks under (see PoolKey).
    cache_ns: String,
    /// yield lane — admission class + prefill budget bucket + batch priority.
    lane: crate::lanes::Lane,
    /// legacy tokenwise cache — None on the spec path (SpecSession owns its own caches; the
    /// double-alloc cost 2GB/128k-session and OOM'd the 27B serve — fixed 2026-07-05).
    cache: Option<Cache>,
    /// SPEC-DECODE serving (2026-07-05): sessions on MTP models decode in
    /// generate_spec_session BURSTS (K-token draft chains + batched verify) instead of one
    /// decode_step per tick — the CLI-measured spec win (27B p3: 79 vs 40 tok/s) brought to the
    /// serve path. `Some` when: model has an MTP head + MEMRA_SERVE_SPEC!=0 + the sampler is
    /// EITHER greedy (argmax verify) OR sampled (temperature>0 -> the rejection-sampling
    /// verify, filters and penalties applied symmetrically to draft q and target p; landed
    /// 2026-07-09/10, feat/sampled-graph-draft + feat/filtered-spec). Greedy-with-penalties
    /// is the one excluded class (`greedy_penalized`) — the argmax verify would ignore them.
    /// See `spec_eligible` in step_admit for the authoritative predicate.
    /// The SpecSession owns its OWN cache/scratch; `cache` above stays as the (unused) admit
    /// allocation on this path (kept to avoid restructuring admit; ~small VRAM overhead until
    /// a follow-up drops it). committed == every token whose state the spec caches hold.
    spec: Option<memra_engine::spec::SpecSession>,
    /// SINGLE-SESSION CUDA-GRAPH serving (2026-07-26, +34% measured at B=1): a greedy
    /// interactive session admitted ALONE rides GraphSession replay (one step/tick, 4B
    /// D2H). Degrades to the batched-eager path the moment a second session admits —
    /// legal because dc==eager is bit-identical, so the graph cache continues seamlessly.
    graph: Option<memra_engine::decode::GraphSession>,
    /// The token produced by the last graph step (next INPUT; emitted on the next tick).
    graph_pending: Option<u32>,
    /// Live acceptance telemetry (hqmtp axis-D): cumulative drafted/accepted across the
    /// session's bursts, logged per burst so serve-regime acceptance-vs-context is measurable.
    spec_drafted: usize,
    spec_accepted: usize,
    sampler: Sampler,
    last_logits: Vec<f32>,
    /// Token pre-sampled ON DEVICE by the last batched tick (decode_step_batch_sampled) —
    /// consumed by the next advance_sample_emit instead of the O(n_vocab) host sample
    /// (measured 1.36 ms/row at 248k vocab). None = host-sample from last_logits (fallback
    /// rows: penalties/top-k/top-p/min-p configs; non-batched paths). Dropped un-consumed
    /// when a session finishes — same semantics as an unsampled last_logits.
    device_next: Option<u32>,
    /// Constrained decoding (`response_format`): per-session llguidance grammar state.
    /// `Some` masks the logits BEFORE every sample and advances with each accepted token.
    /// FULL path (2026-08-03): the packed mask uploads to `mask_dev` each step and
    /// mask_logits_f32 bans on DEVICE before the device sampler — constrained rows ride
    /// the same device-sample/lean-logits tick as everyone else. Fallback sampler configs
    /// (penalties/top-k/top-p/min-p) and MEMRA_CONSTRAIN_HOST=1 keep the v1 host-side
    /// masked-copy sample.
    constraint: Option<crate::constrained::SessionConstraint>,
    /// Device grammar-mask buffer (packed SimpleVob words). Allocated once at first use,
    /// STABLE POINTER thereafter — contents re-uploaded per step (~n_vocab/8 bytes), the
    /// graph-capture contract for the in-graph mask read.
    mask_dev: Option<CudaSlice<u32>>,
    /// Words uploaded this step (0 = no mask staged for the pending batch step).
    mask_words: usize,
    /// Every token actually FED to decode_step, in order (prompt prime + generated feedback).
    /// This is exactly the sequence whose KV + recurrent state live in `cache` — the resume
    /// point for KV PREFIX REUSE on retire (see ReusePool).
    fed: Vec<u32>,
    /// prompt tokens still to be primed (consumed one per scheduler tick during prefill).
    prefill_queue: std::collections::VecDeque<u32>,
    prefill_done: bool,
    generated: Vec<u32>,
    params: GenParams,
    stop_strings: Vec<String>,
    trace_id: Option<String>,
    /// detokenized text already emitted (to compute incremental deltas + stop-string matching).
    emitted_bytes: usize,
    budget: usize,        // max tokens we may still generate
    /// usage accounting (worker-truth): total prompt tokens this session feeds/resumes, and
    /// how many came from a cache (continuation pool / spec resume / prefix cache).
    n_prompt: usize,
    n_cached: usize,
    /// PREFIX-CACHE LCP SPLIT: prime exactly up to this fed-length, snapshot the cache into
    /// the prefix cache there, then continue with the rest of the prompt (the learning step).
    snapshot_at: Option<usize>,
    /// PREFIX-CACHE SEED: park the full primed prompt at prefill-done (cold sessions only).
    seed_prefix: bool,
    tx: tokio::sync::mpsc::UnboundedSender<Event>,
    t0: Instant,
}

impl Session {
    /// The (model, namespace) reuse-pool key this session hits and parks under (PC-ISO).
    fn pool_key(&self) -> PoolKey {
        (self.model.clone(), self.cache_ns.clone())
    }
}

/// The worker entry point. Runs on its OWN std::thread. Builds the Engine + loads every model on
/// THIS thread (CUDA-context affinity), then runs the scheduler loop until the command channel
/// closes. `models` = (name, gguf_path) pairs. Sends `ready_tx` once load completes (or the error).
pub fn run(
    models: Vec<(String, String, Option<String>)>,
    rx: Receiver<Cmd>,
    ready_tx: Sender<Result<(Vec<String>, HashMap<String, ModelCaps>), String>>,
    metrics: SharedMetrics,
) {
    // ---- one-time init on the worker thread: Engine + all models resident ----
    let engine = match Engine::new(0) {
        Ok(e) => e,
        Err(err) => { let _ = ready_tx.send(Err(format!("Engine::new failed: {err}"))); return; }
    };
    // MEMRA_FAST is read ONCE here (same handling as run_gen): the matmul path consults the env var
    // per-call, but logging it once keeps the worker's behavior explicit and stable for the run.
    let fast = std::env::var("MEMRA_FAST").as_deref() != Ok("0");
    eprintln!("[worker] Engine ready (MEMRA_FAST={})", fast);

    let mut loaded: HashMap<String, LoadedModel> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for (name, path, draft) in &models {
        eprintln!("[worker] loading model {name:?} <- {path}");
        // DIRECTORY path = safetensors HF checkpoint or a manifest-backed memra repack/overlay;
        // file = GGUF. Repack tokenizers live in the manifest's source_dir.
        let from_dir = std::path::Path::new(path).is_dir();
        let (model, tok) = if from_dir {
            let dir = std::path::Path::new(path);
            let (src, tok_dir): (Box<dyn memra_gguf::source::TensorSource>, std::path::PathBuf) =
                if dir.join("manifest.json").exists() {
                    let repack = match memra_gguf::source::Hy3RepackSource::open(dir) {
                        Ok(source) => source,
                        Err(err) => { let _ = ready_tx.send(Err(format!("open {path}: {err}"))); return; }
                    };
                    let tok_dir = repack.source_dir()
                        .filter(|source| source.join("tokenizer.json").exists())
                        .unwrap_or(dir).to_path_buf();
                    (Box::new(repack), tok_dir)
                } else {
                    let st = match memra_gguf::source::SafetensorsSource::open(dir) {
                        Ok(source) => source,
                        Err(err) => { let _ = ready_tx.send(Err(format!("open {path}: {err}"))); return; }
                    };
                    (Box::new(st), dir.to_path_buf())
                };
            let model = match HybridModel::load_from_source(&engine, src.as_ref()) {
                Ok(m) => m,
                Err(err) => { let _ = ready_tx.send(Err(format!("load {name}: {err}"))); return; }
            };
            let tok = match Tokenizer::from_hf_dir(&tok_dir) {
                Ok(t) => t,
                Err(err) => { let _ = ready_tx.send(Err(format!("tokenizer {name}: {err}"))); return; }
            };
            (model, tok)
        } else {
            let g = match GgufFile::open(path) {
                Ok(g) => g,
                Err(err) => { let _ = ready_tx.send(Err(format!("open {path}: {err}"))); return; }
            };
            let model = match HybridModel::load(&engine, &g) {
                Ok(m) => m,
                Err(err) => { let _ = ready_tx.send(Err(format!("load {name}: {err}"))); return; }
            };
            let tok = match Tokenizer::from_gguf(&g) {
                Ok(t) => t,
                Err(err) => { let _ = ready_tx.send(Err(format!("tokenizer {name}: {err}"))); return; }
            };
            (model, tok)
        };
        // Per-model regime draft (MEMRA_MODELS "+<draft.gguf>" syntax): replace the embedded
        // MTP head with the standalone regime draft — same load path as MEMRA_MTP_DRAFT but
        // scoped to THIS model, so a multi-model server drafts each model with its own file.
        let model = {
            let mut model = model;
            if let Some(dpath) = draft {
                let dg = match GgufFile::open(dpath) {
                    Ok(g) => g,
                    Err(err) => { let _ = ready_tx.send(Err(format!("draft {name}: {err}"))); return; }
                };
                match memra_engine::hybrid::MtpHead::load_draft(&engine, &dg, &model.cfg) {
                    Ok(head) => {
                        eprintln!("[worker] {name}: regime draft attached ({dpath})");
                        model.mtp = Some(head);
                    }
                    Err(err) => { let _ = ready_tx.send(Err(format!("draft {name}: {err}"))); return; }
                }
            }
            model
        };

        let eos_id = tok.eos_id();
        eprintln!("[worker]   loaded {name:?}: {} layers, eos={eos_id}", model.cfg.n_layer);
        // (#68 closed 2026-08-04: the former ST-spec quarantine notice lived here — dir
        // checkpoints are spec-eligible again, research/fp8ship-20260804/RESULTS.md.)
        loaded.insert(name.clone(), LoadedModel {
            model, tok, eos_id, from_dir, constraints: std::cell::OnceCell::new(),
        });
        order.push(name.clone());
    }
    // Template capability probe (serve-tools lane): same substring laws the renderer uses.
    // + /v1/models metadata (serve-tail lane): context length from the model config,
    // tokenizer family from the pre-tokenizer name, instruct family from the template's
    // turn markers. Unknown stays 0/""/None — the route reports honest nulls.
    let caps: HashMap<String, ModelCaps> = loaded.iter().map(|(n, lm)| {
        let t = lm.tok.chat_template();
        let caps = ModelCaps {
            tools_branch: t.is_some_and(|t| t.contains("<tools>")
                && !t.contains("hy_User") && !t.contains("<|turn>")),
            qwen_think: t.is_some_and(|t| t.contains("<think>") && t.contains("add_generation_prompt")),
            think_switch: t.is_some_and(|t| t.contains("enable_thinking")),
            // GGUF keeps the historical ChatML fallback for template-less models; a dir
            // checkpoint (safetensors/repack) must CARRY its template (tokenizer_config
            // chat_template or chat_template.jinja) or chat requests 400 (serve-st v1).
            chat_ok: t.is_some() || !lm.from_dir,
            context_length: lm.model.cfg.context_length as usize,
            tokenizer: lm.tok.pre().to_string(),
            instruct_type: t.and_then(|t| {
                if t.contains("<|im_start|>") { Some("chatml".to_string()) }
                else if t.contains("<start_of_turn>") { Some("gemma".to_string()) }
                else { None }
            }),
        };
        eprintln!("[worker] {n}: template caps tools={} think={} think_switch={} chat_ok={} \
                   ctx={} tok={:?} instruct={:?}",
                  caps.tools_branch, caps.qwen_think, caps.think_switch, caps.chat_ok,
                  caps.context_length, caps.tokenizer, caps.instruct_type);
        (n.clone(), caps)
    }).collect();
    let _ = ready_tx.send(Ok((order.clone(), caps)));

    // Per-model decode chunk width (inc3 3a): computed once — model tensors and mirrors
    // are fixed after load.
    let chunk_caps: HashMap<String, usize> =
        loaded.iter().map(|(n, lm)| (n.clone(), chunk_cap_for(lm))).collect();
    for (n, c) in &chunk_caps {
        eprintln!("[worker] {n}: decode chunk cap {c}{}",
                  if *c > 8 { " (exact-16 tier)" } else { "" });
    }

    // ---- scheduler loop ----
    let mut active: Vec<Session> = Vec::new();
    let mut queue: std::collections::VecDeque<Box<Request>> = std::collections::VecDeque::new();
    // KV prefix-reuse pool (append-only continuation; see ReuseEntry doc). Keyed by
    // (model, namespace) — cross-request continuation state is tenant-scoped too (PC-ISO).
    let mut reuse: HashMap<PoolKey, Vec<ReuseEntry>> = HashMap::new();
    let mut spec_reuse: HashMap<PoolKey, Vec<SpecReuseEntry>> = HashMap::new();
    // F5: learned spec-session sizing (evict-first models + right-sized ctx asks).
    let mut spec_sizing = SpecSizing::default();
    // Cross-request prefix cache (token-prefix keyed, budget-bound; see the module doc above).
    let mut px = PrefixCache::default();
    if prefix_cache_budget_bytes() > 0 && serve_batching() {
        eprintln!("[prefix-cache] on: budget {:.0}MB (MEMRA_PREFIX_CACHE_MB), min prefix {} tokens",
                  prefix_cache_budget_bytes() as f64 / 1e6, PREFIX_CACHE_MIN_TOKENS);
    }
    // Observed VRAM cost of one admitted session, per model (free-VRAM delta across the first
    // successful admit) — feeds the VRAM-aware admission wait below.
    let mut session_vram_cost: HashMap<String, usize> = HashMap::new();

    // ---- serving counters + engine-truth step stats (30s percentile window) ----
    // Lane machinery (x-lane QoS gate, lane/dl-metering port): policy from env; step_stats
    // is the INTERACTIVE SLO sensor (records only ticks that advanced an interactive
    // session — on naked traffic every session is interactive, so /metrics is unchanged).
    let policy = crate::lanes::LanePolicy::from_env();
    let mut step_stats = StepStats::new(
        std::env::var("MEMRA_LANE_WINDOW_S").ok().and_then(|v| v.parse().ok()).unwrap_or(30.0));
    let mut n_admitted = 0u64;
    let mut n_completed = 0u64;
    let mut n_tokens_out = 0u64;
    let mut n_prompt_in = 0u64;
    let mut n_cached_in = 0u64;
    let mut lane_admitted = [0u64; 3];
    let mut lane_shed = [0u64; 3];
    let mut lane_completed = [0u64; 3];
    let mut lane_tokens = [0u64; 3];
    let mut last_batch = 0usize;
    // Starvation sentinel (estimator blind spot, 2026-07-26 native-judge battery): last
    // time an interactive session decoded. Interactive work waiting with no interactive
    // decode tick inside the SLO age IS an SLO breach the percentile window can't see.
    let mut last_interactive_decode = Instant::now();
    let mut tick_n: u64 = 0;

    loop {
        // 1. Drain pending commands. Block ONLY when there is no work at all (no active sessions),
        //    otherwise poll non-blocking so the decode loop keeps interleaving.
        if active.is_empty() && queue.is_empty() {
            match rx.recv() {
                Ok(cmd) => handle_cmd(cmd, &loaded, &order, &mut queue),
                Err(_) => break, // all senders dropped -> shutdown
            }
        }
        loop {
            match rx.try_recv() {
                Ok(cmd) => handle_cmd(cmd, &loaded, &order, &mut queue),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    if active.is_empty() { return; } else { break; }
                }
            }
        }

        // 2. ADMISSION + LANE GATE (x-lane yield gate, engine-side): interactive admits up
        //    to the cap and WAITS beyond it (FIFO, never rejected — its queue wait is the
        //    protected tenant's own backlog). Judge/harvest are gated on the measured
        //    interactive step p99 vs their SLO fraction and SHED with an immediate
        //    retryable error (HTTP 429 at the handler) — dark-lane work is NEVER queued
        //    inside the engine (the B2 lesson: the engine queue is where the tail dies).
        //    Interactive cap stays the legacy MEMRA_MAX_SESSIONS knob (naked-path
        //    preserving; policy.max_sessions[0] is the sidecar's knob, unused here);
        //    judge/harvest caps come from the lane policy.
        let max_active = if confidence_trace_enabled() { 1 } else { MAX_ACTIVE };
        let mut requeue: std::collections::VecDeque<Box<Request>> = Default::default();
        while let Some(req) = queue.pop_front() {
            // DISCONNECT ABORT (gap-scan F8): a queued request whose client already hung
            // up (receiver dropped) never reaches the GPU — dropped here, logged for the
            // metering record (0 generated; prompt never primed).
            if req.tx.is_closed() {
                eprintln!("[abort] client disconnected while queued (model {:?}); dropped",
                          req.model);
                continue;
            }
            let lane = req.lane;
            let batching_on = std::env::var("MEMRA_SERVE_BATCH").map(|v| v != "0").unwrap_or(true);
            let cap = if lane == crate::lanes::Lane::Interactive {
                if batching_on {
                    std::env::var("MEMRA_MAX_SESSIONS").ok()
                        .and_then(|v| v.parse().ok()).unwrap_or(64)
                } else {
                    max_active
                }
            } else {
                policy.max_sessions[lane.idx()]
            };
            let lane_count = active.iter().filter(|s| s.lane == lane).count();
            if lane_count >= cap {
                if lane == crate::lanes::Lane::Interactive {
                    requeue.push_back(req);   // waits (FIFO), never shed
                } else {
                    lane_shed[lane.idx()] += 1;
                    let _ = req.tx.send(Event::Error(format!(
                        "shed:{}:lane at capacity, retry", lane.as_str())));
                }
                continue;
            }
            // Starvation sentinel closes the estimator's blind spot (2026-07-26 native-judge
            // battery): interactive work EXISTS but no interactive decode tick ran within
            // the SLO age — starvation IS a breach even though the p99 window can't see it.
            let interactive_active_or_waiting = active.iter()
                .any(|s| s.lane == crate::lanes::Lane::Interactive);
            let starved = interactive_active_or_waiting
                && last_interactive_decode.elapsed().as_secs_f32() * 1000.0 > policy.slo_p99_ms;
            if !policy.admit(lane, &mut step_stats, starved) {
                lane_shed[lane.idx()] += 1;
                let _ = req.tx.send(Event::Error(format!(
                    "shed:{}:interactive p99 over budget, retry", lane.as_str())));
                continue;
            }
            // VRAM-AWARE ADMISSION (lane/fast-router, 2026-08-02). Evidence: c=16 on the
            // RESIDENT Ornith-35B 400'd 1-3 requests per burst with a quoted `cache alloc
            // failed: CUDA_ERROR_OUT_OF_MEMORY` (research/fast-router-20260802/greedy-hash-
            // o35b-batch-default-try{1,2}) — resident-if-fits plans a ~2GB reserve while
            // sixteen 8192-ctx session caches want more than that. Admission already WAITS
            // on the session-count axis; extend the same FIFO-wait to the VRAM axis: once a
            // model's per-session cost has been observed, admit the next session only while
            // free VRAM covers TWO of them (one to allocate + one of headroom for prefill/
            // decode transients); otherwise the request waits for a finisher. The first
            // session always passes (the residency planner's reserve guarantees one), and an
            // OOM with no active sessions still errors — that is a real capacity failure,
            // quoted to the client.
            if !active.is_empty() {
                if let (Some(&cost), Ok((free, _))) =
                    (session_vram_cost.get(&req.model), engine.ctx().mem_get_info()) {
                    if free < cost.saturating_mul(2) {
                        requeue.push_back(req);   // waits (FIFO), never rejected
                        continue;
                    }
                }
            }
            let model_key = req.model.clone();
            let free_before = engine.ctx().mem_get_info().map(|(f, _)| f).ok();
            match admit(&engine, &loaded, &mut reuse, &mut spec_reuse, &mut spec_sizing,
                        &mut px, *req) {
                Ok(s) => {
                    n_admitted += 1;
                    lane_admitted[lane.idx()] += 1;
                    n_prompt_in += s.n_prompt as u64;
                    n_cached_in += s.n_cached as u64;
                    active.push(s);
                    if !session_vram_cost.contains_key(&model_key) {
                        if let (Some(fb), Ok((fa, _))) = (free_before, engine.ctx().mem_get_info()) {
                            let cost = fb.saturating_sub(fa);
                            if cost > 0 {
                                eprintln!("[worker] observed session VRAM cost for {model_key:?}: \
                                           {:.0}MB (admission gate = 2x)", cost as f64 / 1e6);
                                session_vram_cost.insert(model_key, cost);
                            }
                        }
                    }
                }
                Err((tx, msg)) => { let _ = tx.send(Event::Error(msg)); }
            }
        }
        queue = requeue;

        // 3. The tick. Three phases (MEMRA_SERVE_BATCH=0 restores legacy round-robin):
        //    (a) spec sessions burst solo (spec x batch composition is a later step);
        //    (b) prefilling sessions prime at the full tick chunk (PREFILL_TICK_T);
        //    (c) decoding sessions advance through BATCHED steps: sample+emit host-side, then
        //        decode_step_batch over survivors in chunks of <= 8.
        let batching = serve_batching();
        let mut finished: Vec<usize> = Vec::new();
        // DISCONNECT ABORT (gap-scan F8): every send in the tick loop is `let _ =
        // s.tx.send(..)` — send errors ignored — so an aborted client used to burn GPU
        // until max_tokens/EOS and hold a slot against admission. The per-tick sweep
        // retires closed-channel sessions BEFORE any phase steps them; the log line is
        // the metering record (bill-to-abort-point: prompt/cached/generated so far).
        // Retire still parks reusable KV — the state is consistent at the abort point.
        for (i, s) in active.iter().enumerate() {
            if s.tx.is_closed() {
                abort_log(s);
                finished.push(i);
            }
        }
        if !batching {
            for i in 0..active.len() {
                if finished.contains(&i) { continue; }
                match step_session(&engine, &loaded, &mut active[i]) {
                    Ok(true) => {}
                    Ok(false) => finished.push(i),
                    Err(err) => {
                        let _ = active[i].tx.send(Event::Error(format!("step error: {err}")));
                        finished.push(i);
                    }
                }
            }
        } else {
            // (a0) SINGLE-SESSION GRAPH PATH (MEMRA_SERVE_GS=0 disables). Promote a lone cold
            // greedy interactive session to GraphSession replay (+34% at B=1); degrade back
            // to batched-eager the moment concurrency arrives (dc==eager bit-identity makes
            // the cache handoff seamless).
            let gs_on = {
                static G: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *G.get_or_init(|| std::env::var("MEMRA_SERVE_GS").map(|v| v != "0").unwrap_or(true))
            };
            if gs_on && active.len() > 1 {
                for i in 0..active.len() {
                    if finished.contains(&i) || active[i].graph.is_none() { continue; }
                    let s = &mut active[i];
                    let g = s.graph.take().unwrap();
                    s.cache = Some(g.cache);
                    if let Some(pend) = s.graph_pending.take() {
                        let (cont, _) = advance_token_emit(&loaded, s, pend);
                        if !cont {
                            finished.push(i);
                        } else {
                            let lm = &loaded[&s.model];
                            match lm.model.decode_step(&engine, pend, s.cache.as_mut().unwrap()) {
                                Ok(l) => { s.last_logits = l; s.fed.push(pend); }
                                Err(err) => {
                                    let _ = s.tx.send(Event::Error(format!("degrade: {err}")));
                                    finished.push(i);
                                }
                            }
                        }
                    }
                }
            }
            if gs_on && active.len() == 1 && !finished.contains(&0) {
                let s = &mut active[0];
                // Promote only generations long enough to amortize the one-time
                // capture+snapshot (~340ms measured = ~330-token break-even at the
                // 1.03ms/tok graph saving). Short requests stay eager-batched.
                let gs_min: usize = std::env::var("MEMRA_GS_MIN").ok()
                    .and_then(|v| v.parse().ok()).unwrap_or(384);
                // POST-PREFILL promotion (round 35): the old cold promotion re-primed the
                // prompt TOKEN-WISE inside graph_session_new — a live ~3x end-to-end LOSS
                // for solo long prompts (measured 871-tok/400-gen: 6.4s vs ~2.2s eager).
                // Now the normal chunked/batched prefill primes first; the graph session
                // captures OVER that cache (graph_session_from_cache). TTFT pays only the
                // one-time capture (~340ms), amortized by the gs_min budget gate.
                // CONSTRAINED sessions promote too (constrained-full, 2026-08-03): the
                // captured step bans the packed grammar mask on device before its argmax
                // (stable mask pointer, contents re-uploaded per step). Host-oracle and
                // fallback-sampler constrained sessions stay eager.
                // SAMPLED sessions do NOT graph-promote (`is_greedy()` below) — a separate,
                // untaken lever. It costs nothing today: this promotion only fires for
                // `s.spec.is_none()` solo sessions, and on an MTP model every sampled
                // session already rides the (faster) sampled spec burst path instead. It
                // would only matter for sampled sessions on a NON-MTP model; capturing the
                // seeded gumbel draw needs the in-graph RNG-counter bump the spec draft
                // chain already has (spec.rs sctr_inc), so it is wiring, not new math.
                let constr_graph_ok = s.constraint.is_none()
                    || (!constrain_host() && devsample_meta(s).is_some());
                if s.graph.is_none() && s.spec.is_none() && s.sampler.is_greedy()
                    && constr_graph_ok
                    && s.lane == crate::lanes::Lane::Interactive
                    && s.budget >= gs_min
                    && s.prefill_done && s.generated.is_empty() && s.cache.is_some()
                    && !s.last_logits.is_empty()
                {
                    let lm = &loaded[&s.model];
                    // first generated token: MASKED argmax for constrained sessions (the
                    // grammar's initial state), plain argmax otherwise.
                    let (first, mask0) = match s.constraint.as_mut() {
                        Some(c) => match c.compute_mask() {
                            Ok(m) => {
                                let mut row = s.last_logits.clone();
                                crate::constrained::apply_mask(&m, &mut row);
                                (memra_engine::forward::argmax(&row) as u32, Some(m))
                            }
                            Err(err) => {
                                let _ = s.tx.send(Event::Error(format!("constraint mask: {err}")));
                                finished.push(0);
                                (0, None)
                            }
                        },
                        None => (memra_engine::forward::argmax(&s.last_logits) as u32, None),
                    };
                    if !finished.contains(&0) {
                    let cache = s.cache.take().unwrap();
                    match lm.model.graph_session_from_cache_masked(
                        &engine, cache, first, s.budget + 2,
                        mask0.as_ref().map(|m| m.as_slice())) {
                        Ok((g, first)) => {
                            s.graph = Some(g);
                            s.graph_pending = Some(first);
                        }
                        Err(err) => {
                            // capture failed with the cache consumed — degrade the session
                            // via the graph-less error path (rare: capture-time errors only).
                            let _ = s.tx.send(Event::Error(format!("graph promote failed: {err}")));
                            finished.push(0);
                        }
                    }
                    }
                }
                // step the (possibly just-promoted) graph session: one token per tick
                let s = &mut active[0];
                if let Some(pend) = s.graph_pending.take() {
                    let t_g = Instant::now();
                    let (cont, _) = advance_token_emit(&loaded, s, pend);
                    if !cont {
                        finished.push(0);
                    } else {
                        s.fed.push(pend);
                        // CONSTRAINED: fresh post-consume mask into the graph's stable
                        // buffer before the replay (the KV-pointer update pattern).
                        let mut mask_err = None;
                        if let Some(c) = s.constraint.as_mut() {
                            match c.compute_mask() {
                                Ok(m) => {
                                    if let Err(err) = s.graph.as_mut().unwrap()
                                        .upload_mask(&engine, m.as_slice()) {
                                        mask_err = Some(err.to_string());
                                    }
                                }
                                Err(err) => mask_err = Some(err),
                            }
                        }
                        if let Some(err) = mask_err {
                            let _ = s.tx.send(Event::Error(format!("constraint mask: {err}")));
                            finished.push(0);
                        } else {
                        let lm = &loaded[&s.model];
                        let g = s.graph.as_mut().unwrap();
                        match g.step(&engine, &lm.model) {
                            Ok(next) => { s.graph_pending = Some(next); }
                            Err(_) => { finish(s, StopReason::MaxNew); finished.push(0); }
                        }
                        n_tokens_out += 1;
                        lane_tokens[0] += 1;
                        step_stats.record(t_g.elapsed().as_secs_f32() * 1000.0);
                        last_interactive_decode = Instant::now();
                        }
                    }
                }
            }
            // (a) spec bursts
            for i in 0..active.len() {
                if finished.contains(&i) { continue; }
                if active[i].spec.is_some() {
                    match step_session(&engine, &loaded, &mut active[i]) {
                        Ok(true) => {}
                        Ok(false) => finished.push(i),
                        Err(err) => {
                            let _ = active[i].tx.send(Event::Error(format!("step error: {err}")));
                            finished.push(i);
                        }
                    }
                }
            }
            // (b) INTERACTIVE prefill only (TTFT priority, full tick chunk budgets[0]).
            // Dark-lane (judge/harvest) prefill runs AFTER decode (phase d) so a judge
            // prime can never sit between an interactive stream and its next token (the
            // 282ms-p99 lesson, 2026-07-26 native-judge battery).
            // task #13 (2026-07-26): BATCH fresh short primes across sessions —
            // one concat trunk, GEMMs at m = sum_T. Measured regime (prime-batch-gate --bench):
            // +80% at B=8 T=64, +44-49% at T=128, crossover ~T=320 (above it, single primes
            // win — per-seq m already at the GEMM plateau). Gate: prime-batch-gate ALL GREEN
            // (per-seq argmax + decode-stream equality). MEMRA_PRIME_BATCH=1 disables.
            let budgets = policy.prefill_budget;
            let (cand, held) = 'pb: loop {
                // default 6 (2026-07-26): with the varlen GDN core (task #18) the
                // concat sweet spot moved from B=4 to B=6-8 (16501 vs 15950 tok/s
                // at T=152 — the per-seq core train no longer scales with B).
                let pb_max: usize = std::env::var("MEMRA_PRIME_BATCH").ok()
                    .and_then(|v| v.parse().ok()).unwrap_or(6);
                // 320 -> 2048 (2026-07-27): the old T=320 crossover ("above it, single
                // primes win") was measured on the per-seq core train. With the wgmma
                // varlen cores (task #22 vl twins) batched wins at EVERY tested T:
                // +30.1% at T=320, +12.6% at 512, +5.9% at 937, +3.0% at 1536
                // (prime-batch-gate --bench, B=3). budgets[0] still caps per-tick load.
                let pb_maxt: usize = std::env::var("MEMRA_PRIME_BATCH_MAX_T").ok()
                    .and_then(|v| v.parse().ok()).unwrap_or(2048);
                let min_t = memra_engine::hybrid_forward::PRIME_MIN_T.max(2);
                let mut cand: Vec<usize> = Vec::new();
                let mut cand_model: Option<String> = None;
                if pb_max >= 2 && !confidence_trace_enabled() {
                    for i in 0..active.len() {
                        if finished.contains(&i) { continue; }
                        let s = &active[i];
                        let ql = s.prefill_queue.len();
                        if s.spec.is_none() && !s.prefill_done && s.graph.is_none()
                            && s.lane == crate::lanes::Lane::Interactive
                            && s.fed.is_empty()
                            && s.cache.as_ref().is_some_and(|c| c.pos == 0)
                            // prefix-cache LCP split primes alone (the boundary snapshot
                            // needs a per-session stop inside the prompt; concat can't stop).
                            && s.snapshot_at.is_none()
                            && ql >= min_t && ql <= pb_maxt && ql <= budgets[0]
                            && cand_model.as_ref().is_none_or(|m| *m == s.model)
                        {
                            cand_model.get_or_insert_with(|| s.model.clone());
                            cand.push(i);
                            if cand.len() == pb_max { break; }
                        }
                    }
                }
                // BATCH-FORMATION HOLD: a lone fresh candidate that arrived <hold_ms ago is
                // deferred (skipped by the single-prime loop below via the same predicate NOT
                // firing — it stays queued) so staggered arrivals can coalesce. Telemetry
                // 2026-07-26: without the hold only 25% of a 32-concurrent burst batched
                // (ticks ~1ms, arrivals staggered). TTFT cost <= hold_ms on a ~40ms prime.
                let hold_ms: u64 = std::env::var("MEMRA_PRIME_BATCH_HOLD_MS").ok()
                    .and_then(|v| v.parse().ok()).unwrap_or(4);
                let mut held = false;
                if cand.len() == 1 && hold_ms > 0 {
                    let s = &active[cand[0]];
                    if s.t0.elapsed().as_millis() < hold_ms as u128 {
                        held = true;
                    }
                }
                let mut fired = false;
                if cand.len() >= 2 {
                    let prompts: Vec<Vec<u32>> = cand.iter()
                        .map(|&i| active[i].prefill_queue.drain(..).collect())
                        .collect();
                    let prompt_refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
                    let mut cache_refs: Vec<&mut memra_engine::cache::Cache> = active.iter_mut()
                        .enumerate()
                        .filter(|(i, _)| cand.contains(i))
                        .map(|(_, s)| s.cache.as_mut().unwrap())
                        .collect();
                    let lm = &loaded[cand_model.as_ref().unwrap()];
                    let t_pb = Instant::now();
                    match lm.model.prime_cache_batch(&engine, &prompt_refs, &mut cache_refs) {
                        Ok(outs) => {
                            let toks: usize = prompts.iter().map(|p| p.len()).sum();
                            eprintln!("[prime-batch] B={} tokens={} in {:.1}ms",
                                      cand.len(), toks, t_pb.elapsed().as_secs_f64() * 1e3);
                            for ((&i, prompt), (l, _h, _x)) in
                                cand.iter().zip(&prompts).zip(outs)
                            {
                                let s = &mut active[i];
                                s.last_logits = l;
                                for &tok in prompt { s.fed.push(tok); s.sampler.accept(tok); }
                                s.prefill_done = true;
                                // prefix-cache seed: batch-primed bytes are the concat
                                // config — the entry stores whatever config ran (contract).
                                maybe_prefix_seed(&engine, &mut px, s);
                            }
                            fired = true;
                        }
                        Err(err) => {
                            // fall back: restore queues, the per-session path serves this tick
                            eprintln!("[prime-batch] failed ({err}); single primes serve");
                            for (&i, prompt) in cand.iter().zip(&prompts) {
                                active[i].prefill_queue = prompt.iter().copied().collect();
                            }
                        }
                    }
                }
                // ROUNDS (telemetry 2026-07-26: a tick with 8 pending batched 4 and
                // single-primed the rest): keep batching while >= 2 candidates remain.
                if fired { continue 'pb; }
                break 'pb (cand, held);
            };
            for i in 0..active.len() {
                if finished.contains(&i) { continue; }
                if held && cand.first() == Some(&i) { continue; }   // batch-formation hold
                let s = &mut active[i];
                if s.spec.is_some() || s.prefill_done { continue; }
                if s.lane != crate::lanes::Lane::Interactive { continue; }
                match prefill_tick(&engine, &loaded, &mut px, s, budgets[0]) {
                    Ok(_) => {}
                    Err(err) => {
                        let _ = s.tx.send(Event::Error(format!("prefill error: {err}")));
                        finished.push(i);
                    }
                }
            }
            // (c) batched decode, interactive rows first (stable sort by lane index: chunks
            // fill with protected-class rows before dark rows).
            let t_decode = Instant::now();
            let mut decoding: Vec<usize> = (0..active.len())
                .filter(|&i| !finished.contains(&i)
                        && active[i].spec.is_none() && active[i].prefill_done
                        && active[i].cache.is_some())
                .collect();
            decoding.sort_by_key(|&i| active[i].lane.idx());
            let mut had_interactive = false;
            // sample + emit + stop checks (host); survivors carry their next token
            let mut ready: Vec<(usize, u32)> = Vec::new();
            for &i in &decoding {
                let (cont, next) = advance_sample_emit(&loaded, &mut active[i]);
                match (cont, next) {
                    (false, _) => finished.push(i),
                    (true, Some(t)) => {
                        // GRAMMAR MASK STAGING (constrained-full): compute the post-consume
                        // token mask and H2D the packed bitset into the session's stable
                        // device buffer — the batched step bans on device BEFORE its device
                        // sampler, so this row rides the same lean tick as everyone else.
                        if let Err(err) = stage_grammar_mask(&engine, &mut active[i]) {
                            let _ = active[i].tx.send(Event::Error(
                                format!("constraint mask: {err}")));
                            finished.push(i);
                            continue;
                        }
                        had_interactive |= active[i].lane == crate::lanes::Lane::Interactive;
                        ready.push((i, t));
                    }
                    (true, None) => {} // nothing to do this tick
                }
            }
            // batched steps in per-model chunks (chunk_cap_for: exact-16 tier models chunk
            // at 16, everything else 8; MEMRA_DECODE_BATCH_CAP is the explicit door).
            // D2H audit (inc3 3c): the per-chunk [B]-u32 device-token readback inside the
            // step is the tick's ONLY steady-state D2H — one per chunk, none per seq. A
            // deferred one-per-TICK variant measured FLAT (±0.7%, N=4, c=8/16/32, 5090 —
            // research/batched-tick-inc3-20260801) and was killed per the flags doctrine.
            for chunk in group_chunks(&active, &ready, &chunk_caps) {
                let toks: Vec<u32> = chunk.iter().map(|&(_, t)| t).collect();
                let idxs: Vec<usize> = chunk.iter().map(|&(i, _)| i).collect();
                let model_name = active[idxs[0]].model.clone();
                let lm = &loaded[&model_name];
                // DEVICE-SIDE SAMPLING metas (MEMRA_SERVE_DEVSAMPLE=0 reverts to host): rows
                // whose sampler is greedy-no-penalties (device argmax, bit-identical) or
                // pure-temperature (seeded gumbel draw — top-k/top-p/min-p/penalty configs
                // keep the host path) sample on device inside the batched step; the next
                // tick's advance_sample_emit consumes the token instead of the O(n_vocab)
                // host sample. Counter = generated.len() — a session-progress function,
                // independent of batch composition (the isolation contract, gate3).
                let samp: Vec<Option<(f32, u64, u32)>> = idxs
                    .iter()
                    .map(|&i| {
                        let s = &active[i];
                        // constrained rows: device-sample iff a mask was staged this tick
                        // (fallback sampler configs / MEMRA_CONSTRAIN_HOST keep the v1
                        // host masked-copy sample — mask_words stays 0 for them).
                        if s.constraint.is_some() && s.mask_words == 0 {
                            return None;
                        }
                        devsample_meta(s)
                    })
                    .collect();
                // GRAMMAR MASKS: staged rows pass (stable device buffer, word count). Raw
                // pointers here because the caches split-borrow below takes as_mut_ptr on
                // `active` — the fields are disjoint (mask_dev vs cache), same soundness
                // class as the existing unique-index split-borrow.
                let mask_ptrs: Vec<Option<(*const CudaSlice<u32>, usize)>> = idxs
                    .iter()
                    .map(|&i| {
                        let s = &active[i];
                        if s.mask_words > 0 {
                            s.mask_dev.as_ref().map(|d| (d as *const _, s.mask_words))
                        } else {
                            None
                        }
                    })
                    .collect();
                let logits = {
                    // split-borrow: pull the caches out via split_at_mut-style indexing
                    let mut caches: Vec<&mut Cache> = Vec::with_capacity(idxs.len());
                    // SAFETY: idxs are unique indices into `active`; we take disjoint &mut.
                    let base = active.as_mut_ptr();
                    for &i in &idxs {
                        let s = unsafe { &mut *base.add(i) };
                        caches.push(s.cache.as_mut().unwrap());
                    }
                    // LEAN LOGITS (inc2 component 3): device-sampled rows skip the
                    // [n_vocab] D2H — their last_logits comes back EMPTY and the row is
                    // parked on-device (cache.last_logits_dev) for the retire-time pool
                    // park below. MEMRA_SERVE_LEANLOGITS=0 restores the full D2H.
                    // SAFETY: mask_ptrs point at Session.mask_dev fields — disjoint from
                    // the caches taken above; nothing mutates them for this call's life.
                    let masks: Vec<Option<(&CudaSlice<u32>, usize)>> = mask_ptrs
                        .iter()
                        .map(|m| m.map(|(p, w)| (unsafe { &*p }, w)))
                        .collect();
                    lm.model.decode_step_batch_sampled_lean_masked(
                        &engine, &toks, &mut caches, &samp, &masks, serve_leanlogits())
                };
                match logits {
                    Ok((rows, next_toks)) => {
                        for (k, &i) in idxs.iter().enumerate() {
                            active[i].last_logits = rows[k].clone();
                            active[i].device_next = next_toks[k];
                            active[i].fed.push(toks[k]);
                            n_tokens_out += 1;
                            lane_tokens[active[i].lane.idx()] += 1;
                        }
                    }
                    Err(err) => {
                        for &i in &idxs {
                            let _ = active[i].tx.send(Event::Error(format!("batch step: {err}")));
                            finished.push(i);
                        }
                    }
                }
            }
            if had_interactive {
                last_interactive_decode = Instant::now();
            }
            last_batch = ready.len();
            // MEMRA_TICK_TRACE=1: per-tick phase timing to stderr (diagnosis only).
            if std::env::var("MEMRA_TICK_TRACE").as_deref() == Ok("1") {
                let n_int = active.iter()
                    .filter(|s| s.lane == crate::lanes::Lane::Interactive).count();
                let n_pref = active.iter().filter(|s| !s.prefill_done).count();
                eprintln!("[tick] act={} int={} priming={} ready={} decode_ms={:.1}",
                          active.len(), n_int, n_pref, ready.len(),
                          t_decode.elapsed().as_secs_f32() * 1000.0);
            }
            // (d) dark-lane prefill, ADAPTIVE: the tick period IS the client TPOT, so dark
            // primes may only consume the SLO headroom decode left over (2026-07-26 yield
            // battery: fixed 256-tok chunks pushed client p99 42 -> 91ms while the
            // decode-only estimator read 44ms). Chunk tokens = headroom_ms x prime rate.
            let decode_ms = t_decode.elapsed().as_secs_f32() * 1000.0;
            let headroom_ms = (policy.slo_p99_ms - decode_ms).max(0.0);
            let prime_tok_per_ms: f32 = std::env::var("MEMRA_PRIME_TOK_PER_MS").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(8.0);
            let adaptive_cap = (headroom_ms * prime_tok_per_ms) as usize;
            // task #17 increment (2026-07-30): CONCAT small FRESH dark prefills — the
            // harvest profile (many short prompts) previously burned one tick per
            // session; a single prime_cache_batch serves them together at m = sum_T,
            // INSIDE the same headroom budget (sum_T <= lane budget AND adaptive cap,
            // so the 282ms-p99 lesson holds: dark work never exceeds the SLO headroom).
            // Same lane + same model only (budget accounting stays per-lane); >= 2
            // candidates, else the single-chunk path below serves as before.
            let mut dark_batched = false;
            {
                let min_t = memra_engine::hybrid_forward::PRIME_MIN_T.max(2);
                let mut dcand: Vec<usize> = Vec::new();
                let mut dmodel: Option<String> = None;
                let mut dlane: Option<usize> = None;
                let mut dsum = 0usize;
                for i in 0..active.len() {
                    if finished.contains(&i) { continue; }
                    let s = &active[i];
                    let li = s.lane.idx();
                    let ql = s.prefill_queue.len();
                    if li == 0 || budgets[li] == 0 { continue; }
                    // FRESH (pos==0, nothing fed) or CONTINUATION (cache primed exactly
                    // through fed): both prime from cache.pos. Carried gemma4 stays
                    // single-chunk (no continuation prime; engine rejects). LCP-split
                    // sessions prime alone (the boundary snapshot needs a per-session
                    // stop inside the prompt; concat can't stop).
                    if s.spec.is_some() || s.prefill_done || s.graph.is_some()
                        || s.snapshot_at.is_some()
                        || !s.cache.as_ref().is_some_and(|c| c.pos == s.fed.len()) { continue; }
                    if !s.fed.is_empty() && loaded[&s.model].model.cfg.gemma4.is_some() { continue; }
                    let cap = budgets[li].min(adaptive_cap);
                    if ql < min_t || dsum + ql > cap { continue; }
                    if dlane.is_some_and(|l| l != li) { continue; }
                    if dmodel.as_ref().is_some_and(|m| *m != s.model) { continue; }
                    dlane.get_or_insert(li);
                    dmodel.get_or_insert_with(|| s.model.clone());
                    dsum += ql;
                    dcand.push(i);
                }
                if dcand.len() >= 2 {
                    let prompts: Vec<Vec<u32>> = dcand.iter()
                        .map(|&i| active[i].prefill_queue.drain(..).collect())
                        .collect();
                    let prompt_refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
                    let mut cache_refs: Vec<&mut memra_engine::cache::Cache> = active.iter_mut()
                        .enumerate()
                        .filter(|(i, _)| dcand.contains(i))
                        .map(|(_, s)| s.cache.as_mut().unwrap())
                        .collect();
                    let lm = &loaded[dmodel.as_ref().unwrap()];
                    match lm.model.prime_cache_batch(&engine, &prompt_refs, &mut cache_refs) {
                        Ok(outs) => {
                            let ncar = dcand.iter()
                                .filter(|&&i| !active[i].fed.is_empty()).count();
                            eprintln!("[prime-batch dark] lane={} B={} tokens={dsum} carried={ncar}",
                                      dlane.unwrap(), dcand.len());
                            for ((&i, prompt), (l, _h, _x)) in
                                dcand.iter().zip(&prompts).zip(outs)
                            {
                                let s = &mut active[i];
                                s.last_logits = l;
                                for &tok in prompt { s.fed.push(tok); s.sampler.accept(tok); }
                                s.prefill_done = true;
                            }
                        }
                        Err(err) => {
                            eprintln!("[prime-batch dark] failed ({err}); chunks serve");
                            for (&i, prompt) in dcand.iter().zip(&prompts) {
                                active[i].prefill_queue = prompt.iter().copied().collect();
                            }
                            dcand.clear();
                        }
                    }
                    dark_batched = !dcand.is_empty(); // the batch WAS this tick's dark action
                }
            }
            for i in 0..active.len() {
                if dark_batched { break; }
                if finished.contains(&i) { continue; }
                let s = &mut active[i];
                if s.spec.is_some() || s.prefill_done { continue; }
                let li = s.lane.idx();
                if li == 0 || budgets[li] == 0 { continue; }
                let chunk = budgets[li].min(adaptive_cap);
                if chunk < memra_engine::hybrid_forward::PRIME_MIN_T { break; }
                if let Err(err) = prefill_tick(&engine, &loaded, &mut px, s, chunk) {
                    let _ = s.tx.send(Event::Error(format!("prefill error: {err}")));
                    finished.push(i);
                }
                break; // one dark chunk per tick — the headroom budget is tick-global
            }
            // Engine-truth interactive TPOT = the FULL client-visible tick (decode + any
            // dark prime). Only interactive-carrying ticks feed the SLO estimator; on
            // naked (all-interactive) traffic this is exactly the pre-gate had_decode.
            if had_interactive {
                step_stats.record(t_decode.elapsed().as_secs_f32() * 1000.0);
            }
        }
        // retire finished sessions (reverse order so indices stay valid). Long-enough sessions
        // park their (fed, cache, last_logits) in the reuse pool instead of dropping the cache.
        finished.sort_unstable();
        finished.dedup();
        for &i in finished.iter().rev() {
            let s = active.remove(i);
            let pool_key = s.pool_key(); // before the partial moves below (PC-ISO park key)
            n_completed += 1;
            lane_completed[s.lane.idx()] += 1;
            if let Some(mut sess) = s.spec {
                // PENDING-CARRY flush before parking: a parked session must be fully committed
                // (committed_text drives the text-prefix resume match — an uncommitted pending
                // would double-feed on resume). One T=1 pass per RETIRED request, not per burst.
                if sess.pending_tok.is_some() {
                    if let Err(err) = loaded[&s.model].model.spec_flush_pending(&engine, &mut sess) {
                        eprintln!("[worker] spec pending flush failed ({err}); dropping session");
                        continue;
                    }
                }
                if sess.committed.len() >= REUSE_MIN_PREFIX && sess.next_pred.is_some() {
                    // skip the leading BOS when rendering: the client's prompt STRING never
                    // contains it (encode() adds it), so it would poison the text-prefix match.
                    let toks = &sess.committed;
                    let skip = loaded[&s.model].tok.bos_id()
                        .map(|b| toks.first() == Some(&b)).unwrap_or(false) as usize;
                    let committed_text = loaded[&s.model].tok.decode_special(&toks[skip..], true);
                    let pool = spec_reuse.entry(pool_key).or_default();
                    if pool.len() >= reuse_pool_per_model() { pool.remove(0); }
                    pool.push(SpecReuseEntry { sess, committed_text });
                }
            } else if s.fed.len() >= REUSE_MIN_PREFIX && s.prefill_done {
                if let Some(cache) = s.cache {
                    // LEAN LOGITS (inc2 component 3): device-sampled sessions carried no
                    // host last_logits — recover the final row from the device park with
                    // ONE D2H here (retire-time, pool-bound sessions only). A session with
                    // neither host nor device logits cannot serve an empty-suffix resume:
                    // skip parking it rather than park a poisoned entry.
                    let last_logits = if s.last_logits.is_empty() {
                        cache.last_logits_dev.as_ref()
                            .and_then(|d| engine.dtoh(d).ok())
                            .unwrap_or_default()
                    } else {
                        s.last_logits
                    };
                    if !last_logits.is_empty() {
                        let pool = reuse.entry(pool_key).or_default();
                        if pool.len() >= reuse_pool_per_model() { pool.remove(0); } // LRU: oldest first
                        let cap = cache.max_ctx;
                        pool.push(ReuseEntry {
                            fed: s.fed, cache, last_logits, cap,
                        });
                    }
                }
            }
        }
        // publish serving metrics (worker owns the counters; axum reads the snapshot).
        // THROTTLED: the per-tick mutex+percentile cost ~1.7ms/token of B=1 TPOT
        // (2026-07-26 live A/B) — publish every 32nd tick.
        tick_n = tick_n.wrapping_add(1);
        if tick_n % 32 == 0 { if let Ok(mut m) = metrics.lock() {
            m.admitted = n_admitted;
            m.completed = n_completed;
            m.tokens_out = n_tokens_out;
            m.step_p50_ms = step_stats.p(50.0).unwrap_or(0.0);
            m.step_p99_ms = step_stats.p(99.0).unwrap_or(0.0);
            m.prompt_tokens_in = n_prompt_in;
            m.cached_tokens_in = n_cached_in;
            m.prefix_hits = px.hits;
            m.prefix_entries = px.n_entries() as u64;
            m.prefix_bytes = px.total_bytes as u64;
            m.lane_admitted = lane_admitted;
            m.lane_shed = lane_shed;
            m.lane_completed = lane_completed;
            m.lane_tokens = lane_tokens;
            m.batch_size_last = last_batch;
        } }
        if !finished.is_empty() && std::env::var("MEMRA_SPILL_STATS").as_deref() == Ok("1") {
            if let Some((reads, bytes, errors, short, fallbacks, waits, ring_full)) =
                engine.moe_pread_stats() {
                eprintln!("[spill-pread] snapshot reads={reads} bytes={bytes} errors={errors} \
                           short_reads={short} fallbacks={fallbacks} buffer_waits={waits} \
                           ring_full={ring_full}");
            }
            if let Some((hits, misses, staged_bytes, slots)) = engine.moe_cache_stats() {
                let accesses = hits.saturating_add(misses);
                let hit_rate = if accesses == 0 {
                    0.0
                } else {
                    100.0 * hits as f64 / accesses as f64
                };
                eprintln!("[moe-cache] snapshot hits={hits} misses={misses} \
                           hit_rate={hit_rate:.3} staged_bytes={staged_bytes} slots={slots}");
            }
        }
    }
}

fn handle_cmd(
    cmd: Cmd,
    loaded: &HashMap<String, LoadedModel>,
    order: &[String],
    queue: &mut std::collections::VecDeque<Box<Request>>,
) {
    match cmd {
        Cmd::Generate(req) => {
            if !loaded.contains_key(&req.model) {
                let _ = req.tx.send(Event::Error(format!(
                    "unknown model {:?}; loaded: {:?}", req.model, order)));
                return;
            }
            queue.push_back(req);
        }
    }
}

/// Build a Session: tokenize the prompt (worker owns the Tokenizer), allocate the per-session Cache,
/// build the per-session Sampler. The prompt is NOT primed here — it's fed one token per scheduler
/// tick so prefill of a new session interleaves with other sessions' decode (the BASE-4 interleave).
fn admit(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    reuse: &mut HashMap<PoolKey, Vec<ReuseEntry>>,
    spec_reuse: &mut HashMap<PoolKey, Vec<SpecReuseEntry>>,
    spec_sizing: &mut SpecSizing,
    px: &mut PrefixCache,
    req: Request,
) -> Result<Session, (tokio::sync::mpsc::UnboundedSender<Event>, String)> {
    let lm = &loaded[&req.model];
    // PC-ISO: every reuse-pool probe below scans ONLY this (model, namespace) pool.
    let pool_key: PoolKey = (req.model.clone(), req.cache_ns.clone());

    // Tokenize: prefer explicit prompt_ids (raw-id path, for the exact-token validation gate); else
    // tokenize the text, optionally wrapping in the chat template.
    let prompt: Vec<u32> = if !req.prompt_ids.is_empty() {
        req.prompt_ids.clone()
    } else if !req.chat_turns.is_empty() {
        // ISOLATION CONTRACT (serve-tools lane): a request with no tools features renders
        // through the EXACT legacy path — the tools renderer is entered only when the
        // request carries tools / tool turns / a non-default think switch.
        let plain = req.tools_json.is_empty()
            && req.think == memra_tokenizer::chat::ThinkMode::Default
            && req.chat_turns.iter().all(|t| t.role != "tool" && t.tool_calls.is_empty());
        let rendered = if plain {
            let messages: Vec<_> = req.chat_turns.iter()
                .map(|t| (t.role.as_str(), t.content.as_str()))
                .collect();
            lm.tok.apply_chat_template(&messages, true)
        } else {
            match lm.tok.apply_chat_template_tools(&req.chat_turns, true,
                                                   &req.tools_json, req.think) {
                Ok(rendered) => rendered,
                Err(err) => return Err((req.tx, format!("chat template: {err}"))),
            }
        };
        lm.tok.encode(&rendered, true)
    } else if req.chat {
        let rendered = lm.tok.apply_chat_template(&[("user", req.prompt_text.as_str())], true);
        lm.tok.encode(&rendered, true)
    } else {
        lm.tok.encode(&req.prompt_text, true)
    };
    if prompt.is_empty() {
        return Err((req.tx, "empty prompt after tokenization".into()));
    }

    // Context guard mirrors generate_with: prompt + generated must fit ctx_cap.
    // MEMRA_CTX (default 8192): FLOOR for session cache allocation — per-request-sized caches can
    // never serve a LONGER continuation, which made the KV-reuse pool structurally unhittable in
    // multi-turn (parked cap 168 < next turn's need 240). Fixed-size sessions are also how the
    // reference server allocates (--ctx-size). KV cost @8192 on the 9B ≈ 119MB/session.
    let ctx_floor: usize = std::env::var("MEMRA_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
    // max_tokens OMITTED (MAX_NEW_CTX_BOUNDED sentinel, gap-scan F2): the session runs at
    // the serving context (MEMRA_CTX / explicit max_ctx), capped at the model's trained
    // context — budget becomes ctx_cap - prompt below, the vLLM/OpenAI default-when-omitted
    // semantics. Explicit max_tokens keeps the exact legacy sizing (prompt + max_new + 8,
    // floored at MEMRA_CTX) — honored exactly.
    let ctx_cap = match (req.params.max_ctx, req.params.max_new) {
        (Some(c), _) => c.max(ctx_floor),
        (None, MAX_NEW_CTX_BOUNDED) => {
            // serving ctx (the MEMRA_CTX floor) normally; a prompt that doesn't fit it
            // grows the session to prompt + one serving ctx of room; always capped at
            // the model's trained context (prompts past THAT are a real 400 below).
            let model_ctx = lm.model.cfg.context_length as usize;
            let mut c = ctx_floor;
            if prompt.len() + 16 > c { c = prompt.len().saturating_add(ctx_floor); }
            if model_ctx > 0 { c = c.min(model_ctx); }
            c
        }
        (None, max_new) => (prompt.len() + max_new + 8).max(ctx_floor),
    };
    if prompt.len() >= ctx_cap {
        return Err((req.tx, format!(
            "prompt ({} tok) >= context cap ({})", prompt.len(), ctx_cap)));
    }
    let room = ctx_cap - prompt.len();
    let budget = req.params.max_new.min(room);

    // KV PREFIX REUSE probe: a parked session whose fed sequence is an EXACT PREFIX of this
    // prompt (and whose cache has room) resumes — only the suffix gets primed. The sampler's
    // penalty history is replayed on host (cheap) so sampling matches a cold run exactly.
    let mut reused: Option<ReuseEntry> = None;
    // DEFAULT-ON (2026-07-05): the identity gate now exists at the engine level — session-gate
    // (bins) pins 3-turn continuation-prime output == fresh-greedy oracle on both models, and the
    // continuation path the reuse pool takes (prime_cache with cache.pos>0 / decode_step) is
    // exactly what it validates. MEMRA_KV_REUSE=0 disables.
    let reuse_on = !confidence_trace_enabled()
        && std::env::var("MEMRA_KV_REUSE").map(|v| v != "0").unwrap_or(true);
    if let (true, Some(pool)) = (reuse_on, reuse.get_mut(&pool_key)) {
        if let Some(idx) = pool.iter().rposition(|e|
            e.fed.len() >= REUSE_MIN_PREFIX && e.cap >= ctx_cap
                && prompt.len() >= e.fed.len() && prompt.starts_with(&e.fed)) {
            reused = Some(pool.remove(idx));
        }
    }

    // SPEC ELIGIBILITY decides the prefix-cache policy up front: spec sessions bypass the
    // cross-request prefix cache entirely (SpecSession owns trunk + draft caches; restoring a
    // trunk-only prefix would leave draft state unprimed — the spec tier keeps its own
    // continuation pool below). Mirrors the spec-branch condition exactly.
    // ST-SPEC QUARANTINE LIFTED (#68 closed, 2026-08-04): the serve-spec divergence on
    // dir-loaded checkpoints was never ST-specific — the per-session persistent draft
    // graph replayed with dangling pool addresses (capture transients not retained +
    // fa_part_pool freeing grown-past buffers the capture baked; fixed in spec.rs/lib.rs,
    // receipts research/fp8ship-20260804/RESULTS.md — the same corruption reproduced on
    // GGUF session bursts at n>=600). Dir checkpoints are spec-eligible again; the
    // serve-st gate pins default-serve text == the run-gen CLI tokenwise oracle.
    let serve_spec = !confidence_trace_enabled()
        && std::env::var("MEMRA_SERVE_SPEC").map(|v| v != "0").unwrap_or(true);
    let mut sampler = Sampler::new(req.sampler_cfg);
    // GREEDY + penalties keeps the legacy tokenwise path (gap-scan F3 plumbing): the greedy
    // spec arm verifies by pure argmax (sampling=None), which would silently ignore the
    // penalties the host sampler applies pre-argmax. Sampled requests carry penalties into
    // the rejection-sampling verify (SpecSampling) and stay spec-eligible.
    let greedy_penalized = sampler.is_greedy()
        && (sampler.penalty_repeat() != 1.0 || sampler.penalty_freq() != 0.0
            || sampler.penalty_present() != 0.0);
    // CONSTRAINED DECODING (response_format): compile the request's grammar against this
    // model's lazily-built vocab factory. Compile errors are clean request errors here —
    // never a mid-stream worker failure. Unconstrained requests skip everything.
    let constraint = match &req.grammar {
        None => None,
        Some(spec) => {
            let factory = lm.constraints.get_or_init(||
                crate::constrained::ConstraintFactory::new(&lm.tok));
            match factory {
                Err(err) => return Err((req.tx, format!("constrained decoding: {err}"))),
                Ok(f) => {
                    let sc = f.matcher(spec);
                    if let Some(err) = sc.error() {
                        return Err((req.tx, format!("response_format: {err}")));
                    }
                    Some(sc)
                }
            }
        }
    };
    // SPEC x CONSTRAINED (constrained-full, 2026-08-03): greedy constrained sessions ride
    // spec bursts — the grammar truncates acceptance AFTER the exactness verify and forces
    // the masked argmax at the cut slot (generate_spec_session_constrained). Sampled
    // constrained and the MEMRA_CONSTRAIN_HOST oracle keep plain decode.
    let spec_eligible = serve_spec
        && (constraint.is_none() || (sampler.is_greedy() && !constrain_host()))
        && (sampler.is_greedy() || sampler.temperature() > 0.0)
        && !greedy_penalized
        && lm.model.mtp.is_some();

    // CROSS-REQUEST PREFIX CACHE probe (2026-08-02; module doc at PrefixCache). Only when the
    // continuation pool missed, the session won't go spec, and batched scheduling is live.
    // A hit deep-copies the longest matching entry into a fresh session cache (entries are
    // reusable); a miss with a long-enough LCP against an existing entry arms the split-prime
    // learning insert; cold long prompts arm the seed insert.
    let prefix_on = reuse_on && serve_batching() && prefix_cache_budget_bytes() > 0;
    let mut prefix_hit = false;
    let mut snapshot_at: Option<usize> = None;
    let mut seed_prefix = false;
    if prefix_on && reused.is_none() && !spec_eligible {
        if let Some(i) = px.lookup(&pool_key, &prompt) {
            let restored = {
                let e = &px.entries[&pool_key][i];
                match Cache::new(engine, &lm.model.cfg, ctx_cap) {
                    Ok(mut c) => match prefix_restore(engine, &mut c, e) {
                        Ok(()) => Ok(ReuseEntry {
                            fed: e.toks.clone(),
                            cache: c,
                            last_logits: e.last_logits.clone(),
                            cap: ctx_cap,
                        }),
                        Err(err) => Err(format!("restore failed: {err}")),
                    },
                    Err(err) => Err(format!("session cache alloc failed: {err}")),
                }
            };
            match restored {
                Ok(entry) => {
                    let pool = px.entries.get_mut(&pool_key).unwrap();
                    pool[i].last_use = Instant::now();
                    px.hits += 1;
                    px.hit_tokens += entry.fed.len() as u64;
                    prefix_hit = true;
                    eprintln!("[prefix-cache] hit: {} of {} prompt tokens from cache (model {})",
                              entry.fed.len(), prompt.len(), req.model);
                    reused = Some(entry);
                }
                Err(msg) => {
                    // headroom discipline: sessions win over the cache — on alloc pressure
                    // drop every entry so the cold path (and the retries behind it) can fit.
                    if msg.starts_with("session cache alloc failed") {
                        let n = px.evict_all();
                        eprintln!("[prefix-cache] {msg}; evicted {n} entries, cold path serves");
                    } else {
                        eprintln!("[prefix-cache] {msg}; cold path serves");
                    }
                }
            }
        }
        if reused.is_none() {
            px.misses += 1;
            let l = px.best_lcp(&pool_key, &prompt);
            if l >= PREFIX_CACHE_MIN_TOKENS && l < prompt.len()
                && !px.has_key(&pool_key, &prompt[..l])
            {
                snapshot_at = Some(l);
            }
            if prompt.len() >= PREFIX_CACHE_MIN_TOKENS {
                seed_prefix = true; // re-checked against covering entries at prefill-done
            }
        }
    }

    let (cache, seed_fed, seed_logits) = match reused {
        Some(e) => {
            if !prefix_hit {
                eprintln!("[worker] kv-reuse: {} of {} prompt tokens resumed (model {})",
                          e.fed.len(), prompt.len(), req.model);
            }
            (Some(e.cache), e.fed, e.last_logits)
        }
        // legacy cache deferred: allocated below ONLY if the spec path doesn't take the session.
        None => (None, Vec::new(), Vec::new()),
    };

    // EOS: union of caller-supplied eos + the model's own eos id.
    let mut params = req.params;
    if !params.eos.contains(&lm.eos_id) { params.eos.push(lm.eos_id); }

    // Suffix-only prefill on a reuse hit; sampler penalty history replayed over the whole prefix.
    for &t in &seed_fed { sampler.accept(t); }
    let suffix: Vec<u32> = prompt[seed_fed.len()..].to_vec();
    let prefill_done_at_admit = suffix.is_empty();
    // SPEC-DECODE serve path (2026-07-05): greedy + MTP head + not a KV-reuse resume (the spec
    // session owns its own caches; folding the reuse pool into SpecSession is a follow-up) +
    // MEMRA_SERVE_SPEC!=0. The whole prompt goes to the spec session as turn 1's suffix; the
    // legacy prefill/decode path is bypassed entirely in step_session.
    let mut spec_resumed = 0usize;
    let mut text_suffix: Option<Vec<u32>> = None;
    // Sampled-spec serve: temperature + filters + penalties ALL ride the rejection-sampling
    // spec path (transforms applied to p and q symmetrically) — the legacy per-token path
    // remains only as the no-MTP/resume fallback.
    let spec = if spec_eligible && seed_fed.is_empty() {
        // POOL RESUME: a parked spec session whose committed sequence exactly prefixes this
        // prompt (with cache room) resumes — only the suffix primes; equal-length = pure burst.
        // Match order: exact token prefix (bit-clean), else TEXT prefix (survives BPE boundary
        // divergence — the ~50% chat-turn miss class). Text hits re-tokenize only the remainder.
        // CONSTRAINED requests never resume parked spec sessions: the park's stashed
        // next_pred/pending is unconstrained state, and the grammar must own generation
        // from token 1. Cold spec session instead (still spec — just no pool hit).
        let resumed = if constraint.is_some() { None } else {
            spec_reuse.get_mut(&pool_key).and_then(|pool| {
            if let Some(idx) = pool.iter().rposition(|e|
                e.sess.cache_max_ctx() >= ctx_cap
                    && prompt.len() >= e.sess.committed.len()
                    && prompt.starts_with(&e.sess.committed)) {
                return Some(pool.remove(idx).sess);
            }
            if !req.prompt_text.is_empty() {
                if let Some(idx) = pool.iter().rposition(|e|
                    e.sess.cache_max_ctx() >= ctx_cap
                        && req.prompt_text.len() >= e.committed_text.len()
                        && req.prompt_text.starts_with(e.committed_text.as_str())) {
                    let e = pool.remove(idx);
                    let rem = &req.prompt_text[e.committed_text.len()..];
                    text_suffix = Some(lm.tok.encode(rem, false));
                    return Some(e.sess);
                }
            }
            None
        })};
        match resumed {
            Some(sess) => {
                spec_resumed = sess.committed.len();
                eprintln!("[worker] spec-reuse: {} committed tokens resumed{} (model {})",
                          spec_resumed,
                          if text_suffix.is_some() { " [text-prefix]" } else { "" }, req.model);
                Some(sess)
            }
            None => {
                // POOL MISS: a parked session's caches (~4GB at 128k: 17-layer trunk KV + draft
                // scratch) can starve the NEW allocation — 2 x 128k sessions + weights don't fit
                // 24GB. Misses happen when the text->token roundtrip diverges at a turn boundary
                // (detok+retok isn't prefix-stable), so the parked session is DEAD WEIGHT for
                // this conversation: evict the pool, then allocate. (Session-id affinity API is
                // the structural fix — follow-up.)
                //
                // F5 (spec-pool thrash, 2026-08-05 — research/specpool-20260804): on a
                // VRAM-tight rig EVERY turn of the daily driver is a miss (the client
                // rewrites history), so the old fail->evict->realloc walk ran once per
                // request, progressively slower as the doomed full-size ask grew the churn.
                // Two learned behaviors replace it:
                //   1. EVICT-FIRST: once a model has observed "parked ghost + new session
                //      don't fit" (evict_first), later misses evict the dead-weight pool
                //      BEFORE allocating — same eviction the failure forced anyway, minus
                //      the failed alloc. Roomy rigs never set the flag and keep the pool.
                //   2. RIGHT-SIZE LADDER: a post-evict (genuine) failure no longer dumps
                //      the whole burst to the tokenwise path. Shrink the ask toward
                //      `need` = prompt + budget + SPEC_SHRINK_SLACK — the exact cap this
                //      request's emission needs (MaxNew preempts ContextFull by
                //      construction, so a shrunken session emits identical tokens).
                //      The landing size is memoized (learned_ctx) so later misses ladder
                //      from it instead of re-walking. Below `need` = tokenwise fallback.
                if spec_sizing.evict_first.contains(&req.model) {
                    if let Some(n) = spec_reuse.get_mut(&pool_key)
                        .map(|p| { let n = p.len(); p.clear(); n }).filter(|&n| n > 0)
                    {
                        eprintln!("[worker] spec pool evicted ({n}) pre-alloc \
                                   (learned VRAM-tight; model {})", req.model);
                    }
                }
                let need = prompt.len().saturating_add(budget).saturating_add(SPEC_SHRINK_SLACK);
                match lm.model.new_session(engine, ctx_cap) {
                    Ok(sess) => Some(sess),
                    Err(first_err) => {
                        let evicted = spec_reuse.get_mut(&pool_key)
                            .map(|p| { let n = p.len(); p.clear(); n }).unwrap_or(0);
                        if evicted > 0 {
                            spec_sizing.evict_first.insert(req.model.clone());
                            eprintln!("[worker] spec pool evicted ({evicted}) after alloc \
                                       failure; retrying (evict-first learned)");
                        }
                        let retried = if evicted > 0 {
                            lm.model.new_session(engine, ctx_cap).ok()
                        } else { None };
                        match retried {
                            Some(sess) => Some(sess),
                            None => {
                                // Genuine capacity failure (pool empty). Right-size:
                                // ladder down from the learned/half ask toward `need`.
                                let mut sess = None;
                                if need <= ctx_cap {
                                    let mut ask = spec_sizing.learned_ctx.get(&req.model)
                                        .copied().unwrap_or(ctx_cap / 2)
                                        .clamp(need, ctx_cap);
                                    loop {
                                        let landed = match lm.model.new_session(engine, ask) {
                                            Ok(s) => {
                                                // transient reserve (see SPEC_SHRINK_RESERVE):
                                                // a fit that leaves no headroom panics later on
                                                // a lazy upload — treat as a miss, shrink on.
                                                // (a) the embed table is the biggest lazy
                                                // transient: make it resident FALLIBLY now;
                                                // (b) on a NEW landing size only (ask > learned
                                                // — a size that already served a burst has
                                                // proven its transients resident), PROBE-
                                                // allocate the reserve and drop it. A probe,
                                                // not a mem_get_info read, is the fit signal:
                                                // the async pool's pinned release threshold
                                                // keeps freed blocks cached and invisible to
                                                // free-VRAM queries — and re-probing after the
                                                // transients are resident double-counts them
                                                // (observed: turn-1 ladder walked to need and
                                                // still failed while a 16k session + resident
                                                // transients served fine on turn 0).
                                                let proven = spec_sizing.learned_ctx.get(&req.model)
                                                    .is_some_and(|&l| ask <= l);
                                                let ok = lm.model.ensure_embed_resident(engine).is_ok()
                                                    && (proven
                                                        || engine.alloc_u8_uninit(SPEC_SHRINK_RESERVE).is_ok());
                                                if ok { Some(s) } else { drop(s); None }
                                            }
                                            Err(_) => None,
                                        };
                                        match landed {
                                            Some(s) => {
                                                eprintln!("[worker] spec session right-sized: \
                                                           ctx {ask} of {ctx_cap} (prompt {} + \
                                                           budget {budget}; model {})",
                                                          prompt.len(), req.model);
                                                spec_sizing.learned_ctx.insert(req.model.clone(), ask);
                                                sess = Some(s);
                                                break;
                                            }
                                            None if ask > need => { ask = (ask / 2).max(need); }
                                            None => break,
                                        }
                                    }
                                }
                                if sess.is_none() {
                                    eprintln!("[worker] spec session alloc failed ({first_err}); \
                                               tokenwise path");
                                }
                                sess
                            }
                        }
                    }
                }
            }
        }
    } else { None };
    // spec-resume: replay sampler penalty history over the resumed prefix; queue only the suffix.
    // (text-prefix hit: replay the SESSION's committed ids — the prompt's own ids diverge at the
    // boundary; greedy sessions ignore penalties anyway, this keeps sampled-future-proofing sane.)
    if spec_resumed > 0 {
        match (&spec, &text_suffix) {
            (Some(sess), Some(_)) => { for &t in &sess.committed { sampler.accept(t); } }
            _ => { for &t in &prompt[..spec_resumed] { sampler.accept(t); } }
        }
    }
    // legacy tokenwise cache only when the spec path did NOT take the session (spec owns its own).
    let cache = match (&spec, cache) {
        (Some(_), c) => c,        // reuse hit carried a cache? keep it parked as-is (rare; None normally)
        (None, Some(c)) => Some(c),
        (None, None) => match Cache::new(engine, &lm.model.cfg, ctx_cap) {
            Ok(c) => Some(c),
            Err(err) => {
                // headroom discipline: the prefix cache yields before a session errors.
                let evicted = px.evict_all();
                if evicted > 0 {
                    eprintln!("[prefix-cache] evicted {evicted} entries after cache alloc failure; retrying");
                    match Cache::new(engine, &lm.model.cfg, ctx_cap) {
                        Ok(c) => Some(c),
                        Err(err) => return Err((req.tx, format!("cache alloc failed: {err}"))),
                    }
                } else {
                    return Err((req.tx, format!("cache alloc failed: {err}")));
                }
            }
        },
    };
    // WORKER-TRUTH usage accounting: total prompt tokens (as the worker actually feeds/resumes
    // them — the text-prefix spec resume re-tokenizes only the remainder) + how many came from
    // a cache instead of being computed.
    let (n_prompt, n_cached) = if spec_resumed > 0 {
        let suffix_len = text_suffix.as_ref().map(|t| t.len())
            .unwrap_or_else(|| prompt.len() - spec_resumed);
        (spec_resumed + suffix_len, spec_resumed)
    } else {
        (prompt.len(), seed_fed.len())
    };
    Ok(Session {
        model: req.model,
        cache_ns: req.cache_ns,
        lane: req.lane,
        cache,
        sampler,
        spec,
        graph: None,
        graph_pending: None,
        spec_drafted: 0,
        spec_accepted: 0,
        last_logits: seed_logits,
        device_next: None,
        constraint,
        mask_dev: None,
        mask_words: 0,
        fed: seed_fed,
        prefill_queue: if let Some(ts) = text_suffix { ts.into_iter().collect() }
                       else if spec_resumed > 0 { prompt[spec_resumed..].to_vec().into_iter().collect() }
                       else { suffix.into_iter().collect() },
        prefill_done: prefill_done_at_admit,
        generated: Vec::new(),
        params,
        stop_strings: req.stop_strings,
        trace_id: req.trace_id,
        emitted_bytes: 0,
        budget,
        n_prompt,
        n_cached,
        snapshot_at,
        seed_prefix,
        tx: req.tx,
        t0: Instant::now(),
    })
}

/// Return only the newly completed UTF-8 text. Tokenizer byte-fallback sequences may span token
/// boundaries; retain an incomplete suffix until a later token completes it instead of emitting a
/// permanent replacement character. Truly invalid bytes are consumed as U+FFFD so they cannot
/// stall every later delta.
fn utf8_delta(decoded: &[u8], emitted_bytes: &mut usize) -> String {
    if *emitted_bytes > decoded.len() {
        return String::new();
    }
    let mut cursor = *emitted_bytes;
    let mut delta = String::new();
    while cursor < decoded.len() {
        match std::str::from_utf8(&decoded[cursor..]) {
            Ok(text) => {
                delta.push_str(text);
                cursor = decoded.len();
            }
            Err(err) => {
                let valid = err.valid_up_to();
                if valid != 0 {
                    // SAFETY: `valid_up_to` is the exact valid UTF-8 prefix certified by Rust.
                    delta.push_str(unsafe {
                        std::str::from_utf8_unchecked(&decoded[cursor..cursor + valid])
                    });
                    cursor += valid;
                }
                match err.error_len() {
                    None => break,
                    Some(invalid) => {
                        delta.push('\u{fffd}');
                        cursor += invalid;
                    }
                }
            }
        }
    }
    *emitted_bytes = cursor;
    delta
}

/// One scheduler tick for one session. Returns Ok(true) if still running, Ok(false) if retired.
/// Decomposes `generate_with`'s loop body into a single per-session step (same semantics):
///   - prefill phase: consume ONE prompt token via decode_step, accept it into the sampler.
///   - decode phase: sample from last_logits, accept, stream the token, check EOS/stop/ctx, then
///     run ONE decode_step to produce the next logits.
/// One prefill tick for a session under a token budget. Returns tokens consumed.
/// Same chunking laws as step_session's prefill phase (PRIME_MIN_T floor, tail handling).
fn prefill_tick(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    px: &mut PrefixCache,
    s: &mut Session,
    budget: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let lm = &loaded[&s.model];
    let q = s.prefill_queue.len();
    if q == 0 {
        s.prefill_done = true;
        maybe_prefix_seed(engine, px, s);
        return Ok(0);
    }
    let mut consumed = 0usize;
    // PREFIX-CACHE LCP SPLIT: tokens left until the snapshot boundary. Chunks stop exactly
    // there; a residual below the prime floor rides the tokenwise path (unreachable at the
    // current PREFILL_TICK_T, guarded for smaller budgets).
    let bound_rem = s.snapshot_at.map(|b| b - s.fed.len());
    if !confidence_trace_enabled()
        && q >= memra_engine::hybrid_forward::PRIME_MIN_T.max(2)
        && budget >= memra_engine::hybrid_forward::PRIME_MIN_T
        && bound_rem.is_none_or(|r| r >= memra_engine::hybrid_forward::PRIME_MIN_T)
    {
        let mut take = q.min(budget);
        if q - take > 0 && q - take < memra_engine::hybrid_forward::PRIME_MIN_T {
            take = if q <= budget { q } else { take };
        }
        if let Some(r) = bound_rem {
            if take >= r {
                take = r; // stop exactly at the snapshot boundary
            } else if r - take < memra_engine::hybrid_forward::PRIME_MIN_T {
                // keep the boundary chunk itself primeable next tick
                take = (r - memra_engine::hybrid_forward::PRIME_MIN_T)
                    .max(memra_engine::hybrid_forward::PRIME_MIN_T);
            }
        }
        let chunk: Vec<u32> = s.prefill_queue.drain(..take).collect();
        let (l, _h, _x) = lm.model.prime_cache(engine, &chunk, s.cache.as_mut().unwrap())?;
        s.last_logits = l;
        for &tok in &chunk { s.fed.push(tok); s.sampler.accept(tok); }
        consumed = take;
    } else if let Some(tok) = s.prefill_queue.pop_front() {
        s.last_logits = lm.model.decode_step(engine, tok, s.cache.as_mut().unwrap())?;
        if let Some(&target) = s.prefill_queue.front() {
            write_confidence_trace(s, tok, target, &s.last_logits)?;
        }
        s.fed.push(tok);
        s.sampler.accept(tok);
        consumed = 1;
    }
    // Boundary reached: snapshot the primed prefix into the cache, then keep priming the rest
    // of the prompt as a continuation (the LCP-split learning insert).
    if s.snapshot_at == Some(s.fed.len()) {
        s.snapshot_at = None;
        prefix_insert_from_session(engine, px, s, "lcp-split");
    }
    if s.prefill_queue.is_empty() {
        s.prefill_done = true;
        maybe_prefix_seed(engine, px, s);
    }
    Ok(consumed)
}

/// GRAMMAR MASK STAGING (constrained-full, 2026-08-03): compute the session's current
/// llguidance token mask and H2D the packed bitset into its STABLE device buffer for the
/// upcoming batched step. Runs AFTER advance_sample_emit consumed the tick's token — the
/// mask reflects the post-consume grammar state, exactly the set legal for the NEXT token.
/// No-op (mask_words = 0 -> host fallback) for unconstrained sessions, fallback sampler
/// configs (penalties/filters host-sample), and the MEMRA_CONSTRAIN_HOST=1 oracle.
fn stage_grammar_mask(engine: &Engine, s: &mut Session) -> Result<(), String> {
    s.mask_words = 0;
    if s.constraint.is_none() || constrain_host() || devsample_meta(s).is_none() {
        return Ok(());
    }
    let mask = s.constraint.as_mut().unwrap().compute_mask()?;
    let words = mask.as_slice();
    match s.mask_dev.as_mut() {
        Some(d) if d.len() >= words.len() => {
            engine.htod_u32_into(d, words).map_err(|e| e.to_string())?;
        }
        _ => {
            let mut d = engine.alloc_u32_zeroed(words.len()).map_err(|e| e.to_string())?;
            engine.htod_u32_into(&mut d, words).map_err(|e| e.to_string())?;
            s.mask_dev = Some(d);
        }
    }
    s.mask_words = words.len();
    Ok(())
}

/// The decode tick's HOST half: sample from last_logits, emit the token, run the stop
/// battery. Returns (continue?, Some(next_token) to feed the next step). Extracted from
/// step_session so the batched scheduler can drive many sessions into ONE engine step.
fn advance_sample_emit(
    loaded: &HashMap<String, LoadedModel>,
    s: &mut Session,
) -> (bool, Option<u32>) {
    let lm = &loaded[&s.model];
    if s.generated.len() >= s.budget {
        finish(s, StopReason::MaxNew);
        return (false, None);
    }
    // Device-presampled token from the last batched tick (Session.device_next): skips the
    // O(n_vocab) host sample (measured 1.36 ms/row at 248k vocab). Greedy device rows are
    // bit-identical to host argmax; temp rows are the seeded device draw (gate3 contract).
    // CONSTRAINED rows never device-sample (their samp meta is None) — they host-sample
    // from a grammar-masked COPY of last_logits (the pristine row still parks into the
    // reuse pool at retire, so continuations resume unmasked).
    let next = match (s.device_next.take(), s.constraint.as_mut()) {
        (Some(t), _) => t,
        (None, Some(c)) => {
            let mut row = s.last_logits.clone();
            if let Err(err) = c.mask_logits(&mut row) {
                let _ = s.tx.send(Event::Error(format!("constraint mask: {err}")));
                return (false, None);
            }
            s.sampler.sample(&row)
        }
        (None, None) => s.sampler.sample(&s.last_logits),
    };
    s.sampler.accept(next);
    s.generated.push(next);
    if s.params.eos.contains(&next) {
        finish(s, StopReason::Eos);
        return (false, None);
    }
    // Advance the grammar with the accepted (non-EOS) token. The token came from this
    // state's own mask, so an error here is a real bug — stop LOUDLY, never emit
    // schema-violating text as if it conformed.
    if let Some(c) = s.constraint.as_mut() {
        if let Err(err) = c.consume(next) {
            let _ = s.tx.send(Event::Error(format!("constraint advance: {err}")));
            return (false, None);
        }
    }
    let decoded = lm.tok.decode_bytes_special(&s.generated, true);
    let delta = utf8_delta(&decoded, &mut s.emitted_bytes);
    let full = String::from_utf8_lossy(&decoded);
    // DISCONNECT ABORT (gap-scan F8): a failed send = receiver dropped = client gone.
    // Stop generating THIS tick (the tick-top sweep would only catch it next tick).
    if s.tx.send(Event::Token { id: next, text: delta }).is_err() {
        abort_log(s);
        return (false, None);
    }
    if !s.stop_strings.is_empty() && s.stop_strings.iter().any(|ss| full.contains(ss.as_str())) {
        finish(s, StopReason::Callback);
        return (false, None);
    }
    if s.cache.as_ref().map(|c| c.pos >= c.max_ctx).unwrap_or(false) {
        finish(s, StopReason::ContextFull);
        return (false, None);
    }
    (true, Some(next))
}

/// Token-driven twin of `advance_sample_emit` for the graph path: the token was produced
/// by the DEVICE argmax (greedy), so there is no sampling — accept, emit, stop battery.
/// Returns (continue?, ()).
fn advance_token_emit(
    loaded: &HashMap<String, LoadedModel>,
    s: &mut Session,
    tok: u32,
) -> (bool, ()) {
    let lm = &loaded[&s.model];
    if s.generated.len() >= s.budget {
        finish(s, StopReason::MaxNew);
        return (false, ());
    }
    s.sampler.accept(tok);
    s.generated.push(tok);
    if s.params.eos.contains(&tok) {
        finish(s, StopReason::Eos);
        return (false, ());
    }
    // CONSTRAINED graph sessions: the token came from the in-graph masked argmax — advance
    // the grammar (post-EOS-check, same ordering as advance_sample_emit). An error here is
    // a real bug: loud stop, never emit schema-violating text as if it conformed.
    if let Some(c) = s.constraint.as_mut() {
        if let Err(err) = c.consume(tok) {
            let _ = s.tx.send(Event::Error(format!("constraint advance: {err}")));
            return (false, ());
        }
    }
    let decoded = lm.tok.decode_bytes_special(&s.generated, true);
    let delta = utf8_delta(&decoded, &mut s.emitted_bytes);
    let full = String::from_utf8_lossy(&decoded);
    // DISCONNECT ABORT (gap-scan F8): failed send = client gone, stop this tick.
    if s.tx.send(Event::Token { id: tok, text: delta }).is_err() {
        abort_log(s);
        return (false, ());
    }
    if !s.stop_strings.is_empty() && s.stop_strings.iter().any(|ss| full.contains(ss.as_str())) {
        finish(s, StopReason::Callback);
        return (false, ());
    }
    (true, ())
}

/// Group ready (session_idx, token) pairs into batched-step chunks: same model, <= 8 rows
/// (the exactness-tier cap), input order preserved (caller sorted interactive first).
/// Device-side batched-tick sampling (default ON — measured 1.36 ms/row host temp-sample at
/// the 9B's 248k vocab, ~45% of the B=8 serving tick). MEMRA_SERVE_DEVSAMPLE=0 is the
/// rollback/A-B seam: every row host-samples from last_logits as before.
fn serve_devsample() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SERVE_DEVSAMPLE").as_deref() != Ok("0"))
}

/// LEAN LOGITS (inc2 component 3, default ON): device-sampled rows skip the [n_vocab]
/// logits D2H; the last row parks on-device per cache and is D2H'd once at retire (the
/// reuse-pool consumer). MEMRA_SERVE_LEANLOGITS=0 is the rollback/A-B seam (full D2H,
/// the exact pre-change tick).
fn serve_leanlogits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SERVE_LEANLOGITS").as_deref() != Ok("0"))
}

/// MEMRA_CONSTRAIN_HOST=1 (rollback oracle): constrained rows keep the v1 host-side
/// masked-copy sample (full-row D2H + O(n_vocab) host sample) instead of the device
/// grammar mask. Diagnostics/A-B only — the device path is the shipped default.
fn constrain_host() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_CONSTRAIN_HOST").as_deref() == Ok("1"))
}

/// Device-sample meta for a session's row in the batched step (the ONE eligibility rule —
/// the samp closure and the grammar-mask staging pass must agree): greedy-no-penalties
/// (device argmax, bit-identical) or pure-temperature (seeded gumbel). Penalty/top-k/
/// top-p/min-p configs host-sample. Counter = generated.len() — a session-progress
/// function, independent of batch composition (the isolation contract, gate3).
fn devsample_meta(s: &Session) -> Option<(f32, u64, u32)> {
    if !serve_devsample() {
        return None;
    }
    let sm = &s.sampler;
    let no_pen = sm.penalty_repeat() == 1.0
        && sm.penalty_freq() == 0.0
        && sm.penalty_present() == 0.0;
    if !no_pen || sm.top_k() != 0 || sm.top_p() < 1.0 || sm.min_p() > 0.0 {
        return None;
    }
    if sm.is_greedy() {
        Some((0.0, 0, 0))
    } else {
        Some((sm.temperature(), sm.seed(), s.generated.len() as u32))
    }
}

/// Per-model decode chunk width. MEMRA_DECODE_BATCH_CAP (explicit door) wins; otherwise
/// models that qualify for the EXACT-16 tier (decode_batch_exact16_ok — every matmul has
/// a bit-exact b16-class kernel; Q8_0 needs the q8rp mirror) default to chunk 16, the
/// measured winner on the 5090 (+12% aggregate over chunk 8 at 32 seqs, same mirror
/// config — research/batched-tick-inc3-20260801/chunksweep.log); everything else keeps
/// the chunk-8 exactness tier. Isolation contract unchanged either way (gate2
/// bit-strength PASS at both widths).
fn chunk_cap_for(lm: &LoadedModel) -> usize {
    if let Some(c) = std::env::var("MEMRA_DECODE_BATCH_CAP").ok().and_then(|v| v.parse().ok()) {
        return usize::clamp(c, 1, 32);
    }
    if lm.model.decode_batch_exact16_ok() { 16 } else { 8 }
}

fn group_chunks(
    active: &[Session],
    ready: &[(usize, u32)],
    caps: &HashMap<String, usize>,
) -> Vec<Vec<(usize, u32)>> {
    let mut chunks: Vec<Vec<(usize, u32)>> = Vec::new();
    for &(i, t) in ready {
        let model = &active[i].model;
        let cap = caps.get(model).copied().unwrap_or(8);
        match chunks.last_mut() {
            Some(c) if c.len() < cap && active[c[0].0].model == *model => c.push((i, t)),
            _ => chunks.push(vec![(i, t)]),
        }
    }
    chunks
}

fn step_session(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    s: &mut Session,
) -> Result<bool, Box<dyn std::error::Error>> {
    let lm = &loaded[&s.model];

    // ---- SPEC-BURST arm (2026-07-05): MTP sessions decode in generate_spec_session
    // bursts — turn 1 primes the prompt (suffix = the whole prefill queue), later ticks are
    // ZERO-prime continuation bursts (SpecSession.next_pred). Each burst emits up to
    // SPEC_BURST_T tokens; between bursts the scheduler round-robins other sessions. Exactness:
    // GREEDY bursts — the session-gate oracle (4 turns incl empty-suffix) pins burst output ==
    // fresh greedy, byte-identical. SAMPLED bursts (temperature>0, `sampling` below) are
    // DISTRIBUTIONALLY exact instead: the rejection-sampling verify draws from the same
    // filtered/penalized target as plain sampled decode, but consumes its own Philox streams
    // (sess.sctr/uctr), so the token stream is reproducible per (seed, session) rather than
    // byte-equal to a plain-sampled run. That is the contract, not a gap.
    if let Some(spec) = s.spec.as_mut() {
        // Burst size trades round-robin latency (other sessions wait a whole burst) against
        // per-burst fixed cost. The dominant cost — the per-call draft-graph recapture,
        // measured ~16ms/burst on H100 q27 — is gone since 2026-08-01: the captured graph
        // persists on the SpecSession (spec.rs DraftGraphCtx) and later bursts replay it.
        // MEMRA_SPEC_BURST overrides for measurement; 32 = latency-safe default.
        let burst_t: usize = std::env::var("MEMRA_SPEC_BURST").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(32);
        let k: usize = std::env::var("MEMRA_SPEC_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let room = s.budget.saturating_sub(s.generated.len()).min(burst_t);
        if room == 0 { finish(s, StopReason::MaxNew); return Ok(false); }
        let suffix: Vec<u32> = s.prefill_queue.drain(..).collect();
        s.prefill_done = true;
        if suffix.is_empty() && spec.next_pred.is_none() && spec.pending_tok.is_none() {
            // nothing primed and nothing to prime — shouldn't happen (admit rejects empty prompts)
            finish(s, StopReason::MaxNew); return Ok(false);
        }
        let sampling = if s.sampler.temperature() > 0.0 {
            Some(memra_engine::spec::SpecSampling {
                temp: s.sampler.temperature(),
                seed: s.sampler.seed(),
                top_k: s.sampler.top_k() as i32,
                top_p: s.sampler.top_p(),
                min_p: s.sampler.min_p(),
                penalty_last_n: s.sampler.penalty_last_n(),
                penalty_repeat: s.sampler.penalty_repeat(),
                penalty_freq: s.sampler.penalty_freq(),
                penalty_present: s.sampler.penalty_present(),
            })
        } else { None };
        // SPEC x CONSTRAINED: greedy constrained bursts carry the grammar hook — verify-side
        // truncation + masked-argmax cut slots (engine contract; sampled never gets here).
        let (burst, d, a) = match s.constraint.as_mut() {
            Some(c) => {
                let mut g = crate::constrained::SpecGrammar::new(c, lm.eos_id);
                lm.model.generate_spec_session_constrained(
                    engine, spec, &suffix, room, k, sampling, Some(&mut g))?
            }
            None => lm.model.generate_spec_session_sampled(
                engine, spec, &suffix, room, k, sampling)?,
        };
        s.spec_drafted += d;
        s.spec_accepted += a;
        if d > 0 {
            eprintln!("[spec-acc] ctx={} burst={}/{} cum={}/{}={:.3}",
                      s.fed.len() + suffix.len(), a, d, s.spec_accepted, s.spec_drafted,
                      s.spec_accepted as f64 / s.spec_drafted.max(1) as f64);
        }
        for &tok in &suffix { s.fed.push(tok); s.sampler.accept(tok); }
        let mut stop: Option<StopReason> = None;
        for &tok in &burst {
            s.sampler.accept(tok);
            s.generated.push(tok);
            s.fed.push(tok);
            if s.params.eos.contains(&tok) { stop = Some(StopReason::Eos); break; }
        }
        // stream the burst's incremental text in ONE event (per-token events are per-tick anyway).
        // EOS text is never streamed (serve-compat, 2026-08-03): the tokenwise path stops
        // BEFORE emitting the EOS token's text, but the burst used to detokenize the whole
        // tail — clients saw a literal `<|im_end|>` in content (caught by the SDK gate's G4
        // receipt). The token still counts (generated/fed keep it; committed state intact).
        let visible = match stop {
            Some(StopReason::Eos) => &s.generated[..s.generated.len() - 1],
            _ => &s.generated[..],
        };
        let decoded = lm.tok.decode_bytes_special(visible, true);
        let delta = utf8_delta(&decoded, &mut s.emitted_bytes);
        let full = String::from_utf8_lossy(&decoded);
        if !delta.is_empty()
            && s.tx.send(Event::Token { id: *burst.last().unwrap_or(&0), text: delta }).is_err()
        {
            // DISCONNECT ABORT (gap-scan F8): client gone — retire at the abort point
            // (session still parks; committed state is consistent post-burst).
            abort_log(s);
            return Ok(false);
        }
        if stop.is_none() && !s.stop_strings.is_empty()
            && s.stop_strings.iter().any(|ss| full.contains(ss.as_str())) {
            stop = Some(StopReason::Callback);
        }
        if stop.is_none() && s.generated.len() >= s.budget { stop = Some(StopReason::MaxNew); }
        // +3 (was +2): committed excludes a carried pending token (pending-carry, 2026-08-01).
        if stop.is_none() && spec.committed.len() + k + 3 >= spec.cache_max_ctx() {
            stop = Some(StopReason::ContextFull);
        }
        if let Some(r) = stop { finish(s, r); return Ok(false); }
        return Ok(true);
    }

    // ---- prefill phase: BATCHED chunk prime (2026-07-05). prime_cache now supports
    // continuation (cache.pos > 0 attends to the quantized past), so the worker primes up to
    // PREFILL_TICK_T prompt tokens per tick at prefill throughput (~2000-5900 tok/s) instead of
    // one decode_step (~38-100 tok/s) — a 32k prompt drops from ~15min of ticks to ~a minute,
    // while the per-tick cap keeps round-robin latency for concurrent sessions bounded.
    // Tails below PRIME_MIN_T keep the tokenwise decode_step path (prime_cache floor).
    if !s.prefill_done {
        let q = s.prefill_queue.len();
        if !confidence_trace_enabled() && q >= memra_engine::hybrid_forward::PRIME_MIN_T.max(2) {
            // leave a tail chunk >= PRIME_MIN_T if this tick doesn't finish the queue
            let mut take = q.min(PREFILL_TICK_T);
            if q - take > 0 && q - take < memra_engine::hybrid_forward::PRIME_MIN_T { take = q; }
            let chunk: Vec<u32> = s.prefill_queue.drain(..take).collect();
            let (l, _h, _x) = lm.model.prime_cache(engine, &chunk, s.cache.as_mut().unwrap())?;
            s.last_logits = l;
            for &tok in &chunk { s.fed.push(tok); s.sampler.accept(tok); }
        } else if let Some(tok) = s.prefill_queue.pop_front() {
            s.last_logits = lm.model.decode_step(engine, tok, s.cache.as_mut().unwrap())?;
            if let Some(&target) = s.prefill_queue.front() {
                write_confidence_trace(s, tok, target, &s.last_logits)?;
            }
            s.fed.push(tok);
            s.sampler.accept(tok);
        }
        if s.prefill_queue.is_empty() { s.prefill_done = true; }
        // If after this the prompt is fully primed AND budget==0, we still fall through to decode
        // (which will immediately hit MaxNew). Keep prefill and decode as distinct ticks otherwise.
        return Ok(true);
    }

    // ---- decode phase ----
    if s.generated.len() >= s.budget {
        finish(s, StopReason::MaxNew);
        return Ok(false);
    }

    // CONSTRAINED rows host-sample from a grammar-masked copy (same seam as
    // advance_sample_emit — the batched path; kept in lockstep).
    let next = match (s.device_next.take(), s.constraint.as_mut()) {
        (Some(t), _) => t,
        (None, Some(c)) => {
            let mut row = s.last_logits.clone();
            c.mask_logits(&mut row).map_err(|e| format!("constraint mask: {e}"))?;
            s.sampler.sample(&row)
        }
        (None, None) => s.sampler.sample(&s.last_logits),
    };
    s.sampler.accept(next);
    s.generated.push(next);

    // EOS stop (before streaming the EOS token as text — we still report it in the count).
    if s.params.eos.contains(&next) {
        finish(s, StopReason::Eos);
        return Ok(false);
    }
    if let Some(c) = s.constraint.as_mut() {
        c.consume(next).map_err(|e| format!("constraint advance: {e}"))?;
    }

    // Detokenize the full generated tail, compute the incremental text delta vs what we've emitted.
    let decoded = lm.tok.decode_bytes_special(&s.generated, true);
    let delta = utf8_delta(&decoded, &mut s.emitted_bytes);
    let full = String::from_utf8_lossy(&decoded);
    // DISCONNECT ABORT (gap-scan F8): failed send = client gone, retire at the abort point.
    if s.tx.send(Event::Token { id: next, text: delta }).is_err() {
        abort_log(s);
        return Ok(false);
    }

    // stop-string match on the detokenized tail.
    if !s.stop_strings.is_empty() && s.stop_strings.iter().any(|ss| full.contains(ss.as_str())) {
        finish(s, StopReason::Callback);
        return Ok(false);
    }

    // context guard.
    if s.cache.as_ref().map(|c| c.pos >= c.max_ctx).unwrap_or(false) {
        finish(s, StopReason::ContextFull);
        return Ok(false);
    }

    // produce next logits (the ONE decode_step that advances this session).
    s.last_logits = lm.model.decode_step(engine, next, s.cache.as_mut().unwrap())?;
    s.fed.push(next);
    Ok(true)
}

fn confidence_trace_enabled() -> bool {
    std::env::var("MEMRA_CONFIDENCE_TRACE").is_ok()
}

#[derive(Debug)]
struct ConfidenceSummary {
    reference_logprob: f64,
    top1_token: u32,
    top1_correct: bool,
    top1_top2_margin: f32,
    entropy: f64,
}

fn summarize_confidence(logits: &[f32], target: u32) -> Result<ConfidenceSummary, String> {
    let target = target as usize;
    if logits.is_empty() || target >= logits.len() {
        return Err(format!("target token {target} outside {} logits", logits.len()));
    }
    let mut top1 = (0usize, f32::NEG_INFINITY);
    let mut top2 = f32::NEG_INFINITY;
    for (index, &logit) in logits.iter().enumerate() {
        if logit > top1.1 {
            top2 = top1.1;
            top1 = (index, logit);
        } else if logit > top2 {
            top2 = logit;
        }
    }
    let max_logit = top1.1 as f64;
    let mut sum_exp = 0.0f64;
    let mut weighted_logit = 0.0f64;
    for &logit in logits {
        let exp = ((logit as f64) - max_logit).exp();
        sum_exp += exp;
        weighted_logit += exp * logit as f64;
    }
    let logsumexp = max_logit + sum_exp.ln();
    Ok(ConfidenceSummary {
        reference_logprob: logits[target] as f64 - logsumexp,
        top1_token: top1.0 as u32,
        top1_correct: top1.0 == target,
        top1_top2_margin: top1.1 - top2,
        entropy: logsumexp - weighted_logit / sum_exp,
    })
}

fn write_confidence_trace(
    session: &Session,
    input_token: u32,
    target_token: u32,
    logits: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let Ok(path) = std::env::var("MEMRA_CONFIDENCE_TRACE") else { return Ok(()) };
    let summary = summarize_confidence(logits, target_token).map_err(std::io::Error::other)?;
    let record = serde_json::json!({
        "format": "memra-token-confidence-v1",
        "trace_id": session.trace_id,
        "input_position": session.fed.len(),
        "input_token": input_token,
        "target_token": target_token,
        "reference_logprob": summary.reference_logprob,
        "top1_token": summary.top1_token,
        "top1_correct": summary.top1_correct,
        "top1_top2_margin": summary.top1_top2_margin,
        "entropy": summary.entropy,
    });
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{record}")?;
    Ok(())
}

/// DISCONNECT ABORT metering record (gap-scan F8): one log line per aborted session —
/// prompt/cached/generated at the abort point (bill-to-abort). Called from every
/// send-failure retire; the tick-top sweep prints the same shape.
fn abort_log(s: &Session) {
    eprintln!("[abort] client disconnected: model {:?}, prompt {} ({} cached), \
               {} generated — billed to abort point, {:.2}s",
              s.model, s.n_prompt, s.n_cached, s.generated.len(),
              s.t0.elapsed().as_secs_f64());
}

fn finish(s: &Session, reason: StopReason) {
    let elapsed = s.t0.elapsed().as_secs_f64();
    // constrained-session mask-cost receipt (the perf ledger line): steps + total/mean
    // host-side mask compute time. Unconstrained sessions log nothing.
    if let Some(c) = s.constraint.as_ref() {
        if c.steps > 0 {
            eprintln!("[constrained] {}: {} masked steps, mask total {:.2} ms ({:.3} ms/step)",
                      s.model, c.steps, c.mask_ns as f64 / 1e6,
                      c.mask_ns as f64 / 1e6 / c.steps as f64);
        }
        // DRAFT-SIDE MASKING receipt (lane/draft-mask): the speculative Matcher clone (one per
        // spec round) and the draft-position masks computed on it — the two costs the lane adds.
        if c.spec_clones > 0 {
            eprintln!("[draft-mask] {}: {} clones {:.2} ms ({:.3} ms/clone), \
                       {} draft masks {:.2} ms ({:.3} ms/mask)",
                      s.model, c.spec_clones, c.spec_ns as f64 / 1e6,
                      c.spec_ns as f64 / 1e6 / c.spec_clones as f64,
                      c.draft_masks, c.draft_mask_ns as f64 / 1e6,
                      c.draft_mask_ns as f64 / 1e6 / c.draft_masks.max(1) as f64);
        }
    }
    let reason = format!("{reason:?}");
    let _ = s.tx.send(Event::Done {
        stop_reason: reason,
        n_tokens: s.generated.len(),
        n_prompt: s.n_prompt,
        n_cached: s.n_cached,
        elapsed_s: elapsed,
    });
}

/// Convenience: spawn the worker thread and block until it reports ready (or fails). Returns the
/// command Sender (clone into the axum state) + the loaded model names + template caps.
#[allow(clippy::type_complexity)]
pub fn spawn(models: Vec<(String, String, Option<String>)>)
    -> Result<(Sender<Cmd>, Arc<Vec<String>>, Arc<HashMap<String, ModelCaps>>, SharedMetrics), String> {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
    let (ready_tx, ready_rx) =
        std::sync::mpsc::channel::<Result<(Vec<String>, HashMap<String, ModelCaps>), String>>();
    let metrics: SharedMetrics = Default::default();
    let m2 = metrics.clone();
    std::thread::Builder::new()
        .name("memra-gpu-worker".into())
        .spawn(move || run(models, cmd_rx, ready_tx, m2))
        .map_err(|e| format!("spawn worker thread: {e}"))?;
    match ready_rx.recv() {
        Ok(Ok((names, caps))) => Ok((cmd_tx, Arc::new(names), Arc::new(caps), metrics)),
        Ok(Err(err)) => Err(err),
        Err(_) => Err("worker died during init".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::{summarize_confidence, utf8_delta};
    use super::{PoolKey, PrefixCache, PrefixEntry, PREFIX_CACHE_MIN_TOKENS};

    /// Device-free PrefixEntry (empty kv/conv/ssm planes) — the namespace-visibility laws
    /// under test live entirely in the host-side key/toks matching.
    fn entry(toks: Vec<u32>) -> PrefixEntry {
        PrefixEntry {
            toks,
            kv: Vec::new(),
            conv: Vec::new(),
            ssm: Vec::new(),
            pos: 0,
            last_logits: vec![0.0],
            bytes: 1,
            last_use: std::time::Instant::now(),
        }
    }

    fn key(ns: &str) -> PoolKey {
        ("m".to_string(), ns.to_string())
    }

    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn prefix_cache_same_namespace_same_prefix_hits() {
        let mut px = PrefixCache::default();
        let prefix = toks(PREFIX_CACHE_MIN_TOKENS);
        px.insert(&key("tenant-a"), entry(prefix.clone()), "test");
        // same namespace + same prefix (prompt extends the entry) -> hit.
        assert!(px.lookup(&key("tenant-a"), &toks(PREFIX_CACHE_MIN_TOKENS + 32)).is_some());
        assert!(px.has_covering(&key("tenant-a"), &prefix));
        assert_eq!(px.best_lcp(&key("tenant-a"), &prefix), prefix.len());
    }

    #[test]
    fn prefix_cache_namespaces_isolate_both_directions() {
        let mut px = PrefixCache::default();
        let prompt = toks(PREFIX_CACHE_MIN_TOKENS + 32);
        // tenant-a seeds; the identical prefix is INVISIBLE to tenant-b and to the
        // default namespace (a -> b direction).
        px.insert(&key("tenant-a"), entry(toks(PREFIX_CACHE_MIN_TOKENS)), "test");
        assert!(px.lookup(&key("tenant-b"), &prompt).is_none());
        assert!(px.lookup(&key(""), &prompt).is_none());
        // ... and the learning/seed signals stay scoped too (no cross-ns LCP split).
        assert_eq!(px.best_lcp(&key("tenant-b"), &prompt), 0);
        assert!(!px.has_covering(&key("tenant-b"), &prompt));
        // tenant-b seeds its OWN copy (no cross-ns dedupe: has_key is per key) and hits
        // it, while tenant-a still hits only its own (b -> a direction).
        px.insert(&key("tenant-b"), entry(toks(PREFIX_CACHE_MIN_TOKENS)), "test");
        assert_eq!(px.n_entries(), 2);
        assert!(px.lookup(&key("tenant-a"), &prompt).is_some());
        assert!(px.lookup(&key("tenant-b"), &prompt).is_some());
        assert!(px.lookup(&key("tenant-c"), &prompt).is_none());
    }

    #[test]
    fn prefix_cache_default_namespace_preserves_single_tenant_behavior() {
        // No salt = the "" namespace on every request: inserts, covering dedupe, LCP
        // learning, and longest-match lookup all behave exactly as the pre-PC-ISO
        // model-keyed cache.
        let mut px = PrefixCache::default();
        let short = toks(PREFIX_CACHE_MIN_TOKENS);
        let long = toks(PREFIX_CACHE_MIN_TOKENS + 16);
        px.insert(&key(""), entry(short.clone()), "test");
        px.insert(&key(""), entry(long.clone()), "test");
        px.insert(&key(""), entry(long.clone()), "test"); // exact-key dedupe still holds
        assert_eq!(px.n_entries(), 2);
        // longest entry prefixing the prompt wins, floor PREFIX_CACHE_MIN_TOKENS.
        let hit = px.lookup(&key(""), &toks(PREFIX_CACHE_MIN_TOKENS + 64)).unwrap();
        assert_eq!(px.entries[&key("")][hit].toks.len(), long.len());
        assert!(px.lookup(&key(""), &toks(PREFIX_CACHE_MIN_TOKENS - 1)).is_none());
    }

    // ---------------- SESSION AFFINITY (lane/session-affinity, 2026-08-05) ----------------

    /// Token-stream stand-in for a chat-template-rendered conversation. `IM` plays the
    /// template's turn-marker (control) token; every other id is ordinary text.
    const IM: u32 = 1000;
    fn is_marker(t: u32) -> bool {
        t == IM
    }
    /// Render a conversation as the flat token stream a client-side template would post:
    /// each segment = marker + its body tokens.
    fn convo(segs: &[&[u32]]) -> Vec<u32> {
        let mut v = Vec::new();
        for s in segs {
            v.push(IM);
            v.extend_from_slice(s);
        }
        v
    }
    /// Fingerprint chain of a REQUEST (the live turn is excluded from identity).
    fn fp(toks: &[u32]) -> Vec<u64> {
        super::conversation_fingerprint(toks, &is_marker, true)
    }
    /// Fingerprint chain of a PARKED session's committed stream (no live tail to drop).
    fn fp_parked(toks: &[u32]) -> Vec<u64> {
        super::conversation_fingerprint(toks, &is_marker, false)
    }
    fn shared(a: &[u64], b: &[u64]) -> usize {
        super::fingerprint_affinity(a, b)
    }
    /// A body long enough that head and tail windows do not overlap (so interior edits are
    /// genuinely invisible to the fingerprint rather than trivially absent).
    fn body(tag: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| tag * 100 + i).collect()
    }

    #[test]
    fn fingerprint_survives_an_assistant_interior_rewrite() {
        // THE lane's target case: the client strips a <think> block out of a PRIOR assistant
        // turn. Segment boundaries, roles, opening and closing tokens are unchanged; only the
        // interior shrinks. Same conversation => same fingerprint => the parked session is
        // nominated instead of discarded.
        let sys = body(1, 24);
        let user1 = body(2, 24);
        let mut asst1 = body(3, 40);
        let user2 = body(4, 24);
        let live = body(9, 8);
        let before = convo(&[&sys, &user1, &asst1, &user2, &live]);
        // strip the interior (keep >= FP_WINDOW head and tail tokens intact).
        asst1.drain(super::FP_WINDOW..asst1.len() - super::FP_WINDOW);
        let after = convo(&[&sys, &user1, &asst1, &user2, &live]);
        assert_ne!(before, after, "the rewrite must actually change the token stream");
        assert!(!after.starts_with(&before[..before.len() - 1]),
                "the rewrite must break plain prefix-extension (else the old probe would hit)");
        assert_eq!(fp(&before), fp(&after));
        assert!(fp(&before).len() >= super::FP_MIN_SEGMENTS);
    }

    #[test]
    fn fingerprint_nominates_the_parked_session_across_a_rewritten_turn() {
        // END TO END on the lane's actual case. Parked session committed turns 1-2 of a
        // conversation; the next request re-sends that history with the assistant turn's
        // interior stripped AND a new user turn appended. Plain prefix-extension is broken,
        // but the fingerprint chains share their whole leading run -> nominated.
        let (sys, user1, user2, live) = (body(1, 24), body(2, 24), body(4, 24), body(9, 8));
        let mut asst1 = body(3, 40);
        let parked = convo(&[&sys, &user1, &asst1]);
        asst1.drain(super::FP_WINDOW..asst1.len() - super::FP_WINDOW);
        let request = convo(&[&sys, &user1, &asst1, &user2, &live]);
        let n = shared(&fp(&request), &fp_parked(&parked));
        assert_eq!(n, 3, "system + user1 + rewritten assistant1 all match");
        assert!(n >= super::FP_MIN_SEGMENTS, "clears the nomination bar");
    }

    #[test]
    fn fingerprint_degrades_gracefully_when_a_rewrite_reaches_a_head_window() {
        // A think-strip can start right after the role marker, inside the head window. That
        // segment's hash changes — but identity is a PREFIX relation, so the stable opener
        // (system + early user turns, which no client rewrites) still nominates. Nomination
        // is a guess; affinity_match decides on bytes.
        let (sys, user1, user2, live) = (body(1, 24), body(2, 24), body(4, 24), body(9, 8));
        let asst1 = body(3, 40);
        let parked = convo(&[&sys, &user1, &asst1, &user2]);
        let mut wrecked = asst1.clone();
        wrecked.drain(..super::FP_WINDOW); // rewrite eats the head window too
        let request = convo(&[&sys, &user1, &wrecked, &user2, &live]);
        let n = shared(&fp(&request), &fp_parked(&parked));
        assert_eq!(n, 2, "shared run ends at the damaged segment, not at zero");
    }

    #[test]
    fn fingerprint_ignores_the_live_turn() {
        // A request's last segment is the turn being generated — new every turn by
        // construction. Two consecutive turns of one conversation share a chain.
        let (sys, user1, asst1) = (body(1, 24), body(2, 24), body(3, 24));
        let turn_a = convo(&[&sys, &user1, &asst1, &body(7, 12)]);
        let turn_b = convo(&[&sys, &user1, &asst1, &body(8, 30)]);
        assert_eq!(fp(&turn_a), fp(&turn_b));
    }

    #[test]
    fn fingerprint_separates_different_conversations() {
        // A different system prompt or a different first user turn must not clear the
        // nomination bar — affinity must never cross-link unrelated conversations.
        let (sys, user1, asst1, live) = (body(1, 24), body(2, 24), body(3, 24), body(9, 8));
        let base = fp(&convo(&[&sys, &user1, &asst1, &live]));
        let other_sys = fp(&convo(&[&body(5, 24), &user1, &asst1, &live]));
        let other_user = fp(&convo(&[&sys, &body(6, 24), &asst1, &live]));
        assert_eq!(shared(&base, &other_sys), 0, "different system prompt: nothing shared");
        assert_eq!(shared(&base, &other_user), 1, "only the system prompt is shared");
        assert!(shared(&base, &other_user) < super::FP_MIN_SEGMENTS, "below the bar");
    }

    #[test]
    fn fingerprint_declines_short_generic_openers() {
        // A bare system prompt (+ first user turn) is the SAME opener for every fresh
        // conversation with this client, so its shared run must stay under the bar.
        let sys = body(1, 24);
        let a = fp(&convo(&[&sys, &body(2, 24)]));
        let b = fp(&convo(&[&sys, &body(7, 24)]));
        assert!(shared(&a, &b) < super::FP_MIN_SEGMENTS);
        // a real multi-turn conversation does clear it.
        let long = convo(&[&sys, &body(2, 24), &body(3, 24), &body(9, 8)]);
        assert!(shared(&fp(&long), &fp(&long)) >= super::FP_MIN_SEGMENTS);
    }

    #[test]
    fn fingerprint_handles_a_prompt_with_no_markers() {
        // Raw non-chat completions (no template markers) have no segment structure: a
        // 1-segment chain, which for a request is also the live turn -> empty. Never clears
        // the bar, so those callers keep the plain prefix probes exactly as before.
        assert!(fp(&toks(512)).is_empty());
        assert!(shared(&fp(&toks(512)), &fp_parked(&toks(512))) < super::FP_MIN_SEGMENTS);
    }

    #[test]
    fn affinity_resume_requires_the_whole_committed_prefix() {
        use super::{affinity_match, AffinityMatch};
        // EXACT: the prompt carries every committed token, then new text -> prime the tail only.
        assert_eq!(
            affinity_match(&toks(100), &toks(60)),
            AffinityMatch::Exact { suffix_from: 60 }
        );
        // EXACT, empty suffix: pure continuation burst (nothing left to prime).
        assert_eq!(
            affinity_match(&toks(60), &toks(60)),
            AffinityMatch::Exact { suffix_from: 60 }
        );
    }

    #[test]
    fn affinity_refuses_to_resume_across_a_committed_range_divergence() {
        use super::{affinity_match, AffinityMatch};
        // The rewrite reached text the session ALREADY committed: the parked caches hold
        // recurrent state for tokens this request does not have, and a parked session carries
        // no checkpoint at the divergence boundary. Full re-prime — exactness over speed.
        let mut prompt = toks(100);
        prompt[42] = 999;
        assert_eq!(
            affinity_match(&prompt, &toks(60)),
            AffinityMatch::Diverged { at: 42 }
        );
        // A prompt SHORTER than committed (client dropped its own tail) is divergence too:
        // the extra committed rows cannot be trimmed away.
        assert_eq!(
            affinity_match(&toks(40), &toks(60)),
            AffinityMatch::Diverged { at: 40 }
        );
    }

    #[test]
    fn streaming_utf8_waits_for_a_complete_multibyte_sequence() {
        let mut emitted = 0;
        assert_eq!(utf8_delta(b"caf\xc3", &mut emitted), "caf");
        assert_eq!(emitted, 3);
        assert_eq!(utf8_delta(b"caf\xc3\xa9\n", &mut emitted), "é\n");
        assert_eq!(emitted, 6);
    }

    #[test]
    fn streaming_utf8_consumes_truly_invalid_bytes_once() {
        let mut emitted = 0;
        assert_eq!(utf8_delta(b"a\xffb", &mut emitted), "a\u{fffd}b");
        assert_eq!(emitted, 3);
        assert_eq!(utf8_delta(b"a\xffbc", &mut emitted), "c");
    }

    #[test]
    fn confidence_summary_tracks_reference_and_margin() {
        let summary = summarize_confidence(&[0.0, 2.0, 1.0], 1).unwrap();
        assert_eq!(summary.top1_token, 1);
        assert!(summary.top1_correct);
        assert!((summary.top1_top2_margin - 1.0).abs() < 1e-6);
        let expected = 2.0f64 - (0.0f64.exp() + 2.0f64.exp() + 1.0f64.exp()).ln();
        assert!((summary.reference_logprob - expected).abs() < 1e-12);
        assert!(summary.entropy > 0.0);
    }
}
