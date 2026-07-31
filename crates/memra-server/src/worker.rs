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

/// Per-tick prefill chunk cap: tokens primed per scheduler tick per session. Priming runs at
/// prefill throughput instead of tokenwise decode, while the per-tick cap keeps round-robin
/// latency for concurrent sessions bounded.
const PREFILL_TICK_T: usize = 1024;

/// A model loaded resident on the worker thread: weights + its own tokenizer + config snapshot.
struct LoadedModel {
    model: HybridModel,
    tok: Tokenizer,
    eos_id: u32,
}

/// What the worker streams back to one request, over its per-request tokio mpsc channel.
#[derive(Debug, Clone)]
pub enum Event {
    /// One decoded token: the raw id + the incremental text delta (detokenized tail minus prefix).
    Token { id: u32, text: String },
    /// Terminal event: why we stopped + final token count + timing.
    Done { stop_reason: String, n_tokens: usize, elapsed_s: f64 },
    /// The request could not start (bad model name, ctx full at admit, etc).
    Error(String),
}

/// A generation request submitted by an HTTP handler to the worker.
pub struct Request {
    pub model: String,
    pub prompt_ids: Vec<u32>,   // already tokenized? no — worker tokenizes (it owns the Tokenizer)
    pub prompt_text: String,
    pub chat: bool,
    pub chat_messages: Vec<(String, String)>,
    pub params: GenParams,
    pub sampler_cfg: SamplerConfig,
    pub stop_strings: Vec<String>,
    pub trace_id: Option<String>,
    /// per-request stream back to the handler. tokio mpsc so the async side can await it.
    pub tx: tokio::sync::mpsc::UnboundedSender<Event>,
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
}
pub type SharedMetrics = std::sync::Arc<std::sync::Mutex<Metrics>>;

/// Windowed percentile over decode-step latencies (ms). Engine ground truth: the worker
/// records the wall time of each batched decode tick that advanced at least one session —
/// that IS the client-visible TPOT for that tick.
struct StepStats {
    window: std::collections::VecDeque<(Instant, f32)>,
    window_s: f32,
}

impl StepStats {
    fn new(window_s: f32) -> Self {
        Self { window: std::collections::VecDeque::with_capacity(4096), window_s }
    }
    fn record(&mut self, ms: f32) {
        self.window.push_back((Instant::now(), ms));
        if self.window.len() > 16384 {
            self.window.pop_front();
        }
    }
    fn evict(&mut self) {
        let cutoff = self.window_s;
        while let Some(&(t, _)) = self.window.front() {
            if t.elapsed().as_secs_f32() > cutoff {
                self.window.pop_front();
            } else {
                break;
            }
        }
    }
    /// q in [0,100]. None until the window has signal.
    fn p(&mut self, q: f32) -> Option<f32> {
        self.evict();
        if self.window.is_empty() {
            return None;
        }
        let mut v: Vec<f32> = self.window.iter().map(|&(_, g)| g).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let i = ((q / 100.0) * (v.len() - 1) as f32).round() as usize;
        Some(v[i.min(v.len() - 1)])
    }
}

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

struct Session {
    model: String,
    /// legacy tokenwise cache — None on the spec path (SpecSession owns its own caches; the
    /// double-alloc cost 2GB/128k-session and OOM'd the 27B serve — fixed 2026-07-05).
    cache: Option<Cache>,
    /// SPEC-DECODE serving (2026-07-05): greedy sessions on MTP models decode in
    /// generate_spec_session BURSTS (K-token draft chains + batched verify) instead of one
    /// decode_step per tick — the CLI-measured spec win (27B p3: 79 vs 40 tok/s) brought to the
    /// serve path. `Some` only when: sampler greedy + model has an MTP head + MEMRA_SERVE_SPEC!=0.
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
    tx: tokio::sync::mpsc::UnboundedSender<Event>,
    t0: Instant,
}

/// The worker entry point. Runs on its OWN std::thread. Builds the Engine + loads every model on
/// THIS thread (CUDA-context affinity), then runs the scheduler loop until the command channel
/// closes. `models` = (name, gguf_path) pairs. Sends `ready_tx` once load completes (or the error).
pub fn run(
    models: Vec<(String, String, Option<String>)>,
    rx: Receiver<Cmd>,
    ready_tx: Sender<Result<Vec<String>, String>>,
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
        let (model, tok) = if std::path::Path::new(path).is_dir() {
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
        loaded.insert(name.clone(), LoadedModel { model, tok, eos_id });
        order.push(name.clone());
    }
    let _ = ready_tx.send(Ok(order.clone()));

    // ---- scheduler loop ----
    let mut active: Vec<Session> = Vec::new();
    let mut queue: std::collections::VecDeque<Box<Request>> = std::collections::VecDeque::new();
    // KV prefix-reuse pool (append-only continuation; see ReuseEntry doc).
    let mut reuse: HashMap<String, Vec<ReuseEntry>> = HashMap::new();
    let mut spec_reuse: HashMap<String, Vec<SpecReuseEntry>> = HashMap::new();

    // ---- serving counters + engine-truth step stats (30s percentile window) ----
    let mut step_stats = StepStats::new(30.0);
    let mut n_admitted = 0u64;
    let mut n_completed = 0u64;
    let mut n_tokens_out = 0u64;
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

        // 2. ADMISSION: sessions admit up to the cap; requests over the cap wait in FIFO
        //    order (never rejected). Batched scheduling decouples concurrency from batch
        //    width (decode runs ceil(N/8) chunks per tick), so its cap is a session-count
        //    knob (MEMRA_MAX_SESSIONS); the legacy MAX_ACTIVE bound applies only in
        //    round-robin mode (MEMRA_SERVE_BATCH=0).
        let max_active = if confidence_trace_enabled() { 1 } else { MAX_ACTIVE };
        let mut requeue: std::collections::VecDeque<Box<Request>> = Default::default();
        while let Some(req) = queue.pop_front() {
            let batching_on = std::env::var("MEMRA_SERVE_BATCH").map(|v| v != "0").unwrap_or(true);
            let cap = if batching_on {
                std::env::var("MEMRA_MAX_SESSIONS").ok().and_then(|v| v.parse().ok()).unwrap_or(64)
            } else {
                max_active
            };
            if active.len() >= cap {
                requeue.push_back(req);   // waits (FIFO), never rejected
                continue;
            }
            match admit(&engine, &loaded, &mut reuse, &mut spec_reuse, *req) {
                Ok(s) => { n_admitted += 1; active.push(s); }
                Err((tx, msg)) => { let _ = tx.send(Event::Error(msg)); }
            }
        }
        queue = requeue;

        // 3. The tick. Three phases (MEMRA_SERVE_BATCH=0 restores legacy round-robin):
        //    (a) spec sessions burst solo (spec x batch composition is a later step);
        //    (b) prefilling sessions prime at the full tick chunk (PREFILL_TICK_T);
        //    (c) decoding sessions advance through BATCHED steps: sample+emit host-side, then
        //        decode_step_batch over survivors in chunks of <= 8.
        let batching = {
            static B: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *B.get_or_init(|| std::env::var("MEMRA_SERVE_BATCH").map(|v| v != "0").unwrap_or(true))
        };
        let mut finished: Vec<usize> = Vec::new();
        if !batching {
            for i in 0..active.len() {
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
                    if active[i].graph.is_none() { continue; }
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
                if s.graph.is_none() && s.spec.is_none() && s.sampler.is_greedy()
                    && s.budget >= gs_min
                    && s.prefill_done && s.generated.is_empty() && s.cache.is_some()
                    && !s.last_logits.is_empty()
                {
                    let lm = &loaded[&s.model];
                    let first = memra_engine::forward::argmax(&s.last_logits) as u32;
                    let cache = s.cache.take().unwrap();
                    match lm.model.graph_session_from_cache(&engine, cache, first, s.budget + 2) {
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
                // step the (possibly just-promoted) graph session: one token per tick
                let s = &mut active[0];
                if let Some(pend) = s.graph_pending.take() {
                    let t_g = Instant::now();
                    let (cont, _) = advance_token_emit(&loaded, s, pend);
                    if !cont {
                        finished.push(0);
                    } else {
                        s.fed.push(pend);
                        let lm = &loaded[&s.model];
                        let g = s.graph.as_mut().unwrap();
                        match g.step(&engine, &lm.model) {
                            Ok(next) => { s.graph_pending = Some(next); }
                            Err(_) => { finish(s, StopReason::MaxNew); finished.push(0); }
                        }
                        n_tokens_out += 1;
                        step_stats.record(t_g.elapsed().as_secs_f32() * 1000.0);
                    }
                }
            }
            // (a) spec bursts
            for i in 0..active.len() {
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
            // (b) prefill (TTFT priority, full tick chunk).
            // task #13 (2026-07-26): BATCH fresh short primes across sessions —
            // one concat trunk, GEMMs at m = sum_T. Measured regime (prime-batch-gate --bench):
            // +80% at B=8 T=64, +44-49% at T=128, crossover ~T=320 (above it, single primes
            // win — per-seq m already at the GEMM plateau). Gate: prime-batch-gate ALL GREEN
            // (per-seq argmax + decode-stream equality). MEMRA_PRIME_BATCH=1 disables.
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
                // (prime-batch-gate --bench, B=3). PREFILL_TICK_T still caps per-tick load.
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
                            && s.fed.is_empty()
                            && s.cache.as_ref().is_some_and(|c| c.pos == 0)
                            && ql >= min_t && ql <= pb_maxt && ql <= PREFILL_TICK_T
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
                match prefill_tick(&engine, &loaded, s, PREFILL_TICK_T) {
                    Ok(_) => {}
                    Err(err) => {
                        let _ = s.tx.send(Event::Error(format!("prefill error: {err}")));
                        finished.push(i);
                    }
                }
            }
            // (c) batched decode
            let t_decode = Instant::now();
            let decoding: Vec<usize> = (0..active.len())
                .filter(|&i| !finished.contains(&i)
                        && active[i].spec.is_none() && active[i].prefill_done
                        && active[i].cache.is_some())
                .collect();
            let mut had_decode = false;
            // sample + emit + stop checks (host); survivors carry their next token
            let mut ready: Vec<(usize, u32)> = Vec::new();
            for &i in &decoding {
                let (cont, next) = advance_sample_emit(&loaded, &mut active[i]);
                match (cont, next) {
                    (false, _) => finished.push(i),
                    (true, Some(t)) => {
                        had_decode = true;
                        ready.push((i, t));
                    }
                    (true, None) => {} // nothing to do this tick
                }
            }
            // batched steps in chunks of <= 8 (the exactness-tier cap), same model per chunk
            for chunk in group_chunks(&active, &ready) {
                let toks: Vec<u32> = chunk.iter().map(|&(_, t)| t).collect();
                let idxs: Vec<usize> = chunk.iter().map(|&(i, _)| i).collect();
                let model_name = active[idxs[0]].model.clone();
                let lm = &loaded[&model_name];
                let logits = {
                    // split-borrow: pull the caches out via split_at_mut-style indexing
                    let mut caches: Vec<&mut Cache> = Vec::with_capacity(idxs.len());
                    // SAFETY: idxs are unique indices into `active`; we take disjoint &mut.
                    let base = active.as_mut_ptr();
                    for &i in &idxs {
                        let s = unsafe { &mut *base.add(i) };
                        caches.push(s.cache.as_mut().unwrap());
                    }
                    lm.model.decode_step_batch(&engine, &toks, &mut caches)
                };
                match logits {
                    Ok(rows) => {
                        for (k, &i) in idxs.iter().enumerate() {
                            active[i].last_logits = rows[k].clone();
                            active[i].fed.push(toks[k]);
                            n_tokens_out += 1;
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
            // MEMRA_TICK_TRACE=1: per-tick phase timing to stderr (diagnosis only).
            if std::env::var("MEMRA_TICK_TRACE").as_deref() == Ok("1") {
                let n_pref = active.iter().filter(|s| !s.prefill_done).count();
                eprintln!("[tick] act={} priming={} ready={} decode_ms={:.1}",
                          active.len(), n_pref, ready.len(),
                          t_decode.elapsed().as_secs_f32() * 1000.0);
            }
            // Engine-truth TPOT = the client-visible decode tick.
            if had_decode {
                step_stats.record(t_decode.elapsed().as_secs_f32() * 1000.0);
            }
        }
        // retire finished sessions (reverse order so indices stay valid). Long-enough sessions
        // park their (fed, cache, last_logits) in the reuse pool instead of dropping the cache.
        finished.sort_unstable();
        finished.dedup();
        for &i in finished.iter().rev() {
            let s = active.remove(i);
            n_completed += 1;
            if let Some(sess) = s.spec {
                if sess.committed.len() >= REUSE_MIN_PREFIX && sess.next_pred.is_some() {
                    // skip the leading BOS when rendering: the client's prompt STRING never
                    // contains it (encode() adds it), so it would poison the text-prefix match.
                    let toks = &sess.committed;
                    let skip = loaded[&s.model].tok.bos_id()
                        .map(|b| toks.first() == Some(&b)).unwrap_or(false) as usize;
                    let committed_text = loaded[&s.model].tok.decode_special(&toks[skip..], true);
                    let pool = spec_reuse.entry(s.model.clone()).or_default();
                    if pool.len() >= reuse_pool_per_model() { pool.remove(0); }
                    pool.push(SpecReuseEntry { sess, committed_text });
                }
            } else if s.fed.len() >= REUSE_MIN_PREFIX && s.prefill_done {
                if let Some(cache) = s.cache {
                    let pool = reuse.entry(s.model.clone()).or_default();
                    if pool.len() >= reuse_pool_per_model() { pool.remove(0); } // LRU: oldest first
                    let cap = cache.max_ctx;
                    pool.push(ReuseEntry {
                        fed: s.fed, cache, last_logits: s.last_logits, cap,
                    });
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
    reuse: &mut HashMap<String, Vec<ReuseEntry>>,
    spec_reuse: &mut HashMap<String, Vec<SpecReuseEntry>>,
    req: Request,
) -> Result<Session, (tokio::sync::mpsc::UnboundedSender<Event>, String)> {
    let lm = &loaded[&req.model];

    // Tokenize: prefer explicit prompt_ids (raw-id path, for the exact-token validation gate); else
    // tokenize the text, optionally wrapping in the chat template.
    let prompt: Vec<u32> = if !req.prompt_ids.is_empty() {
        req.prompt_ids.clone()
    } else if !req.chat_messages.is_empty() {
        let messages: Vec<_> = req.chat_messages.iter()
            .map(|(role, content)| (role.as_str(), content.as_str()))
            .collect();
        let rendered = lm.tok.apply_chat_template(&messages, true);
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
    let ctx_cap = req.params.max_ctx.unwrap_or(prompt.len() + req.params.max_new + 8).max(ctx_floor);
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
    if let (true, Some(pool)) = (reuse_on, reuse.get_mut(&req.model)) {
        if let Some(idx) = pool.iter().rposition(|e|
            e.fed.len() >= REUSE_MIN_PREFIX && e.cap >= ctx_cap
                && prompt.len() >= e.fed.len() && prompt.starts_with(&e.fed)) {
            reused = Some(pool.remove(idx));
        }
    }
    let (cache, seed_fed, seed_logits) = match reused {
        Some(e) => {
            eprintln!("[worker] kv-reuse: {} of {} prompt tokens resumed (model {})",
                      e.fed.len(), prompt.len(), req.model);
            (Some(e.cache), e.fed, e.last_logits)
        }
        // legacy cache deferred: allocated below ONLY if the spec path doesn't take the session.
        None => (None, Vec::new(), Vec::new()),
    };

    // EOS: union of caller-supplied eos + the model's own eos id.
    let mut params = req.params;
    if !params.eos.contains(&lm.eos_id) { params.eos.push(lm.eos_id); }

    // Suffix-only prefill on a reuse hit; sampler penalty history replayed over the whole prefix.
    let mut sampler = Sampler::new(req.sampler_cfg);
    for &t in &seed_fed { sampler.accept(t); }
    let suffix: Vec<u32> = prompt[seed_fed.len()..].to_vec();
    let prefill_done_at_admit = suffix.is_empty();
    // SPEC-DECODE serve path (2026-07-05): greedy + MTP head + not a KV-reuse resume (the spec
    // session owns its own caches; folding the reuse pool into SpecSession is a follow-up) +
    // MEMRA_SERVE_SPEC!=0. The whole prompt goes to the spec session as turn 1's suffix; the
    // legacy prefill/decode path is bypassed entirely in step_session.
    let serve_spec = !confidence_trace_enabled()
        && std::env::var("MEMRA_SERVE_SPEC").map(|v| v != "0").unwrap_or(true);
    let mut spec_resumed = 0usize;
    let mut text_suffix: Option<Vec<u32>> = None;
    // Sampled-spec serve: temperature + filters + penalties ALL ride the rejection-sampling
    // spec path (transforms applied to p and q symmetrically) — the legacy per-token path
    // remains only as the no-MTP/resume fallback.
    let spec = if serve_spec && (sampler.is_greedy() || sampler.temperature() > 0.0) && lm.model.mtp.is_some()
        && seed_fed.is_empty() {
        // POOL RESUME: a parked spec session whose committed sequence exactly prefixes this
        // prompt (with cache room) resumes — only the suffix primes; equal-length = pure burst.
        // Match order: exact token prefix (bit-clean), else TEXT prefix (survives BPE boundary
        // divergence — the ~50% chat-turn miss class). Text hits re-tokenize only the remainder.
        let resumed = spec_reuse.get_mut(&req.model).and_then(|pool| {
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
        });
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
                match lm.model.new_session(engine, ctx_cap) {
                    Ok(sess) => Some(sess),
                    Err(first_err) => {
                        let evicted = spec_reuse.get_mut(&req.model).map(|p| { let n = p.len(); p.clear(); n }).unwrap_or(0);
                        if evicted > 0 {
                            eprintln!("[worker] spec pool evicted ({evicted}) after alloc failure; retrying");
                            match lm.model.new_session(engine, ctx_cap) {
                                Ok(sess) => Some(sess),
                                Err(err) => { eprintln!("[worker] spec session alloc failed after evict ({err}); tokenwise path"); None }
                            }
                        } else {
                            eprintln!("[worker] spec session alloc failed ({first_err}); tokenwise path"); None
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
            Err(err) => return Err((req.tx, format!("cache alloc failed: {err}"))),
        },
    };
    Ok(Session {
        model: req.model,
        cache,
        sampler,
        spec,
        graph: None,
        graph_pending: None,
        spec_drafted: 0,
        spec_accepted: 0,
        last_logits: seed_logits,
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
    s: &mut Session,
    budget: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let lm = &loaded[&s.model];
    let q = s.prefill_queue.len();
    if q == 0 {
        s.prefill_done = true;
        return Ok(0);
    }
    let mut consumed = 0usize;
    if !confidence_trace_enabled()
        && q >= memra_engine::hybrid_forward::PRIME_MIN_T.max(2)
        && budget >= memra_engine::hybrid_forward::PRIME_MIN_T
    {
        let mut take = q.min(budget);
        if q - take > 0 && q - take < memra_engine::hybrid_forward::PRIME_MIN_T {
            take = if q <= budget { q } else { take };
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
    if s.prefill_queue.is_empty() { s.prefill_done = true; }
    Ok(consumed)
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
    let next = s.sampler.sample(&s.last_logits);
    s.sampler.accept(next);
    s.generated.push(next);
    if s.params.eos.contains(&next) {
        finish(s, StopReason::Eos);
        return (false, None);
    }
    let decoded = lm.tok.decode_bytes_special(&s.generated, true);
    let delta = utf8_delta(&decoded, &mut s.emitted_bytes);
    let full = String::from_utf8_lossy(&decoded);
    let _ = s.tx.send(Event::Token { id: next, text: delta });
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
    let decoded = lm.tok.decode_bytes_special(&s.generated, true);
    let delta = utf8_delta(&decoded, &mut s.emitted_bytes);
    let full = String::from_utf8_lossy(&decoded);
    let _ = s.tx.send(Event::Token { id: tok, text: delta });
    if !s.stop_strings.is_empty() && s.stop_strings.iter().any(|ss| full.contains(ss.as_str())) {
        finish(s, StopReason::Callback);
        return (false, ());
    }
    (true, ())
}

/// Group ready (session_idx, token) pairs into batched-step chunks: same model, <= 8 rows
/// (the exactness-tier cap), input order preserved (caller sorted interactive first).
fn group_chunks(active: &[Session], ready: &[(usize, u32)]) -> Vec<Vec<(usize, u32)>> {
    let mut chunks: Vec<Vec<(usize, u32)>> = Vec::new();
    for &(i, t) in ready {
        let model = &active[i].model;
        match chunks.last_mut() {
            Some(c) if c.len() < 8 && active[c[0].0].model == *model => c.push((i, t)),
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

    // ---- SPEC-BURST arm (2026-07-05): greedy MTP sessions decode in generate_spec_session
    // bursts — turn 1 primes the prompt (suffix = the whole prefill queue), later ticks are
    // ZERO-prime continuation bursts (SpecSession.next_pred). Each burst emits up to
    // SPEC_BURST_T tokens; between bursts the scheduler round-robins other sessions. Exactness:
    // the session-gate oracle (4 turns incl empty-suffix) pins burst output == fresh greedy.
    if let Some(spec) = s.spec.as_mut() {
        // Burst size trades round-robin latency (other sessions wait a whole burst) against
        // per-burst fixed cost (generate_spec_session re-runs draft-graph capture + session
        // setup every call). MEMRA_SPEC_BURST overrides for measurement; 32 = latency-safe default.
        let burst_t: usize = std::env::var("MEMRA_SPEC_BURST").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(32);
        let k: usize = std::env::var("MEMRA_SPEC_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let room = s.budget.saturating_sub(s.generated.len()).min(burst_t);
        if room == 0 { finish(s, StopReason::MaxNew); return Ok(false); }
        let suffix: Vec<u32> = s.prefill_queue.drain(..).collect();
        s.prefill_done = true;
        if suffix.is_empty() && spec.next_pred.is_none() {
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
        let (burst, d, a) = lm.model.generate_spec_session_sampled(engine, spec, &suffix, room, k, sampling)?;
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
        let decoded = lm.tok.decode_bytes_special(&s.generated, true);
        let delta = utf8_delta(&decoded, &mut s.emitted_bytes);
        let full = String::from_utf8_lossy(&decoded);
        if !delta.is_empty() {
            let _ = s.tx.send(Event::Token { id: *burst.last().unwrap_or(&0), text: delta });
        }
        if stop.is_none() && !s.stop_strings.is_empty()
            && s.stop_strings.iter().any(|ss| full.contains(ss.as_str())) {
            stop = Some(StopReason::Callback);
        }
        if stop.is_none() && s.generated.len() >= s.budget { stop = Some(StopReason::MaxNew); }
        if stop.is_none() && spec.committed.len() + k + 2 >= spec.cache_max_ctx() {
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

    let next = s.sampler.sample(&s.last_logits);
    s.sampler.accept(next);
    s.generated.push(next);

    // EOS stop (before streaming the EOS token as text — we still report it in the count).
    if s.params.eos.contains(&next) {
        finish(s, StopReason::Eos);
        return Ok(false);
    }

    // Detokenize the full generated tail, compute the incremental text delta vs what we've emitted.
    let decoded = lm.tok.decode_bytes_special(&s.generated, true);
    let delta = utf8_delta(&decoded, &mut s.emitted_bytes);
    let full = String::from_utf8_lossy(&decoded);
    let _ = s.tx.send(Event::Token { id: next, text: delta });

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

fn finish(s: &Session, reason: StopReason) {
    let elapsed = s.t0.elapsed().as_secs_f64();
    let reason = format!("{reason:?}");
    let _ = s.tx.send(Event::Done {
        stop_reason: reason,
        n_tokens: s.generated.len(),
        elapsed_s: elapsed,
    });
}

/// Convenience: spawn the worker thread and block until it reports ready (or fails). Returns the
/// command Sender (clone into the axum state) + the list of loaded model names.
pub fn spawn(models: Vec<(String, String, Option<String>)>)
    -> Result<(Sender<Cmd>, Arc<Vec<String>>, SharedMetrics), String> {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<Vec<String>, String>>();
    let metrics: SharedMetrics = Default::default();
    let m2 = metrics.clone();
    std::thread::Builder::new()
        .name("memra-gpu-worker".into())
        .spawn(move || run(models, cmd_rx, ready_tx, m2))
        .map_err(|e| format!("spawn worker thread: {e}"))?;
    match ready_rx.recv() {
        Ok(Ok(names)) => Ok((cmd_tx, Arc::new(names), metrics)),
        Ok(Err(err)) => Err(err),
        Err(_) => Err("worker died during init".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::{summarize_confidence, utf8_delta};

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
