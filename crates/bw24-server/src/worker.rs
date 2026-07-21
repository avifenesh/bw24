//! The single GPU worker thread + step-interleave scheduler (BASE-4, BW24-BUILD-MAP §4e).
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

use bw24_engine::Engine;
use bw24_engine::cache::Cache;
use bw24_engine::decode::{GenParams, StopReason};
use bw24_engine::hybrid::HybridModel;
use bw24_engine::sampler::{Sampler, SamplerConfig};
use bw24_gguf::GgufFile;
use bw24_tokenizer::Tokenizer;

/// Max concurrently-active sessions the scheduler interleaves. Admits beyond this queue (FIFO).
pub const MAX_ACTIVE: usize = 4;

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
/// callers (agent loops) always hit. Pool: at most REUSE_POOL_PER_MODEL entries per model, LRU.
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
    sess: bw24_engine::spec::SpecSession,
    /// detok(committed) — TEXT-level prefix matching (2026-07-06). Token-level starts_with
    /// missed ~50% of chat turn boundaries (detok->retok BPE merges differ at the seam). Text
    /// matching resumes whenever the new prompt string literally extends the parked
    /// conversation; only the remainder is tokenized (no BOS). Same acceptable-divergence class
    /// as llama serve's cache_prompt: the suffix's boundary tokenization may differ from a cold
    /// full-retok — committed tokens stay authoritative, spec==greedy exactness is untouched.
    committed_text: String,
}
const REUSE_POOL_PER_MODEL: usize = 2;
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
    /// serve path. `Some` only when: sampler greedy + model has an MTP head + BW24_SERVE_SPEC!=0.
    /// The SpecSession owns its OWN cache/scratch; `cache` above stays as the (unused) admit
    /// allocation on this path (kept to avoid restructuring admit; ~small VRAM overhead until
    /// a follow-up drops it). committed == every token whose state the spec caches hold.
    spec: Option<bw24_engine::spec::SpecSession>,
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
) {
    // ---- one-time init on the worker thread: Engine + all models resident ----
    let engine = match Engine::new(0) {
        Ok(e) => e,
        Err(err) => { let _ = ready_tx.send(Err(format!("Engine::new failed: {err}"))); return; }
    };
    // BW24_FAST is read ONCE here (same handling as run_gen): the matmul path consults the env var
    // per-call, but logging it once keeps the worker's behavior explicit and stable for the run.
    let fast = std::env::var("BW24_FAST").as_deref() != Ok("0");
    eprintln!("[worker] Engine ready (BW24_FAST={})", fast);

    let mut loaded: HashMap<String, LoadedModel> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for (name, path, draft) in &models {
        eprintln!("[worker] loading model {name:?} <- {path}");
        // DIRECTORY path = safetensors HF checkpoint or a manifest-backed bw24 repack/overlay;
        // file = GGUF. Repack tokenizers live in the manifest's source_dir.
        let (model, tok) = if std::path::Path::new(path).is_dir() {
            let dir = std::path::Path::new(path);
            let (src, tok_dir): (Box<dyn bw24_gguf::source::TensorSource>, std::path::PathBuf) =
                if dir.join("manifest.json").exists() {
                    let repack = match bw24_gguf::source::Hy3RepackSource::open(dir) {
                        Ok(source) => source,
                        Err(err) => { let _ = ready_tx.send(Err(format!("open {path}: {err}"))); return; }
                    };
                    let tok_dir = repack.source_dir()
                        .filter(|source| source.join("tokenizer.json").exists())
                        .unwrap_or(dir).to_path_buf();
                    (Box::new(repack), tok_dir)
                } else {
                    let st = match bw24_gguf::source::SafetensorsSource::open(dir) {
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
        // Per-model regime draft (BW24_MODELS "+<draft.gguf>" syntax): replace the embedded
        // MTP head with the standalone regime draft — same load path as BW24_MTP_DRAFT but
        // scoped to THIS model, so a multi-model server drafts each model with its own file.
        let model = {
            let mut model = model;
            if let Some(dpath) = draft {
                let dg = match GgufFile::open(dpath) {
                    Ok(g) => g,
                    Err(err) => { let _ = ready_tx.send(Err(format!("draft {name}: {err}"))); return; }
                };
                match bw24_engine::hybrid::MtpHead::load_draft(&engine, &dg, &model.cfg) {
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

        // 2. Admit queued requests into free slots (up to MAX_ACTIVE).
        let max_active = if confidence_trace_enabled() { 1 } else { MAX_ACTIVE };
        while active.len() < max_active {
            let Some(req) = queue.pop_front() else { break };
            match admit(&engine, &loaded, &mut reuse, &mut spec_reuse, *req) {
                Ok(s) => active.push(s),
                Err((tx, msg)) => { let _ = tx.send(Event::Error(msg)); }
            }
        }

        // 3. One round-robin pass: step every active session by exactly ONE decode_step.
        let mut finished: Vec<usize> = Vec::new();
        for i in 0..active.len() {
            match step_session(&engine, &loaded, &mut active[i]) {
                Ok(true) => {}                 // still running
                Ok(false) => finished.push(i), // retired this tick
                Err(err) => {
                    let _ = active[i].tx.send(Event::Error(format!("step error: {err}")));
                    finished.push(i);
                }
            }
        }
        // retire finished sessions (reverse order so indices stay valid). Long-enough sessions
        // park their (fed, cache, last_logits) in the reuse pool instead of dropping the cache.
        for &i in finished.iter().rev() {
            let s = active.remove(i);
            if let Some(sess) = s.spec {
                if sess.committed.len() >= REUSE_MIN_PREFIX && sess.next_pred.is_some() {
                    // skip the leading BOS when rendering: the client's prompt STRING never
                    // contains it (encode() adds it), so it would poison the text-prefix match.
                    let toks = &sess.committed;
                    let skip = loaded[&s.model].tok.bos_id()
                        .map(|b| toks.first() == Some(&b)).unwrap_or(false) as usize;
                    let committed_text = loaded[&s.model].tok.decode_special(&toks[skip..], true);
                    let pool = spec_reuse.entry(s.model.clone()).or_default();
                    if pool.len() >= REUSE_POOL_PER_MODEL { pool.remove(0); }
                    pool.push(SpecReuseEntry { sess, committed_text });
                }
            } else if s.fed.len() >= REUSE_MIN_PREFIX && s.prefill_done {
                if let Some(cache) = s.cache {
                    let pool = reuse.entry(s.model.clone()).or_default();
                    if pool.len() >= REUSE_POOL_PER_MODEL { pool.remove(0); } // LRU: oldest first
                    let cap = cache.max_ctx;
                    pool.push(ReuseEntry {
                        fed: s.fed, cache, last_logits: s.last_logits, cap,
                    });
                }
            }
        }
        if !finished.is_empty() && std::env::var("BW24_SPILL_STATS").as_deref() == Ok("1") {
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
    // BW24_CTX (default 8192): FLOOR for session cache allocation — per-request-sized caches can
    // never serve a LONGER continuation, which made the KV-reuse pool structurally unhittable in
    // multi-turn (parked cap 168 < next turn's need 240). Fixed-size sessions are also how the
    // reference server allocates (--ctx-size). KV cost @8192 on the 9B ≈ 119MB/session.
    let ctx_floor: usize = std::env::var("BW24_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
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
    // exactly what it validates. BW24_KV_REUSE=0 disables.
    let reuse_on = !confidence_trace_enabled()
        && std::env::var("BW24_KV_REUSE").map(|v| v != "0").unwrap_or(true);
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
    // BW24_SERVE_SPEC!=0. The whole prompt goes to the spec session as turn 1's suffix; the
    // legacy prefill/decode path is bypassed entirely in step_session.
    let serve_spec = !confidence_trace_enabled()
        && std::env::var("BW24_SERVE_SPEC").map(|v| v != "0").unwrap_or(true);
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
        // setup every call). BW24_SPEC_BURST overrides for measurement; 32 = latency-safe default.
        let burst_t: usize = std::env::var("BW24_SPEC_BURST").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(32);
        let k: usize = std::env::var("BW24_SPEC_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let room = s.budget.saturating_sub(s.generated.len()).min(burst_t);
        if room == 0 { finish(s, StopReason::MaxNew); return Ok(false); }
        let suffix: Vec<u32> = s.prefill_queue.drain(..).collect();
        s.prefill_done = true;
        if suffix.is_empty() && spec.next_pred.is_none() {
            // nothing primed and nothing to prime — shouldn't happen (admit rejects empty prompts)
            finish(s, StopReason::MaxNew); return Ok(false);
        }
        let sampling = if s.sampler.temperature() > 0.0 {
            Some(bw24_engine::spec::SpecSampling {
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
        const PREFILL_TICK_T: usize = 1024;
        let q = s.prefill_queue.len();
        if !confidence_trace_enabled() && q >= bw24_engine::hybrid_forward::PRIME_MIN_T.max(2) {
            // leave a tail chunk >= PRIME_MIN_T if this tick doesn't finish the queue
            let mut take = q.min(PREFILL_TICK_T);
            if q - take > 0 && q - take < bw24_engine::hybrid_forward::PRIME_MIN_T { take = q; }
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
    std::env::var("BW24_CONFIDENCE_TRACE").is_ok()
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
    let Ok(path) = std::env::var("BW24_CONFIDENCE_TRACE") else { return Ok(()) };
    let summary = summarize_confidence(logits, target_token).map_err(std::io::Error::other)?;
    let record = serde_json::json!({
        "format": "bw24-token-confidence-v1",
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
pub fn spawn(models: Vec<(String, String, Option<String>)>) -> Result<(Sender<Cmd>, Arc<Vec<String>>), String> {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<Vec<String>, String>>();
    std::thread::Builder::new()
        .name("bw24-gpu-worker".into())
        .spawn(move || run(models, cmd_rx, ready_tx))
        .map_err(|e| format!("spawn worker thread: {e}"))?;
    match ready_rx.recv() {
        Ok(Ok(names)) => Ok((cmd_tx, Arc::new(names))),
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
