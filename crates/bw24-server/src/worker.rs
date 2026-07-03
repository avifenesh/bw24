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
    pub params: GenParams,
    pub sampler_cfg: SamplerConfig,
    pub stop_strings: Vec<String>,
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
const REUSE_POOL_PER_MODEL: usize = 2;
/// Minimum parked prefix worth reusing (below this, cold prime is cheaper than bookkeeping).
const REUSE_MIN_PREFIX: usize = 16;

struct Session {
    model: String,
    cache: Cache,
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
    /// detokenized text already emitted (to compute incremental deltas + stop-string matching).
    emitted_text: String,
    budget: usize,        // max tokens we may still generate
    tx: tokio::sync::mpsc::UnboundedSender<Event>,
    t0: Instant,
}

/// The worker entry point. Runs on its OWN std::thread. Builds the Engine + loads every model on
/// THIS thread (CUDA-context affinity), then runs the scheduler loop until the command channel
/// closes. `models` = (name, gguf_path) pairs. Sends `ready_tx` once load completes (or the error).
pub fn run(
    models: Vec<(String, String)>,
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
    let fast = std::env::var("BW24_FAST").is_ok();
    eprintln!("[worker] Engine ready (BW24_FAST={})", fast);

    let mut loaded: HashMap<String, LoadedModel> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for (name, path) in &models {
        eprintln!("[worker] loading model {name:?} <- {path}");
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
        while active.len() < MAX_ACTIVE {
            let Some(req) = queue.pop_front() else { break };
            match admit(&engine, &loaded, &mut reuse, *req) {
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
            if s.fed.len() >= REUSE_MIN_PREFIX && s.prefill_done {
                let pool = reuse.entry(s.model.clone()).or_default();
                if pool.len() >= REUSE_POOL_PER_MODEL { pool.remove(0); } // LRU: oldest first
                let cap = s.cache.max_ctx;
                pool.push(ReuseEntry {
                    fed: s.fed, cache: s.cache, last_logits: s.last_logits, cap,
                });
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
    req: Request,
) -> Result<Session, (tokio::sync::mpsc::UnboundedSender<Event>, String)> {
    let lm = &loaded[&req.model];

    // Tokenize: prefer explicit prompt_ids (raw-id path, for the exact-token validation gate); else
    // tokenize the text, optionally wrapping in the chat template.
    let prompt: Vec<u32> = if !req.prompt_ids.is_empty() {
        req.prompt_ids.clone()
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
    let ctx_cap = req.params.max_ctx.unwrap_or(prompt.len() + req.params.max_new + 8);
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
    let reuse_on = std::env::var("BW24_KV_REUSE").is_ok();  // opt-in until the identity gate runs
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
            (e.cache, e.fed, e.last_logits)
        }
        None => match Cache::new(engine, &lm.model.cfg, ctx_cap) {
            Ok(c) => (c, Vec::new(), Vec::new()),
            Err(err) => return Err((req.tx, format!("cache alloc failed: {err}"))),
        },
    };

    // EOS: union of caller-supplied eos + the model's own eos id.
    let mut params = req.params;
    if !params.eos.contains(&lm.eos_id) { params.eos.push(lm.eos_id); }

    // Suffix-only prefill on a reuse hit; sampler penalty history replayed over the whole prefix.
    let mut sampler = Sampler::new(req.sampler_cfg);
    for &t in &seed_fed { sampler.accept(t); }
    let suffix: Vec<u32> = prompt[seed_fed.len()..].to_vec();
    let prefill_done_at_admit = suffix.is_empty();
    Ok(Session {
        model: req.model,
        cache,
        sampler,
        last_logits: seed_logits,
        fed: seed_fed,
        prefill_queue: suffix.into_iter().collect(),
        prefill_done: prefill_done_at_admit,
        generated: Vec::new(),
        params,
        stop_strings: req.stop_strings,
        emitted_text: String::new(),
        budget,
        tx: req.tx,
        t0: Instant::now(),
    })
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

    // ---- prefill phase: prime exactly ONE prompt token this tick ----
    if !s.prefill_done {
        if let Some(tok) = s.prefill_queue.pop_front() {
            s.last_logits = lm.model.decode_step(engine, tok, &mut s.cache)?;
            s.fed.push(tok);
            s.sampler.accept(tok);
            if s.prefill_queue.is_empty() { s.prefill_done = true; }
        }
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
    let full = lm.tok.decode(&s.generated);
    let delta = if full.len() >= s.emitted_text.len() && full.starts_with(&s.emitted_text) {
        full[s.emitted_text.len()..].to_string()
    } else {
        // detok is not strictly prefix-stable across multibyte boundaries; fall back to last-piece.
        lm.tok.decode(&[next])
    };
    s.emitted_text = full.clone();
    let _ = s.tx.send(Event::Token { id: next, text: delta });

    // stop-string match on the detokenized tail.
    if !s.stop_strings.is_empty() && s.stop_strings.iter().any(|ss| full.contains(ss.as_str())) {
        finish(s, StopReason::Callback);
        return Ok(false);
    }

    // context guard.
    if s.cache.pos >= s.cache.max_ctx {
        finish(s, StopReason::ContextFull);
        return Ok(false);
    }

    // produce next logits (the ONE decode_step that advances this session).
    s.last_logits = lm.model.decode_step(engine, next, &mut s.cache)?;
    s.fed.push(next);
    Ok(true)
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
pub fn spawn(models: Vec<(String, String)>) -> Result<(Sender<Cmd>, Arc<Vec<String>>), String> {
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
