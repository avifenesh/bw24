//! memra-server (BASE-4): a minimal OpenAI-ish HTTP server that serves 2-4 concurrent agents across
//! DIFFERENT models on one endpoint via a single GPU worker thread + step-interleave scheduler.
//!
//! Architecture (see worker.rs): axum runs on a tokio runtime; ONE dedicated std::thread owns the
//! Engine + every loaded HybridModel (CUDA context is thread-affine). Handlers submit `Cmd`s over a
//! std mpsc channel and receive tokens back over a per-request tokio mpsc channel.
//!
//! Endpoints:
//!   GET  /health                 -> {"status":"ok","models":[...]}
//!   GET  /models                 -> {"data":[{"id":name},...]}  (OpenAI-ish)
//!   GET  /metrics                -> flat serving counters + step latency percentiles.
//!   POST /v1/completions         -> {model,prompt|prompt_ids,max_tokens,temperature?,top_p?,top_k?,
//!                                     seed?,stop?,chat?,stream?,cache_salt?}. stream=true => SSE
//!                                     token-by-token; else a single JSON {text,tokens,stop_reason}.
//!   POST /v1/chat/completions    -> OpenAI chat messages rendered by the GGUF chat template;
//!                                     OpenAI message/chunk response shapes. `tools`/`tool_choice`
//!                                     (auto|none) + role:"tool" turns render through the
//!                                     template's own <tools> branch; emitted <tool_call> blocks
//!                                     parse into OpenAI `tool_calls` (+"tool_calls" finish);
//!                                     `reasoning_effort`/`reasoning` map onto the template's
//!                                     think switch (serve-tools lane, 2026-08-02).
//!
//! CONFIG: MEMRA_MODELS="name=/path.gguf[+/draft.gguf],name2=hf:owner/repo" (comma-separated;
//! `+draft.gguf` attaches that model's regime draft — docs/DRAFT-REGIME.md).
//! Defaults to the BASE-4 test pair (main=27B, judge=9B) if unset. MEMRA_ADDR sets the bind addr.

/// x-lane QoS (lane/dl-metering gate, QoS-only extraction 2026-08-02): lane types, SLO
/// admission policy, engine-truth step stats live in the memra-lanes crate so out-of-process
/// controllers (the sidecar shape) can share them.
pub(crate) mod lanes { pub use memra_lanes::*; }
mod toolcall;
mod worker;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::Sender;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{sse::{Event as SseEvent, Sse}, IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use memra_engine::decode::GenParams;
use memra_engine::sampler::SamplerConfig;
use memra_tokenizer::chat::{ThinkMode, ToolCall as TmplToolCall, Turn as TmplTurn};
use toolcall::{ParsedToolCall, Piece, ToolStreamParser};
use worker::{Cmd, Event, ModelCaps, Request, SharedMetrics};

#[derive(Clone)]
struct AppState {
    cmd_tx: Sender<Cmd>,
    models: Arc<Vec<String>>,
    caps: Arc<HashMap<String, ModelCaps>>,
    metrics: SharedMetrics,
}

/// POST /v1/completions request body.
#[derive(Deserialize)]
struct CompletionReq {
    model: String,
    #[serde(default)]
    prompt: String,
    /// raw token-id prompt (the exact-token validation-gate path; bypasses the tokenizer).
    #[serde(default)]
    prompt_ids: Vec<u32>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: f32,
    #[serde(default = "one")]
    top_p: f32,
    #[serde(default)]
    top_k: usize,
    #[serde(default)]
    min_p: f32,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    stop: StopSequences,
    /// wrap the prompt in the model's chat template (single user turn).
    #[serde(default)]
    chat: bool,
    /// stream tokens via SSE; else return one JSON when done.
    #[serde(default)]
    stream: bool,
    /// optional hard context cap.
    #[serde(default)]
    max_ctx: Option<usize>,
    /// Stable calibration-record identity written only when confidence tracing is enabled.
    #[serde(default)]
    trace_id: Option<String>,
    /// PC-ISO prefix-cache namespace (vLLM `cache_salt` convention, optional): requests
    /// only share cached prefixes with requests carrying the SAME salt. Absent/"" = the
    /// default single-tenant namespace (pre-PC-ISO behavior). See `cache_namespace`.
    #[serde(default)]
    cache_salt: Option<String>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    /// string, null, or an array of `{type:"text",text}` parts (OpenAI content shapes).
    #[serde(default)]
    content: serde_json::Value,
    /// OpenAI assistant-history tool calls, re-rendered into the template on the next turn.
    #[serde(default)]
    tool_calls: Vec<ReqToolCall>,
    /// Accepted for OpenAI-shape compat; result pairing in the template is positional.
    #[serde(default)]
    #[allow(dead_code)]
    tool_call_id: Option<String>,
}

#[derive(Deserialize)]
struct ReqToolCall {
    #[serde(default)]
    #[allow(dead_code)]
    id: Option<String>,
    function: ReqToolFunction,
}

#[derive(Deserialize)]
struct ReqToolFunction {
    name: String,
    /// OpenAI sends a JSON-encoded STRING; inline objects are accepted too.
    #[serde(default)]
    arguments: serde_json::Value,
}

#[derive(Clone, Default, Deserialize)]
#[serde(untagged)]
enum StopSequences {
    One(String),
    Many(Vec<String>),
    #[default]
    None,
}

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(stop) => vec![stop],
            Self::Many(stops) => stops,
            Self::None => Vec::new(),
        }
    }
}

/// OpenAI-compatible multi-turn chat request. `tools`/`tool_choice`/role:"tool" are accepted
/// (serve-tools lane, 2026-08-02): tool schemas render into the model chat template's own
/// <tools> branch and emitted `<tool_call>` blocks parse back into OpenAI `tool_calls` — the
/// model's GGUF chat template remains the sole source of prompt formatting, and the tools
/// path is TEMPLATE + PARSING only (zero engine changes).
#[derive(Deserialize)]
struct ChatCompletionReq {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens", alias = "max_completion_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: f32,
    #[serde(default = "one")]
    top_p: f32,
    #[serde(default)]
    top_k: usize,
    #[serde(default)]
    min_p: f32,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    stop: StopSequences,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_ctx: Option<usize>,
    /// OpenAI tool schemas: `[{"type":"function","function":{name,description?,parameters?}}]`.
    #[serde(default)]
    tools: Vec<serde_json::Value>,
    /// "auto" (default) | "none". "required"/named-function need constrained decoding -> 400.
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    /// OpenAI reasoning effort: none|minimal|low -> the template's no-think switch;
    /// medium|high -> the template default (open think). Models without a think switch
    /// ignore the parameter gracefully.
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// OpenRouter object form: {"effort": "...", "enabled": bool} (max_tokens etc ignored).
    #[serde(default)]
    reasoning: Option<serde_json::Value>,
    /// PC-ISO prefix-cache namespace (vLLM `cache_salt` convention, optional): requests
    /// only share cached prefixes with requests carrying the SAME salt. Absent/"" = the
    /// default single-tenant namespace (pre-PC-ISO behavior). See `cache_namespace`.
    #[serde(default)]
    cache_salt: Option<String>,
}
fn default_max_tokens() -> usize { 128 }
fn one() -> f32 { 1.0 }

#[derive(Serialize)]
struct CompletionResp {
    model: String,
    text: String,
    tokens: Vec<u32>,
    stop_reason: String,
    n_tokens: usize,
    /// worker-truth prompt accounting (prompt caching): total prompt tokens, and how many
    /// were served from cache (continuation pool / spec resume / cross-request prefix cache).
    prompt_tokens: usize,
    cached_tokens: usize,
    elapsed_s: f64,
}

/// OpenAI-schema usage object, shared by every response shape. `prompt_tokens_details.
/// cached_tokens` is the marketplace prompt-caching field (cache reads bill at a discount;
/// the value is worker-truth — tokens whose KV was resumed instead of computed).
fn usage_json(n_prompt: usize, n_tokens: usize, n_cached: usize, elapsed_s: f64) -> serde_json::Value {
    json!({
        "prompt_tokens": n_prompt,
        "completion_tokens": n_tokens,
        "total_tokens": n_prompt + n_tokens,
        "prompt_tokens_details": { "cached_tokens": n_cached },
        "elapsed_s": elapsed_s,
    })
}

/// OpenAI-compat mapping (2026-07-05, serve-parity arc): the pi daily client speaks
/// `openai-completions` — POST /v1/completions with the OpenAI body, expecting
/// `{choices:[{text, finish_reason, index}], usage:{...}}` and, when streaming, OpenAI SSE
/// chunks (`data: {choices:[{text}]}` ... `data: [DONE]`). pi renders the chat template
/// CLIENT-side (thinkingFormat qwen-chat-template), so raw-prompt completions is the whole
/// contract. MEMRA_COMPAT=openai (default when MEMRA_API_KEY is set — the pi setup) switches the
/// response shape; the native memra shape stays default otherwise (validation harnesses use it).
fn openai_compat() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        match std::env::var("MEMRA_COMPAT").as_deref() {
            Ok("openai") => true,
            Ok(_) => false,
            Err(_) => std::env::var("MEMRA_API_KEY").is_ok(),
        }
    })
}

/// PC-ISO (lane/pc-iso, 2026-08-02): derive the cache namespace for a request — the vLLM
/// `cache_salt` design (research/cache-tools-20260802/REPORT.md §4). Priority order:
/// (a) the explicit `cache_salt` body field (OpenAI-compatible extension); (b) a per-API-key
/// namespace would slot in here IF the auth layer ever distinguished keys — today
/// MEMRA_API_KEY is one shared bearer, a single trust domain, so there is no per-key
/// identity to fold in; (c) "" — the default single-tenant namespace, byte-identical to
/// pre-PC-ISO behavior. Cross-request KV reuse (prefix cache, continuation pool, spec pool)
/// only ever matches entries with an IDENTICAL namespace, so the `cached_tokens` hit oracle
/// can only reveal the caller's own namespace's history (CacheProbe/PROMPTPEEK mitigation).
fn cache_namespace(cache_salt: &Option<String>) -> String {
    cache_salt.clone().unwrap_or_default()
}

fn stop_reason_to_finish(r: &str) -> &'static str {
    match r {
        "Eos" | "Callback" => "stop",
        "MaxNew" | "ContextFull" => "length",
        _ => "stop",
    }
}

// ---- tools surface helpers (serve-tools lane, 2026-08-02) ----

/// Flatten an OpenAI `content` value to text: string, null (-> ""), or `{type:"text"}` parts.
fn content_to_text(v: &serde_json::Value) -> Result<String, String> {
    match v {
        serde_json::Value::Null => Ok(String::new()),
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("text") | None => match p.get("text").and_then(|t| t.as_str()) {
                        Some(t) => out.push_str(t),
                        None => return Err("content part has no text field".into()),
                    },
                    Some(other) => {
                        return Err(format!("unsupported content part type {other:?} (text only)"));
                    }
                }
            }
            Ok(out)
        }
        _ => Err("content must be a string, null, or an array of text parts".into()),
    }
}

/// Render a JSON value the way the reference template's `tojson` does (python json.dumps:
/// `", "` / `": "` separators, insertion-order keys — serde_json preserve_order — non-ASCII
/// left raw). The tools block is prompt bytes, so the training-time convention is the law.
fn pyjson(v: &serde_json::Value, out: &mut String) {
    match v {
        serde_json::Value::Object(m) => {
            out.push('{');
            for (i, (k, val)) in m.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                out.push_str(&serde_json::Value::String(k.clone()).to_string());
                out.push_str(": ");
                pyjson(val, out);
            }
            out.push('}');
        }
        serde_json::Value::Array(a) => {
            out.push('[');
            for (i, val) in a.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                pyjson(val, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

fn pyjson_str(v: &serde_json::Value) -> String {
    let mut s = String::new();
    pyjson(v, &mut s);
    s
}

#[derive(PartialEq)]
enum ToolChoice { Auto, None }

fn parse_tool_choice(v: &Option<serde_json::Value>) -> Result<ToolChoice, String> {
    match v {
        None | Some(serde_json::Value::Null) => Ok(ToolChoice::Auto),
        Some(serde_json::Value::String(s)) => match s.as_str() {
            "auto" => Ok(ToolChoice::Auto),
            "none" => Ok(ToolChoice::None),
            "required" => Err("tool_choice \"required\" is not supported (no constrained \
                               decoding); use \"auto\"".into()),
            other => Err(format!("bad tool_choice {other:?} (auto|none)")),
        },
        Some(serde_json::Value::Object(_)) =>
            Err("named-function tool_choice is not supported; use \"auto\"".into()),
        Some(other) => Err(format!("bad tool_choice: {other}")),
    }
}

/// Map OpenAI `reasoning_effort` / OpenRouter `reasoning` onto the template's think switch.
/// Absent -> the template's own default (open think on the qwen3.5/3.6 class — verified
/// against the committed template dumps; NOT overridden here, the byte-identity contract).
fn parse_think(reasoning_effort: &Option<String>, reasoning: &Option<serde_json::Value>)
    -> Result<ThinkMode, String> {
    let mut effort = reasoning_effort.clone();
    if let Some(r) = reasoning {
        match r {
            serde_json::Value::Null => {}
            serde_json::Value::Object(obj) => {
                if obj.get("enabled").and_then(|v| v.as_bool()) == Some(false) {
                    return Ok(ThinkMode::NoThink);
                }
                if let Some(e) = obj.get("effort").and_then(|v| v.as_str()) {
                    effort = Some(e.to_string());
                }
            }
            _ => return Err("reasoning must be an object".into()),
        }
    }
    match effort.as_deref() {
        None => Ok(ThinkMode::Default),
        Some("none") | Some("minimal") | Some("low") => Ok(ThinkMode::NoThink),
        Some("medium") | Some("high") => Ok(ThinkMode::Default),
        Some(other) => Err(format!(
            "bad reasoning_effort {other:?} (none|minimal|low|medium|high)")),
    }
}

/// Validate tool schemas and pre-serialize them for the template's <tools> block; also
/// extract declared parameter types (function -> parameter -> type) for argument coercion.
#[allow(clippy::type_complexity)]
fn prepare_tools(tools: &[serde_json::Value])
    -> Result<(Vec<String>, HashMap<String, HashMap<String, String>>), String> {
    let mut tools_json = Vec::with_capacity(tools.len());
    let mut schemas: HashMap<String, HashMap<String, String>> = HashMap::new();
    for t in tools {
        let f = t.get("function").ok_or("each tool needs a function object")?;
        let name = f.get("name").and_then(|n| n.as_str())
            .ok_or("each tool needs function.name")?;
        let mut params: HashMap<String, String> = HashMap::new();
        if let Some(props) = f.get("parameters").and_then(|p| p.get("properties"))
            .and_then(|p| p.as_object()) {
            for (p, def) in props {
                if let Some(ty) = def.get("type").and_then(|t| t.as_str()) {
                    params.insert(p.clone(), ty.to_string());
                }
            }
        }
        schemas.insert(name.to_string(), params);
        tools_json.push(pyjson_str(t));
    }
    Ok((tools_json, schemas))
}

/// Re-render an assistant-history tool call for the template. Value law mirrors the
/// template's `args_value | tojson if mapping/sequence else | string`: strings raw,
/// objects/arrays python-style JSON; scalars use their JSON text (`true`/`3`/`null` —
/// JSON spelling, not python's, so a parse round-trip stays self-consistent).
fn render_req_tool_call(tc: &ReqToolCall) -> Result<TmplToolCall, String> {
    let parsed: serde_json::Value = match &tc.function.arguments {
        serde_json::Value::Null => json!({}),
        serde_json::Value::String(s) if s.trim().is_empty() => json!({}),
        serde_json::Value::String(s) => serde_json::from_str(s)
            .map_err(|e| format!("tool_calls arguments is not valid JSON: {e}"))?,
        v @ serde_json::Value::Object(_) => v.clone(),
        _ => return Err("tool_calls arguments must be a JSON object".into()),
    };
    let obj = parsed.as_object()
        .ok_or("tool_calls arguments must decode to a JSON object")?;
    let params = obj.iter().map(|(k, v)| {
        let rendered = match v {
            serde_json::Value::String(s) => s.clone(),
            v @ (serde_json::Value::Object(_) | serde_json::Value::Array(_)) => pyjson_str(v),
            scalar => scalar.to_string(),
        };
        (k.clone(), rendered)
    }).collect();
    Ok(TmplToolCall { name: tc.function.name.clone(), params })
}

/// OpenAI response entry for one parsed call.
fn tool_call_json(c: &ParsedToolCall) -> serde_json::Value {
    json!({ "id": c.id, "type": "function",
            "function": { "name": c.name, "arguments": c.arguments } })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = parse_models_config();
    eprintln!("[server] starting; models config = {models:?}");

    // Spawn the GPU worker thread and block until every model is loaded (or it fails).
    let (cmd_tx, model_names, caps, metrics) = match worker::spawn(models) {
        Ok(v) => v,
        Err(err) => { eprintln!("[server] FATAL: worker init failed: {err}"); std::process::exit(1); }
    };
    eprintln!("[server] worker ready; serving models: {model_names:?}");

    let state = AppState { cmd_tx, models: model_names, caps, metrics };
    let app = Router::new()
        .route("/health", get(health))
        .route("/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/metrics", get(get_metrics))
        .route("/yield/metrics", get(yield_metrics))
        .with_state(state);

    let addr = std::env::var("MEMRA_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("[server] listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

/// MEMRA_MODELS="name=/path.gguf[+/draft.gguf],name2=hf:owner/repo". Falls back to the
/// BASE-4 test pair. `+<draft.gguf>` after a model path attaches that model's regime
/// draft (docs/DRAFT-REGIME.md) — per model, not the global MEMRA_MTP_DRAFT env, so a
/// multi-model server gives each model its own draft. Both parts accept hf: specs.
fn parse_models_config() -> Vec<(String, String, Option<String>)> {
    if let Ok(spec) = std::env::var("MEMRA_MODELS") {
        let mut out = Vec::new();
        for entry in spec.split(',').filter(|s| !s.trim().is_empty()) {
            if let Some((name, path)) = entry.split_once('=') {
                // Paths accept hf:owner/repo[:file] specs — resolved (downloaded on first
                // use) before the worker sees them.
                let (mpath, dpath) = match path.trim().split_once('+') {
                    Some((m, d)) => (m.trim(), Some(d.trim())),
                    None => (path.trim(), None),
                };
                let resolve = |p: &str| memra_gguf::hf::resolve_arg(p).unwrap_or_else(|err| {
                    eprintln!("[server] FATAL: model {name:?}: {err}");
                    std::process::exit(1);
                });
                out.push((name.trim().to_string(), resolve(mpath), dpath.map(resolve)));
            } else {
                eprintln!("[server] WARN: bad MEMRA_MODELS entry {entry:?} (want name=/path[+/draft]); skipping");
            }
        }
        if !out.is_empty() { return out; }
    }
    // Default: the BASE-4 test pair (main=27B, judge=9B).
    vec![
        ("main".into(),  "/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf".into(), None),
        ("judge".into(), "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf".into(), None),
    ]
}

async fn health(State(st): State<AppState>) -> impl IntoResponse {
    Json(json!({ "status": "ok", "models": *st.models }))
}

/// Flat serving counters + engine-truth step latency percentiles.
async fn get_metrics(State(st): State<AppState>) -> impl IntoResponse {
    let m = st.metrics.lock().map(|m| m.clone()).unwrap_or_default();
    Json(json!({
        "admitted": m.admitted,
        "completed": m.completed,
        "tokens_out": m.tokens_out,
        "step_p50_ms": m.step_p50_ms,
        "step_p99_ms": m.step_p99_ms,
        // worker-truth prompt caching split (cached = resumed from any KV cache tier).
        "prompt_tokens_in": m.prompt_tokens_in,
        "cached_tokens_in": m.cached_tokens_in,
        "prefix_cache_hits": m.prefix_hits,
        "prefix_cache_entries": m.prefix_entries,
        "prefix_cache_bytes": m.prefix_bytes,
    }))
}

async fn list_models(State(st): State<AppState>) -> impl IntoResponse {
    let data: Vec<_> = st.models.iter().map(|m| json!({ "id": m, "object": "model" })).collect();
    Json(json!({ "object": "list", "data": data }))
}

/// Per-lane counters + engine-truth interactive step latency (sidecar-compatible shape —
/// the x-lane QoS gate's receipts endpoint).
async fn yield_metrics(State(st): State<AppState>) -> impl IntoResponse {
    let m = st.metrics.lock().map(|m| m.clone()).unwrap_or_default();
    let lane = |i: usize| json!({
        "admitted": m.lane_admitted[i], "shed": m.lane_shed[i],
        "completed": m.lane_completed[i], "tokens_out": m.lane_tokens[i],
    });
    Json(json!({
        "lanes": {
            "interactive": lane(0), "judge": lane(1), "harvest": lane(2),
        },
        "interactive_step_ms": { "p50": m.step_p50_ms, "p99": m.step_p99_ms },
        "batch_size_last": m.batch_size_last,
    }))
}

/// Extract the yield lane from `x-lane` (default interactive; unknown value = 400).
fn lane_from_headers(headers: &axum::http::HeaderMap) -> Result<lanes::Lane, Response> {
    match headers.get("x-lane").map(|v| v.to_str().unwrap_or("?")) {
        None => Ok(lanes::Lane::Interactive),
        Some(v) => lanes::Lane::parse(v).ok_or_else(|| {
            (StatusCode::BAD_REQUEST,
             Json(json!({ "error": format!("unknown x-lane {v:?}") }))).into_response()
        }),
    }
}

/// Dark lanes (judge/harvest) can be SHED at admission — surface that as HTTP 429 +
/// Retry-After before committing to a streaming response. Interactive never sheds, so it
/// skips the peek (its first token may be legitimately far away; don't hold headers).
async fn peek_shed(
    lane: lanes::Lane,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
) -> Result<tokio::sync::mpsc::UnboundedReceiver<Event>, Response> {
    if lane == lanes::Lane::Interactive {
        return Ok(rx);
    }
    match rx.recv().await {
        Some(Event::Error(e)) if e.starts_with("shed:") => Err((
            StatusCode::TOO_MANY_REQUESTS,
            [(axum::http::header::RETRY_AFTER, "2")],
            Json(json!({ "error": e })),
        ).into_response()),
        first => {
            let (tx2, rx2) = tokio::sync::mpsc::unbounded_channel();
            if let Some(ev) = first {
                let _ = tx2.send(ev);
            }
            tokio::spawn(async move {
                while let Some(ev) = rx.recv().await {
                    if tx2.send(ev).is_err() { break; }
                }
            });
            Ok(rx2)
        }
    }
}

/// Build the (GenParams, SamplerConfig, stop, prompt) from a request body.
fn build_request(req: &CompletionReq, tx: tokio::sync::mpsc::UnboundedSender<Event>,
                 lane: lanes::Lane) -> Request {
    let params = GenParams {
        max_new: req.max_tokens,
        max_ctx: req.max_ctx,
        eos: Vec::new(), // worker adds the model's own eos id
    };
    let sampler_cfg = SamplerConfig {
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        min_p: req.min_p,
        seed: req.seed,
        ..Default::default()
    };
    Request {
        model: req.model.clone(),
        prompt_ids: req.prompt_ids.clone(),
        prompt_text: req.prompt.clone(),
        chat: req.chat,
        chat_turns: Vec::new(),
        tools_json: Vec::new(),
        think: ThinkMode::Default,
        params,
        sampler_cfg,
        stop_strings: req.stop.clone().into_vec(),
        trace_id: req.trace_id.clone(),
        cache_ns: cache_namespace(&req.cache_salt),
        lane,
        tx,
    }
}

/// Everything the chat handler derives from the request body before submitting to the
/// worker: the worker Request plus the parser arming state for the response side.
struct ChatPlan {
    request: Request,
    /// Some(parser) when a <tools> block was rendered — the ONLY case the emission parser
    /// runs (non-tools traffic keeps byte-identical streams, chunk boundaries included).
    parser: Option<ToolStreamParser>,
}

fn build_chat_request(req: ChatCompletionReq, caps: Option<&ModelCaps>,
                      tx: tokio::sync::mpsc::UnboundedSender<Event>,
                      lane: lanes::Lane)
                      -> Result<ChatPlan, String> {
    let tool_choice = parse_tool_choice(&req.tool_choice)?;
    let think = parse_think(&req.reasoning_effort, &req.reasoning)?;

    // tool_choice "none" = OpenAI "the model will not call tools": the prompt renders
    // WITHOUT the tools block (byte-identical to a no-tools request) and no parser runs.
    let (tools_json, schemas) = if !req.tools.is_empty() && tool_choice == ToolChoice::Auto {
        prepare_tools(&req.tools)?
    } else {
        (Vec::new(), HashMap::new())
    };

    let mut turns: Vec<TmplTurn> = Vec::with_capacity(req.messages.len());
    for msg in &req.messages {
        let content = content_to_text(&msg.content)
            .map_err(|e| format!("{} message: {e}", msg.role))?;
        let tool_calls = msg.tool_calls.iter().map(render_req_tool_call)
            .collect::<Result<Vec<_>, _>>()?;
        if !tool_calls.is_empty() && msg.role != "assistant" {
            return Err("tool_calls are only valid on assistant messages".into());
        }
        turns.push(TmplTurn { role: msg.role.clone(), content, tool_calls });
    }

    // Capability gate: reject tools on models whose template has no tools branch BEFORE
    // the request reaches the GPU worker (clean 400 instead of a mid-stream error).
    let has_tool_features = !tools_json.is_empty()
        || turns.iter().any(|t| t.role == "tool" || !t.tool_calls.is_empty());
    if has_tool_features && !caps.map(|c| c.tools_branch).unwrap_or(false) {
        return Err(format!("model {:?} chat template has no tools branch", req.model));
    }

    // Parser think gate: armed only when the rendered prompt will end with an OPEN think
    // tail (template default, not switched off by reasoning_effort on a switch-carrying
    // template) — reasoning text is content, not tool-call surface.
    let think_open = caps.map(|c| c.qwen_think
        && !(think == ThinkMode::NoThink && c.think_switch)).unwrap_or(false);
    let parser = (!tools_json.is_empty())
        .then(|| ToolStreamParser::new(schemas, think_open));

    Ok(ChatPlan {
        request: Request {
            model: req.model,
            prompt_ids: Vec::new(),
            prompt_text: String::new(),
            chat: false,
            chat_turns: turns,
            tools_json,
            think,
            params: GenParams {
                max_new: req.max_tokens,
                max_ctx: req.max_ctx,
                eos: Vec::new(),
            },
            sampler_cfg: SamplerConfig {
                temperature: req.temperature,
                top_k: req.top_k,
                top_p: req.top_p,
                min_p: req.min_p,
                seed: req.seed,
                ..Default::default()
            },
            stop_strings: req.stop.into_vec(),
            trace_id: None,
            cache_ns: cache_namespace(&req.cache_salt),
            lane,
            tx,
        },
        parser,
    })
}

fn authorized(headers: &axum::http::HeaderMap) -> bool {
    let Ok(key) = std::env::var("MEMRA_API_KEY") else { return true };
    headers.get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|candidate| candidate == key)
}

async fn completions(State(st): State<AppState>, headers: axum::http::HeaderMap,
                     Json(req): Json<CompletionReq>) -> Response {
    // API key (MEMRA_API_KEY): OpenAI-style `Authorization: Bearer <key>`. Absent env = open.
    if !authorized(&headers) {
        return (StatusCode::UNAUTHORIZED, "invalid api key").into_response();
    }
    let lane = match lane_from_headers(&headers) {
        Ok(l) => l,
        Err(resp) => return resp,
    };
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let model = req.model.clone();
    let stream = req.stream;
    let request = build_request(&req, tx, lane);
    let stop_strings = request.stop_strings.clone();

    if st.cmd_tx.send(Cmd::Generate(Box::new(request))).is_err() {
        return (StatusCode::SERVICE_UNAVAILABLE, "worker unavailable").into_response();
    }
    let rx = match peek_shed(lane, rx).await {
        Ok(rx) => rx,
        Err(resp) => return resp,
    };

    if stream {
        sse_response(rx, model, false, None).into_response()
    } else {
        blocking_response(rx, model, false, stop_strings, None).await.into_response()
    }
}

async fn chat_completions(State(st): State<AppState>, headers: axum::http::HeaderMap,
                          Json(req): Json<ChatCompletionReq>) -> Response {
    if !authorized(&headers) {
        return (StatusCode::UNAUTHORIZED, "invalid api key").into_response();
    }
    if req.messages.is_empty() || req.messages.iter().any(|message| {
        !matches!(message.role.as_str(), "system" | "user" | "assistant" | "tool")
    }) {
        return (StatusCode::BAD_REQUEST,
                Json(json!({ "error": "messages must use system/user/assistant/tool roles" })))
            .into_response();
    }
    let lane = match lane_from_headers(&headers) {
        Ok(l) => l,
        Err(resp) => return resp,
    };
    let model = req.model.clone();
    let stream = req.stream;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let plan = match build_chat_request(req, st.caps.get(&model), tx, lane) {
        Ok(plan) => plan,
        Err(err) => {
            return (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response();
        }
    };
    let stop_strings = plan.request.stop_strings.clone();
    if st.cmd_tx.send(Cmd::Generate(Box::new(plan.request))).is_err() {
        return (StatusCode::SERVICE_UNAVAILABLE, "worker unavailable").into_response();
    }
    let rx = match peek_shed(lane, rx).await {
        Ok(rx) => rx,
        Err(resp) => return resp,
    };
    if stream {
        sse_response(rx, model, true, plan.parser).into_response()
    } else {
        blocking_response(rx, model, true, stop_strings, plan.parser).await.into_response()
    }
}

/// Streaming (SSE): forward each Token as an SSE `data:` line; emit a final `done` event.
/// `parser`: Some only for tools-armed chat requests — content routes through the tool-call
/// parser and parsed calls stream as OpenAI `tool_calls` deltas (one header chunk carrying
/// id/type/name, one arguments chunk), with `finish_reason:"tool_calls"` on the final chunk.
fn sse_response(mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>, model: String, chat: bool,
                mut parser: Option<ToolStreamParser>)
    -> Sse<impl futures_core::Stream<Item = Result<SseEvent, std::convert::Infallible>>> {
    let stream = async_stream::stream! {
        let mut call_index: usize = 0;
        // renders Piece -> chat.completion.chunk payloads (tools-armed path only).
        macro_rules! piece_chunks {
            ($piece:expr) => {{
                let mut payloads: Vec<String> = Vec::new();
                match $piece {
                    Piece::Content(text) => payloads.push(
                        json!({ "object": "chat.completion.chunk", "model": model,
                                "choices": [{ "index": 0, "delta": { "content": text },
                                              "finish_reason": null }] }).to_string()),
                    Piece::Call(call) => {
                        payloads.push(json!({ "object": "chat.completion.chunk", "model": model,
                            "choices": [{ "index": 0, "delta": { "tool_calls": [{
                                "index": call_index, "id": call.id, "type": "function",
                                "function": { "name": call.name, "arguments": "" } }] },
                                "finish_reason": null }] }).to_string());
                        payloads.push(json!({ "object": "chat.completion.chunk", "model": model,
                            "choices": [{ "index": 0, "delta": { "tool_calls": [{
                                "index": call_index,
                                "function": { "arguments": call.arguments } }] },
                                "finish_reason": null }] }).to_string());
                        call_index += 1;
                    }
                }
                payloads
            }};
        }
        while let Some(ev) = rx.recv().await {
            match ev {
                Event::Token { id, text } => {
                    if let Some(p) = parser.as_mut() {
                        for piece in p.push(&text) {
                            for payload in piece_chunks!(piece) {
                                yield Ok(SseEvent::default().data(payload));
                            }
                        }
                        continue;
                    }
                    let payload = if chat {
                        json!({ "object": "chat.completion.chunk", "model": model,
                                "choices": [{ "index": 0, "delta": { "content": text },
                                              "finish_reason": null }] }).to_string()
                    } else if openai_compat() {
                        json!({ "object": "text_completion", "model": model,
                                "choices": [{ "index": 0, "text": text, "finish_reason": null }] })
                            .to_string()
                    } else {
                        json!({ "model": model, "id": id, "text": text }).to_string()
                    };
                    yield Ok(SseEvent::default().data(payload));
                }
                Event::Done { stop_reason, n_tokens, n_prompt, n_cached, elapsed_s } => {
                    let mut finish = stop_reason_to_finish(&stop_reason);
                    if let Some(p) = parser.as_mut() {
                        for piece in p.finish() {
                            for payload in piece_chunks!(piece) {
                                yield Ok(SseEvent::default().data(payload));
                            }
                        }
                        if p.n_calls() > 0 { finish = "tool_calls"; }
                    }
                    if chat || openai_compat() {
                        let usage = usage_json(n_prompt, n_tokens, n_cached, elapsed_s);
                        let fin = if chat {
                            json!({ "object": "chat.completion.chunk", "model": model,
                                "choices": [{ "index": 0, "delta": {},
                                              "finish_reason": finish }],
                                "usage": usage })
                        } else {
                            json!({ "object": "text_completion", "model": model,
                                "choices": [{ "index": 0, "text": "",
                                              "finish_reason": finish }],
                                "usage": usage })
                        }.to_string();
                        yield Ok(SseEvent::default().data(fin));
                        yield Ok(SseEvent::default().data("[DONE]".to_string()));
                    } else {
                        let payload = json!({
                            "stop_reason": stop_reason, "n_tokens": n_tokens,
                            "prompt_tokens": n_prompt, "cached_tokens": n_cached,
                            "elapsed_s": elapsed_s
                        }).to_string();
                        yield Ok(SseEvent::default().event("done").data(payload));
                    }
                    break;
                }
                Event::Error(msg) => {
                    let payload = json!({ "error": msg }).to_string();
                    yield Ok(SseEvent::default().event("error").data(payload));
                    break;
                }
            }
        }
    };
    Sse::new(stream)
}

/// Blocking JSON: collect all tokens, return one {text, tokens, stop_reason} when done.
fn truncate_at_stop(text: &mut String, stop_strings: &[String]) {
    if let Some(offset) = stop_strings.iter().filter_map(|stop| text.find(stop)).min() {
        text.truncate(offset);
    }
}

async fn blocking_response(mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>, model: String,
                           chat: bool, stop_strings: Vec<String>,
                           mut parser: Option<ToolStreamParser>) -> Response {
    let mut text = String::new();
    let mut tokens: Vec<u32> = Vec::new();
    let mut calls: Vec<ParsedToolCall> = Vec::new();
    let mut consume = |pieces: Vec<Piece>, text: &mut String, calls: &mut Vec<ParsedToolCall>| {
        for piece in pieces {
            match piece {
                Piece::Content(t) => text.push_str(&t),
                Piece::Call(c) => calls.push(c),
            }
        }
    };
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Token { id, text: delta } => {
                tokens.push(id);
                match parser.as_mut() {
                    Some(p) => consume(p.push(&delta), &mut text, &mut calls),
                    None => text.push_str(&delta),
                }
            }
            Event::Done { stop_reason, n_tokens, n_prompt, n_cached, elapsed_s } => {
                if let Some(p) = parser.as_mut() {
                    consume(p.finish(), &mut text, &mut calls);
                }
                truncate_at_stop(&mut text, &stop_strings);
                let finish = if calls.is_empty() { stop_reason_to_finish(&stop_reason) }
                             else { "tool_calls" };
                if chat {
                    // OpenAI shape: content is null on a pure tool-call turn.
                    let content = if !calls.is_empty() && text.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::Value::String(text)
                    };
                    let mut message = json!({ "role": "assistant", "content": content });
                    if !calls.is_empty() {
                        message["tool_calls"] = serde_json::Value::Array(
                            calls.iter().map(tool_call_json).collect());
                    }
                    return Json(json!({
                        "object": "chat.completion", "model": model,
                        "choices": [{ "index": 0,
                                      "message": message,
                                      "finish_reason": finish }],
                        "usage": usage_json(n_prompt, n_tokens, n_cached, elapsed_s)
                    })).into_response();
                }
                if openai_compat() {
                    return Json(json!({
                        "object": "text_completion", "model": model,
                        "choices": [{ "index": 0, "text": text,
                                      "finish_reason": finish }],
                        "usage": usage_json(n_prompt, n_tokens, n_cached, elapsed_s)
                    })).into_response();
                }
                return Json(CompletionResp {
                    model, text, tokens, stop_reason, n_tokens,
                    prompt_tokens: n_prompt, cached_tokens: n_cached, elapsed_s,
                }).into_response();
            }
            Event::Error(msg) => {
                return (StatusCode::BAD_REQUEST, Json(json!({ "error": msg }))).into_response();
            }
        }
    }
    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "worker closed stream" }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_caps() -> ModelCaps {
        ModelCaps { tools_branch: true, qwen_think: true, think_switch: true }
    }

    #[test]
    fn chat_request_preserves_turns_and_openai_stop_forms() {
        let payload = serde_json::json!({
            "model": "plain_quant",
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "work"}
            ],
            "max_tokens": 64,
            "temperature": 0.0,
            "stop": "<stop>"
        });
        let req: ChatCompletionReq = serde_json::from_value(payload).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let plan = build_chat_request(req, None, tx, lanes::Lane::Interactive).unwrap();
        let request = plan.request;
        assert!(plan.parser.is_none(), "no tools -> no parser (isolation contract)");
        assert!(request.tools_json.is_empty());
        assert_eq!(request.think, ThinkMode::Default);
        assert_eq!(request.model, "plain_quant");
        assert_eq!(request.params.max_new, 64);
        let turns: Vec<(String, String)> = request.chat_turns.iter()
            .map(|t| (t.role.clone(), t.content.clone())).collect();
        assert_eq!(turns, vec![
            ("system".into(), "rules".into()),
            ("user".into(), "task".into()),
            ("assistant".into(), "work".into()),
        ]);
        assert!(request.chat_turns.iter().all(|t| t.tool_calls.is_empty()));
        assert_eq!(request.stop_strings, vec!["<stop>"]);

        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "plain_quant", "messages": [{"role": "user", "content": "task"}],
            "stop": ["a", "b"]
        })).unwrap();
        assert_eq!(req.stop.into_vec(), vec!["a", "b"]);

        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "plain_quant", "messages": [{"role": "user", "content": "task"}],
            "stop": null
        })).unwrap();
        assert!(req.stop.into_vec().is_empty());
    }

    #[tokio::test]
    async fn chat_response_has_openai_message_shape() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Token { id: 1, text: "hello".into() }).unwrap();
        tx.send(Event::Done {
            stop_reason: "Eos".into(), n_tokens: 1, n_prompt: 42, n_cached: 30, elapsed_s: 0.5,
        }).unwrap();
        drop(tx);
        let response = blocking_response(rx, "plain_quant".into(), true, Vec::new(), None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["object"], "chat.completion");
        assert_eq!(payload["choices"][0]["message"]["role"], "assistant");
        assert_eq!(payload["choices"][0]["message"]["content"], "hello");
        assert_eq!(payload["choices"][0]["finish_reason"], "stop");
        // OpenAI prompt-caching usage schema (worker-truth cached vs computed split).
        assert_eq!(payload["usage"]["prompt_tokens"], 42);
        assert_eq!(payload["usage"]["completion_tokens"], 1);
        assert_eq!(payload["usage"]["total_tokens"], 43);
        assert_eq!(payload["usage"]["prompt_tokens_details"]["cached_tokens"], 30);
    }

    fn weather_request(extra: serde_json::Value) -> ChatCompletionReq {
        let mut payload = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object",
                               "properties": {"city": {"type": "string"},
                                              "days": {"type": "integer"}},
                               "required": ["city"]}}}],
        });
        if let Some(obj) = extra.as_object() {
            for (k, v) in obj { payload[k] = v.clone(); }
        }
        serde_json::from_value(payload).unwrap()
    }

    #[test]
    fn tools_request_renders_client_key_order_and_arms_parser() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let plan = build_chat_request(weather_request(json!({})), Some(&tool_caps()), tx, lanes::Lane::Interactive).unwrap();
        assert!(plan.parser.is_some());
        assert_eq!(plan.request.tools_json.len(), 1);
        // client key order preserved + python-dumps separators (the template's tojson law).
        assert_eq!(plan.request.tools_json[0],
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \
             \"description\": \"Get current weather\", \"parameters\": {\"type\": \"object\", \
             \"properties\": {\"city\": {\"type\": \"string\"}, \"days\": {\"type\": \
             \"integer\"}}, \"required\": [\"city\"]}}}");
    }

    #[test]
    fn tool_choice_none_strips_tools_and_parser() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let plan = build_chat_request(weather_request(json!({"tool_choice": "none"})),
                                      Some(&tool_caps()), tx, lanes::Lane::Interactive).unwrap();
        assert!(plan.parser.is_none());
        assert!(plan.request.tools_json.is_empty());
        // unsupported tool_choice forms are clean 400s, not silent downgrades.
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(build_chat_request(weather_request(json!({"tool_choice": "required"})),
                                   Some(&tool_caps()), tx, lanes::Lane::Interactive).is_err());
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(build_chat_request(weather_request(json!({"tool_choice":
            {"type": "function", "function": {"name": "get_weather"}}})),
                                   Some(&tool_caps()), tx, lanes::Lane::Interactive).is_err());
    }

    #[test]
    fn tools_on_model_without_tools_branch_is_rejected() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let caps = ModelCaps { tools_branch: false, qwen_think: false, think_switch: false };
        assert!(build_chat_request(weather_request(json!({})), Some(&caps), tx, lanes::Lane::Interactive).is_err());
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(build_chat_request(weather_request(json!({})), None, tx, lanes::Lane::Interactive).is_err());
    }

    #[test]
    fn reasoning_effort_maps_to_think_switch() {
        for (extra, want) in [
            (json!({}), ThinkMode::Default),
            (json!({"reasoning_effort": "low"}), ThinkMode::NoThink),
            (json!({"reasoning_effort": "none"}), ThinkMode::NoThink),
            (json!({"reasoning_effort": "minimal"}), ThinkMode::NoThink),
            (json!({"reasoning_effort": "high"}), ThinkMode::Default),
            (json!({"reasoning_effort": "medium"}), ThinkMode::Default),
            (json!({"reasoning": {"enabled": false}}), ThinkMode::NoThink),
            (json!({"reasoning": {"effort": "low"}}), ThinkMode::NoThink),
            (json!({"reasoning": {"enabled": true}}), ThinkMode::Default),
        ] {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let plan = build_chat_request(weather_request(extra.clone()),
                                          Some(&tool_caps()), tx, lanes::Lane::Interactive).unwrap();
            assert_eq!(plan.request.think, want, "extra={extra}");
        }
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(build_chat_request(weather_request(json!({"reasoning_effort": "extreme"})),
                                   Some(&tool_caps()), tx, lanes::Lane::Interactive).is_err());
    }

    #[test]
    fn assistant_history_tool_calls_and_tool_role_render_into_turns() {
        let payload = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_x", "type": "function", "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\": \"Paris\", \"days\": 3}"}}]},
                {"role": "tool", "tool_call_id": "call_x", "content": "{\"temp_c\": 21}"}
            ],
        });
        let req: ChatCompletionReq = serde_json::from_value(payload).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let plan = build_chat_request(req, Some(&tool_caps()), tx, lanes::Lane::Interactive).unwrap();
        let turns = &plan.request.chat_turns;
        assert_eq!(turns[1].tool_calls, vec![TmplToolCall {
            name: "get_weather".into(),
            params: vec![("city".into(), "Paris".into()), ("days".into(), "3".into())],
        }]);
        assert_eq!(turns[2].role, "tool");
        assert_eq!(turns[2].content, "{\"temp_c\": 21}");
        // no tools field on this follow-up turn: parser stays unarmed, tool turns still render.
        assert!(plan.parser.is_none());
    }

    #[tokio::test]
    async fn blocking_tools_response_carries_tool_calls_and_finish_reason() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Token { id: 1, text: "plan</think>\n\n".into() }).unwrap();
        tx.send(Event::Token { id: 2, text: "<tool_call>\n<function=get_weather>\n\
<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>".into() }).unwrap();
        tx.send(Event::Done {
            stop_reason: "Eos".into(), n_tokens: 2, n_prompt: 40, n_cached: 0, elapsed_s: 0.5,
        }).unwrap();
        drop(tx);
        let parser = ToolStreamParser::new(HashMap::new(), true);
        let response = blocking_response(rx, "m".into(), true, Vec::new(), Some(parser)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(payload["choices"][0]["message"]["content"], "plan</think>\n\n");
        let call = &payload["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        assert_eq!(call["function"]["arguments"], "{\"city\":\"Paris\"}");
        // THE INTERSECTION (integrate-cache): a tools response's usage carries the same
        // worker-truth prompt/cached split as any other shape — one source of truth.
        assert_eq!(payload["usage"]["prompt_tokens"], 40);
        assert_eq!(payload["usage"]["completion_tokens"], 2);
        assert_eq!(payload["usage"]["total_tokens"], 42);
        assert_eq!(payload["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    }

    #[test]
    fn cache_salt_plumbs_to_the_worker_namespace() {
        // PC-ISO: explicit cache_salt -> the request's cache namespace, on BOTH bodies.
        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "task", "cache_salt": "tenant-a"
        })).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert_eq!(build_request(&req, tx, lanes::Lane::Interactive).cache_ns, "tenant-a");

        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "task"}],
            "cache_salt": "tenant-b"
        })).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert_eq!(build_chat_request(req, None, tx, lanes::Lane::Interactive).unwrap().request.cache_ns, "tenant-b");

        // no salt -> "" (the default single-tenant namespace; pre-PC-ISO behavior).
        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "task"
        })).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert_eq!(build_request(&req, tx, lanes::Lane::Interactive).cache_ns, "");
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "task"}]
        })).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert_eq!(build_chat_request(req, None, tx, lanes::Lane::Interactive).unwrap().request.cache_ns, "");
    }

    #[test]
    fn completions_accept_openai_stop_forms() {
        for (value, expected) in [
            (serde_json::json!("Problem:"), vec!["Problem:"]),
            (serde_json::json!(["Question:", "Problem:"]), vec!["Question:", "Problem:"]),
            (serde_json::Value::Null, Vec::<&str>::new()),
        ] {
            let req: CompletionReq = serde_json::from_value(serde_json::json!({
                "model": "plain_quant", "prompt": "task", "stop": value
            })).unwrap();
            assert_eq!(req.stop.into_vec(), expected);
        }
    }

    #[tokio::test]
    async fn blocking_response_excludes_stop_text_across_token_events() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Token { id: 1, text: "answer\nPro".into() }).unwrap();
        tx.send(Event::Token { id: 2, text: "blem: leaked prompt".into() }).unwrap();
        tx.send(Event::Done {
            stop_reason: "Callback".into(), n_tokens: 2, n_prompt: 8, n_cached: 0, elapsed_s: 0.5,
        }).unwrap();
        drop(tx);
        let response = blocking_response(
            rx, "plain_quant".into(), false, vec!["Problem:".into()], None
        ).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["text"], "answer\n");
        assert_eq!(payload["stop_reason"], "Callback");
    }
}
