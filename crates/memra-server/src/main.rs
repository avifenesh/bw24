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
pub(crate) mod constrained;
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
    /// unix seconds at worker-ready — the /v1/models `created` value (when this server
    /// instance made the model available; the honest timestamp we actually know).
    started: u64,
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
    /// Omitted (gap-scan F2) => context-bounded (session ctx - prompt, model-capped), the
    /// OpenAI default-when-omitted semantics — NOT a silent 128-token truncation.
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: f32,
    #[serde(default = "one")]
    top_p: f32,
    #[serde(default)]
    top_k: usize,
    #[serde(default)]
    min_p: f32,
    /// OpenAI penalties (gap-scan F3): implemented in SamplerConfig all along, now plumbed.
    #[serde(default)]
    frequency_penalty: f32,
    #[serde(default)]
    presence_penalty: f32,
    /// OpenRouter/HF-convention multiplicative penalty (1.0 = off).
    #[serde(default = "one")]
    repetition_penalty: f32,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    stop: StopSequences,
    /// Unsupported-but-semantic fields (gap-scan F4): captured so they 400 loudly instead
    /// of being silently swallowed by serde (policy: clean 400s, not silent downgrades).
    #[serde(default)]
    logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    logprobs: Option<serde_json::Value>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    best_of: Option<usize>,
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
    /// Omitted (gap-scan F2) => context-bounded (session ctx - prompt, model-capped), the
    /// OpenAI default-when-omitted semantics — NOT a silent 128-token truncation.
    #[serde(default, alias = "max_completion_tokens")]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: f32,
    #[serde(default = "one")]
    top_p: f32,
    #[serde(default)]
    top_k: usize,
    #[serde(default)]
    min_p: f32,
    /// OpenAI penalties (gap-scan F3): implemented in SamplerConfig all along, now plumbed.
    #[serde(default)]
    frequency_penalty: f32,
    #[serde(default)]
    presence_penalty: f32,
    /// OpenRouter/HF-convention multiplicative penalty (1.0 = off).
    #[serde(default = "one")]
    repetition_penalty: f32,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    stop: StopSequences,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_ctx: Option<usize>,
    /// OpenAI `response_format` (constrained decoding, lane/constrained 2026-08-03):
    /// `{"type":"text"}` (no-op), `{"type":"json_object"}`, and
    /// `{"type":"json_schema","json_schema":{...,"schema":{...}}}` are supported — the
    /// grammar masks logits per decode step (llguidance). Unknown types 400 loudly.
    #[serde(default)]
    response_format: Option<serde_json::Value>,
    #[serde(default)]
    logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    logprobs: Option<serde_json::Value>,
    #[serde(default)]
    top_logprobs: Option<usize>,
    #[serde(default)]
    n: Option<usize>,
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
    /// OpenRouter object form: {"effort": "...", "enabled": bool, "exclude": bool}
    /// (max_tokens etc ignored).
    #[serde(default)]
    reasoning: Option<serde_json::Value>,
    /// OpenRouter legacy switch: false = think text is separated AND dropped from the
    /// response (`reasoning.exclude` in the object form does the same).
    #[serde(default)]
    include_reasoning: Option<bool>,
    /// PC-ISO prefix-cache namespace (vLLM `cache_salt` convention, optional): requests
    /// only share cached prefixes with requests carrying the SAME salt. Absent/"" = the
    /// default single-tenant namespace (pre-PC-ISO behavior). See `cache_namespace`.
    #[serde(default)]
    cache_salt: Option<String>,
}
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

// ---- OpenAI response envelope (serve-compat lane, 2026-08-03; gap-scan F1) ----
//
// The official `openai` SDKs pydantic-validate every response: `ChatCompletion` /
// `ChatCompletionChunk` REQUIRE `id: str` and `created: int`, so a response without them
// is rejected client-side before the caller ever sees the content. Every OpenAI-shape
// completion and every stream chunk therefore carries `id` + `created` +
// `system_fingerprint`; the id doubles as the `x-request-id` response header (vLLM
// convention, serving_engine.py) for support/tracing. The memra-native response shape
// (non-chat, MEMRA_COMPAT unset) is untouched — validation harnesses depend on it.

/// Backend-config fingerprint: the build's git SHA (baked by build.rs). Together with
/// `seed`, responses are checkable for determinism across deploys — the OpenAI
/// `system_fingerprint` contract.
const SYSTEM_FINGERPRINT: &str = concat!("memra-", env!("MEMRA_BUILD_SHA"));

/// 128 random-ish hex bits: two RandomState-seeded hashes over a process counter + time.
/// Uniqueness class (request ids), not crypto.
fn gen_hex128() -> String {
    use std::hash::{BuildHasher, Hasher};
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut h1 = std::collections::hash_map::RandomState::new().build_hasher();
    h1.write_u64(n);
    h1.write_u64(t);
    let mut h2 = std::collections::hash_map::RandomState::new().build_hasher();
    h2.write_u64(t.rotate_left(17));
    h2.write_u64(n);
    format!("{:016x}{:016x}", h1.finish(), h2.finish())
}

/// One request's envelope identity: the completion `id` (`chatcmpl-…` chat, `cmpl-…`
/// text) + `created` unix seconds, shared by the response and every chunk of its stream.
#[derive(Clone)]
struct Envelope {
    id: String,
    created: u64,
}

impl Envelope {
    fn new(chat: bool) -> Self {
        Envelope {
            id: format!("{}-{}", if chat { "chatcmpl" } else { "cmpl" }, gen_hex128()),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Stamp the envelope fields onto one completion/chunk payload.
    fn stamp(&self, mut v: serde_json::Value) -> serde_json::Value {
        v["id"] = json!(self.id);
        v["created"] = json!(self.created);
        v["system_fingerprint"] = json!(SYSTEM_FINGERPRINT);
        v
    }
}

/// Attach the request id as the `x-request-id` response header.
fn with_request_id(id: &str, mut resp: Response) -> Response {
    if let Ok(v) = axum::http::HeaderValue::from_str(id) {
        resp.headers_mut()
            .insert(axum::http::HeaderName::from_static("x-request-id"), v);
    }
    resp
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

/// OpenAI error body: `{"error": {"message", "type", "param", "code"}}` — the object
/// shape every OpenAI SDK parses (gap-scan F1; the old `{"error": "<string>"}` made
/// clients show a blank error). `type` follows the OpenAI vocabulary:
/// invalid_request_error / authentication_error / not_found_error / server_error.
fn error_body(message: &str, etype: &str, param: Option<&str>, code: Option<&str>)
    -> serde_json::Value {
    json!({ "error": {
        "message": message,
        "type": etype,
        "param": param,
        "code": code,
    } })
}

fn error_response(status: StatusCode, message: &str, etype: &str, param: Option<&str>)
    -> Response {
    (status, Json(error_body(message, etype, param, None))).into_response()
}

fn bad_request(message: &str, param: Option<&str>) -> Response {
    error_response(StatusCode::BAD_REQUEST, message, "invalid_request_error", param)
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

/// Sampler wiring shared by both bodies (gap-scan F3): the penalties existed in
/// SamplerConfig end-to-end (host sampler + spec rejection-sampling verify) — this is
/// pure request-struct plumbing. OpenAI penalties apply over the full context window;
/// SamplerConfig models that as a last-n history window, so an active penalty arms the
/// whole-history window (usize::MAX — `saturating_sub` makes it the full history).
fn sampler_config(temperature: f32, top_k: usize, top_p: f32, min_p: f32,
                  frequency_penalty: f32, presence_penalty: f32, repetition_penalty: f32,
                  seed: u64) -> SamplerConfig {
    let penalties_on = frequency_penalty != 0.0 || presence_penalty != 0.0
        || repetition_penalty != 1.0;
    SamplerConfig {
        temperature,
        top_k,
        top_p,
        min_p,
        penalty_last_n: if penalties_on { usize::MAX } else { 0 },
        penalty_repeat: repetition_penalty,
        penalty_freq: frequency_penalty,
        penalty_present: presence_penalty,
        seed,
    }
}

/// Honesty gate (gap-scan F4): semantic params we cannot honor are explicit 400s with the
/// offending param named — never silent downgrades (a client sending response_format:
/// json_object would get unvalidated free text and no error). Cosmetic fields (`user`,
/// `stream_options`) stay accept-and-ignore.
fn reject_unsupported(fields: &[(&str, bool, &str)]) -> Result<(), (String, String)> {
    for (param, present, why) in fields {
        if *present {
            return Err((format!("{param} is not supported{why}"), param.to_string()));
        }
    }
    Ok(())
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

    let state = AppState {
        cmd_tx, models: model_names, caps, metrics,
        started: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs()).unwrap_or(0),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/models", get(list_models))
        .route("/v1/models", get(list_models_v1))
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

/// One /v1/models entry in the OpenRouter model schema (serve-tail lane, 2026-08-04).
/// Values are worker truth from the loaded model plan (ModelCaps probed at spawn);
/// anything the plan doesn't know is an honest null, never an invented value.
/// Pricing is the self-hosted stub ("0" USD strings, the OR convention for an
/// unpriced endpoint) — a marketplace listing overrides it on the OR side.
fn model_entry_v1(name: &str, caps: Option<&ModelCaps>, created: u64) -> serde_json::Value {
    let ctx = caps.map(|c| c.context_length).filter(|&c| c > 0);
    let tokenizer = caps.map(|c| c.tokenizer.as_str()).filter(|t| !t.is_empty());
    let instruct = caps.and_then(|c| c.instruct_type.as_deref());
    json!({
        "id": name,
        "name": name,
        "object": "model",
        "created": created,
        "context_length": ctx,
        "architecture": {
            // text-only serving surface (no image/audio inputs on this server).
            "modality": "text->text",
            "tokenizer": tokenizer,
            "instruct_type": instruct,
        },
        "pricing": {
            "prompt": "0",
            "completion": "0",
            "request": "0",
            "image": "0",
        },
        "top_provider": {
            "context_length": ctx,
            // no static per-request completion cap: max_tokens is context-bounded
            // (gap-scan F2), so the honest schema value is null.
            "max_completion_tokens": serde_json::Value::Null,
        },
    })
}

/// GET /v1/models — the OpenAI/OpenRouter model listing, enriched with per-model
/// metadata from the loaded plan (context length, tokenizer, instruct family).
async fn list_models_v1(State(st): State<AppState>) -> impl IntoResponse {
    let data: Vec<_> = st.models.iter()
        .map(|m| model_entry_v1(m, st.caps.get(m), st.started))
        .collect();
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
        max_new: req.max_tokens.unwrap_or(worker::MAX_NEW_CTX_BOUNDED),
        max_ctx: req.max_ctx,
        eos: Vec::new(), // worker adds the model's own eos id
    };
    let sampler_cfg = sampler_config(
        req.temperature, req.top_k, req.top_p, req.min_p,
        req.frequency_penalty, req.presence_penalty, req.repetition_penalty, req.seed);
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
        grammar: None, // /v1/completions carries no response_format (chat surface only)
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
    let mut think = parse_think(&req.reasoning_effort, &req.reasoning)?;
    // response_format -> grammar spec (constrained decoding). None/text = unconstrained,
    // the exact legacy path; unknown/malformed forms are loud 400s.
    let grammar = constrained::parse_response_format(req.response_format.as_ref())?;
    // GRAMMAR x THINK (measured live 2026-08-03): the grammar masks from the FIRST
    // generated token, so an open <think> tail can never be closed — the forced JSON
    // lands in the think segment and `content` comes back empty. Constrained requests
    // force the template's no-think switch; a think-tail template WITHOUT the switch is
    // a loud 400 (honesty gate), not a silently broken stream.
    if grammar.is_some() {
        if let Some(c) = caps {
            if c.qwen_think && think != ThinkMode::NoThink {
                if c.think_switch {
                    think = ThinkMode::NoThink;
                } else {
                    return Err("response_format requires disabling the model's think tail, \
                                but this chat template has no enable_thinking switch".into());
                }
            }
        }
    }

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

    // Parser think gate: the rendered prompt ends with an OPEN think tail (template
    // default, not switched off by reasoning_effort on a switch-carrying template).
    let think_open = caps.map(|c| c.qwen_think
        && !(think == ThinkMode::NoThink && c.think_switch)).unwrap_or(false);
    // REASONING SEPARATION (gap-scan F13): think-segment text routes to the OpenRouter
    // `reasoning` response field on EVERY chat request against a think-open prompt —
    // content is post-think only. `include_reasoning:false` / `reasoning.exclude:true`
    // drops the separated text. Tools requests keep the full tool-call scanner; non-tools
    // think-open requests get the reasoning-only splitter (post-think text unscanned).
    // Models without a think tail keep a byte-identical no-parser stream.
    let include_reasoning = req.include_reasoning.unwrap_or(true)
        && req.reasoning.as_ref()
            .and_then(|r| r.get("exclude")).and_then(|v| v.as_bool()) != Some(true);
    let parser = if !tools_json.is_empty() {
        Some(ToolStreamParser::new(schemas, think_open)
            .with_include_reasoning(include_reasoning))
    } else if think_open {
        Some(ToolStreamParser::reasoning_only(include_reasoning))
    } else {
        None
    };

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
                max_new: req.max_tokens.unwrap_or(worker::MAX_NEW_CTX_BOUNDED),
                max_ctx: req.max_ctx,
                eos: Vec::new(),
            },
            sampler_cfg: sampler_config(
                req.temperature, req.top_k, req.top_p, req.min_p,
                req.frequency_penalty, req.presence_penalty, req.repetition_penalty,
                req.seed),
            stop_strings: req.stop.into_vec(),
            trace_id: None,
            cache_ns: cache_namespace(&req.cache_salt),
            lane,
            grammar,
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
    let env = Envelope::new(false);
    if !authorized(&headers) {
        return with_request_id(&env.id, error_response(
            StatusCode::UNAUTHORIZED, "invalid api key", "authentication_error", None));
    }
    // HONESTY GATE (gap-scan F4): semantic params we can't honor 400 loudly.
    if let Err((msg, param)) = reject_unsupported(&[
        ("logit_bias", req.logit_bias.is_some(),
         " (device-side sampling has no bias hook yet)"),
        ("logprobs", req.logprobs.is_some(), ""),
        ("n", req.n.is_some_and(|n| n != 1), " for n != 1 (single choice only)"),
        ("best_of", req.best_of.is_some_and(|n| n != 1), " (single choice only)"),
    ]) {
        return with_request_id(&env.id, bad_request(&msg, Some(&param)));
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
        return with_request_id(&env.id, error_response(
            StatusCode::SERVICE_UNAVAILABLE, "worker unavailable", "server_error", None));
    }
    let rx = match peek_shed(lane, rx).await {
        Ok(rx) => rx,
        Err(resp) => return resp,
    };

    let resp = if stream {
        sse_response(rx, model, false, None, env.clone(), stop_strings).into_response()
    } else {
        blocking_response(rx, model, false, stop_strings, None, env.clone()).await.into_response()
    };
    with_request_id(&env.id, resp)
}

async fn chat_completions(State(st): State<AppState>, headers: axum::http::HeaderMap,
                          Json(req): Json<ChatCompletionReq>) -> Response {
    let env = Envelope::new(true);
    if !authorized(&headers) {
        return with_request_id(&env.id, error_response(
            StatusCode::UNAUTHORIZED, "invalid api key", "authentication_error", None));
    }
    if req.messages.is_empty() || req.messages.iter().any(|message| {
        !matches!(message.role.as_str(), "system" | "user" | "assistant" | "tool")
    }) {
        return with_request_id(&env.id, bad_request(
            "messages must use system/user/assistant/tool roles", Some("messages")));
    }
    // HONESTY GATE (gap-scan F4): semantic params we can't honor 400 loudly, never
    // silent downgrades. response_format json_object/json_schema are now REAL
    // (constrained decoding, lane/constrained) — parsed below; bad forms 400 with the
    // parser's own message.
    if let Err((msg, param)) = reject_unsupported(&[
        ("logit_bias", req.logit_bias.is_some(),
         " (device-side sampling has no bias hook yet)"),
        ("logprobs", req.logprobs.as_ref().is_some_and(|v| v.as_bool() != Some(false)), ""),
        ("top_logprobs", req.top_logprobs.is_some(), ""),
        ("n", req.n.is_some_and(|n| n != 1), " for n != 1 (single choice only)"),
    ]) {
        return with_request_id(&env.id, bad_request(&msg, Some(&param)));
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
            return with_request_id(&env.id, bad_request(&err, None));
        }
    };
    let stop_strings = plan.request.stop_strings.clone();
    if st.cmd_tx.send(Cmd::Generate(Box::new(plan.request))).is_err() {
        return with_request_id(&env.id, error_response(
            StatusCode::SERVICE_UNAVAILABLE, "worker unavailable", "server_error", None));
    }
    let rx = match peek_shed(lane, rx).await {
        Ok(rx) => rx,
        Err(resp) => return resp,
    };
    let resp = if stream {
        sse_response(rx, model, true, plan.parser, env.clone(), stop_strings).into_response()
    } else {
        blocking_response(rx, model, true, stop_strings, plan.parser, env.clone()).await
            .into_response()
    };
    with_request_id(&env.id, resp)
}

/// Streaming (SSE): forward each Token as an SSE `data:` line; emit a final `done` event.
/// `parser`: Some only for tools-armed chat requests — content routes through the tool-call
/// parser and parsed calls stream as OpenAI `tool_calls` deltas (one header chunk carrying
/// id/type/name, one arguments chunk), with `finish_reason:"tool_calls"` on the final chunk.
/// ENVELOPE (gap-scan F1): every OpenAI-shape chunk is stamped with the request's
/// id/created/system_fingerprint; the FIRST chat delta carries `role:"assistant"` (SDK
/// stream-accumulator contract); mid-stream worker errors go out as a `data:` error chunk
/// (OpenAI clients never parse named SSE events) followed by [DONE].
fn sse_response(mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>, model: String, chat: bool,
                mut parser: Option<ToolStreamParser>, env: Envelope,
                stop_strings: Vec<String>)
    -> Sse<impl futures_core::Stream<Item = Result<SseEvent, std::convert::Infallible>>> {
    // STOP-LEAK holdback (gap-scan F9), OpenAI shapes only: content deltas buffer until
    // they can't start a stop string; matched stop text is excluded exactly like the
    // non-stream shape. The memra-native stream stays byte-identical (no scrubber).
    let mut scrub = (!stop_strings.is_empty() && (chat || openai_compat()))
        .then(|| StopScrubber::new(stop_strings));
    let stream = async_stream::stream! {
        let mut call_index: usize = 0;
        // first chat delta carries the role (applied to whatever delta comes first —
        // content, reasoning, or the tool-call header).
        let mut role_sent = false;
        macro_rules! chat_chunk {
            ($delta:expr, $finish:expr) => {{
                let mut delta = $delta;
                if chat && !role_sent {
                    role_sent = true;
                    delta["role"] = json!("assistant");
                }
                env.stamp(json!({ "object": "chat.completion.chunk", "model": model,
                                  "choices": [{ "index": 0, "delta": delta,
                                                "finish_reason": $finish }] }))
                    .to_string()
            }};
        }
        // renders Piece -> chat.completion.chunk payloads (tools-armed path only).
        macro_rules! piece_chunks {
            ($piece:expr) => {{
                let mut payloads: Vec<String> = Vec::new();
                match $piece {
                    Piece::Content(text) => {
                        let text = match scrub.as_mut() {
                            Some(sc) => sc.push(&text),
                            None => text,
                        };
                        if !text.is_empty() {
                            payloads.push(chat_chunk!(json!({ "content": text }),
                                                      serde_json::Value::Null));
                        }
                    }
                    // OR reasoning dialect (gap-scan F13): think text streams as
                    // delta.reasoning, never as content (stop strings scrub content only,
                    // same as the non-stream truncate law).
                    Piece::Reasoning(text) => payloads.push(
                        chat_chunk!(json!({ "reasoning": text }), serde_json::Value::Null)),
                    Piece::Call(call) => {
                        payloads.push(chat_chunk!(json!({ "tool_calls": [{
                            "index": call_index, "id": call.id, "type": "function",
                            "function": { "name": call.name, "arguments": "" } }] }),
                            serde_json::Value::Null));
                        payloads.push(chat_chunk!(json!({ "tool_calls": [{
                            "index": call_index,
                            "function": { "arguments": call.arguments } }] }),
                            serde_json::Value::Null));
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
                    let text = match scrub.as_mut() {
                        Some(sc) => sc.push(&text),
                        None => text,
                    };
                    if text.is_empty() && scrub.is_some() {
                        continue; // held back (possible stop prefix) or post-stop
                    }
                    let payload = if chat {
                        chat_chunk!(json!({ "content": text }), serde_json::Value::Null)
                    } else if openai_compat() {
                        env.stamp(json!({ "object": "text_completion", "model": model,
                                "choices": [{ "index": 0, "text": text, "finish_reason": null }] }))
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
                    // stop-scrubber flush: held-back text that never became a stop.
                    if let Some(sc) = scrub.as_mut() {
                        let tail = sc.finish();
                        if !tail.is_empty() {
                            let payload = if chat {
                                chat_chunk!(json!({ "content": tail }),
                                            serde_json::Value::Null)
                            } else {
                                env.stamp(json!({ "object": "text_completion",
                                    "model": model,
                                    "choices": [{ "index": 0, "text": tail,
                                                  "finish_reason": null }] })).to_string()
                            };
                            yield Ok(SseEvent::default().data(payload));
                        }
                    }
                    if chat || openai_compat() {
                        let usage = usage_json(n_prompt, n_tokens, n_cached, elapsed_s);
                        let fin = if chat {
                            let mut v = env.stamp(json!({
                                "object": "chat.completion.chunk", "model": model,
                                "choices": [{ "index": 0, "delta": {},
                                              "finish_reason": finish }],
                                "usage": usage }));
                            // zero-token stream: the role must still arrive (SDK contract).
                            if !role_sent {
                                v["choices"][0]["delta"]["role"] = json!("assistant");
                            }
                            v
                        } else {
                            env.stamp(json!({ "object": "text_completion", "model": model,
                                "choices": [{ "index": 0, "text": "",
                                              "finish_reason": finish }],
                                "usage": usage }))
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
                    if chat || openai_compat() {
                        // OpenAI clients only parse `data:` lines — a named `event: error`
                        // reads as a silent hang. Error object as the final data chunk.
                        let payload = error_body(&msg, "server_error", None, None).to_string();
                        yield Ok(SseEvent::default().data(payload));
                        yield Ok(SseEvent::default().data("[DONE]".to_string()));
                    } else {
                        let payload = json!({ "error": msg }).to_string();
                        yield Ok(SseEvent::default().event("error").data(payload));
                    }
                    break;
                }
            }
        }
    };
    Sse::new(stream).keep_alive(
        // OR cancels + fails over on silent phases (fetch timeout) — long-prompt prefill
        // streams nothing for many seconds before first token. SSE comment every 5s.
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(5)),
    )
}

/// Blocking JSON: collect all tokens, return one {text, tokens, stop_reason} when done.
fn truncate_at_stop(text: &mut String, stop_strings: &[String]) {
    if let Some(offset) = stop_strings.iter().filter_map(|stop| text.find(stop)).min() {
        text.truncate(offset);
    }
}

/// Longest PROPER prefix of `tag` (on tag char boundaries) that `s` ends with — the
/// char-boundary-safe twin of toolcall's ASCII-tag helper (stop strings are client text).
fn partial_stop_suffix(s: &str, tag: &str) -> usize {
    let mut best = 0;
    for (k, _) in tag.char_indices().skip(1) {
        if k <= s.len() && s.ends_with(&tag[..k]) {
            best = k;
        }
    }
    best
}

/// STREAMING STOP SCRUBBER (gap-scan F9): the worker emits the token delta BEFORE its
/// stop check, so streams used to leak the stop text (and same-token overshoot) that
/// non-stream clients never see. Content deltas route through this holdback buffer:
/// text is released only once it can no longer be the start of a stop string, and a
/// completed stop truncates exactly like the non-stream `truncate_at_stop`.
struct StopScrubber {
    stops: Vec<String>,
    buf: String,
    done: bool,
}

impl StopScrubber {
    fn new(stops: Vec<String>) -> Self {
        Self { stops, buf: String::new(), done: false }
    }

    /// Feed a content delta; returns the text now safe to emit.
    fn push(&mut self, text: &str) -> String {
        if self.done {
            return String::new();
        }
        self.buf.push_str(text);
        if let Some(i) = self.stops.iter().filter_map(|s| self.buf.find(s.as_str())).min() {
            self.done = true;
            let out = self.buf[..i].to_string();
            self.buf.clear();
            return out;
        }
        let keep = self.stops.iter()
            .map(|s| partial_stop_suffix(&self.buf, s)).max().unwrap_or(0);
        let emit_to = self.buf.len() - keep;
        let out = self.buf[..emit_to].to_string();
        self.buf.drain(..emit_to);
        out
    }

    /// End of stream: release held-back text (it never became a stop).
    fn finish(&mut self) -> String {
        if self.done {
            self.buf.clear();
            return String::new();
        }
        std::mem::take(&mut self.buf)
    }
}

async fn blocking_response(mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>, model: String,
                           chat: bool, stop_strings: Vec<String>,
                           mut parser: Option<ToolStreamParser>, env: Envelope) -> Response {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tokens: Vec<u32> = Vec::new();
    let mut calls: Vec<ParsedToolCall> = Vec::new();
    let consume = |pieces: Vec<Piece>, text: &mut String, reasoning: &mut String,
                   calls: &mut Vec<ParsedToolCall>| {
        for piece in pieces {
            match piece {
                Piece::Content(t) => text.push_str(&t),
                Piece::Reasoning(t) => reasoning.push_str(&t),
                Piece::Call(c) => calls.push(c),
            }
        }
    };
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Token { id, text: delta } => {
                tokens.push(id);
                match parser.as_mut() {
                    Some(p) => consume(p.push(&delta), &mut text, &mut reasoning, &mut calls),
                    None => text.push_str(&delta),
                }
            }
            Event::Done { stop_reason, n_tokens, n_prompt, n_cached, elapsed_s } => {
                if let Some(p) = parser.as_mut() {
                    consume(p.finish(), &mut text, &mut reasoning, &mut calls);
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
                    // OR reasoning dialect (gap-scan F13): think text is a dedicated
                    // message field (+ reasoning_details), content is post-think only.
                    if !reasoning.is_empty() {
                        message["reasoning"] = json!(reasoning);
                        message["reasoning_details"] = json!([{
                            "type": "reasoning.text", "text": reasoning }]);
                    }
                    if !calls.is_empty() {
                        message["tool_calls"] = serde_json::Value::Array(
                            calls.iter().map(tool_call_json).collect());
                    }
                    return Json(env.stamp(json!({
                        "object": "chat.completion", "model": model,
                        "choices": [{ "index": 0,
                                      "message": message,
                                      "finish_reason": finish }],
                        "usage": usage_json(n_prompt, n_tokens, n_cached, elapsed_s)
                    }))).into_response();
                }
                if openai_compat() {
                    return Json(env.stamp(json!({
                        "object": "text_completion", "model": model,
                        "choices": [{ "index": 0, "text": text,
                                      "finish_reason": finish }],
                        "usage": usage_json(n_prompt, n_tokens, n_cached, elapsed_s)
                    }))).into_response();
                }
                return Json(CompletionResp {
                    model, text, tokens, stop_reason, n_tokens,
                    prompt_tokens: n_prompt, cached_tokens: n_cached, elapsed_s,
                }).into_response();
            }
            Event::Error(msg) => {
                return bad_request(&msg, None);
            }
        }
    }
    error_response(StatusCode::INTERNAL_SERVER_ERROR, "worker closed stream", "server_error", None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_caps() -> ModelCaps {
        ModelCaps {
            tools_branch: true, qwen_think: true, think_switch: true,
            ..Default::default()
        }
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
        // OMITTED max_tokens (gap-scan F2): the context-bounded sentinel, not 128.
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "plain_quant", "messages": [{"role": "user", "content": "task"}]
        })).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let plan = build_chat_request(req, None, tx, lanes::Lane::Interactive).unwrap();
        assert_eq!(plan.request.params.max_new, worker::MAX_NEW_CTX_BOUNDED);
        // max_completion_tokens alias still honored exactly.
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "plain_quant", "messages": [{"role": "user", "content": "task"}],
            "max_completion_tokens": 7
        })).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert_eq!(build_chat_request(req, None, tx, lanes::Lane::Interactive).unwrap().request.params.max_new, 7);
        // completions body: same omission law.
        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "plain_quant", "prompt": "task"
        })).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert_eq!(build_request(&req, tx, lanes::Lane::Interactive).params.max_new, worker::MAX_NEW_CTX_BOUNDED);
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
        let response = blocking_response(rx, "plain_quant".into(), true, Vec::new(), None,
                                         Envelope::new(true)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["object"], "chat.completion");
        // OpenAI envelope (gap-scan F1): the official SDK pydantic-REQUIRES id + created.
        assert!(payload["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert!(payload["created"].as_u64().unwrap() > 1_700_000_000);
        assert!(payload["system_fingerprint"].as_str().unwrap().starts_with("memra-"));
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
        // tools stripped: no tool-call scanning; the think-open prompt still arms the
        // reasoning-only splitter (F13) — a <tool_call> in post-think prose stays prose.
        let mut p = plan.parser.expect("think-open chat arms the reasoning splitter");
        let pieces = p.push("x</think>\n\n<tool_call> stays prose");
        assert_eq!(pieces, vec![
            Piece::Reasoning("x".into()),
            Piece::Content("<tool_call> stays prose".into()),
        ]);
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
        let caps = ModelCaps::default();
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
        // no tools field on this follow-up turn: no tool-call scanning — but the think-open
        // prompt still arms the reasoning-only splitter (gap-scan F13).
        let mut p = plan.parser.expect("think-open chat arms the reasoning splitter");
        let pieces = p.push("thought</think>\n\nanswer <tool_call> is prose here");
        assert_eq!(pieces, vec![
            Piece::Reasoning("thought".into()),
            Piece::Content("answer <tool_call> is prose here".into()),
        ]);
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
        let response = blocking_response(rx, "m".into(), true, Vec::new(), Some(parser),
                                         Envelope::new(true)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["choices"][0]["finish_reason"], "tool_calls");
        // reasoning separation (gap-scan F13): think text -> message.reasoning (+details),
        // content is post-think only (null here — a pure tool-call turn).
        assert_eq!(payload["choices"][0]["message"]["content"], serde_json::Value::Null);
        assert_eq!(payload["choices"][0]["message"]["reasoning"], "plan");
        assert_eq!(payload["choices"][0]["message"]["reasoning_details"][0]["text"], "plan");
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

    /// Drain an Sse response into its `data:` payload lines (keep-alive comments skipped).
    async fn sse_data_lines(resp: Response) -> Vec<String> {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
            .lines()
            .filter_map(|l| l.strip_prefix("data: ").map(str::to_string))
            .collect()
    }

    #[tokio::test]
    async fn stream_chunks_carry_envelope_and_first_delta_role() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Token { id: 1, text: "he".into() }).unwrap();
        tx.send(Event::Token { id: 2, text: "llo".into() }).unwrap();
        tx.send(Event::Done {
            stop_reason: "Eos".into(), n_tokens: 2, n_prompt: 10, n_cached: 0, elapsed_s: 0.1,
        }).unwrap();
        drop(tx);
        let resp = sse_response(rx, "m".into(), true, None, Envelope::new(true), Vec::new())
            .into_response();
        let lines = sse_data_lines(resp).await;
        assert_eq!(lines.last().map(String::as_str), Some("[DONE]"));
        let chunks: Vec<serde_json::Value> = lines[..lines.len() - 1].iter()
            .map(|l| serde_json::from_str(l).unwrap()).collect();
        // every chunk: id (chatcmpl-, SAME id) + created + system_fingerprint + object.
        let id = chunks[0]["id"].as_str().unwrap().to_string();
        assert!(id.starts_with("chatcmpl-"));
        for c in &chunks {
            assert_eq!(c["id"], id.as_str());
            assert!(c["created"].as_u64().unwrap() > 1_700_000_000);
            assert!(c["system_fingerprint"].as_str().unwrap().starts_with("memra-"));
            assert_eq!(c["object"], "chat.completion.chunk");
        }
        // FIRST delta carries role:"assistant" (SDK accumulator contract); later ones don't.
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "he");
        assert!(chunks[1]["choices"][0]["delta"].get("role").is_none());
        // final chunk: finish_reason + usage.
        let fin = chunks.last().unwrap();
        assert_eq!(fin["choices"][0]["finish_reason"], "stop");
        assert_eq!(fin["usage"]["prompt_tokens"], 10);
    }

    #[tokio::test]
    async fn stream_excludes_stop_text_like_non_stream_does() {
        // gap-scan F9: the worker emits the delta BEFORE its stop check — the stream
        // shape must still exclude the stop text (and same-token overshoot) exactly
        // like the non-stream truncate. Stop spans two token events here.
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Token { id: 1, text: "answer\nPro".into() }).unwrap();
        tx.send(Event::Token { id: 2, text: "blem: leaked prompt".into() }).unwrap();
        tx.send(Event::Done {
            stop_reason: "Callback".into(), n_tokens: 2, n_prompt: 8, n_cached: 0,
            elapsed_s: 0.1,
        }).unwrap();
        drop(tx);
        let resp = sse_response(rx, "m".into(), true, None, Envelope::new(true),
                                vec!["Problem:".into()]).into_response();
        let lines = sse_data_lines(resp).await;
        let content: String = lines.iter()
            .filter(|l| *l != "[DONE]")
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str()
                .map(str::to_string))
            .collect();
        assert_eq!(content, "answer\n");

        // held-back text that never becomes a stop is flushed at Done.
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Token { id: 1, text: "ends in Pro".into() }).unwrap();
        tx.send(Event::Done {
            stop_reason: "Eos".into(), n_tokens: 1, n_prompt: 8, n_cached: 0, elapsed_s: 0.1,
        }).unwrap();
        drop(tx);
        let resp = sse_response(rx, "m".into(), true, None, Envelope::new(true),
                                vec!["Problem:".into()]).into_response();
        let lines = sse_data_lines(resp).await;
        let content: String = lines.iter()
            .filter(|l| *l != "[DONE]")
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str()
                .map(str::to_string))
            .collect();
        assert_eq!(content, "ends in Pro");
    }

    #[tokio::test]
    async fn stream_worker_error_is_a_data_chunk_not_a_named_event() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Error("boom".into())).unwrap();
        drop(tx);
        let resp = sse_response(rx, "m".into(), true, None, Envelope::new(true), Vec::new())
            .into_response();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        // OpenAI clients only parse `data:` lines — no named `event: error` on the chat shape.
        assert!(!body.contains("event: error"), "named SSE event leaked: {body}");
        let lines: Vec<&str> = body.lines()
            .filter_map(|l| l.strip_prefix("data: ")).collect();
        let err: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(err["error"]["message"], "boom");
        assert_eq!(err["error"]["type"], "server_error");
        assert_eq!(lines.last(), Some(&"[DONE]"));
    }

    #[tokio::test]
    async fn error_bodies_use_the_openai_object_shape() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Error("unknown model \"x\"".into())).unwrap();
        drop(tx);
        let response = blocking_response(rx, "m".into(), true, Vec::new(), None,
                                         Envelope::new(true)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // {"error": {message, type, param, code}} — the object every OpenAI SDK parses.
        assert_eq!(payload["error"]["message"], "unknown model \"x\"");
        assert_eq!(payload["error"]["type"], "invalid_request_error");
        assert!(payload["error"].get("param").is_some());
        assert!(payload["error"].get("code").is_some());
    }

    #[test]
    fn penalties_plumb_from_http_to_sampler_config() {
        // gap-scan F3: the fields existed in SamplerConfig all along — assert the HTTP
        // layer actually delivers them, with the whole-history window armed.
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "task"}],
            "frequency_penalty": 0.5, "presence_penalty": 0.25, "repetition_penalty": 1.1
        })).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let cfg = build_chat_request(req, None, tx, lanes::Lane::Interactive).unwrap().request.sampler_cfg;
        assert_eq!(cfg.penalty_freq, 0.5);
        assert_eq!(cfg.penalty_present, 0.25);
        assert_eq!(cfg.penalty_repeat, 1.1);
        assert_eq!(cfg.penalty_last_n, usize::MAX);

        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "task", "frequency_penalty": 1.5
        })).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let cfg = build_request(&req, tx, lanes::Lane::Interactive).sampler_cfg;
        assert_eq!(cfg.penalty_freq, 1.5);
        assert_eq!(cfg.penalty_last_n, usize::MAX);

        // no penalties set -> window off, byte-identical legacy config.
        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "task"
        })).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let cfg = build_request(&req, tx, lanes::Lane::Interactive).sampler_cfg;
        assert_eq!(cfg.penalty_last_n, 0);
        assert_eq!(cfg.penalty_repeat, 1.0);
    }

    #[test]
    fn response_format_builds_grammar_only_when_present() {
        // NO-OP CONTRACT (lane/constrained): absent / {"type":"text"} => grammar None —
        // the worker Request is field-identical to a pre-lane request, no llguidance
        // object is ever built. json_object / json_schema arm the grammar.
        let mk = |rf: Option<serde_json::Value>| {
            let mut body = serde_json::json!({
                "model": "m", "messages": [{"role": "user", "content": "t"}]});
            if let Some(rf) = rf { body["response_format"] = rf; }
            let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            build_chat_request(req, None, tx, lanes::Lane::Interactive)
        };
        assert!(mk(None).unwrap().request.grammar.is_none());
        assert!(mk(Some(serde_json::json!({"type": "text"})))
            .unwrap().request.grammar.is_none());
        assert!(matches!(mk(Some(serde_json::json!({"type": "json_object"})))
            .unwrap().request.grammar,
            Some(constrained::GrammarSpec::JsonObject)));
        assert!(matches!(mk(Some(serde_json::json!({"type": "json_schema",
            "json_schema": {"schema": {"type": "object"}}})))
            .unwrap().request.grammar,
            Some(constrained::GrammarSpec::JsonSchema(_))));
        // unknown type: loud error, never silent.
        assert!(mk(Some(serde_json::json!({"type": "yaml"}))).is_err());
    }

    #[test]
    fn unsupported_semantic_params_are_named_rejections() {
        // gap-scan F4: fields serde used to swallow now deserialize into rejection slots.
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "t"}],
            "response_format": {"type": "json_object"}
        })).unwrap();
        assert!(req.response_format.is_some());
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "t"}],
            "response_format": {"type": "text"}, "logprobs": false, "n": 1,
            "user": "u-1", "stream_options": {"include_usage": true}
        })).unwrap();
        // the no-op forms + cosmetic fields: all fine (accept-and-ignore class).
        assert_eq!(req.response_format.as_ref().unwrap()["type"], "text");
        assert_eq!(req.logprobs.as_ref().unwrap().as_bool(), Some(false));
        assert_eq!(req.n, Some(1));
        // the gate law itself: present -> named error, absent -> Ok.
        assert!(reject_unsupported(&[("logit_bias", false, "")]).is_ok());
        let (msg, param) = reject_unsupported(&[("logit_bias", true, " (why)")]).unwrap_err();
        assert_eq!(param, "logit_bias");
        assert_eq!(msg, "logit_bias is not supported (why)");
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

    #[test]
    fn v1_models_entry_matches_or_schema_with_honest_nulls() {
        // KNOWN plan metadata populates every OR-schema field from worker truth.
        let caps = ModelCaps {
            tools_branch: true, qwen_think: true, think_switch: true,
            context_length: 262144,
            tokenizer: "qwen2".into(),
            instruct_type: Some("chatml".into()),
        };
        let e = model_entry_v1("main", Some(&caps), 1_754_000_000);
        assert_eq!(e["id"], "main");
        assert_eq!(e["name"], "main");
        assert_eq!(e["object"], "model");
        assert_eq!(e["created"], 1_754_000_000u64);
        assert_eq!(e["context_length"], 262144);
        assert_eq!(e["architecture"]["modality"], "text->text");
        assert_eq!(e["architecture"]["tokenizer"], "qwen2");
        assert_eq!(e["architecture"]["instruct_type"], "chatml");
        // pricing stub: OR-convention USD strings, self-hosted zeros.
        assert_eq!(e["pricing"]["prompt"], "0");
        assert_eq!(e["pricing"]["completion"], "0");
        assert_eq!(e["top_provider"]["context_length"], 262144);
        // no static completion cap (context-bounded, gap-scan F2) -> honest null.
        assert!(e["top_provider"]["max_completion_tokens"].is_null());

        // UNKNOWN metadata (no caps / empty fields) -> honest nulls, never invented.
        let e = model_entry_v1("m", None, 7);
        assert!(e["context_length"].is_null());
        assert!(e["architecture"]["tokenizer"].is_null());
        assert!(e["architecture"]["instruct_type"].is_null());
        assert!(e["top_provider"]["context_length"].is_null());
        let bare = ModelCaps::default(); // caps present, fields unknown (0/""/None)
        let e = model_entry_v1("m", Some(&bare), 7);
        assert!(e["context_length"].is_null());
        assert!(e["architecture"]["tokenizer"].is_null());
        assert!(e["architecture"]["instruct_type"].is_null());
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
            rx, "plain_quant".into(), false, vec!["Problem:".into()], None, Envelope::new(false)
        ).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["text"], "answer\n");
        assert_eq!(payload["stop_reason"], "Callback");
    }
}
