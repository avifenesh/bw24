//! bw24-server (BASE-4): a minimal OpenAI-ish HTTP server that serves 2-4 concurrent agents across
//! DIFFERENT models on one endpoint via a single GPU worker thread + step-interleave scheduler.
//!
//! Architecture (see worker.rs): axum runs on a tokio runtime; ONE dedicated std::thread owns the
//! Engine + every loaded HybridModel (CUDA context is thread-affine). Handlers submit `Cmd`s over a
//! std mpsc channel and receive tokens back over a per-request tokio mpsc channel.
//!
//! Endpoints:
//!   GET  /health                 -> {"status":"ok","models":[...]}
//!   GET  /models                 -> {"data":[{"id":name},...]}  (OpenAI-ish)
//!   POST /v1/completions         -> {model,prompt|prompt_ids,max_tokens,temperature?,top_p?,top_k?,
//!                                     seed?,stop?,chat?,stream?}. stream=true => SSE token-by-token;
//!                                     else a single JSON {text,tokens,stop_reason}.
//!   POST /v1/chat/completions    -> OpenAI chat messages rendered by the GGUF chat template;
//!                                     OpenAI message/chunk response shapes.
//!
//! CONFIG: BW24_MODELS="name=/path.gguf,name2=/path2.gguf" (comma-separated name=path pairs).
//! Defaults to the BASE-4 test pair (main=27B, judge=9B) if unset. BW24_ADDR sets the bind addr.

mod worker;

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

use bw24_engine::decode::GenParams;
use bw24_engine::sampler::SamplerConfig;
use worker::{Cmd, Event, Request};

#[derive(Clone)]
struct AppState {
    cmd_tx: Sender<Cmd>,
    models: Arc<Vec<String>>,
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
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
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

/// OpenAI-compatible multi-turn chat request. Tool calls are intentionally not accepted: the
/// Terminus agent emits its terminal actions as JSON/XML text, and the model's GGUF chat template
/// remains the sole source of prompt formatting.
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
    elapsed_s: f64,
}

/// OpenAI-compat mapping (2026-07-05, serve-parity arc): the pi daily client speaks
/// `openai-completions` — POST /v1/completions with the OpenAI body, expecting
/// `{choices:[{text, finish_reason, index}], usage:{...}}` and, when streaming, OpenAI SSE
/// chunks (`data: {choices:[{text}]}` ... `data: [DONE]`). pi renders the chat template
/// CLIENT-side (thinkingFormat qwen-chat-template), so raw-prompt completions is the whole
/// contract. BW24_COMPAT=openai (default when BW24_API_KEY is set — the pi setup) switches the
/// response shape; the native bw24 shape stays default otherwise (validation harnesses use it).
fn openai_compat() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        match std::env::var("BW24_COMPAT").as_deref() {
            Ok("openai") => true,
            Ok(_) => false,
            Err(_) => std::env::var("BW24_API_KEY").is_ok(),
        }
    })
}

fn stop_reason_to_finish(r: &str) -> &'static str {
    match r {
        "Eos" | "Callback" => "stop",
        "MaxNew" | "ContextFull" => "length",
        _ => "stop",
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = parse_models_config();
    eprintln!("[server] starting; models config = {models:?}");

    // Spawn the GPU worker thread and block until every model is loaded (or it fails).
    let (cmd_tx, model_names) = match worker::spawn(models) {
        Ok(v) => v,
        Err(err) => { eprintln!("[server] FATAL: worker init failed: {err}"); std::process::exit(1); }
    };
    eprintln!("[server] worker ready; serving models: {model_names:?}");

    let state = AppState { cmd_tx, models: model_names };
    let app = Router::new()
        .route("/health", get(health))
        .route("/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = std::env::var("BW24_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("[server] listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

/// BW24_MODELS="name=/path.gguf,name2=/path2.gguf". Falls back to the BASE-4 test pair.
fn parse_models_config() -> Vec<(String, String)> {
    if let Ok(spec) = std::env::var("BW24_MODELS") {
        let mut out = Vec::new();
        for entry in spec.split(',').filter(|s| !s.trim().is_empty()) {
            if let Some((name, path)) = entry.split_once('=') {
                out.push((name.trim().to_string(), path.trim().to_string()));
            } else {
                eprintln!("[server] WARN: bad BW24_MODELS entry {entry:?} (want name=/path); skipping");
            }
        }
        if !out.is_empty() { return out; }
    }
    // Default: the BASE-4 test pair (main=27B, judge=9B).
    vec![
        ("main".into(),  "/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf".into()),
        ("judge".into(), "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf".into()),
    ]
}

async fn health(State(st): State<AppState>) -> impl IntoResponse {
    Json(json!({ "status": "ok", "models": *st.models }))
}

async fn list_models(State(st): State<AppState>) -> impl IntoResponse {
    let data: Vec<_> = st.models.iter().map(|m| json!({ "id": m, "object": "model" })).collect();
    Json(json!({ "object": "list", "data": data }))
}

/// Build the (GenParams, SamplerConfig, stop, prompt) from a request body.
fn build_request(req: &CompletionReq, tx: tokio::sync::mpsc::UnboundedSender<Event>) -> Request {
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
        chat_messages: Vec::new(),
        params,
        sampler_cfg,
        stop_strings: req.stop.clone().into_vec(),
        tx,
    }
}

fn build_chat_request(req: ChatCompletionReq,
                      tx: tokio::sync::mpsc::UnboundedSender<Event>) -> Request {
    Request {
        model: req.model,
        prompt_ids: Vec::new(),
        prompt_text: String::new(),
        chat: false,
        chat_messages: req.messages.into_iter().map(|msg| (msg.role, msg.content)).collect(),
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
        tx,
    }
}

fn authorized(headers: &axum::http::HeaderMap) -> bool {
    let Ok(key) = std::env::var("BW24_API_KEY") else { return true };
    headers.get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|candidate| candidate == key)
}

async fn completions(State(st): State<AppState>, headers: axum::http::HeaderMap,
                     Json(req): Json<CompletionReq>) -> Response {
    // API key (BW24_API_KEY): OpenAI-style `Authorization: Bearer <key>`. Absent env = open.
    if !authorized(&headers) {
        return (StatusCode::UNAUTHORIZED, "invalid api key").into_response();
    }
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let model = req.model.clone();
    let stream = req.stream;
    let request = build_request(&req, tx);
    let stop_strings = request.stop_strings.clone();

    if st.cmd_tx.send(Cmd::Generate(Box::new(request))).is_err() {
        return (StatusCode::SERVICE_UNAVAILABLE, "worker unavailable").into_response();
    }

    if stream {
        sse_response(rx, model, false).into_response()
    } else {
        blocking_response(rx, model, false, stop_strings).await.into_response()
    }
}

async fn chat_completions(State(st): State<AppState>, headers: axum::http::HeaderMap,
                          Json(req): Json<ChatCompletionReq>) -> Response {
    if !authorized(&headers) {
        return (StatusCode::UNAUTHORIZED, "invalid api key").into_response();
    }
    if req.messages.is_empty() || req.messages.iter().any(|message| {
        !matches!(message.role.as_str(), "system" | "user" | "assistant")
    }) {
        return (StatusCode::BAD_REQUEST,
                Json(json!({ "error": "messages must use system/user/assistant roles" })))
            .into_response();
    }
    let model = req.model.clone();
    let stream = req.stream;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let request = build_chat_request(req, tx);
    let stop_strings = request.stop_strings.clone();
    if st.cmd_tx.send(Cmd::Generate(Box::new(request))).is_err() {
        return (StatusCode::SERVICE_UNAVAILABLE, "worker unavailable").into_response();
    }
    if stream {
        sse_response(rx, model, true).into_response()
    } else {
        blocking_response(rx, model, true, stop_strings).await.into_response()
    }
}

/// Streaming (SSE): forward each Token as an SSE `data:` line; emit a final `done` event.
fn sse_response(mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>, model: String, chat: bool)
    -> Sse<impl futures_core::Stream<Item = Result<SseEvent, std::convert::Infallible>>> {
    let stream = async_stream::stream! {
        while let Some(ev) = rx.recv().await {
            match ev {
                Event::Token { id, text } => {
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
                Event::Done { stop_reason, n_tokens, elapsed_s } => {
                    if chat || openai_compat() {
                        let fin = if chat {
                            json!({ "object": "chat.completion.chunk", "model": model,
                                "choices": [{ "index": 0, "delta": {},
                                              "finish_reason": stop_reason_to_finish(&stop_reason) }],
                                "usage": { "completion_tokens": n_tokens,
                                           "total_tokens": n_tokens,
                                           "elapsed_s": elapsed_s } })
                        } else {
                            json!({ "object": "text_completion", "model": model,
                                "choices": [{ "index": 0, "text": "",
                                              "finish_reason": stop_reason_to_finish(&stop_reason) }],
                            "usage": { "completion_tokens": n_tokens,
                                       "total_tokens": n_tokens,
                                       "elapsed_s": elapsed_s } })
                        }.to_string();
                        yield Ok(SseEvent::default().data(fin));
                        yield Ok(SseEvent::default().data("[DONE]".to_string()));
                    } else {
                        let payload = json!({
                            "stop_reason": stop_reason, "n_tokens": n_tokens, "elapsed_s": elapsed_s
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
                           chat: bool, stop_strings: Vec<String>) -> Response {
    let mut text = String::new();
    let mut tokens: Vec<u32> = Vec::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Token { id, text: delta } => { tokens.push(id); text.push_str(&delta); }
            Event::Done { stop_reason, n_tokens, elapsed_s } => {
                truncate_at_stop(&mut text, &stop_strings);
                if chat {
                    return Json(json!({
                        "object": "chat.completion", "model": model,
                        "choices": [{ "index": 0,
                                      "message": { "role": "assistant", "content": text },
                                      "finish_reason": stop_reason_to_finish(&stop_reason) }],
                        "usage": { "completion_tokens": n_tokens, "total_tokens": n_tokens,
                                   "elapsed_s": elapsed_s }
                    })).into_response();
                }
                if openai_compat() {
                    return Json(json!({
                        "object": "text_completion", "model": model,
                        "choices": [{ "index": 0, "text": text,
                                      "finish_reason": stop_reason_to_finish(&stop_reason) }],
                        "usage": { "completion_tokens": n_tokens, "total_tokens": n_tokens,
                                   "elapsed_s": elapsed_s }
                    })).into_response();
                }
                return Json(CompletionResp {
                    model, text, tokens, stop_reason, n_tokens, elapsed_s,
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
        let request = build_chat_request(req, tx);
        assert_eq!(request.model, "plain_quant");
        assert_eq!(request.params.max_new, 64);
        assert_eq!(request.chat_messages, vec![
            ("system".into(), "rules".into()),
            ("user".into(), "task".into()),
            ("assistant".into(), "work".into()),
        ]);
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
            stop_reason: "Eos".into(), n_tokens: 1, elapsed_s: 0.5,
        }).unwrap();
        drop(tx);
        let response = blocking_response(rx, "plain_quant".into(), true, Vec::new()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["object"], "chat.completion");
        assert_eq!(payload["choices"][0]["message"]["role"], "assistant");
        assert_eq!(payload["choices"][0]["message"]["content"], "hello");
        assert_eq!(payload["choices"][0]["finish_reason"], "stop");
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
            stop_reason: "Callback".into(), n_tokens: 2, elapsed_s: 0.5,
        }).unwrap();
        drop(tx);
        let response = blocking_response(
            rx, "plain_quant".into(), false, vec!["Problem:".into()]
        ).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["text"], "answer\n");
        assert_eq!(payload["stop_reason"], "Callback");
    }
}
