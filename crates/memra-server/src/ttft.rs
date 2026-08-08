use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

const UNSET: u64 = u64::MAX;

#[derive(Default)]
struct Meta {
    request_id: String,
    model: String,
    path: String,
    prompt_tokens: usize,
}

/// Per-request TTFT phase trace. Allocated only when MEMRA_TTFT_TRACE=1.
///
/// Every timestamp is a microsecond offset from HTTP request arrival. Atomics let the
/// axum and GPU-worker threads stamp one shared timeline without locks; the metadata
/// mutex is touched only on the debug path and only a few times per request.
pub struct Trace {
    started: Instant,
    meta: Mutex<Meta>,
    logged: AtomicBool,
    parsed: AtomicU64,
    submitted: AtomicU64,
    tokenize_start: AtomicU64,
    tokenize_end: AtomicU64,
    prime_start: AtomicU64,
    prime_end: AtomicU64,
    first_decode: AtomicU64,
    first_sse_byte: AtomicU64,
}

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MEMRA_TTFT_TRACE").as_deref() == Ok("1"))
}

pub fn start(path: &str) -> Option<Arc<Trace>> {
    if !matches!(path, "/v1/completions" | "/v1/chat/completions") {
        return None;
    }
    enabled().then(|| {
        Arc::new(Trace {
            started: Instant::now(),
            meta: Mutex::new(Meta {
                path: path.to_string(),
                ..Meta::default()
            }),
            logged: AtomicBool::new(false),
            parsed: AtomicU64::new(UNSET),
            submitted: AtomicU64::new(UNSET),
            tokenize_start: AtomicU64::new(UNSET),
            tokenize_end: AtomicU64::new(UNSET),
            prime_start: AtomicU64::new(UNSET),
            prime_end: AtomicU64::new(UNSET),
            first_decode: AtomicU64::new(UNSET),
            first_sse_byte: AtomicU64::new(UNSET),
        })
    })
}

impl Trace {
    fn now_us(&self) -> u64 {
        self.started.elapsed().as_micros().min(u64::MAX as u128) as u64
    }

    fn mark(&self, slot: &AtomicU64) {
        let _ = slot.compare_exchange(UNSET, self.now_us(), Ordering::AcqRel, Ordering::Acquire);
    }

    pub fn bind_request(&self, request_id: &str, model: &str) {
        if let Ok(mut meta) = self.meta.lock() {
            meta.request_id = request_id.to_string();
            meta.model = model.to_string();
        }
    }

    pub fn mark_parsed(&self) {
        self.mark(&self.parsed);
    }

    pub fn mark_submitted(&self) {
        self.mark(&self.submitted);
    }

    pub fn mark_tokenize_start(&self) {
        self.mark(&self.tokenize_start);
    }

    pub fn mark_tokenize_end(&self, prompt_tokens: usize) {
        if let Ok(mut meta) = self.meta.lock() {
            meta.prompt_tokens = prompt_tokens;
        }
        self.mark(&self.tokenize_end);
    }

    pub fn mark_prime_start(&self) {
        self.mark(&self.prime_start);
    }

    pub fn mark_prime_end(&self) {
        self.mark(&self.prime_end);
    }

    pub fn mark_first_decode(&self) {
        // A fully cached prompt has no prime call. Represent that as a zero-duration
        // prime at the first decode boundary so the timeline remains ordered.
        if self.prime_start.load(Ordering::Acquire) == UNSET {
            self.mark_prime_start();
        }
        if self.prime_end.load(Ordering::Acquire) == UNSET {
            self.mark_prime_end();
        }
        self.mark(&self.first_decode);
    }

    pub fn mark_first_sse_byte(&self) {
        self.mark(&self.first_sse_byte);
        self.log_once("first_sse_byte");
    }

    fn value(slot: &AtomicU64) -> Option<u64> {
        match slot.load(Ordering::Acquire) {
            UNSET => None,
            value => Some(value),
        }
    }

    fn ms(value: Option<u64>) -> String {
        value
            .map(|us| format!("{:.3}", us as f64 / 1_000.0))
            .unwrap_or_else(|| "na".to_string())
    }

    fn delta_ms(start: Option<u64>, end: Option<u64>) -> String {
        match (start, end) {
            (Some(start), Some(end)) => Self::ms(Some(end.saturating_sub(start))),
            _ => "na".to_string(),
        }
    }

    fn log_once(&self, outcome: &str) {
        if self
            .logged
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        let parsed = Self::value(&self.parsed);
        let submitted = Self::value(&self.submitted);
        let tokenize_start = Self::value(&self.tokenize_start);
        let tokenize_end = Self::value(&self.tokenize_end);
        let prime_start = Self::value(&self.prime_start);
        let prime_end = Self::value(&self.prime_end);
        let first_decode = Self::value(&self.first_decode);
        let first_sse_byte = Self::value(&self.first_sse_byte);
        let meta = self.meta.lock().ok();
        let request_id = meta
            .as_ref()
            .map(|m| m.request_id.as_str())
            .filter(|v| !v.is_empty())
            .unwrap_or("unknown");
        let model = meta
            .as_ref()
            .map(|m| m.model.as_str())
            .filter(|v| !v.is_empty())
            .unwrap_or("unknown");
        let path = meta.as_ref().map(|m| m.path.as_str()).unwrap_or("unknown");
        let prompt_tokens = meta.as_ref().map(|m| m.prompt_tokens).unwrap_or(0);

        eprintln!(
            "[ttft] id={request_id} model={model:?} path={path:?} prompt_tokens={prompt_tokens} \
             outcome={outcome} request_parse_ms={} admission_ms={} queue_wait_ms={} \
             tokenize_ms={} prime_wait_ms={} prime_ms={} decode_wait_ms={} sse_handoff_ms={} \
             request_parse_end_ms={} admission_end_ms={} tokenize_start_ms={} tokenize_end_ms={} \
             prime_start_ms={} prime_end_ms={} first_decode_ms={} first_sse_byte_ms={} total_ms={}",
            Self::ms(parsed),
            Self::delta_ms(parsed, submitted),
            Self::delta_ms(submitted, tokenize_start),
            Self::delta_ms(tokenize_start, tokenize_end),
            Self::delta_ms(tokenize_end, prime_start),
            Self::delta_ms(prime_start, prime_end),
            Self::delta_ms(prime_end, first_decode),
            Self::delta_ms(first_decode, first_sse_byte),
            Self::ms(parsed),
            Self::ms(submitted),
            Self::ms(tokenize_start),
            Self::ms(tokenize_end),
            Self::ms(prime_start),
            Self::ms(prime_end),
            Self::ms(first_decode),
            Self::ms(first_sse_byte),
            Self::ms(
                first_sse_byte
                    .or(first_decode)
                    .or(prime_end)
                    .or(tokenize_end)
                    .or(parsed)
            ),
        );
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        self.log_once("ended_without_sse");
    }
}
