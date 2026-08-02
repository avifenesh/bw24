//! Streaming parser for template-law tool-call emissions (serve-tools lane, 2026-08-02).
//!
//! The qwen3.5/3.6-class templates instruct the model to emit
//!
//! ```text
//! optional prose...
//! <tool_call>
//! <function=get_weather>
//! <parameter=city>
//! Paris
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! This module turns that text stream into OpenAI-shape `tool_calls` while passing everything
//! else through as content. It is PARSING ONLY — it sits between the worker's token stream and
//! the HTTP response and never touches generation. It is constructed ONLY for requests that
//! rendered a `<tools>` block (non-tools traffic bypasses it entirely: byte-identical streams,
//! including chunk boundaries — the isolation contract).
//!
//! MALFORMED-EMISSION POLICY (gate c): a `<tool_call>...</tool_call>` block that does not parse
//! (missing/garbled `<function=`, unpaired `<parameter=`) is surfaced VERBATIM as content —
//! tags included — and the stream continues; an unterminated `<tool_call>` at end-of-generation
//! flushes raw. Never an error, never dropped bytes: content + parsed calls always reassemble
//! to the exact generated text.
//!
//! THINK GATE: when the rendered prompt ended with an open `<think>\n` tail (the template
//! default), everything up to and including `</think>` passes through as content unscanned —
//! a `<tool_call>` mentioned while reasoning is not a call.

use std::collections::HashMap;

/// One parsed call, OpenAI-shape: `arguments` is a compact JSON object STRING.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Content(String),
    Call(ParsedToolCall),
}

enum State {
    /// Prompt ended with an open `<think>` — pass text through until `</think>`.
    Prethink,
    /// Scanning content for `<tool_call>`.
    Scan,
    /// Inside a `<tool_call>` block, buffering until `</tool_call>`.
    InCall,
}

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const THINK_END: &str = "</think>";

pub struct ToolStreamParser {
    state: State,
    /// Held-back text: in Prethink/Scan at most a partial tag suffix; in InCall the block body.
    buf: String,
    /// Declared JSON-schema `type` per (function, parameter) — drives argument coercion.
    schemas: HashMap<String, HashMap<String, String>>,
    n_calls: usize,
}

/// Length of the longest PROPER prefix of `tag` that `s` ends with (all tags are ASCII, so
/// the returned byte offset is always a char boundary).
fn partial_suffix_len(s: &str, tag: &str) -> usize {
    let max = (tag.len() - 1).min(s.len());
    for k in (1..=max).rev() {
        if s.ends_with(&tag[..k]) {
            return k;
        }
    }
    0
}

impl ToolStreamParser {
    /// `schemas`: function name -> parameter -> declared JSON-schema type string.
    /// `skip_think`: true when the rendered prompt ends with an open `<think>\n` tail.
    pub fn new(schemas: HashMap<String, HashMap<String, String>>, skip_think: bool) -> Self {
        Self {
            state: if skip_think { State::Prethink } else { State::Scan },
            buf: String::new(),
            schemas,
            n_calls: 0,
        }
    }

    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.state {
                State::Prethink => {
                    if let Some(i) = self.buf.find(THINK_END) {
                        let cut = i + THINK_END.len();
                        emit_content(&mut out, self.buf[..cut].to_string());
                        self.buf.drain(..cut);
                        self.state = State::Scan;
                        continue;
                    }
                    let keep = partial_suffix_len(&self.buf, THINK_END);
                    let emit_to = self.buf.len() - keep;
                    if emit_to > 0 {
                        emit_content(&mut out, self.buf[..emit_to].to_string());
                        self.buf.drain(..emit_to);
                    }
                    break;
                }
                State::Scan => {
                    if let Some(i) = self.buf.find(OPEN) {
                        if i > 0 {
                            emit_content(&mut out, self.buf[..i].to_string());
                        }
                        self.buf.drain(..i + OPEN.len());
                        self.state = State::InCall;
                        continue;
                    }
                    let keep = partial_suffix_len(&self.buf, OPEN);
                    let emit_to = self.buf.len() - keep;
                    if emit_to > 0 {
                        emit_content(&mut out, self.buf[..emit_to].to_string());
                        self.buf.drain(..emit_to);
                    }
                    break;
                }
                State::InCall => {
                    let Some(i) = self.buf.find(CLOSE) else { break };
                    let inner: String = self.buf[..i].to_string();
                    self.buf.drain(..i + CLOSE.len());
                    self.state = State::Scan;
                    match self.parse_block(&inner) {
                        Some(call) => out.push(Piece::Call(call)),
                        // malformed: surfaced verbatim, tags included, stream continues.
                        None => emit_content(&mut out, format!("{OPEN}{inner}{CLOSE}")),
                    }
                    continue;
                }
            }
        }
        out
    }

    /// End of generation: flush any held-back text. An unterminated `<tool_call>` block is
    /// surfaced raw (opening tag restored) — same malformed policy.
    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        if !self.buf.is_empty() {
            let tail = std::mem::take(&mut self.buf);
            let text = match self.state {
                State::InCall => format!("{OPEN}{tail}"),
                _ => tail,
            };
            emit_content(&mut out, text);
        }
        self.state = State::Scan;
        out
    }

    pub fn n_calls(&self) -> usize {
        self.n_calls
    }

    /// Parse one block body (the text between the `<tool_call>` tags). None = malformed.
    fn parse_block(&mut self, inner: &str) -> Option<ParsedToolCall> {
        let s = inner.trim();
        let rest = s.strip_prefix("<function=")?;
        let gt = rest.find('>')?;
        let name = &rest[..gt];
        if name.is_empty() || name.contains(['<', '>', '\n']) {
            return None;
        }
        let mut body = rest[gt + 1..].strip_suffix("</function>")?;
        let mut args = serde_json::Map::new();
        loop {
            let t = body.trim_start();
            if t.is_empty() {
                break;
            }
            let r = t.strip_prefix("<parameter=")?;
            let gt = r.find('>')?;
            let key = &r[..gt];
            if key.is_empty() || key.contains(['<', '>', '\n']) {
                return None;
            }
            // rendered form is `<parameter=k>\n{value}\n</parameter>` — the delimiter
            // newlines belong to the syntax, inner newlines belong to the value.
            let after = &r[gt + 1..];
            let after = after.strip_prefix('\n').unwrap_or(after);
            let end = after.find("</parameter>")?;
            let raw = after[..end].strip_suffix('\n').unwrap_or(&after[..end]);
            args.insert(key.to_string(), self.coerce(name, key, raw));
            body = &after[end + "</parameter>".len()..];
        }
        let arguments = serde_json::to_string(&serde_json::Value::Object(args)).ok()?;
        // Deterministic id (greedy serve receipts stay hashable): FNV-1a over index+name+args.
        let id = format!("call_{:016x}", fnv1a64(&[
            &self.n_calls.to_le_bytes(), name.as_bytes(), arguments.as_bytes(),
        ]));
        self.n_calls += 1;
        Some(ParsedToolCall { id, name: name.to_string(), arguments })
    }

    /// Coercion law: a parameter whose declared schema type is non-"string" is parsed as
    /// JSON (integer/number/boolean/object/array); parse failure or a declared/unknown
    /// string type keeps the raw text.
    fn coerce(&self, func: &str, param: &str, raw: &str) -> serde_json::Value {
        let declared = self.schemas.get(func).and_then(|m| m.get(param)).map(String::as_str);
        match declared {
            Some("string") | None => serde_json::Value::String(raw.to_string()),
            Some(_) => serde_json::from_str::<serde_json::Value>(raw.trim())
                .unwrap_or_else(|_| serde_json::Value::String(raw.to_string())),
        }
    }
}

/// Coalesce adjacent content pieces (chunk boundaries are not part of any contract, but
/// fewer SSE events is strictly kinder to clients).
fn emit_content(out: &mut Vec<Piece>, text: String) {
    if text.is_empty() {
        return;
    }
    if let Some(Piece::Content(prev)) = out.last_mut() {
        prev.push_str(&text);
        return;
    }
    out.push(Piece::Content(text));
}

fn fnv1a64(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for part in parts {
        for &b in *part {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_schema() -> HashMap<String, HashMap<String, String>> {
        let mut params = HashMap::new();
        params.insert("city".to_string(), "string".to_string());
        params.insert("days".to_string(), "integer".to_string());
        params.insert("metric".to_string(), "boolean".to_string());
        let mut m = HashMap::new();
        m.insert("get_weather".to_string(), params);
        m
    }

    const EMISSION: &str = "I'll check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n\
Paris\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n\
</function>\n</tool_call>";

    fn reassemble(pieces: &[Piece]) -> (String, Vec<ParsedToolCall>) {
        let mut content = String::new();
        let mut calls = Vec::new();
        for p in pieces {
            match p {
                Piece::Content(t) => content.push_str(t),
                Piece::Call(c) => calls.push(c.clone()),
            }
        }
        (content, calls)
    }

    #[test]
    fn parses_call_with_schema_coercion() {
        let mut p = ToolStreamParser::new(weather_schema(), false);
        let mut pieces = p.push(EMISSION);
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(content, "I'll check.\n\n");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris","days":3,"metric":true}"#);
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn char_by_char_deltas_produce_the_same_result() {
        let mut p = ToolStreamParser::new(weather_schema(), false);
        let mut pieces: Vec<Piece> = Vec::new();
        for ch in EMISSION.chars() {
            pieces.extend(p.push(&ch.to_string()));
        }
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(content, "I'll check.\n\n");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"city":"Paris","days":3,"metric":true}"#);
    }

    #[test]
    fn think_gate_passes_tool_call_mentions_through() {
        let mut p = ToolStreamParser::new(weather_schema(), true);
        let text = "planning a <tool_call> here...</think>\n\n<tool_call>\n\
<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";
        let mut pieces = p.push(text);
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(content, "planning a <tool_call> here...</think>\n\n");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"city":"Oslo"}"#);
    }

    #[test]
    fn malformed_block_is_surfaced_verbatim() {
        // broken JSON-ish emission: no <function= wrapper at all.
        let text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {broken\n</tool_call>done";
        let mut p = ToolStreamParser::new(weather_schema(), false);
        let mut pieces = p.push(text);
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(content, text); // byte-exact surfacing, tags included
        assert!(calls.is_empty());
    }

    #[test]
    fn unterminated_block_flushes_raw_on_finish() {
        let mut p = ToolStreamParser::new(weather_schema(), false);
        let mut pieces = p.push("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis");
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(content, "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis");
        assert!(calls.is_empty());
    }

    #[test]
    fn two_calls_and_multiline_string_values() {
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nline one\nline two\n\
</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=days>\n\
not-a-number\n</parameter>\n</function>\n</tool_call>";
        let mut p = ToolStreamParser::new(weather_schema(), false);
        let mut pieces = p.push(text);
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(content, "\n"); // the separator newline between the two blocks
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, r#"{"city":"line one\nline two"}"#);
        // integer-declared param that fails JSON parse falls back to the raw string.
        assert_eq!(calls[1].arguments, r#"{"days":"not-a-number"}"#);
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn partial_tag_holdback_never_loses_bytes() {
        // a "<tool" that never becomes a tag must still be emitted.
        let mut p = ToolStreamParser::new(HashMap::new(), false);
        let mut pieces = p.push("a <tool");
        pieces.extend(p.push("box holds bytes"));
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(content, "a <toolbox holds bytes");
        assert!(calls.is_empty());
    }
}
