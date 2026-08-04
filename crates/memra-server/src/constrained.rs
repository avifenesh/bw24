//! Constrained decoding (OpenAI `response_format`): JSON-mode + JSON-schema grammars.
//!
//! llguidance (the vLLM/SGLang/llama.cpp guided-decoding engine) compiles the schema into a
//! token-level grammar; each decode step computes the set of vocab tokens the grammar can
//! consume and bans everything else (-inf on the host logits row) BEFORE the sampler runs.
//! The accepted token then advances the grammar state.
//!
//! ISOLATION CONTRACT (the serve-tools convention): a request WITHOUT `response_format`
//! builds no factory, no matcher, and takes zero new branches — every hook below is behind
//! `Option`s that stay `None`. Unconstrained serving is byte-identical to pre-lane behavior
//! (proved by the A/B gate in research/constrained-20260803/).
//!
//! FULL path (lane/constrained-full, 2026-08-03 — v1's host-only seams closed):
//!   - the packed mask (SimpleVob words) H2Ds per step into a stable per-session device
//!     buffer; `mask_logits_f32` bans on device BEFORE the device sampler — constrained
//!     rows ride the same device-sample/lean-logits tick as everyone else.
//!   - constrained greedy sessions graph-promote (in-graph mask node, stable pointer,
//!     contents re-uploaded per step) and spec-decode (verify-side grammar truncation +
//!     masked-argmax cut slot; SpecGrammar below adapts the engine's SpecConstraint hook).
//!   - fallback sampler configs (penalties/top-k/top-p/min-p) and MEMRA_CONSTRAIN_HOST=1
//!     (the rollback oracle) keep the v1 host masked-copy sample.
//! Receipts: research/constrained-full-20260803/ (battery + three-way perf + gates).

use std::sync::Arc;

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{SimpleVob, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};
use memra_tokenizer::Tokenizer;

/// What the HTTP layer parsed out of `response_format` — carried on the worker `Request`.
#[derive(Debug, Clone)]
pub enum GrammarSpec {
    /// `{"type":"json_object"}` — any JSON object (schema `{"type":"object"}`).
    JsonObject,
    /// `{"type":"json_schema","json_schema":{"schema":{...}}}` — the client's schema.
    JsonSchema(serde_json::Value),
}

/// Parse the OpenAI `response_format` value. `None`/`{"type":"text"}` = unconstrained.
/// Unknown types / malformed bodies are loud errors (the honesty-gate policy: clean 400s,
/// never silent downgrades).
pub fn parse_response_format(v: Option<&serde_json::Value>)
    -> Result<Option<GrammarSpec>, String>
{
    let Some(v) = v else { return Ok(None) };
    let ty = v.get("type").and_then(|t| t.as_str())
        .ok_or("response_format.type must be a string")?;
    match ty {
        "text" => Ok(None),
        "json_object" => Ok(Some(GrammarSpec::JsonObject)),
        "json_schema" => {
            let js = v.get("json_schema")
                .ok_or("response_format.json_schema is required for type json_schema")?;
            if !js.is_object() {
                return Err("response_format.json_schema must be an object".into());
            }
            // OpenAI nests the schema under json_schema.schema; some clients send the
            // schema directly under json_schema. Accept both (the vLLM convention).
            let schema = js.get("schema").unwrap_or(js).clone();
            Ok(Some(GrammarSpec::JsonSchema(schema)))
        }
        other => Err(format!("response_format type {other:?} is not supported \
                              (text | json_object | json_schema)")),
    }
}

/// The token-vocabulary bridge: memra's Tokenizer vocab rendered as a llguidance TokTrie.
/// Declared NON-canonical (`tokenize_is_canonical = false`) so llguidance never fast-forwards
/// tokens it tokenized itself — every token the model emits is validated through the mask,
/// which is exactly the per-step contract the worker enforces.
struct MemraTokEnv {
    trie: TokTrie,
}

impl TokenizerEnv for MemraTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }
    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        // mask-only integration (non-canonical): greedy trie walk is sufficient — this is
        // never used to force tokens into the stream.
        self.trie.greedy_tokenize(s)
    }
    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}

/// Per-model grammar factory: the TokTrie build (one pass over the vocab) + llguidance's
/// slicer preprocessing happen ONCE, lazily on the first constrained request against the
/// model, then every request compiles only its own schema.
pub struct ConstraintFactory {
    factory: ParserFactory,
}

impl ConstraintFactory {
    pub fn new(tok: &Tokenizer) -> Result<Self, String> {
        let n = tok.vocab_size();
        let mut words: Vec<Vec<u8>> = Vec::with_capacity(n);
        for id in 0..n as u32 {
            if tok.token_is_control(id) {
                // control/protocol tokens: llguidance special-token marker form — never
                // matchable as literal grammar bytes (a JSON string must not be able to
                // smuggle <|im_start|>).
                let mut w = vec![TokTrie::SPECIAL_TOKEN_MARKER];
                w.extend_from_slice(format!("[{id}]").as_bytes());
                words.push(w);
            } else {
                words.push(tok.decode_bytes_special(&[id], true));
            }
        }
        let info = TokRxInfo::new(n as u32, tok.eos_id());
        let trie = TokTrie::from(&info, &words);
        let env: TokEnv = Arc::new(MemraTokEnv { trie });
        let mut factory = ParserFactory::new_simple(&env)
            .map_err(|e| format!("constraint factory: {e}"))?;
        factory.quiet();
        Ok(Self { factory })
    }

    /// Compile one request's grammar. Compile errors (bad schema) surface via
    /// `SessionConstraint::error()` at admit — a clean client error, not a worker panic.
    pub fn matcher(&self, spec: &GrammarSpec) -> SessionConstraint {
        let schema = match spec {
            GrammarSpec::JsonObject => serde_json::json!({"type": "object"}),
            GrammarSpec::JsonSchema(s) => s.clone(),
        };
        let grammar = TopLevelGrammar::from_json_schema(schema);
        SessionConstraint::new(Matcher::new(self.factory.create_parser(grammar)))
    }
}

/// -inf every vocab token the grammar cannot consume. Logits rows longer than the tokenizer
/// vocab (padded lm_head) get their tail banned too — padding ids are never decodable.
pub fn apply_mask(mask: &SimpleVob, logits: &mut [f32]) {
    let n = logits.len();
    mask.iter_unset_entries(|i| {
        if i < n {
            logits[i] = f32::NEG_INFINITY;
        }
    });
    if mask.len() < n {
        for l in &mut logits[mask.len()..] {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Per-session grammar state + the mask-cost meter (the perf receipt: steps and total
/// mask-compute time are logged at finish).
pub struct SessionConstraint {
    m: Matcher,
    pub steps: u64,
    pub mask_ns: u128,
    /// draft-side masking receipt (lane/draft-mask): speculative clones + their wall, and the
    /// draft-position masks computed on the cloned state.
    pub spec_clones: u64,
    pub spec_ns: u128,
    pub draft_masks: u64,
    pub draft_mask_ns: u128,
}

impl SessionConstraint {
    pub fn new(m: Matcher) -> Self {
        Self { m, steps: 0, mask_ns: 0,
               spec_clones: 0, spec_ns: 0, draft_masks: 0, draft_mask_ns: 0 }
    }

    /// Grammar-compile / parser error (checked once at admit).
    pub fn error(&self) -> Option<String> {
        self.m.get_error()
    }

    /// Compute the current token mask (timed — the mask-cost receipt). When the grammar
    /// has finished, the mask collapses to EOS-only — the normal Eos stop fires. The
    /// packed form (`SimpleVob::as_slice`) is what the device path H2Ds verbatim.
    pub fn compute_mask(&mut self) -> Result<SimpleVob, String> {
        let t0 = std::time::Instant::now();
        let mask = self.m.compute_mask_or_eos().map_err(|e| e.to_string())?;
        self.steps += 1;
        self.mask_ns += t0.elapsed().as_nanos();
        Ok(mask)
    }

    /// Compute the current token mask and apply it to `logits` (the HOST path: fallback
    /// sampler configs + the MEMRA_CONSTRAIN_HOST=1 oracle).
    pub fn mask_logits(&mut self, logits: &mut [f32]) -> Result<(), String> {
        let mask = self.compute_mask()?;
        apply_mask(&mask, logits);
        Ok(())
    }

    /// Advance the grammar with the accepted token. Cannot legitimately fail (the token
    /// was sampled from this state's own mask) — an error here is a loud session stop.
    pub fn consume(&mut self, tok: u32) -> Result<(), String> {
        self.m.consume_token(tok).map_err(|e| e.to_string())
    }

    /// SPECULATIVE CLONE of the committed grammar state (draft-side masking): llguidance's
    /// Matcher is Clone, so a draft chain walks a throwaway copy and the real state stays
    /// pinned at the last EMITTED token. Cost is metered separately (`spec_ns`) — one clone
    /// per spec round, never on the plain path.
    pub fn clone_matcher(&mut self) -> Matcher {
        let t0 = std::time::Instant::now();
        let m = self.m.clone();
        self.spec_clones += 1;
        self.spec_ns += t0.elapsed().as_nanos();
        m
    }
}

/// SpecConstraint adapter (constrained x spec-decode, 2026-08-03): SessionConstraint behind
/// the engine's grammar hook, with a per-state CACHED mask — the verify walk probes
/// `is_allowed` once per accepted token and the mask only changes on `consume`, so each
/// grammar state computes its mask exactly once (the same 0.02-0.06 ms/step cost as plain
/// constrained decode). EOS is never consumed (the plain path's EOS-before-consume ordering):
/// a finished grammar collapses its mask to EOS-only, so post-EOS drafts truncate naturally.
///
/// DRAFT-SIDE MASKING (lane/draft-mask, 2026-08-04, default ON — MEMRA_DRAFT_MASK=0 reverts):
/// `draft_begin` clones the matcher into `spec` and each draft position's mask is computed on
/// that CLONE, advanced by the PROPOSED token. The real matcher is untouched until `consume`
/// (an emitted token), so verify-side truncation remains the correctness backstop and the
/// emitted stream is byte-identical with masking on or off — masking only changes which
/// tokens get proposed. The clone is dropped at the next `draft_begin`/`consume`.
pub struct SpecGrammar<'a> {
    c: &'a mut SessionConstraint,
    eos: u32,
    cur: Option<SimpleVob>,
    /// speculative (draft-chain) matcher: a clone of `c`'s state at chain start.
    spec: Option<Matcher>,
    on: bool,
}

/// MEMRA_DRAFT_MASK=0 turns draft-side grammar masking off (the rollback seam / A-B arm).
pub fn draft_mask_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_DRAFT_MASK").map(|v| v != "0").unwrap_or(true))
}

impl<'a> SpecGrammar<'a> {
    pub fn new(c: &'a mut SessionConstraint, eos: u32) -> Self {
        Self { c, eos, cur: None, spec: None, on: draft_mask_on() }
    }
    fn cur_mask(&mut self) -> Result<&SimpleVob, String> {
        if self.cur.is_none() {
            self.cur = Some(self.c.compute_mask()?);
        }
        Ok(self.cur.as_ref().unwrap())
    }
}

impl memra_engine::spec::SpecConstraint for SpecGrammar<'_> {
    fn mask_logits(&mut self, logits: &mut [f32]) -> Result<(), String> {
        let mask = self.cur_mask()?;
        apply_mask(mask, logits);
        Ok(())
    }
    fn mask_words(&mut self) -> Result<Vec<u32>, String> {
        Ok(self.cur_mask()?.as_slice().to_vec())
    }
    fn is_allowed(&mut self, tok: u32) -> Result<bool, String> {
        let mask = self.cur_mask()?;
        // ids past the mask (padded lm_head tail) are banned; EOS defers to the mask
        // (a finished grammar's mask is EOS-only, an unfinished one usually bans it).
        Ok((tok as usize) < mask.len() && mask.is_allowed(tok))
    }
    fn consume(&mut self, tok: u32) -> Result<(), String> {
        // the speculative chain is dead as soon as the real state moves.
        self.spec = None;
        if tok == self.eos {
            return Ok(()); // EOS ends the stream — never fed to the grammar (plain-path order)
        }
        self.c.consume(tok)?;
        self.cur = None;
        Ok(())
    }

    fn draft_begin(&mut self) -> Result<(), String> {
        if !self.on {
            self.spec = None;
            return Ok(());
        }
        self.spec = Some(self.c.clone_matcher());
        Ok(())
    }

    fn draft_mask_words(&mut self) -> Result<Option<Vec<u32>>, String> {
        if !self.on {
            return Ok(None);
        }
        // position 0 of the chain shares the committed state's mask — reuse the cached one
        // (`cur`) instead of recomputing on the clone; identical set, zero mask cost.
        let Some(spec) = self.spec.as_mut() else { return Ok(None) };
        let t0 = std::time::Instant::now();
        let mask = spec.compute_mask_or_eos().map_err(|e| e.to_string())?;
        self.c.draft_masks += 1;
        self.c.draft_mask_ns += t0.elapsed().as_nanos();
        Ok(Some(mask.as_slice().to_vec()))
    }

    fn draft_advance(&mut self, tok: u32) -> Result<bool, String> {
        if !self.on {
            return Ok(false);
        }
        let Some(spec) = self.spec.as_mut() else { return Ok(false) };
        if tok == self.eos {
            return Ok(false); // EOS proposed: the chain ends here (plain-path EOS order)
        }
        // A masked draft is legal by construction; a token from a slot the mask could not
        // reach (p-min break, trimmed-vocab miss) simply ends the speculative chain — the
        // proposal still rides verify, where truncation arbitrates.
        match spec.consume_token(tok) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llguidance::toktrie::ApproximateTokEnv;

    #[test]
    fn parse_response_format_forms() {
        // absent / text = unconstrained (the no-op contract).
        assert!(parse_response_format(None).unwrap().is_none());
        let text = serde_json::json!({"type": "text"});
        assert!(parse_response_format(Some(&text)).unwrap().is_none());
        // json_object
        let jo = serde_json::json!({"type": "json_object"});
        assert!(matches!(parse_response_format(Some(&jo)).unwrap(),
                         Some(GrammarSpec::JsonObject)));
        // OpenAI nested form
        let js = serde_json::json!({"type": "json_schema", "json_schema": {
            "name": "x", "schema": {"type": "object", "required": ["a"]}}});
        match parse_response_format(Some(&js)).unwrap() {
            Some(GrammarSpec::JsonSchema(s)) => assert_eq!(s["required"][0], "a"),
            other => panic!("wrong parse: {other:?}"),
        }
        // direct-schema form (vLLM convention)
        let js2 = serde_json::json!({"type": "json_schema",
                                     "json_schema": {"type": "object"}});
        match parse_response_format(Some(&js2)).unwrap() {
            Some(GrammarSpec::JsonSchema(s)) => assert_eq!(s["type"], "object"),
            other => panic!("wrong parse: {other:?}"),
        }
        // loud errors: unknown type, missing schema, malformed.
        let bad = serde_json::json!({"type": "yaml"});
        assert!(parse_response_format(Some(&bad)).is_err());
        let bad2 = serde_json::json!({"type": "json_schema"});
        assert!(parse_response_format(Some(&bad2)).is_err());
        let bad3 = serde_json::json!({"type": 3});
        assert!(parse_response_format(Some(&bad3)).is_err());
    }

    #[test]
    fn apply_mask_bans_unset_and_padding_tail() {
        let mut vob = SimpleVob::alloc(8);
        vob.allow_token(2);
        vob.allow_token(5);
        // logits longer than the mask: the padded tail must be banned too.
        let mut logits = vec![1.0f32; 10];
        apply_mask(&vob, &mut logits);
        for (i, &l) in logits.iter().enumerate() {
            if i == 2 || i == 5 {
                assert_eq!(l, 1.0, "allowed token {i} must be untouched");
            } else {
                assert_eq!(l, f32::NEG_INFINITY, "banned token {i} must be -inf");
            }
        }
    }

    /// schema -> mask -> forced token sequence: greedy-walk the grammar (always take the
    /// lowest allowed token) and assert the emitted bytes parse as JSON AND satisfy the
    /// schema's required key. Uses llguidance's byte-level test env — the machinery under
    /// test is grammar/mask/consume, identical to the serve path.
    #[test]
    fn schema_mask_forced_sequence() {
        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": false
        });
        let mut m = Matcher::new(
            factory.create_parser(TopLevelGrammar::from_json_schema(schema)));
        assert!(m.get_error().is_none(), "{:?}", m.get_error());
        let eos = env.tok_trie().eos_token();
        let mut out: Vec<u8> = Vec::new();
        for _ in 0..256 {
            let mask = m.compute_mask_or_eos().unwrap();
            // the serve-path invariant: something is always allowed (worst case EOS).
            assert!(mask.num_set() > 0, "empty mask");
            // lowest allowed NON-whitespace token (JSON grammars allow unbounded
            // whitespace — a pure lowest-token walk would emit tabs forever).
            let mut pick: Option<u32> = None;
            mask.iter_set_entries(|i| {
                let ws = matches!(i as u8, b'\t' | b'\n' | b'\r' | b' ') && i < 128;
                if !ws && pick.is_none() {
                    pick = Some(i as u32);
                }
            });
            let t = pick.expect("only whitespace allowed — walker stuck");
            if t == eos {
                break;
            }
            m.consume_token(t).unwrap();
            out.extend_from_slice(env.tok_trie().token(t));
        }
        let text = String::from_utf8(out).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("forced output is not JSON: {e}: {text:?}"));
        assert!(v.is_object(), "not an object: {text:?}");
        // the walk picks '-' before digits, producing -0 — a valid JSON-schema integer
        // (serde parses it as f64; schema-wise -0 == 0). Number-with-zero-fraction is
        // exactly the draft-2020 "integer" definition.
        let a = v.get("a").unwrap_or_else(|| panic!("required key missing: {text:?}"));
        assert!(a.as_f64().is_some_and(|f| f.fract() == 0.0),
                "required integer key not an integer: {text:?}");
    }

    /// DRAFT-SIDE MASKING (lane/draft-mask): the speculative clone must (a) hand out the same
    /// legal set as the committed state at chain position 0, (b) advance INDEPENDENTLY of the
    /// real matcher across the chain, (c) mask out a token the grammar cannot take at that
    /// position, and (d) leave the real state exactly where it was (the byte-identity
    /// precondition — only `consume` may move it).
    #[test]
    fn speculative_clone_masks_illegal_draft_and_leaves_real_state() {
        use memra_engine::spec::SpecConstraint;
        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": false
        });
        let mut sc = SessionConstraint::new(Matcher::new(
            factory.create_parser(TopLevelGrammar::from_json_schema(schema))));
        assert!(sc.error().is_none());
        let eos = env.tok_trie().eos_token();
        let mut g = SpecGrammar::new(&mut sc, eos);
        assert!(g.on, "draft masking must default ON");

        // chain start: clone. Position 0 of this grammar can only take '{' (or whitespace).
        g.draft_begin().unwrap();
        let w0 = g.draft_mask_words().unwrap().expect("draft mask must be present when ON");
        let allowed = |words: &[u32], t: u32| -> bool {
            let w = (t >> 5) as usize;
            w < words.len() && (words[w] >> (t & 31)) & 1 == 1
        };
        assert!(allowed(&w0, b'{' as u32), "'{{' must be legal at draft pos 0");
        assert!(!allowed(&w0, b'x' as u32), "'x' must be MASKED at draft pos 0");
        assert!(!allowed(&w0, b'a' as u32), "bare 'a' (unquoted key) must be masked at pos 0");

        // propose the legal token: the clone advances, the REAL state must not.
        assert!(g.draft_advance(b'{' as u32).unwrap(), "legal draft must extend the chain");
        let w1 = g.draft_mask_words().unwrap().unwrap();
        assert!(allowed(&w1, b'"' as u32), "after '{{' a quoted key must be legal");
        assert!(!allowed(&w1, b'{' as u32), "a second '{{' must be masked at draft pos 1");
        // (d) the real (committed) state is still at position 0 — its own mask is unchanged.
        let real: Vec<u32> = SpecConstraint::mask_words(&mut g).unwrap();
        assert_eq!(real, w0, "real matcher moved during a draft chain (byte-identity break)");

        // an illegal proposal ends the speculative chain instead of erroring out.
        assert!(!g.draft_advance(b'{' as u32).unwrap(),
                "illegal draft token must end the chain, not error");
        // and the real state STILL has not moved.
        let real2: Vec<u32> = SpecConstraint::mask_words(&mut g).unwrap();
        assert_eq!(real2, w0, "real matcher moved after a dead speculative chain");

        // emitted token -> real state advances; a new chain clones from there.
        SpecConstraint::consume(&mut g, b'{' as u32).unwrap();
        g.draft_begin().unwrap();
        let w2 = g.draft_mask_words().unwrap().unwrap();
        assert_eq!(w2, w1, "a fresh chain after emitting '{{' must match the pos-1 mask");
        assert!(sc.spec_clones >= 2, "clone meter must count each chain start");
        assert!(sc.draft_masks >= 3, "draft-mask meter must count each masked position");
    }

    /// A token sampled OUTSIDE the mask must be rejected by consume — the guard the
    /// worker relies on for its loud-stop path.
    #[test]
    fn consume_outside_mask_is_error() {
        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let mut m = Matcher::new(factory.create_parser(
            TopLevelGrammar::from_json_schema(serde_json::json!({"type": "object"}))));
        // 'x' (0x78) cannot start a JSON object.
        assert!(m.consume_token(b'x' as u32).is_err());
    }
}
