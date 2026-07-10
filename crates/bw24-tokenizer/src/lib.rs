//! bw24-tokenizer — host-only GPT-2/BPE tokenizer (encode + decode + chat template).
//!
//! Algorithm TAKEn ~1:1 from llama.cpp's GPT-2 BPE path (`src/llama-vocab.cpp`,
//! `src/unicode.cpp`), Rust glue hand-rolled. Built from the model's own GGUF
//! tokenizer metadata (`tokenizer.ggml.*`) so it is integer-exact for that model.
//!
//! Scope: the `gpt2` vocab model with the `qwen35` pre-tokenizer (Qwen3.5). Other
//! pre-tokenizers are not ported (we only need this model's).

mod chat;
mod json;
mod unicode;
mod unicode_data;

pub use chat::apply_chat_template_str;

use bw24_gguf::{GgufFile, MetaValue};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// ggml token_type values (llama.cpp `LLAMA_TOKEN_TYPE_*`).
const TT_UNKNOWN: i64 = 2;
const TT_CONTROL: i64 = 3;
const TT_USER_DEFINED: i64 = 4;
const TT_BYTE: i64 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokAttr {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte,
    Other,
}

impl TokAttr {
    fn from_toktype(t: i64) -> Self {
        match t {
            TT_UNKNOWN => TokAttr::Unknown,
            TT_CONTROL => TokAttr::Control,
            TT_USER_DEFINED => TokAttr::UserDefined,
            TT_BYTE => TokAttr::Byte,
            1 => TokAttr::Normal,
            _ => TokAttr::Other,
        }
    }
    /// Tokens that participate in `tokenizer_st_partition` (special-token splitting):
    /// CONTROL | USER_DEFINED | UNKNOWN.
    fn is_special(self) -> bool {
        matches!(self, TokAttr::Control | TokAttr::UserDefined | TokAttr::Unknown)
    }
}

pub struct Tokenizer {
    /// id -> raw vocab piece string (byte-encoded GPT-2 form, e.g. "Ġworld").
    id_to_token: Vec<String>,
    /// piece string -> id.
    token_to_id: HashMap<String, u32>,
    /// per-token attribute.
    attrs: Vec<TokAttr>,
    /// (left, right) merge pair -> rank (lower = higher priority).
    bpe_ranks: HashMap<(String, String), i32>,
    /// special-token ids, sorted by descending piece length (llama's cache order).
    special_tokens: Vec<u32>,
    eos_id: u32,
    bos_id: Option<u32>,
    add_bos: bool,
    pre: String,
    chat_template: Option<String>,
    /// SPM-style BPE (gemma4): \u2581 whitespace escaping, raw-UTF-8 merges, <0xXX> byte fallback.
    spm_style: bool,
}

/// A bigram in the BPE work queue. Ordering matches llama.cpp's comparator:
/// the priority_queue pops the *smallest* (rank, left) under the std comparator
/// `l.rank > r.rank || (l.rank == r.rank && l.left > r.left)`. We implement `Ord`
/// so a max-heap pops that same element (min rank, then min left).
#[derive(Clone, Eq, PartialEq)]
struct Bigram {
    left: i32,
    right: i32,
    rank: i32,
    text: String,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; we want the element with the lowest rank
        // (ties: lowest left index) to be "greatest" so it pops first.
        match other.rank.cmp(&self.rank) {
            Ordering::Equal => other.left.cmp(&self.left),
            o => o,
        }
    }
}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A symbol (one or more codepoints) in the BPE chain. Mirrors `llm_symbol`.
struct Symbol {
    text: String,
    prev: i32,
    next: i32,
    n: usize, // codepoint count (0 == merged away)
}

impl Tokenizer {
    /// Build a tokenizer from a model's GGUF tokenizer metadata.
    pub fn from_gguf(g: &GgufFile) -> Result<Self, String> {
        let model = g
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .ok_or("missing tokenizer.ggml.model")?;
        if model != "gpt2" && model != "gemma4" {
            return Err(format!("unsupported tokenizer model '{model}' (only gpt2/gemma4)"));
        }
        // gemma4 = SPM-style BPE (llama-vocab.cpp): spaces escaped to \u2581 by the normalizer,
        // merges over raw UTF-8 (NO gpt2 byte-encoding), whole-line pre-split, <0xXX> byte
        // fallback tokens, add_bos force-true (PR #21500 workaround).
        let spm_style = model == "gemma4";
        let pre = g
            .metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or(if spm_style { "gemma4" } else { "default" })
            .to_string();

        // tokens[]
        let tokens = match g.metadata.get("tokenizer.ggml.tokens") {
            Some(MetaValue::Array(a)) => a,
            _ => return Err("missing tokenizer.ggml.tokens array".into()),
        };
        let n = tokens.len();
        let mut id_to_token = Vec::with_capacity(n);
        let mut token_to_id = HashMap::with_capacity(n);
        for (i, t) in tokens.iter().enumerate() {
            let s = t.as_str().ok_or("non-string in tokens[]")?.to_string();
            // first-id-wins on duplicates (llama keeps the map's first insert)
            token_to_id.entry(s.clone()).or_insert(i as u32);
            id_to_token.push(s);
        }

        // token_type[] -> attrs
        let mut attrs = vec![TokAttr::Normal; n];
        if let Some(MetaValue::Array(a)) = g.metadata.get("tokenizer.ggml.token_type") {
            for (i, v) in a.iter().enumerate().take(n) {
                if let Some(t) = v.as_u64() {
                    attrs[i] = TokAttr::from_toktype(t as i64);
                } else if let MetaValue::I32(t) = v {
                    attrs[i] = TokAttr::from_toktype(*t as i64);
                }
            }
        }

        // merges[] -> ranks. Each entry is "first second" (split on first space at idx>=1).
        let mut bpe_ranks = HashMap::new();
        if let Some(MetaValue::Array(a)) = g.metadata.get("tokenizer.ggml.merges") {
            for (i, v) in a.iter().enumerate() {
                let word = v.as_str().ok_or("non-string in merges[]")?;
                // llama: pos = word.find(' ', 1) — a *byte* search starting at byte 1.
                // (The space separating the two pieces is always single-byte ASCII; the
                // pieces themselves may contain multibyte chars like 'Ġ', so we search bytes.)
                let bytes = word.as_bytes();
                if let Some(pos) = bytes.iter().skip(1).position(|&b| b == b' ').map(|p| p + 1) {
                    let first = word[..pos].to_string();
                    let second = word[pos + 1..].to_string();
                    bpe_ranks.insert((first, second), i as i32);
                }
            }
        } else {
            return Err("missing tokenizer.ggml.merges array".into());
        }

        // special-token cache: CONTROL|USER_DEFINED|UNKNOWN, sorted by descending text length.
        let mut special_tokens: Vec<u32> = (0..n as u32)
            .filter(|&id| attrs[id as usize].is_special())
            .collect();
        special_tokens.sort_by(|&a, &b| {
            id_to_token[b as usize]
                .len()
                .cmp(&id_to_token[a as usize].len())
        });

        let eos_id = g
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or("missing tokenizer.ggml.eos_token_id")?;
        let bos_id = g
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let add_bos = g
            .metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| match v {
                MetaValue::Bool(b) => Some(*b),
                _ => v.as_u64().map(|x| x != 0),
            })
            .unwrap_or(false);
        let add_bos = add_bos || spm_style;

        let chat_template = g
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(Tokenizer {
            id_to_token,
            token_to_id,
            attrs,
            bpe_ranks,
            special_tokens,
            eos_id,
            bos_id,
            add_bos,
            pre,
            chat_template,
            spm_style,
        })
    }

    /// Build a tokenizer from an HF fast-tokenizer checkpoint directory
    /// (`tokenizer.json` + optional `tokenizer_config.json` / `generation_config.json` /
    /// `chat_template.jinja`). Only byte-level BPE (the gpt2 class — MiniMax-M3, Qwen,
    /// Llama-3 style) is supported: `model.type == "BPE"` with a ByteLevel pre-tokenizer.
    ///
    /// Mapping to the GGUF-built struct:
    ///   - model.vocab (token -> id map)             -> id_to_token / token_to_id
    ///   - model.merges ("a b" strings OR [a,b] pairs; both HF serializations) -> bpe_ranks
    ///   - added_tokens special=true -> Control class (split before BPE + hidden on decode);
    ///     non-special added tokens stay Normal.
    ///   - eos/bos: tokenizer_config eos_token/bos_token (string or {content} object),
    ///     generation_config eos_token_id (int or array) as the eos fallback.
    ///   - add_bos: tokenizer_config add_bos_token (default false).
    ///   - chat template: tokenizer_config chat_template, else chat_template.jinja.
    ///   - pre = "default": every `pre` class routes to the gpt2-style byte-level split
    ///     in bpe_tokenize (see the match there) — correct for the M3 tokenizer class.
    pub fn from_hf_dir(dir: &std::path::Path) -> Result<Self, String> {
        let tj_path = dir.join("tokenizer.json");
        let text = std::fs::read_to_string(&tj_path)
            .map_err(|e| format!("read {}: {e}", tj_path.display()))?;
        let tj = json::parse(&text).map_err(|e| format!("{}: {e}", tj_path.display()))?;

        let model = tj.get("model").ok_or("tokenizer.json: missing model")?;
        if let Some(t) = model.get("type").and_then(|v| v.as_str()) {
            if t != "BPE" {
                return Err(format!("unsupported tokenizer.json model type '{t}' (only BPE)"));
            }
        }
        // byte-level check: pre_tokenizer.type == ByteLevel (possibly inside a Sequence).
        let pre_tok = tj.get("pre_tokenizer").ok_or("tokenizer.json: missing pre_tokenizer")?;
        if !pre_tokenizer_is_byte_level(pre_tok) {
            return Err("tokenizer.json: pre_tokenizer is not ByteLevel — only byte-level \
                        BPE is supported"
                .into());
        }

        // ---- vocab (token -> id). ids may exceed the map len (added_tokens append). ----
        let vocab = model
            .get("vocab")
            .and_then(|v| v.as_obj())
            .ok_or("tokenizer.json: missing model.vocab")?;
        let empty: Vec<json::Value> = Vec::new();
        let added = tj
            .get("added_tokens")
            .and_then(|v| v.as_arr())
            .unwrap_or(&empty);
        let mut max_id = 0u32;
        for v in vocab.values() {
            let id = v.as_u64().ok_or("tokenizer.json: non-integer id in model.vocab")? as u32;
            max_id = max_id.max(id);
        }
        for a in added {
            if let Some(id) = a.get("id").and_then(|v| v.as_u64()) {
                max_id = max_id.max(id as u32);
            }
        }
        let n = max_id as usize + 1;
        let mut id_to_token = vec![String::new(); n];
        let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(n);
        let mut attrs = vec![TokAttr::Normal; n];
        for (tok, v) in vocab {
            let id = v.as_u64().unwrap() as u32;
            id_to_token[id as usize] = tok.clone();
            token_to_id.entry(tok.clone()).or_insert(id);
        }
        // added_tokens: register content + special flag. special=true -> Control (the class
        // that is split out before BPE and hidden by decode_special(.., false)).
        for a in added {
            let id = a
                .get("id")
                .and_then(|v| v.as_u64())
                .ok_or("tokenizer.json: added_tokens entry missing id")? as u32;
            let content = a
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or("tokenizer.json: added_tokens entry missing content")?;
            if id_to_token[id as usize].is_empty() {
                id_to_token[id as usize] = content.to_string();
            }
            token_to_id.entry(content.to_string()).or_insert(id);
            if a.get("special").and_then(|v| v.as_bool()).unwrap_or(false) {
                attrs[id as usize] = TokAttr::Control;
            } else {
                // HF's AddedVocabulary matches EVERY added token whole (special or not) before
                // the BPE model runs; `special` only controls skip_special_tokens on decode.
                // UserDefined = split whole before BPE but NOT hidden on decode — exactly the
                // HF non-special class (Hy3's `<think:opensource>`/`<｜reasoning_mode…｜>` chat
                // tokens are special=false and MUST encode as single ids, 2026-07-09).
                attrs[id as usize] = TokAttr::UserDefined;
            }
        }

        // ---- merges: array of "a b" strings OR [a, b] pairs (HF emits both). ----
        let merges = model
            .get("merges")
            .and_then(|v| v.as_arr())
            .ok_or("tokenizer.json: missing model.merges")?;
        let mut bpe_ranks = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            let (first, second) = match m {
                json::Value::Str(s) => {
                    // byte search for the separating space from byte 1 (same as the GGUF
                    // path: pieces may contain multibyte chars like 'Ġ', the space is ASCII).
                    let bytes = s.as_bytes();
                    let pos = bytes
                        .iter()
                        .skip(1)
                        .position(|&b| b == b' ')
                        .map(|p| p + 1)
                        .ok_or_else(|| format!("tokenizer.json: merges[{i}] has no space"))?;
                    (s[..pos].to_string(), s[pos + 1..].to_string())
                }
                json::Value::Arr(a) if a.len() == 2 => {
                    let f = a[0]
                        .as_str()
                        .ok_or_else(|| format!("tokenizer.json: merges[{i}] non-string pair"))?;
                    let s2 = a[1]
                        .as_str()
                        .ok_or_else(|| format!("tokenizer.json: merges[{i}] non-string pair"))?;
                    (f.to_string(), s2.to_string())
                }
                _ => {
                    return Err(format!(
                        "tokenizer.json: merges[{i}] is neither \"a b\" string nor [a, b] pair"
                    ));
                }
            };
            bpe_ranks.insert((first, second), i as i32);
        }

        // special-token cache: same construction as from_gguf.
        let mut special_tokens: Vec<u32> = (0..n as u32)
            .filter(|&id| attrs[id as usize].is_special())
            .collect();
        special_tokens.sort_by(|&a, &b| {
            id_to_token[b as usize]
                .len()
                .cmp(&id_to_token[a as usize].len())
        });

        // ---- sidecars: tokenizer_config.json + generation_config.json ----
        let tc = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|t| json::parse(&t).ok());
        let gc = std::fs::read_to_string(dir.join("generation_config.json"))
            .ok()
            .and_then(|t| json::parse(&t).ok());

        // eos_token/bos_token: plain string OR {"content": "..."} AddedToken object.
        let tok_content = |v: &json::Value| -> Option<String> {
            v.as_str()
                .map(|s| s.to_string())
                .or_else(|| v.get("content").and_then(|c| c.as_str()).map(|s| s.to_string()))
        };
        let eos_from_cfg = tc
            .as_ref()
            .and_then(|c| c.get("eos_token"))
            .and_then(&tok_content)
            .and_then(|s| token_to_id.get(&s).copied());
        // generation_config eos_token_id: int or array of ints (first entry wins).
        let eos_from_gen = gc
            .as_ref()
            .and_then(|c| c.get("eos_token_id"))
            .and_then(|v| match v {
                json::Value::Num(_) => v.as_u64(),
                json::Value::Arr(a) => a.first().and_then(|x| x.as_u64()),
                _ => None,
            })
            .map(|v| v as u32);
        let eos_id = eos_from_cfg.or(eos_from_gen).ok_or(
            "no eos token: need tokenizer_config.json eos_token or \
             generation_config.json eos_token_id",
        )?;
        let bos_id = tc
            .as_ref()
            .and_then(|c| c.get("bos_token"))
            .and_then(&tok_content)
            .and_then(|s| token_to_id.get(&s).copied());
        let add_bos = tc
            .as_ref()
            .and_then(|c| c.get("add_bos_token"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // chat template: tokenizer_config chat_template string, else chat_template.jinja file.
        let chat_template = tc
            .as_ref()
            .and_then(|c| c.get("chat_template"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| std::fs::read_to_string(dir.join("chat_template.jinja")).ok());

        Ok(Tokenizer {
            id_to_token,
            token_to_id,
            attrs,
            bpe_ranks,
            special_tokens,
            eos_id,
            bos_id,
            add_bos,
            pre: "default".to_string(),
            chat_template,
            spm_style: false,
        })
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
    /// End-of-generation ids: eos + the common turn-end control tokens present in the vocab
    /// (llama's special_eog set — <|im_end|> chatml, <turn|>/<end_of_turn> gemma).
    pub fn eog_ids(&self) -> Vec<u32> {
        let mut ids = vec![self.eos_id];
        for t in ["<|im_end|>", "<turn|>", "<end_of_turn>"] {
            if let Some(&id) = self.token_to_id.get(t) {
                if !ids.contains(&id) { ids.push(id); }
            }
        }
        ids
    }
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
    pub fn pre(&self) -> &str {
        &self.pre
    }
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    #[inline]
    fn text_to_token(&self, s: &str) -> Option<u32> {
        self.token_to_id.get(s).copied()
    }

    fn find_bpe_rank(&self, left: &str, right: &str) -> i32 {
        self.bpe_ranks
            .get(&(left.to_string(), right.to_string()))
            .copied()
            .unwrap_or(-1)
    }

    /// Encode text -> token ids.
    ///
    /// `add_special` controls whether a BOS is prepended when the model asks for it.
    /// `parse_special` (always true here) splits control/user-defined/unknown tokens
    /// (e.g. `<|im_start|>`) out before BPE — matching llama's default tokenize().
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        self.encode_special(text, add_special, true)
    }

    pub fn encode_special(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let mut output: Vec<u32> = Vec::new();
        if add_special && self.add_bos {
            if let Some(b) = self.bos_id {
                output.push(b);
            }
        }
        if text.is_empty() {
            return output;
        }

        // fragment buffer: alternate raw-text spans and resolved special-token ids.
        for frag in self.st_partition(text, parse_special) {
            match frag {
                Fragment::Token(id) => output.push(id),
                Fragment::Text(span) => self.bpe_tokenize(&span, &mut output),
            }
        }
        output
    }

    /// `tokenizer_st_partition` — split out special tokens (longest first) before BPE.
    fn st_partition(&self, text: &str, parse_special: bool) -> Vec<Fragment> {
        let mut frags = vec![Fragment::Text(text.to_string())];
        for &sid in &self.special_tokens {
            let attr = self.attrs[sid as usize];
            // when parse_special is false, skip CONTROL/UNKNOWN (user-defined still split).
            if !parse_special && matches!(attr, TokAttr::Control | TokAttr::Unknown) {
                continue;
            }
            let needle = &self.id_to_token[sid as usize];
            if needle.is_empty() {
                continue;
            }
            let mut next: Vec<Fragment> = Vec::with_capacity(frags.len());
            for f in frags.drain(..) {
                match f {
                    Fragment::Token(id) => next.push(Fragment::Token(id)),
                    Fragment::Text(s) => {
                        let mut rest: &str = &s;
                        let mut acc = String::new();
                        while let Some(m) = rest.find(needle.as_str()) {
                            acc.push_str(&rest[..m]);
                            if !acc.is_empty() {
                                next.push(Fragment::Text(std::mem::take(&mut acc)));
                            }
                            next.push(Fragment::Token(sid));
                            rest = &rest[m + needle.len()..];
                        }
                        acc.push_str(rest);
                        if !acc.is_empty() {
                            next.push(Fragment::Text(acc));
                        }
                    }
                }
            }
            frags = next;
        }
        frags
    }

    /// Core BPE over one raw-text fragment (`llm_tokenizer_bpe_session::tokenize`).
    fn bpe_tokenize(&self, text: &str, output: &mut Vec<u32>) {
        if self.spm_style {
            // gemma4 (llama PRE_TYPE_GEMMA4): escape spaces to \u2581 on the raw fragment,
            // split whole lines ([^\n]+|[\n]+), run BPE on raw UTF-8 chars.
            let escaped: String = text.chars().map(|c| if c == ' ' { '\u{2581}' } else { c }).collect();
            let mut words: Vec<String> = Vec::new();
            let mut cur = String::new();
            let mut cur_nl: Option<bool> = None;
            for c in escaped.chars() {
                let nl = c == '\n';
                if cur_nl != Some(nl) && !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
                cur_nl = Some(nl);
                cur.push(c);
            }
            if !cur.is_empty() { words.push(cur); }
            for word in &words {
                // newline-run fix (llama PR #21343): whole-word vocab hit short-circuits BPE.
                if word.chars().all(|c| c == '\n') {
                    if let Some(tok) = self.text_to_token(word) {
                        output.push(tok);
                        continue;
                    }
                }
                self.bpe_merge_word(word, output);
            }
            return;
        }
        // 1) pre-tokenizer split (qwen35), then 2) GPT-2 byte-encode each word.
        let words: Vec<String> = match self.pre.as_str() {
            "qwen35" => unicode::split_qwen35(text),
            other => {
                // Fall back to qwen35 split for the closely-related qwen2 family;
                // anything else is unsupported and would not be integer-exact.
                if other == "qwen2" {
                    unicode::split_qwen35(text)
                } else {
                    unicode::split_qwen35(text)
                }
            }
        };

        for word in &words {
            let word = unicode::byte_encode(word);
            self.bpe_merge_word(&word, output);
        }
    }

    /// BPE merge over one pre-split word (symbols = unicode chars), emitting token ids with
    /// byte fallback (gpt2 single-char byte tokens, or SPM <0xXX> tokens when spm_style).
    fn bpe_merge_word(&self, word: &str, output: &mut Vec<u32>) {
        {
            let word = word.to_string();

            // build the symbol chain, one symbol per unicode char initially.
            let chars: Vec<char> = word.chars().collect();
            let mut symbols: Vec<Symbol> = Vec::with_capacity(chars.len());
            for (i, &c) in chars.iter().enumerate() {
                symbols.push(Symbol {
                    text: c.to_string(),
                    prev: i as i32 - 1,
                    next: if i + 1 == chars.len() { -1 } else { i as i32 + 1 },
                    n: 1,
                });
            }

            // seed the work queue with adjacent bigrams.
            let mut queue: BinaryHeap<Bigram> = BinaryHeap::new();
            for i in 1..symbols.len() {
                self.add_bigram(&symbols, i as i32 - 1, i as i32, &mut queue);
            }

            // merge by rank.
            while let Some(bigram) = queue.pop() {
                let li = bigram.left as usize;
                let ri = bigram.right as usize;
                if symbols[li].n == 0 || symbols[ri].n == 0 {
                    continue;
                }
                let combined = format!("{}{}", symbols[li].text, symbols[ri].text);
                if combined != bigram.text {
                    continue; // outdated bigram
                }
                // merge right into left
                symbols[li].text = combined;
                symbols[li].n += symbols[ri].n;
                symbols[ri].n = 0;
                let r_next = symbols[ri].next;
                symbols[li].next = r_next;
                if r_next >= 0 {
                    symbols[r_next as usize].prev = bigram.left;
                }
                let l_prev = symbols[li].prev;
                let l_next = symbols[li].next;
                self.add_bigram(&symbols, l_prev, bigram.left, &mut queue);
                self.add_bigram(&symbols, bigram.left, l_next, &mut queue);
            }

            // emit final symbols in chain order, with byte-level fallback.
            for sym in &symbols {
                if sym.n == 0 {
                    continue;
                }
                match self.text_to_token(&sym.text) {
                    Some(tok) => output.push(tok),
                    None => {
                        // byte fallback: each *byte* of the piece must be its own token.
                        for b in sym.text.bytes() {
                            let bs = if self.spm_style {
                                format!("<0x{b:02X}>")   // SPM-style byte tokens (gemma4)
                            } else {
                                (b as char).to_string()
                            };
                            if let Some(t) = self.text_to_token(&bs) {
                                output.push(t);
                            }
                        }
                    }
                }
            }
        }
    }

    fn add_bigram(&self, symbols: &[Symbol], left: i32, right: i32, queue: &mut BinaryHeap<Bigram>) {
        if left == -1 || right == -1 {
            return;
        }
        let lt = &symbols[left as usize].text;
        let rt = &symbols[right as usize].text;
        let rank = self.find_bpe_rank(lt, rt);
        if rank < 0 {
            return;
        }
        queue.push(Bigram {
            left,
            right,
            rank,
            text: format!("{lt}{rt}"),
        });
    }

    /// Decode token ids -> String. `special=false` drops control tokens (chat tags);
    /// `special=true` renders them as their literal text.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_special(ids, true)
    }

    pub fn decode_special(&self, ids: &[u32], special: bool) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let i = id as usize;
            if i >= self.id_to_token.len() {
                continue;
            }
            let attr = self.attrs[i];
            let piece = &self.id_to_token[i];
            match attr {
                TokAttr::Normal | TokAttr::Byte => {
                    if self.spm_style {
                        // gemma4: <0xXX> byte tokens -> raw byte; else unescape \u2581 -> space.
                        if matches!(attr, TokAttr::Byte)
                            || (piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>')) {
                            if let Ok(b) = u8::from_str_radix(&piece[3..5], 16) {
                                bytes.push(b);
                                continue;
                            }
                        }
                        for c in piece.chars() {
                            if c == '\u{2581}' { bytes.push(b' '); }
                            else {
                                let mut buf = [0u8; 4];
                                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                            }
                        }
                    } else {
                        // undo GPT-2 byte encoding: each char -> one raw byte.
                        self.piece_to_bytes(piece, &mut bytes);
                    }
                }
                TokAttr::UserDefined => {
                    // user-defined tokens are literal text (not byte-encoded).
                    bytes.extend_from_slice(piece.as_bytes());
                }
                TokAttr::Control | TokAttr::Unknown => {
                    if special {
                        bytes.extend_from_slice(piece.as_bytes());
                    }
                    // else: render nothing
                }
                TokAttr::Other => {}
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn piece_to_bytes(&self, piece: &str, out: &mut Vec<u8>) {
        for c in piece.chars() {
            match unicode::unicode_to_byte(c) {
                Some(b) => out.push(b),
                None => {
                    // not in the byte map — emit the char's utf-8 bytes verbatim.
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }

    /// Apply the chat template (from GGUF, or a chatml fallback) to a list of
    /// (role, content) turns, producing the prompt string. Then `encode` it.
    pub fn apply_chat_template(
        &self,
        messages: &[(&str, &str)],
        add_generation_prompt: bool,
    ) -> String {
        chat::apply_chat_template_str(self.chat_template.as_deref(), messages, add_generation_prompt)
    }
}

enum Fragment {
    Text(String),
    Token(u32),
}

/// True when an HF `pre_tokenizer` object is byte-level BPE: type == "ByteLevel", or a
/// "Sequence" whose pretokenizers include a ByteLevel step (the common Split+ByteLevel combo).
fn pre_tokenizer_is_byte_level(pt: &json::Value) -> bool {
    match pt.get("type").and_then(|v| v.as_str()) {
        Some("ByteLevel") => true,
        Some("Sequence") => pt
            .get("pretokenizers")
            .and_then(|v| v.as_arr())
            .map(|arr| arr.iter().any(pre_tokenizer_is_byte_level))
            .unwrap_or(false),
        _ => false,
    }
}

#[cfg(test)]
mod hf_tests {
    use super::*;

    /// Inline tokenizer.json fixture: byte-level BPE, ~20 tokens incl one special added
    /// token, merges deliberately MIXED between the "a b" string format and the [a, b]
    /// pair format (HF emits both across tokenizers versions).
    const TOKENIZER_JSON: &str = r#"{
      "version": "1.0",
      "added_tokens": [
        {"id": 15, "content": "<|end|>", "special": true},
        {"id": 16, "content": "<think>", "special": false}
      ],
      "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
          {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated"},
          {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false}
        ]
      },
      "model": {
        "type": "BPE",
        "vocab": {
          "h": 0, "e": 1, "l": 2, "o": 3, "Ġ": 4, "w": 5, "r": 6, "d": 7,
          "he": 8, "ll": 9, "hell": 10, "hello": 11, "Ġw": 12, "or": 13, "!": 14
        },
        "merges": [
          "h e",
          ["l", "l"],
          "he ll",
          ["hell", "o"],
          ["Ġ", "w"],
          "o r"
        ]
      }
    }"#;

    fn write_fixture(name: &str, tokenizer_config: Option<&str>, generation_config: Option<&str>,
                     jinja: Option<&str>) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("bw24-tok-hf-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).unwrap();
        if let Some(tc) = tokenizer_config {
            std::fs::write(dir.join("tokenizer_config.json"), tc).unwrap();
        }
        if let Some(gc) = generation_config {
            std::fs::write(dir.join("generation_config.json"), gc).unwrap();
        }
        if let Some(j) = jinja {
            std::fs::write(dir.join("chat_template.jinja"), j).unwrap();
        }
        dir
    }

    #[test]
    fn hf_dir_encode_decode_roundtrip_and_specials() {
        // eos as an AddedToken OBJECT + chat_template string in tokenizer_config.
        let tc = r#"{
          "eos_token": {"content": "<|end|>", "lstrip": false},
          "add_bos_token": false,
          "chat_template": "{{ messages }}<|end|>"
        }"#;
        let dir = write_fixture("full", Some(tc), None, None);
        let tok = Tokenizer::from_hf_dir(&dir).expect("from_hf_dir");

        assert_eq!(tok.eos_id(), 15);
        assert_eq!(tok.bos_id(), None);
        assert_eq!(tok.pre(), "default");
        assert_eq!(tok.vocab_size(), 17); // ids 0..16 (added tokens extend the table)
        assert_eq!(tok.chat_template(), Some("{{ messages }}<|end|>"));

        // BPE over both merge formats: "hello world" -> hello(11) Ġw(12) or(13) l(2) d(7).
        // The 'hello' chain exercises string merges (h e / he ll), the pair merges
        // ([l,l] / [hell,o] / [Ġ,w]) fire inside the same words -> both formats load.
        let ids = tok.encode("hello world", true);
        assert_eq!(ids, vec![11, 12, 13, 2, 7]);
        assert_eq!(tok.decode(&ids), "hello world");

        // special handling: <|end|> (Control) is split out BEFORE BPE and never byte-merged.
        let ids = tok.encode("hello<|end|> world", true);
        assert_eq!(ids, vec![11, 15, 12, 13, 2, 7]);
        // decode with specials rendered vs dropped
        assert_eq!(tok.decode_special(&ids, true), "hello<|end|> world");
        assert_eq!(tok.decode_special(&ids, false), "hello world");

        // non-special added token stays Normal: decodes as literal text.
        assert_eq!(tok.decode(&[16]), "<think>");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hf_dir_generation_config_eos_fallback_and_jinja() {
        // no tokenizer_config eos -> generation_config eos_token_id (array form) must win;
        // chat template comes from chat_template.jinja.
        let gc = r#"{"eos_token_id": [15, 14]}"#;
        let dir = write_fixture("genconf", None, Some(gc), Some("JINJA {{ messages }}"));
        let tok = Tokenizer::from_hf_dir(&dir).expect("from_hf_dir");
        assert_eq!(tok.eos_id(), 15);
        assert!(!tok.encode("hello", true).is_empty());
        assert_eq!(tok.chat_template(), Some("JINJA {{ messages }}"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hf_dir_rejects_non_byte_level() {
        let dir = std::env::temp_dir()
            .join(format!("bw24-tok-hf-nonbl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let bad = TOKENIZER_JSON.replace("\"ByteLevel\"", "\"Metaspace\"");
        std::fs::write(dir.join("tokenizer.json"), bad).unwrap();
        assert!(Tokenizer::from_hf_dir(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
