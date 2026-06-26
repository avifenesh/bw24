//! bw24-tokenizer — host-only GPT-2/BPE tokenizer (encode + decode + chat template).
//!
//! Algorithm TAKEn ~1:1 from llama.cpp's GPT-2 BPE path (`src/llama-vocab.cpp`,
//! `src/unicode.cpp`), Rust glue hand-rolled. Built from the model's own GGUF
//! tokenizer metadata (`tokenizer.ggml.*`) so it is integer-exact for that model.
//!
//! Scope: the `gpt2` vocab model with the `qwen35` pre-tokenizer (Qwen3.5). Other
//! pre-tokenizers are not ported (we only need this model's).

mod chat;
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
        if model != "gpt2" {
            return Err(format!("unsupported tokenizer model '{model}' (only gpt2)"));
        }
        let pre = g
            .metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
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
        })
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
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
                            let bs = (b as char).to_string();
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
                    // undo GPT-2 byte encoding: each char -> one raw byte.
                    self.piece_to_bytes(piece, &mut bytes);
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
