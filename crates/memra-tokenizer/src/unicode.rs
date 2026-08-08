//! Unicode helpers ported 1:1 from llama.cpp `src/unicode.cpp` / `src/unicode.h`.
//!
//! Only the pieces the GPT-2/qwen35/deepseek-v3 BPE paths need are ported:
//!   - codepoint flags (`\p{L}`, `\p{N}`, `\p{M}`, `\p{P}`, `\p{S}`, `\s`) via the generated
//!     range table, plus `category_flag()` for the collapsed-text map
//!   - `unicode_tolower` (binary search over the lowercase map)
//!   - the GPT-2 byte<->unicode map (`bytes_to_unicode`)
//!   - the hand-written `unicode_regex_split_custom_qwen35` pre-tokenizer state machine
//!   - `split_deepseek_v3`: an equivalent for `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM`, which
//!     upstream runs through generic `std::regex` (three ordered passes over collapsed text)
//!     rather than a custom splitter. Required by Step-3.5/3.7-Flash.
//!
//! Keeping the classification table identical to llama.cpp is what makes the
//! pre-tokenizer split — and therefore the final token ids — integer-exact.

use crate::unicode_data::{UNICODE_MAP_LOWERCASE, UNICODE_RANGES_FLAGS, UNICODE_SET_WHITESPACE};
use std::collections::HashMap;
use std::sync::OnceLock;

const MAX_CODEPOINTS: usize = 0x110000;

// flag bits (llama.cpp `unicode_cpt_flags` enum). A few are unused by the qwen35
// split but kept for completeness / documentation of the table layout.
#[allow(dead_code)]
mod flag {
    pub const UNDEFINED: u16 = 0x0001;
    pub const NUMBER: u16 = 0x0002; // \p{N}
    pub const LETTER: u16 = 0x0004; // \p{L}
    pub const SEPARATOR: u16 = 0x0008; // \p{Z}
    pub const ACCENT_MARK: u16 = 0x0010; // \p{M}
    pub const PUNCTUATION: u16 = 0x0020; // \p{P}
    pub const SYMBOL: u16 = 0x0040; // \p{S}
    pub const CONTROL: u16 = 0x0080; // \p{C}
    pub const WHITESPACE: u16 = 0x0100; // \s
}
pub use flag::{
    ACCENT_MARK as FLAG_ACCENT_MARK, LETTER as FLAG_LETTER, NUMBER as FLAG_NUMBER,
    UNDEFINED as FLAG_UNDEFINED, WHITESPACE as FLAG_WHITESPACE,
};

/// Codepoint classification flags, mirroring `unicode_cpt_flags`.
/// We only carry the bits the qwen35 pre-tokenizer reads.
#[derive(Clone, Copy, Default)]
pub struct CptFlags(pub u16);

impl CptFlags {
    #[inline]
    pub fn is_number(self) -> bool {
        self.0 & FLAG_NUMBER != 0
    }
    #[inline]
    pub fn is_letter(self) -> bool {
        self.0 & FLAG_LETTER != 0
    }
    #[inline]
    pub fn is_accent_mark(self) -> bool {
        self.0 & FLAG_ACCENT_MARK != 0
    }
    #[inline]
    pub fn is_whitespace(self) -> bool {
        self.0 & FLAG_WHITESPACE != 0
    }
    /// matches `unicode_cpt_flags::as_uint()` for the bits we keep — used by the
    /// qwen35 split to test "any defined category at all".
    #[inline]
    pub fn as_uint(self) -> u16 {
        self.0
    }
    /// `unicode_cpt_flags::category_flag()` — `as_uint() & MASK_CATEGORIES`. Note this is the
    /// whole low byte, not a single bit: a codepoint carrying two category bits yields a value
    /// absent from llama.cpp's `k_ucat_cpt` map and therefore collapses to the 0xD0 fallback.
    #[inline]
    pub fn category_flag(self) -> u16 {
        self.0 & 0x00FF
    }
}

/// Build the full codepoint->flags table, exactly like `unicode_cpt_flags_array()`.
fn cpt_flags_table() -> &'static Vec<u16> {
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut flags = vec![FLAG_UNDEFINED; MAX_CODEPOINTS];
        // ranges: [start_i, start_{i+1}) gets range_i.flags
        for i in 1..UNICODE_RANGES_FLAGS.len() {
            let (ini, fl) = UNICODE_RANGES_FLAGS[i - 1];
            let (end, _) = UNICODE_RANGES_FLAGS[i];
            for cpt in ini..end {
                flags[cpt as usize] = fl;
            }
        }
        // whitespace OR-in (note: this OR matches llama's `is_whitespace = true`)
        for &cpt in UNICODE_SET_WHITESPACE.iter() {
            flags[cpt as usize] |= FLAG_WHITESPACE;
        }
        // (lowercase/uppercase/nfd bits are unused by the qwen35 pre-tokenizer)
        flags
    })
}

/// `unicode_cpt_flags_from_cpt` — out-of-range cpts get UNDEFINED (0x0001), matching llama.
#[inline]
pub fn cpt_flags_from_cpt(cpt: u32) -> CptFlags {
    let table = cpt_flags_table();
    if (cpt as usize) < MAX_CODEPOINTS {
        CptFlags(table[cpt as usize])
    } else {
        CptFlags(FLAG_UNDEFINED)
    }
}

/// `unicode_tolower` — binary search over the lowercase map, identity if absent.
#[inline]
pub fn tolower(cpt: u32) -> u32 {
    match UNICODE_MAP_LOWERCASE.binary_search_by(|&(k, _)| k.cmp(&cpt)) {
        Ok(idx) => UNICODE_MAP_LOWERCASE[idx].1,
        Err(_) => cpt,
    }
}

// ---- GPT-2 byte <-> unicode map (`bytes_to_unicode`) ----------------------------------

/// (byte -> unicode-codepoint, codepoint -> byte). Mirrors `unicode_byte_to_utf8_map`.
fn byte_unicode_maps() -> &'static (Vec<char>, HashMap<char, u8>) {
    static MAPS: OnceLock<(Vec<char>, HashMap<char, u8>)> = OnceLock::new();
    MAPS.get_or_init(|| {
        // byte -> char, exactly like the C++ map build order.
        let mut byte_to_char: Vec<Option<char>> = vec![None; 256];
        let mut set = |ch: u32| {
            byte_to_char[ch as usize] = Some(char::from_u32(ch).unwrap());
        };
        for ch in 0x21..=0x7E {
            set(ch);
        }
        for ch in 0xA1..=0xAC {
            set(ch);
        }
        for ch in 0xAE..=0xFF {
            set(ch);
        }
        let mut n: u32 = 0;
        for ch in 0..256u32 {
            if byte_to_char[ch as usize].is_none() {
                byte_to_char[ch as usize] = Some(char::from_u32(256 + n).unwrap());
                n += 1;
            }
        }
        let b2c: Vec<char> = byte_to_char.into_iter().map(|c| c.unwrap()).collect();
        let mut c2b: HashMap<char, u8> = HashMap::with_capacity(256);
        for (b, &c) in b2c.iter().enumerate() {
            c2b.insert(c, b as u8);
        }
        (b2c, c2b)
    })
}

/// Map one raw byte to its GPT-2 unicode char (`unicode_byte_to_utf8`).
#[inline]
pub fn byte_to_unicode(byte: u8) -> char {
    byte_unicode_maps().0[byte as usize]
}

/// Map a GPT-2 unicode char back to its raw byte (`unicode_utf8_to_byte`); None if not in map.
#[inline]
pub fn unicode_to_byte(c: char) -> Option<u8> {
    byte_unicode_maps().1.get(&c).copied()
}

/// GPT-2 byte-encode a raw &str: each *byte* becomes one unicode char.
/// Mirrors `unicode_byte_encoding_process` (which encodes per-byte, not per-cpt).
pub fn byte_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        out.push(byte_to_unicode(b));
    }
    out
}

// ---- qwen35 pre-tokenizer split -------------------------------------------------------

/// Port of `unicode_regex_split_custom_qwen35` (llama.cpp `src/unicode.cpp`).
///
/// Splits `text` (a UTF-8 string) into pre-token word boundaries, returning the
/// byte slices for each word. This is a deterministic codepoint-class state machine,
/// NOT a regex engine — that is exactly why it can be ported integer-exact.
///
/// The qwen35 regex it implements:
///   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N}
///     | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
pub fn split_qwen35(text: &str) -> Vec<String> {
    // codepoints + the byte length of each, so we can recover substrings.
    let cpts: Vec<u32> = text.chars().map(|c| c as u32).collect();
    let cpt_bytes: Vec<usize> = text.chars().map(|c| c.len_utf8()).collect();
    let n = cpts.len();

    const OOR: u32 = 0xFFFF_FFFF;
    let get_cpt = |pos: usize| -> u32 {
        if pos < n {
            cpts[pos]
        } else {
            OOR
        }
    };
    let get_flags = |pos: usize| -> CptFlags {
        if pos < n {
            cpt_flags_from_cpt(cpts[pos])
        } else {
            CptFlags::default()
        }
    };

    // emit token boundaries as codepoint counts, then convert to byte substrings.
    let mut lens: Vec<usize> = Vec::new(); // codepoint-length of each word
    let mut prev_end = 0usize;
    let add_token = |end: usize, prev_end: &mut usize, lens: &mut Vec<usize>| -> usize {
        debug_assert!(*prev_end <= end && end <= n);
        let len = end - *prev_end;
        if len > 0 {
            lens.push(len);
        }
        *prev_end = end;
        len
    };

    let mut pos = 0usize;
    while pos < n {
        let cpt = get_cpt(pos);
        let flags = get_flags(pos);

        // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if cpt == b'\'' as u32 && pos + 1 < n {
            let cpt_next = tolower(get_cpt(pos + 1));
            if cpt_next == 's' as u32
                || cpt_next == 't' as u32
                || cpt_next == 'm' as u32
                || cpt_next == 'd' as u32
            {
                pos += add_token(pos + 2, &mut prev_end, &mut lens);
                continue;
            }
            if pos + 2 < n {
                let cpt_nn = tolower(get_cpt(pos + 2));
                if (cpt_next == 'r' as u32 && cpt_nn == 'e' as u32)
                    || (cpt_next == 'v' as u32 && cpt_nn == 'e' as u32)
                    || (cpt_next == 'l' as u32 && cpt_nn == 'l' as u32)
                {
                    pos += add_token(pos + 3, &mut prev_end, &mut lens);
                    continue;
                }
            }
        }

        // regex: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if !(cpt == '\r' as u32 || cpt == '\n' as u32 || flags.is_number()) {
            if flags.is_letter()
                || flags.is_accent_mark()
                || get_flags(pos + 1).is_accent_mark()
                || get_flags(pos + 1).is_letter()
            {
                pos += 1;
                while get_flags(pos).is_letter() || get_flags(pos).is_accent_mark() {
                    pos += 1;
                }
                add_token(pos, &mut prev_end, &mut lens);
                continue;
            }
        }

        // regex: \p{N}
        if flags.is_number() {
            pos += 1;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // regex: <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        let mut flags2 = if cpt == ' ' as u32 {
            get_flags(pos + 1)
        } else {
            flags
        };
        if !(flags2.is_whitespace() || flags2.is_letter() || flags2.is_accent_mark() || flags2.is_number())
            && flags.as_uint() != 0
        {
            pos += (cpt == ' ' as u32) as usize;
            while !(flags2.is_whitespace()
                || flags2.is_letter()
                || flags2.is_accent_mark()
                || flags2.is_number())
                && flags2.as_uint() != 0
            {
                pos += 1;
                flags2 = get_flags(pos);
            }
            let mut cpt2 = get_cpt(pos);
            while cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                pos += 1;
                cpt2 = get_cpt(pos);
            }
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // count run of whitespace, remember last \r/\n end
        let mut num_ws = 0usize;
        let mut last_end_rn = 0usize;
        while get_flags(pos + num_ws).is_whitespace() {
            let cpt2 = get_cpt(pos + num_ws);
            if cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                last_end_rn = pos + num_ws + 1;
            }
            num_ws += 1;
        }

        // regex: \s*[\r\n]+
        if last_end_rn > 0 {
            pos = last_end_rn;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // regex: \s+(?!\S)
        if num_ws > 1 && get_cpt(pos + num_ws) != OOR {
            pos += num_ws - 1;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // regex: \s+
        if num_ws > 0 {
            pos += num_ws;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // no matches
        pos += 1;
        add_token(pos, &mut prev_end, &mut lens);
    }

    // convert codepoint-length words to byte substrings
    let mut words = Vec::with_capacity(lens.len());
    let mut cpt_i = 0usize;
    let mut byte_i = 0usize;
    for &len in &lens {
        let mut nbytes = 0usize;
        for k in 0..len {
            nbytes += cpt_bytes[cpt_i + k];
        }
        words.push(text[byte_i..byte_i + nbytes].to_string());
        cpt_i += len;
        byte_i += nbytes;
    }
    words
}

// ---- deepseek-v3 pre-tokenizer split --------------------------------------------------
//
// Step-3.7-Flash's `tokenizer.ggml.pre` is `deepseek-v3`
// (llama.cpp `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM`). Upstream has NO custom state machine for
// it — it runs three regexes through the generic `std::regex` path, in order, each pass
// subdividing the previous pass's offsets. This is a hand-written equivalent.
//
// The three patterns (`src/llama-vocab.cpp:318-325`):
//   1. "\p{N}{1,3}"
//   2. "[一-龥぀-ゟ゠-ヿ]+"                       (CJK ideographs + hiragana + katakana)
//   3. "[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
//       |[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
//       | ?[\p{P}\p{S}]+[\r\n]*
//       |\s*[\r\n]+ |\s+(?!\S) |\s+"
//
// Differences from qwen2/qwen35 that make a fall-through silently wrong: digits group in runs
// of up to 3 (not one per token), CJK/kana is its own isolated pass, the letter-run's optional
// leading character excludes \p{P}/\p{S} instead of \p{N}, the non-letter run is
// \p{P}/\p{S}-only (so undefined/control codepoints are NOT absorbed into it), and there is no
// contraction ('s/'t/...) alternative at all.
//
// Two upstream details that are easy to get wrong and are load-bearing here:
//   - Pass 3 runs on the COLLAPSED text (one byte per codepoint, non-ASCII mapped to a
//     category byte), so `\s` is std::regex's ASCII-only \s and every non-ASCII whitespace
//     codepoint has already become 0x0B (which IS ASCII \s). A codepoint carrying two category
//     bits collapses to the 0xD0 fallback and therefore matches NONE of \p{L}/\p{P}/\p{S}.
//   - `unicode_regex_split_stl` emits the GAP before each match as its own word, so a pass
//     never drops text; unmatched spans survive to the next pass.

/// Collapsed-text category byte for one codepoint (`unicode_regex_split`'s `k_ucat_cpt`).
#[inline]
fn collapse_cpt(cpt: u32) -> u8 {
    if cpt < 128 {
        return cpt as u8;
    }
    let fl = cpt_flags_from_cpt(cpt);
    if fl.is_whitespace() {
        return 0x0B; // <vertical tab> — llama's non-ASCII whitespace stand-in
    }
    match fl.category_flag() {
        FLAG_NUMBER => 0xD1,
        FLAG_LETTER => 0xD2,
        flag::PUNCTUATION => 0xD3,
        FLAG_ACCENT_MARK => 0xD4,
        flag::SYMBOL => 0xD5,
        _ => 0xD0, // undefined/separator/control, or any multi-category codepoint
    }
}

/// Classes over the collapsed byte alphabet. Each is `k_ucat_cpt[cat]` plus `k_ucat_map[cat]`
/// (the sub-128 codepoints of that category), exactly as the collapsed regex is built.
#[inline]
fn c_is_letter(b: u8) -> bool {
    b == 0xD2 || b.is_ascii_alphabetic()
}
#[inline]
fn c_is_mark(b: u8) -> bool {
    b == 0xD4 // no sub-128 accent marks
}
#[inline]
fn c_is_punct(b: u8) -> bool {
    b == 0xD3
        || matches!(b,
            0x21..=0x23 | 0x25..=0x2A | 0x2C..=0x2F | 0x3A..=0x3B | 0x3F..=0x40
            | 0x5B..=0x5D | 0x5F | 0x7B | 0x7D)
}
/// DELIBERATE divergence from upstream llama.cpp: `0x7E` (`~`, U+007E, category Sm) is
/// included here but MISSING from upstream's `k_ucat_map` SYMBOL expansion
/// (``"$+<=>^`|"``, `unicode.cpp:1244`, verified on master 2026-08-07) — the single
/// printable-ASCII codepoint where that map disagrees with real Unicode P/S (enumerated
/// over 0x21..0x7E). The HF reference tokenizer's `\p{S}` DOES match `~`, so upstream
/// splits `" ~"` as `[" ", "~"]` while the tokenizer the model was TRAINED with produces
/// `[" ~"]` (one pre-token, `Ġ~`). memra matches the training-time ground truth; receipt:
/// `research/step-sku-20260807/raw/tok-parity-20260807T0640Z.log` (the one corpus
/// mismatch before this fix, `symbols-spaced`, id 6883 `Ġ~` vs `223,96`).
#[inline]
fn c_is_symbol(b: u8) -> bool {
    b == 0xD5 || matches!(b, 0x24 | 0x2B | 0x3C..=0x3E | 0x5E | 0x60 | 0x7C | 0x7E)
}
#[inline]
fn c_is_number(b: u8) -> bool {
    b == 0xD1 || b.is_ascii_digit()
}
/// std::regex `\s` over the collapsed alphabet: ASCII space/\t/\n/\v/\f/\r only (every
/// non-ASCII whitespace codepoint is already 0x0B).
#[inline]
fn c_is_space(b: u8) -> bool {
    matches!(b, 0x20 | 0x09..=0x0D)
}
/// The ASCII punctuation literal class of pattern 3's first alternative,
/// `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]` — written out because it is NOT the same set as
/// `\p{P}` ∪ `\p{S}` collapsed (it is ASCII-literal and excludes the 0xD3/0xD5 category bytes).
#[inline]
fn c_is_ascii_punct_lit(b: u8) -> bool {
    matches!(b,
        0x21..=0x2F | 0x3A..=0x40 | 0x5B..=0x60 | 0x7B..=0x7E)
}

/// Emit a word boundary of `len` codepoints (the `_add_token` of the custom splitters).
#[inline]
fn push_len(lens: &mut Vec<usize>, len: usize) {
    if len > 0 {
        lens.push(len);
    }
}

/// One `unicode_regex_split_stl` pass: apply `matcher` inside each existing offset window,
/// emitting the gap before each match and then the match itself. `matcher(win_start, pos)`
/// returns the end index of a match starting at `pos`, or `None`.
fn split_pass<F>(offsets: &[usize], mut matcher: F) -> Vec<usize>
where
    F: FnMut(usize, usize, usize) -> Option<usize>,
{
    let mut out: Vec<usize> = Vec::with_capacity(offsets.len());
    let mut start = 0usize;
    for &off in offsets {
        let end = start + off;
        let mut gap = start; // start of the not-yet-emitted gap
        let mut pos = start;
        while pos < end {
            match matcher(start, end, pos) {
                Some(m_end) if m_end > pos => {
                    push_len(&mut out, pos - gap);
                    push_len(&mut out, m_end - pos);
                    gap = m_end;
                    pos = m_end;
                }
                // zero-width or no match: advance the scan, the span stays in the gap.
                _ => pos += 1,
            }
        }
        push_len(&mut out, end - gap);
        start = end;
    }
    out
}

/// Pattern 2's class: `[一-龥぀-ゟ゠-ヿ]` — CJK unified ideographs (U+4E00..U+9FA5, note the
/// upper bound is 龥 not 鿿), hiragana (U+3040..U+309F), katakana (U+30A0..U+30FF).
#[inline]
fn is_cjk_kana(cpt: u32) -> bool {
    (0x4E00..=0x9FA5).contains(&cpt)
        || (0x3040..=0x309F).contains(&cpt)
        || (0x30A0..=0x30FF).contains(&cpt)
}

/// Port of the `deepseek-v3` (DEEPSEEK3_LLM) pre-tokenizer split. Returns the pre-token word
/// substrings of `text`, in order, concatenating back to `text` exactly.
///
/// Cross-checked against an independent reference implementation driven by a different regex
/// engine: `research/step37-p2-20260806/pretok-ref-deepseek-v3.py`.
pub fn split_deepseek_v3(text: &str) -> Vec<String> {
    let cpts: Vec<u32> = text.chars().map(|c| c as u32).collect();
    let cpt_bytes: Vec<usize> = text.chars().map(|c| c.len_utf8()).collect();
    let n = cpts.len();
    let coll: Vec<u8> = cpts.iter().map(|&c| collapse_cpt(c)).collect();

    // ---- pass 1: \p{N}{1,3} (greedy, up to 3) ----
    let mut offsets = vec![n];
    offsets = split_pass(&offsets, |_s, end, pos| {
        if !c_is_number(coll[pos]) {
            return None;
        }
        let mut e = pos + 1;
        while e < end && e - pos < 3 && c_is_number(coll[e]) {
            e += 1;
        }
        Some(e)
    });

    // ---- pass 2: [CJK|kana]+ (runs on the codepoint text: no \p{} class, non-ASCII literals) ----
    offsets = split_pass(&offsets, |_s, end, pos| {
        if !is_cjk_kana(cpts[pos]) {
            return None;
        }
        let mut e = pos + 1;
        while e < end && is_cjk_kana(cpts[e]) {
            e += 1;
        }
        Some(e)
    });

    // ---- pass 3: the six-alternative pattern, leftmost-first (std::regex ECMAScript order) ----
    offsets = split_pass(&offsets, |_s, end, pos| {
        let b = coll[pos];

        // alt 1: [ASCII punct literal][A-Za-z]+
        if c_is_ascii_punct_lit(b) && pos + 1 < end && coll[pos + 1].is_ascii_alphabetic() {
            let mut e = pos + 2;
            while e < end && coll[e].is_ascii_alphabetic() {
                e += 1;
            }
            return Some(e);
        }

        // alt 2: [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
        // the optional lead is any single cpt that is NOT \r \n letter punct symbol; then one or
        // more letter/mark. Try WITH the lead first (leftmost-longest is not the rule here, but
        // ECMAScript takes the first alternative that matches at all and `X?Y+` is greedy on X).
        {
            let lead_ok = b != b'\r'
                && b != b'\n'
                && !c_is_letter(b)
                && !c_is_punct(b)
                && !c_is_symbol(b);
            let mut e = pos;
            if lead_ok && pos + 1 < end && (c_is_letter(coll[pos + 1]) || c_is_mark(coll[pos + 1]))
            {
                e = pos + 1;
            } else if !(c_is_letter(b) || c_is_mark(b)) {
                e = usize::MAX; // no viable letter run here
            }
            if e != usize::MAX {
                let run_start = e;
                while e < end && (c_is_letter(coll[e]) || c_is_mark(coll[e])) {
                    e += 1;
                }
                if e > run_start {
                    return Some(e);
                }
            }
        }

        // alt 3: ' ?[\p{P}\p{S}]+[\r\n]*'
        {
            let mut e = pos;
            if b == b' ' {
                e += 1;
            }
            let run_start = e;
            while e < end && (c_is_punct(coll[e]) || c_is_symbol(coll[e])) {
                e += 1;
            }
            if e > run_start {
                while e < end && (coll[e] == b'\r' || coll[e] == b'\n') {
                    e += 1;
                }
                return Some(e);
            }
        }

        // alt 4: \s*[\r\n]+
        if c_is_space(b) {
            // greedy \s* then require at least one \r\n; ECMAScript backtracks, so find the
            // LAST \r/\n reachable through an unbroken whitespace run.
            let mut e = pos;
            let mut last_rn = None;
            while e < end && c_is_space(coll[e]) {
                if coll[e] == b'\r' || coll[e] == b'\n' {
                    last_rn = Some(e + 1);
                }
                e += 1;
            }
            if let Some(rn_end) = last_rn {
                return Some(rn_end);
            }
            // alt 5: \s+(?!\S) — a whitespace run that is not followed by a non-space char.
            // With no \r\n in the run: if the run is followed by a non-space, backtrack one
            // codepoint so the lookahead sees whitespace; else take the whole run.
            let run_end = e;
            if run_end < end {
                if run_end - pos > 1 {
                    return Some(run_end - 1);
                }
                // single space followed by a non-space: alts 4/5 fail, alt 6 \s+ takes it.
                return Some(pos + 1);
            }
            return Some(run_end); // alt 5 at end-of-window
        }

        None
    });

    // ---- codepoint-length words back to byte substrings ----
    let mut words = Vec::with_capacity(offsets.len());
    let mut cpt_i = 0usize;
    let mut byte_i = 0usize;
    for &len in &offsets {
        let nbytes: usize = cpt_bytes[cpt_i..cpt_i + len].iter().sum();
        words.push(text[byte_i..byte_i + nbytes].to_string());
        cpt_i += len;
        byte_i += nbytes;
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    // generated by research/step37-p2-20260806/pretok-ref-deepseek-v3.py --rust
    const DS3_CASES: &[(&str, &[&str])] = &[
        ("Hello world", &["Hello", " world"]),
        ("Hello, world!", &["Hello", ",", " world", "!"]),
        (" leading and trailing ", &[" leading", " and", " trailing", " "]),
        ("don't can't we're I've I'm you'll he'd", &["don", "'t", " can", "'t", " we", "'re", " I", "'ve", " I", "'m", " you", "'ll", " he", "'d"]),
        ("1234567 89 0", &["123", "456", "7", " ", "89", " ", "0"]),
        ("v0.71.0 and 128K ctx", &["v", "0", ".", "71", ".", "0", " and", " ", "128", "K", " ctx"]),
        ("Step-3.7-Flash: 196B-A11B (45 blocks)", &["Step", "-", "3", ".", "7", "-Flash", ":", " ", "196", "B", "-A", "11", "B", " (", "45", " blocks", ")"]),
        ("line1\nline2\r\nline3", &["line", "1", "\n", "line", "2", "\r\n", "line", "3"]),
        ("trailing newlines\n\n\n", &["trailing", " newlines", "\n\n\n"]),
        ("tabs\tand\t\tspaces   x", &["tabs", "\tand", "\t", "\tspaces", "  ", " x"]),
        ("\n\n  \n indented", &["\n\n  \n", " indented"]),
        ("中文测试", &["中文测试"]),
        ("混合 English 中文 123", &["混合", " English", " ", "中文", " ", "123"]),
        ("日本語のテスト、カタカナ", &["日本語のテスト", "、", "カタカナ"]),
        ("한국어 테스트", &["한국어", " 테스트"]),
        ("emoji 🚀 and symbols ~ ^ | $ +", &["emoji", " 🚀", " and", " symbols", " ~", " ^", " |", " $", " +"]),
        ("naïve café résumé", &["naïve", " café", " résumé"]),
        ("Ünïcödé mÄrks", &["Ünïcödé", " mÄrks"]),
        ("áb̧c", &["áb̧c"]),
        ("MoE top-8 288 experts@4096", &["MoE", " top", "-", "8", " ", "288", " experts", "@", "409", "6"]),
        ("  ", &["  "]),
        (" ", &[" "]),
        ("", &[]),
        ("\t", &["\t"]),
        ("\n", &["\n"]),
        ("x", &["x"]),
        ("@#$%^&*()", &["@#$%^&*()"]),
        ("snake_case camelCase kebab-case", &["snake", "_case", " camelCase", " kebab", "-case"]),
        ("path/to/file.gguf", &["path", "/to", "/file", ".gguf"]),
        ("{\"key\": [1, 2, 3]}", &["{\"", "key", "\":", " [", "1", ",", " ", "2", ",", " ", "3", "]}"]),
        ("5e6 vs 1e4 rope base", &["5", "e", "6", " vs", " ", "1", "e", "4", " rope", " base"]),
        ("ЖИВЁТ русский текст", &["ЖИВЁТ", " русский", " текст"]),
        ("Ελληνικά κείμενα", &["Ελληνικά", " κείμενα"]),
        ("العربية نص", &["العربية", " نص"]),
        ("▁escaped▁space", &["▁", "escaped", "▁", "space"]),
        ("100%% sure? yes!!!", &["100", "%%", " sure", "?", " yes", "!!!"]),
        (" .a", &[" .", "a"]),
        (" a", &[" a"]),
        ("..", &[".."]),
        ("a1", &["a", "1"]),
        ("1a", &["1", "a"]),
        ("12345678901234", &["123", "456", "789", "012", "34"]),
        (" 123", &[" ", "123"]),
        ("123 ", &["123", " "]),
        ("  123  ", &["  ", "123", "  "]),
        ("-abc", &["-abc"]),
        ("-abc1", &["-abc", "1"]),
        ("~abc", &["~abc"]),
        ("~", &["~"]),
        ("~ ^", &["~", " ^"]),
        (" nbsp", &[" nbsp"]),
        ("a  b", &["a", " ", " b"]),
        ("x \n y", &["x", " \n", " y"]),
        ("x  \n\n  y", &["x", "  \n\n", " ", " y"]),
        ("end with space ", &["end", " with", " space", " "]),
        ("end with spaces   ", &["end", " with", " spaces", "   "]),
        ("\r", &["\r"]),
        ("\r\r\n\n", &["\r\r\n\n"]),
        (" \n ", &[" \n", " "]),
        ("́leading mark", &["́leading", " mark"]),
        ("中1文2", &["中", "1", "文", "2"]),
        ("ーヽヾ", &["ーヽヾ"]),
        ("龥龦", &["龥", "龦"]),
        ("぀〿", &["぀", "〿"]),
    ];

    /// `split_deepseek_v3` vs an independent reference driven by a DIFFERENT regex engine
    /// (`research/step37-p2-20260806/pretok-ref-deepseek-v3.py`, which executes llama.cpp's
    /// three-pass collapsed-text algorithm through Python's `re`). Corpus covers the cases where
    /// deepseek-v3 diverges from qwen2/qwen35: digit grouping in runs of <=3, the isolated
    /// CJK/kana pass, accented letter runs, the punct/symbol-only run, whitespace/newline runs,
    /// and the absence of any contraction alternative.
    #[test]
    fn deepseek_v3_split_matches_reference() {
        for (text, want) in DS3_CASES {
            let got = split_deepseek_v3(text);
            assert_eq!(got, *want, "split_deepseek_v3({text:?})");
            // a pre-tokenizer must never drop or reorder bytes
            assert_eq!(got.concat(), *text, "reassembly of {text:?}");
        }
    }

    /// Why the fall-through was a real bug, mechanism by mechanism. Each case below is one
    /// concrete way `split_qwen35` mis-splits deepseek-v3 text — if any of these ever start
    /// agreeing, the corresponding alternative has been ported wrong (or dropped).
    #[test]
    fn deepseek_v3_differs_from_qwen35_per_mechanism() {
        let q = |t: &str| split_qwen35(t);
        let d = |t: &str| split_deepseek_v3(t);

        // \p{N}{1,3} vs \p{N}: digits group in runs of up to three, left to right.
        assert_eq!(d("12345678901234"), ["123", "456", "789", "012", "34"]);
        assert_eq!(q("1234").len(), 4, "qwen35 emits one token per digit");

        // The isolated CJK/kana pass splits a kana/CJK run away from following punctuation
        // and from a preceding space that qwen35 would absorb.
        assert_eq!(d("日本語のテスト、カタカナ"), ["日本語のテスト", "、", "カタカナ"]);
        assert_eq!(q("日本語のテスト、カタカナ"), ["日本語のテスト", "、カタカナ"]);
        assert_eq!(d(" 中文"), [" ", "中文"], "pass 2 runs before the letter alternative");
        assert_eq!(q(" 中文"), [" 中文"]);

        // alt 3 is [\p{P}\p{S}]+ only, so a codepoint in NEITHER class (U+2581 ▁, category So?
        // no — it is \p{S}o... U+2581 is LOWER ONE EIGHTH BLOCK, \p{So}) still differs because
        // qwen35's alternative is the complement [^\s\p{L}\p{M}\p{N}] and absorbs the
        // following letters' leading position differently.
        assert_eq!(d("▁escaped▁space"), ["▁", "escaped", "▁", "space"]);
        assert_eq!(q("▁escaped▁space"), ["▁escaped", "▁space"]);

        // alt 3 is ' ?[\p{P}\p{S}]+' — a STRICT class, unlike qwen35's complement
        // [^\s\p{L}\p{N}]+ which also swallows format/control codepoints. ZWSP (U+200B, Cf)
        // is in neither \p{P} nor \p{S}, so deepseek-v3 leaves it as its own gap word while
        // qwen35 absorbs it into the punct run.
        assert_eq!(d("a\u{200b}!"), ["a", "\u{200b}", "!"]);
        assert_eq!(q("a\u{200b}!"), ["a", "\u{200b}!"]);
        // '~' IS \p{S} (U+007E, Sm) — upstream llama.cpp's k_ucat_map omits it from the
        // sub-128 SYMBOL expansion; memra deliberately includes it to match the HF
        // training-time tokenizer (see c_is_symbol). Pre-fix this split as
        // [" symbols", " ", "~", " ^"], which is what upstream still produces.
        assert_eq!(d(" symbols ~ ^"), [" symbols", " ~", " ^"]);

        // alt 1 has no counterpart in qwen35 at all: ASCII punctuation immediately followed by
        // ASCII letters is ONE token, and it stops at a non-letter.
        assert_eq!(d("-abc1"), ["-abc", "1"]);

        // No contraction alternative in deepseek-v3 — but the outcome coincides with qwen35
        // here because alt 2's optional lead picks up the apostrophe. Pinned so a future
        // "add the contractions back" edit has to justify itself.
        assert_eq!(d("don't"), ["don", "'t"]);
    }

    /// Sanity on the collapse map itself (`unicode_regex_split`'s k_ucat_cpt).
    #[test]
    fn collapse_map_category_bytes() {
        assert_eq!(collapse_cpt(b'a' as u32), b'a', "ASCII passes through");
        assert_eq!(collapse_cpt(0x4E2D), 0xD2, "CJK ideograph is a LETTER");
        assert_eq!(collapse_cpt(0x00E9), 0xD2, "e-acute is a LETTER");
        assert_eq!(collapse_cpt(0x0301), 0xD4, "combining acute is an ACCENT_MARK");
        assert_eq!(collapse_cpt(0x3001), 0xD3, "ideographic comma is PUNCTUATION");
        assert_eq!(collapse_cpt(0x00A0), 0x0B, "NBSP collapses to the ws stand-in");
        assert_eq!(collapse_cpt(0x0660), 0xD1, "Arabic-Indic digit is a NUMBER");
    }
}
