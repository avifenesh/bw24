//! Unicode helpers ported 1:1 from llama.cpp `src/unicode.cpp` / `src/unicode.h`.
//!
//! Only the pieces the GPT-2/qwen35 BPE path needs are ported:
//!   - codepoint flags (`\p{L}`, `\p{N}`, `\p{M}`, `\s`) via the generated range table
//!   - `unicode_tolower` (binary search over the lowercase map)
//!   - the GPT-2 byte<->unicode map (`bytes_to_unicode`)
//!   - the hand-written `unicode_regex_split_custom_qwen35` pre-tokenizer state machine
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
