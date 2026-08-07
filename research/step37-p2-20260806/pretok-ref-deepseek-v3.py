#!/usr/bin/env python3
"""Independent reference for the `deepseek-v3` pre-tokenizer split (Step-3.7-Flash's
`tokenizer.ggml.pre`), used to cross-check memra's hand-written state machine in
`crates/memra-tokenizer/src/unicode.rs::split_deepseek_v3`.

Method — llama.cpp's algorithm, verbatim, but executed by a DIFFERENT regex engine:

  1. Read the codepoint-class table out of memra's own `unicode_data.rs` (which is itself a 1:1
     port of llama.cpp's `unicode-data.cpp`). Classification is therefore identical by
     construction and this script tests the SPLIT ALGORITHM, not the tables.
  2. Build the "collapsed" one-byte-per-codepoint text exactly as `unicode_regex_split` does
     (ASCII kept; non-ASCII -> 0x0B whitespace / 0xD1 N / 0xD2 L / 0xD3 P / 0xD4 M / 0xD5 S /
     0xD0 fallback).
  3. Apply the three DEEPSEEK3_LLM regexes in order over the accumulated offsets, emulating
     `unicode_regex_split_stl`'s regex_iterator + gap-emission loop. Regex 2 has non-ASCII
     literals and no \\p{} class, so upstream runs it on the CODEPOINT text (wregex path), not
     the collapsed text — reproduced here.

Patterns (llama.cpp `src/llama-vocab.cpp`, LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM):
    "\\p{N}{1,3}"
    "[一-龥぀-ゟ゠-ヿ]+"
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

Byte patterns are used for the collapsed passes so `\\s` is ASCII-only, matching C++
`std::regex` (Python's str-mode `\\s` also matches \\x1c-\\x1f, which would diverge).

Usage: python3 pretok-ref-deepseek-v3.py [--rust]   # --rust emits the Rust test corpus literal
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "crates" / "memra-tokenizer" / "src" / "unicode_data.rs"

NUMBER, LETTER, SEPARATOR, ACCENT_MARK = 0x0002, 0x0004, 0x0008, 0x0010
PUNCTUATION, SYMBOL, CONTROL, WHITESPACE = 0x0020, 0x0040, 0x0080, 0x0100
UNDEFINED = 0x0001
MASK_CATEGORIES = 0x00FF
MAX_CODEPOINTS = 0x110000


def load_flags():
    """Rebuild the cpt->flags table from memra's unicode_data.rs (same source as llama.cpp)."""
    src = DATA.read_text(encoding="utf-8")
    # Bound each array to its own block: UNICODE_MAP_LOWERCASE is ALSO a list of (0x..,0x..)
    # pairs, and sweeping the whole file would silently overwrite the flag table with
    # lowercase-map entries (observed: it mis-classified every non-ASCII letter).
    r_ini = src.index("UNICODE_RANGES_FLAGS")
    ws_ini = src.index("UNICODE_SET_WHITESPACE")
    lc_ini = src.index("UNICODE_MAP_LOWERCASE")
    ranges = [
        (int(a, 16), int(b, 16))
        for a, b in re.findall(
            r"\(0x([0-9A-Fa-f]+),\s*0x([0-9A-Fa-f]+)\)", src[r_ini:ws_ini]
        )
    ]
    assert len(ranges) == 2273, f"expected 2273 flag ranges, parsed {len(ranges)}"
    ws = [int(x, 16) for x in re.findall(r"0x([0-9A-Fa-f]+)", src[ws_ini:lc_ini])]
    assert len(ws) == 25, f"expected 25 whitespace cpts, parsed {len(ws)}"

    flags = bytearray(MAX_CODEPOINTS * 2)
    tbl = [UNDEFINED] * MAX_CODEPOINTS
    for i in range(1, len(ranges)):
        ini, fl = ranges[i - 1]
        end, _ = ranges[i]
        for cpt in range(ini, end):
            tbl[cpt] = fl
    for cpt in ws:
        if cpt < MAX_CODEPOINTS:
            tbl[cpt] |= WHITESPACE
    del flags
    return tbl


FLAGS = load_flags()

K_UCAT_CPT = {NUMBER: 0xD1, LETTER: 0xD2, PUNCTUATION: 0xD3, ACCENT_MARK: 0xD4, SYMBOL: 0xD5}


def collapse(cpts):
    out = bytearray()
    for c in cpts:
        if c < 128:
            out.append(c)
            continue
        fl = FLAGS[c] if c < MAX_CODEPOINTS else UNDEFINED
        if fl & WHITESPACE:
            out.append(0x0B)
        else:
            out.append(K_UCAT_CPT.get(fl & MASK_CATEGORIES, 0xD0))
    return bytes(out)


# collapsed character classes (k_ucat_cpt byte + k_ucat_map ASCII ranges), verbatim
L_CLS = rb"\xd2\x41-\x5a\x61-\x7a"
M_CLS = rb"\xd4"
P_CLS = rb"\xd3\x21-\x23\x25-\x2a\x2c-\x2f\x3a-\x3b\x3f-\x40\x5b-\x5d\x5f\x7b\x7d"
# S_CLS diverges from upstream's k_ucat_map ON PURPOSE: \x7e (~, category Sm) is missing from
# upstream's SYMBOL expansion ("$+<=>^`|", unicode.cpp:1244) — the one printable-ASCII codepoint
# where that map disagrees with real Unicode P/S. The HF training-time tokenizer's \p{S} DOES
# match '~' (" ~" pre-tokenizes to one word, 'Ġ~' id 6883). memra matches HF, not the upstream
# omission. Receipt: research/step-sku-20260807/raw/tok-parity-20260807T0640Z.log.
S_CLS = rb"\xd5\x24\x2b\x3c-\x3e\x5e\x60\x7c\x7e"

P1 = re.compile(rb"[\xd1\x30-\x39]{1,3}")
P2 = re.compile("[一-龥぀-ゟ゠-ヿ]+")
P3 = re.compile(
    rb"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"
    rb"|[^\r\n" + L_CLS + P_CLS + S_CLS + rb"]?[" + L_CLS + M_CLS + rb"]+"
    rb"| ?[" + P_CLS + S_CLS + rb"]+[\r\n]*"
    rb"|\s*[\r\n]+"
    rb"|\s+(?!\S)"
    rb"|\s+"
)


def split_pass(seq, offsets, pattern):
    """`unicode_regex_split_stl`: regex_iterator over each offset window, gaps emitted as words."""
    out = []
    start = 0
    for off in offsets:
        window = seq[start:start + off]
        start_idx = 0
        for m in pattern.finditer(window):
            if m.start() > start_idx:
                out.append(m.start() - start_idx)
            out.append(m.end() - m.start())
            start_idx = m.end()
        if start_idx < off:
            out.append(off - start_idx)
        start += off
    return out


def split_deepseek_v3(text):
    cpts = [ord(c) for c in text]
    coll = collapse(cpts)
    offsets = [len(cpts)]
    offsets = split_pass(coll, offsets, P1)
    offsets = split_pass(text, offsets, P2)
    offsets = split_pass(coll, offsets, P3)
    words, i = [], 0
    for off in offsets:
        words.append(text[i:i + off])
        i += off
    return words


CORPUS = [
    "Hello world",
    "Hello, world!",
    " leading and trailing ",
    "don't can't we're I've I'm you'll he'd",
    "1234567 89 0",
    "v0.71.0 and 128K ctx",
    "Step-3.7-Flash: 196B-A11B (45 blocks)",
    "line1\nline2\r\nline3",
    "trailing newlines\n\n\n",
    "tabs\tand\t\tspaces   x",
    "\n\n  \n indented",
    "中文测试",
    "混合 English 中文 123",
    "日本語のテスト、カタカナ",
    "한국어 테스트",
    "emoji 🚀 and symbols ~ ^ | $ +",
    "naïve café résumé",
    "Ünïcödé mÄrks",
    "áb̧c",
    "MoE top-8 288 experts@4096",
    "  ",
    " ",
    "",
    "\t",
    "\n",
    "x",
    "@#$%^&*()",
    "snake_case camelCase kebab-case",
    "path/to/file.gguf",
    "{\"key\": [1, 2, 3]}",
    "5e6 vs 1e4 rope base",
    "ЖИВЁТ русский текст",
    "Ελληνικά κείμενα",
    "العربية نص",
    "▁escaped▁space",
    "100%% sure? yes!!!",
    # --- adversarial: alternative-ordering and lookahead edges ---
    " .a",
    " a",
    "..",
    "a1",
    "1a",
    "12345678901234",     # digit run length not a multiple of 3
    " 123",
    "123 ",
    "  123  ",
    "-abc",               # alt1: ASCII-punct + [A-Za-z]+
    "-abc1",              # alt1 stops at the digit
    "~abc",               # alt1 (ASCII literal + letters) wins over alt3's \\p{S} run
    "~",                  # ~ alone is a \\p{S} run (memra includes \\x7e; upstream omits it)
    "~ ^",
    " nbsp",         # non-ASCII whitespace -> 0x0B stand-in
    "a  b",
    "x \n y",
    "x  \n\n  y",
    "end with space ",
    "end with spaces   ",
    "\r",
    "\r\r\n\n",
    " \n ",
    "́leading mark",  # combining mark with no base
    "中1文2",              # CJK pass interleaved with the digit pass
    "ーヽヾ",  # katakana tail of the pass-2 range
    "龥龦",        # 龥 is IN the class, 龦 is NOT (upper bound check)
    "぀〿",        # hiragana lower bound / just below it
]


def main():
    rust = "--rust" in sys.argv
    if rust:
        print("// generated by research/step37-p2-20260806/pretok-ref-deepseek-v3.py --rust")
        print("const DS3_CASES: &[(&str, &[&str])] = &[")
        for t in CORPUS:
            ws = split_deepseek_v3(t)
            body = ", ".join(rust_lit(w) for w in ws)
            print(f"    ({rust_lit(t)}, &[{body}]),")
        print("];")
        return
    for t in CORPUS:
        print(f"{t!r}\n  -> {split_deepseek_v3(t)!r}")


def rust_lit(s):
    out = '"'
    for ch in s:
        if ch == '"':
            out += '\\"'
        elif ch == "\\":
            out += "\\\\"
        elif ch == "\n":
            out += "\\n"
        elif ch == "\r":
            out += "\\r"
        elif ch == "\t":
            out += "\\t"
        elif ord(ch) < 0x20:
            out += "\\u{%x}" % ord(ch)
        else:
            out += ch
    return out + '"'


if __name__ == "__main__":
    main()
