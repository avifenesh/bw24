#!/usr/bin/env python3
"""Build the step35 tokenizer byte-parity corpus (Task #53 item 1, listing gate).

Emits corpus.tsv: one case per line, `<name>\t<hex of utf-8 bytes>`. Hex transport is
deliberate — the corpus is full of newlines, tabs, NBSP, emoji and template control tokens,
and hex is the one encoding both sides (this script's HF reference runner and the Rust
`tok-corpus` bin) can decode with zero escaping ambiguity.

Two layers of cases:
  1. RAW adversarial text — every mechanism of the deepseek-v3 pre-tokenizer split
     (\\p{N}{1,3} digit grouping, the CJK/kana literal ranges incl. their exact bounds,
     punct+letters, the collapsed-text \\s classes, the (?!\\S) end lookahead), plus
     byte-level BPE stressors (emoji ZWJ, combining marks, NBSP family) and special-token
     literals (<|im_start|> etc. must resolve to single ids on both sides).
  2. CHAT-TEMPLATE renders — the shipped chat_template.jinja rendered under the exact HF
     environment (trim_blocks/lstrip_blocks True, same as render_step35_template.py, whose
     goldens already pin memra's Rust renderer byte-for-byte). Tokenizing these strings
     end-to-ends the serve path by composition: render parity (goldens) + id parity (here).

Run:  python3 research/step-sku-20260807/build-tok-corpus.py
Out:  research/step-sku-20260807/corpus.tsv
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
TMPL = HERE.parent / "step37-bringup-20260802" / "raw" / "chat_template.jinja"
OUT = HERE / "corpus.tsv"

cases: list[tuple[str, str]] = []


def add(name: str, text: str) -> None:
    assert "\t" not in name and "\n" not in name, name
    cases.append((name, text))


# ---------------------------------------------------------------- raw: basics
add("empty", "")
add("ascii-hello", "Hello, world!")
add("leading-space-word", " hello")
add("contraction", "I don't, she'll, they're, it's")
add("mixed-case-runs", "XMLHttpRequest HTMLParser iOS macOS")

# ------------------------------------------- raw: digit grouping (\p{N}{1,3})
for n in range(1, 11):
    add(f"digits-len{n}", "1234567890"[:n])
add("digits-14", "12345678901234")
add("digits-in-words", "abc123def4567gh89")
add("decimal", "3.14159265358979")
add("negative", "-273.15 degrees")
add("thousands", "1,234,567.89")
add("version-string", "v2.7.18-rc3+build.4521")
add("arabic-indic-digits", "\u0661\u0662\u0663\u0664\u0665\u0666\u0667")  # ١٢٣٤٥٦٧ — \p{N}
add("superscript-digits", "x\u00b2 + y\u00b3 = z\u2074")  # ²³⁴ are \p{N}
add("fullwidth-digits", "\uff11\uff12\uff13\uff14\uff15")  # １２３４５
add("date-iso", "2026-08-07T06:00:00Z")
add("phone", "+1 (555) 010-4477 ext. 42")

# ------------------------- raw: CJK / kana literal ranges (incl. exact bounds)
add("cjk-sentence", "\u65e5\u672c\u8a9e\u306e\u30c6\u30b9\u30c8\u3001\u30ab\u30bf\u30ab\u30ca")
add("cjk-range-low", "\u4e00")           # 一 = low bound of [一-龥]
add("cjk-range-high", "\u9fa5")          # 龥 = high bound
add("cjk-past-high", "\u9fa6\u9fff")     # beyond the literal range — NOT in the class
add("hiragana-low", "\u3040")            # ぀ = low bound of [぀-ゟ] (unassigned cp, real bound)
add("hiragana-high", "\u309f")           # ゟ
add("katakana-low", "\u30a0")            # ゠
add("katakana-high", "\u30ff")           # ヿ
add("cjk-mixed-ascii", "Rust\u3067\u66f8\u304f\u30b3\u30fc\u30c9\u306f\u901f\u3044\u3002fast!")
add("cjk-space-cjk", "\u4e2d\u6587 \u4e2d\u6587\u3000\u4e2d\u6587")  # ascii + ideographic space
add("leading-space-cjk", " \u4e2d\u6587")
add("korean", "\uc548\ub155\ud558\uc138\uc694 \uc138\uacc4")  # Hangul — NOT in the CJK class
add("chinese-punct", "\u4f60\u597d\uff0c\u4e16\u754c\uff01\u300c\u5f15\u7528\u300d")

# ------------------------------------------------ raw: other scripts + marks
add("cyrillic", "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440!")
add("greek", "\u03b1\u03b2\u03b3\u03b4 \u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac")
add("arabic", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645")
add("hebrew", "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd")
add("thai-no-spaces", "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01")
add("devanagari-marks", "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e")
add("naive-diaeresis", "a na\u00efve caf\u00e9 r\u00e9sum\u00e9")
add("combining-accent", "e\u0301le\u0300ve co\u0308o\u0308perate")  # NFD combining marks (\p{M})
add("vietnamese", "Ti\u1ebfng Vi\u1ec7t r\u1ea5t hay")
add("turkish-dotless", "I\u0131\u0130i \u011f\u00fc\u015f\u00f6\u00e7")

# ----------------------------------------------------------------- raw: emoji
add("emoji-single", "hello \U0001f600 world")
add("emoji-zwj-family", "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466")
add("emoji-skin-tone", "\U0001f44d\U0001f3fd\U0001f44b\U0001f3ff")
add("emoji-flag", "\U0001f1ee\U0001f1f1 \U0001f1ef\U0001f1f5 flags")
add("emoji-variation-selector", "\u2764\ufe0f vs \u2764")
add("emoji-keycap", "#\ufe0f\u20e3 1\ufe0f\u20e3")
add("emoji-run", "\U0001f602\U0001f602\U0001f602\U0001f923\U0001f60a")

# ---------------------------------------------- raw: whitespace / \s classes
add("spaces-run", "a    b")
add("tabs", "a\tb\t\tc")
add("trailing-spaces", "end   ")
add("trailing-space-1", "end ")
add("spaces-then-word", "   indent")
add("newline-single", "line1\nline2")
add("newline-run", "a\n\n\nb")
add("crlf", "win\r\nline\r\n\r\nend")
add("cr-only", "old\rmac")
add("space-before-newline", "text  \n  more")
add("nbsp", "a\u00a0b \u00a0 c")
add("thin-space", "a\u2009b\u200af")
add("ideographic-space", "a\u3000b")
add("newline-tab-mix", "\n\t \n \t\n")
add("only-spaces", "     ")
add("only-newlines", "\n\n\n\n")
add("zero-width-space", "a\u200bb")   # ZWSP is Cf, not \s
add("indent-python", "def f():\n    if x:\n        return  # done\n")

# -------------------------------------------------- raw: punct+letters, code
add("punct-letters", "-abc1 .method ->foo #include @user")
add("escaped-space-char", "\u2581escaped\u2581space")  # ▁ (the SPM metachar, \p{S} here)
add("symbols-spaced", " symbols ~ ^ | \\ ")
add("rust-snippet", "fn main() { let x: Vec<u32> = (0..10).map(|i| i * 2).collect(); }")
add("json-snippet", '{"key": "value", "n": 42, "arr": [1, 2, 3], "nested": {"ok": true}}')
add("c-pointer", "int *p = &arr[0]; p += 3; // comment")
add("shell", "grep -rn 'foo.*bar' /tmp | awk '{print $2}' >> out.log 2>&1")
add("url", "https://example.com/path?q=hello%20world&lang=en#frag")
add("email", "user.name+tag@sub.example.co.uk")
add("markdown", "# Title\n\n- item **bold** `code`\n\n```rust\nlet a = 1;\n```\n")
add("regex-literal", r"^\d{3}-\d{4}\s+(?!\S)[A-Za-z]+$")
add("operators", "a==b != c <= d >= e && f || !g ? h : i")
add("underscores", "snake_case_name __dunder__ _private mixedCase_2")
add("quotes-nested", "He said \"she said 'nested' twice\" loudly.")
add("punct-run", "!!!???...,,,;;;:::")
add("brackets", "([{<>}]) (({[]}))")

# ------------------------------------- raw: special-token literals (id parity)
add("special-im-start-end", "<|im_start|>user\nhi<|im_end|>")
add("special-bos-literal", "<\uff5cbegin\u2581of\u2581sentence\uff5c>text")
add("special-eos-pad", "<\uff5cend\u2581of\u2581sentence\uff5c><\uff5c\u2581pad\u2581\uff5c>")
add("special-partial", "<|im_start and <|im_end without close")
add("special-think", "<think>\nreasoning here\n</think>\nanswer")
add("special-toolcall", "<tool_call>\n<function=get_x>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>")
add("special-tool-response", "<tool_response>output bytes</tool_response>")
add("special-vlm-tokens", "<im_start><im_patch><im_end><patch_start><patch_newline>")
add("special-adjacent", "<|im_end|><|im_start|>assistant")
add("special-case-miss", "<|IM_START|>not a token<|Im_End|>")

# ------------------------------------------------------------ raw: long mixes
add("mixed-kitchen-sink",
    "On 2026-08-07, \u5f20\u4f1f (age 34) paid \u20ac1,234.56 \u2014 \u0441\u043f\u0430\u0441\u0438\u0431\u043e! "
    "\U0001f389\U0001f389 See https://t.co/abc?x=1&y=22 \u0648\u0634\u0643\u0631\u0627\u064b.\n\n"
    "\tdef f(n: int) -> str:\n\t\treturn f\"{n:>8,}\"  # 12345678901234\n")
add("repeat-word", ("token " * 50).rstrip())
add("long-digit-mix", " ".join(str(7 ** k) for k in range(1, 20)))

# ------------------------------------------------- layer 2: template renders
import jinja2  # noqa: E402

env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True)
env.filters["tojson"] = lambda o, ensure_ascii=True, **kw: json.dumps(o, ensure_ascii=ensure_ascii, **kw)
env.filters["fromjson"] = json.loads
tmpl = env.from_string(TMPL.read_text())

WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}

TCASES = [
    ("tmpl-plain-user",
     dict(messages=[{"role": "user", "content": "Hello"}], add_generation_prompt=True)),
    ("tmpl-system-user",
     dict(messages=[{"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"}], add_generation_prompt=True)),
    ("tmpl-empty-user",
     dict(messages=[{"role": "user", "content": ""}], add_generation_prompt=True)),
    ("tmpl-empty-system",
     dict(messages=[{"role": "system", "content": ""},
                    {"role": "user", "content": "q"}], add_generation_prompt=True)),
    ("tmpl-reasoning-low",
     dict(messages=[{"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "Hi"}],
          add_generation_prompt=True, reasoning_effort="low")),
    ("tmpl-reasoning-medium",
     dict(messages=[{"role": "user", "content": "Hi"}],
          add_generation_prompt=True, reasoning_effort="medium")),
    ("tmpl-reasoning-high-tools",
     dict(messages=[{"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "Weather?"}],
          add_generation_prompt=True, tools=[WEATHER_TOOL], reasoning_effort="high")),
    ("tmpl-multiturn-think",
     dict(messages=[{"role": "user", "content": "task"},
                    {"role": "assistant", "content": "<think>\nplan\n</think>\nanswer one"},
                    {"role": "user", "content": "more"},
                    {"role": "assistant", "content": "<think>\nplan two\n</think>\nanswer two"}],
          add_generation_prompt=False)),
    ("tmpl-toolcall-roundtrip",
     dict(messages=[{"role": "user", "content": "Weather in Paris?"},
                    {"role": "assistant", "content": "",
                     "tool_calls": [{"type": "function", "function": {
                         "name": "get_weather", "arguments": {"city": "Paris"}}}]},
                    {"role": "tool", "content": '{"temp_c": 21}'},
                    {"role": "tool", "content": "second result"}],
          add_generation_prompt=True, tools=[WEATHER_TOOL])),
    ("tmpl-unicode-content",
     dict(messages=[{"role": "system", "content": "\u4f60\u662f\u52a9\u624b\u3002"},
                    {"role": "user", "content": "12345678901234 \U0001f600  \u65e5\u672c\u8a9e\n\n  indent"}],
          add_generation_prompt=True, reasoning_effort="high")),
]
for name, kw in TCASES:
    add(name, tmpl.render(bos_token="", **kw))

# ---------------------------------------------------------------------- emit
with OUT.open("w") as f:
    for name, text in cases:
        f.write(f"{name}\t{text.encode('utf-8').hex()}\n")
print(f"wrote {OUT}: {len(cases)} cases "
      f"({sum(1 for n, _ in cases if n.startswith('tmpl-'))} template renders)")
