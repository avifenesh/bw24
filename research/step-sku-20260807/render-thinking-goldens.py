#!/usr/bin/env python3
"""Golden renders of every supported arch's THINKING control, from the real templates.

Owner directive 2026-08-07: all supported models are thinking models; the serve surface
(`reasoning_effort`) must map to each arch's native mechanism. These goldens are the ground
truth the Rust arms in crates/memra-tokenizer/src/chat.rs are pinned against — rendered by
jinja2 (trim_blocks/lstrip_blocks True, the HF/minja settings) from the REAL shipped
templates:

  qwen class   research/onboard-ornith-20260801/templates/ref-qwen36-35b.chat_template.jinja
               (same enable_thinking/<think> block in ornith9b/35b, agentworld, kat — all
               five committed dumps carry identical markers)
  gemma4       research/step-sku-20260807/templates/gemma4-12b-qat.chat_template.jinja
               (dumped from the local gemma-4-12b-it-qat-q4_0.gguf header,
               sha256 36e3a42e...; model sha256 faff1a63...)
  hy3          research/step-sku-20260807/templates/hy3.chat_template.jinja
               (= tencent/Hy3 blob 4fdb6c56..., pinned snapshot 716aa724,
               sha256 7fc351fe...)
  step35       research/step37-bringup-20260802/raw/chat_template.jinja (f428623f...)
               — already pinned by render_step35_template.py; re-rendered here only for
               the cross-arch summary table.

Run: python3 research/step-sku-20260807/render-thinking-goldens.py
Out: research/step-sku-20260807/raw/thinking-goldens.txt (committed)
"""
import json
import pathlib

import jinja2

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "raw" / "thinking-goldens.txt"

env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True)
env.filters["tojson"] = lambda o, ensure_ascii=True, **kw: json.dumps(o, ensure_ascii=ensure_ascii, **kw)
env.filters["fromjson"] = json.loads


def tmpl(path):
    return env.from_string(pathlib.Path(path).read_text())


QWEN = tmpl(HERE.parent / "onboard-ornith-20260801/templates/ref-qwen36-35b.chat_template.jinja")
GEMMA = tmpl(HERE / "templates/gemma4-12b-qat.chat_template.jinja")
HY3 = tmpl(HERE / "templates/hy3.chat_template.jinja")
STEP = tmpl(HERE.parent / "step37-bringup-20260802/raw/chat_template.jinja")

MSGS_PLAIN = [{"role": "user", "content": "Hi"}]
MSGS_SYS = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}]

CASES = [
    # (label, template, kwargs)
    ("qwen default (thinking ON, open <think>)",
     QWEN, dict(messages=MSGS_PLAIN, add_generation_prompt=True)),
    ("qwen enable_thinking=false (closed think block)",
     QWEN, dict(messages=MSGS_PLAIN, add_generation_prompt=True, enable_thinking=False)),
    ("gemma4 default (enable_thinking undefined -> CLOSED thought channel)",
     GEMMA, dict(messages=MSGS_PLAIN, add_generation_prompt=True)),
    ("gemma4 enable_thinking=true, no system",
     GEMMA, dict(messages=MSGS_PLAIN, add_generation_prompt=True, enable_thinking=True)),
    ("gemma4 enable_thinking=true, with system",
     GEMMA, dict(messages=MSGS_SYS, add_generation_prompt=True, enable_thinking=True)),
    ("gemma4 enable_thinking=false explicit (must equal default)",
     GEMMA, dict(messages=MSGS_SYS, add_generation_prompt=True, enable_thinking=False)),
    ("hy3 default (reasoning_effort undefined -> no_think)",
     HY3, dict(messages=MSGS_PLAIN, add_generation_prompt=True)),
    ("hy3 reasoning_effort=low (OPEN think)",
     HY3, dict(messages=MSGS_PLAIN, add_generation_prompt=True, reasoning_effort="low")),
    ("hy3 reasoning_effort=high (OPEN think)",
     HY3, dict(messages=MSGS_SYS, add_generation_prompt=True, reasoning_effort="high")),
    ("hy3 reasoning_effort=no_think explicit (must equal default)",
     HY3, dict(messages=MSGS_PLAIN, add_generation_prompt=True, reasoning_effort="no_think")),
    ("hy3 assistant history stays closed-think at low",
     HY3, dict(messages=[{"role": "user", "content": "q"},
                         {"role": "assistant", "content": "a"},
                         {"role": "user", "content": "more"}],
               add_generation_prompt=True, reasoning_effort="low")),
    ("step35 reasoning_effort=medium (string in system turn; tail unconditional)",
     STEP, dict(messages=MSGS_PLAIN, add_generation_prompt=True, reasoning_effort="medium",
                bos_token="")),
]

lines = []
for label, t, kw in CASES:
    kw.setdefault("bos_token", "")
    r = t.render(**kw)
    lines.append(f"### {label}\n{r!r}\n")

OUT.write_text("\n".join(lines))
print(f"wrote {OUT}: {len(CASES)} goldens")
for label, t, kw in CASES:
    kw.setdefault("bos_token", "")
    print(f"--- {label}\n{t.render(**kw)!r}\n")
