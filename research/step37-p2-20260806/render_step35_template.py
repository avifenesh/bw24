#!/usr/bin/env python3
"""Golden renders of the Step-3.7-Flash chat template, straight from the shipped jinja.

The GGUF's `tokenizer.chat_template` (5723 chars) is byte-identical to the HF repo's
`chat_template.jinja` committed at research/step37-bringup-20260802/raw/chat_template.jinja
(both dumped in phase 1). memra ships no jinja engine — `crates/memra-tokenizer/src/chat.rs`
reproduces each dialect in Rust — so this script is the ORACLE: it renders the real template
under jinja2 and emits the exact strings the Rust arm must produce.

Run: python3 research/step37-p2-20260806/render_step35_template.py
Output: research/step37-p2-20260806/raw/step35-template-goldens.txt (committed)

`bos_token` is rendered as the empty string on purpose: memra's `encode(add_special=true)`
prepends BOS from `tokenizer.ggml.add_bos_token` (True) / `bos_token_id` (0), so re-emitting
`{{bos_token}}` in the template text would double it — the same double-BOS trap the gemma4
arm documents.
"""
import json
import pathlib
import jinja2

HERE = pathlib.Path(__file__).resolve().parent
TMPL = HERE.parent / "step37-bringup-20260802" / "raw" / "chat_template.jinja"
OUT = HERE / "raw" / "step35-template-goldens.txt"

# trim_blocks/lstrip_blocks MUST be True: that is what HF transformers'
# _compile_jinja_template uses (ImmutableSandboxedEnvironment(trim_blocks=True,
# lstrip_blocks=True)) AND what llama.cpp's minja chat_template parses with. With them False
# the newline after this template's `{% endmacro %}` leaks into every render as a leading
# "\n" — an artifact of the harness, not of the model's prompt format.
env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True)
env.filters["tojson"] = lambda o, ensure_ascii=True, **kw: json.dumps(
    o, ensure_ascii=ensure_ascii, **kw)
env.filters["fromjson"] = json.loads
tmpl = env.from_string(TMPL.read_text())

WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}

CASES = [
    ("plain user", dict(messages=[{"role": "user", "content": "Hello"}],
                        add_generation_prompt=True)),
    ("system + user", dict(messages=[{"role": "system", "content": "You are helpful."},
                                     {"role": "user", "content": "Hi"}],
                           add_generation_prompt=True)),
    ("multi-turn (prior assistant is BEFORE last query -> NO think block)",
     dict(messages=[{"role": "system", "content": "rules"},
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "work"},
                    {"role": "user", "content": "more"}],
          add_generation_prompt=True)),
    ("no generation prompt", dict(messages=[{"role": "user", "content": "Hello"}],
                                  add_generation_prompt=False)),
    ("reasoning_effort, no system", dict(messages=[{"role": "user", "content": "Hi"}],
                                         add_generation_prompt=True,
                                         reasoning_effort="high")),
    ("reasoning_effort + system", dict(messages=[{"role": "system", "content": "Be terse."},
                                                 {"role": "user", "content": "Hi"}],
                                       add_generation_prompt=True,
                                       reasoning_effort="low")),
    ("tools + system", dict(messages=[{"role": "system", "content": "Be terse."},
                                      {"role": "user", "content": "Weather in Paris?"}],
                            add_generation_prompt=True, tools=[WEATHER_TOOL])),
    ("tools, no system", dict(messages=[{"role": "user", "content": "Weather in Paris?"}],
                              add_generation_prompt=True, tools=[WEATHER_TOOL])),
    ("tools + reasoning_effort + system",
     dict(messages=[{"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "Weather?"}],
          add_generation_prompt=True, tools=[WEATHER_TOOL], reasoning_effort="medium")),
    ("assistant tool_calls then tool response",
     dict(messages=[{"role": "user", "content": "Weather in Paris?"},
                    {"role": "assistant", "content": "",
                     "tool_calls": [{"type": "function", "function": {
                         "name": "get_weather", "arguments": {"city": "Paris"}}}]},
                    {"role": "tool", "content": "{\"temp_c\": 21}"}],
          add_generation_prompt=True, tools=[WEATHER_TOOL])),
    ("two consecutive tool turns group into ONE tool_response turn",
     dict(messages=[{"role": "user", "content": "both"},
                    {"role": "assistant", "content": "checking",
                     "tool_calls": [{"type": "function", "function": {
                                        "name": "a", "arguments": {"x": "1"}}},
                                    {"type": "function", "function": {
                                        "name": "b", "arguments": {}}}]},
                    {"role": "tool", "content": "r1"},
                    {"role": "tool", "content": "r2"}],
          add_generation_prompt=True, tools=[WEATHER_TOOL])),
    ("assistant with <think> in content (last-turn split)",
     dict(messages=[{"role": "user", "content": "q"},
                    {"role": "assistant", "content": "<think>\nreasoned\n</think>\nanswer"}],
          add_generation_prompt=False)),
    ("system-role observation turn (not first)",
     dict(messages=[{"role": "system", "content": "rules"},
                    {"role": "user", "content": "q"},
                    {"role": "system", "content": "obs text", "name": "observation"}],
          add_generation_prompt=True)),
    ("content NOT trimmed (no |trim in this template)",
     dict(messages=[{"role": "user", "content": "  padded  "}],
          add_generation_prompt=True)),
    ("user turn that IS a <tool_response> wrapper does NOT move last_query_index",
     dict(messages=[{"role": "user", "content": "real question"},
                    {"role": "assistant", "content": "thinking about it"},
                    {"role": "user", "content": "<tool_response>r</tool_response>"}],
          add_generation_prompt=True)),
    ("explicit reasoning_content field on an assistant turn",
     dict(messages=[{"role": "user", "content": "q"},
                    {"role": "assistant", "content": "answer",
                     "reasoning_content": "because"}],
          add_generation_prompt=False)),
    ("assistant AFTER last query with no think markers -> empty reasoning block",
     dict(messages=[{"role": "user", "content": "q"},
                    {"role": "assistant", "content": "plain"}],
          add_generation_prompt=False)),
    ("two tools in the <tools> block",
     dict(messages=[{"role": "user", "content": "q"}], add_generation_prompt=True,
          tools=[WEATHER_TOOL, {"type": "function", "function": {"name": "search"}}])),
    ("non-ASCII tool schema (template uses tojson(ensure_ascii=False))",
     dict(messages=[{"role": "user", "content": "q"}], add_generation_prompt=True,
          tools=[{"type": "function", "function": {"name": "f", "description": "café"}}])),
]

lines = [
    "# Step-3.7-Flash (arch step35) chat-template goldens",
    "# oracle: jinja2 %s over research/step37-bringup-20260802/raw/chat_template.jinja" % jinja2.__version__,
    "#         (byte-identical to the GGUF tokenizer.chat_template, 5723 chars)",
    "# bos_token rendered as '' — memra's encode(add_special) supplies BOS (double-BOS trap)",
    "",
]
for name, kwargs in CASES:
    kwargs.setdefault("bos_token", "")
    out = tmpl.render(**kwargs)
    lines.append("=== %s" % name)
    lines.append("--- args: %s" % json.dumps(
        {k: v for k, v in kwargs.items() if k != "bos_token"}, ensure_ascii=False))
    lines.append("--- rendered (python repr, exact bytes):")
    lines.append(repr(out))
    lines.append("")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines))
print("wrote %s (%d cases)" % (OUT, len(CASES)))
