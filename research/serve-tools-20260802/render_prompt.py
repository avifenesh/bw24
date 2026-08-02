#!/usr/bin/env python3
"""Python mirror of memra's qwen3.5/3.6-class chat rendering (crates/memra-tokenizer/src/chat.rs)
including the serve-tools lane's tools branch, plus a port of the emission parser
(crates/memra-server/src/toolcall.rs). Used by the gates to (a) reconstruct the exact rendered
prompt for tok-check token-count crosschecks and the /v1/completions bijection run, and (b)
verify parser equivalence on the raw completions text.

The byte laws here are copied from the committed template dumps
(research/onboard-ornith-20260801/templates/{ref-qwen36-35b,agentworld}.chat_template.jinja),
which were verified byte-identical to the deployed GGUFs' embedded templates on 2026-08-02.
"""

import json

INSTRUCTION = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n- If there is "
    "no function call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>"
)


def pyjson(v):
    """python json.dumps with default separators == the server's pyjson (tojson law)."""
    return json.dumps(v, ensure_ascii=False)


def render_param_value(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return pyjson(v)
    return json.dumps(v)  # true/3/null — JSON spelling (server law)


def render_tool_call_block(name, arguments):
    """Canonical <tool_call> block for a call (name, arguments dict, order preserved)."""
    s = "<tool_call>\n<function=" + name + ">\n"
    for k, v in arguments.items():
        s += "<parameter=" + k + ">\n" + render_param_value(v) + "\n</parameter>\n"
    s += "</function>\n</tool_call>"
    return s


def render_prompt(messages, tools=None, think="default"):
    """Mirror of apply_chat_template_tools for the qwen think-class templates,
    add_generation_prompt=True. `messages` = OpenAI-shape dicts (content str/None,
    optional tool_calls with function.name / function.arguments JSON-string)."""
    out = ""
    skip_first = False
    if tools:
        out += "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>"
        for t in tools:
            out += "\n" + pyjson(t)
        out += "\n</tools>" + INSTRUCTION
        if messages and messages[0]["role"] == "system":
            skip_first = True
            content = (messages[0].get("content") or "").strip()
            if content:
                out += "\n\n" + content
        out += "<|im_end|>\n"
    for i, m in enumerate(messages):
        if i == 0 and skip_first:
            continue
        role = m["role"]
        content = (m.get("content") or "").strip()
        if role in ("system", "user"):
            out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        elif role == "assistant":
            out += "<|im_start|>assistant\n" + content
            for k, tc in enumerate(m.get("tool_calls") or []):
                fn = tc["function"]
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) \
                    else fn["arguments"]
                if k == 0:
                    if content:
                        out += "\n\n"
                else:
                    out += "\n"
                out += render_tool_call_block(fn["name"], args)
            out += "<|im_end|>\n"
        elif role == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                out += "<|im_start|>user"
            out += "\n<tool_response>\n" + content + "\n</tool_response>"
            if i + 1 >= len(messages) or messages[i + 1]["role"] != "tool":
                out += "<|im_end|>\n"
        else:
            out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    out += "<think>\n\n</think>\n\n" if think == "nothink" else "<think>\n"
    return out


def parse_emission(text, schemas, skip_think):
    """Port of ToolStreamParser (whole-text mode): returns (content, [(name, args_dict)])."""
    content = ""
    calls = []
    pos = 0
    if skip_think:
        i = text.find("</think>")
        if i >= 0:
            pos = i + len("</think>")
            content += text[:pos]
        else:
            return text, []
    while True:
        i = text.find("<tool_call>", pos)
        if i < 0:
            content += text[pos:]
            break
        content += text[pos:i]
        j = text.find("</tool_call>", i)
        if j < 0:
            content += text[i:]  # unterminated: surfaced raw
            break
        inner = text[i + len("<tool_call>"):j]
        parsed = parse_block(inner, schemas)
        if parsed is None:
            content += text[i:j + len("</tool_call>")]  # malformed: surfaced verbatim
        else:
            calls.append(parsed)
        pos = j + len("</tool_call>")
    return content, calls


def parse_block(inner, schemas):
    s = inner.strip()
    if not s.startswith("<function="):
        return None
    rest = s[len("<function="):]
    gt = rest.find(">")
    if gt <= 0:
        return None
    name = rest[:gt]
    if any(c in name for c in "<>\n"):
        return None
    body = rest[gt + 1:]
    if not body.endswith("</function>"):
        return None
    body = body[:-len("</function>")]
    args = {}
    while True:
        t = body.lstrip()
        if not t:
            break
        if not t.startswith("<parameter="):
            return None
        r = t[len("<parameter="):]
        gt = r.find(">")
        if gt <= 0:
            return None
        key = r[:gt]
        if any(c in key for c in "<>\n"):
            return None
        after = r[gt + 1:]
        if after.startswith("\n"):
            after = after[1:]
        end = after.find("</parameter>")
        if end < 0:
            return None
        raw = after[:end]
        if raw.endswith("\n"):
            raw = raw[:-1]
        declared = schemas.get(name, {}).get(key)
        if declared in (None, "string"):
            args[key] = raw
        else:
            try:
                args[key] = json.loads(raw.strip())
            except json.JSONDecodeError:
                args[key] = raw
        body = after[end + len("</parameter>"):]
    return name, args
