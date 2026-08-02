//! Minimal chat-template renderer for the Qwen3.5 / ChatML format.
//!
//! The model's GGUF `tokenizer.chat_template` is a large jinja template covering
//! tools, vision, and multi-step reasoning. We do NOT ship a jinja engine; instead
//! we reproduce the text-only system/user/assistant path of that template exactly,
//! which is the path memra's text-in/text-out CLI uses. The reproduced behavior
//! (verified against the dumped template):
//!
//!   - a leading `system` turn renders `<|im_start|>system\n{content}<|im_end|>\n`
//!   - `user`      -> `<|im_start|>user\n{content}<|im_end|>\n`
//!   - `assistant` -> `<|im_start|>assistant\n{content}<|im_end|>\n`
//!   - with `add_generation_prompt`, Qwen3.5 appends `<|im_start|>assistant\n<think>\n`
//!     (its default, since `enable_thinking` is undefined => the else-branch fires).
//!
//! `content` is trimmed (the template applies `|trim`). If the GGUF has no template
//! we fall back to plain ChatML (no `<think>` tail).

/// One tool call attached to a prior assistant turn, pre-rendered for the template:
/// `params` values are already strings per the template law (string arguments raw,
/// everything else JSON-rendered by the caller — this crate stays serde-free).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub params: Vec<(String, String)>,
}

/// One chat turn for the tools-capable renderer (`apply_chat_template_tools`).
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Generation-prompt think tail. `Default` = the template's own default (the qwen3.5/3.6
/// class opens `<think>\n` — verified against the committed dumps in
/// research/onboard-ornith-20260801/templates/); `NoThink` = the template's
/// `enable_thinking=false` switch (closed `<think>\n\n</think>\n\n`). On templates
/// without an `enable_thinking` switch the mode is ignored (graceful no-op).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkMode {
    Default,
    NoThink,
}

/// Render messages into the prompt string.
///
/// `template` is the raw GGUF chat_template (used only to decide qwen3.5-vs-plain
/// chatml behavior — we detect the `<think>` generation tail by substring). When
/// `None`, plain ChatML is produced.
pub fn apply_chat_template_str(
    template: Option<&str>,
    messages: &[(&str, &str)],
    add_generation_prompt: bool,
) -> String {
    // Tencent Hy3 (`hy_v3`): a completely different special-token dialect (no ChatML).
    // Detected by its `hy_User` token literal; rendered by the dedicated arm below.
    if template.is_some_and(|t| t.contains("hy_User")) {
        return apply_hy3_template(messages, add_generation_prompt);
    }
    // gemma4: `<|turn>role\n{content}<turn|>\n` dialect; generation prompt appends
    // `<|turn>model\n` + the CLOSED thought channel (`<|channel>thought\n<channel|>` — the
    // template's enable_thinking-false default). bos comes from encode(add_special) — the
    // template's `{{ bos_token }}` is NOT re-emitted here (double-BOS trap).
    if template.is_some_and(|t| t.contains("<|turn>")) {
        return apply_gemma4_template(messages, add_generation_prompt);
    }
    // qwen3.5 template emits a `<think>\n` tail on the generation prompt by default.
    let qwen_think = template
        .map(|t| t.contains("<think>") && t.contains("add_generation_prompt"))
        .unwrap_or(false);

    let mut out = String::new();
    for (i, (role, content)) in messages.iter().enumerate() {
        let content = content.trim();
        match *role {
            "system" => {
                // template requires system at the beginning; we render it wherever
                // it appears at index 0 (the common case).
                let _ = i;
                out.push_str("<|im_start|>system\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            other => {
                // unsupported role in this minimal renderer; emit as a generic turn.
                out.push_str("<|im_start|>");
                out.push_str(other);
                out.push('\n');
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
        }
    }

    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if qwen_think {
            out.push_str("<think>\n");
        }
    }

    out
}

/// The fixed tool-calling instruction block of the qwen3.5/3.6-class templates. Byte-for-byte
/// the string literal shared by ornith9b / agentworld / ref-qwen36-35b
/// (research/onboard-ornith-20260801/templates/*.jinja) and the deployed GGUF dumps.
const QWEN_TOOLS_INSTRUCTION: &str = "\n\nIf you choose to call a function ONLY reply in the \
following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n\
This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified \
format: an inner <function=...></function> block must be nested within <tool_call></tool_call> \
XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for \
your function call in natural language BEFORE the function call, but NOT after\n- If there is \
no function call available, answer the question like normal with your current knowledge and do \
not tell the user about function calls\n</IMPORTANT>";

/// Tools-capable chat rendering (serve-tools lane, 2026-08-02). Reproduces the TOOLS branch of
/// the qwen3.5/3.6-class ChatML templates exactly (verified against the committed dumps AND the
/// deployed GGUFs' embedded templates, byte-identical):
///
///   - tools present  -> `<|im_start|>system\n# Tools\n\nYou have access to the following
///     functions:\n\n<tools>` + `\n{tool json}` each + `\n</tools>` + the fixed instruction
///     block; a leading system turn's trimmed content is appended after `\n\n`; `<|im_end|>\n`.
///   - assistant turns with `tool_calls` -> content then `<tool_call>\n<function=NAME>\n`
///     (+`\n\n` separator when content is non-empty; later calls separated by `\n`),
///     `<parameter=K>\nV\n</parameter>\n` each, `</function>\n</tool_call>`, then `<|im_end|>\n`.
///   - `tool` turns -> grouped into ONE user turn: `<|im_start|>user` opens a run of
///     consecutive tool messages, each `\n<tool_response>\n{content}\n</tool_response>`,
///     `<|im_end|>\n` closes the run.
///   - generation prompt -> `<|im_start|>assistant\n` + `<think>\n` (template default) or
///     `<think>\n\n</think>\n\n` (`ThinkMode::NoThink` = the template's `enable_thinking=false`
///     switch; ignored when the template has no `enable_thinking`).
///
/// The no-tools/no-tool-turns/`Default`-think case renders byte-identically to
/// `apply_chat_template_str` (pinned by `tools_renderer_matches_legacy_when_plain`); callers
/// that want the hard isolation guarantee keep calling the legacy function on that path.
/// Errors (never on the plain path): tools/tool turns on a template without a tools branch
/// (hy3 / gemma4 / bare ChatML).
pub fn apply_chat_template_tools(
    template: Option<&str>,
    turns: &[Turn],
    add_generation_prompt: bool,
    tools_json: &[String],
    think: ThinkMode,
) -> Result<String, String> {
    let has_tool_features = !tools_json.is_empty()
        || turns.iter().any(|t| t.role == "tool" || !t.tool_calls.is_empty());
    let tools_branch = template.is_some_and(|t| t.contains("<tools>"));
    if has_tool_features && !tools_branch {
        return Err("model chat template has no tools branch".into());
    }
    if template.is_some_and(|t| t.contains("hy_User") || t.contains("<|turn>")) {
        // hy3 / gemma4 dialects: no committed tools rendering reference — reject tool
        // features even if the raw jinja happens to mention <tools>; the plain path stays
        // on the legacy arms and ThinkMode is ignored (graceful, per the mission contract).
        if has_tool_features {
            return Err("tools are not supported on this model's chat-template dialect".into());
        }
        let messages: Vec<(&str, &str)> =
            turns.iter().map(|t| (t.role.as_str(), t.content.as_str())).collect();
        return Ok(apply_chat_template_str(template, &messages, add_generation_prompt));
    }
    let qwen_think = template
        .map(|t| t.contains("<think>") && t.contains("add_generation_prompt"))
        .unwrap_or(false);
    let think_switch = template.is_some_and(|t| t.contains("enable_thinking"));

    let mut out = String::new();
    // Tools system header replaces the plain system turn (template law: the leading system
    // turn's content is folded INTO the tools block).
    let mut skip_leading_system = false;
    if !tools_json.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in tools_json {
            out.push('\n');
            out.push_str(tool);
        }
        out.push_str("\n</tools>");
        out.push_str(QWEN_TOOLS_INSTRUCTION);
        if let Some(first) = turns.first() {
            if first.role == "system" {
                skip_leading_system = true;
                let content = first.content.trim();
                if !content.is_empty() {
                    out.push_str("\n\n");
                    out.push_str(content);
                }
            }
        }
        out.push_str("<|im_end|>\n");
    }

    for (i, turn) in turns.iter().enumerate() {
        if i == 0 && skip_leading_system {
            continue;
        }
        let content = turn.content.trim();
        match turn.role.as_str() {
            "system" => {
                out.push_str("<|im_start|>system\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n");
                out.push_str(content);
                for (k, call) in turn.tool_calls.iter().enumerate() {
                    if k == 0 {
                        if !content.is_empty() {
                            out.push_str("\n\n");
                        }
                    } else {
                        out.push('\n');
                    }
                    out.push_str("<tool_call>\n<function=");
                    out.push_str(&call.name);
                    out.push_str(">\n");
                    for (key, value) in &call.params {
                        out.push_str("<parameter=");
                        out.push_str(key);
                        out.push_str(">\n");
                        out.push_str(value);
                        out.push_str("\n</parameter>\n");
                    }
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                if i == 0 || turns[i - 1].role != "tool" {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response>");
                if i + 1 >= turns.len() || turns[i + 1].role != "tool" {
                    out.push_str("<|im_end|>\n");
                }
            }
            other => {
                // parity with the legacy renderer's generic-turn arm.
                out.push_str("<|im_start|>");
                out.push_str(other);
                out.push('\n');
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
        }
    }

    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if qwen_think {
            if think == ThinkMode::NoThink && think_switch {
                out.push_str("<think>\n\n</think>\n\n");
            } else {
                out.push_str("<think>\n");
            }
        }
    }
    Ok(out)
}

/// Text-only reproduction of the Hy3 `chat_template.jinja` default path (no tools, no
/// `is_training`, `reasoning_effort` undefined => template defaults it to `'no_think'`):
///   - `{bos}{system…}<｜reasoning_mode:opensource｜>reasoning_effort:no_think` header
///     (system turns concatenate into the header, before any user turn);
///   - `user`      -> `<｜hy_User:opensource｜>{content}`
///   - `assistant` -> `<｜hy_Assistant:opensource｜><think:opensource></think:opensource>{content}<｜hy_eos:opensource｜>`
///     (non-last turns; thinking is not preserved on the text path);
///   - generation prompt (no_think): `<｜hy_Assistant:opensource｜><think:opensource></think:opensource>`.
/// Content is NOT trimmed (the Hy3 template applies no `|trim`).
fn apply_hy3_template(messages: &[(&str, &str)], add_generation_prompt: bool) -> String {
    const BOS: &str = "<\u{ff5c}hy_begin_of_sentence:opensource\u{ff5c}>";
    const USER: &str = "<\u{ff5c}hy_User:opensource\u{ff5c}>";
    const ASSISTANT: &str = "<\u{ff5c}hy_Assistant:opensource\u{ff5c}>";
    const EOS: &str = "<\u{ff5c}hy_eos:opensource\u{ff5c}>";
    const REASONING: &str = "<\u{ff5c}reasoning_mode:opensource\u{ff5c}>";
    const THINK_BEGIN: &str = "<think:opensource>";
    const THINK_END: &str = "</think:opensource>";

    let mut out = String::from(BOS);
    for (role, content) in messages.iter().filter(|(r, _)| *r == "system") {
        let _ = role;
        out.push_str(content);
    }
    out.push_str(REASONING);
    out.push_str("reasoning_effort:no_think");

    let mut last_is_assistant = false;
    let n = messages.len();
    for (i, (role, content)) in messages.iter().enumerate() {
        last_is_assistant = false;
        match *role {
            "user" => { out.push_str(USER); out.push_str(content); }
            "assistant" => {
                out.push_str(ASSISTANT);
                out.push_str(THINK_BEGIN);
                out.push_str(THINK_END);
                out.push_str(content);
                if i + 1 < n { out.push_str(EOS); }   // template: `not loop.last` gets eos
                last_is_assistant = true;
            }
            _ => {} // system handled in the header; tool turns are out of scope here
        }
    }
    if add_generation_prompt && !last_is_assistant {
        out.push_str(ASSISTANT);
        out.push_str(THINK_BEGIN);
        out.push_str(THINK_END);
    }
    out
}


/// gemma4 turn dialect (text-only path of the GGUF template, verified against the dumped
/// jinja): roles map assistant->model; each turn = `<|turn>{role}\n{content|trim}<turn|>\n`;
/// generation prompt = `<|turn>model\n<|channel>thought\n<channel|>`.
fn apply_gemma4_template(messages: &[(&str, &str)], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        let role = if *role == "assistant" { "model" } else { role };
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(content.trim());
        out.push_str("<turn|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_chatml() {
        let s = apply_chat_template_str(
            None,
            &[("user", "Hello")],
            true,
        );
        assert_eq!(s, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");
    }

    /// A template stand-in carrying every marker the real qwen3.5/3.6 dumps carry
    /// (tools branch + think tail + enable_thinking switch).
    const QWEN_TOOLS_TMPL: &str =
        "... <tools> ... add_generation_prompt ... enable_thinking ... '<think>\\n' ...";

    /// Isolation contract: the tools renderer on a PLAIN request (no tools, no tool turns,
    /// Default think) is byte-identical to the legacy renderer, across the message shapes
    /// the serve path sees.
    #[test]
    fn tools_renderer_matches_legacy_when_plain() {
        let batteries: &[&[(&str, &str)]] = &[
            &[("user", "Hello")],
            &[("system", "You are helpful."), ("user", "Hi")],
            &[("system", "rules"), ("user", "task"), ("assistant", "work"), ("user", "more")],
            &[("user", "  padded  "), ("assistant", "reply\nwith lines")],
        ];
        for tmpl in [None, Some(QWEN_TOOLS_TMPL)] {
            for msgs in batteries {
                let legacy = apply_chat_template_str(tmpl, msgs, true);
                let turns: Vec<Turn> = msgs.iter().map(|(r, c)| Turn {
                    role: r.to_string(), content: c.to_string(), tool_calls: Vec::new(),
                }).collect();
                let ext = apply_chat_template_tools(tmpl, &turns, true, &[], ThinkMode::Default)
                    .unwrap();
                assert_eq!(legacy, ext, "template={tmpl:?} msgs={msgs:?}");
            }
        }
    }

    #[test]
    fn tools_header_and_tool_response_render_per_template_law() {
        let tools = vec![r#"{"type": "function", "function": {"name": "get_weather"}}"#.to_string()];
        let turns = vec![
            Turn { role: "system".into(), content: "Be terse.".into(), tool_calls: Vec::new() },
            Turn { role: "user".into(), content: "Weather in Paris?".into(), tool_calls: Vec::new() },
            Turn { role: "assistant".into(), content: "".into(), tool_calls: vec![ToolCall {
                name: "get_weather".into(),
                params: vec![("city".into(), "Paris".into())],
            }] },
            Turn { role: "tool".into(), content: "{\"temp_c\": 21}".into(), tool_calls: Vec::new() },
        ];
        let s = apply_chat_template_tools(Some(QWEN_TOOLS_TMPL), &turns, true, &tools,
                                          ThinkMode::Default).unwrap();
        let expected = concat!(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n",
            "<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\"}}\n</tools>",
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:",
            "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n",
            "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the ",
            "second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>",
            "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner ",
            "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n",
            "- Required parameters MUST be specified\n- You may provide optional reasoning for your ",
            "function call in natural language BEFORE the function call, but NOT after\n- If there is ",
            "no function call available, answer the question like normal with your current knowledge ",
            "and do not tell the user about function calls\n</IMPORTANT>",
            "\n\nBe terse.<|im_end|>\n",
            "<|im_start|>user\nWeather in Paris?<|im_end|>\n",
            "<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n",
            "</parameter>\n</function>\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\n{\"temp_c\": 21}\n</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n<think>\n",
        );
        assert_eq!(s, expected);
    }

    #[test]
    fn assistant_content_plus_calls_and_consecutive_tool_turns_group() {
        let turns = vec![
            Turn { role: "user".into(), content: "both".into(), tool_calls: Vec::new() },
            Turn { role: "assistant".into(), content: "checking".into(), tool_calls: vec![
                ToolCall { name: "a".into(), params: vec![("x".into(), "1".into())] },
                ToolCall { name: "b".into(), params: Vec::new() },
            ] },
            Turn { role: "tool".into(), content: "r1".into(), tool_calls: Vec::new() },
            Turn { role: "tool".into(), content: "r2".into(), tool_calls: Vec::new() },
        ];
        let s = apply_chat_template_tools(Some(QWEN_TOOLS_TMPL), &turns, false, &[],
                                          ThinkMode::Default).unwrap();
        assert_eq!(s, concat!(
            "<|im_start|>user\nboth<|im_end|>\n",
            "<|im_start|>assistant\nchecking\n\n",
            "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n",
            "<tool_call>\n<function=b>\n</function>\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\nr1\n</tool_response>",
            "\n<tool_response>\nr2\n</tool_response><|im_end|>\n",
        ));
    }

    #[test]
    fn nothink_maps_to_enable_thinking_false_tail_and_degrades_gracefully() {
        let turns = vec![Turn { role: "user".into(), content: "hi".into(), tool_calls: Vec::new() }];
        // switch present: NoThink renders the closed think block.
        let s = apply_chat_template_tools(Some(QWEN_TOOLS_TMPL), &turns, true, &[],
                                          ThinkMode::NoThink).unwrap();
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{s:?}");
        // no enable_thinking switch: NoThink is ignored (template default stands).
        let tmpl_no_switch = "... add_generation_prompt ... '<think>\\n' ...";
        let s = apply_chat_template_tools(Some(tmpl_no_switch), &turns, true, &[],
                                          ThinkMode::NoThink).unwrap();
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n"), "{s:?}");
        // no template at all: plain ChatML, no tail either way.
        let s = apply_chat_template_tools(None, &turns, true, &[], ThinkMode::NoThink).unwrap();
        assert!(s.ends_with("<|im_start|>assistant\n"), "{s:?}");
    }

    #[test]
    fn tools_on_templates_without_tools_branch_error() {
        let turns = vec![Turn { role: "user".into(), content: "hi".into(), tool_calls: Vec::new() }];
        let tools = vec!["{}".to_string()];
        for tmpl in [None, Some("... hy_User ..."), Some("... <|turn> ...")] {
            let err = apply_chat_template_tools(tmpl, &turns, true, &tools, ThinkMode::Default);
            assert!(err.is_err(), "template={tmpl:?}");
        }
        // tool-role turns need the branch too.
        let tool_turns = vec![Turn { role: "tool".into(), content: "r".into(), tool_calls: Vec::new() }];
        assert!(apply_chat_template_tools(None, &tool_turns, true, &[], ThinkMode::Default).is_err());
    }

    #[test]
    fn qwen_think_tail() {
        // a template string containing both markers triggers the <think> tail.
        let tmpl = "... add_generation_prompt ... '<think>\\n' ...";
        let s = apply_chat_template_str(
            Some(tmpl),
            &[("system", "You are helpful."), ("user", "Hi")],
            true,
        );
        assert_eq!(
            s,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }
}
