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
//!
//! Non-qwen dialects each get their own arm, dispatched by a marker substring in the raw
//! template: Tencent Hy3 (`hy_User`), gemma4 (`<|turn>`), and StepFun Step-3.7-Flash /
//! arch `step35` (`render_message_content`). The step35 check must come BEFORE the qwen
//! `<think>`-tail detection — its template contains every qwen marker, so the qwen arm would
//! render the right generation tail on the wrong turn bodies.

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

/// Thinking control (owner directive 2026-08-07: every supported model is a thinking model,
/// one serve surface maps to each arch's native mechanism).
///
/// - `Default` = the template's OWN default, byte-identical to the pre-surface render:
///   qwen class opens `<think>\n` (thinking ON), gemma4 renders the CLOSED thought channel
///   (its `enable_thinking | default(false)`), hy3 renders `reasoning_effort:no_think`.
/// - `NoThink` = thinking OFF via the arch's native off-switch: qwen
///   `enable_thinking=false` (closed `<think>\n\n</think>\n\n`), gemma4 closed thought
///   channel, hy3 `no_think`. On step35 — whose `<think>` tail is unconditional — it clamps
///   to the lowest effort level instead (`Reasoning: low`).
/// - `Think` = thinking explicitly ON: qwen open `<think>\n` (same bytes as its default),
///   gemma4 `<|think|>\n` injected into the system turn + an OPEN generation turn, hy3
///   an open `<think:opensource>` channel at the requested effort.
///
/// On templates with no switch at all the non-native direction is a graceful no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkMode {
    Default,
    NoThink,
    Think,
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
    // Legacy path = the template's own default ("no_think") — byte-identical to history.
    if template.is_some_and(|t| t.contains("hy_User")) {
        return apply_hy3_template(messages, add_generation_prompt, "no_think");
    }
    // StepFun Step-3.7-Flash (arch `step35`): a ChatML *dialect* — same `<|im_start|>` framing,
    // different everything else (see `apply_step35_template`). Detected by its
    // `render_message_content` macro, which no other committed template defines. This check MUST
    // precede the qwen `<think>`-tail detection below: the step35 template contains both markers,
    // so the qwen arm would produce the right generation tail with the wrong turn bodies.
    if template.is_some_and(|t| t.contains("render_message_content")) {
        let turns: Vec<Turn> = messages.iter().map(|(r, c)| Turn {
            role: r.to_string(), content: c.to_string(), tool_calls: Vec::new(),
        }).collect();
        return apply_step35_template(&turns, add_generation_prompt, &[], None);
    }
    // gemma4: `<|turn>role\n{content}<turn|>\n` dialect; generation prompt appends
    // `<|turn>model\n` + the CLOSED thought channel (`<|channel>thought\n<channel|>` — the
    // template's enable_thinking-false default). bos comes from encode(add_special) — the
    // template's `{{ bos_token }}` is NOT re-emitted here (double-BOS trap).
    // Legacy path = thinking OFF (the template's `default(false)`) — byte-identical to history.
    if template.is_some_and(|t| t.contains("<|turn>")) {
        return apply_gemma4_template(messages, add_generation_prompt, false);
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
///
/// `reasoning_effort` is the step35 dialect's three-level control ("low"/"medium"/"high" —
/// a STRING rendered into the system turn, not a think switch; see `apply_step35_template`).
/// Every other dialect ignores it (their templates have no `reasoning_effort` input), and
/// `None` is the step35 template's own default (no `Reasoning:` line). The server only
/// supplies `Some` for models whose template consumes it (`ModelCaps::effort_levels`), so
/// non-step35 prompts stay byte-identical by construction, not by luck.
pub fn apply_chat_template_tools(
    template: Option<&str>,
    turns: &[Turn],
    add_generation_prompt: bool,
    tools_json: &[String],
    think: ThinkMode,
    reasoning_effort: Option<&str>,
) -> Result<String, String> {
    let has_tool_features = !tools_json.is_empty()
        || turns.iter().any(|t| t.role == "tool" || !t.tool_calls.is_empty());
    let tools_branch = template.is_some_and(|t| t.contains("<tools>"));
    if has_tool_features && !tools_branch {
        return Err("model chat template has no tools branch".into());
    }
    // step35: its own dialect all the way through, tools included (unlike hy3/gemma4, which
    // reject tool features — step35 HAS a tools branch and it is reproduced). Must precede the
    // qwen arm: the step35 template contains `<tools>`, `<think>` and `add_generation_prompt`,
    // so every qwen marker check below matches it. `ThinkMode` is ignored (no `enable_thinking`
    // in this template => `think_switch` is false => NoThink is already a documented no-op);
    // `reasoning_effort` is this dialect's own control and is honored here.
    if template.is_some_and(|t| t.contains("render_message_content")) {
        return Ok(apply_step35_template(turns, add_generation_prompt, tools_json,
                                        reasoning_effort));
    }
    if template.is_some_and(|t| t.contains("hy_User") || t.contains("<|turn>")) {
        // hy3 / gemma4 dialects: no committed tools rendering reference — reject tool
        // features even if the raw jinja happens to mention <tools>. ThinkMode maps to each
        // arch's native mechanism (thinking goldens, render-thinking-goldens.py):
        //   hy3    -> the template's own reasoning_effort input: no_think (its default,
        //             = ThinkMode::Default/NoThink) or low/high (open think, ThinkMode::Think
        //             at the level the caller resolved — effort carries it).
        //   gemma4 -> enable_thinking: default(false) = Default/NoThink;
        //             Think = <|think|> system token + open generation turn.
        if has_tool_features {
            return Err("tools are not supported on this model's chat-template dialect".into());
        }
        let messages: Vec<(&str, &str)> =
            turns.iter().map(|t| (t.role.as_str(), t.content.as_str())).collect();
        if template.is_some_and(|t| t.contains("hy_User")) {
            // hy3's accepted set is exactly no_think|low|high; OpenAI medium clamps to low
            // (the template has no medium level and raises on unknown strings).
            let effort = match (think, reasoning_effort) {
                (ThinkMode::Think, Some("high")) => "high",
                (ThinkMode::Think, _) => "low",
                _ => "no_think",
            };
            return Ok(apply_hy3_template(&messages, add_generation_prompt, effort));
        }
        return Ok(apply_gemma4_template(&messages, add_generation_prompt,
                                        think == ThinkMode::Think));
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

/// The fixed tool-calling instruction block of the StepFun `step35` template. NOT the same
/// string as `QWEN_TOOLS_INSTRUCTION` — three differences, all load-bearing: the header says
/// "in JSONSchema format", the nesting reminder carries literal `\n...\n` inside the
/// `<function=...>` / `<tool_call>` examples, and the Reminder list has 2 bullets instead of 4
/// (no "optional reasoning BEFORE the call" and no "answer normally if no function is
/// available"). Copied byte-for-byte out of the shipped template
/// (`research/step37-bringup-20260802/raw/chat_template.jinja`, == the GGUF's own
/// `tokenizer.chat_template`).
const STEP35_TOOLS_INSTRUCTION: &str = "\n\nIf you choose to call a function ONLY reply in the \
following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n\
This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified \
format: an inner <function=...>\n...\n</function> block must be nested within <tool_call>\n\
...\n</tool_call> XML tags\n- Required parameters MUST be specified\n</IMPORTANT>";

/// StepFun Step-3.7-Flash (GGUF arch `step35`) chat template.
///
/// A ChatML *dialect*, not ChatML: it shares the `<|im_start|>role\n…<|im_end|>\n` frame and
/// nothing else. Reproduced from the shipped jinja, and pinned test-by-test against goldens
/// rendered from that jinja under jinja2 with `trim_blocks`/`lstrip_blocks` — the settings HF
/// transformers and llama.cpp's minja both parse chat templates with
/// (`research/step37-p2-20260806/render_step35_template.py`, goldens committed under `raw/`).
///
/// Where it differs from the qwen3.5/3.6 arms above — every one of these silently corrupts the
/// prompt if the qwen arm is reused:
///
/// | | qwen3.5/3.6 | step35 |
/// |---|---|---|
/// | reasoning level | `enable_thinking` bool | `Reasoning: {low,medium,high}\n\n` prefix inside the system turn |
/// | `<think>` tail | switchable | **unconditional** — no `enable_thinking`, so `ThinkMode::NoThink` is a no-op |
/// | prior assistant turns | content only | turns AFTER the last real user query also carry `<think>\n{reasoning}\n</think>\n` |
/// | tool results | grouped into a `user` turn, `\n<tool_response>\n…\n</tool_response>` | own **`tool_response`** role, `<tool_response>…</tool_response>` with NO inner newlines |
/// | content | `\|trim`med | **not** trimmed |
/// | tools header | `following functions:` | `following functions in JSONSchema format:` |
/// | call separators | `\n\n` after content, `\n` between calls | **none** |
/// | leading system + tools | appended AFTER the instruction block | folded in BEFORE `# Tools` |
///
/// `reasoning_effort` is the model's headline three-level control (low/medium/high per the
/// StepFun model card). It is a parameter here rather than a `ThinkMode`: the value is a
/// *string in the system turn*, so a bool cannot carry it. The serve path supplies it through
/// `apply_chat_template_tools` (worker `Request::reasoning_effort`, mapped from the OpenAI
/// `reasoning_effort` body field when `ModelCaps::effort_levels` is set); `None` — the
/// legacy-str path and every non-step35 model — renders the template's own default
/// (no `Reasoning:` line at all).
///
/// BOS is NOT emitted (the jinja's `{{bos_token}}` is dropped): memra's `encode(add_special)`
/// prepends it from `tokenizer.ggml.add_bos_token`/`bos_token_id` — the same double-BOS trap the
/// gemma4 arm documents.
///
/// ONE deliberate divergence: the jinja's body loop has no `else`, so a role outside
/// {system, user, assistant, tool} renders as **nothing at all** — the turn silently vanishes
/// from the prompt. memra renders it as a generic `<|im_start|>{role}\n{content}<|im_end|>\n`
/// turn instead, matching the other arms here. A dropped turn is the worse failure, and this
/// branch cannot fire on the serve surface: OpenAI roles are exactly system/user/assistant/tool,
/// all four of which are reproduced byte-for-byte.
///
/// Not reproduced (needs data `Turn` does not carry, tracked, cannot fire from an OpenAI client):
/// the `name == "observation"` alias that renames a non-leading `system` turn's role to
/// `observation`, and the `<im_patch>` image-content path (this is a VLM; memra is text-only here).
fn apply_step35_template(turns: &[Turn], add_generation_prompt: bool, tools_json: &[String],
                         reasoning_effort: Option<&str>) -> String {
    let mut out = String::new();
    let leading_system = turns.first().filter(|t| t.role == "system");

    // --- system header. Two branches in the jinja, and the ORDER differs between them.
    if !tools_json.is_empty() {
        out.push_str("<|im_start|>system\n");
        if let Some(effort) = reasoning_effort {
            out.push_str("Reasoning: ");
            out.push_str(effort);
            out.push_str("\n\n");
        }
        if let Some(sys) = leading_system {
            // unconditional `content + '\n\n'` — no emptiness check, unlike the qwen arm.
            out.push_str(&sys.content);
            out.push_str("\n\n");
        }
        out.push_str("# Tools\n\nYou have access to the following functions in JSONSchema \
                      format:\n\n<tools>");
        for tool in tools_json {
            out.push('\n');
            out.push_str(tool);
        }
        out.push_str("\n</tools>");
        out.push_str(STEP35_TOOLS_INSTRUCTION);
        out.push_str("<|im_end|>\n");
    } else if let Some(sys) = leading_system {
        out.push_str("<|im_start|>system\n");
        if let Some(effort) = reasoning_effort {
            out.push_str("Reasoning: ");
            out.push_str(effort);
            out.push_str("\n\n");
        }
        out.push_str(&sys.content);
        out.push_str("<|im_end|>\n");
    } else if let Some(effort) = reasoning_effort {
        out.push_str("<|im_start|>system\nReasoning: ");
        out.push_str(effort);
        out.push_str("\n\n<|im_end|>\n");
    }

    // --- last_query_index: the index of the LAST `user` turn that is a real query, i.e. whose
    // content is not itself a `<tool_response>…</tool_response>` wrapper (a client replaying tool
    // output as a user turn must not reset the reasoning boundary). Default len-1 when there is
    // no such turn, exactly as the jinja's namespace initializer does.
    let last_query_index = turns.iter().enumerate().rev()
        .find(|(_, t)| t.role == "user"
              && !(t.content.starts_with("<tool_response>")
                   && t.content.ends_with("</tool_response>")))
        .map(|(i, _)| i)
        .unwrap_or(turns.len().saturating_sub(1));

    for (i, turn) in turns.iter().enumerate() {
        let content = &turn.content;    // NOT trimmed: this template applies no `|trim`
        match turn.role.as_str() {
            // the leading system turn lives in the header above; later ones are body turns.
            "system" if i == 0 => {}
            "system" | "user" => {
                out.push_str("<|im_start|>");
                out.push_str(&turn.role);
                out.push('\n');
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                // Split an inline `<think>…</think>` out of content, mirroring the jinja's
                // string surgery exactly: reasoning = text before the FIRST `</think>`, with
                // trailing newlines stripped, then everything after the LAST `<think>` in that
                // prefix, with leading newlines stripped; body = after the LAST `</think>`,
                // leading newlines stripped.
                let (reasoning, body): (String, &str) = match content.find("</think>") {
                    Some(first) => {
                        let pre = content[..first].trim_end_matches('\n');
                        let pre = match pre.rfind("<think>") {
                            Some(o) => &pre[o + "<think>".len()..],
                            None => pre,
                        };
                        let last = content.rfind("</think>").unwrap();
                        (pre.trim_start_matches('\n').to_string(),
                         content[last + "</think>".len()..].trim_start_matches('\n'))
                    }
                    None => (String::new(), content.as_str()),
                };
                out.push_str("<|im_start|>assistant\n");
                if i > last_query_index {
                    out.push_str("<think>\n");
                    out.push_str(&reasoning);
                    out.push_str("\n</think>\n");
                }
                out.push_str(body);
                // NO separator before or between calls (the qwen arm's `\n\n`/`\n` would corrupt).
                for call in &turn.tool_calls {
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
                // own role, and consecutive tool turns share ONE `tool_response` turn.
                if i == 0 || turns[i - 1].role != "tool" {
                    out.push_str("<|im_start|>tool_response\n");
                }
                out.push_str("<tool_response>");
                out.push_str(content);
                out.push_str("</tool_response>");
                if i + 1 >= turns.len() || turns[i + 1].role != "tool" {
                    out.push_str("<|im_end|>\n");
                }
            }
            other => {
                // the jinja drops this turn entirely; see the divergence note above.
                out.push_str("<|im_start|>");
                out.push_str(other);
                out.push('\n');
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
        }
    }

    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n<think>\n");
    }
    out
}

/// Text-only reproduction of the Hy3 `chat_template.jinja` (no tools, no `is_training`).
/// `effort` is the template's own `reasoning_effort` input — `"no_think"` / `"low"` /
/// `"high"`, its full accepted set (the jinja `raise_exception`s on anything else; undefined
/// defaults to `'no_think'`, so callers with no opinion pass `"no_think"`):
///   - `{bos}{system…}<｜reasoning_mode:opensource｜>reasoning_effort:{effort}` header
///     (system turns concatenate into the header, before any user turn);
///   - `user`      -> `<｜hy_User:opensource｜>{content}`
///   - `assistant` -> `<｜hy_Assistant:opensource｜><think:opensource></think:opensource>{content}<｜hy_eos:opensource｜>`
///     (non-last turns; history turns render CLOSED think at every effort — the template
///     opens only turns past `last_user_index`, and OpenAI history carries no reasoning);
///   - generation prompt: `<｜hy_Assistant:opensource｜><think:opensource></think:opensource>`
///     at no_think, `…<think:opensource>` (OPEN think) at low/high.
/// Content is NOT trimmed (the Hy3 template applies no `|trim`). Goldens: rendered from the
/// pinned tencent/Hy3 template (sha 7fc351fe…, snapshot 716aa724) by
/// `research/step-sku-20260807/render-thinking-goldens.py`.
fn apply_hy3_template(messages: &[(&str, &str)], add_generation_prompt: bool,
                      effort: &str) -> String {
    const BOS: &str = "<\u{ff5c}hy_begin_of_sentence:opensource\u{ff5c}>";
    const USER: &str = "<\u{ff5c}hy_User:opensource\u{ff5c}>";
    const ASSISTANT: &str = "<\u{ff5c}hy_Assistant:opensource\u{ff5c}>";
    const EOS: &str = "<\u{ff5c}hy_eos:opensource\u{ff5c}>";
    const REASONING: &str = "<\u{ff5c}reasoning_mode:opensource\u{ff5c}>";
    const THINK_BEGIN: &str = "<think:opensource>";
    const THINK_END: &str = "</think:opensource>";

    debug_assert!(matches!(effort, "no_think" | "low" | "high"),
                  "hy3 reasoning_effort must be no_think|low|high, got {effort:?}");
    let mut out = String::from(BOS);
    for (role, content) in messages.iter().filter(|(r, _)| *r == "system") {
        let _ = role;
        out.push_str(content);
    }
    out.push_str(REASONING);
    out.push_str("reasoning_effort:");
    out.push_str(effort);

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
        if effort == "no_think" {
            out.push_str(THINK_END); // low/high leave the think channel OPEN (the golden)
        }
    }
    out
}


/// gemma4 turn dialect (text-only path of the GGUF template, verified against the dumped
/// jinja — sha 36e3a42e…, goldens `research/step-sku-20260807/raw/thinking-goldens.txt`):
/// roles map assistant->model; each turn = `<|turn>{role}\n{content|trim}<turn|>\n`.
///
/// THINKING is `enable_thinking`, and its default is OFF (`enable_thinking | default(false)`)
/// — the inverse of the qwen class:
///   - thinking OFF (default): generation prompt = `<|turn>model\n<|channel>thought\n<channel|>`
///     (the CLOSED thought channel — the model may not think);
///   - thinking ON: a `<|think|>\n` token is injected at the very top of the FIRST system
///     turn (a system turn is CREATED if the request has none), and the generation prompt is
///     the bare `<|turn>model\n` — the thought channel is left to the model.
fn apply_gemma4_template(messages: &[(&str, &str)], add_generation_prompt: bool,
                         thinking: bool) -> String {
    let mut out = String::new();
    let mut msgs = messages;
    // System header block: fires when thinking is on OR a leading system turn exists.
    let leading_system = msgs.first().filter(|(r, _)| *r == "system");
    if thinking || leading_system.is_some() {
        out.push_str("<|turn>system\n");
        if thinking {
            out.push_str("<|think|>\n");
        }
        if let Some((_, content)) = leading_system {
            out.push_str(content.trim());
            msgs = &msgs[1..];
        }
        out.push_str("<turn|>\n");
    }
    for (role, content) in msgs {
        let role = if *role == "assistant" { "model" } else { role };
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(content.trim());
        out.push_str("<turn|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|turn>model\n");
        if !thinking {
            out.push_str("<|channel>thought\n<channel|>");
        }
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
                let ext = apply_chat_template_tools(tmpl, &turns, true, &[], ThinkMode::Default, None)
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
                                          ThinkMode::Default, None).unwrap();
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
                                          ThinkMode::Default, None).unwrap();
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
                                          ThinkMode::NoThink, None).unwrap();
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{s:?}");
        // no enable_thinking switch: NoThink is ignored (template default stands).
        let tmpl_no_switch = "... add_generation_prompt ... '<think>\\n' ...";
        let s = apply_chat_template_tools(Some(tmpl_no_switch), &turns, true, &[],
                                          ThinkMode::NoThink, None).unwrap();
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n"), "{s:?}");
        // no template at all: plain ChatML, no tail either way.
        let s = apply_chat_template_tools(None, &turns, true, &[], ThinkMode::NoThink, None).unwrap();
        assert!(s.ends_with("<|im_start|>assistant\n"), "{s:?}");
    }

    #[test]
    fn tools_on_templates_without_tools_branch_error() {
        let turns = vec![Turn { role: "user".into(), content: "hi".into(), tool_calls: Vec::new() }];
        let tools = vec!["{}".to_string()];
        for tmpl in [None, Some("... hy_User ..."), Some("... <|turn> ...")] {
            let err = apply_chat_template_tools(tmpl, &turns, true, &tools, ThinkMode::Default, None);
            assert!(err.is_err(), "template={tmpl:?}");
        }
        // tool-role turns need the branch too.
        let tool_turns = vec![Turn { role: "tool".into(), content: "r".into(), tool_calls: Vec::new() }];
        assert!(apply_chat_template_tools(None, &tool_turns, true, &[], ThinkMode::Default, None).is_err());
    }

    // ---- per-arch thinking control (owner directive 2026-08-07) -------------------------
    // Every `expected` below is the EXACT string the arch's REAL shipped template renders,
    // from research/step-sku-20260807/raw/thinking-goldens.txt (render-thinking-goldens.py:
    // jinja2 trim_blocks/lstrip_blocks over the pinned template dumps — gemma4 sha 36e3a42e
    // from the local QAT GGUF header, hy3 sha 7fc351fe from the pinned tencent/Hy3 snapshot).

    fn one_user() -> Vec<Turn> {
        vec![turn("user", "Hi")]
    }

    #[test]
    fn gemma4_thinking_maps_to_the_think_token_and_open_turn() {
        let g = |think: ThinkMode| {
            apply_chat_template_tools(Some("... <|turn> ..."), &one_user(), true, &[], think,
                                      None).unwrap()
        };
        // Default AND NoThink = the template's own default(false): closed thought channel.
        // Byte-identical to the legacy renderer (no silent behavior change).
        let closed = "<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
        assert_eq!(g(ThinkMode::Default), closed);
        assert_eq!(g(ThinkMode::NoThink), closed);
        assert_eq!(apply_chat_template_str(Some("... <|turn> ..."), &[("user", "Hi")], true),
                   closed, "legacy renderer = the default arm");
        // Think = enable_thinking=true: <|think|> injected into a CREATED system turn and
        // the generation turn left open (golden: gemma4 enable_thinking=true, no system).
        assert_eq!(g(ThinkMode::Think),
                   "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n");
        // with a client system turn the token lands at the very top of it (golden).
        let turns = vec![turn("system", "Be terse."), turn("user", "Hi")];
        let s = apply_chat_template_tools(Some("... <|turn> ..."), &turns, true, &[],
                                          ThinkMode::Think, None).unwrap();
        assert_eq!(s, "<|turn>system\n<|think|>\nBe terse.<turn|>\n\
                       <|turn>user\nHi<turn|>\n<|turn>model\n");
    }

    #[test]
    fn hy3_thinking_maps_to_its_reasoning_effort_levels() {
        const HY_TMPL: Option<&str> = Some("... hy_User ...");
        let h = |think: ThinkMode, effort: Option<&str>| {
            apply_chat_template_tools(HY_TMPL, &one_user(), true, &[], think, effort).unwrap()
        };
        // Default AND NoThink = the template's own default: no_think header + CLOSED think.
        // Byte-identical to the legacy renderer.
        let closed = "<\u{ff5c}hy_begin_of_sentence:opensource\u{ff5c}>\
                      <\u{ff5c}reasoning_mode:opensource\u{ff5c}>reasoning_effort:no_think\
                      <\u{ff5c}hy_User:opensource\u{ff5c}>Hi\
                      <\u{ff5c}hy_Assistant:opensource\u{ff5c}>\
                      <think:opensource></think:opensource>";
        assert_eq!(h(ThinkMode::Default, None), closed);
        assert_eq!(h(ThinkMode::NoThink, Some("low")), closed,
                   "NoThink wins over a level: thinking off IS no_think");
        assert_eq!(apply_chat_template_str(HY_TMPL, &[("user", "Hi")], true), closed,
                   "legacy renderer = the default arm");
        // Think at low/high = the template's own open-think levels (goldens: header carries
        // the level, generation prompt ends with an OPEN <think:opensource>).
        let low = h(ThinkMode::Think, Some("low"));
        assert!(low.contains("reasoning_effort:low"), "{low:?}");
        assert!(low.ends_with("<think:opensource>"), "{low:?}");
        let high = h(ThinkMode::Think, Some("high"));
        assert!(high.contains("reasoning_effort:high"), "{high:?}");
        assert!(high.ends_with("<think:opensource>"), "{high:?}");
        // medium clamps to low (hy3's accepted set is exactly no_think|low|high — the jinja
        // raise_exceptions on anything else); Think with no level also lands at low.
        assert_eq!(h(ThinkMode::Think, Some("medium")), low);
        assert_eq!(h(ThinkMode::Think, None), low);
        // History assistant turns stay CLOSED-think at every effort (the template opens only
        // turns past last_user_index; golden: "hy3 assistant history stays closed-think").
        let turns = vec![turn("user", "q"), turn("assistant", "a"), turn("user", "more")];
        let s = apply_chat_template_tools(HY_TMPL, &turns, true, &[], ThinkMode::Think,
                                          Some("low")).unwrap();
        assert_eq!(s, "<\u{ff5c}hy_begin_of_sentence:opensource\u{ff5c}>\
                       <\u{ff5c}reasoning_mode:opensource\u{ff5c}>reasoning_effort:low\
                       <\u{ff5c}hy_User:opensource\u{ff5c}>q\
                       <\u{ff5c}hy_Assistant:opensource\u{ff5c}>\
                       <think:opensource></think:opensource>a\
                       <\u{ff5c}hy_eos:opensource\u{ff5c}>\
                       <\u{ff5c}hy_User:opensource\u{ff5c}>more\
                       <\u{ff5c}hy_Assistant:opensource\u{ff5c}><think:opensource>");
    }

    #[test]
    fn qwen_think_mode_covers_all_three_directions() {
        let q = |think: ThinkMode| {
            apply_chat_template_tools(Some(QWEN_TOOLS_TMPL), &one_user(), true, &[], think,
                                      None).unwrap()
        };
        // qwen's template default IS thinking-on, so Default and Think render identically.
        assert!(q(ThinkMode::Default).ends_with("<|im_start|>assistant\n<think>\n"));
        assert_eq!(q(ThinkMode::Think), q(ThinkMode::Default));
        assert!(q(ThinkMode::NoThink)
            .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    // ---- StepFun Step-3.7-Flash (arch step35) -------------------------------------------
    // Every `expected` below is the EXACT string the shipped jinja renders, taken from
    // research/step37-p2-20260806/raw/step35-template-goldens.txt (generated by
    // render_step35_template.py under jinja2 with trim_blocks/lstrip_blocks — the settings HF
    // transformers and llama.cpp's minja use). `{{bos_token}}` renders as "" there because
    // encode(add_special) supplies BOS.

    /// A step35 template stand-in: the real one is 5723 chars, and the detector keys on
    /// `render_message_content` (the macro no other committed template defines). The other
    /// markers are present to prove the step35 arm WINS the dispatch — a qwen-marker template
    /// carrying `<tools>`/`<think>`/`add_generation_prompt` would otherwise take the qwen arm.
    const STEP35_TMPL: &str =
        "{% macro render_message_content(message) %}... <tools> ... add_generation_prompt ... '<think>\\n' ...";

    fn s35(msgs: &[(&str, &str)], genp: bool) -> String {
        apply_chat_template_str(Some(STEP35_TMPL), msgs, genp)
    }

    fn s35_turns(turns: Vec<Turn>, genp: bool, tools: &[String]) -> String {
        apply_chat_template_tools(Some(STEP35_TMPL), &turns, genp, tools, ThinkMode::Default, None)
            .unwrap()
    }

    fn turn(role: &str, content: &str) -> Turn {
        Turn { role: role.into(), content: content.into(), tool_calls: Vec::new() }
    }

    #[test]
    fn step35_plain_paths_match_the_shipped_jinja() {
        assert_eq!(s35(&[("user", "Hello")], true),
                   "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n");
        assert_eq!(s35(&[("user", "Hello")], false), "<|im_start|>user\nHello<|im_end|>\n");
        assert_eq!(s35(&[("system", "You are helpful."), ("user", "Hi")], true),
                   "<|im_start|>system\nYou are helpful.<|im_end|>\n\
                    <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");
        // multi-turn: the prior assistant is BEFORE the last user query, so it carries NO
        // think block — the reasoning boundary the qwen arms have no concept of.
        assert_eq!(
            s35(&[("system", "rules"), ("user", "task"), ("assistant", "work"),
                  ("user", "more")], true),
            "<|im_start|>system\nrules<|im_end|>\n<|im_start|>user\ntask<|im_end|>\n\
             <|im_start|>assistant\nwork<|im_end|>\n<|im_start|>user\nmore<|im_end|>\n\
             <|im_start|>assistant\n<think>\n");
        // content is NOT trimmed (this template applies no `|trim`) — the qwen arms trim.
        assert_eq!(s35(&[("user", "  padded  ")], true),
                   "<|im_start|>user\n  padded  <|im_end|>\n<|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn step35_dispatch_beats_the_qwen_marker_arm() {
        // The step35 template carries every qwen marker. If the dispatch order regressed, the
        // think tail would still be right and the BODY would be wrong (trimmed content, wrong
        // tools header) — so assert a body-shaped difference, not the tail.
        let qwen = apply_chat_template_str(Some(QWEN_TOOLS_TMPL), &[("user", " pad ")], true);
        let step = s35(&[("user", " pad ")], true);
        assert_eq!(qwen, "<|im_start|>user\npad<|im_end|>\n<|im_start|>assistant\n<think>\n");
        assert_eq!(step, "<|im_start|>user\n pad <|im_end|>\n<|im_start|>assistant\n<think>\n");
        assert_ne!(qwen, step);
    }

    #[test]
    fn step35_reasoning_effort_renders_in_the_system_turn() {
        assert_eq!(
            apply_step35_template(&[turn("user", "Hi")], true, &[], Some("high")),
            "<|im_start|>system\nReasoning: high\n\n<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");
        assert_eq!(
            apply_step35_template(&[turn("system", "Be terse."), turn("user", "Hi")], true, &[],
                                  Some("low")),
            "<|im_start|>system\nReasoning: low\n\nBe terse.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");
        // with tools the order flips: Reasoning, then the system content, then `# Tools`.
        let tools = vec![r#"{"type": "function", "function": {"name": "f"}}"#.to_string()];
        let s = apply_step35_template(&[turn("system", "Be terse."), turn("user", "q")], true,
                                      &tools, Some("medium"));
        assert!(s.starts_with("<|im_start|>system\nReasoning: medium\n\nBe terse.\n\n# Tools\n"),
                "{s:?}");
    }

    #[test]
    fn reasoning_effort_reaches_step35_through_the_public_entry_and_only_step35() {
        // The serve path enters via apply_chat_template_tools: the level must land in the
        // rendered system turn on the step35 dialect...
        let turns = vec![turn("user", "Hi")];
        let s = apply_chat_template_tools(Some(STEP35_TMPL), &turns, true, &[],
                                          ThinkMode::Default, Some("high")).unwrap();
        assert!(s.starts_with("<|im_start|>system\nReasoning: high\n\n<|im_end|>\n"), "{s:?}");
        // ...None keeps the template's own default (no Reasoning: line at all)...
        let s = apply_chat_template_tools(Some(STEP35_TMPL), &turns, true, &[],
                                          ThinkMode::Default, None).unwrap();
        assert!(!s.contains("Reasoning:"), "{s:?}");
        // ...and every non-step35 dialect ignores the parameter (their templates have no
        // reasoning_effort input) — byte-identical with and without it.
        for tmpl in [None, Some(QWEN_TOOLS_TMPL), Some("... hy_User ..."), Some("... <|turn> ...")] {
            let with = apply_chat_template_tools(tmpl, &turns, true, &[],
                                                 ThinkMode::Default, Some("high")).unwrap();
            let without = apply_chat_template_tools(tmpl, &turns, true, &[],
                                                    ThinkMode::Default, None).unwrap();
            assert_eq!(with, without, "template={tmpl:?}");
        }
    }

    #[test]
    fn step35_tools_header_is_not_the_qwen_header() {
        let tools = vec![
            r#"{"type": "function", "function": {"name": "get_weather"}}"#.to_string(),
            r#"{"type": "function", "function": {"name": "search"}}"#.to_string(),
        ];
        let s = s35_turns(vec![turn("system", "Be terse."), turn("user", "Weather in Paris?")],
                          true, &tools);
        assert_eq!(s, concat!(
            // leading system folds in BEFORE `# Tools` (the qwen arm appends it AFTER the
            // instruction block), and the header says "in JSONSchema format".
            "<|im_start|>system\nBe terse.\n\n# Tools\n\n",
            "You have access to the following functions in JSONSchema format:\n\n<tools>\n",
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\"}}\n",
            "{\"type\": \"function\", \"function\": {\"name\": \"search\"}}\n</tools>",
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:",
            "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n",
            "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the ",
            "second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>",
            // the nesting reminder carries literal \n...\n INSIDE the example tags, and the
            // Reminder list stops after 2 bullets (the qwen block has 4).
            "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner ",
            "<function=...>\n...\n</function> block must be nested within <tool_call>\n...\n",
            "</tool_call> XML tags\n- Required parameters MUST be specified\n</IMPORTANT>",
            "<|im_end|>\n",
            "<|im_start|>user\nWeather in Paris?<|im_end|>\n",
            "<|im_start|>assistant\n<think>\n",
        ));
        // and it is NOT the qwen instruction block.
        assert!(!s.contains(QWEN_TOOLS_INSTRUCTION));
    }

    #[test]
    fn step35_tool_results_take_their_own_role_and_group() {
        let tools = vec![r#"{"type": "function", "function": {"name": "get_weather"}}"#.to_string()];
        let turns = vec![
            turn("user", "both"),
            Turn { role: "assistant".into(), content: "checking".into(), tool_calls: vec![
                ToolCall { name: "a".into(), params: vec![("x".into(), "1".into())] },
                ToolCall { name: "b".into(), params: Vec::new() },
            ] },
            turn("tool", "r1"),
            turn("tool", "r2"),
        ];
        let s = s35_turns(turns, true, &tools);
        let body = s.split("<|im_end|>\n").skip(1).collect::<Vec<_>>().join("<|im_end|>\n");
        assert_eq!(body, concat!(
            "<|im_start|>user\nboth<|im_end|>\n",
            // the assistant is AFTER the last user query, so it carries a think block — empty,
            // because its content has no `</think>` marker.
            "<|im_start|>assistant\n<think>\n\n</think>\nchecking",
            // NO separator before the first call and NONE between calls.
            "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=b>\n</function>\n</tool_call><|im_end|>\n",
            // own `tool_response` ROLE (not a user turn), and NO newlines inside the wrappers.
            "<|im_start|>tool_response\n<tool_response>r1</tool_response>",
            "<tool_response>r2</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n<think>\n",
        ));
    }

    #[test]
    fn step35_assistant_think_split_and_the_reasoning_boundary() {
        // inline <think>…</think> in content splits into the reasoning block + body.
        assert_eq!(
            s35(&[("user", "q"), ("assistant", "<think>\nreasoned\n</think>\nanswer")], false),
            "<|im_start|>user\nq<|im_end|>\n\
             <|im_start|>assistant\n<think>\nreasoned\n</think>\nanswer<|im_end|>\n");
        // no markers, but still after the last query -> an EMPTY reasoning block is emitted.
        assert_eq!(
            s35(&[("user", "q"), ("assistant", "plain")], false),
            "<|im_start|>user\nq<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\nplain<|im_end|>\n");
        // a user turn that IS a <tool_response> wrapper does NOT move the boundary: the
        // assistant before it still counts as after-the-last-real-query.
        assert_eq!(
            s35(&[("user", "real question"), ("assistant", "thinking about it"),
                  ("user", "<tool_response>r</tool_response>")], true),
            "<|im_start|>user\nreal question<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\nthinking about it<|im_end|>\n\
             <|im_start|>user\n<tool_response>r</tool_response><|im_end|>\n\
             <|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn step35_think_tail_is_unconditional_and_nothink_is_a_noop() {
        // No `enable_thinking` in this template, so ThinkMode::NoThink cannot close the tail —
        // the same graceful-no-op contract the other switchless templates get. A NoThink that
        // silently emitted `<think>\n\n</think>\n\n` would be a prompt the model never saw.
        let turns = vec![turn("user", "hi")];
        for mode in [ThinkMode::Default, ThinkMode::NoThink] {
            let s = apply_chat_template_tools(Some(STEP35_TMPL), &turns, true, &[], mode, None).unwrap();
            assert!(s.ends_with("<|im_start|>assistant\n<think>\n"), "mode={mode:?} {s:?}");
        }
    }

    #[test]
    fn step35_plain_path_is_identical_through_both_renderers() {
        // same isolation contract the qwen arms hold: a plain request renders byte-identically
        // whether it enters via apply_chat_template_str or apply_chat_template_tools.
        let batteries: &[&[(&str, &str)]] = &[
            &[("user", "Hello")],
            &[("system", "You are helpful."), ("user", "Hi")],
            &[("system", "rules"), ("user", "task"), ("assistant", "work"), ("user", "more")],
            &[("user", "  padded  "), ("assistant", "reply\nwith lines")],
        ];
        for msgs in batteries {
            let legacy = s35(msgs, true);
            let ext = s35_turns(msgs.iter().map(|(r, c)| turn(r, c)).collect(), true, &[]);
            assert_eq!(legacy, ext, "msgs={msgs:?}");
        }
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
