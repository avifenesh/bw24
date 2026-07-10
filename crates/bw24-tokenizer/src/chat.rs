//! Minimal chat-template renderer for the Qwen3.5 / ChatML format.
//!
//! The model's GGUF `tokenizer.chat_template` is a large jinja template covering
//! tools, vision, and multi-step reasoning. We do NOT ship a jinja engine; instead
//! we reproduce the text-only system/user/assistant path of that template exactly,
//! which is the path bw24's text-in/text-out CLI uses. The reproduced behavior
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
