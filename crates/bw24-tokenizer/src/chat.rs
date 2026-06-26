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
