#!/usr/bin/env python3
"""Build the three cell prompts (owner's daily usage shapes), Qwen chat template
rendered CLIENT-side — the exact pi contract both daily providers use
(api: openai-completions, thinkingFormat qwen-chat-template, raw /v1/completions).

Cells:
  a) short-agentic — the owner's tool-check shape (~100-200 tok outputs)
  b) long-gen — 512-token generation
  c) ctx4k — ~4k-token context continuation (mid-session agentic shape)

NOT board material: dogfood-experience diagnostic (owner-asked, 2026-08-05).
"""
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "prompts")
os.makedirs(OUT, exist_ok=True)

def qwen(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")

SYS_AGENT = (
    "You are pi, a coding agent running on the owner's workstation. You have these tools:\n"
    "- bash(command: string): run a shell command, returns stdout+stderr\n"
    "- read(path: string): read a file\n"
    "- write(path: string, content: string): write a file\n"
    "Respond with a tool call in the form <tool_call>{\"name\": ..., \"arguments\": {...}}"
    "</tool_call> when a tool is needed, otherwise answer directly and concisely."
)

# a) short-agentic: the tool-check shape
short_agentic = qwen(
    SYS_AGENT,
    "Check whether /etc/hosts contains an entry for the host 'gpu-rig', and if the file "
    "is readable also report how many total entries it has. Use the bash tool.")

# b) long-gen: 512-token essay-class generation
long_gen = qwen(
    "You are a careful technical writer. Write precisely and without filler.",
    "Explain how speculative decoding with an MTP draft head works in an LLM serving "
    "stack: the draft/verify loop, acceptance criteria under sampled (non-greedy) "
    "decoding, why acceptance drops as temperature rises, and what determines the "
    "end-to-end speedup. Cover KV-cache handling for rejected tokens. Be thorough.")

# c) ctx4k: ~4k-token document continuation (deterministic, varied filler so the
# continuation doesn't degenerate)
topics = [
    ("storage staging", "NVMe reads land in pinned host buffers sized to the transfer window"),
    ("PCIe overlap", "H2D copies run on a dedicated stream so kernels never wait on the bus"),
    ("KV quantization", "q8_0 keys survive rotation while q5_1 values hold retrieval quality"),
    ("expert residency", "hot experts stay device-resident while cold ones spill to host"),
    ("draft acceptance", "acceptance decays with depth so the ladder trims tail positions"),
    ("graph capture", "decode graphs amortize launch overhead across steady-state steps"),
    ("prefill chunking", "chunk size trades transient VRAM against per-chunk launch cost"),
    ("session parking", "parked caches resume continuations without re-priming the past"),
]
paras = []
for i in range(44):
    t, s = topics[i % len(topics)]
    paras.append(
        f"Section {i}: {t}. {s}. Measurement note {i}: the run recorded its thermal "
        f"regime and its N, and the median moved by less than the run-to-run spread. "
        f"The follow-up sweep varied one knob at a time and kept the raw log next to "
        f"the summary row so the claim stays auditable. ")
doc = "".join(paras)
ctx4k = qwen(
    SYS_AGENT,
    "Here is the engineering log you were analyzing:\n\n" + doc +
    "\n\nContinue the analysis: summarize the three most load-bearing mechanisms in "
    "this log and state which single knob you would sweep next and why.")

# warmup (excluded from stats)
warmup = qwen("You are a helpful assistant.", "Say OK.")

for name, text in [("short-agentic", short_agentic), ("long-gen", long_gen),
                   ("ctx4k", ctx4k), ("warmup", warmup)]:
    with open(os.path.join(OUT, f"{name}.txt"), "w") as f:
        f.write(text)
    print(name, len(text), "chars (~", len(text)//4, "tok)")
