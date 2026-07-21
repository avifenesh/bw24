# Learning Guide: README Best Practices for Technical Inference Repositories

**Generated**: 2026-07-16  
**Sources**: 20 resources analyzed  
**Depth**: medium  

## Prerequisites

Before writing a README for an LLM inference engine or performance-focused systems project, you should understand:

- Your target audience (researchers, production engineers, hobbyists, or all three)
- The performance characteristics and hardware requirements of your system
- How your project relates to alternatives in the ecosystem
- What constitutes a reproducible performance claim in your domain
- Version control and documentation-as-code workflows

**Required tools/knowledge:**
- Markdown syntax
- Basic understanding of your project's build system
- Performance measurement methodology
- Your project's licensing and legal constraints

## TL;DR

**For technical inference repositories, your README must:**

1. **Lead with clarity** — one-line value proposition, then immediate action (install/run)
2. **Show, don't just tell** — runnable examples with expected output before prose
3. **Anchor performance claims** — name specific hardware, cite methodology, link full reports
4. **Route by audience** — quickstart for users, build-from-source for contributors, architecture docs external
5. **Disclose honestly** — limitations inline, compatibility matrices explicit, breaking changes surfaced
6. **Delegate depth** — README as navigation hub, not comprehensive manual
7. **Maintain trust** — version-pin dependencies, validate examples, update benchmarks or remove them

## Core Concepts Overview

Seven patterns govern inference-repo READMEs; each has its own section below.

## The README as a Router, Not a Manual

Modern high-quality READMEs (llama.cpp, vLLM, PyTorch, Rust) follow a **hub-and-spoke pattern**: the README provides just enough information for readers to self-qualify and take the next step, then delegates to purpose-built documents.

**Why this matters for inference engines:**
- Your audience spans researchers (want papers/theory), production engineers (want deployment guides), and hobbyists (want quickstarts)
- Build complexity varies wildly by platform (CUDA versions, ROCm, CPU-only, Metal, etc.)
- Performance numbers go stale faster than code

**Implementation pattern from llama.cpp:**
```markdown
## Quick start
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

For detailed build instructions, see docs/build.md
For server deployment, see tools/server/README.md
For supported hardware, see docs/backends.md
```

Notice: one runnable command inline, everything else linked.

## Performance Claims Must Be Falsifiable

**The credibility hierarchy** (from most to least trusted):

| Claim Type | Example | Trust Level | Why |
|------------|---------|-------------|-----|
| **Reproducible command output** | llama.cpp shows literal `llama-bench` output with build hash, uncertainty bars | ⭐⭐⭐⭐⭐ | Reader can verify |
| **Linked versioned report** | mistral.rs links `releases/v0.8.2/report.md` with commands, model IDs, host metadata | ⭐⭐⭐⭐ | Auditable trail |
| **Named mechanisms** | vLLM: "PagedAttention" + arXiv link | ⭐⭐⭐ | Verifiable via paper |
| **Quantified with hardware** | InfluxDB: "<10ms for last-value queries on X hardware" | ⭐⭐ | Specific but not reproducible |
| **Vague claims** | "Blazing fast", "10x faster" (no baseline) | ⭐ | Marketing, not evidence |

**mistral.rs benchmark pattern** (exemplary):
```markdown
## Benchmarks

Mean tokens/sec across prompt lengths 128-16384, decode depths 128-16384.

| Model | Hardware | Format | Prefill | Decode |
|-------|----------|--------|---------|--------|
| Qwen3.6-27B | GB10 | UQFF Q8 | 14,520 | 182 |

Full methodology: releases/v0.8.2/report.md
Commands, model revisions, host metadata in appendix.
```

**What makes this work:**
- States the workload clearly ("prompt lengths 128-16384")
- Names competitors in like-for-like formats ("vs llama.cpp GGUF Q8_0")
- Links a full reproducible report
- Includes unfavorable numbers (honesty signal)

## Hardware Disclosure as a Design Constraint

For inference engines, **hardware compatibility isn't a nice-to-have—it's core scope**. Three disclosure patterns observed:

#### 1. Support Matrix (bitsandbytes, TensorRT-LLM)
```markdown
| OS | Arch | Accelerator | Status | Notes |
|----|------|-------------|--------|-------|
| Linux | x86_64 | NVIDIA (SM75+) | ✅ | Recommended |
| Linux | x86_64 | AMD gfx90a | 🟡 | Partial, no 8-bit optimizers |
| macOS | arm64 | Metal | 🐢 | Slow, in progress |
```

**Pros:** Comprehensive, scannable  
**Cons:** Maintenance burden (must update with every release)

#### 2. Tiered Prose (Triton, Candle)
```markdown
**Supported:**
- Linux: NVIDIA GPUs (Compute Capability 8.0+), AMD GPUs (ROCm 6.2+)
- macOS: Apple Silicon (experimental)

**In Development:**
- CPU (interpreter only, no performance tuning)
```

**Pros:** Low maintenance, easy to read  
**Cons:** Harder to scan, imprecise boundaries

#### 3. Implicit via Tooling (Meta Llama, Nerfstudio)
```markdown
## Prerequisites
- NVIDIA GPU
- CUDA 11.8 (tested) or 11.7
- conda environment with PyTorch
```

**Pros:** Minimal text  
**Cons:** Doesn't disclose tier-2 platforms, no negative space

**For single-target repos** (e.g., "RTX 5090 optimized CUDA kernels"), state the constraint **in the title** or first sentence: "bw24: LLM inference for sm_120 (RTX 50-series)". Don't make readers guess scope.

## Scoping and Limitations: The Trust Moat

Projects with **explicit limitations disclosed inline** are trusted more than those claiming universal capability. Pattern observed across high-quality READMEs:

**Where to surface limitations:**

1. **Inline with the feature** (ExLlamaV2):
   ```markdown
   - Dynamic batching via Flash Attention 2.5.7+
   - FP8 cache not yet supported in dynamic mode; use Q4 cache instead
   ```

2. **Support matrix negatives** (bitsandbytes):
   - macOS Metal: 🐢 slow
   - Intel Gaudi: QLoRA only, no 8-bit optimizers

3. **Explicit scope boundaries** (llama.cpp):
   ```markdown
   ## What llama.cpp is NOT
   - Not a training framework (inference only)
   - Not a hosted service (local execution)
   - Not a model hub (see ggml.ai for models)
   ```

4. **Version/compatibility tables** (InfluxDB):
   ```markdown
   | Version | Branch | Query Languages | Status |
   |---------|--------|----------------|---------|
   | 3.x | main | SQL, InfluxQL, Flight SQL | Active |
   | 2.x | main-2.x | Flux, InfluxQL | Maintenance |
   | 1.x | master-1.x | InfluxQL | Legacy |
   ```

**Anti-pattern:** Hiding limitations in a GitHub Issues label or Discord FAQ. If it's a known constraint, surface it in docs.

## Quickstart vs. Deep Docs: The Two-Tier Split

**Golden rule from Art of README:** "The ideal README is as short as it can be without being any shorter."

**The split pattern** (observed in llama.cpp, vLLM, PyTorch, Rust):

| README Contains | External Docs Contain |
|-----------------|----------------------|
| One command install (binary/pip) | Source build instructions |
| One example with expected output | Full API reference |
| Supported model families (list) | Per-model configuration |
| Quickstart for common case | Edge cases, multi-GPU, Docker |
| Links to paper/blog | Full methodology, related work |

**Example: vLLM quickstart**
```markdown
## Installation
uv pip install vllm

## Quick Start
See: docs.vllm.ai/quickstart

## Supported Models
200+ architectures on Hugging Face. See: docs.vllm.ai/models
```

Three external links, zero inline detail. The README stays scannable forever.

**Example: PyTorch complexity management**
- README: "Visit pytorch.org for binary selector (choose OS/CUDA)"
- Source builds: Nested TOC in README, full prerequisites per platform, but prominently framed as "advanced" with time estimates ("30-60 minutes initial build")

**For inference engines specifically:**
- **Inference quickstart in README** — most users want to run, not build
- **Build instructions external** — platform matrix explosion
- **Performance tuning external** — too workload-specific
- **Model support external** — changes frequently

## Badges, Perf Cards, and Generated Content

**Badges observed in top repos:**

| Badge Type | Purpose | Example Projects |
|------------|---------|------------------|
| **Build status** | CI health signal | llama.cpp, PyTorch, Triton |
| **Version** | Release cadence | vLLM, mistral.rs, Candle |
| **License** | Legal clarity | All reviewed projects |
| **Docs status** | Doc freshness | Nerfstudio, TensorRT-LLM |
| **Downloads/PyPI** | Adoption signal | vLLM, bitsandbytes |
| **Test coverage** | Code quality | (Less common in systems projects) |

**Anti-patterns to avoid:**
- 15+ badges making the README look like a racecar (Standard Readme: "too many badges can reduce readability")
- Unmaintained badges (broken links, stale CI)
- Vanity metrics (GitHub stars) unless community size is a selling point

**Generated content patterns:**

1. **Model support lists** (mistral.rs): "This list is generated from the loader registry — it never drifts"
2. **Benchmark tables** (llama.cpp): Literal tool output with build hash
3. **Performance cards** (TensorRT-LLM): Link to auto-updated charts rather than static images
4. **Contributor avatars** (common in "Awesome README" examples): Helps community feel seen

**Trade-off:** Generated content stays accurate but adds CI complexity. Only automate if staleness would erode trust (model lists, benchmarks yes; contributor names no).

## Correctness and Evidence Discipline

The **Linux kernel submission checklist** sets the bar for systems-level evidence standards. Applicable patterns for inference engines:

**Documentation requirements:**
- Every user-facing config option needs help text
- Memory barriers and non-obvious optimizations require inline rationale comments
- Module parameters self-document via docstrings
- Performance claims must survive multi-config builds

**Testing requirements adapted for ML inference:**
- Benchmark reproducibility: publish commands + hardware + versions
- Numerical correctness: test against reference implementations
- Multi-backend testing: CUDA/ROCm/Metal/CPU variants
- Stress testing: long contexts, large batches, OOM scenarios

**Example: Redis build validation**
- Every platform's build instructions tested against named Docker images ("ubuntu:24.04")
- Version pins with rationale ("LLVM 21 required for cross-language LTO")
- Post-build smoke test: `redis-cli` session with expected output
- Automated tests: `make test`

**For inference repos:**
```markdown
## Verification

After building, run:
./scripts/validate.sh

Expected output:
  ✓ CUDA detected: 12.1
  ✓ NCCL multi-GPU: enabled
  ✓ Smoke test (Llama-2-7B): 83.4 tok/s ±2%
```

**Anti-pattern:** "Trust me, it's fast." No measurements, no reproducibility. Even a simple script that runs inference and reports tok/s builds trust.

## Code Examples

### Example 1: Minimal Inference Engine README

```markdown
# fastinfer — CPU-optimized LLM inference

Minimal LLM inference engine for x86_64 CPUs with AVX512. Single-header C++.

## Quick Start

git clone https://github.com/you/fastinfer
cd fastinfer
./scripts/download_model.sh llama-3-8b
./fastinfer --model llama-3-8b.gguf --prompt "Hello"

Expected output:
> Hello! How can I help you today? [42 tok/s, AVX512-BF16]

## Performance

Measured on Intel Xeon Ice Lake, Llama-3-8B Q8_0:
- Prefill: 1,240 tok/s
- Decode: 42 tok/s

Methodology: releases/v1.0/bench-report.md

## Supported Hardware

Required:
- x86_64 CPU with AVX512 (Ice Lake or later)
- 16GB RAM

Not supported:
- GPUs (use llama.cpp for CUDA)
- ARM (use llama.cpp for Apple Silicon)

## Installation

Binary: ./scripts/install.sh
Source: See BUILDING.md

## Documentation

- API Reference: docs/api.md
- Tuning Guide: docs/tuning.md
- Architecture: docs/design.md

## License

MIT. See LICENSE file.
```

**What makes this work:**
- Scope in subtitle ("CPU-optimized")
- Runnable example with expected output
- Hardware requirements explicit (required + not supported)
- Performance numbers with hardware + methodology link
- Everything deep lives in external docs

### Example 2: Performance Claim Table (mistral.rs style)

```markdown
## Benchmarks

All benchmarks: mean tok/s across prompt lengths 128–16384 and decode depths 128–16384.

### Q8 Quantization vs llama.cpp GGUF Q8_0

| Model | Hardware | mistral.rs | llama.cpp | Delta |
|-------|----------|------------|-----------|-------|
| Llama-3-8B | B200 (1x) | 14,520 | 14,105 | +2.9% |
| Llama-3-70B | H100 SXM (8x) | 9,834 | 9,621 | +2.2% |

### BF16 vs vLLM BF16

| Model | Hardware | mistral.rs | vLLM | Delta |
|-------|----------|------------|------|-------|
| Llama-3-8B | H100 | 3,241 | 3,580 | -9.5% |

Full report with commands, model IDs, and hyperparameters:  
releases/v1.2.0/benchmark-report.md

### Test Environment
- mistral.rs: v1.2.0 (commit abc1234)
- llama.cpp: b1234 (2024-12-15)
- vLLM: v0.8.1
- Driver: 560.28.03, CUDA 12.4
```

**What makes this credible:**
- Explicit workload definition
- Named competitors with versions
- Includes unfavorable results (vLLM BF16 loss)
- Links full report for reproducibility
- Test environment disclosed

### Example 3: Hardware Support Matrix

```markdown
## Supported Hardware

| Platform | Architecture | Accelerator | Status | Notes |
|----------|--------------|-------------|--------|-------|
| Linux | x86_64 | NVIDIA (SM 8.0+) | ✅ Fully Supported | Recommended for production |
| Linux | x86_64 | AMD MI250/MI300 | ✅ Fully Supported | ROCm 6.0+ required |
| Linux | x86_64 | Intel GPU | 🟡 Experimental | SYCL backend, beta quality |
| Linux | x86_64 | CPU (AVX512) | ✅ Supported | BF16 compute, see CPU.md |
| Linux | aarch64 | NVIDIA | ✅ Supported | Jetson tested, see JETSON.md |
| macOS | arm64 | Apple Silicon | 🟡 In Progress | Metal backend, contributions welcome |
| Windows | x86_64 | NVIDIA | ⚠️ Untested | Should work, need CI |

**Legend:**
- ✅ Fully Supported — production-ready, CI tested
- 🟡 Experimental — functional but not performance-tuned
- ⚠️ Untested — may work, no automated testing

## Minimum Requirements

All platforms:
- C++17 compiler
- CMake 3.20+
- 8GB RAM

NVIDIA GPUs:
- Compute Capability 8.0+ (A100, H100, RTX 30/40/50 series)
- CUDA 11.8+ or 12.x
- cuDNN 8.9+

AMD GPUs:
- ROCm 6.0+
- MI200/MI300 series

See docs/hardware.md for tuning recommendations per platform.
```

**Key elements:**
- Visual status indicators
- Explicit legend defining symbols
- Negative space (untested platforms)
- Links to platform-specific guides
- Minimum requirements separate from support matrix

## Common Pitfalls

### 1. Overclaiming Performance Without Evidence

**Anti-pattern:**
```markdown
# SuperFastLLM

The fastest LLM inference engine ever! 100x faster than llama.cpp!
```

**Problems:**
- No hardware specified
- No methodology
- Vague baseline ("100x faster at what?")
- Unfalsifiable

**Fix:**
```markdown
# SuperFastLLM

LLM inference optimized for NVIDIA B200 GPUs.

Llama-3-8B decode: 187 tok/s on B200 vs 162 tok/s (llama.cpp b3501, Q8_0)

Methodology: benchmarks/report.md
```

### 2. Stale Performance Numbers

**Pattern observed:** READMEs with 2-year-old benchmark tables that no longer reflect current performance (either worse due to feature bloat, or better due to optimizations).

**Fixes:**
1. **Date your benchmarks:** "(Benchmarked 2024-11-15)"
2. **Link CI-generated reports:** Auto-update or make staleness visible
3. **Remove if you can't maintain:** Better no number than a wrong number

**Example (TensorRT-LLM):** Performance claims in **dated news items** ("2024-11: Llama 4 at 40k tok/s on B200") rather than a static table. Old items collapse into "Previous News."

### 3. Missing Reproducibility Steps

**Anti-pattern:**
```markdown
Our benchmarks show 5.2x speedup on H100.
```

**What's missing:**
- Baseline comparison (5.2x vs what?)
- Model and quantization
- Batch size, context length
- Commands to reproduce
- Software versions

**Fix:** Provide a reproduction script:
```bash
# In benchmarks/reproduction/h100_llama3_8b.sh
./build/bench \
  --model llama-3-8b-q8.gguf \
  --ctx 2048 \
  --batch 32 \
  --prompt-len 512 \
  --decode-len 128 \
  --runs 10

# Expected: 3,420 tok/s ±120 (95% CI)
```

### 4. Installation Instructions That Break

**Anti-pattern:**
```markdown
Just run `pip install myproject` and you're good!
```

**Why this breaks:**
- Compiled extensions need matching CUDA versions
- System dependencies (cuDNN, ROCm) not mentioned
- Python version requirements unstated

**Pattern from bitsandbytes:**
- Defer to external install docs
- Ship post-install verification script: `check_install.py`
- Document ABI constraints: "PyTorch C++ extension ABI breaks with every PyTorch version"
- Prebuilt wheels with explicit platform tags: `cp310-cu118`

**Better install section:**
```markdown
## Installation

Prerequisites:
- Python 3.10+
- CUDA 11.8 or 12.x
- cuDNN 8.9+

pip install myproject

Verify installation:
python -c "import myproject; myproject.check_cuda()"

If verification fails, see TROUBLESHOOTING.md

Source builds: See BUILDING.md
```

### 5. No Limitations Section

**Why this is a pitfall:** Users discover limitations the hard way (after investing time), leading to frustration and GitHub issues.

**Pattern from ExLlamaV2:**
```markdown
## Known Limitations

- FP8 cache not supported in dynamic generator (use Q4 cache)
- GQA unavailable for 13B quantized models → max context 2048
- Conversion is slow (~45 min for 70B models)
- MacBooks: unlikely to run 13B+ alongside desktop environment
```

**Benefits:**
- Manages expectations
- Reduces support burden
- Builds trust (honesty signal)

### 6. Assuming Ecosystem Knowledge

**Anti-pattern (from "Changelog: Top 10 Reasons"):**
- No explanation of what the project does
- Assumes readers know the problem space
- Jargon without definitions

**Example bad opening:**
```markdown
# FastLLM

Implements FlashAttention 3.0 with paged KV-cache.
```

**Problems:**
- What is this for? (inference? training? research?)
- Who should use it?
- Why vs alternatives?

**Fix:**
```markdown
# FastLLM — Production LLM Inference Server

Serve LLaMA, Qwen, and Mistral models with sub-50ms latency.

Use FastLLM when you need:
- Production serving (vs llama.cpp for local/edge)
- Multi-tenant workloads (vs vLLM for single-user)
- Custom kernels for latest GPUs (B200, GB300)

Built on FlashAttention 3.0 with paged KV-cache.
```

### 7. Broken or Missing Examples

**From "Art of README":** "The ideal example is copy-pasteable and runs without modification."

**Anti-pattern:**
```python
# example.py
import fastllm
model = fastllm.load("model.gguf")  # File doesn't exist
output = model.generate("Hello")    # Wrong API (missing parameters)
print(output)
```

**Fix:**
- Test examples in CI
- Use downloadable models or include test fixtures
- Show expected output

**Pattern from Nerfstudio:**
```bash
# Download test data
ns-download-data nerfstudio --capture-name=poster

# Train (expected: 15 min on RTX 4090)
ns-train nerfacto --data data/nerfstudio/poster

# Expected output in terminal:
#   Step 0: loss=0.421
#   ...
#   Step 1000: loss=0.012 | 42 fps
```

## Best Practices Summary

### Structure Checklist

✅ **Above the fold (first screenful):**
- [ ] Project name + one-line description
- [ ] Badges (CI, version, license)
- [ ] Link bar to docs/Discord/issues
- [ ] One command that runs immediately
- [ ] Visual if applicable (GIF, screenshot)

✅ **First scroll:**
- [ ] What it does and why (vs alternatives)
- [ ] Supported hardware (positive + negative space)
- [ ] Installation (simple path, link to detailed)
- [ ] Quick example with expected output

✅ **Second scroll and beyond:**
- [ ] Feature list (link to guides for each)
- [ ] Limitations/known issues
- [ ] Performance (link to full reports)
- [ ] Documentation links
- [ ] Contributing, license, contact

✅ **External docs:**
- [ ] Build from source (BUILDING.md)
- [ ] API reference (docs/ or wiki)
- [ ] Performance methodology (benchmarks/report.md)
- [ ] Architecture/design docs (ARCHITECTURE.md)
- [ ] Contributing guide (CONTRIBUTING.md)

### Performance Documentation Checklist

For any performance claim in your README:

- [ ] **Hardware specified** — model, GPU generation, CPU, memory
- [ ] **Workload defined** — model size, quantization, context length, batch size
- [ ] **Baseline named** — "vs llama.cpp b3501" not "vs competitors"
- [ ] **Methodology linked** — commands, model IDs, software versions
- [ ] **Date provided** — performance changes over time
- [ ] **Uncertainty shown** — error bars, confidence intervals, or run count
- [ ] **Reproducibility** — script or commands to verify
- [ ] **Honest about gaps** — include unfavorable results if doing comparisons

**Remember:** One reproducible number is worth more than ten vague claims.

### Hardware-Specific Single-Target Repos

If your project targets **one specific GPU or architecture** (e.g., "Optimized for RTX 5090" or "Apple M4 only"):

- [ ] State the constraint **in the README title or subtitle**
- [ ] Explain why (often: latest hardware features not widely supported)
- [ ] Link to general-purpose alternatives for other hardware
- [ ] Document the specific features you leverage (e.g., "sm_120 block-FP4")

**Example:**
```markdown
# bw24-inference — RTX 50-series LLM Inference

Inference engine for NVIDIA sm_120 GPUs (RTX 5090 laptop/desktop).
Leverages block-FP4 instructions not available in prior generations.

For other GPUs, see: llama.cpp (NVIDIA), mistral.rs (multi-platform)
```

### Writing Style

- **Use first-screenful rule:** Most important info in first scroll
- **Write for skimmers:** Headings, bullets, tables over paragraphs
- **Show output:** Every code example should show expected result
- **Link aggressively:** External docs, papers, issues, related projects
- **Use relative links:** `docs/BUILDING.md` not full URLs (works in clones)
- **Test your instructions:** Have someone else follow your quickstart
- **Date time-sensitive content:** Benchmarks, news, compatibility
- **Separate audiences:** "For users" vs "For contributors"

### Maintenance Practices

1. **README-driven development:** Write the README first (shows API clarity)
2. **Test examples in CI:** Catch bitrot immediately
3. **Version-pin claims:** "As of v1.2.0" prevents future confusion
4. **Deprecation notices:** Banner at top if project is superseded
5. **Contribution instructions:** Lower the barrier for doc updates
6. **Automate what goes stale:** Model lists, contributor avatars, benchmark CI
7. **Archive old news:** Collapsible sections keep the README scannable

## Further Reading

### Official Project READMEs (Exemplary)

| Project | Why Study It | Key Lesson |
|---------|--------------|------------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | Hub-and-spoke, reproducible benchmarks | README as router; literal tool output |
| [vLLM](https://github.com/vllm-project/vllm) | Production focus, clean scope | Mechanism > metrics; external docs |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | Benchmark credibility | Linked full reports, honest losses |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | Compiled extensions | Support matrix, post-install verification |
| [Candle](https://github.com/huggingface/candle) | Rust/systems project | Runnable example first, PyTorch cheatsheet |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Enterprise governance | Deprecation policy, telemetry transparency |
| [Redis](https://github.com/redis/redis) | Mature systems project | Docker-verified builds, smoke tests |
| [PyTorch](https://github.com/pytorch/pytorch) | Massive ecosystem | Audience routing, install complexity management |
| [Rust](https://github.com/rust-lang/rust) | Language-scale docs | Minimal README, everything delegated |

### Guides and Frameworks

| Resource | Type | Key Takeaway |
|----------|------|--------------|
| [Art of README](https://github.com/noffle/art-of-readme) | Essay | Cognitive funnel, README as product |
| [Standard Readme](https://github.com/RichardLitt/standard-readme) | Spec | Required sections, badge guidelines |
| [makeareadme.com](https://www.makeareadme.com) | Quick guide | Essential sections, common mistakes |
| [Diátaxis](https://diataxis.fr) | Framework | 4 doc types (tutorial/how-to/reference/explanation) |
| [Write The Docs Guide](https://www.writethedocs.org/guide/writing/beginners-guide-to-docs/) | Community resource | Why document, types, maintenance |
| [GitHub README Docs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) | Official | Placement, size limits, relative links |
| [18F Open Source Guide](https://github.com/18F/open-source-guide/blob/18f-pages/pages/making-readmes-readable.md) | Gov guide | Common problems, validate with users |
| [Linux Kernel Checklist](https://www.kernel.org/doc/html/latest/process/submit-checklist.html) | Systems standard | Evidence requirements, testing rigor |

### Academic/Research

| Resource | Focus | Relevance |
|----------|-------|-----------|
| Meta Llama README | Model distribution, gated access | Licensing, responsible AI disclosure |
| alpaca-lora README | Experimental results | Reproducible commands, honest framing |
| Nerfstudio README | Research to production | Visual results, quickstart design |

### Anti-Patterns and Pitfalls

| Resource | Type | Value |
|----------|------|-------|
| [Changelog: Top 10 Reasons I Won't Use Your OSS](https://changelog.com/posts/top-ten-reasons-why-i-wont-use-your-open-source-project) | Blog (2011) | Adoption blockers, missing elements |
| [Awesome README Examples](https://github.com/matiassingers/awesome-readme) | Curated list | 200+ examples with pattern descriptions |

---

## Actionable Next Steps

1. **Audit your README against the structure checklist** — are critical elements above the fold?
2. **Run someone else through your quickstart** — does it work without your help?
3. **Review your performance claims** — can a reader reproduce them? Add methodology links.
4. **Disclose limitations** — add a "Known Limitations" or "Not Supported" section
5. **Create a BUILDING.md** — move complex install instructions out of the README
6. **Add verification** — post-install or post-build smoke test
7. **Test your examples in CI** — prevent bitrot
8. **Study 3 exemplary READMEs** — llama.cpp, mistral.rs, vLLM are good starts

---

*This guide was synthesized from 20 sources spanning production inference engines, research projects, documentation frameworks, and systems engineering standards. See `resources/readme-technical-inference-repos-sources.json` for full source list and quality ratings.*
