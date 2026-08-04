# Research notes — what converts on dev-tool sites (web, fetched 2026-08-05)

Supporting evidence for SPEC.md §6 (design), §7 (site architecture), §10 (OG image).
All URLs fetched live 2026-08-05 unless noted.

## 1. Conversion evidence (published teardowns)

- Evil Martians, "How to kill conversions on your developer tool's landing page"
  (2025-03-05, evilmartians.com/chronicles/how-to-kill-conversions-on-your-developer-tool-landing-page):
  replacing vague messaging with specific metrics moved their conversion 0.1% → 2.0%.
  "Blazing fast" loses to "handles JPEG images at least 35% faster than leading
  alternatives"; "leading companies trust us" loses to a named-customer number. CTA
  wording measurable ("Build with us" → "Hire Martians": 1.3% → 2.0%). Strip first-screen
  distractions. "The best way to impress developers is to respect their time enough to
  not waste it."
- daily.dev Ads, "How to Create Developer-First Landing Pages That Convert" (2026-05-09,
  business.daily.dev/resources/create-developer-first-landing-pages-convert/): code block
  more than ~1,000 px down loses up to 50% of technical traffic — snippet above the fold
  is table stakes. Headline = what/who/does in specifics. Pricing behind "Contact Sales"
  is a trust-killer. Dual-CTA hero (primary + "View Docs"). Micro-copy under CTA ("No
  credit card required"). Dark mode mirrors editors; real terminal output beats abstract
  3D art. North-star activation metric: "Time to First Hello World."
- Heavybit, "Developer Marketing Mistakes: What Developers Hate"
  (heavybit.com/library/article/developer-marketing-mistakes): devs don't hate marketing,
  they hate spam; transparent guides to what the product can AND CAN'T do are the
  antidote — an argument for darklanes publishing its documented numeric-drift limits.
- dev.to "Landing Pages That Convert for Developer Tools" (2026-04-09, AI-authored,
  weight low): 3-second hero decision; specifics beat outcomes; devs read the whole page
  — repeat the CTA at the bottom.

Convergent pattern: specificity with numbers, code above the fold, transparent pricing,
docs one click away, one low-friction action, no stock imagery, no superlatives.

## 2. Exemplar devices worth stealing (all homepages fetched 2026-08-05)

| Site | Headline | Device to steal |
|---|---|---|
| Linear.app | "A new species of product tool" | "FIG 0.2/0.3" schematic labeling on feature panels — marketing dressed as engineering diagrams |
| Vercel.com | "Build agents on infrastructure that thinks like them" | Geist Sans + Geist Mono (OFL, free); customer metrics woven into hero copy |
| Fly.io | "Computers for agents" | Hero stat strip: "18+ regions · <1 second machine boot · 500ms deploy · 99.9% uptime SLA" — four mono numbers doing the whole positioning job |
| Modal.com | "AI infrastructure that developers love" | Public per-second GPU price table as the honesty signal; $30/mo starter credits |
| Resend.com | "Email for developers" | Multi-language SDK tab switcher in the hero; exemplary pricing transparency (Free $0/3k, Pro $20, overage $0.90/1k stated on-page) |
| ClickHouse.com | "millisecond queries at petabyte scale" | Animated mechanism explainer on the homepage; GitHub stats strip; dedicated public /benchmarks page |

## 3. Dark-theme branding patterns

- Frontend Horse, "The Linear Look" (frontend.horse/articles/the-linear-look/): the
  catalogued trend across Linear/Vercel/Railway/Resend/Raycast/Supabase — dark
  backgrounds, thin 1px lines, dot/grid substrates, screenshots/code instead of photos,
  gradient headings, bento grids, blurry glows. Now well-worn: take the substrate
  (near-black + thin lines + grid), skip the clichés (glow blobs, bento-for-bento's-sake).
- Vercel proves zero-accent near-black/white works if typography carries it
  (shadcn.io/theme/vercel). Dark-first is the current dev-tool norm (adminlte.io,
  2026-07). Recurring formula: near-black surfaces (not pure #000), ONE restrained
  accent, mono numerals for anything measured.

## 4. Live metrics / status as marketing (the genre darklanes fits)

- Groq: brand built on third-party speed proof — ArtificialAnalysis "doubled its chart
  axis to fit Groq" (groq.com/blog/artificialanalysis-ai-llm-benchmark-doubles-axis...);
  console demo with a visible tokens/s counter as the conversion device.
- Cerebras launch post: "2.4x faster than Groq" with live demo links — speed-demo
  one-upmanship is the established inference-provider genre.
- OpenRouter publishes rolling p50/p75/p90/p99 latency + throughput charts per provider
  endpoint, uptime, and quantization disclosure, and teaches buyers to shop on TAIL
  latency ("How to Evaluate LLM Provider Performance," openrouter.ai/blog/insights/,
  ~2026-07). Undisclosed quantization framed as "the hidden quality variable" — the
  exact opening for darklanes' exactness positioning. Rule: publish your own percentile
  graphs before someone else measures you; market p99 where competitors market p50.
- Status pages as trust: Hyperping (2025-08-13) — "some companies even use their
  excellent uptime statistics as marketing material"; UptimeRobot guide (2026-06-17):
  host status on separate infrastructure/domain.

## 5. OG-image guidance

- GitHub's OG framework (github.blog, 2021-06-30): repo-card aesthetic — name, one-liner,
  big metric row — is the visual language devs already trust in feeds; HTML template →
  headless screenshot; generated per page.
- PageThen (2026-03-30): 1200×630; ≥60 px bold headline, ~40 chars; center safe zone;
  dark bg + light text explicitly recommended; test at 200×105 thumbnail; 3–5x CTR
  difference claimed between good and bad cards. Scanly (2026-05-20) / Slick Media
  (2025-10-20): feeds render in light AND dark — mid-gray cards die; ≥4.5:1 contrast.
- Dynamic generation: Vercel OG / `ImageResponse` pattern.
