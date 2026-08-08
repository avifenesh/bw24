# AI Horde bridge selection

Source audit date: **2026-08-08**. The revisions below were checked directly
against each repository's live `HEAD`; none is inferred from package indexes or
old issue text.

| Candidate | Live revision and last commit | Result |
|---|---|---|
| [Haidra-Org/AI-Horde-Worker](https://github.com/Haidra-Org/AI-Horde-Worker) | `d0bacd83f996550d934c105ea25ac7a0e0fb380e`, 2026-03-07 | Official text worker and still the reference implementation. Its OpenAI path does not provide a configured Authorization header for the local backend, so it cannot call memra while the trial keyring is enabled. |
| [the-crypt-keeper/Medusa-Bridge](https://github.com/the-crypt-keeper/Medusa-Bridge) | `21f156cabe910ac70a52e7fb7172779a2b3fb026`, 2025-05-05 | Older and not a direct fit: its vLLM engine targets `/generate`, and its OpenAI-like paths do not cover memra's keyed endpoint cleanly. |
| [whs/hordebridge](https://github.com/whs/hordebridge) | `8a56dab5af4e0b768a67ed23a6ce4319180ca96c`, 2026-07-31 | Freshest code and direct OpenAI auth, but one commit/no release, a Go 1.26.4 source build with a large dependency graph, and prompt truncation is still marked TODO. Kept as the first fallback if it gains releases. |
| [Belarrius1/belarrius-ai-horde-bridge](https://github.com/Belarrius1/belarrius-ai-horde-bridge) | `36659f95cb9cbe3caf847a36ecbb41d08fb913f7`, 2026-03-31 | Selected current-best operational fit. It has direct Bearer auth, generic `/v1/completions`, bridge-side context rejection, output caps, and one-thread polling. It is also a one-commit project with no releases, so this lane pins the full revision and package lock rather than claiming a mature maintenance history. |

The selected bridge reports package version `1.5.0`; its package metadata says
ISC. The checkout has no top-level license file at the pinned revision. The setup
script clones rather than vendors the source and records the exact revision.

The Node runtime is pinned to upstream-recommended Node 22:
`node-v22.23.2-linux-x64.tar.xz`, SHA-256
`d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307`.
The checksum was checked against Node's `latest-v22.x/SHASUMS256.txt` on the
source-audit date.
