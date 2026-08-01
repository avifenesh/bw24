# Lane status — updated per completed stage

| stage | ornith9b | ornith35b | katcoder |
|---|---|---|---|
| own-gen corpus (254 prompts, ~130k tok) | pending | pending | pending |
| ranks artifact (owngen-ranks-32768) | pending | pending | pending |
| drafter built (donor block + own trim) | pending | pending | pending |
| run-spec K=1..8 self-consistency | pending | pending | pending |
| acceptance table K=2..4 (p1/p2/p3) | pending | pending | pending |
| e2e spec/plain x3 (serving K) | pending | pending | pending |

Rig notes: corpus runs chunked via `gen-corpus-chunk.sh` (64 prompts/chunk under
`flock /tmp/gpu5090.lock`). 2026-08-01: a sibling lane's memra-server sits resident with
18.2 GiB VRAM — the 35B-class runs (21.2/18.8 GiB) need it gone; 9B (9.5 GiB) fits
alongside. Priority Ornith-35B > Ornith-9B > KAT, VRAM permitting.
