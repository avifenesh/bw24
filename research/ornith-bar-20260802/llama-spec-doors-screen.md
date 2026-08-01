# llama.cpp draftless speculative doors on Ornith-9B — screen verdict (2026-08-01)

Mission: the 9B best-vs-best cell gives llama a fair best-effort at a speculative config,
since llama has no Ornith draft artifact. This fork (local build, HEAD `bb090d1f1`, libs
0.0.9839) ships no lookup/lookahead binaries; both were built from the fork's own source for
this screen (`make llama-lookup llama-lookahead llama-completion`, examples configured ON).

## llama-lookahead: NOT SUPPORTED on this arch

Fails immediately on the qwen3.5 hybrid position rule (`o9b-llama-lookahead-*-rep0.log`):

    the last position stored in the memory module ... for sequence 1 is X = 30
    the tokens for sequence 1 in the input batch have a starting position of Y = 28
    for M-RoPE, it is required that the position satisfies: X < Y
    decode: failed to initialize batch / llama_decode failed - increase KV cache size

Lookahead assigns overlapping positions to parallel branches by design — structural, not a
config problem.

## llama-lookup (n-gram): BROKEN on this arch — numbers are not evidence

Screen + draft-max sweep (`o9b-llama-lookup-*-rep0.log`, `o9b-lookup-sweep-*.log`):

| config | class | "decoded" t/s | accept | disqualifier |
|---|---|---|---|---|
| dm=3 | p1 | 219.5 | 64.2% | mid-run `inconsistent sequence positions` error in-stream; output != plain greedy |
| dm=3 | p2 | 192.6 | 62.7% | same error class |
| dm=3 | p3 | 200.9 | 75.3% | same error class (after -b 8192 fix for the single-shot prompt decode) |
| dm=8 | p1 | 402.6 | 23.2% | output != dm=3 output != plain greedy |
| dm=16 | p1 | 513.1 | 11.8% | output DEGENERATE ("insert, insert, insert, ..." x86) |
| dm=16 | p2 | — | — | ggml_abort in common_sampler_sample (backtrace in log) |

Greedy lookup decoding is supposed to be output-lossless vs plain greedy — the bar memra's
own spec arm is held to (run-spec self-consistency gate). Here dm3 != dm8 != dm16 != plain,
positions go inconsistent mid-run on the M-RoPE arch, and the highest readings are throughput
on degenerate repetition (repetitive text is exactly what an n-gram cache accelerates). None
of these qualify as a llama best config.

(Also fixed for the screen: the lookup/lookahead examples single-shot the whole prompt into
one llama_decode — p3's 6.3k-token prompt needs `-b 8192` or it ggml_aborts; receipts
`o9b-llama-lookup-p3-agentic-long-rep0.log` (crash) vs `o9b-lookup-sweep-dm3-p3-agentic-long.log`.)

## Verdict

llama's best Ornith-9B config on this rig = its best PLAIN: llama-completion, swept-best
flags from the board convention (`-ngl 999 -fa on -ctk q8_0 -ctv q5_1`), greedy, --ignore-eos.
(this fork's llama-cli refuses non-conversation mode and defers to llama-completion; its
plain decode reading, 84.4 t/s on p1, matches the serve-lane llama-bench tg128 83.8 within
regime spread.)
