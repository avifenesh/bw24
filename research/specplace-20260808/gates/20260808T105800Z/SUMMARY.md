# specplace gate summary

- Host: ip-172-31-11-170
- Commit: 8d8ba1eaad71c4d2c36426f20d24f7e4d28d14be
- Script-detected failures: 0

## run-spec
runspec-pp2-01.log:=== SELF-CONSISTENCY PASS ===
runspec-pp2-10.log:=== SELF-CONSISTENCY PASS ===
runspec-single.log:=== SELF-CONSISTENCY PASS ===

## policy
policy-pp2-default-server.log:[spec-gate] policy placement=pp2-cross-device LOW=0 HIGH=1 source=placement-default spec-admission=off
policy-single-default-server.log:[spec-gate] policy placement=single-or-non-pp2 LOW=2 HIGH=4 source=placement-default spec-admission=on
crash-pp2-forced-spec-server.log:[spec-gate] policy disabled by MEMRA_SPEC_GATE=0: always-spec

## load points
- policy-pp2-default-c1: ok=4 err=0 shed=0 agg=221.8 tok/s
- policy-single-default-c1: ok=4 err=0 shed=0 agg=377.4 tok/s
- crash-c2: ok=8 err=0 shed=0 agg=250.6 tok/s
- crash-c4: ok=16 err=0 shed=0 agg=245.8 tok/s
- crash-recovery-c1: ok=4 err=0 shed=0 agg=250.2 tok/s

## serve-smoke
== serve-smoke: plain serving ==
  ok: /models lists the model
  ok: chat non-stream (text + usage + finish_reason)
  ok: chat stream (SSE chunks + [DONE])
  ok: /v1/completions
  ok: greedy determinism (2 runs identical)
  ok: 3 concurrent chats
  ok: long generation (>=100 tok)
== serve-smoke: cache-metering exactness ==
  ok: A-req1 cached_tokens == 0
  ok: A-req1 prompt_tokens == 272
  ok: A-req2 cached_tokens == 0
  ok: A-req2 prompt_tokens == 272
  ok: A-req3 cached_tokens == 256
  ok: A-req3 prompt_tokens == 272
  ok: A-req4 cached_tokens == 256
  ok: A-req4 prompt_tokens == 272
  ok: A-req5 cached_tokens == 256
  ok: A-req5 prompt_tokens == 272
  ok: B-req1 cached_tokens == 0 (cross-salt blindness)
  ok: prompt_tokens_in == 1632
  ok: cached_tokens_in == 768
  ok: computed_tokens_in == prompt - cached
  ok: cache_hit_token_ratio matches arithmetic
  ok: prefix_cache_hits == N-2
  ok: prefix_cache_misses == 3
  ok: prefix_cache_inserts == 3 (A seed + A split + B seed)
  ok: prefix_cache_hit_tokens == 768
  ok: lcp_histogram: 6 probes total
  ok: lcp_histogram: 2 cold probes in bucket [0]
  ok: lcp_histogram: 4 probes in K's bucket (edge 256)
  ok: tick-seg [64,512) window carries the shared-prefix probes
  ok: tenants[meter-A] split exact
  ok: tenants[meter-B] split exact (0 cached)
  ok: economics revenue_multiplier == 1.8889 (factor 1.0)
  ok: cache-metering accounting exact (per-request + /metrics + economics)
== serve-smoke: spec serving (draft attached) ==
  ok: spec == plain greedy text (serving exactness)
== serve-smoke: sampled truncation matrix (spec server) ==
  ok: trunc top_k=40 (reproducible, bangs=0 <= baseline 0)
  ok: trunc top_p=0.95 (reproducible, bangs=0 <= baseline 0)
  ok: trunc min_p=0.05 (reproducible, bangs=0 <= baseline 0)
  ok: trunc llama-default k40+p0.95+m0.05 (reproducible, bangs=0 <= baseline 0)
== serve-smoke: session-affinity resume exactness ==
  ok: affinity resume is deterministic across servers (4 rewritten turns)
  ok: replay arm resumed too (3 rewind(s))
  ok: affinity fired (3 rewind(s) on a rewritten history)
  ok: no failed rewinds
== serve-smoke: gemma4 arm SKIP (no model at /data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf)
serve-smoke: 0 failed
