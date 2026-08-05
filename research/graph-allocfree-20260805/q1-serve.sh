#!/bin/bash
# Q1 serve c=1 arm: interleaved auto_free vs priority, N=5. Serve path = where the generic
# GraphSession actually gets promoted (worker.rs:1334 graph_session_from_cache_masked).
# c=1 greedy is the arm that promotes (worker.rs:1251 "greedy interactive session ALONE").
cd /root/bw24-q1
export PATH=/root/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:$LD_LIBRARY_PATH
Q27=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
export MEMRA_MODELS="qwen=$Q27"
export MEMRA_ADDR=127.0.0.1:8137
export MEMRA_CTX=8192
for rep in 1 2 3 4 5; do
  for flag in auto_free priority; do
    pkill -f 'memra-server' 2>/dev/null; sleep 4
    MEMRA_GRAPH_IFLAG=$flag ./target/release/memra-server > /tmp/q1srv-$flag-$rep.log 2>&1 &
    ok=0
    for i in $(seq 1 120); do
      curl -sf http://127.0.0.1:8137/v1/models >/dev/null 2>&1 && { ok=1; break; }; sleep 2
    done
    if [ $ok -eq 0 ]; then echo "=== rep$rep flag=$flag SERVER FAILED TO START ==="; tail -5 /tmp/q1srv-$flag-$rep.log; continue; fi
    sleep 3
    echo "=== rep$rep flag=$flag ==="
    python3 tools/load-serve.py --base http://127.0.0.1:8137 --concurrency 1 --requests 4 \
      --model qwen --greedy --label "q1-$flag-$rep" 2>&1 | grep -iE "tok/s|p50|p95|error|shed|n_ok"
    pkill -f 'memra-server' 2>/dev/null; sleep 4
  done
done
pkill -f 'memra-server' 2>/dev/null
