set -uo pipefail
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:${LD_LIBRARY_PATH:-}
export PATH=/root/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
cd /root/wt-rpst
OUT=/root/rpst-verdict; mkdir -p $OUT
ST=/root/models/st27
PORT=8314; BASE=http://127.0.0.1:$PORT
SPID=0
# THE WIDTH QUESTION the fp8dec lane could not have asked: MEMRA_ST_E4M3=0 sends the ST
# checkpoint back to the ARM B' Q8_0 slab, whose batched inner loop is dp4a (4 int8 MACs
# per instruction) instead of e4m3's per-weight fmaf chain. That arm LOST at m=1 by 2.58pp
# (research/fp8dec-20260805/), but m=1 has no column loop, so it never priced the ALU wall
# this lane found. If the ST batched path is ALU-bound at width, the dp4a slab may WIN at
# c=16 even while losing at c=1 — and both are the same checkpoint bytes on disk.
#   e4m3  = MEMRA_ST_E4M3 default ON  (native e4m3 residency, hoisted)
#   slab  = MEMRA_ST_E4M3=0           (Q8_0 slab residency, dp4a batched loop)
start() {
  case "$1" in
    e4m3) ENVV="MEMRA_MODELS=m=$ST" ;;
    slab) ENVV="MEMRA_MODELS=m=$ST MEMRA_ST_E4M3=0" ;;
  esac
  env $ENVV MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:$PORT MEMRA_SERVE_SPEC=0 \
      MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=40 \
      target/release/memra-server > $OUT/ssrv-$1.log 2>&1 &
  SPID=$!
  for i in $(seq 1 500); do curl -s -m 2 $BASE/models >/dev/null 2>&1 && break; sleep 3; done
  sleep 5
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' ' > $OUT/svram-$1.txt
  grep -iE "chunk cap|residency census|F8_E4M3|Q8_0" $OUT/ssrv-$1.log | head -6
}
stop() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 10; }
for R in 1 2 3 4 5; do
  for ARM in e4m3 slab; do
    echo "### slab-round=$R arm=$ARM $(date -u +%H:%M:%S)"
    start $ARM
    for C in 1 16 32; do
      REQ=$((C*3)); [ $REQ -lt 8 ] && REQ=8
      python3 tools/load-serve.py --base $BASE --model m --concurrency $C \
        --requests $REQ --max-tokens 128 --warmup 1 \
        --label "$ARM-c$C-r$R" --out $OUT/slab-points.jsonl 2>&1 | tail -1
    done
    echo "svram-$ARM: $(cat $OUT/svram-$ARM.txt)"
    stop
  done
done
echo "### DONE $(date -u +%H:%M:%S)"
