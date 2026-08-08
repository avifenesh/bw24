#!/bin/bash
# 122B capacity receipt: c=8 sessions at 8k ctx, no-OOM, completion + VRAM peak.
set -u
cd /root/bw24-122b
export PATH=/root/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
MODEL=/dev/shm/122b/Qwen3.5-122B-A10B-UD-IQ4_XS.gguf
OUT=/root/receipts-122b/logs
ADDR=127.0.0.1:8199
exec 9>/tmp/gpu5090.lock
flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 1; }

MEMRA_COMPAT=openai MEMRA_MODELS="m122b=$MODEL" MEMRA_ADDR=$ADDR MEMRA_CTX=8192 \
  target/release/memra-server > "$OUT/cap-server.log" 2>&1 &
SPID=$!
up=0
for _ in $(seq 200); do
  curl -sf "http://$ADDR/v1/models" 2>/dev/null | grep -q m122b && { up=1; break; }
  kill -0 $SPID 2>/dev/null || { echo "FATAL server died"; tail -5 "$OUT/cap-server.log"; exit 1; }
  sleep 2
done
[ "$up" = 1 ] || { echo "FATAL never up"; kill $SPID; exit 1; }
echo "server up"

# VRAM sampler
( while kill -0 $SPID 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > "$OUT/cap-vram.csv" &
VPID=$!

python3 - "http://$ADDR" 8 "$OUT/cap-rows.jsonl" <<'PY'
import json,sys,threading,time,urllib.request
base,n,rows=sys.argv[1],int(sys.argv[2]),sys.argv[3]
# ~6k-char prompts -> >1k tokens each, 8k ctx sessions
SEED="The storage-to-compute pipeline of a modern LLM inference engine spans mmap fallback, explicit positioned reads, local-NVMe access, bounded pinned host buffers, residency caching, asynchronous prefetch and overlap, PCIe transfer, and GPU kernels. "
res=[];lk=threading.Lock()
def w(i):
    prompt=f"[session {i}] "+SEED*24+" Summarize the pipeline stages and their failure modes."
    body={"model":"m122b","messages":[{"role":"user","content":prompt}],
          "max_tokens":256,"temperature":0.0,"seed":9000+i,"stream":True}
    r=urllib.request.Request(base+"/v1/chat/completions",data=json.dumps(body).encode(),
                             headers={"Content-Type":"application/json"})
    t0=time.time();ttfb=None;nchunk=0;ok=False;err=""
    try:
        with urllib.request.urlopen(r,timeout=1800) as resp:
            for raw in resp:
                if ttfb is None: ttfb=time.time()-t0
                l=raw.decode("utf-8","replace").strip()
                if not l.startswith("data: "): continue
                p=l[6:]
                if p=="[DONE]": ok=True; break
                try:
                    ch=json.loads(p)
                    d=ch.get("choices",[{}])[0].get("delta",{}) or {}
                    if d.get("content") or d.get("reasoning"): nchunk+=1
                    if "error" in json.dumps(ch).lower() and "step error" in json.dumps(ch): err="step-error"
                except Exception: pass
    except Exception as e: err=str(e)[:120]
    row={"i":i,"ok":ok,"ttfb":ttfb,"wall":time.time()-t0,"chunks":nchunk,"err":err}
    with lk: res.append(row)
ts=[threading.Thread(target=w,args=(i,)) for i in range(n)]
[t.start() for t in ts];[t.join() for t in ts]
with open(rows,"w") as f:
    for r in sorted(res,key=lambda x:x["i"]): f.write(json.dumps(r)+"\n")
nok=sum(1 for r in res if r["ok"] and r["chunks"]>0 and not r["err"])
print(f"CAPACITY c=8: {nok}/8 completed clean")
PY
kill $VPID 2>/dev/null
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
echo "VRAM peak MiB: $(sort -n "$OUT/cap-vram.csv" | tail -1)"
grep -c "out of memory\|CUDA_ERROR_OUT_OF_MEMORY" "$OUT/cap-server.log" | xargs echo "OOM lines in server log:"
echo "CAP DONE"
