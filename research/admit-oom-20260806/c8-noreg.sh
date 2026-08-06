#!/usr/bin/env bash
# c=8 no-regression: fix-on (defaults) vs fix-off (MEMRA_STEP_OOM_RETRIES=0 +
# MEMRA_ADMIT_RESERVE_MB=0 => reserve 0, i.e. gate reduces to `free >= cost`, the
# pre-fix behavior class at c=8 where the old 2x term never bound anyway).
# N=5 INTERLEAVED (on,off,on,off,...) per the interleaving law.
set -uo pipefail
cd /home/avifenesh/projects/wt-admitoom
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
OUT=research/admit-oom-20260806/logs
ADDR=127.0.0.1:8188
run_arm() {
  local arm="$1" rep="$2"
  local envs=(MEMRA_COMPAT=openai "MEMRA_MODELS=stress=$MODEL+$DRAFT" MEMRA_ADDR=$ADDR MEMRA_CTX=8192)
  [ "$arm" = off ] && envs+=(MEMRA_STEP_OOM_RETRIES=0 MEMRA_ADMIT_RESERVE_MB=0)
  env "${envs[@]}" target/release/memra-server > "$OUT/c8-$arm-r$rep-server.log" 2>&1 &
  local spid=$!
  local up=0
  for _ in $(seq 150); do
    if curl -sf "http://$ADDR/v1/models" 2>/dev/null | grep -q '"stress"'; then up=1; break; fi
    kill -0 $spid 2>/dev/null || { echo "FATAL: server died ($arm r$rep):"; tail -5 "$OUT/c8-$arm-r$rep-server.log"; exit 1; }
    sleep 2
  done
  [ "$up" = 1 ] || { echo "FATAL: our server never came up on $ADDR ($arm r$rep)"; tail -5 "$OUT/c8-$arm-r$rep-server.log"; kill $spid 2>/dev/null; exit 1; }
  python3 - http://$ADDR 8 "$arm" "$rep" "$OUT/c8-rows.jsonl" <<'PY'
import json,statistics,sys,threading,time,urllib.request
base,n,arm,rep,rows=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5]
PROMPTS=["Explain speculative decoding in three sentences.",
         "List three failure modes of a GPU serving stack under high concurrency.",
         "Summarize what a KV cache stores during LLM decoding.",
         "Describe the difference between p50 and p99 latency."]
res=[];lk=threading.Lock()
def w(i):
    body={"model":"stress","messages":[{"role":"user","content":f"[c{i}] "+PROMPTS[i%4]}],
          "max_tokens":192,"temperature":0.0,"seed":7000+i,"stream":True}
    r=urllib.request.Request(base+"/v1/chat/completions",data=json.dumps(body).encode(),
                             headers={"Content-Type":"application/json"})
    t0=time.time();ttfb=None;nt=0;nchunk=0;acc=None;ok=False
    try:
        with urllib.request.urlopen(r,timeout=600) as resp:
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
                    u=ch.get("usage")
                    if u:
                        nt=u.get("completion_tokens",0)
                        sp=u.get("spec") or {}
                        acc=sp.get("acceptance_rate")
                except Exception: pass
    except Exception as e:
        with lk: res.append({"i":i,"err":str(e)[:200]}); return
    wall=time.time()-t0
    with lk: res.append({"i":i,"ok":ok,"toks":nt,"chunks":nchunk,"acc":acc,"ttfb_s":ttfb,"wall_s":wall})
ts=[threading.Thread(target=w,args=(i,)) for i in range(n)]
for t in ts: t.start()
t0=time.time()
for t in ts: t.join()
batch=time.time()-t0
good=[r for r in res if r.get("ok")]
tot=sum(r["toks"] for r in good)
walls=sorted(r["wall_s"] for r in good); ttfbs=sorted(r["ttfb_s"] for r in good)
def pc(v,p): return v[min(len(v)-1,int(p*len(v)))] if v else float("nan")
accs=[r["acc"] for r in good if r.get("acc") is not None]
row={"arm":arm,"rep":int(rep),"c":n,"ok":len(good),"agg_tok_s":round(tot/batch,2),
     "p50_wall_s":round(pc(walls,.5),3),"p95_wall_s":round(pc(walls,.95),3),
     "p50_ttfb_s":round(pc(ttfbs,.5),3),"batch_s":round(batch,2),"tokens":tot,
     "acc_mean":round(sum(accs)/len(accs),4) if accs else None}
open(rows,"a").write(json.dumps(row)+"\n")
print(json.dumps(row))
PY
  kill $spid 2>/dev/null; wait $spid 2>/dev/null
  sleep 3
}
for rep in 1 2 3 4 5; do
  run_arm on  "$rep"
  run_arm off "$rep"
done
