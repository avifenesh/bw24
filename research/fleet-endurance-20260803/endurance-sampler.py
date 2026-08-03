#!/usr/bin/env python3
import json, subprocess, time, calendar, os, urllib.request
OUT = "/home/ubuntu/receipts/fleet-endurance-20260803"
FLEET = OUT + "/fleet"
END = calendar.timegm((2026, 8, 3, 10, 48, 0))

def gpu_sample():
    out = subprocess.run(["nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = []
    for l in out.strip().splitlines():
        p = [x.strip() for x in l.split(",")]
        rows.append({"i": int(p[0]), "util": int(p[1]), "mem_mib": int(p[2]),
                     "temp": int(p[3]), "pw_w": float(p[4])})
    return rows

def rss_of(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for l in f:
                if l.startswith("VmRSS:"):
                    return int(l.split()[1])
    except Exception:
        return None
    return None

def pidfile(p):
    try:
        return int(open(p).read().strip())
    except Exception:
        return None

def replicas():
    out = []
    for port in range(9085, 9093):
        pid = pidfile(f"{FLEET}/replica-{port}.pid")
        out.append({"port": port, "pid": pid,
                    "rss_kb": rss_of(pid) if pid else None})
    return out

def proxy_metrics():
    try:
        with urllib.request.urlopen("http://127.0.0.1:9080/metrics", timeout=3) as r:
            return json.load(r)
    except Exception as e:
        return {"err": f"{type(e).__name__}: {e}"}

def last_window():
    try:
        with open(OUT + "/load-windows.jsonl", "rb") as f:
            f.seek(0, 2)
            sz = f.tell()
            f.seek(max(0, sz - 8192))
            lines = f.read().decode(errors="replace").strip().splitlines()
        d = json.loads(lines[-1])
        return {k: d.get(k) for k in ("label", "agg_tok_s", "lat_p50_s",
                                      "lat_p95_s", "n_ok", "n_err", "n_shed")}
    except Exception:
        return None

while time.time() < END:
    ppid = pidfile(FLEET + "/proxy.pid")
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpus": gpu_sample(),
        "replicas": replicas(),
        "proxy": {"pid": ppid, "rss_kb": rss_of(ppid) if ppid else None},
        "proxy_metrics": proxy_metrics(),
        "last_window": last_window(),
    }
    with open(OUT + "/samples.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    time.sleep(60)
