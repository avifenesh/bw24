# ACF search: fa3 harness v11 (full-TMA staging) on H100, ptxas 13.3 apply-controls.
# Objective = v11 us/call (min), gated on the harness correctness MATCH lines.
import pathlib
from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.utils.helpers import save_compiler_config

def objective(config):
    import subprocess, re, os, fcntl
    from uuid import uuid4
    from compileiq.utils.helpers import save_compiler_config
    INVALID = "*"
    acf = f"/tmp/ciq_{uuid4().hex}.acf"
    exe = f"/tmp/ciq_fa3_{uuid4().hex}"
    try:
        save_compiler_config(acf, config)
        r = subprocess.run(
            [os.path.expanduser("~/cuda-13.3.1/bin/nvcc"),
             "-gencode", "arch=compute_90a,code=sm_90a", "-O3",
             "-DA_LEAD=128", "-DB_LEAD=128",
             f"-Xptxas=--apply-controls={acf}",
             "-I/usr/local/cuda-13.1/targets/x86_64-linux/include",
             "-L/usr/local/cuda-13.1/lib64", "-lcuda",
             "-o", exe, os.path.expanduser("~/bw24-unified/tools/bench_fa3.cu")],
            capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return INVALID
        with open("/tmp/ciq_gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            out = subprocess.run([exe], capture_output=True, text=True,
                                 timeout=300).stdout
        if "MISMATCH" in out:
            return INVALID
        m = re.findall(r"v11 T=2048: (\d+)us/call", out)
        if not m:
            return INVALID
        return float(m[-1])
    except Exception:
        return INVALID
    finally:
        for p in (acf, exe):
            try:
                os.remove(p)
            except OSError:
                pass

if __name__ == "__main__":
    search = Search(
        objective_function=objective,
        search_space=PtxasSearchSpace(version="13.3"),
        search_config={"pool_size": 32, "generations": 8, "mutate_rate": 0.3,
                       "problem_type": "min", "num_objectives": 1},
        dump_results=pathlib.Path("~/acf-fa3-results.csv").expanduser(),
        disable_progress_bar=True,
    )
    res = search.start(num_workers=6)
    best = res.get_best_result()
    print("BEST score:", best["score_1"])
    save_compiler_config(str(pathlib.Path("~/acf-fa3-best.acf").expanduser()), best["params"])
    print("saved ~/acf-fa3-best.acf")
