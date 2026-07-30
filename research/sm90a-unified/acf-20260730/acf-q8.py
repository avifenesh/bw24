# ACF search: q8 harness on H100, ptxas 13.3 apply-controls (same recipe as acf-fa3.py).
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
    exe = f"/tmp/ciq_q8_{uuid4().hex}"
    try:
        save_compiler_config(acf, config)
        r = subprocess.run(
            [os.path.expanduser("~/cuda-13.3.1/bin/nvcc"),
             "-gencode", "arch=compute_90a,code=sm_90a", "-O3",
             f"-Xptxas=--apply-controls={acf}",
             "-I/usr/local/cuda-13.1/targets/x86_64-linux/include",
             "-L/usr/local/cuda-13.1/lib64", "-lcuda",
             "-o", exe, os.path.expanduser("~/bw24-unified/tools/bench_q8_gemm_wgmma.cu")],
            capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return INVALID
        with open("/tmp/ciq_gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            out = subprocess.run([exe], capture_output=True, text=True,
                                 timeout=300).stdout
        if "FAIL" in out or "MISMATCH" in out or "BAD" in out:
            return INVALID
        m = re.findall(r"MMQ +([0-9]+)us", out)
        if len(m) != 6:
            return INVALID
        return sum(float(x) for x in m)
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
        dump_results=pathlib.Path("~/acf-q8-results.csv").expanduser(),
        disable_progress_bar=True,
    )
    res = search.start(num_workers=5)
    best = res.get_best_result()
    print("BEST score:", best["score_1"])
    save_compiler_config(str(pathlib.Path("~/acf-q8-best.acf").expanduser()), best["params"])
    print("saved ~/acf-q8-best.acf")
