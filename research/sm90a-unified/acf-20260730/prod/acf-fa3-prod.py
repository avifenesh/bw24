# ACF search against the PRODUCTION fa3_prefill TU (carry-over 8, 2026-07-31).
# Objective: acf_fa3_runner linked with fa3_prefill.cu under --apply-controls.
# Candidates must reproduce the baseline output fingerprint (scheduling-only gate).
import pathlib
from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.utils.helpers import save_compiler_config

BASE_FP = "1.975509e+05"

def objective(config):
    import subprocess, re, os, fcntl
    from uuid import uuid4
    from compileiq.utils.helpers import save_compiler_config
    INVALID = "*"
    acf = f"/tmp/ciq_{uuid4().hex}.acf"
    exe = f"/tmp/ciq_prod_{uuid4().hex}"
    try:
        save_compiler_config(acf, config)
        r = subprocess.run(
            [os.path.expanduser("~/cuda-13.3.1/bin/nvcc"),
             "-std=c++17", "-O3", "-gencode", "arch=compute_90a,code=sm_90a",
             "--expt-relaxed-constexpr", f"-Xptxas=--apply-controls={acf}",
             "-I/usr/local/cuda-13.1/targets/x86_64-linux/include",
             "-L/usr/local/cuda-13.1/lib64", "-lcuda", "-o", exe,
             os.path.expanduser("~/bw24-unified/tools/acf_fa3_runner.cu"),
             os.path.expanduser("~/bw24-unified/crates/bw24-engine/cu/fa3_prefill.cu")],
            capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return INVALID
        with open("/tmp/ciq_gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            out = subprocess.run([exe, "2048"], capture_output=True, text=True,
                                 timeout=300).stdout
        if f"fingerprint {BASE_FP}" not in out:
            return INVALID
        m = re.findall(r"T=2048: (\d+)us/call", out)
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
        search_config={"pool_size": 24, "generations": 6, "mutate_rate": 0.3,
                       "problem_type": "min", "num_objectives": 1},
        dump_results=pathlib.Path("~/acf-fa3-prod-results.csv").expanduser(),
        disable_progress_bar=True,
    )
    res = search.start(num_workers=6)
    best = res.get_best_result()
    print("BEST score:", best["score_1"])
    save_compiler_config(str(pathlib.Path("~/acf-fa3-prod-best.acf").expanduser()), best["params"])
    print("saved ~/acf-fa3-prod-best.acf")
