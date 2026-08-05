#!/usr/bin/env python3
"""Hold N MiB of VRAM until killed (F5 ladder-path test ballast)."""
import ctypes, ctypes.util, sys, time

mib = int(sys.argv[1])
for cand in ("libcudart.so", "libcudart.so.13", "libcudart.so.12",
             "/usr/local/cuda/lib64/libcudart.so"):
    try:
        rt = ctypes.CDLL(cand)
        break
    except OSError:
        continue
else:
    sys.exit("no libcudart")
p = ctypes.c_void_p()
rc = rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(mib * 1024 * 1024))
if rc != 0:
    sys.exit(f"cudaMalloc({mib}MiB) rc={rc}")
print(f"ballast holding {mib}MiB", flush=True)
while True:
    time.sleep(60)
