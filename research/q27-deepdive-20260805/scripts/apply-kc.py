import sys
p = "/root/bw24/crates/memra-engine/src/bin/kernel_check.rs"
s = open(p).read()
# NOTE: anchor with the leading newline — the 12-space line is a SUBSTRING of the
# 16-space fused3 line, so a bare match counts 2.
old = "\n            for mm in [2usize, 3, 4] {"
new = "\n            for mm in [2usize, 3, 4, 5, 8] {   // 5,8 = SERVING tier (fused2_b8), q27-deepdive"
if new in s:
    print("SKIP already"); sys.exit(0)
n = s.count(old)
if n != 1:
    print("ANCHOR-FAIL count=%d" % n); sys.exit(1)
open(p, "w").write(s.replace(old, new)); print("OK kernel_check")
