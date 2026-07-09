# MTP acceptance: ceiling (bf16 full-prec) vs quant (NVFP4)  [median of N runs]
# ceiling=research/mtp-heal/out-bf16.jsonl  quant=research/mtp-heal/out-nvfp4.jsonl
prompt          K    bf16   nvfp4   delta   hit% consist
------------------------------------------------------------
p1              1   0.841   0.868  -0.027   -3.2 PASS
p1              2   0.796   0.790  +0.006   +0.8 PASS
p1              3   0.683   0.674  +0.009   +1.3 PASS
p1              4   0.530   0.583  -0.053  -10.0 PASS
p2              1   0.753   0.789  -0.036   -4.8 PASS
p2              2   0.585   0.634  -0.049   -8.4 PASS
p2              3   0.455   0.513  -0.058  -12.7 PASS
p2              4   0.387   0.472  -0.085  -22.0 PASS
p3              1   0.969   0.829  +0.140  +14.4 PASS
p3              2   0.911   0.740  +0.171  +18.8 PASS
p3              3   0.615   0.651  -0.036   -5.9 PASS
p3              4   0.461   0.571  -0.110  -23.9 PASS
agentloop-t1    1   0.947   0.917  +0.030   +3.2 PASS
agentloop-t2    1   0.984   0.947  +0.037   +3.8 PASS
agentloop-t3    1   1.000   0.977  +0.023   +2.3 PASS
agentloop-t4    1   0.984   0.962  +0.022   +2.2 PASS
agentloop-t5    1   0.969   0.947  +0.022   +2.3 PASS
agentloop-t6    1   0.992   0.947  +0.045   +4.5 PASS
agentloop-t7    1   0.992   0.939  +0.053   +5.3 PASS
agentloop-t8    1   1.000   0.925  +0.075   +7.5 PASS

[wrote research/mtp-heal/delta-summary.json]
