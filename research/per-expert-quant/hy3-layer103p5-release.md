# Hy3 Layer103.5 release

`layer103-late20` is the smallest tested Layer100-preserving complement that improved the matched
115-question directional screen. It restores 262 experts only in layers 60-79 and leaves every
Layer100 expert choice and projection quantization unchanged.

| arm | logical bytes | math | code | history | other | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer100 matched | 99,999,322,624 | 27 | 32 | 3 | 11 | 73/115 |
| Layer103.5 late20 | 103,489,802,752 | 29 | 32 | 4 | 11 | 76/115 |

The paired comparison is 5 wins, 2 losses, and 108 ties. This is directional evidence, not a claim
of statistical significance: the paired exact-sign p-value is 0.453125 and the domain-macro
bootstrap interval crosses zero. Public capability results were not used to construct or heal the
arm; selection used private routing displacement and layer position only.

The published artifact is a `bw24-expert-overlay-v2`, not a Transformers checkpoint. It contains
78,490,288,128 bytes of expert payloads and resolves non-expert tensors from `tencent/Hy3` revision
`716aa7241bd6d95896be4ebfc761162a9c4d49ef`. The model and receipts are published at
[Avifenesh/Hy3-REAP-Layer103p5-bw24](https://huggingface.co/Avifenesh/Hy3-REAP-Layer103p5-bw24).

Use `tools/relocate_hy3_expert_overlay.py` to create a zero-copy runtime view that points the
immutable published overlay at a local copy of the pinned source checkpoint.

Receipt anchors:

- source plan: `9606b1b96890b270534237b1143a5f5f25165245d1b5f08f515c25268d1b056c`
- artifact manifest: `08f206aed555752982585a59a7b5096b9cc6e71faf1f84ad5c6dd60476b7509a`
- screen summary: `661e495c02467acf5f28180eb73d562d25fea8724449e69f305831beb435be3d`
- winner receipt: `4c6d4089d5dec428a5575b8c9971521f1f04ed5041e48e02bda67d3dae993ab3`
