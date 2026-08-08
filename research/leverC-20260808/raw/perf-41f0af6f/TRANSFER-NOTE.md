# Lever C performance receipt transfer

The Box2 campaign completed before transfer:

- summary: `perf-summary-20260808T112734Z.log`
- result: `rc=0`
- timed arms: 30/30 exit zero
- remote source: `/tmp/leverC-perf-41f0af6f/`

The first recursive copy was interrupted when the Box2 SSH endpoint stopped accepting
connections. Thirty-one generated files arrived directly. These three small arm logs did not:

- `pp512-grouped-r3-20260808T112734Z.log`
- `pp2048-off-r3-20260808T112734Z.log`
- `pp2048-grouped-r5-20260808T112734Z.log`

Each arm log had already been copied verbatim into the aggregate summary by the committed driver.
The missing files were extracted from their marker-delimited sections. Before recovery, the same
extractor was run against `pp512-off-r1-20260808T112734Z.log`; its output matched the directly
copied original byte-for-byte and had the same SHA-256
`37bb8599b4f1902a3d90d30a2f3cda61a6b0f1b02e3dae94c5a63d5b74cfb319`.

The pp4096 prompt was restored from the tracked identical source
`research/step-sku-20260807/prompt-pp4096.txt`. Its SHA-256 matches the remote receipt:
`23c1d8384a16c7c0bcb7736b412d43e64c0b4d8e238703864e928565f824ae11`.

`SHA256SUMS` covers every retained receipt in this directory except itself.
