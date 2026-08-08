# cx-models OpenRouter listing progress

Fetch/work date: **2026-08-07**
Branch: `lane/cx-or-models`
Train tip: `6afc4f65`

## Research

Official sources fetched on 2026-08-07:

- https://openrouter.ai/docs/guides/community/for-providers
- https://openrouter.ai/docs/assets/provider-monitor-schema-v2.openapi.json
- https://openrouter.ai/docs/guides/community/for-providers-legacy
- https://openrouter.ai/providers/apply

The guide Markdown and OpenAPI asset were fetched directly at 22:28 UTC. The schema
response reported `Last-Modified: Fri, 07 Aug 2026 21:29:01 GMT`; the downloaded JSON
was 62,681 bytes with SHA-256
`ca70e3e89dffc9060768e55d55fb46885c605b69682a86340ec4396249459d1e`.

Local owner context reviewed:

- `/home/avifenesh/projects/darklanes/exp/fast-entry-48h-20260808.md`
- `/home/avifenesh/projects/darklanes/exp/provider-table-stakes-20260806.md`
- `research/or-provider-20260802/REPORT.md`

What the current sources say:

- The machine-readable Provider Monitor document is still version **2.4**.
- Its current `ModelDocumentV2` is the typed-modality format, not the earlier flat
  provider document and not the public catalog shape with `architecture`,
  `top_provider`, or `per_request_limits`.
- Required model fields are `schema_version`, `id`, `name`, `input_modalities`, and
  `output_modalities`; model documents set `additionalProperties: false`.
- Prompt/cache prices belong on the text input modality, completion/reasoning prices
  on the text output modality, and request-scoped prices at the model root.
- Prices are per-unit USD decimal strings. The guide says to omit unbilled SKUs rather
  than fill them with `"0"`; explicit zero is for a genuinely free SKU.
- The old flat format remains documented only for existing integrations. New provider
  integrations are directed to the typed format.

Because the strict model schema rejects the existing OpenAI `object` field, a default
superset would not validate. The implementation therefore uses the owner-approved
query variant: the existing `/models` response stays byte-identical, and the application
URL is `/models?schema=openrouter`.

## Implementation

Commit `883c531b` (`feat(server): add OpenRouter provider model listing`):

- Preserves default `GET /models` and existing `GET /v1/models`.
- Adds current Provider Monitor 2.4 output at
  `GET /models?schema=openrouter`.
- Derives model id/name from the `MEMRA_MODELS` alias, context/tokenizer from the
  loaded plan, and supported parameters from memra's real HTTP/template capabilities.
- Adds optional `MEMRA_MODEL_METADATA=/path/models.toml`.
- Fails before GPU load on unknown TOML fields, invalid prices/quantization/limits, or
  metadata aliases absent from `MEMRA_MODELS`.
- Omits undeclared prices, capacities, and optional fields rather than serializing
  nulls or invented zero prices.

The 48-hour probe prices convert to these per-token strings:

| SKU | displayed price | schema string |
|---|---:|---:|
| prompt | $0.234 / 1M tokens | `"0.000000234"` |
| cached prompt | 25% of prompt | `"0.0000000585"` |
| completion | $1.872 / 1M tokens | `"0.000001872"` |

## Verification

Required crate gates:

```text
cargo check -p memra-server
PASS — dev profile finished; sm_120a auto-detected

cargo test -p memra-server
PASS — 115 passed, 0 failed
```

The captured live Provider Monitor entry also passed `jsonschema` validation against
the directly downloaded `ModelDocumentV2` schema:

```text
PASS: 1 live model entry validates against Provider Monitor schema 2.4
```

The RTX 5090 Laptop was idle enough for a live probe: 0% utilization and 768 MiB /
24,463 MiB used. Existing `llama-server` and Hermes processes were left running.
memra loaded the local 5.3 GiB Qwen3.5 9B GGUF with `MEMRA_SERVE_SPEC=0` on port
`18080`.

Default compatibility response:

```json
{"object":"list","data":[{"id":"probe","object":"model"}]}
```

Live Provider Monitor response:

```json
{"data":[{"schema_version":"2.4","id":"probe","name":"probe","quantization":"nvfp4","tokenizer":"qwen35","description":"Qwen3.5 9B live schema probe.","input_modalities":[{"type":"text","supported_inputs":{"max_context_length":{"value":262144,"unit":"token"},"max_prompt_length":{"value":245760,"unit":"token"}},"pricing":[{"type":"prompt","unit":"token","cost_usd":"0.000000234"},{"type":"cached_prompt","unit":"token","cost_usd":"0.0000000585"}]}],"output_modalities":[{"type":"text","supported_parameters":{"temperature":{"type":"unknown"},"top_p":{"type":"unknown"},"min_p":{"type":"unknown"},"frequency_penalty":{"type":"unknown"},"presence_penalty":{"type":"unknown"},"repetition_penalty":{"type":"unknown"},"stop":{"type":"unknown"},"top_k":{"type":"integer","min":0},"seed":{"type":"integer","min":0,"max":9007199254740991},"max_tokens":{"type":"integer","min":1,"unit":"token","max":16384},"json_mode":{"type":"boolean"},"structured_outputs":{"type":"boolean"},"tools":{"type":"boolean"},"tool_choice":{"type":"enum","values":["auto","none"]},"reasoning":{"type":"boolean"}},"streaming":true,"max_length":{"value":16384,"unit":"token"},"pricing":[{"type":"completion","unit":"token","cost_usd":"0.000001872"}]}],"is_ready":true}]}
```

After shutdown, GPU memory returned to 768 MiB and port `18080` was closed. No nsys
artifacts were produced and nothing was pushed.
