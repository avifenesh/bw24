# AI Horde text worker

This directory connects AI Horde text jobs to memra's keyed OpenAI-compatible
endpoint. It advertises the fixed trial model name `memra-research-preview` and
sends generation to `stepfun/step-3.7-flash` on
`http://127.0.0.1:8002/v1/completions`.

The bridge selection and exact current-source audit are in
[UPSTREAM.md](UPSTREAM.md). The integration is pinned because the selected
project has no release artifacts.

## Account and kudos

1. Register a permanent AI Horde account at
   [aihorde.net/register](https://aihorde.net/register). Do not use the anonymous
   `0000000000` key.
2. Copy the account API key. Jobs completed by this worker earn kudos for that
   account; kudos are AI Horde's internal accounting/reputation unit, not a
   payment credential.
3. Give the worker a stable, unique name. The public model name remains
   `memra-research-preview`.

The live API heartbeat and worker schema were checked at
`https://aihorde.net/api/v2` on 2026-08-08.

## Pod setup

Run this after `deploy/runpod/provision.sh` has installed memra, created the
`memra` system user, staged Step-3.7-Flash, and installed the keyring:

```bash
sudo /opt/memra/deploy/glue/horde-worker/setup.sh
```

Mint one dedicated backend key. The per-key `rate-limit` is a concurrent-request
cap, so `1` matches the bridge's single worker thread:

```bash
sudo memra-server --gen-key horde \
  --lane batch \
  --rate-limit 1 \
  --keys /etc/memra/keys.toml
```

Edit `/etc/memra/horde-worker.env` and replace both placeholder secrets:

```text
AI_HORDE_API_KEY=<permanent account key>
MEMRA_HORDE_BACKEND_KEY=<new mk-... key>
```

Keep that file mode `0600`. Then validate without contacting either service:

```bash
sudo -u memra /opt/memra/deploy/glue/horde-worker/run.sh --dry-run
sudo systemctl start memra-horde-worker.service
sudo systemctl status memra-horde-worker.service
```

`trial-up.sh` owns the actual trial start. The setup script enables the unit but
does not start it.

## Admission posture

The bridge deliberately advertises less than the server can provide:

- one concurrent Horde job;
- 16,384-token context;
- 512-token maximum output;
- five-second idle polling;
- prompt logging disabled;
- NSFW jobs disabled by default;
- local regex CSAM prefilter enabled.

The regex filter is only a bridge-side guard and is not a comprehensive
moderation system. The owner can change `MEMRA_HORDE_NSFW` after making the
trial policy decision. Do not increase threads or context merely because the
model accepts it: the dedicated memra key, global admission controller, and
bridge cap must remain aligned.

Inspect the registered worker after its first poll:

```bash
curl --fail --silent --show-error \
  "https://aihorde.net/api/v2/workers/name/memra-research-preview-runpod" |
  jq '{id,name,online,maintenance_mode,requests_fulfilled,kudos_rewards,models}'
```

The offline dry run renders and validates the complete config but permits
placeholder secrets. `run.sh --check` is stricter: it requires the pinned
installation, real secrets, and a mode-`0600` rendered config, while still
making no network connection.
