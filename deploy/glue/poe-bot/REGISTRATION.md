# Poe server-bot registration

This shim uses Poe's server-bot protocol through `fastapi-poe`, not Poe's
OpenAI-compatible API Bot import path. The protocol package and Creator docs
were checked on 2026-08-08; the pinned package is `fastapi-poe 0.0.83`.

## 1. Configure the host

After the RunPod provisioner has installed memra:

```bash
sudo /opt/memra/deploy/glue/poe-bot/setup.sh
openssl rand -hex 16
sudoedit /etc/memra/poe-bot.env
```

Put the 32-character `openssl` result in `POE_ACCESS_KEY`. Mint a dedicated
memra key with one concurrent request and put it in
`MEMRA_POE_BACKEND_KEY`:

```bash
sudo memra-server --gen-key poe \
  --lane interactive \
  --rate-limit 1 \
  --keys /etc/memra/keys.toml
```

Then start the unit and verify local liveness:

```bash
sudo systemctl start memra-poe-bot.service
curl --fail http://127.0.0.1:8081/poe/livez
```

## 2. Publish the Poe path over TLS

On the same pod, add a more-specific Cloudflare Tunnel published-application
route ahead of the catch-all memra route:

```text
Hostname: api.tiyuvta.ai
Path:     ^/poe(/.*)?$
Service:  http://127.0.0.1:8081
```

The existing catch-all `api.tiyuvta.ai` route continues to send `/v1`,
`/readyz`, and `/metrics` to port `8002`. Reusing the hostname adds no new DNS
record. Verify:

```bash
curl --fail https://api.tiyuvta.ai/poe/livez
```

On a tiny VM, set `MEMRA_POE_BACKEND_URL=https://api.tiyuvta.ai`, keep the
shim bound to loopback behind that VM's TLS proxy, and publish its `/poe` path.

## 3. Register in Poe

1. Open [Create a bot](https://poe.com/create_bot) while signed in.
2. Choose a **Server bot**.
3. Set the server URL to `https://api.tiyuvta.ai/poe`.
4. Enter the exact same 32-character `POE_ACCESS_KEY`.
5. Keep the bot private for the smoke test, then make it public for the trial.
6. Use the trial description:
   `Research preview served by memra on rented hardware; may go offline after the probe.`
7. Leave monetization/rate-card fields unset. `get_settings()` also returns no
   rate card or cost label, so this is a free bot.
8. Leave attachments, image comprehension, tools, and parameter controls off;
   the shim advertises the same restrictions.

Send one short prompt, one multi-turn prompt, and one prompt while a direct API
request occupies the Poe key's only slot. Confirm streaming output and the
expected retryable busy response.

When the pod retires, set the Poe bot private before stopping the shim. A public
bot must not be left pointing at a dead 48-hour endpoint.

Current official references:

- [FastAPI Poe quick start](https://creator.poe.com/docs/server-bots/quick-start)
- [Poe server-bot settings](https://creator.poe.com/docs/server-bots/updating-bot-settings)
- [fastapi-poe source](https://github.com/poe-platform/fastapi_poe)
