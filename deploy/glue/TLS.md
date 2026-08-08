# Trial TLS and tunnel

The preferred public edge is a remotely managed Cloudflare Tunnel terminating TLS for
`api.tiyuvta.ai` and forwarding to memra on `http://127.0.0.1:8002`. The connector makes
an outbound connection, so the pod does not need a public TCP port for this path.

## Cloudflare path

The one dashboard-side DNS action is:

1. In Cloudflare Zero Trust, create or select a remotely managed tunnel.
2. In that tunnel, add one **Published application** route:
   - Hostname: `api.tiyuvta.ai` (or the selected `*.tiyuvta.ai` subdomain)
   - Service: `http://127.0.0.1:8002`

Saving the route creates the proxied DNS record. Copy the connector token shown for that
tunnel. This is the only secret needed by the pod-side installer.

On the provisioned pod:

```bash
sudo install -m 0600 /dev/null /etc/memra/cloudflared-token
sudoedit /etc/memra/cloudflared-token
sudo CLOUDFLARED_TOKEN_FILE=/etc/memra/cloudflared-token \
  deploy/glue/cloudflared-setup.sh
deploy/glue/cloudflared-setup.sh --check
```

The installer is idempotent. If `deploy/runpod/provision.sh` already installed the
connector service, it preserves that service and only enables/checks it. A new install is
pinned to cloudflared `2026.7.3` and verifies Cloudflare's published package checksum.
`--check` requires all of the following:

- `cloudflared.service` is active.
- The loopback `/readyz` endpoint responds.
- The public hostname resolves.
- Public `/readyz` and `/v1/models` respond over verified HTTPS.

Do not place the connector token in shell history. The route should target the origin root,
not `/v1`, because health and metrics use sibling paths.

Current-source references, checked 2026-08-08:

- [Create a remotely managed tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/)
- [Route a public hostname to a tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/routing-to-tunnel/dns/)
- [Run cloudflared as a Linux service](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/local-management/as-a-service/linux/)
- [cloudflared 2026.7.3 release](https://github.com/cloudflare/cloudflared/releases/tag/2026.7.3)

## RunPod proxy fallback

If the Cloudflare hostname is not ready, RunPod's HTTP proxy can expose port `8002` as:

```text
https://<pod-id>-8002.proxy.runpod.net
```

Use the RunPod provisioner's proxy exposure mode rather than adding another server. That
mode binds memra to `0.0.0.0:8002`; the Cloudflare path deliberately keeps it on
`127.0.0.1:8002`. Expose HTTP port `8002` in the pod configuration before launch.

This fallback is suitable for short smoke traffic, not the preferred trial edge:

- RunPod documents a 100-second HTTP proxy timeout, so generation clients must stream and
  long first-token stalls can still fail at the proxy.
- The proxy hostname is tied to the pod id and is not the stable `tiyuvta.ai` API address.
- The whole HTTP service is public. memra protects `/v1` with API keys, but `/metrics` is
  intentionally unauthenticated and should not be left exposed through this path.
- RunPod's proxy has request-size and bandwidth limits outside memra's admission controls.

Reference checked 2026-08-08:
[Expose ports on RunPod](https://docs.runpod.io/pods/configuration/expose-ports).
