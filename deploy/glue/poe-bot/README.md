# Poe research-preview shim

The shim translates Poe server-bot requests into memra
`/v1/chat/completions` requests and streams visible content back through the
Poe protocol. It can run on the two-GPU pod or on a small separate VM.

Properties fixed for the trial:

- keyed Poe protocol endpoint at `/poe`;
- dedicated keyed memra backend;
- one concurrent request;
- 60,000 input characters across at most 64 messages;
- 512 output tokens;
- text only, no attachments or tools;
- no Poe monetization rate card;
- memra reasoning excluded from the visible stream;
- per-conversation hashed `cache_salt` and `session_id`;
- no prompt or response logging in the shim.

Install and validate:

```bash
deploy/glue/poe-bot/run.sh --dry-run
sudo deploy/glue/poe-bot/setup.sh
sudoedit /etc/memra/poe-bot.env
sudo deploy/glue/poe-bot/run.sh --check
```

Registration and TLS routing are in [REGISTRATION.md](REGISTRATION.md).

Run the mock-backed tests:

```bash
python3 -m venv /tmp/memra-poe-test
/tmp/memra-poe-test/bin/pip install -r deploy/glue/poe-bot/requirements-dev.txt
PYTHONPATH=deploy/glue/poe-bot \
  /tmp/memra-poe-test/bin/pytest -q deploy/glue/poe-bot/tests
```
