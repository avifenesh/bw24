#!/usr/bin/env python3
"""CPU-only tests for the agent-shaped fleet replay driver."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest


ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "fleet-replay.py"


def load_replay():
    spec = importlib.util.spec_from_file_location("fleet_replay", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPLAY = load_replay()


class ReplayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), ReplayHandler)
        self.records: list[dict] = []
        self.records_lock = threading.Lock()
        self.received = threading.Event()


class ReplayHandler(BaseHTTPRequestHandler):
    server: ReplayHTTPServer

    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        with self.server.records_lock:
            index = len(self.server.records) + 1
            reply = json.dumps({
                "tool": "read_file",
                "arguments": {"path": f"turn-{index}.txt"},
            })
            record = {
                "body": body,
                "authorization": self.headers.get("Authorization"),
                "replay_label": self.headers.get("X-Memra-Replay"),
                "reply": reply,
            }
            self.server.records.append(record)
            self.server.received.set()

        prompt_tokens = REPLAY.estimate_tokens(body["prompt"])
        completion_tokens = body["max_tokens"]
        payload = {
            "id": f"cmpl-fixture-{index}",
            "object": "text_completion",
            "model": body["model"],
            "choices": [{
                "index": 0,
                "text": reply,
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": 0 if index == 1 else prompt_tokens // 2,
                },
            },
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args) -> None:
        pass


@contextmanager
def running_server():
    server = ReplayHTTPServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def command(server: ReplayHTTPServer, *extra: str) -> list[str]:
    host, port = server.server_address
    return [
        sys.executable,
        str(SCRIPT),
        "--base",
        f"http://{host}:{port}",
        "--model",
        "fixture-model",
        "--api-key",
        "fixture-key",
        "--timeout",
        "2",
        *extra,
    ]


class FleetReplayTests(unittest.TestCase):
    def test_prefix_template_pool_is_realistic_size(self) -> None:
        templates = REPLAY.PREFIX_TEMPLATES
        self.assertGreaterEqual(len(templates), 5)
        self.assertLessEqual(len(templates), 10)
        for template in templates:
            tokens = REPLAY.estimate_tokens(template.text)
            self.assertGreaterEqual(tokens, 1000, template.name)
            self.assertLessEqual(tokens, 4000, template.name)
            self.assertIn("<tools>", template.text)
            self.assertIn('"additionalProperties": false', template.text)

    def test_mock_server_sees_ratio_salt_and_conversation_carry(self) -> None:
        with running_server() as server:
            result = subprocess.run(
                command(
                    server,
                    "--duration",
                    "10",
                    "--requests",
                    "2",
                    "--requests-per-minute",
                    "60000",
                    "--sessions",
                    "1",
                    "--tenants",
                    "1",
                    "--seed",
                    "7",
                ),
                check=True,
                capture_output=True,
                text=True,
                timeout=8,
            )

            summary = json.loads(result.stdout.strip())
            self.assertEqual(summary["label"], "replay-calibrated")
            self.assertEqual(summary["requests_ok"], 2)
            self.assertEqual(summary["stop_reason"], "request-limit")

            with server.records_lock:
                records = list(server.records)
            self.assertEqual(len(records), 2)
            first, second = records
            self.assertEqual(first["authorization"], "Bearer fixture-key")
            self.assertEqual(first["replay_label"], "replay-calibrated")
            self.assertEqual(
                first["body"]["cache_salt"],
                "replay-calibrated-tenant-01",
            )
            self.assertEqual(first["body"]["session_id"], "replay-session-001")
            self.assertEqual(
                first["body"]["max_tokens"],
                REPLAY.completion_budget(
                    REPLAY.estimate_tokens(first["body"]["prompt"]),
                    REPLAY.DEFAULT_PROMPT_COMPLETION_RATIO,
                ),
            )
            self.assertTrue(
                second["body"]["prompt"].startswith(
                    first["body"]["prompt"] + first["reply"]
                )
            )
            self.assertEqual(
                second["body"]["cache_salt"],
                first["body"]["cache_salt"],
            )
            self.assertEqual(
                second["body"]["session_id"],
                first["body"]["session_id"],
            )

    def test_sigterm_stops_during_poisson_wait(self) -> None:
        with running_server() as server:
            process = subprocess.Popen(
                command(
                    server,
                    "--duration",
                    "30",
                    "--requests-per-minute",
                    "0.01",
                    "--sessions",
                    "1",
                    "--tenants",
                    "1",
                    "--seed",
                    "11",
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertTrue(server.received.wait(timeout=4))
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)

            self.assertEqual(process.returncode, 0, stderr)
            summary = json.loads(stdout.strip())
            self.assertEqual(summary["label"], "replay-calibrated")
            self.assertEqual(summary["stop_reason"], "signal")
            self.assertEqual(summary["requests_ok"], 1)


if __name__ == "__main__":
    unittest.main()
