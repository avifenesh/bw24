#!/usr/bin/env python3
"""CPU-only tests for the opencode SFT trace generator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "sft-gen.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("sft_gen", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SftGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_tool()

    def test_battery_has_six_shapes_and_four_task_kinds(self) -> None:
        templates = self.tool.build_templates()
        self.assertEqual(len(templates), 24)
        self.assertEqual(
            {template.category for template in templates},
            {"bug_fix", "refactor", "test_writing", "explain"},
        )
        counts = {}
        for template in templates:
            counts[template.category] = counts.get(template.category, 0) + 1
            self.assertTrue(template.source_paths)
            self.assertIn("CPU-only", template.prompt)
        self.assertEqual(set(counts.values()), {6})

    def test_all_fixture_sources_compile_and_scan_clean(self) -> None:
        templates = self.tool.build_templates()
        self.tool.scan_templates(templates)
        for template in templates:
            for path, content in template.files.items():
                compile(content, path, "exec")

    def test_secret_scanner_detects_assigned_and_prefixed_keys(self) -> None:
        self.assertTrue(
            self.tool.scan_text("fixture", 'api_key = "not-a-real-secret-value"')
        )
        self.assertTrue(
            self.tool.scan_text("fixture", "sk-abcdefghijklmnopqrstuvwxyz123456")
        )
        self.assertFalse(self.tool.scan_text("fixture", "api_key_name = None"))

    def test_content_hash_ignores_ids_times_and_usage(self) -> None:
        template = self.tool.build_templates()[0]
        first = {
            "info": {"directory": "/tmp/memra-sft-first"},
            "messages": [
                {
                    "info": {"role": "assistant", "id": "message-a"},
                    "parts": [
                        {
                            "type": "text",
                            "text": "read /tmp/memra-sft-first/module.py",
                            "id": "part-a",
                            "time": {"start": 1, "end": 2},
                        },
                        {"type": "step-finish", "tokens": {"input": 10}, "cost": 0.1},
                    ],
                }
            ]
        }
        second = {
            "info": {"directory": "/tmp/memra-sft-second"},
            "messages": [
                {
                    "info": {"role": "assistant", "id": "message-b"},
                    "parts": [
                        {
                            "type": "text",
                            "text": "read /tmp/memra-sft-second/module.py",
                            "id": "part-b",
                            "time": {"start": 10, "end": 20},
                        },
                        {"type": "step-finish", "tokens": {"input": 99}, "cost": 9.9},
                    ],
                }
            ]
        }
        self.assertEqual(
            self.tool.content_hash(template, first),
            self.tool.content_hash(template, second),
        )
        second["messages"][0]["parts"][0]["text"] = "different"
        self.assertNotEqual(
            self.tool.content_hash(template, first),
            self.tool.content_hash(template, second),
        )

    def test_event_parser_requires_json_objects_and_one_session(self) -> None:
        events = self.tool.parse_json_events(
            '{"type":"text","sessionID":"s1","part":{"text":"ok"}}\n'
            '{"type":"step_finish","sessionID":"s1","part":{}}\n'
        )
        self.assertEqual(self.tool.session_id_from_events(events), "s1")
        with self.assertRaisesRegex(ValueError, "not JSON"):
            self.tool.parse_json_events("plain text")

    def test_prepare_workspace_accepts_precreated_empty_directory(self) -> None:
        path = Path(tempfile.mkdtemp(prefix="sft-gen-test-"))
        try:
            template = self.tool.build_templates()[0]
            self.tool.prepare_workspace(template, path)
            self.assertTrue((path / ".git").is_dir())
            self.assertTrue((path / template.module_name).is_file())
        finally:
            shutil.rmtree(path)

    def test_file_stdout_capture_preserves_large_single_write(self) -> None:
        result = self.tool.run_with_file_stdout(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 80000)"]
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(result.stdout), 80000)

    def test_opencode_config_requires_exact_provider_pin(self) -> None:
        model_id = "deepseek/deepseek-v4-flash-0731"
        config = {
            "provider": {
                "openrouter": {
                    "models": {
                        model_id: {
                            "name": "DeepSeek trace generator",
                            "options": {
                                "provider": {
                                    "order": list(self.tool.PROVIDER_ONLY),
                                    "only": list(self.tool.PROVIDER_ONLY),
                                    "allow_fallbacks": False,
                                },
                                "usage": {"include": True},
                            },
                        }
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory(prefix="sft-config-test-") as directory:
            path = Path(directory) / "opencode.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            receipt = self.tool.validate_opencode_config(
                path,
                f"openrouter/{model_id}",
            )
            self.assertEqual(receipt["provider"]["only"], list(self.tool.PROVIDER_ONLY))
            self.assertEqual(len(receipt["sha256"]), 64)

            config["provider"]["openrouter"]["models"][model_id]["options"]["provider"][
                "allow_fallbacks"
            ] = True
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "allow_fallbacks must be false"):
                self.tool.validate_opencode_config(path, f"openrouter/{model_id}")

    def test_tool_audit_rejects_rust_and_accelerator_commands(self) -> None:
        session = {
            "messages": [
                {
                    "parts": [
                        {
                            "type": "tool",
                            "tool": "bash",
                            "state": {"input": {"command": "cargo test && nvidia-smi"}},
                        }
                    ]
                }
            ]
        }
        self.assertEqual(
            self.tool.audit_tool_commands(session),
            ["bash: forbidden CPU/GPU command requested"],
        )


if __name__ == "__main__":
    unittest.main()
