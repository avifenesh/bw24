#!/usr/bin/env python3
"""CPU-only tests for the architecture gate scaffold generator."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "generate-arch-gates.py"


def valid_spec() -> dict:
    return {
        "id": "38",
        "artifact_env": "MEMRA_Q38_GGUF",
        "chunk": {
            "label": "qwen38-swa",
            "prompts": ["research/prompts/qwen38-long.txt"],
            "chunks": [4096, 513, 512, 256, 64],
            "steps": 24,
            "seam": "MEMRA_Q38_CHUNK_LEGACY",
        },
        "tick": {
            "label": "qwen38-tick",
            "prompts": ["research/prompts/qwen38-long.txt"],
            "budgets": [0, 1024, 513, 512, 256, 64],
            "splits": [64, 256, 512],
            "steps": 24,
            "seam": "MEMRA_Q38_CALLLOCAL",
        },
        "batch": {
            "model_alias": "qwen38",
            "draft_path": "/data/models/qwen38-draft.gguf",
            "draft_env": "MEMRA_Q38_DRAFT_GGUF",
            "canary_env": {"MEMRA_Q38_BATCH": "0"},
            "required_gpus": 2,
            "pp_stages": 2,
            "pp_devices": [0, 1],
            "concurrency": [2, 4],
            "port": 8094,
            "receipt_dir": "research/qwen38-batch/raw",
            "server_env": {
                "MEMRA_SERVE_B1FAST": "0",
                "MEMRA_SERVE_SPEC": "0",
            },
            "request": {
                "messages": [{"role": "user", "content": "Count to eight."}],
                "max_tokens": 48,
                "temperature": 0.0,
            },
            "liveness": {
                "cap_regex": "qwen38: decode chunk cap [0-9]+",
                "cap_min": 2,
                "walk_regex": "\\[qwen38-batch\\] first B>1",
            },
        },
        "mapping": [
            {
                "path_regex": (
                    "^crates/memra-engine/src/"
                    "(decode|decode_batch|forward|hybrid_forward)\\.rs$"
                ),
                "kernel_scope": "synthetic",
                "base_probes": ["g12", "q9", "q35"],
                "base_spec_probes": ["q35spec"],
                "gate_families": ["chunk", "tick", "batch"],
            },
            {
                "path_regex": "^crates/memra-server/",
                "kernel_scope": "none",
                "base_probes": ["sstress", "accept"],
                "base_spec_probes": [],
                "gate_families": ["tick", "batch"],
            },
        ],
    }


class ArchGateGeneratorTests(unittest.TestCase):
    def run_generator(
        self,
        directory: Path,
        spec: dict,
        *,
        extra: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        spec_path = directory / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "Qwen 3.8",
                "/data/models/Qwen3.8 27B.gguf",
                "--spec",
                str(spec_path),
                "--out-dir",
                "generated/qwen-3-8",
                *extra,
            ],
            cwd=directory,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_generates_scripts_and_registry_fragments(self) -> None:
        with tempfile.TemporaryDirectory(prefix="arch-gate-generator-") as temp:
            directory = Path(temp)
            result = self.run_generator(directory, valid_spec())
            self.assertEqual(result.returncode, 0, result.stderr)
            output = directory / "generated" / "qwen-3-8"

            scripts = sorted(output.glob("*.sh"))
            self.assertEqual(len(scripts), 3)
            for script in scripts:
                self.assertTrue(os.access(script, os.X_OK), script)
                content = script.read_text(encoding="utf-8")
                self.assertNotIn("{{", content)
                syntax = subprocess.run(
                    ["bash", "-n", str(script)],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(syntax.returncode, 0, syntax.stderr)

            missing_artifact = subprocess.run(
                [str(output / "qwen-3-8-chunk-invariance-gate.sh")],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(missing_artifact.returncode, 0)
            self.assertIn("SKIP", missing_artifact.stdout)

            model_rows = [
                row
                for row in (output / "fast-gate-models.tsv")
                .read_text(encoding="utf-8")
                .splitlines()
                if row and not row.startswith("#")
            ]
            self.assertEqual(len(model_rows), 6)
            self.assertEqual(
                {row.split("\t")[0] for row in model_rows},
                {
                    "chunkinv38",
                    "chunkinv38c",
                    "tickinv38",
                    "tickinv38c",
                    "b2geo38",
                    "b2geo38c",
                },
            )
            self.assertTrue(all(len(row.split("\t")) == 6 for row in model_rows))
            self.assertTrue(
                all("generated/qwen-3-8/" in row for row in model_rows)
            )

            map_rows = [
                row
                for row in (output / "fast-gate-map.tsv")
                .read_text(encoding="utf-8")
                .splitlines()
                if row and not row.startswith("#")
            ]
            self.assertEqual(len(map_rows), 2)
            self.assertTrue(all(len(row.split("\t")) == 4 for row in map_rows))
            engine_probes = map_rows[0].split("\t")[2].split(",")
            self.assertTrue(
                {
                    "chunkinv38",
                    "chunkinv38c",
                    "tickinv38",
                    "tickinv38c",
                    "b2geo38",
                    "b2geo38c",
                }.issubset(engine_probes)
            )
            server_probes = map_rows[1].split("\t")[2].split(",")
            self.assertNotIn("chunkinv38", server_probes)
            self.assertIn("tickinv38", server_probes)
            self.assertIn("b2geo38", server_probes)

            normalized = json.loads(
                (output / "gate-spec.json").read_text(encoding="utf-8")
            )
            self.assertEqual(normalized["architecture"], "Qwen 3.8")
            self.assertEqual(
                normalized["artifact"], "/data/models/Qwen3.8 27B.gguf"
            )
            self.assertEqual(len(normalized["generated_sha256"]), 5)

    def test_rejects_invalid_scientific_inputs(self) -> None:
        cases = []
        missing_seam = valid_spec()
        del missing_seam["chunk"]["seam"]
        cases.append((missing_seam, "missing required keys: seam"))

        bad_cap = valid_spec()
        bad_cap["batch"]["liveness"]["cap_regex"] = "qwen38 cap"
        cases.append((bad_cap, "must include a '[0-9]+' capture"))

        bad_pp = valid_spec()
        bad_pp["batch"]["pp_devices"] = [0]
        cases.append((bad_pp, "length must equal"))

        bad_device = valid_spec()
        bad_device["batch"]["pp_devices"] = [0, 2]
        cases.append((bad_device, "index outside"))

        shadowed_canary = valid_spec()
        shadowed_canary["batch"]["server_env"]["MEMRA_Q38_BATCH"] = "1"
        cases.append((shadowed_canary, "conflicts with server"))

        bad_port = valid_spec()
        bad_port["batch"]["port"] = 70000
        cases.append((bad_port, "at most 65535"))

        for spec, expected in cases:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory(
                    prefix="arch-gate-generator-invalid-"
                ) as temp:
                    result = self.run_generator(Path(temp), spec)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(expected, result.stderr)

    def test_refuses_overwrite_without_force(self) -> None:
        with tempfile.TemporaryDirectory(prefix="arch-gate-generator-force-") as temp:
            directory = Path(temp)
            first = self.run_generator(directory, valid_spec())
            self.assertEqual(first.returncode, 0, first.stderr)
            target = (
                directory
                / "generated"
                / "qwen-3-8"
                / "qwen-3-8-chunk-invariance-gate.sh"
            )
            target.write_text("sentinel\n", encoding="utf-8")

            refused = self.run_generator(directory, valid_spec())
            self.assertEqual(refused.returncode, 2)
            self.assertIn("refusing to overwrite", refused.stderr)
            self.assertEqual(target.read_text(encoding="utf-8"), "sentinel\n")

            forced = self.run_generator(directory, valid_spec(), extra=("--force",))
            self.assertEqual(forced.returncode, 0, forced.stderr)
            self.assertNotEqual(target.read_text(encoding="utf-8"), "sentinel\n")

    def test_rejects_output_path_with_whitespace(self) -> None:
        with tempfile.TemporaryDirectory(prefix="arch-gate-generator-path-") as temp:
            directory = Path(temp)
            spec_path = directory / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "Qwen 3.8",
                    "/data/models/qwen38.gguf",
                    "--spec",
                    str(spec_path),
                    "--out-dir",
                    "generated/has space",
                ],
                cwd=directory,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must not contain whitespace", result.stderr)


if __name__ == "__main__":
    unittest.main()
