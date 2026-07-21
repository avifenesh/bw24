#!/usr/bin/env python3
"""CPU-only path-validation regression tests for the Hy3 NVMe builders."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


def load_tool(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NvmeBuilderPathSafetyTests(unittest.TestCase):
    def test_payload_names_are_strictly_relative_expert_paths(self) -> None:
        for module_name in ("build_dual_nvme_expert_view", "build_expert_mirror_map"):
            module = load_tool(module_name)
            self.assertEqual(
                module.payload_name("experts/layer-000.bin"),
                "experts/layer-000.bin",
            )
            for unsafe in (
                "../outside",
                "experts/../../outside",
                "/tmp/outside",
                "experts//outside",
                "./outside",
            ):
                with self.subTest(module=module_name, path=unsafe):
                    with self.assertRaises(ValueError):
                        module.payload_name(unsafe)

    def test_contained_path_rejects_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bw24-nvme-path-test-") as temp:
            root = Path(temp) / "artifact"
            root.mkdir()
            for module_name in ("build_dual_nvme_expert_view", "build_expert_mirror_map"):
                module = load_tool(module_name)
                with self.subTest(module=module_name):
                    with self.assertRaises(ValueError):
                        module.contained_path(root, "experts/../../outside")

    def test_mirror_map_v2_pins_both_file_generations(self) -> None:
        module = load_tool("build_expert_mirror_map")
        source = (11, 22, 33, 44, 55)
        alternate = (66, 77, 33, 88, 99)
        row = module.format_map_row(source, alternate, Path("/mirror/expert.bin"))
        self.assertEqual(
            row,
            "11\t22\t33\t44\t55\t66\t77\t33\t88\t99\t/mirror/expert.bin\n",
        )
        self.assertEqual(len(row.rstrip("\n").split("\t")), 11)


if __name__ == "__main__":
    unittest.main()
