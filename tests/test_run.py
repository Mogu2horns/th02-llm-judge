"""Regression checks for resource gates and exclusive output ownership."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from th02.run import execute, output_lock, verify_manifest


class RunTests(unittest.TestCase):
    def test_manifest_preserved(self):
        self.assertTrue(all(check["matches"] for check in verify_manifest()))

    def test_exclusive_output_lock(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with output_lock(Path(directory)):
                with self.assertRaises(OSError):
                    with output_lock(Path(directory)):
                        self.fail("A second writer acquired the same output directory")
            with output_lock(Path(directory)):
                pass

    def test_full_run_requires_six_gib(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            args = SimpleNamespace(smoke=False, device="cuda", preflight_only=True)
            resources = {"ram_available_gib": 0.5, "gpu": {"memory_free_mib": 4096}}
            with patch("th02.run.inspect_resources", return_value=resources):
                code = execute(args, Path(directory), {"seed": 1})
            self.assertEqual(code, 2)
            state = json.loads((Path(directory) / "run_state.json").read_text())
            self.assertEqual(state["status"], "blocked_resources")
            self.assertFalse((Path(directory) / "metrics.json").exists())

    def test_preflight_passes_with_sufficient_resources(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            args = SimpleNamespace(smoke=False, device="cuda", preflight_only=True)
            resources = {"ram_available_gib": 8, "gpu": {"memory_free_mib": 4096}}
            with patch("th02.run.inspect_resources", return_value=resources):
                self.assertEqual(execute(args, Path(directory), {"seed": 1}), 0)

    def test_smoke_exempts_full_ram_gate(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            args = SimpleNamespace(smoke=True, device="cpu", preflight_only=True)
            with patch("th02.run.inspect_resources", return_value={"ram_available_gib": 0.5}):
                self.assertEqual(execute(args, Path(directory), {"seed": 1}), 0)


if __name__ == "__main__":
    unittest.main()
