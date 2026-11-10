"""Regression checks for resource gates and exclusive output ownership."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from th02.run import execute, inspect_resources, output_lock, verify_manifest


class RunTests(unittest.TestCase):
    def resource_mocks(self, cuda_available=True):
        torch = MagicMock()
        torch.cuda.is_available.return_value = cuda_available
        torch.cuda.current_device.return_value = 0
        torch.cuda.get_device_properties.return_value = SimpleNamespace(
            uuid="4edb0631-c663-d7d9-671c-196d4841cbfe",
        )
        psutil = MagicMock()
        psutil.virtual_memory.return_value = SimpleNamespace(total=32 * 2**30,
                                                            available=16 * 2**30)
        psutil.cpu_count.return_value = 8
        return {"torch": torch, "psutil": psutil}

    def test_gpu_preflight_uses_selected_device_uuid(self):
        modules = self.resource_mocks()
        with patch.dict("sys.modules", modules), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}), \
                patch("th02.run.subprocess.run", return_value=SimpleNamespace(
                    stdout="RTX 3090, 24576, 21000, 570.133.20\n")) as query:
            resources = inspect_resources()
        selector = "GPU-4edb0631-c663-d7d9-671c-196d4841cbfe"
        self.assertEqual(query.call_args.args[0][1:3], ["-i", selector])
        modules["torch"].cuda.get_device_properties.assert_called_once_with(0)
        self.assertEqual(resources["gpu"]["uuid"], selector)
        self.assertEqual(resources["gpu"]["memory_free_mib"], 21000)

    def test_gpu_preflight_without_visibility_override(self):
        modules = self.resource_mocks()
        modules["torch"].cuda.get_device_properties.return_value.uuid = "GPU-first-device"
        with patch.dict("sys.modules", modules), patch.dict(os.environ):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            with patch("th02.run.subprocess.run", return_value=SimpleNamespace(
                    stdout="A100, 81920, 8000, 570.133.20\n")) as query:
                resources = inspect_resources()
        self.assertEqual(query.call_args.args[0][1:3], ["-i", "GPU-first-device"])
        self.assertEqual(resources["gpu"]["logical_index"], 0)

    def test_gpu_preflight_without_cuda_preserves_cpu_resources(self):
        with patch.dict("sys.modules", self.resource_mocks(cuda_available=False)), \
                patch("th02.run.subprocess.run") as query:
            resources = inspect_resources()
        query.assert_not_called()
        self.assertNotIn("gpu", resources)
        self.assertIn("CUDA is unavailable", resources["gpu_error"])
        self.assertEqual(resources["ram_available_gib"], 16)

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
