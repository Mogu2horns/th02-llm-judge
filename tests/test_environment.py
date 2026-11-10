"""Dependency locks must be installable outside Conda's original build machines."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from th02.environment import write_dependency_lock


class EnvironmentTests(unittest.TestCase):
    def test_lock_pins_active_versions_without_local_urls(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "requirements-lock.txt"
            with patch("th02.environment.installed_versions",
                       return_value={"pip": "24.0", "torch": "2.5.1", "filelock": "3.13.1"}):
                write_dependency_lock(path)
            contents = path.read_text(encoding="utf-8")
            self.assertIn("torch==2.5.1", contents)
            self.assertIn("filelock==3.13.1", contents)
            self.assertNotIn("pip==", contents)
            self.assertNotIn("file://", contents)


if __name__ == "__main__":
    unittest.main()
