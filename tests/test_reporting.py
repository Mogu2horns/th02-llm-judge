"""Ensure generated error candidates retain their evidence and review status."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from th02.data import Sample, profile_data
from th02.evaluation import metrics
from th02.reporting import write_reports


class ReportingTests(unittest.TestCase):
    def test_smoke_report_has_twelve_reviewable_distinct_cases(self):
        samples = [Sample(str(index), [f"Question {index}"], [f"Answer A {index}"],
                          [f"Answer B {index}"], (0, 0, 0), index % 3)
                   for index in range(16)]
        probabilities = np.tile([0.25, 0.25, 0.5], (len(samples), 1))
        labels = np.asarray([sample.label for sample in samples])
        predictions = {"baseline": probabilities, "semantic": probabilities,
                       "fusion_calibrated": probabilities}
        experiment = {"smoke": True, "seed": 20260916, "results": {
            "baseline": metrics(labels, probabilities),
            "fusion_calibrated": metrics(labels, probabilities)}}
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output = Path(directory)
            write_reports(output, samples, samples, profile_data(samples, samples),
                          experiment, predictions, {"dev": {"truncated_ids": {}}})
            payload = json.loads((output / "error_cases.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "candidate_requires_review")
            self.assertEqual(len(payload["cases"]), 12)
            self.assertEqual(len({case["id"] for case in payload["cases"]}), 12)
            self.assertIn("raw_fields", payload["cases"][0])
            self.assertIn("小规模", (output / "REPORT.md").read_text(encoding="utf-8"))
            self.assertIn("metrics.json", (output / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
