import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from th02.evaluation import (
    LABELS, metrics, score_official, select_calibration, select_error_indices,
    symmetrize, temperature_scale, validate_probabilities, write_predictions,
)


class EvaluationTests(unittest.TestCase):
    def test_probability_validation_rejects_invalid_values(self):
        invalid = (
            [[0.3, 0.3, 0.3]], [[-0.1, 0.5, 0.6]], [[np.nan, 0.0, 1.0]],
            [[np.inf, 0.0, 0.0]], [[1.0, 0.0]],
        )
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                validate_probabilities(probabilities)
        self.assertEqual(validate_probabilities([[0.2, 0.3, 0.5]]).dtype, np.float64)

    def test_symmetrize_respects_a_b_exchange(self):
        original = np.array([[0.6, 0.1, 0.3], [0.1, 0.2, 0.7]])
        swapped = np.array([[0.2, 0.5, 0.3], [0.3, 0.1, 0.6]])
        result = symmetrize(original, swapped)
        np.testing.assert_allclose(result, [[0.55, 0.15, 0.3], [0.1, 0.25, 0.65]])
        np.testing.assert_allclose(result, symmetrize(swapped, original)[:, [1, 0, 2]])

    def test_temperature_identity_and_flattening(self):
        probabilities = np.array([[0.8, 0.15, 0.05]])
        np.testing.assert_allclose(temperature_scale(probabilities, 1.0), probabilities)
        self.assertLess(temperature_scale(probabilities, 2.0).max(), probabilities.max())
        for value in (0, -1, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                temperature_scale(probabilities, value)

    def test_calibration_grid_returns_legal_probabilities(self):
        probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]])
        result = select_calibration([0, 1, 2], probabilities, probabilities, probabilities)
        self.assertEqual(len(result["grid"]), 35)
        self.assertEqual(result["alpha"], 0.0)
        fused = temperature_scale(probabilities, result["temperature"])
        self.assertAlmostEqual(metrics([0, 1, 2], fused)["log_loss"], result["fused_log_loss"])

    def test_metric_absent_classes_and_invalid_targets(self):
        result = metrics([0], [[0.6, 0.2, 0.2]])
        self.assertEqual(result["per_class"]["winner_tie"]["support"], 0)
        self.assertEqual(result["accuracy"], 1.0)
        with self.assertRaises(ValueError):
            metrics([0.5], [[0.6, 0.2, 0.2]])

    def test_atomic_write_and_official_score_match(self):
        samples = [SimpleNamespace(id="x"), SimpleNamespace(id="y")]
        probabilities = np.array([[0.75, 0.2, 0.05], [0.1, 0.3, 0.6]])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            truth = Path(directory) / "truth.csv"
            prediction = Path(directory) / "prediction.csv"
            with truth.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("id", *LABELS))
                writer.writerow(("x", 1, 0, 0))
                writer.writerow(("y", 0, 0, 1))
            write_predictions(samples, probabilities, prediction)
            self.assertAlmostEqual(
                score_official(truth, prediction), metrics([0, 2], probabilities)["log_loss"], places=14,
            )
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            with prediction.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(("extra", 0.2, 0.3, 0.5))
            with self.assertRaises(ValueError):
                score_official(truth, prediction)

    def test_duplicate_prediction_ids_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, self.assertRaises(ValueError):
            write_predictions(
                [SimpleNamespace(id="same"), SimpleNamespace(id="same")],
                [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], Path(directory) / "prediction.csv",
            )

    def test_error_cases_are_eligible_and_content_distinct(self):
        samples = []
        for index in range(14):
            samples.append(SimpleNamespace(
                id=str(index), prompt=[f"question {index}"], response_a=["A"], response_b=["B"],
                null_counts=(0, 0, 0), texts=(f"question {index}", "A", "B"),
            ))
        samples[13] = samples[0]
        probabilities = np.tile([0.2, 0.6, 0.2], (14, 1))
        selected = select_error_indices(samples, np.zeros(14, dtype=int), probabilities, probabilities)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(set(samples[index].id for index in selected)), 12)


if __name__ == "__main__":
    unittest.main()
