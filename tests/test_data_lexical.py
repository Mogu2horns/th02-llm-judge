import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import joblib

from th02.data import Sample, load_samples, make_splits, structural_features, truncate_chars
from th02.lexical import train_baseline


class DataTests(unittest.TestCase):
    def write_rows(self, path, rows, labelled=True):
        fields = ["id", "prompt", "response_a", "response_b"]
        if labelled:
            fields += ["winner_model_a", "winner_model_b", "winner_tie"]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, **changes):
        row = {
            "id": "case-1", "prompt": '["question", "follow up"]',
            "response_a": '[null, "answer"]', "response_b": '["", "other"]',
            "winner_model_a": "1", "winner_model_b": "0", "winner_tie": "0",
        }
        row.update(changes)
        return row

    def test_json_null_multiturn_and_bom(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "data.csv"
            self.write_rows(path, [self.row()])
            sample = load_samples(path)[0]
        self.assertEqual(sample.response_a, ["", "answer"])
        self.assertEqual(sample.null_counts, (0, 1, 0))
        self.assertIn("[TURN 2]\nanswer", sample.texts[1])
        self.assertEqual(structural_features([sample]).shape, (1, 15))

    def test_invalid_fields_report_id(self):
        for changes in (
            {"prompt": "{}"}, {"response_a": "[3, 4]"},
            {"response_b": '["one"]'}, {"winner_model_b": "1"},
            {"response_a": "not JSON"},
        ):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
                path = Path(directory) / "data.csv"
                self.write_rows(path, [self.row(**changes)])
                with self.assertRaisesRegex(ValueError, "case-1"):
                    load_samples(path)

    def test_inference_csv_without_labels(self):
        row = {key: value for key, value in self.row().items() if not key.startswith("winner_")}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "data.csv"
            self.write_rows(path, [row], labelled=False)
            self.assertEqual(load_samples(path)[0].label, -1)

    def test_limit_stops_before_later_rows(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "data.csv"
            self.write_rows(path, [self.row(), self.row(id="invalid", prompt="{}")])
            self.assertEqual(len(load_samples(path, limit=1)), 1)
            with self.assertRaisesRegex(ValueError, "invalid"):
                load_samples(path)

    def test_normalization_and_head_tail(self):
        first = Sample("1", ["Ａ   B"], ["yes"], ["no"], (0, 0, 0), 0)
        second = Sample("2", ["A B"], ["yes"], ["no"], (0, 0, 0), 1)
        self.assertEqual(first.group_hash, second.group_hash)
        self.assertEqual(truncate_chars("abcdefghij", 6), "abchij")
        self.assertEqual(truncate_chars("short", 6), "short")

    def test_grouped_splits_and_structural_swap(self):
        samples = synthetic_samples()
        samples.append(Sample("duplicate", samples[0].prompt, samples[0].response_a,
                              samples[0].response_b, (0, 0, 0), samples[0].label))
        splits = make_splits(samples, 20260916)
        repeated = make_splits(samples, 20260916)
        for name in splits:
            np.testing.assert_array_equal(splits[name], repeated[name])
        seen = set()
        for name in ("fit", "tune", "calibration"):
            groups = {samples[int(index)].group_hash for index in splits[name]}
            self.assertFalse(seen.intersection(groups))
            seen.update(groups)
        normal = structural_features(samples)
        swapped = structural_features(samples, swapped=True)
        np.testing.assert_array_equal(normal[:, :6], swapped[:, 6:12])
        np.testing.assert_array_equal(normal[:, 14], -swapped[:, 14])


def synthetic_samples():
    return [Sample(str(index), [f"question common topic {index}"],
                   [f"answer useful text group {index % 9}"],
                   [f"other alternative response group {index % 7}"],
                   (0, 0, 0), index % 3) for index in range(120)]


class LexicalTests(unittest.TestCase):
    def test_fit_save_predict_and_swap(self):
        samples = synthetic_samples()
        splits = make_splits(samples, 20260916)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            model, metadata = train_baseline(samples, splits, directory, 20260916)
            probabilities = model.predict_proba(samples[:5])
            swapped = model.predict_proba(samples[:5], swapped=True)
            self.assertEqual(probabilities.shape, (5, 3))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
            self.assertTrue(np.isfinite(swapped).all())
            self.assertIn(metadata["selected_C"], (0.1, 1.0, 10.0))
            self.assertTrue((Path(directory) / "models" / "baseline.joblib").exists())
            reloaded = joblib.load(Path(directory) / "models" / "baseline.joblib")
            np.testing.assert_array_equal(probabilities, reloaded.predict_proba(samples[:5]))
            exchanged = [Sample(sample.id, sample.prompt, sample.response_b, sample.response_a,
                                (sample.null_counts[0], sample.null_counts[2], sample.null_counts[1]),
                                sample.label) for sample in samples[:5]]
            np.testing.assert_array_equal(swapped, model.predict_proba(exchanged))


if __name__ == "__main__":
    unittest.main()
