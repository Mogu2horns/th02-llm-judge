"""Regression checks for semantic caching, order handling, and reproducibility."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from th02.data import Sample, structural_features
from th02.semantic import (
    EMBEDDING_DIM, INPUT_DIM, MAX_TOKENS, _batch_features, _numeric_features,
    _tokenize_batch, encode_samples, load_semantic, train_semantic,
)


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(character) % 97 + 1 for character in text]}

    def prepare_for_model(self, ids, **kwargs):
        values = [101] + list(ids) + [102]
        return {"input_ids": values, "attention_mask": [1] * len(values)}

    def pad(self, encoded, **kwargs):
        length = max(len(item["input_ids"]) for item in encoded)
        return {
            key: torch.tensor([item[key] + [0] * (length - len(item[key])) for item in encoded])
            for key in ("input_ids", "attention_mask")
        }


class FakeEncoder:
    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __call__(self, input_ids, attention_mask):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated interrupted encoding")
        basis = torch.arange(1, EMBEDDING_DIM + 1, device=input_ids.device).float()
        hidden = torch.sin(input_ids.float().unsqueeze(-1) / basis)
        return SimpleNamespace(last_hidden_state=hidden)


def example(index, label=None):
    return Sample(
        id=str(index), prompt=[f"Question {index}"], response_a=[f"Answer A {index}"],
        response_b=[f"Answer B {index}"], null_counts=(0, 0, 0),
        label=index % 3 if label is None else label,
    )


class SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_head_tail_preserves_budget_and_ends(self):
        tokenizer = FakeTokenizer()
        text = "a" * 200 + "b" * 200
        batch, flags = _tokenize_batch(tokenizer, [text, ""])
        self.assertEqual(batch["input_ids"].shape[1], MAX_TOKENS)
        self.assertEqual(flags, [True, False])
        ids = batch["input_ids"][0]
        self.assertEqual(ids[1].item(), ord("a") % 97 + 1)
        self.assertEqual(ids[-2].item(), ord("b") % 97 + 1)
        self.assertEqual(batch["attention_mask"][1].sum().item(), 2)

    def test_interrupted_cache_resumes_and_complete_cache_reuses(self):
        samples = [example(index) for index in range(3)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            directory = Path(temporary)
            model = directory / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"hidden_size": 384}))
            cache = directory / "cache"
            interrupted = FakeEncoder(fail_on_call=2)
            with patch("th02.semantic._load_encoder", return_value=(FakeTokenizer(), interrupted)):
                with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
                    encode_samples(samples, cache, model, device="cpu", batch_size=2)
            self.assertFalse((cache / "metadata.json").exists())
            self.assertEqual(json.loads((cache / "progress.json").read_text())["completed_fields"], 2)
            resumed_encoder = FakeEncoder()
            with patch("th02.semantic._load_encoder", return_value=(FakeTokenizer(), resumed_encoder)):
                vectors, metadata = encode_samples(samples, cache, model, device="cpu", batch_size=2)
            self.assertEqual(metadata["resumed_from_fields"], 2)
            self.assertEqual(vectors.shape, (3, 3, 384))
            self.assertTrue(np.isfinite(vectors).all())
            np.testing.assert_allclose(np.linalg.norm(vectors, axis=2), 1, atol=1e-6)
            with patch("th02.semantic._load_encoder", side_effect=AssertionError("cache should be reused")):
                cached, _ = encode_samples(samples, cache, model, device="cpu")
            np.testing.assert_array_equal(vectors, cached)
            del vectors, cached

    def test_feature_dimension_and_exchange(self):
        samples = [example(0), example(1)]
        vectors = np.random.default_rng(4).normal(size=(2, 3, 384)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
        stats = structural_features(samples)
        scaler = StandardScaler().fit(_numeric_features(vectors, stats))
        features = _batch_features(vectors, stats, scaler, "cpu").numpy()
        self.assertEqual(features.shape, (2, INPUT_DIM))
        np.testing.assert_array_equal(features[:, :384], vectors[:, 0])
        np.testing.assert_allclose(features[:, 1152:1536], vectors[:, 1] - vectors[:, 2])
        exchanged = vectors[:, [0, 2, 1]]
        exchange_stats = structural_features(samples, swapped=True)
        exchange_features = _batch_features(exchanged, exchange_stats, scaler, "cpu").numpy()
        np.testing.assert_allclose(features[:, 1152:1536], -exchange_features[:, 1152:1536])
        np.testing.assert_array_equal(features[:, 1536:1920], exchange_features[:, 1536:1920])

    def test_small_training_reproducible_and_checkpoint_loadable(self):
        samples = [example(index) for index in range(40)]
        vectors = np.random.default_rng(9).normal(size=(40, 3, 384)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
        splits = {
            "fit": np.arange(24), "tune": np.arange(24, 32),
            "calibration": np.arange(32, 40), "refit": np.arange(32),
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            directory = Path(temporary)
            first, metadata = train_semantic(samples, vectors, splits, directory / "first", 123, "cpu", 1)
            second, _ = train_semantic(samples, vectors, splits, directory / "second", 123, "cpu", 1)
            prediction = first.predict_proba(samples, vectors)
            np.testing.assert_array_equal(prediction, second.predict_proba(samples, vectors))
            restored = load_semantic(directory / "first" / "models" / "semantic.pt", "cpu")
            np.testing.assert_array_equal(prediction, restored.predict_proba(samples, vectors))
            self.assertEqual(metadata["selected_epochs"], 1)
            np.testing.assert_allclose(prediction.sum(axis=1), 1)
            self.assertEqual(first.scaler.n_samples_seen_, 64)


if __name__ == "__main__":
    unittest.main()
