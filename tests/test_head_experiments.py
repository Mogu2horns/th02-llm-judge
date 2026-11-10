"""Checks for the new head shapes, feature ordering, and saved checkpoints."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from th02.data import Sample, structural_features
from th02.evaluation import symmetrize
from th02.head_experiments import HEADS, HeadModel, build_network, fit_fixed_head, load_head
from th02.semantic import _batch_features, _numeric_features


class HeadExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_heads_train_three_class_logits(self):
        for hidden in HEADS.values():
            network = build_network(hidden)
            logits = network(torch.randn(7, 2706))
            self.assertEqual(tuple(logits.shape), (7, 3))
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(7) % 3)
            loss.backward()
            self.assertTrue(all(value.grad is not None for value in network.parameters()))
            self.assertEqual(sum(isinstance(module, torch.nn.Dropout) for module in network), bool(hidden))

    def test_checkpoint_predicts_both_orders_and_numeric_ablation(self):
        rng = np.random.default_rng(29)
        embeddings = rng.normal(size=(5, 3, 384)).astype(np.float32)
        samples = [Sample(str(i), [f"q{i}"], ["short"], ["a longer reply"], (0, 0, 0), i % 3)
                   for i in range(5)]
        structural = structural_features(samples)
        scaler = StandardScaler().fit(_numeric_features(embeddings, structural))
        swapped_samples = [Sample(s.id, s.prompt, s.response_b, s.response_a, (0, 0, 0), s.label)
                           for s in samples]
        with tempfile.TemporaryDirectory() as temporary:
            for numeric in (True, False):
                model = HeadModel(build_network((11, 7), 2706 if numeric else 2688), scaler,
                                  (11, 7), "cpu", numeric)
                direct = model.predict_proba(samples, embeddings)
                swapped = model.predict_proba(samples, embeddings, swapped=True)
                np.testing.assert_array_equal(swapped, model.predict_proba(
                    swapped_samples, embeddings[:, [0, 2, 1]]))
                path = Path(temporary) / f"head_{numeric}.pt"
                model.save(path, {"epoch": 3})
                restored = load_head(path, "cpu")
                np.testing.assert_array_equal(direct, restored.predict_proba(samples, embeddings))
                np.testing.assert_array_equal(swapped, restored.predict_proba(samples, embeddings, swapped=True))
                symmetric = symmetrize(direct, swapped)
                np.testing.assert_array_equal(symmetric[:, [1, 0, 2]], symmetrize(swapped, direct))
                np.testing.assert_allclose(symmetric.sum(axis=1), 1.0, atol=1e-15)

    def test_cached_feature_swap_matches_mixed_training_batch(self):
        rng = np.random.default_rng(9)
        embeddings = rng.normal(size=(7, 3, 384)).astype(np.float32)
        samples = [Sample(str(i), ["question"], ["a" * i], ["b" * (2 * i)], (0, 0, 0), i % 3)
                   for i in range(7)]
        direct_stats = structural_features(samples)
        swapped_stats = structural_features(samples, swapped=True)
        scaler = StandardScaler().fit(_numeric_features(embeddings, direct_stats))
        direct = _batch_features(embeddings, direct_stats, scaler, "cpu")
        swapped = _batch_features(embeddings[:, [0, 2, 1]], swapped_stats, scaler, "cpu")
        mask = np.array([True, False, True, False, False, True, True])
        mixed_embeddings, mixed_stats = embeddings.copy(), direct_stats.copy()
        mixed_embeddings[mask, 1], mixed_embeddings[mask, 2] = (
            mixed_embeddings[mask, 2].copy(), mixed_embeddings[mask, 1].copy())
        mixed_stats[mask] = swapped_stats[mask]
        cached = direct.clone()
        cached[mask] = swapped[mask]
        torch.testing.assert_close(cached, _batch_features(mixed_embeddings, mixed_stats, scaler, "cpu"),
                                   rtol=0, atol=0)

    def test_fixed_refit_ignores_heldout_samples_and_labels(self):
        rng = np.random.default_rng(13)
        embeddings = rng.normal(size=(12, 3, 384)).astype(np.float32)
        samples = [Sample(str(i), [f"q{i}"], ["a" * (i + 1)], ["b" * (i + 2)], (0, 0, 0), i % 3)
                   for i in range(12)]
        indices = np.arange(8, dtype=np.int64)
        first, metadata = fit_fixed_head(samples, embeddings, indices, (5,), 2, device="cpu")
        changed_samples = list(samples)
        changed_embeddings = embeddings.copy()
        for i in range(8, 12):
            changed_samples[i] = Sample(str(i), ["changed prompt"], ["unseen" * 50], ["x"],
                                        (0, 0, 0), (i + 1) % 3)
            changed_embeddings[i] *= 100
        second, _ = fit_fixed_head(changed_samples, changed_embeddings, indices, (5,), 2, device="cpu")
        for key, value in first.network.state_dict().items():
            torch.testing.assert_close(value, second.network.state_dict()[key], rtol=0, atol=0)
        np.testing.assert_array_equal(first.scaler.mean_, second.scaler.mean_)
        self.assertEqual(int(first.scaler.n_samples_seen_), 2 * len(indices))
        self.assertEqual(metadata["fit_count"], len(indices))
        self.assertEqual(len(metadata["history"]), 2)


if __name__ == "__main__":
    unittest.main()
