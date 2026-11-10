"""Small offline checks for token accounting and joint-encoder checkpoints."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from transformers import BertConfig, BertTokenizerFast

from th02.pair_encoder import (PairConfig, PairNetwork, PairEncoderModel,
                               head_tail, load_pair_encoder, pair_token_ids)


class PairEncoderTests(unittest.TestCase):
    def test_head_tail_and_zero_budget(self):
        self.assertEqual(head_tail(list(range(10)), 5), [0, 1, 2, 8, 9])
        self.assertEqual(head_tail(list(range(10)), 0), [])
        with self.assertRaises(ValueError):
            head_tail([1], -1)

    def test_pair_budget_short_field_releases_capacity_and_keeps_separators(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nx\n")
            tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"))
            config = PairConfig(max_length=12, prompt_budget=3)
            encoded, flags = pair_token_ids(tokenizer, [5] * 20, [5] * 2, config)
            self.assertEqual(len(encoded["input_ids"]), 12)
            self.assertEqual(sum(encoded["attention_mask"]), 12)
            self.assertEqual(encoded["input_ids"].count(tokenizer.sep_token_id), 2)
            self.assertTrue(flags["prompt_truncated"])
            self.assertFalse(flags["answer_truncated"])
            encoded, flags = pair_token_ids(tokenizer, [5], [5] * 20, config)
            self.assertEqual(sum(encoded["token_type_ids"]), 9)  # eight answer tokens + SEP
            self.assertTrue(flags["answer_truncated"])

    def test_three_logits_swapped_features_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            BertConfig(vocab_size=20, hidden_size=8, num_hidden_layers=1,
                       num_attention_heads=2, intermediate_size=16).save_pretrained(path)
            config = PairConfig(hidden_dim=8)
            network = PairNetwork(path, config, pretrained=False).eval()
            side = torch.randn(3, 2, 16)
            numeric = torch.randn(3, 15)
            self.assertEqual(tuple(network.classify(side, numeric).shape), (3, 3))
            torch.testing.assert_close(network.classify(side, numeric, True),
                                       network.classify(side[:, [1, 0]], numeric))
            scaler = StandardScaler().fit(np.arange(60).reshape(4, 15))
            model = PairEncoderModel(network, scaler, config, path, "cpu")
            model.save(path / "checkpoint.pt", {"test": True})
            loaded = load_pair_encoder(path / "checkpoint.pt").network.eval()
            torch.testing.assert_close(network.classify(side, numeric),
                                       loaded.classify(side, numeric))


if __name__ == "__main__":
    unittest.main()
