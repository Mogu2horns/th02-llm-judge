"""Independent prompt / shared-answer TF-IDF logistic regression baseline."""

from __future__ import annotations

import gc
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .data import Sample, targets, truncate_chars


@dataclass
class LexicalModel:
    prompt_vectorizer: TfidfVectorizer
    answer_vectorizer: TfidfVectorizer
    classifier: LogisticRegression

    def transform(self, samples: Sequence[Sample], swapped: bool = False) -> sparse.csr_matrix:
        first_index, second_index = (2, 1) if swapped else (1, 2)
        prompt = self.prompt_vectorizer.transform(truncate_chars(sample.texts[0]) for sample in samples)
        first = self.answer_vectorizer.transform(truncate_chars(sample.texts[first_index]) for sample in samples)
        second = self.answer_vectorizer.transform(truncate_chars(sample.texts[second_index]) for sample in samples)
        return sparse.hstack((prompt, first, second), format="csr", dtype=np.float32)

    def predict_proba(self, samples: Sequence[Sample], swapped: bool = False) -> np.ndarray:
        output = np.empty((len(samples), 3), dtype=np.float64)
        for start in range(0, len(samples), 512):
            batch = samples[start:start + 512]
            matrix = self.transform(batch, swapped=swapped)
            probabilities = self.classifier.predict_proba(matrix)
            for column, label in enumerate(self.classifier.classes_):
                output[start:start + len(batch), int(label)] = probabilities[:, column]
        return output


def _new_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w+\b",
    )


def _fit_vectorizers(samples: Sequence[Sample]) -> LexicalModel:
    prompt_vectorizer = _new_vectorizer(20000)
    answer_vectorizer = _new_vectorizer(40000)
    prompt_vectorizer.fit(truncate_chars(sample.texts[0]) for sample in samples)
    answer_vectorizer.fit(
        truncate_chars(sample.texts[side]) for sample in samples for side in (1, 2)
    )
    return LexicalModel(prompt_vectorizer, answer_vectorizer, None)


def _fit_classifier(matrix: sparse.csr_matrix, labels: np.ndarray, c_value: float, seed: int):
    classifier = LogisticRegression(
        C=c_value,
        solver="saga",
        max_iter=1000,
        tol=1e-3,
        random_state=seed,
        n_jobs=1,
    )
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(matrix, labels)
    details = {
        "C": c_value,
        "iterations": classifier.n_iter_.tolist(),
        "converged": not any(issubclass(item.category, ConvergenceWarning) for item in caught),
        "warnings": [str(item.message) for item in caught],
        "elapsed_seconds": time.perf_counter() - start,
    }
    return classifier, details


def train_baseline(
    samples: Sequence[Sample], splits: dict[str, np.ndarray], output_dir: str | Path, seed: int
) -> tuple[LexicalModel, dict]:
    output_dir = Path(output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    fit_samples = [samples[int(index)] for index in splits["fit"]]
    tune_samples = [samples[int(index)] for index in splits["tune"]]
    start = time.perf_counter()
    model = _fit_vectorizers(fit_samples)
    fit_matrix = model.transform(fit_samples)
    tune_matrix = model.transform(tune_samples)
    fit_labels, tune_labels = targets(fit_samples), targets(tune_samples)
    trials = []
    for c_value in (0.1, 1.0, 10.0):
        classifier, details = _fit_classifier(fit_matrix, fit_labels, c_value, seed)
        details["tune_log_loss"] = float(log_loss(tune_labels, classifier.predict_proba(tune_matrix), labels=[0, 1, 2]))
        trials.append(details)
        print(json.dumps({"stage": "baseline_tuning", **details}), flush=True)
        del classifier
    best = min(trials, key=lambda trial: trial["tune_log_loss"])
    del model, fit_matrix, tune_matrix, fit_samples, tune_samples
    gc.collect()

    refit_samples = [samples[int(index)] for index in splits["refit"]]
    model = _fit_vectorizers(refit_samples)
    refit_matrix = model.transform(refit_samples)
    model.classifier, final_details = _fit_classifier(refit_matrix, targets(refit_samples), best["C"], seed)
    model_path = output_dir / "models" / "baseline.joblib"
    temporary_model_path = model_path.with_suffix(".joblib.tmp")
    joblib.dump(model, temporary_model_path, compress=3)
    os.replace(temporary_model_path, model_path)
    metadata = {
        "trials": trials,
        "selected_C": best["C"],
        "refit": final_details,
        "fit_rows": len(splits["fit"]),
        "tune_rows": len(splits["tune"]),
        "refit_rows": len(refit_samples),
        "prompt_vocabulary_size": len(model.prompt_vectorizer.vocabulary_),
        "shared_answer_vocabulary_size": len(model.answer_vectorizer.vocabulary_),
        "model_path": str(model_path),
        "elapsed_seconds": time.perf_counter() - start,
    }
    metadata_path = output_dir / "baseline_training.json"
    temporary_metadata_path = metadata_path.with_suffix(".json.tmp")
    with temporary_metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    os.replace(temporary_metadata_path, metadata_path)
    return model, metadata
