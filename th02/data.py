"""Validated Arena data loading, grouped splits, and text statistics."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


LABELS = ("winner_model_a", "winner_model_b", "winner_tie")
TEXT_FIELDS = ("prompt", "response_a", "response_b")
STRUCTURAL_FEATURE_NAMES = (
    "a_log_characters",
    "a_log_words",
    "a_log_nonempty_turns",
    "a_log_null_turns",
    "a_log_code_fences",
    "a_is_empty",
    "b_log_characters",
    "b_log_words",
    "b_log_nonempty_turns",
    "b_log_null_turns",
    "b_log_code_fences",
    "b_is_empty",
    "prompt_log_characters",
    "prompt_log_turns",
    "a_minus_b_log_characters",
)


def _normalize_turn(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _canonical(turn_fields: Sequence[Sequence[str]]) -> str:
    normalized = [[_normalize_turn(turn) for turn in turns] for turns in turn_fields]
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


@dataclass
class Sample:
    id: str
    prompt: list[str]
    response_a: list[str]
    response_b: list[str]
    null_counts: tuple[int, int, int]
    label: int

    @cached_property
    def texts(self) -> tuple[str, str, str]:
        return tuple(
            "\n".join(f"[TURN {index + 1}]\n{text}" for index, text in enumerate(turns))
            for turns in (self.prompt, self.response_a, self.response_b)
        )

    @cached_property
    def group_hash(self) -> str:
        canonical = _canonical((self.prompt, self.response_a, self.response_b))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @cached_property
    def raw_group_hash(self) -> str:
        canonical = json.dumps(
            [self.prompt, self.response_a, self.response_b],
            ensure_ascii=False, separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_turns(value: str, sample_id: str, field: str) -> tuple[list[str], int]:
    try:
        turns = json.loads(value)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Sample {sample_id!r}, field {field}: invalid JSON") from error
    if not isinstance(turns, list):
        raise ValueError(f"Sample {sample_id!r}, field {field}: expected a JSON array")
    if any(turn is not None and not isinstance(turn, str) for turn in turns):
        raise ValueError(f"Sample {sample_id!r}, field {field}: turns must be strings or null")
    return ["" if turn is None else turn for turn in turns], turns.count(None)


def load_samples(path: str | Path, limit: int | None = None) -> list[Sample]:
    """Read labelled train/dev or an unlabelled inference CSV, failing with IDs."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    csv.field_size_limit(2**31 - 1)
    samples = []
    seen_ids = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        required = {"id", *TEXT_FIELDS}
        if not required.issubset(fields):
            raise ValueError(f"CSV {path} missing columns {sorted(required - fields)}")
        present_labels = fields.intersection(LABELS)
        if present_labels and len(present_labels) != len(LABELS):
            raise ValueError(f"CSV {path} must provide all three label columns together")
        for line_number, row in enumerate(reader, start=2):
            sample_id = (row.get("id") or "").strip()
            if not sample_id:
                raise ValueError(f"CSV {path}, line {line_number}: missing sample id")
            if sample_id in seen_ids:
                raise ValueError(f"Sample {sample_id!r}: duplicate id")
            seen_ids.add(sample_id)
            parsed = [_parse_turns(row.get(field), sample_id, field) for field in TEXT_FIELDS]
            turn_lists = [result[0] for result in parsed]
            if len({len(turns) for turns in turn_lists}) != 1:
                raise ValueError(f"Sample {sample_id!r}: prompt/A/B turn counts do not match")
            label = -1
            if present_labels:
                values = [(row.get(field) or "").strip() for field in LABELS]
                if any(value not in {"0", "1"} for value in values) or values.count("1") != 1:
                    raise ValueError(f"Sample {sample_id!r}: invalid one-hot label {values}")
                label = values.index("1")
            samples.append(
                Sample(
                    id=sample_id,
                    prompt=turn_lists[0],
                    response_a=turn_lists[1],
                    response_b=turn_lists[2],
                    null_counts=tuple(result[1] for result in parsed),
                    label=label,
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"CSV {path} contains no samples")
    return samples


def targets(samples: Sequence[Sample]) -> np.ndarray:
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    if np.any((labels < 0) | (labels > 2)):
        raise ValueError("Training/evaluation requires a valid label for every sample")
    return labels


def make_splits(samples: Sequence[Sample], seed: int) -> dict[str, np.ndarray]:
    """Use ten deterministic stratified group folds for 80/10/10 allocation."""
    labels = targets(samples)
    groups = np.asarray([sample.group_hash for sample in samples])
    if len(set(groups)) < 10 or np.min(np.bincount(labels, minlength=3)) < 10:
        raise ValueError("Ten-fold grouped splitting needs at least ten samples per class and ten groups")
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(samples)), labels, groups))
    calibration = np.sort(folds[0][1]).astype(np.int64)
    tune = np.sort(folds[1][1]).astype(np.int64)
    refit = np.setdiff1d(np.arange(len(samples), dtype=np.int64), calibration)
    fit = np.setdiff1d(refit, tune)
    splits = {"fit": fit, "tune": tune, "calibration": calibration, "refit": refit}
    for first, second in (("fit", "tune"), ("fit", "calibration"), ("tune", "calibration")):
        if set(groups[splits[first]]).intersection(groups[splits[second]]):
            raise AssertionError(f"Content leakage between {first} and {second}")
    return splits


def truncate_chars(text: str, max_chars: int = 16000) -> str:
    if max_chars < 2:
        raise ValueError("max_chars must be at least two")
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    return text[:head] + text[-(max_chars - head):]


def _side_statistics(turns: Sequence[str], null_count: int) -> list[float]:
    text = "\n".join(turns)
    counts = (
        len(text),
        len(text.split()),
        sum(bool(turn.strip()) for turn in turns),
        null_count,
        text.count("```"),
    )
    return [float(np.log1p(count)) for count in counts] + [float(not text.strip())]


def structural_features(samples: Sequence[Sample], swapped: bool = False) -> np.ndarray:
    features = np.empty((len(samples), len(STRUCTURAL_FEATURE_NAMES)), dtype=np.float32)
    for index, sample in enumerate(samples):
        first, second = sample.response_a, sample.response_b
        first_null, second_null = sample.null_counts[1:]
        if swapped:
            first, second = second, first
            first_null, second_null = second_null, first_null
        first_stats = _side_statistics(first, first_null)
        second_stats = _side_statistics(second, second_null)
        prompt_text = "\n".join(sample.prompt)
        features[index] = first_stats + second_stats + [
            float(np.log1p(len(prompt_text))),
            float(np.log1p(len(sample.prompt))),
            first_stats[0] - second_stats[0],
        ]
    return features


def _length_summary(lengths: Sequence[int]) -> dict:
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": int(np.max(values)),
    }


def _profile_split(samples: Sequence[Sample]) -> dict:
    group_members = defaultdict(list)
    prompt_counts = Counter()
    lengths = {field: [] for field in TEXT_FIELDS}
    null_samples = Counter()
    empty_samples = Counter()
    truncated_ids = {field: [] for field in TEXT_FIELDS}
    for sample in samples:
        group_members[sample.group_hash].append(sample)
        prompt_counts[_canonical((sample.prompt,))] += 1
        for field_index, field in enumerate(TEXT_FIELDS):
            turns = getattr(sample, field)
            raw_text = "\n".join(turns)
            lengths[field].append(len(raw_text))
            null_samples[field] += int(sample.null_counts[field_index] > 0)
            empty_samples[field] += int(not raw_text.strip())
            if len(sample.texts[field_index]) > 16000:
                truncated_ids[field].append(sample.id)
    conflicts = [
        {"group_hash": group, "sample_ids": [sample.id for sample in members],
         "labels": [int(sample.label) for sample in members]}
        for group, members in group_members.items()
        if len({sample.label for sample in members}) > 1
    ]
    return {
        "rows": len(samples),
        "class_counts": {label: sum(sample.label == index for sample in samples)
                         for index, label in enumerate(LABELS)},
        "multi_turn_samples": sum(len(sample.prompt) > 1 for sample in samples),
        "null_samples": dict(null_samples),
        "empty_samples": dict(empty_samples),
        "identical_responses": sum(sample.response_a == sample.response_b for sample in samples),
        "parsed_exact_duplicate_extra_rows": len(samples) - len({sample.raw_group_hash for sample in samples}),
        "normalized_duplicate_extra_rows": len(samples) - len(group_members),
        "normalized_duplicate_groups": sum(len(members) > 1 for members in group_members.values()),
        "conflicting_label_groups": conflicts,
        "repeated_prompt_extra_rows": sum(count - 1 for count in prompt_counts.values()),
        "length_characters": {field: _length_summary(values) for field, values in lengths.items()},
        "total_length_characters": _length_summary([
            sum(lengths[field][index] for field in TEXT_FIELDS) for index in range(len(samples))
        ]),
        "tfidf_truncated_ids": truncated_ids,
        "tfidf_truncation_rates": {
            field: len(ids) / len(samples) for field, ids in truncated_ids.items()
        },
    }


def profile_data(train: Sequence[Sample], dev: Sequence[Sample]) -> dict:
    train_groups = {sample.group_hash for sample in train}
    dev_groups = {sample.group_hash for sample in dev}
    train_prompts = {_canonical((sample.prompt,)) for sample in train}
    dev_prompts = {_canonical((sample.prompt,)) for sample in dev}
    overlap = train_groups.intersection(dev_groups)
    overlap_samples = [
        {
            "group_hash": group,
            "train_samples": [{"id": sample.id, "label": int(sample.label), "raw_group_hash": sample.raw_group_hash}
                              for sample in train if sample.group_hash == group],
            "dev_samples": [{"id": sample.id, "label": int(sample.label), "raw_group_hash": sample.raw_group_hash}
                            for sample in dev if sample.group_hash == group],
        }
        for group in sorted(overlap)
    ]
    return {
        "train": _profile_split(train),
        "dev": _profile_split(dev),
        "cross_split": {
            "normalized_dialogue_overlap_groups": len(overlap),
            "normalized_dialogue_overlap_hashes": sorted(overlap),
            "normalized_dialogue_overlap_samples": overlap_samples,
            "parsed_exact_dialogue_overlap_groups": len(
                {sample.raw_group_hash for sample in train}.intersection(
                    sample.raw_group_hash for sample in dev
                )
            ),
            "normalized_prompt_overlap_groups": len(train_prompts.intersection(dev_prompts)),
        },
        "normalization": "Unicode NFKC, collapse whitespace, preserve case and turn boundaries, null as empty",
    }
