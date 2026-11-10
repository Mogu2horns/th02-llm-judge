#!/usr/bin/env python3
"""Build and verify the local C-TH-02 delivery using an explicit file whitelist.

The default build requires completed final results. --allow-incomplete creates a
clearly marked draft. Existing codex_sessions is preserved, never removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/disk1/wuchufeng/C-TH-02")
MANIFEST_NAME = "DELIVERY_MANIFEST.json"
SOURCE_DIRECTORIES = ("th02", "scripts", "tests", "evaluation", "templates")
SOURCE_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".sh", ".ps1"}
ROOT_FILES = (
    "C-TH-02-LLM回答偏好预测-候选人题目说明.md", "MANIFEST.sha256",
    "README.md", "VALIDATION.md", "AI_USAGE.md", "EXPERIMENTS.md",
    "IMPROVEMENT_PROTOCOL.json", "environment-conda-explicit.txt", ".gitignore",
)
DATA_FILES = ("data/train.csv", "data/dev.csv", "data/split_manifest.json")
ENCODER_FILES = tuple("models/all-MiniLM-L6-v2/" + name for name in (
    "config.json", "model.safetensors", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.txt", "provenance.json",
))
EXCLUDED_PARTS = {".git", ".venv", ".cache", "__pycache__", "cache", "token_cache"}
REPORT_SUFFIXES = {".json", ".md", ".csv"}
MODEL_SUFFIXES = {".pt", ".joblib", ".json", ".safetensors", ".txt"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def safe_relative(name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe relative path in manifest: {name!r}")
    return Path(*relative.parts)


def permitted_path(path: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    name = path.name.lower()
    if name.startswith(".env") or name in {"auth.json", "credentials", "credentials.json", "resume.pt"}:
        return False
    if name.endswith((".key", ".pem", ".pyc", ".tmp", ".lock")):
        return False
    return True


def inventory(source: Path, head_trials: list[str]) -> tuple[dict[str, Path], list[str]]:
    """Select source files, never walk arbitrary directories into the package."""
    files: dict[str, Path] = {}
    notes: list[str] = []

    def add(path: Path) -> None:
        if not path.is_file() or path.is_symlink():
            return
        relative = path.relative_to(source)
        if not permitted_path(relative):
            return
        if not path.resolve().is_relative_to(source):
            raise ValueError(f"Source escapes project directory: {path}")
        files[relative.as_posix()] = path

    for name in ROOT_FILES:
        add(source / name)
    for pattern in ("requirements*.txt", "*.sh", "*.ps1"):
        for path in source.glob(pattern):
            add(path)
    for directory in SOURCE_DIRECTORIES:
        for path in (source / directory).rglob("*"):
            if path.suffix in SOURCE_SUFFIXES:
                add(path)
    for name in DATA_FILES + ENCODER_FILES:
        add(source / name)

    # Original baseline, original selected head, reports, independent checks.
    for directory in ("artifacts", "artifacts-tuning-audit"):
        base = source / directory
        for path in base.glob("*"):
            if path.suffix in REPORT_SUFFIXES:
                add(path)
        for path in (base / "predictions").glob("*.csv"):
            add(path)
    for name in ("baseline.joblib", "semantic.pt", "calibration.json", "selection_locked.json"):
        add(source / "artifacts/models" / name)
    add(source / "artifacts-tuning-audit/best_selection.pt")

    # Keep all small head-study evidence, but only chosen model checkpoints.
    heads = source / "artifacts-improved-heads"
    for path in heads.glob("*"):
        if path.suffix in REPORT_SUFFIXES:
            add(path)
    for trial_dir in (heads / "trials").glob("*"):
        if not trial_dir.is_dir() or trial_dir.is_symlink():
            continue
        for path in trial_dir.glob("*"):
            if path.suffix in REPORT_SUFFIXES:
                add(path)
    checkpoint_choices: dict[str, set[str]] = {}
    summary_path = heads / "summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        for group_name, checkpoint in (
            ("selected", "best.pt"), ("phase1_selected", "best.pt"),
            ("selected_accuracy", "best_accuracy.pt"),
            ("highest_accuracy_group", "best_accuracy.pt"),
        ):
            group = summary.get(group_name)
            if isinstance(group, dict):
                for trial in group.get("trials", []):
                    checkpoint_choices.setdefault(trial, set()).add(checkpoint)
    for trial in head_trials:
        checkpoint_choices.setdefault(trial, set()).update(("best.pt", "best_accuracy.pt"))
    for trial, checkpoints in checkpoint_choices.items():
        trial_relative = safe_relative(trial)
        if len(trial_relative.parts) != 1:
            raise ValueError(f"Head trial must be a single directory name: {trial}")
        for checkpoint in checkpoints:
            add(heads / "trials" / trial_relative / checkpoint)
    notes.append("Head-study checkpoints are restricted to selected, phase-one selected, highest-Accuracy groups, and explicit --head-trial overrides; all trial curves/configurations/CSV evidence remain available.")

    # Selection-stage encoder weights are redundant with final refit models.
    encoder = source / "artifacts-improved-encoder"
    for path in encoder.glob("*"):
        if path.suffix in REPORT_SUFFIXES:
            add(path)
    notes.append("Encoder selection reports/provenance/epoch metrics/CSV are retained; selection weights, resume.pt and token caches are omitted. Final selected models come from artifacts-improved-final/models.")

    final = source / "artifacts-improved-final"
    for path in final.glob("*"):
        if path.suffix in REPORT_SUFFIXES:
            add(path)
    for directory in ("predictions", "reports", "error_analysis", "verification"):
        for path in (final / directory).rglob("*"):
            if path.suffix in REPORT_SUFFIXES:
                add(path)
    for path in (final / "models").rglob("*"):
        if path.suffix in MODEL_SUFFIXES:
            add(path)
    return dict(sorted(files.items())), notes


def completion_issues(source: Path, files: dict[str, Path]) -> list[str]:
    issues = []
    required = ("README.md", "AI_USAGE.md", "evaluation/metric.py",
                "scripts/build_submission.py", "IMPROVEMENT_PROTOCOL.json") + DATA_FILES + ENCODER_FILES
    for name in required:
        if name not in files:
            issues.append(f"Required source file missing: {name}")
    final = source / "artifacts-improved-final"
    for name in ("metrics.json", "REPORT.md", "verification.json"):
        if not (final / name).is_file():
            issues.append(f"Final result missing: artifacts-improved-final/{name}")
    if (final / "metrics.json").is_file() and not load_json(final / "metrics.json"):
        issues.append("Final metrics.json is empty")
    states = [final / "run_state.json", final / "state.json"]
    state = next((path for path in states if path.is_file()), None)
    if state is None:
        issues.append("Final run_state.json/state.json missing")
    elif load_json(state).get("status") != "complete":
        issues.append("Final run state does not have status=complete")
    locks = [final / name for name in (
        "selection_locked.json", "models/selection_locked.json",
        "locked_configuration.json", "models/locked_configuration.json",
    )]
    lock = next((path for path in locks if path.is_file()), None)
    if lock is None or not load_json(lock):
        issues.append("Final nonempty locked selection/configuration JSON missing")
    verification = final / "verification.json"
    if verification.is_file() and load_json(verification).get("status") != "passed":
        issues.append("Final verification.json does not have status=passed")
    if not any(name.startswith("artifacts-improved-final/predictions/") and name.endswith(".csv") for name in files):
        issues.append("Final prediction CSV files missing")
    if not any(name.startswith("artifacts-improved-final/models/") and Path(name).suffix in {".pt", ".joblib", ".safetensors"} for name in files):
        issues.append("Final selected model weights missing")
    return issues


def session_issues(output: Path) -> list[str]:
    directory = output / "codex_sessions"
    path = directory / "manifest.json"
    if not path.is_file():
        return ["codex_sessions/manifest.json missing; export project sessions first"]
    try:
        manifest = load_json(path)
        rows = manifest["files"]
        if manifest.get("session_count") != len(rows) or not rows:
            raise ValueError("Session count inconsistent or empty")
        expected = set()
        ids = {row["session_id"] for row in rows}
        root_ids = set(manifest["root_session_ids"])
        for row in rows:
            relative = safe_relative(row["destination_file"])
            if len(relative.parts) != 1 or relative.suffix != ".jsonl":
                raise ValueError("Session filename must be a direct JSONL child")
            target = directory / relative
            if target.is_symlink():
                raise ValueError("Session symlinks are not supported")
            expected.add(relative.as_posix())
            if target.stat().st_size != row["copied_bytes"]:
                raise ValueError(f"Session byte count mismatch: {relative}")
            actual = sha256(target)
            if actual != row["destination_sha256"] or actual != row["source_snapshot_sha256"]:
                raise ValueError(f"Session SHA-256 mismatch: {relative}")
            records = 0
            with target.open("rb") as handle:
                for line in handle:
                    if not line.endswith(b"\n"):
                        raise ValueError(f"Incomplete JSONL final record: {relative}")
                    json.loads(line)
                    records += 1
            if records != row["copied_record_count"]:
                raise ValueError(f"Session record count mismatch: {relative}")
            if row["session_id"] not in root_ids and row.get("parent_thread_id") not in ids:
                raise ValueError(f"Session parent is outside selected tree: {relative}")
        actual_files = {path.name for path in directory.glob("*.jsonl")}
        if expected != actual_files:
            raise ValueError("JSONL files and session manifest differ")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"Session verification failed: {exc}"]
    return []


def copy_stable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed while copying; retry after training completes: {source}")
        if temporary.stat().st_size != before.st_size or sha256(temporary) != sha256(source):
            raise RuntimeError(f"Source/copy integrity mismatch: {source}")
        shutil.copymode(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build(source: Path, output: Path, allow_incomplete: bool, head_trials: list[str]) -> dict:
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and delivery directories must be separate, without nesting")
    files, notes = inventory(source, head_trials)
    issues = completion_issues(source, files) + session_issues(output)
    if issues and not allow_incomplete:
        raise ValueError("Delivery is incomplete; no files copied:\n- " + "\n- ".join(issues))
    # Reject symlinks and unexpected existing files; never delete session data.
    previous_manifest_path = output / MANIFEST_NAME
    old_files = {}
    if previous_manifest_path.is_file():
        old_files = {item["path"]: item for item in load_json(previous_manifest_path).get("files", [])}
    extras = []
    for path in output.rglob("*") if output.exists() else ():
        if path.is_symlink():
            raise ValueError(f"Symlink in destination: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        known = relative in files or relative in old_files or relative in {MANIFEST_NAME, "PACKAGING_README.md"} or relative.startswith("codex_sessions/")
        if not known:
            extras.append(relative)
    if extras:
        raise ValueError("Unmanaged existing delivery files; inspect before packaging: " + ", ".join(sorted(extras)))
    output.mkdir(parents=True, exist_ok=True)
    for relative, path in files.items():
        copy_stable(path, output / relative)
    # Remove only files managed by our previous manifest and now absent from
    # the whitelist. codex_sessions and user-maintained packaging notes survive.
    for relative, item in old_files.items():
        if relative not in files and item.get("kind") == "copied_project_file" and not relative.startswith("codex_sessions/"):
            (output / safe_relative(relative)).unlink(missing_ok=True)
    records = []
    session_sources = {}
    session_manifest = output / "codex_sessions/manifest.json"
    if session_manifest.is_file():
        session_sources = {item["destination_file"]: item["source_path"] for item in load_json(session_manifest).get("files", [])}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(output)
        if not permitted_path(relative):
            raise ValueError(f"Prohibited file in delivery: {relative}")
        name = relative.as_posix()
        source_path = str(files[name]) if name in files else str(path)
        kind = "copied_project_file" if name in files else "preserved_local_delivery_file"
        if name.startswith("codex_sessions/"):
            source_path = session_sources.get(path.name, str(path))
            kind = "preserved_project_session_snapshot" if path.suffix == ".jsonl" else "preserved_session_export_metadata"
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path), "source_path": source_path, "kind": kind})
    manifest = {
        "schema_version": 1, "project": "C-TH-02 / th02-llm-judge",
        "built_at_utc": now(), "source_project": str(source),
        "delivery_directory": str(output),
        "delivery_status": "incomplete_draft" if issues else "complete",
        "completion_issues": issues,
        "file_count": len(records), "total_bytes": sum(row["bytes"] for row in records),
        "manifest_self_exclusion": "DELIVERY_MANIFEST.json is excluded from its own hash list to avoid a self-referential hash.",
        "notes": notes + ["No network publish or Git operations were performed. codex_sessions was preserved and independently validated.", "Source MANIFEST.sha256 is retained as original provenance; DELIVERY_MANIFEST.json describes this delivery's full current file set."],
        "files": records,
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify(output: Path) -> dict:
    manifest = load_json(output / MANIFEST_NAME)
    failures = []
    expected = {MANIFEST_NAME}
    for row in manifest.get("files", []):
        relative = safe_relative(row["path"])
        expected.add(relative.as_posix())
        path = output / relative
        if path.is_symlink() or not path.is_file():
            failures.append(f"Missing file or unsupported symlink: {relative}")
        elif path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            failures.append(f"File integrity mismatch: {relative}")
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if actual != expected:
        failures.append(f"File inventory mismatch; extra={sorted(actual - expected)}, missing={sorted(expected - actual)}")
    if manifest.get("file_count") != len(manifest.get("files", [])):
        failures.append("Manifest file count mismatch")
    if manifest.get("total_bytes") != sum(row["bytes"] for row in manifest.get("files", [])):
        failures.append("Manifest byte count mismatch")
    has_session_manifest = (output / "codex_sessions/manifest.json").is_file()
    if has_session_manifest or manifest.get("delivery_status") == "complete":
        failures.extend(session_issues(output))
    if failures:
        raise ValueError("Delivery verification failed:\n- " + "\n- ".join(failures))
    return {"status": "passed", "delivery_status": manifest["delivery_status"], "files_verified": manifest["file_count"], "bytes_verified": manifest["total_bytes"], "session_jsonl_verified": has_session_manifest}


def archive(output: Path, manifest: dict) -> Path:
    if manifest["delivery_status"] != "complete":
        raise ValueError("An incomplete draft cannot be archived as a final submission")
    path = output.with_suffix(".zip")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as handle:
            for relative in sorted([row["path"] for row in manifest["files"]] + [MANIFEST_NAME]):
                handle.write(output / safe_relative(relative), arcname=(Path(output.name) / safe_relative(relative)).as_posix())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-incomplete", action="store_true", help="Build an explicitly marked draft; never creates an archive")
    parser.add_argument("--head-trial", action="append", default=[], help="Also retain this head trial's best and Accuracy checkpoints; repeat as needed")
    parser.add_argument("--verify", action="store_true", help="Only verify the existing delivery manifest and session JSONL files")
    parser.add_argument("--archive", action="store_true", help="After a complete build and verification, create a sibling ZIP locally")
    args = parser.parse_args()
    if args.archive and args.allow_incomplete:
        parser.error("--allow-incomplete cannot be combined with --archive")
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    try:
        if args.verify:
            if args.archive or args.allow_incomplete:
                parser.error("--verify cannot be combined with --archive/--allow-incomplete")
            result = verify(output)
        else:
            manifest = build(source, output, args.allow_incomplete, args.head_trial)
            result = verify(output)
            result["manifest"] = str(output / MANIFEST_NAME)
            if args.archive:
                result["archive"] = str(archive(output, manifest))
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
