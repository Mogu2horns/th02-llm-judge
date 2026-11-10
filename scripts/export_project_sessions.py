#!/usr/bin/env python3
"""Copy complete JSONL snapshots of the explicitly selected project thread tree.

Only session metadata and session_index.jsonl are used for discovery. No auth,
configuration, database, or other Codex files are read. Original bytes are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT_IDS = ("01a1c593-3ad0-7413-8eba-e0ffb84cb36b",)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_metadata(path: Path) -> dict | None:
    """Read only initial records, where Codex stores session_meta."""
    with path.open("rb") as handle:
        for _ in range(32):
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if record.get("type") == "session_meta":
                return record.get("payload", {})
    return None


def parent_info(meta: dict) -> dict:
    source = meta.get("source")
    if not isinstance(source, dict):
        return {}
    subagent = source.get("subagent", {})
    if not isinstance(subagent, dict):
        return {}
    info = subagent.get("thread_spawn", {})
    return info if isinstance(info, dict) else {}


def discover(session_dir: Path, root_ids: set[str]) -> dict[str, dict]:
    sessions = {}
    for path in sorted(session_dir.rglob("*.jsonl")):
        # Resolve and constrain each input to the authorized sessions directory.
        resolved = path.resolve()
        if not resolved.is_relative_to(session_dir):
            continue
        meta = read_metadata(resolved)
        if not meta or not meta.get("id"):
            continue
        session_id = meta["id"]
        if session_id in sessions:
            raise ValueError(f"Duplicate session id: {session_id}")
        sessions[session_id] = {"path": resolved, "meta": meta}
    missing = root_ids - sessions.keys()
    if missing:
        raise ValueError(f"Requested root sessions not found: {sorted(missing)}")
    selected = set(root_ids)
    while True:
        descendants = {
            session_id
            for session_id, session in sessions.items()
            if parent_info(session["meta"]).get("parent_thread_id") in selected
        }
        expanded = selected | descendants
        if expanded == selected:
            break
        selected = expanded
    return {session_id: sessions[session_id] for session_id in sorted(selected)}


def read_titles(index_path: Path, selected_ids: set[str]) -> dict[str, str]:
    titles = {}
    if index_path.is_file():
        with index_path.open("rb") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if row.get("id") in selected_ids:
                    titles[row["id"]] = row.get("thread_name", "")
    return titles


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot(source: Path, destination: Path) -> dict:
    started = utc_now()
    with source.open("rb") as handle:
        before = os.fstat(handle.fileno())
        # Fix the upper bound at open time: appends are deliberately excluded.
        source_bytes = handle.read(before.st_size)
        after = os.fstat(handle.fileno())
    if len(source_bytes) != before.st_size:
        raise RuntimeError(f"Source was truncated during snapshot: {source.name}")
    last_newline = source_bytes.rfind(b"\n")
    if last_newline < 0:
        raise ValueError(f"No complete JSONL record in {source.name}")
    data = source_bytes[: last_newline + 1]
    lines = data.splitlines()
    last_timestamp = None
    for line in lines:
        record = json.loads(line)  # Validate every copied record, keep raw bytes.
        if record.get("timestamp"):
            last_timestamp = record["timestamp"]
    atomic_write(destination, data)
    destination_hash = digest(destination.read_bytes())
    source_snapshot_hash = digest(data)
    if destination_hash != source_snapshot_hash:
        raise RuntimeError(f"Destination SHA-256 mismatch: {destination.name}")
    return {
        "source_path": str(source),
        "destination_file": destination.name,
        "snapshot_started_utc": started,
        "snapshot_finished_utc": utc_now(),
        "source_size_at_open_bytes": before.st_size,
        "source_size_after_read_bytes": after.st_size,
        "source_mtime_at_open_ns": before.st_mtime_ns,
        "source_mtime_after_read_ns": after.st_mtime_ns,
        "source_read_bytes": len(source_bytes),
        "source_read_sha256": digest(source_bytes),
        "copied_bytes": len(data),
        "copied_record_count": len(lines),
        "source_snapshot_sha256": source_snapshot_hash,
        "destination_sha256": destination_hash,
        "excluded_incomplete_trailing_bytes": len(source_bytes) - len(data),
        "last_record_timestamp": last_timestamp,
        "scope": "Original byte prefix through the last complete newline-delimited JSON record.",
        "later_appends_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", type=Path, default=Path("/home/wcf/.codex/sessions"))
    parser.add_argument("--session-index", type=Path, default=Path("/home/wcf/.codex/session_index.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/disk1/wuchufeng/C-TH-02/codex_sessions"))
    parser.add_argument("--root-id", action="append", help="Explicit root id; repeat to select more than one. Defaults to this project's verified root.")
    args = parser.parse_args()
    session_dir = args.sessions_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.is_relative_to(session_dir):
        parser.error("Output must not be inside the original sessions directory")
    root_ids = set(args.root_id or PROJECT_ROOT_IDS)
    sessions = discover(session_dir, root_ids)
    titles = read_titles(args.session_index.resolve(), set(sessions))
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    started = utc_now()
    for session_id, session in sessions.items():
        meta = session["meta"]
        parent = parent_info(meta)
        item = snapshot(session["path"], output_dir / session["path"].name)
        item.update({
            "session_id": session_id,
            "title": titles.get(session_id),
            "cwd": meta.get("cwd"),
            "parent_thread_id": parent.get("parent_thread_id"),
            "agent_path": parent.get("agent_path"),
            "selection_reason": "Explicitly verified project root id" if session_id in root_ids else "Descendant by session_meta parent_thread_id",
        })
        exported.append(item)
    manifest = {
        "schema_version": 1,
        "project": "C-TH-02 / th02-llm-judge",
        "export_started_utc": started,
        "export_finished_utc": utc_now(),
        "root_session_ids": sorted(root_ids),
        "session_count": len(exported),
        "total_copied_bytes": sum(item["copied_bytes"] for item in exported),
        "source_sessions_directory": str(session_dir),
        "selection_policy": "Only exact configured root ids and recursive explicit parent_thread_id descendants; no cwd/title-only matches or unrelated guardian sessions.",
        "snapshot_notice": "These are point-in-time snapshots. Later messages, subsequent tool calls, and the final response after the cutoff are not included. Run the export script again to refresh.",
        "export_script_sha256": digest(Path(__file__).read_bytes()),
        "files": exported,
    }
    atomic_write(output_dir / "manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
    index_lines = [
        "# 项目会话索引", "",
        f"导出时间（UTC）：{manifest['export_finished_utc']}。共 {len(exported)} 个会话。", "",
        "主会话由项目名称和用户请求核对；子会话仅按元数据中的父会话 ID 递归纳入。", "",
        "| 会话 | 文件 | 关联依据 |", "| --- | --- | --- |",
    ]
    for item in exported:
        label = item["title"] or item["agent_path"] or item["session_id"]
        label = label.replace("|", "\\|").replace("\n", " ")
        basis = "已核对的项目主会话" if item["session_id"] in root_ids else f"父 ID：`{item['parent_thread_id']}`"
        index_lines.append(f"| {label} | [{item['destination_file']}]({item['destination_file']}) | {basis} |")
    index_lines.extend(["", "快照截止后的新消息不包含在本次导出中；各文件的 SHA-256、字节数和记录数见 manifest.json。", ""])
    atomic_write(output_dir / "INDEX.md", "\n".join(index_lines).encode())
    print(json.dumps({"output_dir": str(output_dir), "session_count": len(exported), "copied_bytes": manifest["total_copied_bytes"], "manifest": str(output_dir / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
