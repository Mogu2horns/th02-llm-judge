#!/usr/bin/env python3
"""Download or verify the fixed MiniLM encoder using official Hugging Face URLs.

Existing files are checked against the pinned hashes before reuse. A fully
verified existing model needs no network access and keeps its provenance intact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


REPO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
API_URL = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "models/all-MiniLM-L6-v2"

# Sizes and Git/LFS digests are from the official API at the fixed revision.
# SHA256 values for ordinary Git blobs were also verified when first downloaded.
FILES = {
    "config.json": (
        612, "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
        "72b987fd805cfa2b58c4c8c952b274a11bfd5a00",
    ),
    "model.safetensors": (
        90868376, "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        None,
    ),
    "tokenizer.json": (
        466247, "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        "cb202bfe2e3c98645018a6d12f182a434c9d3e02",
    ),
    "tokenizer_config.json": (
        350, "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
        "c79f2b6a0cea6f4b564fed1938984bace9d30ff0",
    ),
    "special_tokens_map.json": (
        112, "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
        "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5",
    ),
    "vocab.txt": (
        231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        "fb140275c155a9c7c5a3b3e0e77a9e839594a938",
    ),
}


class DownloadError(RuntimeError):
    """An error message safe to print without signed redirect URLs."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_url(name: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"


def verified(path: Path, name: str) -> bool:
    size, expected_sha256, expected_git = FILES[name]
    if not path.is_file() or path.stat().st_size != size:
        return False
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    return sha256.hexdigest() == expected_sha256 and (
        expected_git is None or git_blob.hexdigest() == expected_git
    )


def check_official_metadata(requests) -> None:
    try:
        response = requests.get(API_URL, timeout=(20, 60))
        if response.status_code != 200:
            raise DownloadError(f"Official metadata HTTP status {response.status_code}")
        metadata = response.json()
        if metadata.get("sha") != REVISION:
            raise DownloadError("Official metadata revision does not match")
        siblings = {item["rfilename"]: item for item in metadata["siblings"]}
        for name, (size, sha256, git_blob) in FILES.items():
            item = siblings[name]
            digest = item["lfs"]["sha256"] if git_blob is None else item["blobId"]
            if item["size"] != size or digest != (sha256 if git_blob is None else git_blob):
                raise DownloadError(f"Official metadata differs from pinned values: {name}")
    except DownloadError:
        raise
    except Exception as error:
        # requests exceptions can contain expiring, signed redirect URLs.
        raise DownloadError(f"Could not read official metadata ({type(error).__name__})") from None


def download(requests, output: Path, name: str) -> str:
    target = output / name
    temporary = output / (name + ".part")
    for attempt in range(1, 4):
        try:
            with requests.get(source_url(name), stream=True, timeout=(20, 120)) as response:
                if response.status_code != 200:
                    raise DownloadError(f"{name}: HTTP status {response.status_code}")
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            if not verified(temporary, name):
                raise DownloadError(f"{name}: downloaded size or digest does not match")
            temporary.replace(target)
            print(f"{name}: downloaded and verified ({FILES[name][0]} bytes)", flush=True)
            return utc_now()
        except Exception as error:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                message = str(error) if isinstance(error, DownloadError) else type(error).__name__
                raise DownloadError(f"Download failed: {message}") from None
            print(f"{name}: retry {attempt}/3 ({type(error).__name__})", flush=True)
            time.sleep(attempt * 2)
    raise AssertionError("Unreachable")


def write_provenance(output: Path, previous: dict, downloaded: dict[str, str]) -> None:
    old_files = {item["file"]: item for item in previous.get("files", [])}
    files = []
    for name, (size, sha256, git_blob) in FILES.items():
        old = old_files.get(name, {})
        entry = dict(old) if old.get("sha256") == sha256 else {}
        entry.update({
            "file": name, "size": size, "sha256": sha256,
            "source_url": source_url(name),
            "official_digest_algorithm": "sha256" if git_blob is None else "git-blob-sha1",
            "official_digest": sha256 if git_blob is None else git_blob,
            "verified": True,
        })
        if name in downloaded:
            entry["downloaded_at_utc"] = downloaded[name]
            entry["acquisition"] = "downloaded_from_official_url"
        elif not old:
            # A verified preexisting file does not prove when or how it was downloaded.
            entry["acquisition"] = "verified_existing_file"
            entry["verified_at_utc"] = utc_now()
        files.append(entry)
    provenance = dict(previous)
    provenance.update({
        "repo_id": REPO, "revision": REVISION, "official_metadata_url": API_URL,
        "files": sorted(files, key=lambda item: item["file"]), "all_verified": True,
    })
    temporary = output / "provenance.json.part"
    temporary.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "provenance.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    try:
        missing = []
        for name in FILES:
            if verified(output / name, name):
                print(f"{name}: verified existing file ({FILES[name][0]} bytes)", flush=True)
            else:
                missing.append(name)
        provenance_path = output / "provenance.json"
        previous = {}
        if provenance_path.is_file():
            previous = json.loads(provenance_path.read_text(encoding="utf-8"))
            if previous.get("repo_id") != REPO or previous.get("revision") != REVISION:
                raise DownloadError("Existing provenance identifies a different model or revision")
        if not missing and previous:
            print("All six files verified; existing provenance unchanged.")
            return 0
        downloaded = {}
        if missing:
            import requests  # Already included by the project's training dependencies.

            check_official_metadata(requests)
            output.mkdir(parents=True, exist_ok=True)
            for name in missing:
                downloaded[name] = download(requests, output, name)
        write_provenance(output, previous, downloaded)
        print(f"All six files verified; provenance saved in {output / 'provenance.json'}")
        return 0
    except DownloadError as error:
        print(f"Error: {error}", file=sys.stderr)
    except Exception as error:
        print(f"Error: {type(error).__name__}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
