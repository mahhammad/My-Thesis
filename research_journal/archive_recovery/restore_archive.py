#!/usr/bin/env python3
"""Verify or restore archived thesis evidence from the frozen Git tag.

Default: read-only verification, no extraction and no model execution.
Use --dest /outside/repo/path for an explicit, non-overwriting restoration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "ARCHIVE_RECOVERY_2026-11-15.json"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)


def blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def read_verified(item: dict, tag: str) -> bytes:
    source = item["source"]
    if source["type"] == "tag":
        data = git("show", f"{tag}:{source['path']}")
    elif source["type"] == "repo_file":
        path = (ROOT / source["path"]).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), path
        data = path.read_bytes()
    else:
        raise ValueError(f"Unknown source: {source}")
    if len(data) != item["bytes"] or blob_sha1(data) != item["git_blob_sha1"]:
        raise ValueError(f"Content or Git blob mismatch: {item['archive_path']}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, help="Create restored archive under this NEW directory")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text("utf-8"))
    tag = manifest["recovery_tag"]
    tag_commit = git("rev-parse", f"refs/tags/{tag}^{{commit}}").decode().strip()
    if tag_commit != manifest["recovery_commit"]:
        raise ValueError(f"Recovery tag unexpectedly moved: {tag_commit}")
    files = manifest["file_records"]
    if len(files) != manifest["expected_files"]:
        raise ValueError("Manifest file count mismatch")
    if sum(item["bytes"] for item in files) != manifest["expected_bytes"]:
        raise ValueError("Manifest byte count mismatch")
    if len({item["archive_path"] for item in files}) != len(files):
        raise ValueError("Repeated archive paths")
    dest = args.dest.resolve() if args.dest else None
    if dest and dest.exists():
        raise FileExistsError(f"Refuse to overwrite existing directory: {dest}")
    # Fully validate ALL sources before writing any archive contents.
    payloads = [(item, read_verified(item, tag)) for item in files]
    if dest:
        dest.mkdir(parents=True)
        prefix = manifest["archive_prefix"]
        for item, data in payloads:
            name = item["archive_path"]
            if not name.startswith(prefix):
                raise ValueError(f"Unsafe path: {name}")
            target = (dest / name[len(prefix):]).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"Path escaped destination: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print(json.dumps({
        "status": "PASS",
        "tag": tag,
        "commit": tag_commit,
        "verified_files": len(payloads),
        "verified_bytes": sum(len(data) for _, data in payloads),
        "restored": bool(dest),
        "dest": str(dest) if dest else None,
        "training": False,
        "test_access": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
