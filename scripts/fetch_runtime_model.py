#!/usr/bin/env python3
"""Download explicitly selected public model assets at a pinned Hub revision.

LFS checksums are validated when published. Interrupted transfers are resumable;
unverified partial files never acquire the final artifact name.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch(repo: str, files: list[str], output: Path, revision: str = "main"):
    metadata = get_json(f"https://huggingface.co/api/models/{repo}/revision/{revision}")
    revision = metadata["sha"]
    tree = get_json(f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true")
    entries = {item["path"]: item for item in tree}
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for filename in files:
        entry = entries[filename]
        path = output / filename
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError("Unsafe filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = entry.get("lfs", {}).get("oid")
        size = entry["size"]
        partial = path.with_name(path.name + ".partial")
        if path.exists() and path.stat().st_size == size:
            digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            if expected is None or digest == expected:
                print(f"Verified existing {filename}", flush=True)
                records.append({"path": filename, "sha256": digest, "size": size})
                continue
        offset = partial.stat().st_size if partial.exists() else 0
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(filename)}"
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=90) as response:
            if offset and response.status != 206:
                offset = 0
            received, last_print = offset, time.monotonic()
            with partial.open("ab" if offset else "wb") as target:
                for chunk in iter(lambda: response.read(1024**2), b""):
                    target.write(chunk)
                    received += len(chunk)
                    if received > size:
                        raise ValueError(f"Download exceeds published size: {filename}")
                    if time.monotonic() - last_print > 15:
                        print(f"{filename}: {received}/{size} bytes", flush=True)
                        last_print = time.monotonic()
                target.flush()
                os.fsync(target.fileno())
        if received != size:
            raise ValueError(f"Incomplete download: {filename}: {received}/{size}")
        with partial.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if expected and digest != expected:
            raise ValueError(f"Published checksum mismatch: {filename}")
        os.replace(partial, path)
        records.append({"path": filename, "sha256": digest, "size": size})
        print(f"Verified {filename}: {digest}", flush=True)
    record = {"repository": repo, "revision": revision, "files": records}
    (output / "source.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    fetch(args.repo, args.files, args.output, args.revision)
