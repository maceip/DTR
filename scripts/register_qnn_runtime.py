#!/usr/bin/env python3
"""Register already installed ARM64 QNN libraries and runners with DTR."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from packaging.version import Version


def register(root, executorch_version, qnn_version, bundle_sha256=None):
    root = Path(root).resolve()
    Version(executorch_version)
    Version(qnn_version)
    required = ["bin/qnn_llama_runner", "lib/libqnn_executorch_backend.so",
                "lib/libQnnHtp.so", "lib/libQnnSystem.so"]
    for name in required:
        if not (root / name).is_file():
            raise ValueError(f"Missing installed runtime file: {name}")
    if not list((root / "dsp").glob("libQnnHtpV*Skel.so")):
        raise ValueError("Install the matching Hexagon DSP skeleton library in dsp/")
    runner = root / required[0]
    with runner.open("rb") as source:
        header = source.read(20)
    if header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 183:
        raise ValueError("The runner must be a Linux ARM64 ELF binary")
    if not os.access(runner, os.X_OK):
        raise ValueError("Make the installed runner executable before registering it")
    files = {}
    for folder in ("bin", "lib", "dsp"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Copy this dependency into the runtime directory: {path}")
            with path.open("rb") as source:
                files[path.relative_to(root).as_posix()] = hashlib.file_digest(source, "sha256").hexdigest()
    profile = {"format_version": 1, "runner": required[0],
               "executorch_version": executorch_version, "qnn_version": qnn_version, "files": files}
    if bundle_sha256:
        if len(bundle_sha256) != 64 or any(c not in "0123456789abcdef" for c in bundle_sha256):
            raise ValueError("Expected a SHA-256 bundle digest")
        profile["source_bundle_sha256"] = bundle_sha256
    temporary = root / "runtime.json.tmp"
    temporary.write_text(json.dumps(profile, indent=2) + "\n")
    temporary.replace(root / "runtime.json")
    return {"runtime": str(root), "files": len(files), "executorch": executorch_version,
            "qnn": qnn_version}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--executorch-version", required=True)
    parser.add_argument("--qnn-version", required=True)
    parser.add_argument("--bundle-sha256")
    args = parser.parse_args()
    print(json.dumps(register(args.root, args.executorch_version, args.qnn_version, args.bundle_sha256)))
