"""Discover a host-installed QNN runner; model packages cannot provide executables."""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from .errors import RuntimeFault
from .schema import relative_path


def qnn_installation(*, verify=False):
    root = Path(os.environ.get("LOCALKIT_QNN_ROOT", str(Path(sys.prefix).parent / "native/qnn"))).resolve()
    profile = root / "runtime.json"
    if not profile.is_file():
        return None
    try:
        if profile.stat().st_size > 128 * 1024:
            raise ValueError("Runtime profile is too large")
        document = json.loads(profile.read_text())
        if document["format_version"] != 1:
            raise ValueError("Unknown QNN runtime profile")
        from packaging.version import Version

        Version(document["executorch_version"])
        Version(document["qnn_version"])
        if not isinstance(document["files"], dict) or not 1 <= len(document["files"]) <= 256:
            raise ValueError("Invalid runtime file inventory")
        for name in [document["runner"], *document["files"]]:
            relative_path(name)
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing runtime file: {name}")
        if document["runner"] not in document["files"]:
            raise ValueError("Runner is not covered by the installed file hashes")
        if not os.access(root / document["runner"], os.X_OK):
            raise ValueError("Installed QNN runner is not executable")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in document["files"].values()):
            raise ValueError("Invalid runtime file hash")
        if verify:
            for name, expected in document["files"].items():
                with (root / name).open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                if digest != expected:
                    raise ValueError(f"Runtime file changed: {name}")
        return root, document
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if verify:
            raise RuntimeFault("qnn_installation_invalid", str(exc)) from exc
        return None
