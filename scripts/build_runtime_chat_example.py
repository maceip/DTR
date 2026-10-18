#!/usr/bin/env python3
"""Package the downloaded/exported SmolLM2 variants behind one logical assistant."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def build(root: Path, destination: Path, backends=("executorch", "litert")):
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    def add(name, source):
        target = destination / source.name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        artifacts[name] = {"path": target.name, "size": target.stat().st_size, "sha256": digest}
    add("tokenizer", root / "smollm-source/tokenizer.json")
    add("tokenizer_config", root / "smollm-source/tokenizer_config.json")
    variants = []
    for backend in backends:
        if backend == "executorch":
            add("et", root / "smollm-executorch/smollm2.pte")
        else:
            add("lite", root / "smollm-litert/SmolLM2_135M_Instruct.litertlm")
        variants.append({"id": f"{backend}-cpu", "backend": backend, "engine": "llm",
                         "artifact": "et" if backend == "executorch" else "lite",
                         "target": {"accelerator": "cpu", "minimum_memory_mb": 1536,
                                    "runtime_version": ">=1.5.1,<1.6" if backend == "executorch" else ">=0.18,<0.19"},
                         "chat": {"context_length": 1024, "tokenizer": "tokenizer",
                                  "tokenizer_config": "tokenizer_config", "tool_protocol": "json"},
                         "priority": 20 if backend == "executorch" else 10})
    manifest = {
        "schema_version": 1, "name": "local-assistant", "version": "1.0.0",
        "description": "One local assistant and durable conversation across ExecuTorch and LiteRT-LM",
        "artifacts": artifacts, "models": {"assistant": {"task": "chat", "variants": variants}},
        "tools": [{"name": "device_info", "description": "Report the device and installed inference runtimes.",
                   "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                   "permission": "device.read", "side_effect": False}],
        "permissions": ["device.read"],
        "agent": {"model": "assistant", "instruction": "You are a concise local device assistant. Use device_info for device facts.",
                  "tools": ["device_info"], "max_steps": 4},
        "metadata": {"source": "HuggingFaceTB/SmolLM2-135M-Instruct", "license": "Apache-2.0",
                     "note": "A small real model for integration; tool-use quality depends on the model."},
    }
    (destination / "phaseone.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--work", type=Path, default=Path(".work"))
    parser.add_argument("--backend", action="append", choices=["executorch", "litert"])
    args = parser.parse_args()
    print(build(args.work, args.destination, args.backend or ("executorch", "litert")))
