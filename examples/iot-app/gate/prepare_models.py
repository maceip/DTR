"""Attach this application's tool contract to an existing DTR model package."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from gate_demo.core import TOOLS

from phaseone.runtime.package import load_package


def copy_artifact(original, target, output):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(original, target)
    except OSError:
        shutil.copy2(original, target)
    with target.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": target.relative_to(output).as_posix(), "sha256": digest, "size": target.stat().st_size}


def litert_manifest():
    variant = {"id": "litert-cpu", "backend": "litert", "engine": "llm", "artifact": "lt_model",
               "priority": 10, "target": {"systems": ["darwin", "linux"], "architectures": ["arm64"],
               "accelerator": "cpu", "minimum_memory_mb": 4096, "runtime_version": ">=0.18,<0.19"},
               "chat": {"context_length": 4096, "modalities": ["text", "image"],
                        "tool_protocol": "json", "max_images": 1, "vision_accelerator": "cpu"}}
    return {"schema_version": 1, "name": "gate-demo-models", "version": "0.3.0", "artifacts": {},
            "models": {name: {"task": "chat", "variants": [variant]} for name in ("screen", "judge")},
            "metadata": {"vision_model": "Gemma 4 E2B LiteRT-LM", "model_licenses": ["https://ai.google.dev/gemma/terms"]}}


def main():
    parser = argparse.ArgumentParser()
    source_args = parser.add_mutually_exclusive_group(required=True)
    source_args.add_argument("--source", type=Path, help="An existing DTR screen/judge model package")
    source_args.add_argument("--litert-model", type=Path, help="The downloaded Gemma 4 E2B .litertlm file")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=["litert", "executorch"])
    qnn_args = parser.add_mutually_exclusive_group()
    qnn_args.add_argument("--qnn-gemma3", type=Path, help="Sponsor QCS8275 Gemma 3 directory")
    qnn_args.add_argument("--qnn-gemma4", type=Path, help="Sponsor QCS8275 Gemma 4 directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    source = load_package(args.source) if args.source else None
    manifest = source.manifest.model_dump(mode="json", exclude_none=True) if source else litert_manifest()
    if not {"screen", "judge"}.issubset(manifest["models"]):
        parser.error("The source package must contain screen and judge chat models")
    if args.backend:
        for model in manifest["models"].values():
            model["variants"] = [v for v in model["variants"] if v["backend"] == args.backend]
            if not model["variants"]:
                parser.error("The source has no matching variant for a model")
    manifest["name"] = "gate-demo-models"
    manifest["tools"] = TOOLS
    manifest["permissions"] = [tool["permission"] for tool in TOOLS]
    if source:
        manifest.setdefault("metadata", {})["source_package_digest"] = source.digest
    args.output.mkdir(parents=True)
    for artifact in manifest["artifacts"].values():
        original = source.root / artifact["path"]
        target = args.output / artifact["path"]
        copy_artifact(original, target, args.output)
    if args.litert_model:
        manifest["artifacts"]["lt_model"] = copy_artifact(args.litert_model,
            args.output / "models/gemma4.litertlm", args.output)
    if args.qnn_gemma3 or args.qnn_gemma4:
        decoder = "gemma4" if args.qnn_gemma4 else "gemma3"
        model_file = "gemma4_e2b_QCS8275_ctxt_2000.pte" if args.qnn_gemma4 else "gemma3-1b_qcs8275_ctxt-5000.pte"
        files = {"qnn_model": model_file, "qnn_tokenizer": "tokenizer.json",
                 "qnn_tokenizer_config": "tokenizer_config.json", "qnn_template": "chat_template.jinja"}
        for identity, filename in files.items():
            original = (args.qnn_gemma4 or args.qnn_gemma3) / filename
            target = args.output / "models" / ("qnn-" + decoder) / filename
            manifest["artifacts"][identity] = copy_artifact(original, target, args.output)
        manifest["models"]["judge"]["variants"].insert(0, {
            "id": f"qnn-{decoder}-qcs8275", "backend": "executorch", "engine": "llm",
            "artifact": "qnn_model", "priority": 100,
            "target": {"systems": ["linux"], "architectures": ["arm64"],
                       "accelerator": "npu", "delegate": "QnnBackend", "soc": "QCS8275",
                       "runtime_version": ">=1.5,<1.6", "minimum_memory_mb": 4096,
                       "components": {"executorch.qnn_runner": "==1.5.0", "qnn.htp": "==2.46.0"}},
            "chat": {"runner": "qnn", "decoder": decoder, "context_length": 2000 if args.qnn_gemma4 else 5000,
                     "modalities": ["text"], "tokenizer": "qnn_tokenizer",
                     "tokenizer_config": "qnn_tokenizer_config", "template": "qnn_template",
                     "tool_protocol": "json"},
        })
        manifest["version"] = "0.3.0"
        manifest["metadata"]["qnn_judge"] = "Sponsor QCS8275 " + decoder
    (args.output / "phaseone.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(load_package(args.output).describe(), indent=2))


if __name__ == "__main__":
    main()
