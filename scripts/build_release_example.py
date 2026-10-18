#!/usr/bin/env python3
"""Publish a reproducible, signed model-data catalog for desktop/Android checks.

The Pixel-only second CPU release is a deliberate rollout example, not an NPU
performance claim. Fault-injection catalogs are kept under tests/ and never used
as the app's update feed.
"""

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phaseone.runtime import Runtime
from phaseone.runtime.catalog import Catalog, Release, TensorCheck, signed_catalog
from phaseone.runtime.package import atomic_json, create_key, digest_file, load_package
from phaseone.runtime.tensors import decode_tensor, preprocess_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(".work/vision-package"))
    parser.add_argument("--output", type=Path, default=Path(".work/release-server"))
    parser.add_argument("--key", type=Path, default=Path(".work/release-publisher.key"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = load_package(args.package)
    output = args.output.resolve()
    data = output / "vision"
    data.mkdir(parents=True, exist_ok=True)
    if not args.key.exists():
        create_key(args.key)
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    private = args.key.read_bytes()
    public = Ed25519PrivateKey.from_private_bytes(private).public_key().public_bytes_raw()
    (output / "publisher.pub").write_text(public.hex() + "\n")
    manifest = package.manifest.model_dump(mode="json", exclude_none=True)
    model = manifest["models"]["vision"]
    model["variants"] = [v for v in model["variants"] if v["target"].get("systems") != ["web"]]
    for variant in model["variants"]:
        variant["target"]["architectures"] = ["arm64", "x86_64"]
    for name, descriptor in manifest["artifacts"].items():
        shutil.copyfile(package.artifact(name), data / descriptor["path"])
    image = (root / "examples/vision/sample-dog.jpg").read_bytes()
    inputs = preprocess_image(image, package.manifest.models["vision"].image)
    with Runtime(output / ".reference-state") as runtime:
        app = runtime.load(args.package)
        result = app.infer("vision", {"image": inputs}, backend="executorch")
    logits = decode_tensor(result["outputs"]["logits"])
    inputs.astype("<f4").tofile(data / "qualification-input.f32")
    logits.astype("<f4").tofile(data / "qualification-output.f32")
    for name, file in (("qualification.input", "qualification-input.f32"), ("qualification.output", "qualification-output.f32")):
        manifest["artifacts"][name] = {"path": file, "sha256": digest_file(data / file), "size": (data / file).stat().st_size}
    check = TensorCheck.model_validate({"name": "imagenet-reference", "signature": "default",
        "inputs": {"image": {"artifact": "qualification.input", "shape": list(inputs.shape)}},
        "outputs": {"logits": {"artifact": "qualification.output", "shape": list(logits.shape)}}, "absolute_tolerance": 0.003})
    baseline = Release.model_validate({"id": "vision-1", "sequence": 1, "capability": "vision",
        "contract": "imagenet1k.rgb224.v1", "lineage": "torchvision.mobilenet-v3-small.imagenet1k-v1",
        "manifest": manifest, "model": "vision", "base": "vision/", "checks": [check.model_dump()]})
    rollout = baseline.model_copy(deep=True)
    rollout.id = "vision-2-tensor-g6-cpu"
    rollout.sequence = 2
    rollout.manifest.version = "1.1.0"
    chosen = next(v for v in rollout.manifest.models["vision"].variants if v.backend == "litert")
    chosen.id = "litert-tensor-g6-cpu"
    chosen.target.systems = ["android"]
    chosen.target.soc = "Tensor G6"
    chosen.target.minimum_android_api = 31
    rollout.manifest.models["vision"].variants = [chosen]
    catalogs = {
        "baseline.json": [baseline],
        "catalog.json": [baseline, rollout],
    }
    # Deliberate corrupted-transfer and quality failures, all signed by the test
    # publisher. These demonstrate separate integrity/qualification boundaries.
    corrupt = baseline.model_copy(deep=True)
    corrupt.id = "test-corrupt-transfer"
    corrupt.sequence = 3
    corrupt.manifest.artifacts["vision.executorch"].sha256 = "0" * 64
    bad_quality = baseline.model_copy(deep=True)
    bad_quality.id = "test-wrong-reference"
    bad_quality.sequence = 4
    import numpy as np
    np.zeros_like(logits, dtype="<f4").tofile(data / "wrong-reference.f32")
    reference = bad_quality.manifest.artifacts["qualification.output"]
    reference.path = "wrong-reference.f32"
    reference.sha256 = digest_file(data / reference.path)
    future = baseline.model_copy(deep=True)
    future.id = "test-native-upgrade"
    future.sequence = 5
    for variant in future.manifest.models["vision"].variants:
        variant.target.components["phaseone.test-future-native"] = ">=99.0.0"
    catalogs.update({"tests/corrupt.json": [baseline, corrupt], "tests/quality.json": [baseline, bad_quality], "tests/native-upgrade.json": [baseline, future]})
    for name, releases in catalogs.items():
        # A nested test catalog needs a base relative to itself; keep package
        # paths normalized and expose the same data under its own directory.
        catalog = Catalog(name="phaseone-vision", releases=copy.deepcopy(releases))
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(signed_catalog(catalog, private))
    test_data = output / "tests/vision"
    test_data.mkdir(exist_ok=True)
    for item in data.iterdir():
        target = test_data / item.name
        if target.exists():
            target.unlink()
        import os
        os.link(item, target)
    # Public key and a small JPEG are the only model-related APK assets.
    assets = root / "sdk/android/demo/src/main/assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "catalog-publisher.pub").write_text(public.hex() + "\n")
    (assets / "sample-dog.jpg").write_bytes(image)
    schemas = root / "sdk/contracts"
    schemas.mkdir(parents=True, exist_ok=True)
    atomic_json(schemas / "catalog.schema.json", Catalog.model_json_schema())
    fixture = baseline.model_dump(mode="json", exclude_none=True)
    atomic_json(schemas / "release-example.json", fixture)
    atomic_json(output / "reference-evidence.json", {"source_package": package.digest,
                "execution": result["execution"], "reference_top_index": int(logits.argmax()),
                "catalogs": list(catalogs), "note": "Release 2 is a deliberate CPU rollout to Tensor G6; test catalogs inject known failures."})
    print(json.dumps({"output": str(output), "catalogs": list(catalogs), "public_key": public.hex(),
                      "reference_top_index": int(logits.argmax())}, indent=2))


if __name__ == "__main__":
    main()
