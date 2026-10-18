#!/usr/bin/env python3
"""Real native desktop execution through the same release catalog used by Android."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phaseone.runtime import ReleasePolicy, Runtime, RuntimeFault


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(".work/release-server"))
    parser.add_argument("--state", type=Path, default=Path(".work/release-desktop-check"))
    parser.add_argument("--output", type=Path, default=Path("evidence/runtime-release-desktop.json"))
    args = parser.parse_args()
    source = args.source.resolve()
    key = bytes.fromhex((source / "publisher.pub").read_text().strip())
    with Runtime(args.state, catalog_keys={hashlib.sha256(key).hexdigest(): key}, local_development=True) as runtime:
        policy = ReleasePolicy(contract="imagenet1k.rgb224.v1", allowed_lineages=["torchvision.mobilenet-v3-small.imagenet1k-v1"])
        report = {"device": runtime.device, "checks": []}
        for backend in ("executorch", "litert"):
            selected = policy.model_copy(update={"backend": backend})
            update = runtime.releases.update((source / "baseline.json").as_uri(), "vision", selected)
            result = runtime.open("vision").classify("vision", Path("examples/vision/sample-dog.jpg").read_bytes())
            assert result["predictions"][0]["label"] == "Samoyed"
            assert result["execution"]["backend"] == backend
            report["checks"].append({"backend": backend, "update": update, "execution": result["execution"],
                                     "prediction": result["predictions"][0]})
        report["rollback"] = runtime.releases.rollback("vision")
        before = runtime.releases.state("vision")["active"]
        report["fault_injection_checks"] = []
        for name, expected in (("corrupt", "artifact_integrity"), ("quality", "qualification_failed")):
            try:
                runtime.releases.update((source / f"tests/{name}.json").as_uri(), "vision", policy.model_copy(update={"backend": "executorch"}))
                raise AssertionError("Fault injection was not rejected")
            except RuntimeFault as exc:
                assert exc.code == expected, exc
                assert runtime.releases.state("vision")["active"] == before
                report["fault_injection_checks"].append({"name": name, "expected_code": expected, "passed": True})
        fallback = runtime.releases.update((source / "tests/quality.json").as_uri(), "vision", policy.model_copy(update={"backend": "executorch", "allow_fallback": True}))
        assert fallback["status"] == "retained", fallback
        assert runtime.releases.state("vision")["active"] == before
        report["explicit_fallback"] = fallback
        # Deliberate SDK-build change exercises offline requalification. It does
        # not impersonate a different operating system or acceleration device.
        runtime.device = {**runtime.device, "runtime_build": runtime.device["runtime_build"] + "-upgrade-fixture"}
        runtime.open("vision").classify("vision", Path("examples/vision/sample-dog.jpg").read_bytes())
        report["sdk_update_requalification_fixture"] = runtime.releases.state("vision")["active"]
        future = runtime.releases.check((source / "tests/native-upgrade.json").as_uri(), "vision", policy)
        assert future["status"] == "app_update_required", future
        report["native_update_boundary_fixture"] = future
        report["passed"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": True, "backends": [c["backend"] for c in report["checks"]], "output": str(args.output)}))


if __name__ == "__main__":
    main()
