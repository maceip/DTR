"""Contract tests execute real native programs, never mock either backend."""

import json

import numpy as np
import pytest

from phaseone.runtime import Runtime, RuntimeFault
from phaseone.runtime.tensors import decode_tensor


@pytest.fixture(scope="module")
def native_package(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("executorch")
    pytest.importorskip("ai_edge_litert")
    from scripts.build_runtime_fixture import build

    return build(tmp_path_factory.mktemp("native-package"))


@pytest.fixture(scope="module")
def native_runtime(native_package, tmp_path_factory):
    with Runtime(tmp_path_factory.mktemp("native-host")) as runtime:
        yield runtime, runtime.load(native_package)


@pytest.mark.parametrize("variant", ["executorch-program", "litert-compiled", "litert-interpreter"])
def test_real_methods_and_names(native_runtime, variant):
    runtime, app = native_runtime
    left = np.array([[1.0, 2.5, -4.0, 8.0]], dtype=np.float32)
    right = np.array([[0.5, -2.0, 3.0, 9.0]], dtype=np.float32)
    for signature, expected in (("default", left + right), ("difference", left - right)):
        result = app.infer("calculator", {"left": left, "right": right},
                           variant=variant, signature=signature)
        np.testing.assert_array_equal(decode_tensor(result["outputs"]["output"]), expected)
        assert result["execution"]["variant"] == variant
        assert result["execution"]["requested_accelerator"] == "cpu"
        assert result["execution"]["execution_verified"]
        assert not result["execution"]["npu_execution_verified"]
    assert all(worker.reservations == 0 for worker in runtime.workers.values())


def test_rejects_wrong_shape_without_native_crash(native_runtime):
    _, app = native_runtime
    with pytest.raises(RuntimeFault, match="incompatible shape"):
        app.infer("calculator", {"left": [1, 2], "right": [1, 2]}, backend="executorch")
    result = app.infer("calculator", {"left": [[1, 2, 3, 4]], "right": [[1, 1, 1, 1]]})
    assert result["outputs"]["output"]["data"] == [2, 3, 4, 5]


@pytest.mark.parametrize("backend", ["executorch", "litert"])
def test_initial_load_timeout_and_cancellation_reload(native_package, tmp_path, backend):
    import threading
    import time

    with Runtime(tmp_path / "host") as runtime:
        app = runtime.load(native_package)
        inputs = {"left": [[1, 2, 3, 4]], "right": [[1, 1, 1, 1]]}
        started = time.monotonic()
        with pytest.raises(RuntimeFault) as error:
            app.infer("calculator", inputs, backend=backend, timeout_seconds=0.01)
        assert error.value.code == "inference_timeout"
        assert time.monotonic() - started < 5
        assert not runtime.workers
        cancellation = threading.Event()
        cancellation.set()
        with pytest.raises(RuntimeFault) as error:
            app.infer("calculator", inputs, backend=backend, cancellation=cancellation)
        assert error.value.code == "cancelled"
        result = app.infer("calculator", inputs, backend=backend)
        assert result["outputs"]["output"]["data"] == [2, 3, 4, 5]


def test_variant_identity_is_not_only_its_name(native_package, tmp_path):
    manifest = json.loads((native_package / "phaseone.json").read_text())
    et = manifest["models"]["calculator"]["variants"][0]
    lite = manifest["models"]["calculator"]["variants"][1]
    et["id"] = lite["id"] = "cpu"
    manifest["models"] = {
        "add_et": {"task": "tensor", "variants": [et]},
        "add_lite": {"task": "tensor", "variants": [lite]},
    }
    import shutil

    source = tmp_path / "package"
    shutil.copytree(native_package, source)
    (source / "phaseone.json").write_text(json.dumps(manifest))
    with Runtime(tmp_path / "host") as runtime:
        app = runtime.load(source)
        for model, backend in (("add_et", "executorch"), ("add_lite", "litert")):
            result = app.infer(model, {"left": [[1, 1, 1, 1]], "right": [[2, 2, 2, 2]]})
            assert result["execution"]["backend"] == backend
            assert result["outputs"]["output"]["data"] == [3, 3, 3, 3]


def test_http_auth_contract_and_both_backends(native_runtime):
    from fastapi.testclient import TestClient

    from phaseone.runtime.api import create_app

    runtime, application = native_runtime
    token = "test-token-" + "x" * 40
    with TestClient(create_app(runtime, token=token)) as client:
        assert client.get("/health").json()["product"] == "PHASEONE Runtime"
        assert client.get("/v1/device").status_code == 401
        headers = {"Authorization": "Bearer " + token}
        assert client.get("/v1/device", headers={**headers, "Origin": "https://unpaired.invalid"}).status_code == 403
        digest = application.package.digest
        path = f"/v1/packages/{digest}/infer"
        for backend in ("executorch", "litert"):
            result = client.post(path, headers=headers, json={"model": "calculator", "backend": backend,
                "inputs": {"left": [[1, 2, 3, 4]], "right": [[4, 3, 2, 1]]}})
            assert result.status_code == 200, result.text
            body = result.json()
            assert body["execution"]["backend"] == backend
            assert body["outputs"]["output"]["data"] == [5, 5, 5, 5]
            assert body["request_id"]
        failed = client.post(path, headers=headers, json={"model": "calculator", "accelerator": "npu", "inputs": {}})
        assert failed.status_code == 422
        assert failed.json()["error"]["code"] == "no_compatible_variant"
        assert client.get("/v1/receipts/verify", headers=headers).json()["valid"]


@pytest.mark.parametrize("mode", ["quantized", "dynamic"])
def test_native_quantized_and_dynamic_inputs(tmp_path, mode):
    pytest.importorskip("executorch")
    pytest.importorskip("ai_edge_litert")
    from scripts.build_runtime_fixture import build

    source = build(tmp_path / "package", quantized=mode == "quantized", dynamic=mode == "dynamic")
    with Runtime(tmp_path / "host") as runtime:
        app = runtime.load(source)
        for variant in ("executorch-program", "litert-compiled", "litert-interpreter"):
            for batch in ((1, 3, 8) if mode == "dynamic" else (1,)):
                left = np.ones((batch, 4), np.float32)
                right = np.full((batch, 4), 2.5, np.float32)
                result = app.infer("calculator", {"left": left, "right": right}, variant=variant)
                np.testing.assert_allclose(decode_tensor(result["outputs"]["output"]), left + right)
