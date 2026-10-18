"""Host-side worker protocol: crashes, hangs, cancellation and reload.

The spawned child runs `scripted_worker` instead of a native engine, so these
run without ExecuTorch or LiteRT. Native load/inference is in test_runtime_native.
"""

import os
import threading
import time
from types import SimpleNamespace

import pytest

from phaseone.runtime import Runtime, RuntimeFault
from phaseone.runtime import worker as worker_module
from phaseone.runtime.worker import Worker


def scripted_worker(pipe, package_path, expected_digest, variant_data, cache_path):
    # Must be importable by the spawned child, so it lives at module level.
    mode = variant_data["mode"]
    if mode == "load-crash":
        os._exit(7)
    if mode == "load-hang":
        time.sleep(60)
    pipe.send({"type": "ready", "info": {"pid": os.getpid()}, "load_ms": 0.0})
    while True:
        try:
            request = pipe.recv()
        except EOFError:
            return
        operation = request["operation"]
        if operation == "crash":
            os._exit(3)
        if operation == "hang":
            time.sleep(60)
        if operation == "fail":
            pipe.send({"type": "error", "error": RuntimeFault("inference_failed", "scripted").as_dict()})
            continue
        pipe.send({"type": "token", "text": "partial"})
        if operation == "stall":
            time.sleep(60)
        pipe.send({"type": "done", "pid": os.getpid()})


class ScriptedVariant:
    def __init__(self, mode):
        self.id = mode
        self.mode = mode

    def model_dump(self, mode="json"):
        return {"mode": self.mode}


@pytest.fixture(autouse=True)
def scripted(monkeypatch):
    monkeypatch.setattr(worker_module, "worker_main", scripted_worker)


@pytest.fixture
def package(tmp_path):
    return SimpleNamespace(root=tmp_path, digest="0" * 64)


def start(package, tmp_path, mode="ready", **options):
    return Worker(package, ScriptedVariant(mode), tmp_path / "cache", **options)


def request(worker, operation, **options):
    return list(worker.events({"operation": operation}, timeout=options.pop("timeout", 10), **options))


def assert_stopped(worker):
    assert worker.closed
    assert not worker.process.is_alive()


def test_completed_requests_reuse_the_same_process(package, tmp_path):
    worker = start(package, tmp_path)
    try:
        first, second = request(worker, "run"), request(worker, "run")
        assert [e["type"] for e in first] == ["token", "done"]
        assert first[-1]["pid"] == second[-1]["pid"] == worker.process.pid
        # A reported error is a completed request: the process stays usable.
        with pytest.raises(RuntimeFault) as error:
            request(worker, "fail")
        assert error.value.code == "inference_failed"
        assert request(worker, "run")[-1]["pid"] == worker.process.pid
    finally:
        worker.close()


def test_native_crash_is_reported_and_isolated(package, tmp_path):
    worker = start(package, tmp_path)
    with pytest.raises(RuntimeFault) as error:
        request(worker, "crash")
    assert error.value.code == "worker_crashed"
    assert error.value.retryable
    assert error.value.details["exit_code"] == 3
    assert_stopped(worker)
    with pytest.raises(RuntimeFault) as error:
        request(worker, "run")
    assert error.value.code == "worker_closed"


def test_hung_inference_is_terminated_at_its_deadline(package, tmp_path):
    worker = start(package, tmp_path)
    started = time.monotonic()
    with pytest.raises(RuntimeFault) as error:
        request(worker, "hang", timeout=0.5)
    assert error.value.code == "inference_timeout"
    assert time.monotonic() - started < 5
    assert_stopped(worker)


def test_cancellation_stops_running_inference(package, tmp_path):
    worker = start(package, tmp_path)
    cancellation = threading.Event()
    threading.Timer(0.3, cancellation.set).start()
    with pytest.raises(RuntimeFault) as error:
        request(worker, "hang", cancellation=cancellation)
    assert error.value.code == "cancelled"
    assert_stopped(worker)


def test_abandoned_stream_cannot_keep_generating(package, tmp_path):
    worker = start(package, tmp_path)
    events = worker.events({"operation": "stall"}, timeout=10)
    assert next(events)["type"] == "token"
    events.close()  # e.g. the HTTP client disconnected mid-stream.
    assert_stopped(worker)
    assert not worker.lock.locked()


@pytest.mark.parametrize(("mode", "code"), [("load-crash", "worker_crashed"), ("load-hang", "inference_timeout")])
def test_load_failures_do_not_leak_processes(package, tmp_path, monkeypatch, mode, code):
    created = []
    original = Worker.__init__

    def tracking(self, *args, **kwargs):
        created.append(self)
        original(self, *args, **kwargs)

    monkeypatch.setattr(Worker, "__init__", tracking)
    with pytest.raises(RuntimeFault) as error:
        start(package, tmp_path, mode, load_timeout=1)
    assert error.value.code == code
    assert_stopped(created[0])


def test_runtime_replaces_a_crashed_worker_on_the_next_request(package, tmp_path):
    with Runtime(tmp_path / "state") as runtime:
        variant = ScriptedVariant("ready")
        first = runtime._worker(package, variant)
        first.reservations -= 1
        with pytest.raises(RuntimeFault):
            request(first, "crash")

        second = runtime._worker(package, variant)
        second.reservations -= 1
        assert second is not first
        assert request(second, "run")[-1]["pid"] == second.process.pid != first.process.pid
