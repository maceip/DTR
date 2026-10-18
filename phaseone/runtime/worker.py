"""Crash-isolated native inference processes with bounded requests and cancellation.

This is process fault isolation, not an OS security sandbox. Packages and native
runtime binaries are trusted code; tools and their credentials remain in the host.
"""

import multiprocessing
import os
import threading
import time
from pathlib import Path

from .errors import RuntimeFault


def native_failure(exc, cache, code):
    details = {"native_log": str(cache / "native.log")}
    causes, current = [], exc
    while current is not None and len(causes) < 8:
        causes.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    if any("cannot open shared object file" in cause or "Library not loaded:" in cause for cause in causes):
        code = "native_library_unavailable"
        details["causes"] = causes
        if any("libatomic.so.1" in cause for cause in causes):
            details["remedy"] = "On Ubuntu/Debian, install the libatomic1 system package."
    try:
        with (cache / "native.log").open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 128 * 1024))
            lines = log.read().decode("utf-8", errors="replace").splitlines()
        missing = sorted({line.split("Missing operator:", 1)[1].strip() for line in lines if "Missing operator:" in line})
        if missing:
            code = "native_kernel_unavailable"
            details["missing_operators"] = missing[:64]
        elif any("xnn_status_unsupported_hardware" in line for line in lines):
            code = "hardware_unsupported"
            details["reason"] = "This XNNPACK export requires kernels unsupported by the installed backend on this host."
    except OSError:
        pass
    return RuntimeFault(code, " <- ".join(causes), details=details)


def worker_main(pipe, package_path: str, expected_digest: str, variant_data: dict, cache_path: str):
    # Keep runtime/library discovery variables, excluding ambient service credentials.
    keep = {"PATH", "HOME", "TMPDIR", "TMP", "TEMP", "SYSTEMROOT", "WINDIR",
            "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "ADSP_LIBRARY_PATH", "QNN_SDK_ROOT",
            "PYTHONPATH", "VIRTUAL_ENV", "LANG", "LC_ALL", "LITERT_DISPATCH_LIB_DIR"}
    environment = {key: value for key, value in os.environ.items() if key in keep}
    os.environ.clear()
    os.environ.update(environment)
    cache = Path(cache_path)
    cache.mkdir(parents=True, exist_ok=True)
    log = os.open(cache / "native.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.close(log)
    provider = None
    try:
        from .backends import create
        from .package import load_package
        from .schema import Variant

        package = load_package(package_path)
        if package.digest != expected_digest:
            raise RuntimeFault("package_changed", "Package changed after host verification")
        variant = Variant.model_validate(variant_data)
        start = time.monotonic()
        provider = create(package, variant, cache)
        pipe.send({"type": "ready", "info": provider.info,
                   "load_ms": (time.monotonic() - start) * 1000})
        while True:
            try:
                request = pipe.recv()
            except EOFError:
                break
            if request["operation"] == "close":
                break
            started = time.monotonic()
            try:
                if request["operation"] == "infer":
                    result = provider.infer(request["signature"], request["inputs"])
                    pipe.send({"type": "done", "outputs": result,
                               "inference_ms": (time.monotonic() - started) * 1000})
                elif request["operation"] == "chat":
                    result = provider.chat(request["messages"], request["generation"],
                                           request.get("tools", []), pipe.send)
                    pipe.send({"type": "done", **result,
                               "inference_ms": (time.monotonic() - started) * 1000})
                else:
                    raise RuntimeFault("invalid_operation", "Unknown worker operation")
            except RuntimeFault as exc:
                pipe.send({"type": "error", "error": exc.as_dict()})
            except Exception as exc:
                pipe.send({"type": "error", "error": native_failure(exc, cache, "inference_failed").as_dict()})
    except RuntimeFault as exc:
        pipe.send({"type": "error", "error": exc.as_dict()})
    except Exception as exc:
        pipe.send({"type": "error", "error": native_failure(exc, cache, "backend_load_failed").as_dict()})
    finally:
        if provider is not None:
            provider.close()
        pipe.close()


class Worker:
    def __init__(self, package, variant, cache: Path, *, load_timeout: float = 180, cancellation=None):
        self.lock = threading.Lock()
        self.reservations = 0
        self.closed = False
        context = multiprocessing.get_context("spawn")
        self.pipe, child_pipe = context.Pipe()
        self.process = context.Process(target=worker_main,
                                       args=(child_pipe, str(package.root), package.digest,
                                             variant.model_dump(mode="json"), str(cache)),
                                       daemon=True)
        self.process.start()
        child_pipe.close()
        try:
            ready = self.receive(time.monotonic() + load_timeout, cancellation)
            if ready["type"] == "error":
                raise RuntimeFault.from_dict(ready["error"])
            if ready["type"] != "ready":
                raise RuntimeFault("worker_protocol", "Native worker did not initialize")
            self.info, self.load_ms = ready["info"], ready["load_ms"]
        except BaseException:
            self.close()
            raise

    def receive(self, deadline: float, cancellation: threading.Event | None = None):
        while True:
            if cancellation and cancellation.is_set():
                self.close()
                raise RuntimeFault("cancelled", "Inference was cancelled")
            if self.closed:
                raise RuntimeFault("worker_closed", "Native worker is closed", retryable=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise RuntimeFault("inference_timeout", "Inference exceeded its time budget", retryable=True)
            if self.pipe.poll(min(remaining, 0.05)):
                try:
                    return self.pipe.recv()
                except (EOFError, OSError) as exc:
                    self.close()
                    raise RuntimeFault("worker_crashed", "Native inference process exited",
                                       details={"exit_code": self.process.exitcode}, retryable=True) from exc
            if not self.process.is_alive():
                code = self.process.exitcode
                self.close()
                raise RuntimeFault("worker_crashed", "Native inference process exited",
                                   details={"exit_code": code}, retryable=True)

    def events(self, request: dict, *, timeout: float, cancellation=None):
        deadline = time.monotonic() + timeout
        while not self.lock.acquire(timeout=min(timeout, 0.05)):
            if cancellation and cancellation.is_set():
                raise RuntimeFault("cancelled", "Request was cancelled while queued")
            if time.monotonic() >= deadline:
                raise RuntimeFault("queue_timeout", "Model is busy; request exceeded its time budget", retryable=True)
        complete = False
        try:
            if self.closed:
                raise RuntimeFault("worker_closed", "Native worker must be reloaded", retryable=True)
            self.pipe.send(request)
            while True:
                event = self.receive(deadline, cancellation)
                if event["type"] == "error":
                    complete = True
                    raise RuntimeFault.from_dict(event["error"])
                if event["type"] == "done":
                    complete = True
                    yield event
                    break
                yield event
        finally:
            if not complete:
                # A disconnected stream cannot leave generation running invisibly.
                self.close()
            self.lock.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
        self.pipe.close()
