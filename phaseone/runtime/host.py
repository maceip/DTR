"""Public native runtime: one application interface across ExecuTorch and LiteRT."""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .device import candidates, compatibility, discover
from .errors import RuntimeFault
from .package import Package, canonical, install
from .permissions import PermissionPolicy
from .schema import Generation, Message
from .state import State
from .tensors import encode_tensor, prepare_input, prepare_output, preprocess_image
from .worker import Worker


class Runtime:
    def __init__(self, state_dir: str | Path = ".phaseone", *, max_loaded_models: int = 2,
                 trust_store: str | Path | None = None, require_signature: bool = False,
                 granted_permissions: set[str] | None = None, workspace: str | Path | None = None,
                 catalog_keys: dict[str, bytes] | None = None, local_development: bool = False,
                 permissions: PermissionPolicy | dict | None = None, subject: str = "local"):
        self.root = Path(state_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = discover()
        self.state = State(self.root / "runtime.sqlite3")
        self.max_loaded_models = max_loaded_models
        if max_loaded_models < 1:
            raise ValueError("max_loaded_models must be positive")
        self.trust_store = Path(trust_store) if trust_store else self.root / "trusted-keys"
        self.require_signature = require_signature
        self.granted_permissions = set(granted_permissions or [])
        self.permissions = (permissions if isinstance(permissions, PermissionPolicy)
                            else PermissionPolicy(permissions) if permissions is not None else None)
        self.subject = subject  # App/authentication supplied, never a tool argument.
        self.workers: OrderedDict[tuple, Worker] = OrderedDict()
        self.packages: dict[str, Package] = {}
        self.active_sessions: set[str] = set()
        self.observations: dict[tuple, dict] = {}
        self.lock = threading.RLock()
        self.closed = False
        from .tools import ToolGateway, register_builtin_tools

        self.tool_gateway = ToolGateway(self)
        register_builtin_tools(self, workspace=Path(workspace) if workspace is not None else None)
        from .releases import ReleaseManager

        self.releases = ReleaseManager(self, trusted_keys=catalog_keys,
                                       local_development=local_development)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def load(self, source: str | Path) -> "Application":
        package = install(Path(source), self.root / "packages", trust_store=self.trust_store,
                          require_signature=self.require_signature)
        self.packages[package.digest] = package
        self.state.receipt("package_loaded", package.describe())
        return Application(self, package)

    def open(self, capability: str) -> "Application":
        """Open a locally qualified active release, without making a network request."""
        return self.releases.open(capability)

    def application(self, digest: str) -> "Application":
        if digest not in self.packages:
            raise RuntimeFault("package_not_loaded", "Application is not loaded in this host")
        return Application(self, self.packages[digest])

    @contextmanager
    def _registry_lock(self, deadline=None, cancellation=None):
        while True:
            if cancellation and cancellation.is_set():
                raise RuntimeFault("cancelled", "Request was cancelled while selecting a model")
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeFault("inference_timeout", "Request exceeded its time budget while loading a model", retryable=True)
            if self.lock.acquire(timeout=0.05):
                break
        try:
            yield
        finally:
            self.lock.release()

    def _worker(self, package, variant, *, deadline=None, cancellation=None):
        identity = hashlib.sha256(canonical(variant.model_dump(mode="json"))).hexdigest()
        key = (package.digest, identity)
        with self._registry_lock(deadline, cancellation):
            if self.closed:
                raise RuntimeFault("runtime_closed", "Runtime has been closed")
            existing = self.workers.get(key)
            if existing and not existing.closed and existing.process.is_alive():
                self.workers.move_to_end(key)
                existing.reservations += 1
                return existing
            if existing:
                existing.close()
                del self.workers[key]
            while len(self.workers) >= self.max_loaded_models:
                evicted = False
                for old_key, old in list(self.workers.items()):
                    if not old.reservations and old.lock.acquire(blocking=False):
                        try:
                            old.close()
                            del self.workers[old_key]
                        finally:
                            old.lock.release()
                        evicted = True
                        break
                if not evicted:
                    raise RuntimeFault("capacity_busy", "All loaded model slots are busy", retryable=True)
            load_timeout = min(180, deadline - time.monotonic()) if deadline is not None else 180
            if load_timeout <= 0:
                raise RuntimeFault("inference_timeout", "Model selection exhausted the request time budget", retryable=True)
            worker = Worker(package, variant, self.root / "cache" / package.digest / identity,
                            load_timeout=load_timeout, cancellation=cancellation)
            worker.reservations = 1
            self.workers[key] = worker
            self.state.receipt("model_loaded", {
                "package": package.digest, "variant": variant.id,
                "execution": worker.info, "load_ms": worker.load_ms,
            })
            return worker

    def unload(self):
        with self.lock:
            for worker in self.workers.values():
                worker.close()
            self.workers.clear()

    def close(self):
        if not self.closed:
            self.unload()
            self.closed = True
            self.state.close()


class Application:
    def __init__(self, runtime: Runtime, package: Package):
        self.runtime, self.package = runtime, package

    def describe(self) -> dict:
        result = self.package.describe()
        result["device"] = self.runtime.device
        result["compatibility"] = {
            name: [{"variant": variant.id, "backend": variant.backend,
                    "reasons": compatibility(variant, self.runtime.device),
                    "execution_verified": (self.package.digest, name, variant.id) in self.runtime.observations,
                    "last_execution": self.runtime.observations.get((self.package.digest, name, variant.id))}
                   for variant in model.variants]
            for name, model in self.package.manifest.models.items()
        }
        return result

    def _model(self, name):
        try:
            return self.package.manifest.models[name]
        except KeyError as exc:
            raise RuntimeFault("model_not_found", f"Unknown logical model: {name}") from exc

    def capabilities(self, model: str) -> dict:
        definition = self._model(model)
        return {"model": model, "function_calling": {
            "variants": [{"variant": v.id, "protocol": v.chat.tool_protocol if v.chat else "none",
                          "declared": bool(v.chat and v.chat.tool_protocol != "none")}
                         for v in definition.variants],
            "execution": "application_tool_gateway", "quality_verified": False,
        }, "fine_tuning": {"available": False, "status": "not_implemented",
                           "objectives": ["function_calling", "chat"]}}

    def fine_tune(self, request):
        from .training import fine_tune

        return fine_tune(self, request)

    @contextmanager
    def _select(self, name, *, backend=None, variant=None, accelerator=None, allow_fallback=False,
                deadline=None, cancellation=None, modalities=None, image_count=0, require_tools=False):
        model = self._model(name)
        choices, rejected = candidates(model, self.runtime.device, backend=backend,
                                       variant_id=variant, accelerator=accelerator,
                                       modalities=modalities, image_count=image_count, require_tools=require_tools)
        attempts = []
        for choice in choices:
            try:
                worker = self.runtime._worker(self.package, choice, deadline=deadline, cancellation=cancellation)
            except RuntimeFault as exc:
                attempts.append({"variant": choice.id, "error": exc.as_dict()})
                if not allow_fallback or exc.code in ("cancelled", "inference_timeout"):
                    raise
                continue
            try:
                yield choice, worker, attempts
            finally:
                with self.runtime.lock:
                    worker.reservations -= 1
            return
        raise RuntimeFault("backend_unavailable", "Every compatible variant failed to load",
                           details={"attempts": attempts, "rejected": rejected})

    def infer(self, model: str, inputs: dict, *, signature: str = "default", backend=None,
              variant=None, accelerator=None, allow_fallback=False, timeout_seconds=60,
              quantize=True, dequantize=True, cancellation=None) -> dict:
        if self._model(model).task == "chat":
            raise RuntimeFault("wrong_operation", "Use chat() for a conversation model")
        if not 0 < timeout_seconds <= 3600:
            raise RuntimeFault("invalid_timeout", "Timeout must be between 0 and 3600 seconds")
        started = time.monotonic()
        deadline = started + timeout_seconds
        with self._select(model, backend=backend, variant=variant,
                          accelerator=accelerator, allow_fallback=allow_fallback,
                          deadline=deadline, cancellation=cancellation) as (chosen, worker, attempts):
            if signature not in chosen.signatures:
                raise RuntimeFault("signature_not_found", f"Unknown logical signature {signature}")
            contract = chosen.signatures[signature]
            expected = {spec.name for spec in contract.inputs}
            if set(inputs) != expected:
                raise RuntimeFault("input_names", "Input names do not match the model signature",
                                   details={"expected": sorted(expected), "actual": sorted(inputs)})
            prepared = {spec.name: prepare_input(inputs[spec.name], spec, quantize=quantize)
                        for spec in contract.inputs}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeFault("inference_timeout", "Request exhausted its time budget before inference", retryable=True)
            events = worker.events({"operation": "infer", "signature": signature, "inputs": prepared},
                                   timeout=remaining, cancellation=cancellation)
            result = next(events)
            # Exhaust the generator so it releases its worker lock without cancellation.
            for _ in events:
                pass
            outputs = {spec.name: encode_tensor(prepare_output(result["outputs"][spec.name], spec,
                                                               dequantize=dequantize))
                       for spec in contract.outputs}
            execution = {**worker.info, "variant": chosen.id, "package": self.package.digest,
                         "model": model, "signature": signature, "fallback_attempts": attempts,
                         "inference_ms": result["inference_ms"],
                         "elapsed_ms": (time.monotonic() - started) * 1000,
                         "location": "native_host", "execution_verified": True}
            receipt = self.runtime.state.receipt("inference", execution)
            self.runtime.observations[(self.package.digest, model, chosen.id)] = execution
            return {"outputs": outputs, "execution": execution, "receipt": receipt}

    def classify(self, model: str, image: bytes, *, top_k: int = 5, **options) -> dict:
        definition = self._model(model)
        if definition.task != "image.classification":
            raise RuntimeFault("wrong_operation", "Model is not an image classifier")
        if not 1 <= top_k <= 100:
            raise RuntimeFault("invalid_top_k", "top_k must be between 1 and 100")
        array = preprocess_image(image, definition.image)
        result = self.infer(model, {definition.image.input_name: array}, **options)
        from .tensors import decode_tensor

        name = definition.output_name or next(iter(result["outputs"]))
        logits = decode_tensor(result["outputs"][name]).reshape(-1).astype(np.float64)
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        labels = json.loads(self.package.artifact(definition.labels).read_text()) if definition.labels else []
        result["predictions"] = [{"index": int(i), "label": labels[i] if i < len(labels) else str(i),
                                  "score": float(scores[i])}
                                 for i in np.argsort(scores)[::-1][:top_k]]
        return result

    def chat_stream(self, model: str, messages: list, *, generation=None, tools=None,
                    backend=None, variant=None, accelerator=None, allow_fallback=False,
                    cancellation=None):
        if self._model(model).task != "chat":
            raise RuntimeFault("wrong_operation", "Model is not a conversation model")
        messages = [m if isinstance(m, Message) else Message.model_validate(m) for m in messages]
        if not messages:
            raise RuntimeFault("invalid_messages", "Conversation requires at least one message")
        generation = generation if isinstance(generation, Generation) else Generation.model_validate(generation or {})
        deadline = time.monotonic() + generation.timeout_seconds
        from .media import validate_messages_media

        serialized = [m.model_dump(mode="json", exclude_none=True) for m in messages]
        required_modalities, image_count = validate_messages_media(serialized)
        with self._select(model, backend=backend, variant=variant,
                          accelerator=accelerator, allow_fallback=allow_fallback,
                          deadline=deadline, cancellation=cancellation,
                          modalities=required_modalities, image_count=image_count,
                          require_tools=bool(tools)) as (chosen, worker, attempts):
            execution = {**worker.info, "package": self.package.digest, "variant": chosen.id,
                         "model": model, "location": "native_host", "fallback_attempts": attempts,
                         "execution_verified": False}
            yield {"type": "start", "execution": execution}
            request = {"operation": "chat", "messages": serialized,
                       "generation": generation.model_dump(mode="json"), "tools": tools or []}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeFault("inference_timeout", "Request exhausted its time budget before generation", retryable=True)
            for event in worker.events(request, timeout=remaining, cancellation=cancellation):
                if event["type"] == "done":
                    execution = {**execution, "inference_ms": event.pop("inference_ms"),
                                 "usage": event.get("usage", {}), "execution_verified": True}
                    event["execution"] = execution
                    event["receipt"] = self.runtime.state.receipt("chat", execution)
                    self.runtime.observations[(self.package.digest, model, chosen.id)] = execution
                yield event

    def chat(self, model: str, messages: list, **options) -> dict:
        result = None
        for event in self.chat_stream(model, messages, **options):
            if event["type"] == "done":
                result = event
        if result is None:
            raise RuntimeFault("incomplete_generation", "Generation ended without a completion")
        return result

    def create_session(self, model: str, *, messages=None) -> dict:
        if self._model(model).task != "chat":
            raise RuntimeFault("wrong_operation", "Sessions require a chat model")
        normalized = [Message.model_validate(m).model_dump(mode="json", exclude_none=True)
                      for m in (messages or [])]
        return self.runtime.state.create_session(self.package.digest, model, normalized)

    def invoke_tool(self, name: str, arguments: dict, *, action_id: str, cancellation=None):
        return self.runtime.tool_gateway.invoke(self.package, name, arguments,
                                                 action_id=action_id, cancellation=cancellation)

    def session(self, session_id: str):
        session = self.runtime.state.session(session_id)
        if session["package"] != self.package.digest:
            raise RuntimeFault("session_package", "Session belongs to another application version")
        return session

    def recover_session(self, session_id: str):
        with self.runtime.lock:
            self.session(session_id)
            if session_id in self.runtime.active_sessions:
                raise RuntimeFault("session_busy", "Cancel the active turn before recovering this session")
            return self.runtime.state.recover_session(session_id)

    def send_stream(self, session_id: str, content, *, turn_id=None, agent=False, **options):
        from .agent import session_stream

        yield from session_stream(self, session_id, content, turn_id=turn_id, agent=agent, **options)

    def send(self, session_id: str, content, *, turn_id: str | None = None, agent=False, **options) -> dict:
        result = None
        for event in self.send_stream(session_id, content, turn_id=turn_id, agent=agent, **options):
            if event["type"] == "done":
                result = event
        if result is None:
            raise RuntimeFault("incomplete_generation", "Turn ended without a completion")
        return result
