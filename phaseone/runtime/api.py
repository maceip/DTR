"""Authenticated native host contract shared by Python, browsers and remote devices."""

import asyncio
import base64
import hmac
import json
import queue
import re
import threading
import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import Field

from .errors import RuntimeFault
from .host import Runtime
from .package import canonical
from .schema import BackendName, ContentPart, Contract, Generation, Message
from .training import FineTuneRequest


class Selection(Contract):
    backend: BackendName | None = None
    variant: str | None = None
    accelerator: Literal["cpu", "gpu", "npu"] | None = None
    allow_fallback: bool = False

    def selection(self):
        return {key: getattr(self, key) for key in Selection.model_fields}


class Inference(Selection):
    model: str
    inputs: dict[str, Any]
    signature: str = "default"
    timeout_seconds: float = Field(default=60, gt=0, le=3600)
    quantize: bool = True
    dequantize: bool = True


class Chat(Selection):
    model: str
    messages: list[Message] = Field(min_length=1, max_length=4096)
    generation: Generation = Field(default_factory=Generation)
    tools: list[dict[str, Any]] = Field(default_factory=list, max_length=32)
    stream: bool = False


class SessionCreate(Contract):
    model: str
    messages: list[Message] = []


class SessionSend(Selection):
    content: str | list[ContentPart]
    turn_id: str = Field(min_length=1, max_length=128)
    generation: Generation = Field(default_factory=Generation)
    agent: bool = False
    stream: bool = False


class Classification(Selection):
    model: str
    image_base64: str = Field(max_length=45 * 1024**2)
    top_k: int = Field(default=5, ge=1, le=100)


class ToolInvocation(Contract):
    name: str
    arguments: dict[str, Any]
    action_id: str = Field(min_length=1, max_length=256)


class OpenAIChatCompletionRequest(Selection):
    model: str
    messages: list[Message] = Field(min_length=1, max_length=4096)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1, le=10000)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)
    stop: str | list[str] | None = None
    stream: bool = False
    tools: list[dict[str, Any]] = Field(default_factory=list, max_length=32)


class BodyLimit:
    def __init__(self, app, limit=64 * 1024**2):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        total = 0
        async def bounded():
            nonlocal total
            event = await receive()
            total += len(event.get("body", b""))
            if total > self.limit:
                raise RuntimeFault("body_too_large", "Request exceeds 64 MiB")
            return event
        await self.app(scope, bounded, send)


def status_code(error):
    if error.code in ("permission_denied", "untrusted_package"):
        return 403
    if error.code.endswith("not_found") or error.code == "package_not_loaded":
        return 404
    if error.code in ("session_busy", "idempotency_conflict", "turn_incomplete", "action_outcome_unknown", "cancelled"):
        return 409
    if error.code in ("capacity_busy", "queue_timeout"):
        return 429
    if error.code == "inference_timeout":
        return 504
    if error.code == "body_too_large":
        return 413
    if error.code in ("worker_crashed", "backend_load_failed", "inference_failed", "backend_unavailable"):
        return 503
    return 422


def create_app(runtime: Runtime, *, token: str, allowed_origins=()) -> FastAPI:
    if len(token) < 32:
        raise ValueError("Use a random host token of at least 32 characters")
    app = FastAPI(title="PHASEONE Runtime", version="0.3.0")
    app.state.runtime = runtime
    app.state.jobs = {}
    jobs_lock = threading.RLock()
    app.add_middleware(BodyLimit)
    if allowed_origins:
        if "*" in allowed_origins:
            raise ValueError("Choose explicit browser origins")
        app.add_middleware(CORSMiddleware, allow_origins=list(allowed_origins),
                           allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type", "X-Request-ID"])

    @app.middleware("http")
    async def authenticate(request, call_next):
        if request.url.path.startswith("/v1/") and request.method != "OPTIONS":
            authorization = request.headers.get("authorization", "")
            if not hmac.compare_digest(authorization.encode(), ("Bearer " + token).encode()):
                return JSONResponse({"error": {"code": "unauthorized", "message": "Pair with this host's bearer token"}}, status_code=401)
            origin = request.headers.get("origin")
            same_origin = f"{request.url.scheme}://{request.url.netloc}"
            if origin and origin != same_origin and origin not in allowed_origins:
                return JSONResponse({"error": {"code": "origin_denied", "message": "Browser origin is not paired"}}, status_code=403)
        return await call_next(request)

    @app.exception_handler(RuntimeFault)
    async def runtime_error(request, exc):
        return JSONResponse({"error": exc.as_dict()}, status_code=status_code(exc))

    def begin_job(request):
        identity = request.headers.get("x-request-id") or str(uuid.uuid4())
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", identity):
            raise RuntimeFault("invalid_request_id", "X-Request-ID must contain 1 to 128 safe characters")
        cancellation = threading.Event()
        with jobs_lock:
            if identity in app.state.jobs:
                raise RuntimeFault("idempotency_conflict", "A request with this ID is already running")
            app.state.jobs[identity] = cancellation
        return identity, cancellation

    def end_job(identity):
        with jobs_lock:
            app.state.jobs.pop(identity, None)

    async def execute(request, function):
        identity, cancellation = begin_job(request)
        def work():
            try:
                return {**function(cancellation), "request_id": identity}
            finally:
                end_job(identity)
        try:
            return await asyncio.to_thread(work)
        except asyncio.CancelledError:
            cancellation.set()
            raise

    def stream(request, function):
        identity, cancellation = begin_job(request)
        events = queue.Queue(maxsize=32)
        sentinel = object()
        disconnected = threading.Event()
        def put(event):
            # Inference cancellation still needs a terminal protocol event.
            # Only a disconnected consumer should discard buffered responses.
            while not disconnected.is_set():
                try:
                    events.put(event, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False
        def work():
            iterator = None
            try:
                iterator = function(cancellation)
                for event in iterator:
                    if not put({**event, "request_id": identity}):
                        break
            except RuntimeFault as exc:
                put({"type": "error", "error": exc.as_dict(), "request_id": identity})
            except Exception as exc:
                put({"type": "error", "error": {"code": "host_error", "message": str(exc)}, "request_id": identity})
            finally:
                try:
                    if iterator is not None:
                        iterator.close()
                finally:
                    put(sentinel)
                    end_job(identity)
        thread = threading.Thread(target=work, name=f"phaseone-{identity}", daemon=True)
        async def body():
            thread.start()
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.to_thread(events.get, True, 0.2)
                    except queue.Empty:
                        if not thread.is_alive():
                            break
                        continue
                    if event is sentinel:
                        break
                    yield json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
            finally:
                disconnected.set()
                cancellation.set()
                await asyncio.to_thread(thread.join, 5)
        return StreamingResponse(body(), media_type="application/x-ndjson",
                                  headers={"X-Request-ID": identity, "Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/health")
    def health():
        return {"status": "ok", "product": "PHASEONE Runtime", "version": "0.3.0"}

    @app.get("/v1/device")
    def device():
        return {**runtime.device, "granted_permissions": sorted(runtime.granted_permissions)}

    @app.get("/v1/packages")
    def packages():
        return {"packages": [runtime.application(digest).describe() for digest in runtime.packages]}

    @app.get("/v1/packages/{digest}")
    def package(digest: str):
        return runtime.application(digest).describe()

    @app.get("/v1/packages/{digest}/manifest")
    def manifest(digest: str):
        # Send the exact signed/hashable payload bytes. Browser clients hash
        # these bytes directly, avoiding cross-language number reserialization.
        data = json.loads((runtime.application(digest).package.root / "phaseone.json").read_text())
        return Response(canonical(data), media_type="application/json", headers={"ETag": '"' + digest + '"'})

    @app.get("/v1/packages/{digest}/signature")
    def signature(digest: str):
        path = runtime.application(digest).package.root / "phaseone.signature.json"
        if not path.is_file():
            raise RuntimeFault("signature_not_found", "Package is unsigned")
        return FileResponse(path, media_type="application/json")

    @app.get("/v1/packages/{digest}/artifacts/{name}")
    def artifact(digest: str, name: str):
        package = runtime.application(digest).package
        path = package.artifact(name)
        record = package.manifest.artifacts[name]
        return FileResponse(path, media_type=record.media_type, headers={"ETag": '"' + record.sha256 + '"'})

    @app.post("/v1/packages/{digest}/infer")
    async def infer(digest: str, body: Inference, request: Request):
        application = runtime.application(digest)
        return await execute(request, lambda cancellation: application.infer(
            body.model, body.inputs, signature=body.signature, timeout_seconds=body.timeout_seconds,
            quantize=body.quantize, dequantize=body.dequantize, cancellation=cancellation, **body.selection()))

    @app.post("/v1/packages/{digest}/classify")
    async def classify(digest: str, body: Classification, request: Request):
        try:
            data = base64.b64decode(body.image_base64, validate=True)
        except ValueError as exc:
            raise RuntimeFault("invalid_image", "Image is not valid base64") from exc
        application = runtime.application(digest)
        return await execute(request, lambda cancellation: application.classify(
            body.model, data, top_k=body.top_k, cancellation=cancellation, **body.selection()))

    @app.post("/v1/packages/{digest}/chat")
    async def chat(digest: str, body: Chat, request: Request):
        application = runtime.application(digest)
        options = {**body.selection(), "generation": body.generation, "tools": body.tools}
        if body.stream:
            return stream(request, lambda cancellation: application.chat_stream(body.model, body.messages, cancellation=cancellation, **options))
        return await execute(request, lambda cancellation: application.chat(body.model, body.messages, cancellation=cancellation, **options))

    @app.post("/v1/packages/{digest}/sessions")
    def create_session(digest: str, body: SessionCreate):
        return runtime.application(digest).create_session(body.model, messages=body.messages)

    @app.get("/v1/sessions/{identity}")
    def session(identity: str, package: str | None = None):
        value = runtime.state.session(identity)
        if package is not None and package != value["package"]:
            raise RuntimeFault("session_package", "Session belongs to another application version")
        return value

    @app.post("/v1/sessions/{identity}/messages")
    async def send(identity: str, body: SessionSend, request: Request, package: str | None = None):
        current_session = session(identity, package)
        application = runtime.application(current_session["package"])
        content = [part.model_dump(mode="json", exclude_none=True) for part in body.content] if isinstance(body.content, list) else body.content
        options = {**body.selection(), "generation": body.generation, "agent": body.agent, "turn_id": body.turn_id}
        if body.stream:
            return stream(request, lambda cancellation: application.send_stream(identity, content, cancellation=cancellation, **options))
        return await execute(request, lambda cancellation: application.send(identity, content, cancellation=cancellation, **options))

    @app.post("/v1/sessions/{identity}/recover")
    def recover(identity: str, package: str | None = None):
        current_session = session(identity, package)
        return runtime.application(current_session["package"]).recover_session(identity)

    @app.post("/v1/packages/{digest}/tools")
    async def tool(digest: str, body: ToolInvocation, request: Request):
        application = runtime.application(digest)
        return await execute(request, lambda cancellation: application.invoke_tool(
            body.name, body.arguments, action_id=body.action_id, cancellation=cancellation))

    @app.get("/v1/packages/{digest}/capabilities")
    def capabilities(digest: str, model: str):
        return runtime.application(digest).capabilities(model)

    @app.post("/v1/packages/{digest}/fine-tune")
    def fine_tune(digest: str, body: FineTuneRequest):
        return runtime.application(digest).fine_tune(body)

    @app.get("/v1/actions/{identity}")
    def action(identity: str):
        return runtime.state.action(identity)

    @app.post("/v1/jobs/{identity}/cancel")
    def cancel(identity: str):
        with jobs_lock:
            event = app.state.jobs.get(identity)
            if event is None:
                raise RuntimeFault("job_not_found", "Request is not running")
            event.set()
        return {"request_id": identity, "cancellation_requested": True}

    @app.get("/v1/receipts")
    def receipts(after: int = 0, limit: int = 100):
        return {"receipts": runtime.state.receipts(after, limit)}

    @app.get("/v1/receipts/verify")
    def verify_receipts():
        return runtime.state.verify_receipts()

    def _resolve_openai_model(requested_model: str):
        # 1. Check already loaded packages by model name, package name, or digest
        for digest, pkg in list(runtime.packages.items()):
            if requested_model in pkg.manifest.models:
                defaults = pkg.manifest.metadata.get("generation_defaults", {}).get(requested_model, {})
                return runtime.application(digest), requested_model, defaults
            if requested_model in (pkg.manifest.name, digest):
                first_model = next(iter(pkg.manifest.models))
                defaults = pkg.manifest.metadata.get("generation_defaults", {}).get(first_model, {})
                return runtime.application(digest), first_model, defaults
        # 2. Dynamically resolve from local HF cache / packages via resolver
        from .resolver import resolve_model

        resolved = resolve_model(requested_model, state_dir=runtime.root, device=runtime.device)
        application = runtime.load(resolved["package_path"])
        return application, resolved["model"], resolved.get("generation_defaults", {})

    @app.get("/v1/models")
    def list_openai_models():
        data = []
        created = int(time.time())
        for digest, pkg in runtime.packages.items():
            gen_meta = pkg.manifest.metadata.get("generation_defaults", {})
            for model_name, model_def in pkg.manifest.models.items():
                data.append({
                    "id": model_name,
                    "object": "model",
                    "created": created,
                    "owned_by": "localkit",
                    "package_name": pkg.manifest.name,
                    "package_digest": digest,
                    "task": model_def.task,
                    "variants": [v.id for v in model_def.variants],
                    "generation_defaults": gen_meta.get(model_name, {}),
                })
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(body: OpenAIChatCompletionRequest, request: Request):
        application, logical_model, defaults = _resolve_openai_model(body.model)
        stops = [body.stop] if isinstance(body.stop, str) else (body.stop or [])
        max_out = body.max_completion_tokens or body.max_tokens or defaults.get("max_output_tokens") or 1024
        temp = body.temperature if body.temperature is not None else defaults.get("temperature", 0.0)
        top_p = body.top_p if body.top_p is not None else defaults.get("top_p")
        top_k = body.top_k if body.top_k is not None else defaults.get("top_k")
        gen = Generation(
            max_output_tokens=max_out,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            stop=stops,
        )
        options = {**body.selection(), "generation": gen, "tools": body.tools}
        if not body.stream:
            result = await execute(
                request,
                lambda cancellation: application.chat(logical_model, body.messages, cancellation=cancellation, **options),
            )
            usage = result.get("usage") or {}
            in_tok = usage.get("input_tokens") or 0
            out_tok = usage.get("output_tokens") or 0
            return {
                "id": f"chatcmpl-{result['request_id']}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.model,
                "choices": [{
                    "index": 0,
                    "message": result["message"],
                    "finish_reason": result.get("finish_reason", "stop"),
                }],
                "usage": {
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "total_tokens": in_tok + out_tok,
                    **{k: v for k, v in usage.items() if k not in ("input_tokens", "output_tokens")},
                },
                "localkit_execution": result.get("execution"),
                "localkit_receipt": result.get("receipt"),
            }
        return stream(
            request,
            lambda cancellation: application.chat_stream(logical_model, body.messages, cancellation=cancellation, **options),
        )

    return app
