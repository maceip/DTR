"""Python client for the same runtime contract on a paired native host."""

import base64
import json
import uuid

import httpx

from .errors import RuntimeFault
from .tensors import encode_tensor


def json_value(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


class RemoteRuntime:
    def __init__(self, url: str, token: str, *, timeout: float = 3605, transport=None):
        self.client = httpx.Client(base_url=url.rstrip("/"),
                                   headers={"Authorization": "Bearer " + token},
                                   timeout=timeout, transport=transport)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.client.close()

    def _response(self, response):
        try:
            document = response.json()
        except ValueError as exc:
            raise RuntimeFault("host_protocol", f"Host returned HTTP {response.status_code} without JSON") from exc
        if "error" in document and (response.is_error or document.get("status") not in ("denied", "failed", "uncertain")):
            raise RuntimeFault.from_dict(document["error"])
        if response.is_error:
            raise RuntimeFault("host_request", f"Host rejected the request: HTTP {response.status_code}", details=document)
        return document

    def request(self, method, path, body=None):
        try:
            return self._response(self.client.request(method, path, json=json_value(body) if body is not None else None))
        except httpx.HTTPError as exc:
            raise RuntimeFault("host_unreachable", str(exc), retryable=True) from exc

    def events(self, path, body):
        identity = str(uuid.uuid4())
        complete = False
        try:
            with self.client.stream("POST", path, json=json_value({**body, "stream": True}),
                                     headers={"X-Request-ID": identity}) as response:
                if response.is_error:
                    response.read()
                    self._response(response)
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError as exc:
                        raise RuntimeFault("host_protocol", "Host emitted invalid stream JSON") from exc
                    if event.get("type") == "error":
                        complete = True
                        raise RuntimeFault.from_dict(event["error"])
                    if event.get("type") == "done":
                        complete = True
                    if "execution" in event:
                        event["execution"] = {**event["execution"], "location": "remote_host"}
                    yield event
                if not complete:
                    raise RuntimeFault("incomplete_generation", "Host stream disconnected before completion", retryable=True)
        except httpx.HTTPError as exc:
            raise RuntimeFault("host_unreachable", str(exc), retryable=True) from exc
        finally:
            if not complete:
                try:
                    self.client.post(f"/v1/jobs/{identity}/cancel", timeout=3)
                except httpx.HTTPError:
                    pass

    def device(self):
        return {**self.request("GET", "/v1/device"), "location": "remote_host"}

    def packages(self):
        return self.request("GET", "/v1/packages")["packages"]

    def application(self, digest):
        # Validate membership before constructing a client-side reference.
        self.request("GET", f"/v1/packages/{digest}")
        return RemoteApplication(self, digest)


class RemoteApplication:
    def __init__(self, runtime, digest):
        self.runtime, self.digest = runtime, digest
        self.path = f"/v1/packages/{digest}"

    def _post(self, path, body):
        result = self.runtime.request("POST", path, body)
        if "execution" in result:
            result["execution"] = {**result["execution"], "location": "remote_host"}
        return result

    def describe(self):
        return self.runtime.request("GET", self.path)

    def capabilities(self, model):
        from urllib.parse import quote

        return self.runtime.request("GET", self.path + "/capabilities?model=" + quote(model, safe=""))

    def fine_tune(self, request):
        from .training import FineTuneRequest

        return self._post(self.path + "/fine-tune", FineTuneRequest.model_validate(request).model_dump())

    def infer(self, model, inputs, **options):
        values = {key: value if isinstance(value, dict) else encode_tensor(value) for key, value in inputs.items()}
        return self._post(self.path + "/infer", {"model": model, "inputs": values, **options})

    def classify(self, model, image, **options):
        return self._post(self.path + "/classify", {"model": model, "image_base64": base64.b64encode(image).decode(), **options})

    def chat(self, model, messages, **options):
        return self._post(self.path + "/chat", {"model": model, "messages": messages, **options})

    def chat_stream(self, model, messages, **options):
        yield from self.runtime.events(self.path + "/chat", {"model": model, "messages": messages, **options})

    def create_session(self, model, *, messages=None):
        return self._post(self.path + "/sessions", {"model": model, "messages": messages or []})

    def session(self, session_id):
        return self.runtime.request("GET", f"/v1/sessions/{session_id}?package={self.digest}")

    def send(self, session_id, content, *, turn_id=None, **options):
        return self._post(f"/v1/sessions/{session_id}/messages?package={self.digest}", {
            "content": content, "turn_id": turn_id or str(uuid.uuid4()), **options,
        })

    def send_stream(self, session_id, content, *, turn_id=None, **options):
        yield from self.runtime.events(f"/v1/sessions/{session_id}/messages?package={self.digest}", {
            "content": content, "turn_id": turn_id or str(uuid.uuid4()), **options,
        })

    def invoke_tool(self, name, arguments, *, action_id):
        return self._post(self.path + "/tools", {"name": name, "arguments": arguments, "action_id": action_id})

    def recover_session(self, session_id):
        return self._post(f"/v1/sessions/{session_id}/recover?package={self.digest}", {})
