#!/usr/bin/env python3
"""Exercise the live authenticated HTTP host and Python SDK, including cancellation."""

import argparse
import json
import time
import uuid
from pathlib import Path

import httpx

from phaseone.runtime import RemoteRuntime, RuntimeFault


def check(url, token_path, output):
    token = token_path.read_text().strip()
    evidence = []
    assert httpx.get(url + "/health").json()["product"] == "PHASEONE Runtime"
    assert httpx.get(url + "/v1/device").status_code == 401
    with RemoteRuntime(url, token) as runtime:
        packages = {package["name"]: package for package in runtime.packages()}
        calculator = runtime.application(packages["runtime-fixture"]["digest"])
        for backend in ("executorch", "litert"):
            result = calculator.infer("calculator", {"left": [[1, 2, 3, 4]], "right": [[4, 3, 2, 1]]}, backend=backend)
            assert result["outputs"]["output"]["data"] == [5, 5, 5, 5]
            assert result["execution"]["location"] == "remote_host"
            evidence.append({"kind": "tensor", "backend": backend, "execution": result["execution"]})
        assistant = runtime.application(packages["local-assistant"]["digest"])
        for backend in ("executorch", "litert"):
            events = list(assistant.chat_stream("assistant", [{"role": "user", "content": "What is the capital of France?"}],
                                                backend=backend, generation={"max_output_tokens": 24}))
            assert events[-1]["type"] == "done" and "Paris" in events[-1]["message"]["content"]
            assert events[0]["execution"]["execution_verified"] is False
            assert events[-1]["execution"]["execution_verified"] is True
            assert sum(event["type"] == "delta" for event in events) > 0
            evidence.append({"kind": "streaming_chat", "backend": backend, "result": events[-1]})
        session = assistant.create_session("assistant", messages=[{"role": "system", "content": "Answer briefly."}])
        assistant.send(session["id"], "My name is Alex. Say hello.", backend="executorch", generation={"max_output_tokens": 24})
        turn_id = str(uuid.uuid4())
        result = assistant.send(session["id"], "What is my name?", turn_id=turn_id, backend="litert", generation={"max_output_tokens": 24})
        assert "Alex" in result["message"]["content"]
        again = assistant.send(session["id"], "What is my name?", turn_id=turn_id, backend="litert", generation={"max_output_tokens": 24})
        assert again["cached"]
        evidence.append({"kind": "backend_migration_and_retry", "result": result, "retry_cached": True})
        identity = str(uuid.uuid4())
        action = assistant.invoke_tool("device_info", {}, action_id=identity)
        assert action["status"] == "complete" and action["result"]["system"]
        assert assistant.invoke_tool("device_info", {}, action_id=identity)["cached"]
        denied = assistant.invoke_tool("write_text", {"path": "forbidden", "text": "x"}, action_id=str(uuid.uuid4()))
        assert denied["status"] == "denied"
        evidence.append({"kind": "tools", "action": action, "undeclared_tool": denied})
        for backend in ("executorch", "litert"):
            events = assistant.chat_stream("assistant", [{"role": "user", "content": "Write a long story about a traveler. Include many detailed paragraphs."}],
                                            backend=backend, generation={"max_output_tokens": 512})
            request_id = None
            cancelled = False
            try:
                for event in events:
                    request_id = event["request_id"]
                    if event["type"] == "delta":
                        runtime.request("POST", f"/v1/jobs/{request_id}/cancel", {})
                        break
                for _ in events:
                    pass
            except RuntimeFault as exc:
                assert exc.code == "cancelled", exc.as_dict()
                cancelled = True
            finally:
                events.close()
            assert cancelled, f"{backend} request completed before cancellation was observed"
            recovered = assistant.chat("assistant", [{"role": "user", "content": "What is the capital of France?"}],
                                        backend=backend, generation={"max_output_tokens": 24})
            assert "Paris" in recovered["message"]["content"]
            evidence.append({"kind": "cancel_and_reload", "backend": backend, "request_id": request_id, "verified": True})
        evidence.append({"kind": "receipts", **runtime.request("GET", "/v1/receipts/verify")})
        evidence.append({"kind": "host", **runtime.device(), "checked_at": time.time()})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": len(evidence), "evidence": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8920")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evidence/runtime-http.json"))
    args = parser.parse_args()
    check(args.url, args.token_file, args.output)
