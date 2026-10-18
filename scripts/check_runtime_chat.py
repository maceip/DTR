#!/usr/bin/env python3
"""Real streaming, persistence and backend migration check for an assistant package."""

import argparse
import json
from pathlib import Path

from phaseone.runtime import Runtime


def check(package, state):
    evidence = []
    with Runtime(state) as runtime:
        app = runtime.load(package)
        backends = list(dict.fromkeys(v.backend for v in app.package.manifest.models["assistant"].variants))
        for backend in backends:
            events = list(app.chat_stream("assistant", [{"role": "user", "content": "What is the capital of France?"}],
                                          backend=backend, generation={"max_output_tokens": 32}))
            result = events[-1]
            assert result["type"] == "done" and "Paris" in result["message"]["content"], result
            assert "<|im_end|>" not in result["message"]["content"], result
            assert any(e["type"] == "delta" for e in events)
            assert result["usage"]["output_tokens"] <= 32
            evidence.append({"kind": "generation", "backend": backend, "result": result})
        session = app.create_session("assistant", messages=[{"role": "system", "content": "Answer briefly."}])
        first = app.send(session["id"], "My name is Alex. Say hello.", turn_id=session["id"] + ":greeting",
                         backend=backends[0], generation={"max_output_tokens": 32})
        second = app.send(session["id"], "What is my name?", turn_id=session["id"] + ":recall",
                          backend=backends[-1], generation={"max_output_tokens": 32})
        assert "Alex" in second["message"]["content"], second
        cached = app.send(session["id"], "What is my name?", turn_id=session["id"] + ":recall",
                          backend=backends[-1], generation={"max_output_tokens": 32})
        assert cached["cached"]
        evidence.append({"kind": "session_migration", "from": backends[0], "to": backends[-1],
                         "first": first, "second": second, "idempotent_retry": cached["cached"]})
        evidence.append({"kind": "audit", **runtime.state.verify_receipts()})
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    check(args.package, args.state)
