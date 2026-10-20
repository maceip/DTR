"""`localkit compare`: every runnable variant on the same inputs, ranked.

Variants are local-server bindings to one scripted OpenAI-compatible server,
each served model answering differently, so ranking runs through real workers.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from phaseone.runtime import Runtime, RuntimeFault
from phaseone.runtime.build import import_server_binding
from phaseone.runtime.compare import compare, normalize, read_inputs

ANSWERS = {
    "exact": {"capital of France?": "Paris", "2 + 2?": "4"},
    "sloppy": {"capital of France?": "<think>hm</think> It is Paris.", "2 + 2?": "5"},
    "verbose": {"capital of France?": "The capital of France is Paris.", "2 + 2?": "2 + 2 = 4"},
}


class Server(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body, content_type="application/json"):
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(json.dumps({"data": [{"id": name} for name in ANSWERS]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        prompt = body["messages"][-1]["content"]
        text = next(answer for question, answer in ANSWERS[body["model"]].items() if question in prompt)
        chunks = [{"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]},
                  {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}]
        self.reply(b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n",
                   "text/event-stream")


@pytest.fixture
def app(tmp_path):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    package = tmp_path / "package"
    variants = [("exact", "exact", {}), ("sloppy", "sloppy", {}), ("verbose", "verbose", {}),
                ("missing", "not-served", {}), ("windows-only", "exact", {"systems": ["windows"]})]
    for variant, model, target in variants:
        binding = tmp_path / f"{variant}.json"
        binding.write_text(json.dumps({"base_url": url, "model": model}))
        import_server_binding(binding, package, name="compare", model="assistant", variant_id=variant,
                              chat={"context_length": 2048}, target=target)
    with Runtime(tmp_path / "state") as runtime:
        yield runtime.load(package)
    httpd.shutdown()


ROWS = [{"prompt": "What is the capital of France?", "expected": "paris"},
        {"prompt": "What is 2 + 2?", "expected": "4"}]


def test_ranks_by_accuracy_on_expected_answers_then_speed(app):
    result = compare(app, "assistant", ROWS, reference="exact")
    board = {row["variant"]: row for row in result["leaderboard"]}
    order = [row["variant"] for row in result["leaderboard"]]
    # exact and verbose are both fully correct; sloppy misses 2 + 2; missing never loads.
    assert set(order[:2]) == {"exact", "verbose"} and order[2:] == ["sloppy", "missing"]
    assert (board["exact"]["accuracy"], board["verbose"]["accuracy"], board["sloppy"]["accuracy"]) == (1.0, 1.0, 0.5)
    assert board["exact"]["agreement"] == board["exact"]["identical"] == 1.0
    assert board["verbose"]["identical"] == 0.0 and 0 < board["verbose"]["agreement"] < 1
    assert board["missing"]["error"]["code"] == "server_model_unavailable"
    assert [s["variant"] for s in result["skipped"]] == ["windows-only"]
    assert result["outputs"][1] == {"input": ROWS[1], "exact": "4", "sloppy": "5", "verbose": "2 + 2 = 4"}


def test_without_expected_answers_ranks_by_agreement_with_the_reference(app):
    result = compare(app, "assistant", [{"prompt": r["prompt"]} for r in ROWS], reference="exact")
    assert result["labelled_inputs"] == 0
    ranked = [row for row in result["leaderboard"] if "error" not in row]
    assert ranked[0]["variant"] == "exact"
    assert ranked[1]["agreement"] >= ranked[2]["agreement"]
    assert all(row["accuracy"] is None for row in ranked)


def test_reference_must_run_on_this_machine(app):
    for reference, code in (("windows-only", "invalid_reference"), ("missing", "reference_failed")):
        with pytest.raises(RuntimeFault) as error:
            compare(app, "assistant", ROWS, reference=reference)
        assert error.value.code == code


def test_reasoning_blocks_are_not_scored():
    assert normalize("<think>\nlong\nchain</think>\n  Paris ") == normalize("PARIS") == "paris"


@pytest.mark.parametrize("line", ['{"text": "no prompt"}', "not json", "[1, 2]"])
def test_inputs_must_be_prompts_messages_or_images(tmp_path, line):
    path = tmp_path / "inputs.jsonl"
    path.write_text('{"prompt": "ok"}\n' + line + "\n")
    with pytest.raises(RuntimeFault) as error:
        read_inputs(path)
    assert error.value.code == "invalid_inputs"
