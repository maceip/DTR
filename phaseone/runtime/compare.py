"""Run every variant of a model on the developer's own inputs and rank them.

Answers "what can this machine run?" and "what does this quantization change on
my data?": each compatible variant runs the same inputs, its outputs are scored
against a reference variant (and against expected answers when the inputs carry
them), and speed and memory come from what each engine reports.
"""

import difflib
import re
import statistics
import time
from pathlib import Path

from .device import compatibility
from .errors import RuntimeFault
from .package import read_json

TASKS = ("chat", "image.classification")


def read_inputs(path: Path) -> list[dict]:
    """JSONL: {"prompt": ...} or {"messages": [...]} for chat, {"image": path} to classify.

    Any row may add "expected": a string or list of strings that a correct chat
    answer contains, or the correct label.
    """
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = read_json(line.encode())
        except ValueError as exc:
            raise RuntimeFault("invalid_inputs", f"Line {number} is not JSON") from exc
        if not isinstance(row, dict) or not {"prompt", "messages", "image"} & row.keys():
            raise RuntimeFault("invalid_inputs", f"Line {number} needs prompt, messages or image")
        rows.append(row)
    if not 1 <= len(rows) <= 1000:
        raise RuntimeFault("invalid_inputs", "Provide 1 to 1000 inputs")
    return rows


REASONING = re.compile(r"<think>.*?</think>", re.DOTALL)


def normalize(text: str) -> str:
    # Engines differ in whether they return a model's reasoning block; score the answer.
    return " ".join(REASONING.sub(" ", text).split()).casefold()


def correct(output: str, expected, task: str) -> bool:
    if task == "image.classification":
        return normalize(output) == normalize(str(expected))
    wanted = [expected] if isinstance(expected, str) else expected
    return all(normalize(item) in normalize(output) for item in wanted)


def weight_bytes(package, variant) -> int:
    return sum(package.manifest.artifacts[a].size for a in variant.artifact_ids())


def run_variant(app, model, task, variant, rows, base, generation):
    runtime = app.runtime
    runtime.unload()  # One model in memory at a time; load time is measured fresh.
    outputs, latencies, speeds, memory, load_ms = [], [], [], [], None
    for row in rows:
        started = time.monotonic()
        try:
            if task == "chat":
                messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
                result = app.chat(model, messages, variant=variant.id, generation=generation)
                outputs.append(result["message"]["content"])
                usage, execution = result["usage"], result["execution"]
                # Same formula for every engine: output tokens over the engine's inference time.
                if usage.get("output_tokens") and execution.get("inference_ms"):
                    speeds.append(usage["output_tokens"] / (execution["inference_ms"] / 1000))
                if usage.get("peak_memory_gb"):
                    memory.append(usage["peak_memory_gb"])
            else:
                result = app.classify(model, (base / row["image"]).read_bytes(), variant=variant.id, top_k=1)
                outputs.append(result["predictions"][0]["label"])
        except (RuntimeFault, OSError) as exc:
            error = exc.as_dict() if isinstance(exc, RuntimeFault) else {"code": "invalid_input", "message": str(exc)}
            return {"error": error, "outputs": outputs}
        elapsed = (time.monotonic() - started) * 1000
        if load_ms is None and runtime.workers:
            load_ms = next(iter(runtime.workers.values())).load_ms
            elapsed -= load_ms  # The first request also loaded the model; that is reported separately.
        latencies.append(elapsed)
    runtime.unload()
    return {"outputs": outputs, "load_ms": load_ms, "median_latency_ms": statistics.median(latencies),
            "tokens_per_second": statistics.mean(speeds) if speeds else None,
            "peak_memory_gb": max(memory) if memory else None}


def compare(app, model: str, rows: list[dict], *, reference: str | None = None, base: Path = Path("."),
            max_tokens: int = 128, timeout_seconds: float = 120) -> dict:
    package = app.package
    if model not in package.manifest.models:
        raise RuntimeFault("model_not_found", f"Unknown logical model: {model}")
    definition = package.manifest.models[model]
    if definition.task not in TASKS:
        raise RuntimeFault("unsupported_task", f"compare supports {', '.join(TASKS)} models")
    runnable, skipped = [], []
    for variant in definition.variants:
        reasons = compatibility(variant, app.runtime.device)
        if reasons:
            skipped.append({"variant": variant.id, "backend": variant.backend, "reasons": reasons})
        else:
            runnable.append(variant)
    if not runnable:
        raise RuntimeFault("no_compatible_variant", "No variant of this model runs on this machine",
                           details={"skipped": skipped})
    # Without a choice, the variant with the most weight bytes is the reference:
    # usually the highest-precision build of the same model.
    reference = reference or max(runnable, key=lambda v: weight_bytes(package, v)).id
    if reference not in {v.id for v in runnable}:
        raise RuntimeFault("invalid_reference", f"Reference {reference} is not a runnable variant",
                           details={"skipped": skipped})
    generation = {"max_output_tokens": max_tokens, "temperature": 0, "timeout_seconds": timeout_seconds}
    runs = {v.id: run_variant(app, model, definition.task, v, rows, base, generation) for v in runnable}
    baseline = runs[reference]
    if "error" in baseline:
        raise RuntimeFault("reference_failed", f"Reference variant {reference} failed", details=baseline["error"])
    labelled = [i for i, row in enumerate(rows) if "expected" in row]
    board = []
    for variant in runnable:
        run = runs[variant.id]
        entry = {"variant": variant.id, "backend": variant.backend, "accelerator": variant.target.accelerator,
                 "weight_bytes": weight_bytes(package, variant), "reference": variant.id == reference}
        if "error" in run:
            board.append({**entry, "error": run["error"]})
            continue
        pairs = list(zip(run["outputs"], baseline["outputs"], strict=True))
        if definition.task == "chat":
            agreement = statistics.mean(difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio() for a, b in pairs)
        else:
            agreement = statistics.mean(float(normalize(a) == normalize(b)) for a, b in pairs)
        accuracy = (statistics.mean(float(correct(run["outputs"][i], rows[i]["expected"], definition.task)) for i in labelled)
                    if labelled else None)
        board.append({**entry, "accuracy": accuracy, "agreement": agreement,
                      "identical": statistics.mean(float(normalize(a) == normalize(b)) for a, b in pairs),
                      **{key: run[key] for key in ("median_latency_ms", "tokens_per_second", "peak_memory_gb", "load_ms")}})
    # With expected answers: most correct, then fastest. Without: closest to the reference, then fastest.
    board.sort(key=lambda r: ("error" in r, -((r.get("accuracy") if labelled else r.get("agreement")) or 0),
                              r.get("median_latency_ms") or float("inf")))
    return {"model": model, "task": definition.task, "reference": reference, "inputs": len(rows),
            "labelled_inputs": len(labelled), "leaderboard": board, "skipped": skipped,
            "outputs": [{"input": rows[i], **{v: runs[v]["outputs"][i] for v in runs if i < len(runs[v]["outputs"])}}
                        for i in range(len(rows))]}


def render(result: dict, *, width: int = 100) -> str:
    def number(value, digits=0, scale=1.0, suffix=""):
        return "–" if value is None else f"{value * scale:.{digits}f}{suffix}"

    lines = [f"{result['model']} ({result['task']}): {result['inputs']} inputs, "
             f"{result['labelled_inputs']} with expected answers; reference = {result['reference']}", "",
             "| rank | variant | backend | accuracy | agreement | identical | latency | tok/s | peak mem | load |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for rank, row in enumerate(result["leaderboard"], 1):
        name = row["variant"] + (" (reference)" if row["reference"] else "")
        if "error" in row:
            lines.append(f"| {rank} | {name} | {row['backend']} | failed: {row['error']['code']} | | | | | | |")
            continue
        target = "server" if row["backend"] == "local_server" else row["accelerator"]
        lines.append(f"| {rank} | {name} | {row['backend']}/{target} | "
                     f"{number(row['accuracy'], 0, 100, '%')} | {number(row['agreement'], 0, 100, '%')} | "
                     f"{number(row['identical'], 0, 100, '%')} | {number(row['median_latency_ms'], 0, 1, ' ms')} | "
                     f"{number(row['tokens_per_second'])} | {number(row['peak_memory_gb'], 2, 1, ' GB')} | "
                     f"{number(row['load_ms'], 0, 1, ' ms')} |")
    for row in result["skipped"]:
        lines.append(f"| – | {row['variant']} | {row['backend']} | skipped: {'; '.join(row['reasons'])} | | | | | | |")
    for item in result["outputs"]:
        source = item["input"]
        prompt = source.get("prompt") or source.get("image") or source["messages"][-1]["content"]
        lines += ["", f"> {str(prompt)[:width]}" + (f"  (expected: {source['expected']})" if "expected" in source else "")]
        for key, value in item.items():
            if key != "input":
                lines.append(f"  {key}: {' '.join(str(value).split())[:width]}")
    return "\n".join(lines)
