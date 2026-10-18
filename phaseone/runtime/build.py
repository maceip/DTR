"""Explicit developer builds and artifact imports; never invoked by model output."""

import importlib.metadata
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .errors import RuntimeFault
from .package import MANIFEST, SIGNATURE, atomic_json, digest_file, load_package, read_json
from .schema import Manifest, Model, Variant


def record(path: Path, root: Path):
    return {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": digest_file(path)}


def run_command(command: list[str], log: Path, *, timeout=3600):
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("wb") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeFault("build_failed", str(exc), details={"log": str(log)}) from exc
    if result.returncode:
        raise RuntimeFault("build_failed", f"Build command exited with status {result.returncode}", details={"log": str(log)})


def litert_environment(root: Path) -> Path:
    """Keep LiteRT-Torch's torch<2.14 exporter separate from ExecuTorch's torch2.14."""
    environment = root / "litert-torch-0.9.4"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    marker = environment / "phaseone-exporter.json"
    requirements = Path(__file__).with_name("export-requirements.txt")
    required = {"litert-torch": "0.9.4", "requirements_sha256": digest_file(requirements)}
    if marker.is_file() and python.is_file() and read_json(marker.read_bytes()) == required:
        return python
    uv = shutil.which("uv")
    if uv:
        if not python.exists():
            run_command([uv, "venv", "--python", sys.executable, str(environment)], root / "venv.log")
        cpu_index = ["--torch-backend", "cpu"] if sys.platform == "linux" else []
        run_command([uv, "pip", "install", "--python", str(python), *cpu_index, "-r", str(requirements)], root / "install-litert.log")
    else:
        if not python.exists():
            run_command([sys.executable, "-m", "venv", str(environment)], root / "venv.log")
        if sys.platform == "linux":
            run_command([str(python), "-m", "pip", "install", "torch==2.13.0", "torchvision==0.28.0",
                         "--index-url", "https://download.pytorch.org/whl/cpu"], root / "install-torch.log")
        run_command([str(python), "-m", "pip", "install", "-r", str(requirements)], root / "install-litert.log")
    atomic_json(marker, required)
    return python


def manifest_for(destination, name, version):
    if (destination / MANIFEST).is_file():
        package = load_package(destination)
        if package.manifest.name != name:
            raise RuntimeFault("package_name", "Output already contains another application; choose a new directory")
        document = read_json((destination / MANIFEST).read_bytes())
        document["version"] = version
        return document
    return {"schema_version": 1, "name": name, "version": version, "artifacts": {}, "models": {}}


def save_variant(destination, document, model_name, model_definition, variant, artifacts):
    variants = document["models"].get(model_name, {}).get("variants", [])
    variants = [v for v in variants if v["id"] != variant["id"]] + [variant]
    document["models"][model_name] = {**model_definition, "variants": variants}
    document["artifacts"].update(artifacts)
    Manifest.model_validate(document)
    atomic_json(destination / MANIFEST, document)
    # An explicit build creates a new package digest. A prior signature cannot
    # authenticate new artifacts; signing is a separate developer command.
    (destination / SIGNATURE).unlink(missing_ok=True)


def build(source: Path, destination: Path, *, name="my-agent", version="1.0.0", model="model",
          factory="build", backends=("executorch", "litert"), environment_root=None,
          executorch_python=None, litert_python=None) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if not source.is_file():
        raise RuntimeFault("source_not_found", f"No model factory at {source}")
    destination.mkdir(parents=True, exist_ok=True)
    document = manifest_for(destination, name, version)
    results = []
    for backend in backends:
        if backend not in ("executorch", "litert"):
            raise RuntimeFault("unsupported_backend", backend)
        if backend == "executorch":
            python = Path(executorch_python) if executorch_python else Path(sys.executable)
        else:
            python = Path(litert_python) if litert_python else litert_environment(Path(environment_root or ".phaseone/build-environments").resolve())
        variant_id = f"{backend}-cpu"
        filename = f"{model}-{backend}." + ("pte" if backend == "executorch" else "tflite")
        artifact = destination / filename
        report = destination / f"{model}-{backend}.build.json"
        log = destination / f"{model}-{backend}.build.log"
        run_command([str(python), str(Path(__file__).with_name("export_worker.py")),
                     "--source", str(source), "--factory", factory, "--backend", backend,
                     "--destination", str(artifact), "--report", str(report)], log)
        exported = read_json(report.read_bytes())
        definition = exported["definition"]
        definition.setdefault("task", "tensor")
        metadata = definition.pop("metadata", {})
        artifact_id = f"{model}.{backend}"
        artifacts = {artifact_id: record(artifact, destination)}
        if "labels" in definition:
            labels = destination / f"{model}-labels.json"
            atomic_json(labels, definition.pop("labels"))
            label_id = f"{model}.labels"
            artifacts[label_id] = record(labels, destination)
            definition["labels"] = label_id
        variant = Variant(id=variant_id, backend=backend, engine="program" if backend == "executorch" else "compiled",
                          artifact=artifact_id, signatures=exported["signatures"], priority=20 if backend == "executorch" else 10,
                          target={"accelerator": "cpu", "delegate": "XnnpackBackend" if backend == "executorch" else None,
                                  "runtime_version": f"=={exported['exporter']}" if backend == "executorch" else ">=2.2,<3"})
        document.setdefault("metadata", {}).update(metadata)
        document["metadata"].setdefault("builds", {})[artifact_id] = {
            "source_file": source.name, "source_sha256": digest_file(source), "factory": factory,
            "exporter": exported["exporter"], "torch": exported["torch"], "sample_validation": exported["sample_validation"],
        }
        save_variant(destination, document, model, definition, variant.model_dump(mode="json", exclude_none=True), artifacts)
        # LiteRT.js consumes the same portable tensor graph, with a separate
        # target declaration and its own API/version constraints. Compatibility
        # is a candidate until the browser actually loads and runs this graph.
        if backend == "litert" and all(
            spec["dtype"] in ("float32", "int32", "uint8") and -1 not in spec["shape"]
            for signature in exported["signatures"].values()
            for spec in signature["inputs"] + signature["outputs"]
        ):
            web_variant = variant.model_copy(deep=True)
            web_variant.id = "litert-web-cpu"
            web_variant.target.systems = ["web"]
            web_variant.target.architectures = ["wasm32"]
            web_variant.target.runtime_version = ">=2.5.3,<3"
            save_variant(destination, document, model, definition,
                         web_variant.model_dump(mode="json", exclude_none=True), artifacts)
        results.append({"backend": backend, "artifact": str(artifact), "sample_validation": exported["sample_validation"]})
    package = load_package(destination)
    return {"package": package.describe(), "path": str(destination), "builds": results}


def import_model(source: Path, destination: Path, *, backend, name="my-agent", version="1.0.0", model="model",
                 variant_id=None, engine=None, target=None, chat=None, contract: Path | None = None,
                 tokenizer: Path | None = None, tokenizer_config: Path | None = None, template: Path | None = None,
                 data: Path | None = None) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if not source.is_file():
        raise RuntimeFault("source_not_found", f"No artifact at {source}")
    destination.mkdir(parents=True, exist_ok=True)
    document = manifest_for(destination, name, version)
    variant_id = variant_id or f"{backend}-cpu"
    engine = engine or ("llm" if chat else "program" if backend == "executorch" else "interpreter")
    artifacts = {}
    def copy(identifier, path):
        key = f"{model}.{variant_id}.{identifier}"
        target_path = destination / f"{model}-{variant_id}-{identifier}{path.suffix}"
        if path.resolve() != target_path.resolve():
            shutil.copyfile(path, target_path)
        artifacts[key] = record(target_path, destination)
        return key
    artifact_id = copy("model", source)
    data_id = copy("data", data) if data else None
    signatures = {}
    if chat:
        chat = dict(chat)
        for key, path in (("tokenizer", tokenizer), ("tokenizer_config", tokenizer_config), ("template", template)):
            if path:
                chat[key] = copy(key, path)
    elif contract:
        signatures = read_json(contract.read_bytes())
    else:
        with tempfile.TemporaryDirectory(prefix=".inspect-", dir=destination) as work:
            report = Path(work) / "metadata.json"
            log = destination / f"{model}-{variant_id}.inspect.log"
            command = [sys.executable, "-m", "phaseone.runtime.inspect_worker", str(source), backend, str(report)]
            if data:
                command.extend(["--data", str(data.resolve())])
            run_command(command, log, timeout=180)
            signatures = read_json(report.read_bytes())["signatures"]
    variant = Variant(id=variant_id, backend=backend, engine=engine, artifact=artifact_id,
                      data_artifact=data_id, target=target or {}, signatures=signatures, chat=chat)
    definition = Model(task="chat" if chat else "tensor", variants=[variant]).model_dump(mode="json", exclude_none=True)
    definition.pop("variants")
    save_variant(destination, document, model, definition, variant.model_dump(mode="json", exclude_none=True), artifacts)
    return load_package(destination).describe()


def save_chat_variant(destination, document, model, variant, artifacts) -> dict:
    definition = Model(task="chat", variants=[variant]).model_dump(mode="json", exclude_none=True)
    definition.pop("variants")
    save_variant(destination, document, model, definition, variant.model_dump(mode="json", exclude_none=True), artifacts)
    return load_package(destination).describe()


def import_mlx(source: Path, destination: Path, *, name="my-agent", version="1.0.0", model="assistant",
               variant_id=None, chat=None, target=None) -> dict:
    """Package an MLX model directory (e.g. an mlx-community Hugging Face download)."""
    from .catalog import NATIVE_SUFFIXES

    source, destination = source.resolve(), destination.resolve()
    if not (source / "config.json").is_file() or not any(source.glob("*.safetensors")):
        raise RuntimeFault("source_not_found", f"{source} is not an MLX model directory (config.json and *.safetensors)")
    destination.mkdir(parents=True, exist_ok=True)
    document = manifest_for(destination, name, version)
    target = {"accelerator": "gpu", "systems": ["darwin"], "architectures": ["arm64"], **(target or {})}
    if target["accelerator"] == "gpu":
        try:
            metal = importlib.metadata.version("mlx-metal")
        except importlib.metadata.PackageNotFoundError:
            metal = "0.32"
        target.setdefault("components", {"mlx.metal": ">=" + ".".join(metal.split(".")[:2])})
    variant_id = variant_id or f"mlx-{target['accelerator']}"
    chat = dict(chat or {})
    if "context_length" not in chat:
        config = read_json((source / "config.json").read_bytes())
        context = config.get("max_position_embeddings") or config.get("text_config", {}).get("max_position_embeddings")
        if not isinstance(context, int):
            raise RuntimeFault("context_required", "config.json has no max_position_embeddings; pass --context")
        chat["context_length"] = context
    folder = destination / f"{model}-{variant_id}"
    folder.mkdir(exist_ok=True)
    artifacts, ids = {}, {}
    for path in sorted(source.iterdir()):
        # Model code is never packaged: mlx-lm builds the architecture from config.json.
        if not path.is_file() or path.name.startswith(".") or path.name.lower().endswith(NATIVE_SUFFIXES):
            continue
        key = f"{model}.{variant_id}." + re.sub(r"[^A-Za-z0-9_.-]", "-", path.name)
        shutil.copyfile(path, folder / path.name)
        artifacts[key] = record(folder / path.name, destination)
        ids[path.name] = key
    variant = Variant(id=variant_id, backend="mlx", engine="llm", artifact=ids["config.json"],
                      supporting_artifacts=sorted(key for file, key in ids.items() if file != "config.json"),
                      target=target, chat=chat)
    return save_chat_variant(destination, document, model, variant, artifacts)


def import_server_binding(source: Path, destination: Path, *, name="my-agent", version="1.0.0", model="assistant",
                          variant_id=None, chat=None, target=None) -> dict:
    """Package a binding to a model served on this machine (Ollama, LM Studio, llama-server, oMLX)."""
    from .backends.server import read_binding

    source, destination = source.resolve(), destination.resolve()
    if not source.is_file():
        raise RuntimeFault("source_not_found", f"No server binding at {source}")
    read_binding(source)
    if not chat or "context_length" not in chat:
        raise RuntimeFault("context_required", "Local server imports require --context with the served context limit")
    destination.mkdir(parents=True, exist_ok=True)
    document = manifest_for(destination, name, version)
    variant_id = variant_id or "local-server"
    path = destination / f"{model}-{variant_id}-binding.json"
    if source != path:
        shutil.copyfile(source, path)
    key = f"{model}.{variant_id}.binding"
    variant = Variant(id=variant_id, backend="local_server", engine="llm", artifact=key,
                      target=target or {}, chat=chat)
    return save_chat_variant(destination, document, model, variant, {key: record(path, destination)})
