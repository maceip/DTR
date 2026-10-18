"""Developer workflow for PHASEONE Runtime; the hardware gate remains a separate app."""

import argparse
import hashlib
import json
import os
import secrets
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path

from .errors import RuntimeFault
from .package import create_key, install, load_package, pack, read_json, sign_package
from .schema import Manifest


def dump(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def document(value):
    if value == "-":
        return read_json(sys.stdin.buffer.read())
    if value.lstrip().startswith(("{", "[")):
        return read_json(value.encode())
    return read_json(Path(value).read_bytes())


def token_file(path: Path, *, create=False):
    if not path.exists() and create:
        path.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as target:
            target.write(secrets.token_urlsafe(32) + "\n")
    return path.read_text().strip()


def parser():
    root = argparse.ArgumentParser(description="PHASEONE Runtime — one application across ExecuTorch and LiteRT")
    root.add_argument("--version", action="version", version="PHASEONE Runtime 0.3.0")
    commands = root.add_subparsers(dest="command", required=True)
    parsers = {}
    def command(name, help):
        item = commands.add_parser(name, help=help)
        item.add_argument("--state-dir", type=Path, default=Path(".phaseone"))
        item.add_argument("--trust-store", type=Path)
        item.add_argument("--require-signature", action="store_true")
        parsers[name] = item
        return item
    command("doctor", "Report host capabilities and installed engines")
    command("schema", "Print the versioned application package schema")
    command("catalog-schema", "Print the shared model-release catalog schema")
    for name in ("release-check", "release-update", "release-state", "release-rollback", "release-requalify"):
        item = command(name, "Manage qualified model-data releases")
        item.add_argument("capability")
        item.add_argument("--catalog-key", type=Path, action="append", default=[])
        item.add_argument("--local-development", action="store_true", help="Explicitly allow unsigned and loopback/file catalogs")
        if name in ("release-check", "release-update"):
            item.add_argument("catalog")
            item.add_argument("--contract", required=True)
            item.add_argument("--lineage", action="append", required=True)
            item.add_argument("--backend", choices=["executorch", "litert"])
            item.add_argument("--accelerator", choices=["cpu", "gpu", "npu"], action="append")
            item.add_argument("--allow-fallback", action="store_true", help="Try another allowed candidate if qualification fails")
    for name in ("inspect", "verify", "install"):
        command(name, f"{name.capitalize()} an application package").add_argument("package", type=Path)
    key = command("keygen", "Create an Ed25519 signing key")
    key.add_argument("path", type=Path)
    key.add_argument("--trust", action="store_true", help="Trust this new public key on this host")
    trust = command("trust", "Trust a public key stored as 64 hexadecimal characters")
    trust.add_argument("public_key", type=Path)
    sign = command("sign", "Sign a verified application manifest and its artifact hashes")
    sign.add_argument("package", type=Path)
    sign.add_argument("--key", type=Path, required=True)
    package = command("pack", "Create a verified distributable package archive")
    package.add_argument("package", type=Path)
    package.add_argument("output", type=Path)
    build = command("build", "Export a trusted local PyTorch factory to both engines")
    build.add_argument("source", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--factory", default="build")
    build.add_argument("--backend", choices=["executorch", "litert"], action="append")
    build.add_argument("--executorch-python", type=Path)
    build.add_argument("--litert-python", type=Path)
    importer = command("import", "Package an existing .pte, .tflite or .litertlm artifact")
    importer.add_argument("source", type=Path)
    importer.add_argument("output", type=Path)
    importer.add_argument("--backend", choices=["executorch", "litert"], required=True)
    importer.add_argument("--variant")
    importer.add_argument("--engine", choices=["program", "compiled", "interpreter", "llm"])
    importer.add_argument("--accelerator", choices=["cpu", "gpu", "npu"], default="cpu")
    importer.add_argument("--delegate")
    importer.add_argument("--soc")
    importer.add_argument("--runtime-version")
    importer.add_argument("--system", action="append", default=[])
    importer.add_argument("--architecture", action="append", default=[])
    importer.add_argument("--context", type=int)
    importer.add_argument("--prompt-format", choices=["chatml", "llama3"], help="Installed Android ExecuTorch prompt profile")
    importer.add_argument("--tool-protocol", choices=["json", "native", "none"], default="json")
    for option in ("contract", "tokenizer", "tokenizer-config", "template", "data"):
        importer.add_argument("--" + option, type=Path)
    for item in (build, importer):
        item.add_argument("--name", default="my-agent")
        item.add_argument("--package-version", default="1.0.0")
        item.add_argument("--model", default="model")
    def selection(item):
        item.add_argument("--backend", choices=["executorch", "litert"])
        item.add_argument("--variant")
        item.add_argument("--accelerator", choices=["cpu", "gpu", "npu"])
        item.add_argument("--allow-fallback", action="store_true")
    for name in ("run", "classify", "chat", "bench", "tool"):
        item = command(name, {"run": "Run a tensor model", "classify": "Classify an image", "chat": "Chat or run the packaged agent",
                               "bench": "Benchmark real model execution", "tool": "Invoke a declared tool through the host gateway"}[name])
        item.add_argument("package", help="Local package path, or package digest with --remote")
        item.add_argument("--remote", help="URL of an explicitly paired native host")
        item.add_argument("--token-file", type=Path)
        item.add_argument("--grant", action="append", default=[])
        item.add_argument("--workspace", type=Path)
        if name != "tool":
            item.add_argument("--model", default="assistant" if name == "chat" else "model")
            selection(item)
        if name in ("run", "bench"):
            item.add_argument("--inputs", required=True, help="JSON file, literal JSON, or - for stdin")
            item.add_argument("--signature", default="default")
            item.add_argument("--timeout", type=float, default=60)
        if name == "classify":
            item.add_argument("--image", type=Path, required=True)
            item.add_argument("--top-k", type=int, default=5)
        if name == "chat":
            prompt = item.add_mutually_exclusive_group(required=True)
            prompt.add_argument("--prompt")
            prompt.add_argument("--messages", help="JSON conversation file or literal JSON")
            item.add_argument("--session")
            item.add_argument("--turn-id")
            item.add_argument("--agent", action="store_true")
            item.add_argument("--stream", action="store_true")
            item.add_argument("--json", action="store_true", help="Emit stream events as NDJSON")
            item.add_argument("--max-tokens", type=int, default=128)
            item.add_argument("--temperature", type=float, default=0)
            item.add_argument("--timeout", type=float, default=120)
            item.add_argument("--stop", action="append", default=[])
        if name == "bench":
            item.add_argument("--repeat", type=int, default=10)
            item.add_argument("--warmup", type=int, default=1)
        if name == "tool":
            item.add_argument("tool_name")
            item.add_argument("--arguments", default="{}")
            item.add_argument("--action-id", required=True)
    serve = command("serve", "Serve packages through the shared API and browser console")
    serve.add_argument("--package", action="append", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8920)
    serve.add_argument("--token-file", type=Path)
    serve.add_argument("--grant", action="append", default=[])
    serve.add_argument("--workspace", type=Path)
    serve.add_argument("--cors-origin", action="append", default=[])
    serve.add_argument("--max-loaded-models", type=int, default=2)
    token = command("token", "Display this host's pairing token explicitly")
    token.add_argument("--token-file", type=Path)
    receipts = command("receipts", "Read or verify execution records")
    receipts.add_argument("--verify", action="store_true")
    receipts.add_argument("--after", type=int, default=0)
    return root


def run(args):
    trust = args.trust_store or args.state_dir / "trusted-keys"
    if args.command == "schema":
        return dump(Manifest.model_json_schema())
    if args.command == "catalog-schema":
        from .catalog import Catalog

        return dump(Catalog.model_json_schema())
    if args.command.startswith("release-"):
        from .catalog import ReleasePolicy
        from .host import Runtime

        keys = {}
        for path in args.catalog_key:
            public = bytes.fromhex(path.read_text().strip())
            if len(public) != 32:
                raise RuntimeFault("invalid_public_key", "A catalog key must contain 32 bytes")
            keys[hashlib.sha256(public).hexdigest()] = public
        with Runtime(args.state_dir, catalog_keys=keys, local_development=args.local_development,
                     trust_store=trust, require_signature=args.require_signature) as runtime:
            manager = runtime.releases
            if args.command == "release-state":
                return dump(manager.state(args.capability))
            if args.command == "release-rollback":
                return dump(manager.rollback(args.capability))
            if args.command == "release-requalify":
                return dump(manager.requalify(args.capability))
            policy = ReleasePolicy(contract=args.contract, allowed_lineages=args.lineage,
                                   backend=args.backend, accelerators=args.accelerator or ["cpu"],
                                   allow_fallback=args.allow_fallback)
            operation = manager.check if args.command == "release-check" else manager.update
            return dump(operation(args.catalog, args.capability, policy))
    if args.command == "doctor":
        from .device import discover

        return dump(discover())
    if args.command in ("inspect", "verify"):
        with tempfile.TemporaryDirectory(prefix="phaseone-inspect-") as temporary:
            if args.package.is_dir() or args.package.name == "phaseone.json":
                package = load_package(args.package, trust_store=trust, require_signature=args.require_signature)
            else:
                package = install(args.package, Path(temporary), trust_store=trust,
                                  require_signature=args.require_signature)
            if args.command == "inspect":
                from .device import compatibility, discover

                device = discover()
                return dump({**package.describe(), "device": device, "compatibility": {
                    name: [{"variant": v.id, "reasons": compatibility(v, device)} for v in m.variants]
                    for name, m in package.manifest.models.items()}})
            return dump({"verified": True, **package.describe()})
    if args.command == "keygen":
        identity = create_key(args.path, trust_store=trust if args.trust else None)
        return dump({"key_id": identity, "private_key_path": str(args.path), "trusted_on_this_host": args.trust})
    if args.command == "trust":
        public = bytes.fromhex(args.public_key.read_text().strip())
        if len(public) != 32:
            raise RuntimeFault("invalid_public_key", "Ed25519 public key must contain 32 bytes")
        identity = hashlib.sha256(public).hexdigest()
        trust.mkdir(parents=True, exist_ok=True)
        (trust / f"{identity}.pub").write_text(public.hex() + "\n")
        return dump({"trusted_key": identity})
    if args.command == "sign":
        return dump({"signer": sign_package(args.package, args.key)})
    if args.command == "pack":
        return dump(pack(args.package, args.output))
    if args.command == "build":
        from .build import build

        return dump(build(args.source, args.output, name=args.name, version=args.package_version, model=args.model,
                          factory=args.factory, backends=args.backend or ("executorch", "litert"),
                          environment_root=args.state_dir / "build-environments", executorch_python=args.executorch_python,
                          litert_python=args.litert_python))
    if args.command == "import":
        from .build import import_model

        chat = {"context_length": args.context, "tool_protocol": args.tool_protocol} if args.context else None
        if args.prompt_format:
            if chat is None:
                raise RuntimeFault("context_required", "A prompt profile requires --context")
            chat["prompt_format"] = args.prompt_format
        if args.engine == "llm" and chat is None:
            raise RuntimeFault("context_required", "LLM imports require --context with the compiled context limit")
        return dump(import_model(args.source, args.output, backend=args.backend, name=args.name, version=args.package_version,
                                 model=args.model, variant_id=args.variant, engine=args.engine, chat=chat,
                                 target={"accelerator": args.accelerator, "delegate": args.delegate, "soc": args.soc,
                                         "runtime_version": args.runtime_version, "systems": args.system, "architectures": args.architecture},
                                 contract=args.contract, tokenizer=args.tokenizer, tokenizer_config=args.tokenizer_config,
                                 template=args.template, data=args.data))
    if args.command == "token":
        print(token_file(args.token_file or args.state_dir / "host.token"))
        return
    from .host import Runtime

    if getattr(args, "remote", None):
        from .client import RemoteRuntime

        token = token_file(args.token_file) if args.token_file else os.environ.get("PHASEONE_HOST_TOKEN")
        if not token:
            raise RuntimeFault("pairing_required", "Set PHASEONE_HOST_TOKEN or provide --token-file")
        runtime = RemoteRuntime(args.remote, token)
    else:
        runtime = Runtime(args.state_dir, trust_store=trust, require_signature=args.require_signature,
                          granted_permissions=set(getattr(args, "grant", [])), workspace=getattr(args, "workspace", None),
                          max_loaded_models=getattr(args, "max_loaded_models", 2))
    with runtime:
        if args.command == "serve":
            import uvicorn

            from .api import create_app

            for path in args.package:
                runtime.load(path)
            path = args.token_file or args.state_dir / "host.token"
            token = token_file(path, create=True)
            print(f"PHASEONE Runtime at http://{args.host}:{args.port}; pairing token file: {path}", file=sys.stderr)
            app = create_app(runtime, token=token, allowed_origins=args.cors_origin)
            uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
            return
        if args.command == "receipts":
            return dump(runtime.state.verify_receipts() if args.verify else runtime.state.receipts(args.after))
        app = runtime.application(args.package) if getattr(args, "remote", None) else runtime.load(args.package)
        if args.command == "install":
            return dump(app.describe())
        if args.command == "tool":
            return dump(app.invoke_tool(args.tool_name, document(args.arguments), action_id=args.action_id))
        options = {key: getattr(args, key) for key in ("backend", "variant", "accelerator", "allow_fallback")}
        if args.command in ("run", "bench"):
            inputs = document(args.inputs)
            options.update(signature=args.signature, timeout_seconds=args.timeout)
            if args.command == "run":
                return dump(app.infer(args.model, inputs, **options))
            if not 1 <= args.repeat <= 1000 or not 0 <= args.warmup <= 100:
                raise RuntimeFault("invalid_benchmark", "Use 1–1000 repeats and 0–100 warmups")
            for _ in range(args.warmup):
                app.infer(args.model, inputs, **options)
            values, result = [], None
            for _ in range(args.repeat):
                started = time.monotonic()
                result = app.infer(args.model, inputs, **options)
                values.append((time.monotonic() - started) * 1000)
            return dump({"iterations": len(values), "round_trip_ms": {"min": min(values), "median": statistics.median(values), "max": max(values)},
                         "last_execution": result["execution"]})
        if args.command == "classify":
            return dump(app.classify(args.model, args.image.read_bytes(), top_k=args.top_k, **options))
        options["generation"] = {"max_output_tokens": args.max_tokens, "temperature": args.temperature,
                                  "timeout_seconds": args.timeout, "stop": args.stop}
        if args.messages:
            if args.session or args.agent:
                raise RuntimeFault("invalid_chat_options", "Use --prompt for persisted sessions and packaged agents")
            messages = document(args.messages)
            result = app.chat_stream(args.model, messages, **options) if args.stream else app.chat(args.model, messages, **options)
        elif args.session or args.agent:
            session_id = args.session or app.create_session(args.model)["id"]
            options.update(turn_id=args.turn_id or str(uuid.uuid4()), agent=args.agent)
            result = app.send_stream(session_id, args.prompt, **options) if args.stream else app.send(session_id, args.prompt, **options)
        else:
            messages = [{"role": "user", "content": args.prompt}]
            result = app.chat_stream(args.model, messages, **options) if args.stream else app.chat(args.model, messages, **options)
        if args.stream:
            for event in result:
                if args.json:
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                elif event["type"] == "delta":
                    print(event["text"], end="", flush=True)
                elif event["type"] == "tool_result":
                    print(f"\n[{event['action']['tool']}: {event['action']['status']}]", file=sys.stderr)
            if not args.json:
                print()
        else:
            dump(result)


def main(argv=None):
    try:
        run(parser().parse_args(argv))
    except RuntimeFault as exc:
        print(json.dumps({"error": exc.as_dict()}), file=sys.stderr)
        raise SystemExit(2) from exc
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": {"code": "invalid_input", "message": str(exc)}}), file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
