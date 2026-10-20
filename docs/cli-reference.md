# CLI Reference

The `localkit` CLI is installed via `uv run localkit <command>` (or `.venv/bin/localkit <command>`). Every command outputs structured JSON for easy scripting and inspection.

---

## 1. Hardware & Model Discovery

### `localkit device` (alias: `localkit doctor`)
Probes the current host's SoC, physical RAM, live reclaimable RAM (`vm_stat` / `/proc/meminfo`), safe inference memory budget, GPU working-set limits, installed native harnesses, and connected ADB devices.

```sh
uv run localkit device
```

### `localkit models`
Scans `~/.cache/huggingface/hub`, `~/.ollama/models`, `.work/`, and `~/.cache/localkit/packages`. Reports every complete runnable model along with its exact weight footprint, KV-cache bytes per token, recommended context window for this host, and default sampling parameters from `generation_config.json`. Also reports the count of incomplete/interrupted cache directories.

```sh
uv run localkit models
uv run localkit models --show-incomplete
```

---

## 2. Zero-Config Execution & Target Locking

### `localkit chat <model_or_package>`
Resolves a short model name (`qwen3`, `gemma4`, `qwen2.5-coder`) or an existing package path, stages verified weights via zero-copy hardlinks (`os.link`), and runs a chat completion on the best native harness.

```sh
uv run localkit chat qwen3 --prompt "Explain unified memory in one sentence."
uv run localkit chat qwen3 --prompt "Hello" --max-tokens 128 --temperature 0.7
uv run localkit chat gemma4 --prompt "Say hello in five words." --backend litert
```

### `localkit add <model_query>`
Resolves a local model and generates `localkit.toml` plus `localkit.lock.json` containing the physics-based hardware compatibility matrix across `apple_silicon_128gb`, `apple_silicon_16gb`, `android_flagship_12gb`, and `edge_sbc_4gb`.

```sh
uv run localkit add qwen3
```

### `localkit serve <model_or_package>`
Starts the local HTTP runtime server exposing OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints alongside `/health` and native LocalKit endpoints.

```sh
uv run localkit serve qwen3 --host 127.0.0.1 --port 8080
```

### `localkit classify <model_or_package>`
Runs image classification through ExecuTorch or LiteRT and returns top-k predictions, latency, and engine metadata.

```sh
uv run localkit classify .work/vision --model vision --image examples/vision/sample-dog.jpg
```

### `localkit run <model_or_package>`
Runs raw tensor inference against a packaged model variant.

```sh
uv run localkit run .work/vision --model vision --input input.json
```

---

## 3. Evaluation & Benchmarking

### `localkit compare <model_or_package>`
Runs every compatible variant of a model on a JSONL evaluation dataset and ranks them by accuracy, agreement with the reference variant, latency, tokens/sec, and peak memory.

```sh
uv run localkit compare qwen3 --inputs prompts.jsonl
```

### `localkit bench <model_or_package>`
Runs warmup and timed iterations on a model variant and reports p50/p95 latency and throughput.

```sh
uv run localkit bench qwen3 --iterations 5 --warmup 1
```

---

## 4. Packaging, Signing, & Release Lifecycle

| Command | Description |
| --- | --- |
| `localkit import <source> <output> --backend <engine>` | Packages an existing `.pte`, `.tflite`, `.litertlm`, MLX model directory, or local server binding using zero-copy hardlinks where possible. |
| `localkit build <source.py> <output>` | Exports a local PyTorch factory function to ExecuTorch and LiteRT variants. |
| `localkit inspect <package>` | Displays package manifest, variants, and compatibility reasons for the current device. |
| `localkit verify <package>` | Verifies SHA-256 digests of all model artifacts (and Ed25519 signature with `--require-signature`). |
| `localkit keygen <path> [--trust]` | Generates an Ed25519 publisher signing keypair. |
| `localkit sign <package> --key <key>` | Signs a verified package manifest and its artifact hashes. |
| `localkit pack <package> <output.phaseone>` | Creates a verified, distributable `.phaseone` archive. |
| `localkit receipts [--verify]` | Lists recorded execution and tool-invocation audit receipts and verifies hash-chain integrity. |
| `localkit release-check` / `release-update` / `release-rollback` | Checks, qualifies, activates, or rolls back signed model-data releases from a catalog. |
