# Mac engines

Two engines run chat models on Apple Silicon in addition to ExecuTorch and
LiteRT, which run on the CPU there.

## Three layers

Three separate things get called "the model", "the engine" or "the runtime".
This repo uses these names for them:

| Layer | What it is | Examples | Name in this repo |
| --- | --- | --- | --- |
| Model | Trained weights in one format and quantization | Qwen3-0.6B 4-bit MLX, Qwen3-0.6B Q8_0 GGUF | a `variant` inside a package or release |
| Engine | Software that executes those weights on some hardware | ExecuTorch, LiteRT, MLX, llama.cpp, Ollama | `backend` (plus `engine`: `program`, `compiled`, `interpreter`, `llm`) |
| Runtime | Picks a variant this device can run, verifies and loads it, checks updates, and reports what actually ran | LocalKit (phaseone) | `Runtime`, `ReleaseManager` |

## `mlx`: inside the app, on the Metal GPU

- Runs [mlx-lm](https://github.com/ml-explore/mlx-lm) in the runtime's worker
  process. The default target is the Metal GPU; `--accelerator cpu` runs on the
  CPU instead.
- Install it with `uv sync --extra mlx` (Apple Silicon only).
- `localkit import <mlx model dir> <package> --backend mlx` packages an MLX model
  directory, such as an `mlx-community` Hugging Face download.
  - It copies the data files and records their hashes.
  - It skips code files (`.py` and other executable suffixes), because mlx-lm
    builds the architecture from `config.json`.
  - The context length comes from `max_position_embeddings`; `--context`
    overrides it.
  - The variant targets `darwin`/`arm64` and needs the `mlx.metal` component.
    Linux and Android therefore turn it down as "wrong OS" without asking for
    an app update.
- At load time the worker links only the variant's declared files into a
  fresh directory and loads from there. Extra weights left in the package
  directory are never read.
- Results report `device`, `generation_tokens_per_second`,
  `prompt_tokens_per_second` and `peak_memory_gb`.
- Tool calls use the `json` protocol, as on ExecuTorch. Text only.

## `local_server`: a model server already on the machine

- Talks to any server on this machine that speaks the OpenAI-compatible
  `/v1/chat/completions` API, for example Ollama (`11434`), LM Studio
  (`1234`), llama.cpp's `llama-server` (`8080`) or oMLX.
- The package holds a binding file, not weights:

  ```json
  {"base_url": "http://127.0.0.1:11434/v1", "model": "qwen3:0.6b"}
  ```

- `base_url` must point to this machine (`127.0.0.1`, `localhost` or `::1`),
  so a package can't direct the app to a remote host. The binding is hashed
  like any other artifact.
- `localkit import binding.json <package> --backend local_server --context N`
  packages it. `--context` is required, because the runtime can't read the
  server's limit.
- At load, the worker asks `/models` for the list of served models. If the
  server is missing or doesn't serve the model, loading fails with
  `backend_unavailable` or `server_model_unavailable`. With
  `allow_fallback`, the next variant is tried.
- The server loads its own weights, so LocalKit can't verify them. Results report
  `weights_verified: false`, and the server's accelerator is reported as not
  observed. Release checks still run against the served model before a
  release activates.
- Cancelling a request closes the HTTP stream, which stops generation on the
  server.

## Tested on

Apple M5 Max, macOS 25.5, Python 3.12, 2026-10-18:

| Variant (Qwen3 0.6B) | Engine | Speed | Peak memory |
| --- | --- | --- | --- |
| 4-bit MLX | `mlx`, Metal GPU | 426 tok/s | 0.39 GB |
| 8-bit MLX | `mlx`, Metal GPU | 366 tok/s | 0.68 GB |
| Q8_0 GGUF | `local_server` → `llama-server` b8920 | server-side | server-side |

Ollama, LM Studio and oMLX use the same API as `llama-server` but were not run
here.

`tests/test_runtime_mac_engines.py` covers packaging, selection and the
server protocol through a real worker, using a scripted server. It also runs
MLX on the GPU when `.work/models/Qwen3-0.6B-4bit` exists.
