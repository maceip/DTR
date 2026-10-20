# Get Started in 60 Seconds

You do not need to write a manifest, calculate context lengths, or copy model weights to run local inference with LocalKit. If you already have models in `~/.cache/huggingface/hub`, `~/.ollama/models`, or `.work/`, LocalKit discovers and runs them immediately.

---

## 1. Install the Runtime & Native Harnesses

From a checkout on macOS or Linux with Python 3.12 and [`uv`](https://docs.astral.sh/uv/):

```sh
uv sync --extra mlx --extra executorch --extra litert
```

> **Note:** On Apple Silicon Macs, `--extra mlx` installs `mlx` and `mlx-lm` for native Metal GPU execution. On Linux ARM64 or x86_64, omit `--extra mlx`.

---

## 2. Inspect Your Live Hardware (`localkit device`)

Run `localkit device` (or `localkit doctor`) to probe your SoC, live reclaimable RAM, GPU working-set limits, installed harnesses, and any connected Android/Linux ADB targets:

```sh
uv run localkit device
```

```json
{
  "soc": "Apple M5 Max",
  "system": "darwin",
  "architecture": "arm64",
  "memory_mb": 131072,
  "available_memory_mb": 91060,
  "inference_budget_mb": 77401,
  "accelerator": {
    "type": "metal_gpu",
    "device_name": "Apple M5 Max",
    "architecture": "applegpu_g17s",
    "max_recommended_working_set_mb": 110100,
    "max_buffer_length_mb": 82575
  },
  "harnesses": {
    "mlx": "0.32.2",
    "litert_lm": "0.18.0",
    "litert": "2.3.0",
    "executorch": "1.5.1"
  }
}
```

### What LocalKit just calculated:
- **`available_memory_mb` (`91,060 MB`):** Reclaimable memory computed from OS page statistics (`vm_stat` on macOS, `/proc/meminfo` on Linux/Android)—not just total physical RAM.
- **`inference_budget_mb` (`77,401 MB`):** The safe ceiling for model weights + KV cache + activations ($\min(0.85 \times \text{Available RAM}, 0.90 \times \text{Metal Max Working Set})$).

---

## 3. Audit Your Local Models (`localkit models`)

Before loading anything, see which models in your local caches are complete, how much KV-cache memory they consume per 1,000 tokens, and what context window LocalKit recommends on your current machine:

```sh
uv run localkit models
```

| Local Model | Harness | Weights | KV Cache / 1K Tokens | Recommended Context | Author Defaults (`generation_config.json`) |
| --- | --- | --- | --- | --- | --- |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | `mlx` (Metal GPU) | 2.13 GB | 144.0 MB | **65,536 tokens** | `temp=0.7, top_p=0.8, top_k=20` |
| `mlx-community/Qwen2.5-Coder-32B-Instruct-4bit` | `mlx` (Metal GPU) | 17.16 GB | 256.0 MB | **32,768 tokens** | `temp=0.7, top_p=0.8, top_k=20` |
| `Qwen/Qwen3-4B-Instruct-2507` | `mlx` (Metal GPU) | 7.67 GB | 144.0 MB | **65,536 tokens** | `temp=0.7, top_p=0.8, top_k=20` |
| `gemma4.litertlm` (`.work/gate-gemma4-models`) | `litert` (`litert-lm`) | 2.41 GB | Managed | **32,768 tokens** | Engine default |

### Honest Diagnostics for Broken Cache Directories

Hugging Face cache directories frequently contain interrupted downloads or missing `.safetensors` shards. Instead of crashing inside a C++ loader, `localkit models` (and `localkit models --show-incomplete`) inspects `model.safetensors.index.json` and `snapshots/` up front:

```sh
uv run localkit chat deepseek --prompt "Hello"
```

```json
{
  "error": {
    "code": "incomplete_model_cache",
    "message": "Found matching cache entries for 'deepseek', but their downloads are incomplete: mlx-community/DeepSeek-V4-Flash-3bit: Missing snapshots/ directory (download did not complete); mlx-community/DeepSeek-V4-Flash-4bit: Missing snapshots/ directory (download did not complete)",
    "details": {
      "incomplete_matches": [
        {
          "repo_id": "mlx-community/DeepSeek-V4-Flash-3bit",
          "reason": "Missing snapshots/ directory (download did not complete)"
        },
        {
          "repo_id": "mlx-community/DeepSeek-V4-Flash-4bit",
          "reason": "Missing snapshots/ directory (download did not complete)"
        }
      ]
    }
  }
}
```

---

## 4. Run Inference by Short Name (`localkit chat`)

Pass a model family name (`qwen3`, `gemma4`, `qwen2.5-coder`) directly to `localkit chat`. LocalKit selects the best local artifact, stages its weights into `~/.cache/localkit/packages/` using zero-copy hardlinks (`os.link`), applies the author's sampling parameters, and runs it in an isolated worker process:

### Example A: Apple Metal GPU via `mlx-lm`

```sh
uv run localkit chat qwen3 --prompt "Reply with one short sentence confirming what model you are."
```

```json
{
  "model": "qwen3",
  "backend": "mlx",
  "variant": "qwen3-mlx",
  "device": "Device(gpu, 0)",
  "response": "I am Qwen3, a large language model developed by Alibaba Cloud.",
  "generation_tokens_per_second": 227.3,
  "prompt_tokens_per_second": 467.8,
  "peak_memory_gb": 2.38,
  "latency_ms": 106.5,
  "weights_verified": true
}
```

### Example B: Cross-Platform LiteRT-LM (`.litertlm`)

```sh
uv run localkit chat gemma4 --prompt "Say hello in five words."
```

```json
{
  "model": "gemma4",
  "backend": "litert",
  "variant": "litert-cpu",
  "engine": "litert-lm-api 0.18.0",
  "response": "Hello, hope you are well!",
  "time_to_first_token_ms": 223.8,
  "latency_ms": 567.7,
  "weights_verified": true
}
```

---

## Next Steps

- Understand how LocalKit calculates KV-cache physics and stages models in **[How LocalKit Works](how-it-works.md)**.
- Lock a model across Mac, Android, and Linux target profiles in **[Target Locking & `localkit.lock.json`](configuration.md)**.
- Serve an OpenAI-compatible HTTP endpoint or embed the SDK in **[OpenAI Server & Native SDKs](sdk-and-api.md)**.
