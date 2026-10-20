# Target Locking & `localkit.lock.json`

Developers know the model family they want (`qwen3`, `gemma4`, `qwen2.5-coder`), not what context window or quantization fits a customer's 16 GB MacBook Air vs. a 12 GB Android flagship vs. a 4 GB Linux SBC.

Instead of asking you to hand-author JSON manifests with guessed context lengths, LocalKit uses a two-file workflow modeled after modern package managers:
1. **`localkit.toml`** — Your high-level intent (which model capability your project uses).
2. **`localkit.lock.json`** — The machine-generated hardware compatibility and memory-physics lockfile across consumer device tiers.

---

## 1. Lock a Model for Your Project (`localkit add`)

Run `localkit add <model>` in your project directory:

```sh
uv run localkit add qwen3
```

 This writes a minimal `localkit.toml`:

```toml
[models]
default = "qwen3"
```

And generates `localkit.lock.json` by inspecting the model's `config.json`, `generation_config.json`, weight files, and KV-cache growth rate across consumer hardware targets.

---

## 2. What Lives Inside `localkit.lock.json`

For `mlx-community/Qwen3-4B-Instruct-2507-4bit` ($L=36$ layers, $H_{kv}=8$ KV heads, $D_h=128$ head dimension, $2.13\text{ GB}$ weights), LocalKit computes:
- **KV Cache Footprint:** $2 \times 36 \times 8 \times 128 \times 2 = 147,456\text{ bytes/token}$ (**$144.0\text{ MB}$ per 1,000 tokens**).
- **Author Sampling Contract:** `temperature = 0.7`, `top_p = 0.8`, `top_k = 20` (extracted from `generation_config.json`).

It then evaluates the model against four standard consumer hardware profiles:

| Target ID | Hardware Profile | Safe Inference Budget | Locked Context Window | KV Cache at Max Context | Status |
| --- | --- | --- | --- | --- | --- |
| `apple_silicon_128gb` | Apple M5 Max / Studio (128 GB) | 98,304 MB | **65,536 tokens** | 9,216 MB | `ready` (`mlx` Metal GPU) |
| `apple_silicon_16gb` | MacBook Air / Pro (16 GB) | 11,468 MB | **65,536 tokens** | 9,216 MB | `ready` (`mlx` Metal GPU) |
| `android_flagship_12gb` | Snapdragon 8 Elite / Tensor G5 (12 GB) | 5,632 MB | **16,384 tokens** | 2,304 MB | `needs_conversion` (`.litertlm` or `.pte` required) |
| `edge_sbc_4gb` | Arduino Ventuno Q / Linux SBC (4 GB) | 2,457 MB | **8,192 tokens** | 1,152 MB | `needs_conversion` (`.litertlm` or `.pte` required) |

### Why `needs_conversion` Is Reported Honestly
An `mlx-community` 4-bit `.safetensors` checkpoint uses Apple Metal quantization kernels (`mlx.metal`), so an Android phone or Linux SBC cannot execute those weight tensors directly. Rather than pretending an MLX checkpoint will run on Android, `localkit.lock.json` verifies that a 4-bit 4B model **physically fits the phone's 5.6 GB inference budget at a 16,384-token context window**, and explicitly records that a `.litertlm` or `.pte` variant must be imported for non-Apple targets.

---

## 3. Adding Cross-Platform Variants to One Package (`localkit import`)

When you are ready to ship a multi-target package directory that runs on both macOS and Android/Linux, import each target's native artifact under the same logical model name (`assistant`):

```sh
# 1. Apple Silicon Metal GPU variant (zero-copy hardlinked)
uv run localkit import ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/<rev> \
  .work/assistant-pkg --backend mlx --model assistant --variant mlx-4bit

# 2. Cross-platform LiteRT-LM variant for Android / Linux / macOS CPU & GPU
uv run localkit import path/to/model.litertlm \
  .work/assistant-pkg --backend litert --model assistant --variant litert-int4 --context 16384
```

Now `.work/assistant-pkg` contains both variants verified by SHA-256. On a Mac, LocalKit selects `mlx-4bit` on the Metal GPU; on Linux or Android, it selects `litert-int4`.

---

## 4. Ranking Variants on Your Own Data (`localkit compare`)

Before locking a quantization for production, test how every runnable variant performs on your actual prompts:

```sh
echo '{"prompt": "Which planet is the Red Planet? One word.", "expected": "Mars"}' > .work/eval.jsonl
uv run localkit compare .work/assistant-pkg --model assistant --inputs .work/eval.jsonl
```

See **[Comparing Model Variants](compare.md)** for full details on accuracy, agreement, latency, and peak memory ranking.
