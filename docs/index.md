# LocalKit Documentation

**The hardware-aware runtime and resolver for local AI across consumer devices.**

When you ship a local model to users, **three things change on every machine**:
1. **The Native Harness:** Apple Silicon wants `.safetensors` on **MLX Metal** (`mlx-lm`), Android flagships want `.litertlm` on **LiteRT-LM**, and Qualcomm/ARM targets want `.pte` on **ExecuTorch / QNN**.
2. **The Live Memory Envelope:** Static RAM specs lie when other apps are running. A context window that fits on a 128 GB Mac Studio will swap-thrash a 16 GB MacBook Air and immediately OOM-kill a 12 GB phone.
3. **The Author's Sampling Contract:** Every model family ships its own `generation_config.json` (`temperature`, `top_p`, `top_k`, stop tokens, chat templates). Guessing defaults silently degrades accuracy.

LocalKit sits between your application and native vendor engines. It probes live silicon and reclaimable RAM, audits local model caches for missing shards, computes the exact KV-cache memory budget from `config.json`, stages verified weights via zero-copy hardlinks (`os.link`), and executes directly on the fastest installed native harness.

```mermaid
flowchart LR
    subgraph App["Your Application / CLI"]
        CLI["`localkit chat qwen3`\n`localkit serve qwen3`"]
        SDK["OpenAI Client or\nPython · Kotlin · TS SDK"]
    end

    subgraph Control["LocalKit Control Plane"]
        Probe["Live Silicon & RAM Probe\n(`sysctl` · `vm_stat` · Metal · ADB)"]
        Resolver["Zero-Config Resolver\nKV Physics · Shard Audit · `os.link`"]
        Gateway["Tool Permission Gateway\n& Signed Release Manager"]
    end

    subgraph Engines["Native Execution Harnesses"]
        MLX["Apple Metal GPU\n(`mlx-lm`)"]
        LiteRT["Cross-Platform CPU / GPU / NPU\n(`litert-lm` · `ai-edge-litert`)"]
        ET["ARM64 / Qualcomm QNN\n(`executorch` · `xnnpack`)"]
    end

    CLI --> Probe
    SDK --> Probe
    Probe --> Resolver
    Resolver --> Gateway
    Gateway --> MLX
    Gateway --> LiteRT
    Gateway --> ET
```

---

## Follow the Core Arc

| Stage | Guide | What You Will Learn |
| --- | --- | --- |
| **1. First Run** | **[Get Started in 60 Seconds](get-started.md)** | Inspect your live silicon (`localkit device`), audit local model caches (`localkit models`), and run `qwen3` or `gemma4` with zero configuration files. |
| **2. Mental Model** | **[How LocalKit Works](how-it-works.md)** | Understand the Control Plane vs. Execution Plane split, KV-cache sizing physics, shard integrity checks, and zero-copy staging. |
| **3. Lock Targets** | **[Target Locking & `localkit.lock.json`](configuration.md)** | Use `localkit add <model>` to compute safe context windows and memory envelopes across Mac, Android, and Linux SBCs. |
| **4. Build Apps** | **[OpenAI Server & Native SDKs](sdk-and-api.md)** | Point existing OpenAI SDK clients at `localkit serve` or embed the Python, Android/Kotlin, and TypeScript runtimes directly. |
| **5. Reference** | **[CLI Reference](cli-reference.md)** | Complete command reference for discovery, execution, benchmarking, variant comparison, signing, and OTA release updates. |

---

## Platform & Lifecycle Deep Dives

- **[Mac Engines (`mlx` & `local_server`)](mac-engines.md)** — In-process Apple Metal GPU execution and OpenAI-compatible local server bindings.
- **[Comparing Model Variants (`localkit compare`)](compare.md)** — Measure accuracy, agreement, latency, and peak memory across quantizations on your own JSONL eval set.
- **[Tools, Permissions, & Fine-Tuning Contract](tools-and-permissions.md)** — Capability-scoped resource access control across Python, TypeScript (`Permix` + `gdp-ts`), and Android.
- **[Model Updates & Delivery](updates.md)** — Cryptographic release signing, on-device qualification checks, and atomic rollback.
- **[Arduino Ventuno Q (`QCS8275`)](ventuno.md)** — Deploying and verifying packages over USB/ADB to Linux ARM64 hardware.
- **[Windows Target Notes](windows.md)** — Planned Windows ML / Snapdragon X Elite qualification path.

---

## Hardware Verification Matrix

Every claim in LocalKit is backed by recorded execution on physical hardware. Paths not yet qualified on physical silicon are labeled explicitly:

| Platform / Target | Tested Silicon | Verified Harnesses | Status & Evidence |
| --- | --- | --- | --- |
| **macOS (Apple Silicon)** | Apple M5 Max (128 GB, macOS) | `mlx-lm 0.32.2` (Metal GPU), `litert-lm 0.18.0`, `ai-edge-litert 2.3.0`, `executorch 1.5.1` (XNNPACK), `local_server` | **Verified** — 227 tok/s on Qwen3-4B (MLX 4-bit), 426 tok/s on Qwen3-0.6B, 2.7 ms MobileNetV3 on LiteRT. |
| **Linux ARM64 / IoT SBC** | Arduino Ventuno Q (`QCS8275`, Ubuntu 24.04) | `executorch 1.5.1`, `ai-edge-litert 2.3.0`, `litert-lm 0.18.0`, QNN 2.46.0 runner | **Verified** — Vision logits match macOS within `2.9e-5`; multi-engine chat session handoff verified over USB. |
| **Android** | Pixel 11 Pro XL & Pixel probe APK | Native LiteRT CPU & OpenCL GPU (`LITERT_CL`), ExecuTorch CPU, WebGPU LiteRT.js / LiteRT-LM | **Verified** — Signed release qualification, rollback, and GPU delegation verified on device. |
| **Windows (ARM64 / x64)** | Snapdragon X Elite / Windows 11 | Planned Windows ML / ONNX Runtime & native wheels | **Planned** — Not yet qualified on a physical Windows host. |
