# LocalKit

**The hardware-aware runtime and resolver for local AI across consumer devices.**

Shipping a local model across user devices breaks because three things change on every machine: the native execution harness (`MLX` on Apple Metal, `LiteRT-LM` on Google Tensor/GPU, `ExecuTorch` on Qualcomm/ARM), the live available RAM budget, and the safe KV-cache context window. LocalKit sits between your application and native vendor engines—probing live silicon, auditing local model caches for broken shards, computing exact KV-cache memory limits from `config.json`, staging weights via zero-copy hardlinks (`os.link`), and executing directly on the fastest installed native harness.

### 👉 **[Read the Full Documentation & Visual Architecture Guide → `maceip.github.io/LocalKit`](https://maceip.github.io/LocalKit/)**

| Documentation Track | Direct Link |
| --- | --- |
| **1. First Run** | **[Get Started in 60 Seconds](https://maceip.github.io/LocalKit/#/get-started)** — Live hardware discovery, cache audit, and zero-config chat |
| **2. Mental Model** | **[How LocalKit Works](https://maceip.github.io/LocalKit/#/how-it-works)** — Control Plane vs. Execution Plane, KV-cache physics, and zero-copy staging |
| **3. Lock Targets** | **[Target Locking (`localkit.lock.json`)](https://maceip.github.io/LocalKit/#/configuration)** — Lock safe context windows across Mac, Android, and Linux SBCs |
| **4. Build Apps** | **[OpenAI Server & Native SDKs](https://maceip.github.io/LocalKit/#/sdk-and-api)** — Drop-in `/v1/chat/completions` server + Python, Kotlin, & TypeScript SDKs |
| **5. Reference** | **[CLI Reference](https://maceip.github.io/LocalKit/#/cli-reference)** · **[Mac Engines](https://maceip.github.io/LocalKit/#/mac-engines)** · **[Tools & Permissions](https://maceip.github.io/LocalKit/#/tools-and-permissions)** · **[Model Updates](https://maceip.github.io/LocalKit/#/updates)** |

---

## Instant Quickstart (Zero Config Files)

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/) on macOS or Linux. If you already have models in `~/.cache/huggingface/hub` or `.work/`, LocalKit resolves and runs them out of the box:

```sh
# 1. Install runtime + native engines (omit --extra mlx on Linux)
uv sync --extra mlx --extra executorch --extra litert

# 2. Probe live silicon, reclaimable RAM, and GPU working-set limits
uv run localkit device

# 3. Scan local caches, compute KV-cache recipes, and flag broken downloads
uv run localkit models

# 4. Run a model by short name (zero-copy staged onto Metal GPU / LiteRT-LM)
uv run localkit chat qwen3 --prompt "Reply in one sentence confirming what model you are."
uv run localkit chat gemma4 --prompt "Say hello in five words."

# 5. Or start an OpenAI-compatible server on http://127.0.0.1:8080/v1
uv run localkit serve qwen3 --port 8080

# 6. Lock a cross-device compatibility matrix (localkit.toml + localkit.lock.json)
uv run localkit add qwen3
```

---

## Where Things Live

| Path | Contents |
| --- | --- |
| [`localkit/`](localkit/) & [`phaseone/runtime/`](phaseone/runtime/) | Public Python SDK (`localkit`) and core control plane: live silicon probe ([`device.py`](phaseone/runtime/device.py)), zero-config resolver & KV-cache sizer ([`resolver.py`](phaseone/runtime/resolver.py)), zero-copy packager ([`package.py`](phaseone/runtime/package.py)), OpenAI/HTTP server ([`api.py`](phaseone/runtime/api.py)), and CLI ([`cli.py`](phaseone/runtime/cli.py)). |
| [`sdk/android/`](sdk/android/) | Native Android/Kotlin runtime SDK for LiteRT, LiteRT-LM, ExecuTorch, and signed release updates. |
| [`sdk/typescript/`](sdk/typescript/) | TypeScript & browser runtime SDK with `Permix` + `gdp-ts` permission proofs. |
| [`sdk/contracts/`](sdk/contracts/) | Shared JSON schemas and cross-language conformance test vectors (resolver, catalog digests, and permissions). |
| [`docs/`](docs/) | Multi-page documentation site (served via GitHub Pages at **[`maceip.github.io/LocalKit`](https://maceip.github.io/LocalKit/)**). |
| [`tests/`](tests/) | Automated test suite (`uv run --group dev pytest`). |
