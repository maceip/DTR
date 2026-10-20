# OpenAI Server & Native SDKs

LocalKit provides two integration seams so you can adopt it in five minutes without rewriting existing application logic:

1. **Local OpenAI-Compatible Server (`localkit serve`)** — Point any existing OpenAI Python, TypeScript, Go, or Rust client at `http://127.0.0.1:8080/v1`.
2. **Embedded Native SDKs (Python, Android/Kotlin, TypeScript)** — Embed LocalKit in-process with capability-scoped tool permissions and cryptographic model updates.

---

## 1. OpenAI-Compatible Local Server (`localkit serve`)

Start the server using any short model name (`qwen3`, `gemma4`) or package path:

```sh
uv run localkit serve qwen3 --port 8080
```

LocalKit resolves the model, stages its weights via zero-copy hardlinks, sets the hardware-appropriate context window and author sampling defaults, and exposes:

- `GET /v1/models` — Lists resolved models and active hardware backends.
- `POST /v1/chat/completions` — OpenAI-compatible chat completion endpoint.
- `GET /health` — Runtime health and live device telemetry.

### Using the Official OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local",  # No external API key required for loopback
)

response = client.chat.completions.create(
    model="qwen3",
    messages=[
        {"role": "system", "content": "You are a concise systems assistant."},
        {"role": "user", "content": "Why does KV cache grow linearly with sequence length?"},
    ],
)

print(response.choices[0].message.content)
```

### Using `curl`

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Say hello in five words."}]
  }' | jq
```

---

## 2. Embedded Python SDK

When you embed LocalKit inside a Python application, you get direct access to zero-config model resolution, isolated worker processes, vision classification, and the tool permission gateway:

```python
from phaseone.runtime import Runtime
from phaseone.runtime.resolver import resolve_model

# Resolve "qwen3" against live hardware and stage a zero-copy package
resolved = resolve_model("qwen3")

with Runtime() as runtime:
    app = runtime.load(resolved["package_path"])
    result = app.chat(
        resolved["model_name"],
        messages=[{"role": "user", "content": "List three advantages of unified memory."}],
    )
    print(result["response"])
    print(f"Ran on {result['backend']} ({result.get('device')}) at {result.get('generation_tokens_per_second')} tok/s")
```

### Vision Classification (`classify`)

```python
from phaseone.runtime import Runtime

with Runtime() as runtime:
    app = runtime.load(".work/vision")
    prediction = app.classify("vision", "examples/vision/sample-dog.jpg")
    print(prediction["label"], prediction["score"], prediction["backend"])
```

---

## 3. Android SDK (`sdk/android`)

The Android SDK (`ai.phaseone.runtime`) embeds native LiteRT, LiteRT-LM, and ExecuTorch execution inside your APK without requiring users to install a separate runtime app:

```kotlin
val runtime = PhaseoneRuntime(
    context = applicationContext,
    permissions = PermissionPolicy.fromJson(policyJson),
    subject = { currentUser.id }
)

val app = runtime.open("assistant")
val response = app.chat(
    model = "assistant",
    messages = listOf(ChatMessage(role = "user", content = "Summarize my notes."))
)
```

See **[Model Updates & Delivery](updates.md)** for over-the-air model qualification and rollback on Android.

---

## 4. TypeScript / Browser SDK (`sdk/typescript`)

The TypeScript SDK (`@phaseone/runtime`) pairs local WebGPU (`LiteRT.js` / `LiteRT-LM`) or a loopback native host with compile-time and runtime permission proofs (`Permix` + `gdp-ts`):

```ts
import { BrowserRuntime, PermissionPolicy } from '@phaseone/runtime';

const permissions = new PermissionPolicy({
  version: 1,
  rules: [
    { subject: 'alice', resource: 'notes:alice', action: 'read', effect: 'allow' },
  ],
});

const runtime = new BrowserRuntime(host, {
  permissions,
  subject: () => 'alice',
});
```

See **[Tools, Permissions, & Fine-Tuning Contract](tools-and-permissions.md)** for full details on tool registration and resource policies.
