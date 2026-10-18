# DTR Demo Recording Script — Click-by-Click (~2:30)

> **In 100% lockstep with the Demo Director dashboard at `http://127.0.0.1:8766/demo-director.html`**

---

## Pre-Flight Check (All verified green)
- **Deck**: `http://127.0.0.1:8911/static/presentation.html`
- **Site / Console**: `http://127.0.0.1:8911/` *(Operator token: `JiV3bYL4m9fIxcXHXxSd4UPwpMqGcZlem2c58aIwKZQ`)*
- **Demo Director**: `http://127.0.0.1:8766/demo-director.html`
- **Connected Hardware**: Arduino VENTUNO Q via ADB (`serial: 1050156610`)
- **Terminal Working Directory**: `/Users/mac/DTR`

---

## Hook & Problem (0:00–0:20) — Screen: Pitch Deck

**Look:** `http://127.0.0.1:8911/static/presentation.html` (Slide 1 → Slide 2)

| Step | Action |
|---|---|
| 1 | Focus deck tab on Slide 1: **PHASEONE[tpu] · DTR** |
| 2 | Advance to **Slide 2** (`→` or `Space`) — *DEVELOPING FOR EDGE AI IS AGONY TODAY* |
| 3 | Hover the pain cards: **COMPILER HELL → REFLASH → AGENTS WITHOUT PROOFS** |

**Say:**
> “Every edge AI app reimplements the same nightmare — ExecuTorch here, LiteRT there, WASM somewhere else. **DTR ends that.** Your app names a capability. We pick the model, run it, and hand you a receipt. Shipping on-device AI forces app teams to become compiler and driver engineers across an N×M matrix. One missing driver bricks the pipeline; model updates shouldn't mean firmware flashes.”

---

## 1 · Console & Hardware Rail (0:20–0:50) — Screen: GUI `:8911`

**Look:** `http://127.0.0.1:8911/` (Capabilities & Fleet Rail)

| Step | Action |
|---|---|
| 1 | Switch to tab `http://127.0.0.1:8911/` |
| 2 | Point at **CONTRACT FIRST · ENGINE SECOND** in Architecture flow |
| 3 | Scroll down to the **Inference hardware rail** |
| 4 | Point at the **Arduino VENTUNO Q** card alongside Pixel G6 and Apple Silicon |
| 5 | Point at the evidence tags: **OBSERVED** vs **ABSENT / ECOSYSTEM** |

**Say:**
> “You write to a capability — vision, chat — not an engine API. Underneath: ExecuTorch and LiteRT, a model release lifecycle, and Gate as the hardware example on **Arduino VENTUNO Q** — Qualcomm QCS8275 with STM32 co-processor. In the fleet rail, we label evidence honestly: OBSERVED is a real run on that device. ECOSYSTEM or ABSENT is never an unbacked claim.”

---

## 2 · HARDWARE PROOF (REQUIRED) (0:50–1:20) — Screen: Terminal

**Look:** Terminal window in `/Users/mac/DTR`

| Step | Action |
|---|---|
| 1 | Switch to Terminal |
| 2 | Run the verified live board check: |

```bash
cd /Users/mac/DTR
.venv/bin/python scripts/check_runtime_board.py --package .work/vision
```

| Step | Action |
|---|---|
| 3 | Point at the JSON output on screen: `{"passed": true, "serial": "1050156610"}` |
| 4 | (Optional) Show the receipt summary in `.work/runtime-board.json`: ExecuTorch 16.2ms, LiteRT 8.9ms, `maximum_absolute_error: ~3e-5` |

**Say:**
> “Here is the proof on real silicon. Connected Arduino VENTUNO Q — serial **1050156610**. DTR executes the vision capability on the board across both engines: ExecuTorch XNNPACK CPU and LiteRT CPU. Both resolve to identical Samoyed predictions with a maximum absolute error of **3e-5**. Same capability contract, real hardware evidence, zero compiler rewrite.”

---

## 3 · Discovery & Gateway / Permissions (1:20–1:50) — Screen: GUI `:8911`

**Look:** `http://127.0.0.1:8911/` (`#capabilities` & `#permissions`)

| Step | Action |
|---|---|
| 1 | Click **Capabilities** nav → point to resolved candidates |
| 2 | Click **Permissions** nav |
| 3 | Click **Ask Permix · issue proof** → click **Invoke** (show ALLOW receipt) |
| 4 | Click **Tamper proof** → show instant fail-closed DENY |

**Say:**
> “Compiled formats resolve dynamically on device. But tools need cryptographic proof — model reasoning cannot fabricate authority. In DTR Permix, invocation requires a cryptographic ticket. If a prompt injection alters the payload, the proof tampers and fails closed.”

---

## 4 · Operator Console & Labeled Replay (1:50–2:15) — Screen: GUI `:8911`

**Look:** `http://127.0.0.1:8911/#console`

| Step | Action |
|---|---|
| 1 | Scroll to **CONNECT GATE CONSOLE** / **Stage Sequence** |
| 2 | Click **Run first scene ↗** (labeled replay) |
| 3 | Show the green **ALLOW** verdict appear under **Decision receipts** with hash-chained receipt ID |
| 4 | Click **Run next scene ↗** → show the **DENY** verdict on unauthorized injection |

**Say:**
> “In the Gate operator console: camera plus tools. Labeled replay: scoped ALLOW for the confirmed operator intent, DENY when a model oversteps. Approval is strictly bounded to this command, this directory, and this session. Broader work demands fresh confirmation.”

---

## 5 · Code Beat & Close (2:15–2:30) — Screen: Slide / Editor

**Look:** VS Code or Deck Slide showing the clean Python surface

```python
from pathlib import Path
from dtr import Runtime

with Runtime(".dtr") as runtime:
    app = runtime.load(".work/vision")
    result = app.classify("vision", Path("photo.jpg").read_bytes())
```

**Say:**
> “This 6-line block is your entire integration surface. Engine selection, qualification, and rollback live in DTR — not your application code.
> 
> **DTR — Distributed Tensor Routing.** One application API. Real engines on Arduino VENTUNO Q. Cryptographic receipts for what actually ran. **PHASEONE[tpu].**”

---

## Hard Rules While Recording
1. **Never say**: NPU, air-gapped, universal chip support.
2. **Honest claims**: OBSERVED means a verified execution on that exact device.
3. **If a button lags**: Narrate the receipt already visible; never go dead-air.
