# Review follow-ups

Three issues from a code review of the Python runtime and the repository. Each
entry lists the evidence, the proposed fix, and what counts as done. The
TypeScript and Android SDKs were not reviewed in depth.

| Priority | Issue | Area |
| --- | --- | --- |
| 1 | The README leads with hardware paths that have not been tested | Positioning |
| 2 | Long-held locks in tool permissions and release updates | Python runtime |
| 3 | No license, no CI, the version is repeated in several files, and docs are duplicated | Repository |

## 1. The README promises more than has been tested

**Problem.** The README opens with Qualcomm NPU targets: Galaxy S25 Ultra / HTP
v79, Ventuno Q / QCS8275 and Snapdragon X Elite
([README](../README.md), "why this exists"). The device notes in
[team handoff](team-handoff.md#device-notes) say:

- Qualcomm phone NPU execution has not been tested.
- Ventuno Q board NPU and camera testing is still needed. Linux arm64 ran in a VM.
- Windows native integration, packaging and device testing are still needed.

The paths that have been tested are Apple silicon CPU, Pixel CPU/GPU adapter
updates, Linux arm64 CPU in a VM, and web LiteRT. The `examples/mobile-app`,
`examples/iot-app` and `examples/desktop-app` folders contain only `.gitkeep`.

**Fix.** Do one of these:

- Reword the README opening so it leads with the tested paths. List the
  Qualcomm devices as targets and label their status, using the device notes table.
- Or run one Qualcomm NPU path end to end, for example ExecuTorch QNN on the
  S25 (HTP v79) or the Ventuno Q, and record the evidence the way
  `sdk/android/portable-adapters/evidence/` does.

Either way, remove the empty example folders or link them as planned work.

**Done when** every device the README names either has recorded run evidence
or carries an explicit untested or planned label.

## 2. Long-held locks in tool permissions and release updates

### Policy lock held for the whole tool call

**Problem.** `PermissionPolicy.authorize` takes the policy `RLock` and keeps
holding it while it yields to the caller
([permissions.py](../phaseone/runtime/permissions.py), `authorize`).
`ToolGateway.invoke` runs the tool handler inside that context
([tools.py](../phaseone/runtime/tools.py), `invoke`). As a result:

- Every policy-checked tool call in a `Runtime` runs one at a time.
- `permissions.replace(document)` waits until the tool that is currently
  running returns. While a 2 s handler was running, `replace()` was measured
  blocking for 1.90 s. A slow or hung tool delays revocation for every later
  call.

The TypeScript `permissions.ts` and Android `Permissions.kt` have no
equivalent lock.

**Fix.** Evaluate the rules and capture `policy_revision` while holding the
lock, then release it before the handler runs. The documented semantics stay
the same: a check cannot undo an operation that has already started. The
receipt already records the revision that was evaluated.

**Done when** two policy-checked tools can run at the same time, `replace()`
returns while a handler is still running, and the next invocation sees the new
rules. Add a regression test with a blocking handler.

### One global lock for release updates

**Problem.** `ReleaseManager._write_lock` is a single `lock.sqlite3` lock with
a 300 s timeout
([releases.py](../phaseone/runtime/releases.py), `_write_lock`). `update()`
holds it through download and qualification for every capability. An update
for a second capability waits behind the first. If it waits more than 300 s it
fails with SQLite's `database is locked`, not a `RuntimeFault`.

**Fix.** Lock per capability. The objects in `objects/` are content-addressed
and written with `os.replace`, so they don't need a shared lock. Keep a short
global lock only around writes to `identities/`.

**Done when** two capabilities can update concurrently and a lock timeout
raises a `RuntimeFault` with a stable code.

### `open()` can run full qualification

**Problem.** `ReleaseManager.open()` calls `requalify()` when the device
fingerprint has changed (SDK, engine, firmware or security patch). That loads
the model and runs every release check synchronously on the caller's thread
([releases.py](../phaseone/runtime/releases.py), `open`).

**Fix.** Document this cost in [updates](updates.md). Alternatively, return
`requalification_required` from `open()` so the app can call `requalify()` on
a background executor.

**Done when** apps can find out before calling `open()` whether it will
requalify, or `open()` never does.

## 3. Repository hygiene

**Problem.**

- There is no `LICENSE` file at the root. Only third-party notices exist, under
  `sdk/typescript/licenses/`.
- There is no CI configuration. The Python, TypeScript and Kotlin shared
  test-case files in `sdk/contracts/` are only checked when someone runs each
  suite by hand.
- The runtime version `0.3.0` is hardcoded in `pyproject.toml`,
  `phaseone/runtime/api.py` (FastAPI title and `/health`),
  `phaseone/runtime/cli.py` (`--version`) and `phaseone/runtime/device.py`
  (`phaseone_version`, which feeds release resolution and the qualification
  fingerprint).
- `Release.minimum_runtime` in `phaseone/runtime/catalog.py` also defaults to
  `0.3.0`. That default belongs to the catalog format and should not
  automatically follow the package version.
- Two docs exist in two byte-identical copies, because the console serves the
  second copy at `/updates.md` and `/tools.md`:
  - `docs/updates.md` and `phaseone/runtime/static/updates.md`
  - `docs/tools-and-permissions.md` and `phaseone/runtime/static/tools.md`
- `examples/vision/sample-dog.jpg` and
  `sdk/android/demo/src/main/assets/sample-dog.jpg` are the same file.
- Branding is mixed. The repo and CLI are `dtr`, but the package, second
  console script, API title and error messages say `phaseone` / PHASEONE.

**Fix.**

- Choose and add a license.
- Add CI that runs:
  - `uv run --group dev pytest` and `uv run --group dev ruff check .`
  - `npm test`, `npm run typecheck` and `npm run lint:proofs` in
    `sdk/typescript`
  - the Gradle unit tests in `sdk/android`
- Read the version from package metadata (`importlib.metadata.version`) in
  `api.py`, `cli.py` and `device.py`.
- Keep a single source for each duplicated doc. Either copy it into `static/`
  at build time or fail CI when the copies differ.

**Done when** a pull request runs all three SDKs' shared test cases, the version
is changed in one place, and each doc has exactly one copy that people edit.
