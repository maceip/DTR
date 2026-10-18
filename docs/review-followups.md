# Review follow-ups

Issues from a code review of the Python runtime and the repository. Each entry
lists the evidence, the proposed fix, and what counts as done. The TypeScript and
Android SDKs were not reviewed in depth.

| Priority | Issue | Area |
| --- | --- | --- |
| 1 | The README leads with hardware paths that have not been tested | Positioning |
| 2 | Long-held locks in tool permissions and release updates | Python runtime |
| 3 | No license, no CI, the version is repeated in several files, and docs are duplicated | Repository |
| 4 | Python and Android computed different release digests for the same catalog (fixed) | Cross-SDK contract |
| 5 | Smaller runtime issues and remaining test gaps | Python runtime |

## 1. The README promises more than has been tested

**Problem.** The README opens with Qualcomm NPU targets: Galaxy S25 Ultra / HTP
v79, Ventuno Q / QCS8275 and Snapdragon X Elite
([README](../README.md), opening device list). The device notes in
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
- Branding is mixed. The README and Python package say phaseone, while the
  repo, the `dtr` CLI shown in the README and the `dtr/` package say dtr.

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

## 4. Python and Android computed different release digests (fixed)

A release digest is its identity. It names the install directory, drives the
`release_conflict` check, and is recorded in activation metadata.

**Problem.** Python hashed the validated model, which added defaults and
re-spelled numbers through float `repr` (`3e-3` became `0.003`). Android hashes
the catalog JSON as received. The two agreed only for catalogs written by
Python's `signed_catalog`.

**Fix.** A release has one identity: the SHA-256 of its JSON as published.

- Python's `decode_catalog` now hashes each release from the catalog bytes, as
  Android does. It reads the bytes with `read_json(..., keep_number_text=True)`
  and serializes them with `canonical_source()` (`phaseone/runtime/package.py`).
- A release built in Python hashes the expanded form that `signed_catalog`
  publishes, so publisher and device digests agree.
- `canonical_source()` matches Android's Gson-based `canonical()` byte for
  byte:
  - it keeps the source number spelling
  - it sorts keys by UTF-16 code unit
  - it escapes U+2028/U+2029

**Evidence.**

- The shared case `published-number-spelling-and-defaults` in
  `sdk/contracts/resolver-cases.json` omits `minimum_runtime` and writes
  numbers as `3e-3`, `1E-7` and `2.50`.
- The Python vector test decodes every shared case from its source bytes.
- A Java port of the Kotlin `parseObject()`/`canonical()`, run against Gson
  2.14.0, reproduced every expected digest, including the new case. The
  Gradle suite itself was not run.

**Existing installs.** Every signed catalog comes from `signed_catalog`, so a
signed Python install keeps its digest. The exception is a release with
U+2028/U+2029 in a string, which is now escaped. An install from a
hand-written, unsigned development catalog gets a new digest:

- `open()` fails with `artifact_integrity`.
- `update()` reports `up_to_date`, because it compares release IDs.

To recover, delete the runtime's `releases/` directory and update again.
Android is unchanged.

## 5. Smaller runtime issues and remaining test gaps

- **Catalog freshness.** Signed catalogs have no expiry or timestamp. A server
  that keeps serving an old catalog goes unnoticed. `stale_catalog` only
  catches a catalog older than the active release. There is one signing key
  and no rotation.
- **Windows.** `atomic_json(private=True)` calls `os.fchmod`, which Python 3.12
  doesn't provide on Windows. The workspace tools are POSIX-only.
- **Chat qualification checks** pass on substring matches and parse JSON by
  stripping Markdown code fences
  ([releases.py](../phaseone/runtime/releases.py), `_run_checks`).
- **Downloads** have no resume or retry and use a fixed 120 s timeout. Every
  cache hit hashes the whole object again.
- **Test gaps.** Release lifecycle (`tests/test_runtime_release_lifecycle.py`)
  and worker crash isolation (`tests/test_runtime_worker.py`) now run without
  native engines. These areas are still untested:
  - image and media validation (`media.py`)
  - NDJSON streaming, the 64 MiB body limit, origin denial and job
    cancellation in `api.py`
  - the `dtr build` / export path (`build.py`, `export_worker.py`)
  - the native suite, which skips unless the `executorch` and `litert` extras
    are installed
