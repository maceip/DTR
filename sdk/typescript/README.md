# @phaseone/runtime

One application interface for an explicitly paired native ExecuTorch/LiteRT host
and local browser LiteRT inference. The native host owns tool permissions and
execution. Model artifacts remain specific to their supported targets.

```ts
import {NativeHost, BrowserRuntime} from '@phaseone/runtime';

const host = new NativeHost('http://127.0.0.1:8920', pairingToken);
const remote = await host.application(packageDigest);
await remote.classify('vision', photoBlob, {backend: 'executorch'});
await remote.classify('vision', photoBlob, {backend: 'litert'});

const runtime = new BrowserRuntime(host);
const local = await runtime.application(packageDigest);
await local.classify('vision', photoBlob); // Local WASM model execution.
await runtime.close();
```

Use the supplied console or serve `dist/` intact: `index.js`, the classic
`browser-worker.js`, and both `wasm/` directories must remain available. They
are included in the local npm archive. Native-host-only clients do not download
the WASM files.

If your frontend bundler relocates the main module, copy the package's `dist/`
directory into `public/phaseone-runtime/` and set the asset locations explicitly:

```ts
const runtime = new BrowserRuntime(host, {
  workerUrl: new URL('/phaseone-runtime/browser-worker.js', location.origin),
  wasmRoot: new URL('/phaseone-runtime/wasm', location.origin).href,
});
```

Browser inference requires HTTPS or localhost. The host must allow the frontend's
exact origin with `serve --cors-origin ORIGIN` when they differ. Keep the pairing
token in application memory. It is not passed to the inference worker.

Chat uses `chat`/`chatStream`; persistent conversations use `createSession`,
`session`, `send`/`sendStream`, and explicit `recoverSession`. A stable `turn_id`
returns the saved result on retry. Pass an AbortSignal as `signal` to cancel.
To call declared tools through the runtime-owned loop, use `agent: true` in a
session send. Every operation reports its actual execution location.

For app-owned browser tools, supply `permissions: new PermissionPolicy(document)`
and `subject: () => currentUser.id` to `BrowserRuntime`, then register handlers on
`runtime.tools`. That gateway runs locally, uses Permix and gdp-ts, and persists
action records in IndexedDB. Without this option, tools retain the paired-host
path. A denied local call never falls back to a different host or engine.
`capabilities(model)` reports tool protocols and the unavailable training status;
`fineTune(request)` is a consent-checked stub. See [tools and permissions](../../docs/tools-and-permissions.md).

Local browser support is LiteRT.js 2.5.3 tensors and LiteRT-LM 0.18.1 web models.
ExecuTorch uses the paired native host. The current tensor binding exposes
float32/int32/uint8 public I/O and static shapes. Unsupported requests return
structured errors instead of silently moving execution elsewhere.

Run `npm ci`, `npm run build`, `npm run typecheck`, `npm run lint:proofs`, and
`npm test` in this folder. `npm pack` creates
the local installable SDK archive; publishing is disabled. See
`THIRD_PARTY_NOTICES.md` and `patches/README.md` for bundled dependencies and the
guarded LiteRT-LM binding compatibility change.
