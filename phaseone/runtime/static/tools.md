# Tools, data permissions, and fine-tuning

**Keep the same tools and app permissions when DTR changes inference engines.**
A model proposes a function call. DTR validates its arguments, resolves the
affected resources using app code, checks access, then invokes an app-owned
handler. A downloaded model or adapter does not grant itself new permissions.

## One boundary, two complementary libraries

The TypeScript SDK composes [Permix](https://permix.letstri.dev/docs/guide/check)
and [gdp-ts](https://github.com/rauchg/gdp-ts):

1. **Permix makes the permission decision.** The app supplies the current user
   and rules; model arguments cannot supply either authority.
2. **gdp-ts carries the checked decision through TypeScript.** The privileged
   executor requires a proof about that specific user and request. The prover
   stays private; the upstream lint rules and compile-time tests guard its use.
3. **DTR checks it at runtime too.** Each proof gets a unique, instance-owned
   handle, bound to the request, arguments, and policy revision. Forgery, reuse,
   changed arguments, and replacement of the policy invalidate it. No proof is
   accepted over HTTP or reconstructed from JSON.

gdp-ts is a compiler guardrail, not cryptographic attestation or a sandbox.
Python and Kotlin enforce the same [resource policy](../sdk/contracts/permissions.schema.json)
at their native gateways; they do not embed JavaScript to run Permix. Shared
[cases](../sdk/contracts/permissions-cases.json) check the decisions in all three.

## The developer interface

Use your app's existing authentication and consent flow to construct rules.
There is no additional login, DTR app, permission server, or background service.
The first policy format is deliberately small: exact subject/resource matches,
default deny, and explicit deny taking precedence. Every resource required by a
tool must pass. Reading, writing, exporting, executing, and training are separate
actions; none implies another.

```ts
import {BrowserRuntime, PermissionPolicy} from '@phaseone/runtime';

const permissions = new PermissionPolicy({version: 1, rules: [
  {subject: 'alice', resource: 'notes:alice', action: 'read', effect: 'allow'},
  {subject: 'alice', resource: 'code:patch', action: 'write', effect: 'allow'},
]});
const runtime = new BrowserRuntime(host, {
  permissions,
  subject: () => currentUser.id, // From your app's authentication, not a prompt.
});
runtime.tools!.register({
  definition: {
    name: 'read_note', description: 'Read a note', permission: 'notes.read',
    parameters: {type: 'object', properties: {id: {type: 'string'}},
                 required: ['id'], additionalProperties: false},
  },
  access: args => [{resource: 'notes:' + args.id, action: 'read'}],
  handler: args => notes.read(String(args.id)),
});
const app = await runtime.application(packageDigest);
// The package must also declare this tool and include it in its agent tool list.
await app.send(sessionId, 'Read my note', {agent: true});
```

The resource resolver must use trusted identifiers/metadata and enumerate **all**
affected resources. Do data reads and effects in the checked handler, not in the
resolver. For tenant data, resolve the record's ownership from your database;
do not trust a proposed `ownerId`. New rules from the app's consent system can
be installed with `permissions.replace(document)`. The next invocation checks
the new policy, including requests for cached tool results.

In Python, pass `permissions=PermissionPolicy(document), subject="alice"` to
`Runtime`, then register tools with `runtime.tool_gateway.register(...,
access=lambda args: [...])`. Existing workspace tools bind to exact IDs such as
`workspace:patch.py`. Old `granted_permissions` remains a legacy host API;
supplying a resource policy never falls back to those broad grants.

On Android, pass `permissions` and a `subject` callback to `PhaseoneRuntime`.
Register handlers on `app.tools` after opening a release. `app.chat(..., tools=...)`
returns typed `ToolCall` proposals, which the app passes to
`app.tools.invoke(call, actionId)`. Android currently supports the JSON tool
protocol and closed, flat schemas with primitive parameters. Unsupported schemas
or native tool protocols fail explicitly. The automatic Android agent loop is
not implemented; Python/browser packaged-agent loops already use their gateways.

Action IDs come from the application, not the model's guessed call ID. Python,
browser, and Android gateways persist pending actions before handler entry;
uncertain effects cannot blindly replay after a failure. Standalone TypeScript
`ToolGateway` uses bounded memory unless given a durable `ToolActionStore`.
It is suitable for the example below; use durable storage in a long-lived app.
This is crash-conservative execution, not a distributed exactly-once guarantee.

## Function-calling fine-tuning: the stub

The application APIs now have `capabilities(model)` and `fine_tune` (Python) /
`fineTune` (TypeScript/Kotlin). A fine-tuning request names a model and dataset
resource; it accepts the `function_calling` or `chat` objective and the `lora`
method. It requires both `read` and `train` permission on the dataset.

```python
app.fine_tune({
    "model": "assistant",
    "dataset": "dataset:consented-examples",
    "objective": "function_calling",
    "method": "lora",
})
```

**No trainer is installed.** An unauthorized request returns `permission_denied`;
an authorized request returns `fine_tuning_unavailable`. The stub reads no
dataset, uploads nothing, changes no weights, and fabricates no job or success.
Trainer adapters, dataset validation, adapter export and qualification remain
future work. Supporting the tool protocol also does not establish a model's
tool-selection accuracy; capability reports keep that distinction explicit.

## What this protects

A local assistant can read a permitted note and save a generated patch while
being denied another user's note, an upload, code execution, or training use.
The checks stay outside the model and apply to every registered backend's tool
proposals. Try the deterministic example without downloading weights:

```sh
npm --prefix sdk/typescript ci
npm --prefix sdk/typescript run build
node examples/permissions/demo.mjs
```

These are permissions on operations routed through DTR. They do not sandbox
trusted app handlers, track information flow after text enters a conversation,
erase conversation history on revocation, or protect against a compromised
native engine. Apps still own authentication, storage isolation and history
retention. The optional HTTP host has one app-supplied identity per runtime;
it is not a multi-user authentication service.

Protecting generated code means controlling its read/write/export/execute paths.
It does not make code or model weights shipped to a device secret from that
device's owner. No DRM, attestation, or proprietary-code secrecy claim is made.
