import 'fake-indexeddb/auto';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash, generateKeyPairSync, sign} from 'node:crypto';
import {BrowserPackage, BrowserRuntime, NativeHost, RuntimeError, encodeTensor, decodeTensor, prepareTensor, dequantizeTensor} from '../dist/index.js';

globalThis.isSecureContext = true;
globalThis.location = new URL('http://localhost');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
function packageSource(change = () => {}) {
  const artifact = new Uint8Array([1, 2, 3]);
  const manifest = {schema_version: 1, name: 'contract-test', version: '1.0.0', artifacts: {model: {path: 'model.tflite', size: artifact.length, sha256: hash(artifact)}},
    models: {model: {task: 'tensor', variants: [{id: 'web-cpu', backend: 'litert', engine: 'compiled', artifact: 'model', target: {systems: ['web'], accelerator: 'cpu'},
      signatures: {default: {method: 'serving_default', inputs: [{name: 'x', shape: [1], dtype: 'float32'}], outputs: [{name: 'y', shape: [1], dtype: 'float32'}]}}}]}}};
  change(manifest);
  const bytes = new TextEncoder().encode(JSON.stringify(manifest));
  const digest = hash(bytes);
  return {bytes, digest, artifact, host: {
    response: async path => new Response(path.endsWith('/manifest') ? bytes : artifact),
    request: async () => { throw new RuntimeError('signature_not_found', 'Unsigned'); },
  }};
}

test('scalar, binary and quantized tensor contracts round trip', () => {
  const scalar = encodeTensor(new Int32Array([7]), []);
  assert.deepEqual(scalar.shape, []); assert.deepEqual([...decodeTensor(scalar)], [7]);
  const values = Float32Array.from({length: 2048}, (_, i) => i / 4);
  const binary = encodeTensor(values, [64, 32]); assert(binary.base64); assert.deepEqual(decodeTensor(binary), values);
  const spec = {name: 'x', dtype: 'uint8', shape: [4], quantization: {scales: [.5], zero_points: [128], axis: 0}};
  const quantized = prepareTensor({dtype: 'float32', shape: [4], data: [-1, 0, .25, .75]}, spec);
  assert.deepEqual(quantized.data, [126, 128, 128, 130]);
  assert.deepEqual(dequantizeTensor(quantized, spec).data, [-1, 0, 0, 1]);
  assert.throws(() => prepareTensor({dtype: 'float32', shape: [1], data: [256]}, {name: 'x', dtype: 'uint8', shape: [1]}), {code: 'dtype_mismatch'});
});

test('package bytes and artifact checksums are verified independently', async () => {
  const source = packageSource();
  const pkg = await BrowserPackage.open(source.host, source.digest, {cache: false});
  assert.equal(pkg.signature, 'unsigned');
  assert.deepEqual(new Uint8Array(await (await pkg.artifact('model')).arrayBuffer()), source.artifact);
  await assert.rejects(BrowserPackage.open(source.host, '0'.repeat(64)), {code: 'manifest_integrity'});
  const corrupt = packageSource();
  const original = corrupt.host.response;
  corrupt.host.response = path => path.endsWith('/manifest') ? original(path) : new Response(new Uint8Array([9, 2, 3]));
  const broken = await BrowserPackage.open(corrupt.host, corrupt.digest, {cache: false});
  await assert.rejects(broken.artifact('model'), {code: 'artifact_integrity'});
});

test('Ed25519 signatures require independently trusted public keys', async () => {
  const source = packageSource(), keys = generateKeyPairSync('ed25519');
  const publicBytes = keys.publicKey.export({format: 'der', type: 'spki'}).subarray(-32);
  source.host.request = async () => ({algorithm: 'ed25519', key_id: hash(publicBytes), public_key: publicBytes.toString('base64'), signature: sign(null, source.bytes, keys.privateKey).toString('base64')});
  assert.equal((await BrowserPackage.open(source.host, source.digest)).signature, 'valid_untrusted');
  await assert.rejects(BrowserPackage.open(source.host, source.digest, {requireSignature: true}), {code: 'untrusted_package'});
  assert.equal((await BrowserPackage.open(source.host, source.digest, {requireSignature: true, trustedKeys: [publicBytes.toString('hex')]})).signature, 'trusted');
});

test('cross-field manifest validation rejects unsafe paths and invalid dimensions', async () => {
  for (const change of [manifest => { manifest.artifacts.model.path = '../escape'; }, manifest => { manifest.models.model.variants[0].signatures.default.inputs[0].shape = [-2]; }, manifest => { manifest.models.model.variants[0].engine = 'program'; }]) {
    const source = packageSource(change); await assert.rejects(BrowserPackage.open(source.host, source.digest), {code: 'invalid_package'});
  }
});

test('browser selection rejects native device requirements it cannot establish', async () => {
  const runtime = new BrowserRuntime(new NativeHost('http://localhost', 'unused'), {stateName: 'device-requirements-' + crypto.randomUUID()});
  for (const constraint of [{minimum_android_api: 31}, {components: {'litert.google-tensor': '>=2.3,<3'}}, {firmware: ['specific-build']}, {minimum_security_patch: '2026-10-01'}]) {
    const source = packageSource(manifest => Object.assign(manifest.models.model.variants[0].target, constraint));
    const pkg = await BrowserPackage.open(source.host, source.digest, {cache: false});
    const [candidate] = runtime.compatibility(pkg, pkg.manifest.models.model);
    assert.equal(candidate.compatible, false, JSON.stringify(constraint));
    assert(candidate.reasons.length > 0);
  }
  await runtime.close();
});

test('durable browser turns reject blind replay and preserve canonical history', async () => {
  const name = 'contract-' + crypto.randomUUID(), host = new NativeHost('http://localhost', 'test');
  const runtime = new BrowserRuntime(host, {stateName: name});
  const session = await runtime.state.create('a'.repeat(64), 'assistant', [{role: 'system', content: 'Remember the conversation.'}]);
  await runtime.state.begin(session.id, 'one', {content: 'Hello'});
  await assert.rejects(runtime.state.begin(session.id, 'one', {content: 'Hello'}), {code: 'turn_in_progress'});
  await assert.rejects(runtime.state.begin(session.id, 'two', {content: 'Hello again'}), {code: 'session_busy'});
  const messages = [...session.messages, {role: 'user', content: 'Hello'}, {role: 'assistant', content: 'Hi'}];
  const final = {type: 'done', message: messages.at(-1), finish_reason: 'stop', usage: {}, execution: {location: 'browser'}};
  await runtime.state.finish(session.id, 'one', messages, final);
  assert.equal((await runtime.state.begin(session.id, 'one', {content: 'Hello'})).cached.cached, true);
  await assert.rejects(runtime.state.begin(session.id, 'one', {content: 'Different'}), {code: 'idempotency_conflict'});
  await runtime.state.begin(session.id, 'crash', {content: 'Interrupted'});
  await runtime.close();
  const restored = new BrowserRuntime(host, {stateName: name});
  assert.equal((await restored.state.session(session.id)).active_turn, 'crash');
  assert.deepEqual((await restored.state.recover(session.id)).messages, messages);
  await assert.rejects(restored.state.begin(session.id, 'crash', {content: 'Interrupted'}), {code: 'turn_failed'});
  const receipt = await restored.state.append('test', {verified: true});
  assert.equal(receipt.previous_hash, '0'.repeat(64)); assert.equal((await restored.state.receipts()).length, 1);
  await restored.close();
});

test('native transport rejects already-rounded JavaScript integer inputs', () => {
  const host = new NativeHost('http://localhost', 'unused');
  // No request is sent: the precision loss is caught at the application API.
  return import('../dist/index.js').then(({NativeApplication}) => {
    assert.throws(() => new NativeApplication(host, 'a'.repeat(64)).infer('model', {x: {dtype: 'uint64', shape: [1], data: [2 ** 64 - 1]}}), {code: 'integer_precision'});
  });
});

test('native session operations carry their application identity to the host', async () => {
  const {NativeApplication} = await import('../dist/index.js');
  const requests = [], digest = 'a'.repeat(64);
  const host = {request: async path => { requests.push(path); return {}; },
    events: async function* (path) { requests.push(path); yield {type: 'done'}; }};
  const app = new NativeApplication(host, digest);
  await app.session('session-1');
  await app.send('session-1', 'Hello');
  for await (const _event of app.sendStream('session-1', 'Hello again')) {}
  await app.recoverSession('session-1');
  assert.equal(requests.length, 4);
  for (const path of requests) assert.equal(new URL(path, 'http://localhost').searchParams.get('package'), digest);
});
