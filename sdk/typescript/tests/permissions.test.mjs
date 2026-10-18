import 'fake-indexeddb/auto';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {name} from '@gdp-ts/core';
import {PermissionPolicy, ToolGateway, BrowserRuntime, NativeHost} from '../dist/index.js';

const fixture = JSON.parse(await readFile(new URL('../../contracts/permissions-cases.json', import.meta.url)));
const definition = {name: 'read_note', description: 'Read an app-owned note', permission: 'notes.read',
  parameters: {type: 'object', properties: {id: {type: 'string'}}, required: ['id'], additionalProperties: false}};
const pkg = {digest: 'a'.repeat(64), tools: [definition]};
const request = {operation: 'read_note', arguments: {id: 'alice'}, access: [{resource: 'notes:alice', action: 'read'}]};
const allow = () => new PermissionPolicy(fixture.policy);
function setup(options = {}) {
  const policy = options.policy || allow(), effects = [], receipts = [];
  const gateway = new ToolGateway({permissions: policy, subject: () => 'alice', audit: event => receipts.push(event), ...options});
  gateway.register({definition, access: args => [{resource: 'notes:' + args.id, action: 'read'}], handler: args => { effects.push(args.id); return {text: 'private note'}; }});
  return {policy, effects, gateway, receipts};
}

test('Permix implements the shared Python/Android permission decisions and rejects invalid policies', () => {
  const policy = allow();
  for (const {name, subject, resource, action, allowed} of fixture.cases) assert.equal(policy.allows(subject, {resource, action}), allowed, name);
  for (const document of fixture.invalid_documents) assert.throws(() => new PermissionPolicy(document), {code: 'invalid_policy'});
});

test('runtime rejects forged, cross-request, reused, cross-policy and revoked GDP authorizations', () => {
  const policy = allow();
  name('alice', structuredClone(request), (subject, call) => {
    assert.throws(() => policy.consume(subject, call, {kind: 'DtrAuthorized'}), {code: 'invalid_authorization'});
    const proof = policy.authorize(subject, call);
    name(structuredClone(request), other => assert.throws(() => policy.consume(subject, other, proof), {code: 'invalid_authorization'}));
    const once = policy.authorize(subject, call);
    policy.consume(subject, call, once);
    assert.throws(() => policy.consume(subject, call, once), {code: 'invalid_authorization'});
    const foreign = policy.authorize(subject, call);
    assert.throws(() => allow().consume(subject, call, foreign), {code: 'invalid_authorization'});
    const revoked = policy.authorize(subject, call);
    policy.replace({version: 1, rules: []});
    assert.throws(() => policy.consume(subject, call, revoked), {code: 'invalid_authorization'});
  });
});

test('changed arguments cannot reuse a proof; distinct concurrent authorizations stay distinct', () => {
  const policy = allow();
  name('alice', structuredClone(request), (subject, call) => {
    const proof = policy.authorize(subject, call);
    call.value.arguments.id = 'bob';
    assert.throws(() => policy.consume(subject, call, proof), {code: 'invalid_authorization'});
  });
  name('alice', request, (subject, call) => {
    const first = policy.authorize(subject, call), second = policy.authorize(subject, call);
    assert.notEqual(first, second);
    policy.consume(subject, call, first); policy.consume(subject, call, second);
  });
});

test('authorization snapshots reject values with ambiguous JSON encodings', () => {
  const policy = allow();
  for (const value of [NaN, Infinity, undefined, new Array(1), 1n]) {
    name('alice', {...request, arguments: {value}}, (subject, call) => {
      assert.throws(() => policy.authorize(subject, call), {code: 'invalid_tool_arguments'});
    });
  }
});

test('gateway validates model proposals and checks authorization before cached results', async () => {
  const {gateway, effects, policy, receipts} = setup();
  assert.equal((await gateway.invoke(pkg, 'read_note', {id: 'alice'}, 'read')).status, 'complete');
  assert.equal((await gateway.invoke(pkg, 'read_note', {id: 'alice'}, 'read')).cached, true);
  assert.equal((await gateway.invoke(pkg, 'read_note', {id: 'bob'}, 'other')).error.code, 'permission_denied');
  assert.equal((await gateway.invoke(pkg, 'read_note', {id: 'alice', subject: 'admin'}, 'spoof')).error.code, 'invalid_tool_arguments');
  assert.equal((await gateway.invoke(pkg, 'read_note', {id: 42}, 'type')).error.code, 'invalid_tool_arguments');
  await assert.rejects(gateway.invoke(pkg, 'read_note', {id: 'bob'}, 'read'), {code: 'idempotency_conflict'});
  policy.replace({version: 1, rules: []});
  const denied = await gateway.invoke(pkg, 'read_note', {id: 'alice'}, 'read');
  assert.equal(denied.status, 'denied'); assert.equal(denied.result, undefined);
  assert.deepEqual(effects, ['alice']);
  assert(!JSON.stringify(receipts).includes('private note'));
});

test('logout or policy revocation during async work prevents handler entry', async () => {
  let user = 'alice';
  for (const revoke of [policy => { user = 'bob'; }, policy => policy.replace({version: 1, rules: []})]) {
    user = 'alice'; const policy = allow();
    const {gateway, effects} = setup({policy, subject: () => user, audit: event => { if (event.kind === 'tool_authorized') revoke(policy); }});
    assert.equal((await gateway.invoke(pkg, 'read_note', {id: 'alice'}, 'late')).status, 'denied');
    assert.deepEqual(effects, []);
  }
});

test('package cannot weaken host declarations or fetch remote argument schemas', async () => {
  const {gateway, effects} = setup();
  const weakened = {digest: pkg.digest, tools: [{...definition, permission: 'unrestricted'}]};
  assert.equal((await gateway.invoke(weakened, 'read_note', {id: 'alice'}, 'weaken')).error.code, 'tool_contract');
  assert.throws(() => gateway.register({definition: {...definition, name: 'remote', parameters: {type: 'object', $ref: 'https://example.invalid'}}, access: () => request.access, handler: () => ({})}), {code: 'invalid_tool_schema'});
  assert.deepEqual(effects, []);
});

test('browser gateway uses durable action storage across runtime restarts', async () => {
  globalThis.isSecureContext = true;
  const stateName = 'tools-' + crypto.randomUUID(), effects = [];
  async function open() {
    const runtime = new BrowserRuntime(new NativeHost('http://unused.invalid', 'unused'), {stateName, permissions: allow(), subject: () => 'alice'});
    runtime.tools.register({definition: {...definition, side_effect: true}, access: () => request.access, handler: () => { effects.push(1); return {ok: true}; }});
    return runtime;
  }
  const declared = {...pkg, tools: [{...definition, side_effect: true}]};
  let runtime = await open();
  assert.equal((await runtime.tools.invoke(declared, 'read_note', {id: 'alice'}, 'durable')).status, 'complete');
  await runtime.close();
  runtime = await open();
  assert.equal((await runtime.tools.invoke(declared, 'read_note', {id: 'alice'}, 'durable')).cached, true);
  assert.deepEqual(effects, [1]);
  await runtime.state.beginAction('interrupted', 'interrupted-fingerprint');
  await assert.rejects(runtime.state.beginAction('interrupted', 'interrupted-fingerprint'), {code: 'action_outcome_unknown'});
  await runtime.close();
});

test('failed side effects are uncertain and do not execute twice', async () => {
  let effects = 0;
  const gateway = new ToolGateway({permissions: allow(), subject: () => 'alice'});
  const effect = {...definition, side_effect: true};
  gateway.register({definition: effect, access: () => request.access, handler: () => { effects++; throw Error('after execution'); }});
  const declared = {...pkg, tools: [effect]};
  assert.equal((await gateway.invoke(declared, 'read_note', {id: 'alice'}, 'effect')).status, 'uncertain');
  await assert.rejects(gateway.invoke(declared, 'read_note', {id: 'alice'}, 'effect'), {code: 'action_outcome_unknown'});
  assert.equal(effects, 1);
});

test('adding durable tool storage preserves conversations from state format v1', async () => {
  globalThis.isSecureContext = true;
  const stateName = 'migration-' + crypto.randomUUID();
  const previous = await new Promise((resolve, reject) => {
    const open = indexedDB.open(stateName, 1);
    open.onupgradeneeded = () => {
      for (const store of ['sessions', 'turns', 'receipts', 'meta']) open.result.createObjectStore(store, {keyPath: 'id'});
      open.transaction.objectStore('sessions').add({id: 'kept', package: pkg.digest, model: 'assistant', messages: [{role: 'user', content: 'Remember me'}]});
    };
    open.onsuccess = () => resolve(open.result); open.onerror = () => reject(open.error);
  });
  previous.close();
  const runtime = new BrowserRuntime(new NativeHost('http://unused.invalid', 'unused'), {stateName});
  assert.equal((await runtime.state.session('kept')).messages[0].content, 'Remember me');
  assert.equal(await runtime.state.beginAction('new', 'new-fingerprint'), undefined);
  await runtime.close();
});

test('fine-tuning contract checks consent but never invents a completed training job', async () => {
  const {BrowserApplication} = await import('../dist/index.js');
  globalThis.isSecureContext = true;
  const runtime = new BrowserRuntime(new NativeHost('http://unused.invalid', 'unused'), {stateName: 'training-' + crypto.randomUUID(), permissions: allow(), subject: () => 'alice'});
  const app = new BrowserApplication(runtime, {digest: pkg.digest, manifest: {models: {assistant: {task: 'chat', variants: []}}}});
  assert.equal(app.capabilities('assistant').fine_tuning.available, false);
  await assert.rejects(app.fineTune({model: 'assistant', dataset: 'notes:alice'}), {code: 'permission_denied'});
  await assert.rejects(app.fineTune({model: 'assistant', dataset: 'dataset:consented'}), {code: 'fine_tuning_unavailable'});
  await runtime.close();
});
