// Scripted tool proposals demonstrate the boundary without downloading a model.
import assert from 'node:assert/strict';
import {PermissionPolicy, ToolGateway} from '../../sdk/typescript/dist/index.js';

const permissions = new PermissionPolicy({version: 1, rules: [
  {subject: 'alice', resource: 'notes:alice', action: 'read', effect: 'allow'},
  {subject: 'alice', resource: 'code:patch', action: 'write', effect: 'allow'},
]});
const notes = new Map([['alice', 'Refactor tests; leave database migrations alone.'], ['bob', 'Another user’s private notes.']]);
const generated = new Map();
const gateway = new ToolGateway({permissions, subject: () => 'alice'});
const schema = {type: 'object', properties: {id: {type: 'string'}, text: {type: 'string'}}, required: ['id'], additionalProperties: false};
const definitions = [
  {name: 'read_note', description: 'Read a user note', parameters: schema, permission: 'notes.read'},
  {name: 'write_patch', description: 'Save generated code as text', parameters: schema, permission: 'code.write', side_effect: true},
  {name: 'run_patch', description: 'Execute generated code', parameters: schema, permission: 'code.execute', side_effect: true},
  {name: 'upload_note', description: 'Export a note', parameters: schema, permission: 'notes.export', side_effect: true},
];
for (const definition of definitions) {
  const [kind, action] = definition.permission.split('.');
  gateway.register({definition, access: args => [{resource: kind + ':' + args.id, action}], handler: args => {
    if (action === 'read') return {text: notes.get(args.id)};
    if (action === 'write') { generated.set(args.id, args.text); return {saved: true}; }
    throw Error('A denied execute/export handler must never be entered');
  }});
}
const pkg = {digest: 'permission-demo-v1', tools: definitions};
const proposals = [
  ['read_note', {id: 'alice'}],
  ['read_note', {id: 'bob'}],
  ['write_patch', {id: 'patch', text: 'print("generated code is data until execution is authorized")'}],
  ['run_patch', {id: 'patch'}],
  ['upload_note', {id: 'alice'}],
];
const results = [];
for (const [index, [tool, args]] of proposals.entries()) results.push(await gateway.invoke(pkg, tool, args, 'demo-' + index));
assert.deepEqual(results.map(result => result.status), ['complete', 'denied', 'complete', 'denied', 'denied']);
assert.equal(generated.size, 1);
permissions.replace({version: 1, rules: []});
const revoked = await gateway.invoke(pkg, 'read_note', {id: 'alice'}, 'demo-0');
assert.equal(revoked.status, 'denied'); assert.equal(revoked.result, undefined);
console.log('DTR tool boundary · scripted proposals · no model inference or training');
console.table(results.map(({tool, status, error}) => ({tool, status, reason: error?.code || 'allowed'})));
console.log('Revoked permission blocks the cached note:', revoked.status);
