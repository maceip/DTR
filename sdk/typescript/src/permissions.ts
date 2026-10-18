import Ajv from 'ajv/dist/2020.js';
import {createPermix} from 'permix';
import {defineProof, name, type Named, type Proof} from '@gdp-ts/core';
import schema from '../../contracts/permissions.schema.json';
import {ownedJson, jsonFingerprint} from './owned-json.js';
import {RuntimeError} from './types.js';

export type ResourceAction = 'read' | 'write' | 'execute' | 'export' | 'train';
export interface ResourceAccess { readonly resource: string; readonly action: ResourceAction; }
export interface PermissionRule extends ResourceAccess { readonly subject: string; readonly effect: 'allow' | 'deny'; }
export interface PermissionDocument { readonly version: 1; readonly rules: readonly PermissionRule[]; }
export interface AuthorizationRequest {
  readonly operation: string;
  readonly arguments: Record<string, unknown>;
  readonly access: readonly ResourceAccess[];
}
export interface AuthorizationReceipt { subject: string; policy_revision: number; access: readonly ResourceAccess[]; }
interface PermissionCheck extends ResourceAccess { subject: string; }

// This prover never leaves the module. gdp-ts is a compiler guardrail; the
// instance-owned WeakMap below supplies the actual runtime validity checks.
const approved = defineProof('DtrAuthorized');
export interface Authorized<S, R> extends Proof<'DtrAuthorized', [S, R]> {}
const validate = new Ajv({strict: true}).compile<PermissionDocument>(schema);
const identity = /^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$/;

/** App-owned policy, not something a model/catalog/tool call can install. */
export class PermissionPolicy {
  #revision = 0;
  #engine = createPermix<{access: [{name: 'use'; type: PermissionCheck; required: true}]} >();
  #proofs = new WeakMap<object, {subject: object; request: object; binding: string; revision: number}>();
  constructor(document: PermissionDocument) { this.replace(document); }
  get revision(): number { return this.#revision; }
  replace(document: PermissionDocument): void {
    const copy = ownedJson(document, 'invalid_policy');
    if (!validate(copy)) throw new RuntimeError('invalid_policy', 'Invalid permission document', validate.errors);
    this.#engine.setup({access: {use: request => {
      const matches = copy.rules.filter(rule => rule.subject === request.subject && rule.resource === request.resource && rule.action === request.action);
      return matches.some(rule => rule.effect === 'allow') && !matches.some(rule => rule.effect === 'deny');
    }}});
    this.#revision++;
  }
  allows(subject: string, access: ResourceAccess): boolean {
    if (!identity.test(subject) || typeof access.resource !== 'string' || !access.resource.length || [...access.resource].length > 1024 || /[\x00-\x1f\x7f*?]/.test(access.resource) || !['read', 'write', 'execute', 'export', 'train'].includes(access.action)) return false;
    return this.#engine.check('access.use', {subject, resource: access.resource, action: access.action});
  }
  authorize<S, R>(subject: Named<S, string>, request: Named<R, AuthorizationRequest>): Authorized<S, R> {
    const value = ownedJson(request.value);
    if (!Array.isArray(value.access) || !value.access.length || value.access.length > 128) throw new RuntimeError('tool_scope_missing', 'Declare 1 to 128 resource operations');
    for (const access of value.access) if (!this.allows(subject.value, access)) throw new RuntimeError('permission_denied', 'Resource operation is not permitted', {subject: subject.value, ...access, policy_revision: this.#revision});
    // Upstream prove() returns one shared object. Each DTR authorization needs
    // a unique handle so concurrent requests cannot replace each other's proof.
    const proof = Object.freeze({...approved.prove(subject, request)});
    this.#proofs.set(proof, {subject, request, binding: jsonFingerprint(value), revision: this.#revision});
    return proof;
  }
  consume<S, R>(subject: Named<S, string>, request: Named<R, AuthorizationRequest>, proof: Authorized<S, R>): AuthorizationReceipt {
    const known = this.#proofs.get(proof);
    this.#proofs.delete(proof);
    if (!known || known.subject !== subject || known.request !== request || known.revision !== this.#revision || known.binding !== jsonFingerprint(ownedJson(request.value))) throw new RuntimeError('invalid_authorization', 'Authorization is forged, changed, reused, or revoked');
    return {subject: subject.value, policy_revision: known.revision, access: request.value.access};
  }
  run<T>(subject: string, request: AuthorizationRequest, operation: (receipt: AuthorizationReceipt) => T): T {
    return name(subject, ownedJson(request), (user, call) => operation(this.consume(user, call, this.authorize(user, call))));
  }
}
