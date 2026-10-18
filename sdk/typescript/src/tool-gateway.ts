import Ajv from 'ajv/dist/2020.js';
import type {ValidateFunction} from 'ajv';
import {name, type Named} from '@gdp-ts/core';
import {PermissionPolicy, type Authorized, type AuthorizationRequest, type ResourceAccess} from './permissions.js';
import {ownedJson, jsonFingerprint} from './owned-json.js';
import {RuntimeError, type ToolDefinition} from './types.js';

export interface ToolResult extends Record<string, unknown> {
  action_id: string; tool: string; status: 'complete' | 'denied' | 'failed' | 'uncertain'; cached: boolean;
  result?: unknown; error?: {code: string; message: string};
}
export interface ToolActionStore {
  beginAction(id: string, fingerprint: string): Promise<ToolResult | undefined>;
  finishAction(id: string, result: ToolResult): Promise<void>;
}
/** For short-lived embedded hosts. Supply durable storage for restart safety. */
export class MemoryToolActions implements ToolActionStore {
  #actions = new Map<string, {fingerprint: string; result?: ToolResult}>();
  async beginAction(id: string, fingerprint: string): Promise<ToolResult | undefined> {
    const previous = this.#actions.get(id);
    if (previous) {
      if (previous.fingerprint !== fingerprint) throw new RuntimeError('idempotency_conflict', 'Action ID was used for another request');
      if (!previous.result || previous.result.status === 'uncertain') throw new RuntimeError('action_outcome_unknown', 'Inspect the previous action; automatic replay is blocked');
      return structuredClone(previous.result);
    }
    if (this.#actions.size >= 4096) throw new RuntimeError('action_capacity', 'Provide durable action storage for longer sessions');
    this.#actions.set(id, {fingerprint});
  }
  async finishAction(id: string, result: ToolResult): Promise<void> {
    const entry = this.#actions.get(id);
    if (!entry || entry.result) throw new RuntimeError('action_conflict', 'Action is not pending');
    entry.result = structuredClone(result);
  }
}
export interface ToolContext { action_id: string; package: string; subject: string; }
export interface ToolBinding {
  definition: ToolDefinition;
  /** Resolve identifiers/metadata only. Data reads belong in the checked handler. */
  access: (arguments_: Readonly<Record<string, unknown>>) => readonly ResourceAccess[];
  handler: (arguments_: Readonly<Record<string, unknown>>, context: ToolContext) => unknown | Promise<unknown>;
}
export interface ToolGatewayOptions {
  permissions: PermissionPolicy;
  subject: () => string;
  actions?: ToolActionStore;
  audit?: (event: Record<string, unknown>) => void | Promise<void>;
}

function argumentsValidator(schema: Record<string, unknown>): ValidateFunction {
  function inspect(value: unknown): void {
    if (!value || typeof value !== 'object') return;
    for (const [key, child] of Object.entries(value)) {
      if (['$ref', '$dynamicRef'].includes(key) && (typeof child !== 'string' || !child.startsWith('#'))) throw new RuntimeError('invalid_tool_schema', 'Only local schema references are permitted');
      inspect(child);
    }
  }
  inspect(schema);
  if (schema.type !== 'object') throw new RuntimeError('invalid_tool_schema', 'Arguments must be a JSON object');
  try { return new Ajv({strict: false}).compile(schema); }
  catch { throw new RuntimeError('invalid_tool_schema', 'Invalid JSON Schema'); }
}

/** Models propose calls. Only registered application handlers can execute them. */
export class ToolGateway {
  #tools = new Map<string, ToolBinding & {validate: ValidateFunction}>();
  #actions: ToolActionStore;
  constructor(readonly options: ToolGatewayOptions) { this.#actions = options.actions || new MemoryToolActions(); }
  register(binding: ToolBinding): void {
    const definition = ownedJson(binding.definition, 'invalid_tool_schema');
    if (this.#tools.has(definition.name)) throw new RuntimeError('tool_exists', 'Tool is already registered');
    this.#tools.set(definition.name, {...binding, definition, validate: argumentsValidator(definition.parameters)});
  }
  async invoke(pkg: {digest: string; tools: ToolDefinition[]}, tool: string, arguments_: Record<string, unknown>, actionId: string): Promise<ToolResult> {
    if (typeof actionId !== 'string' || !actionId.length || actionId.length > 256) throw new RuntimeError('invalid_action_id', 'Provide an action ID of 1 to 256 characters');
    const args = ownedJson(arguments_), subject = this.options.subject();
    if (!args || Array.isArray(args) || typeof args !== 'object') throw new RuntimeError('invalid_tool_arguments', 'Arguments must be an object');
    const requestId = jsonFingerprint({package: pkg.digest, subject, tool, arguments: args});
    const cached = await this.#actions.beginAction(actionId, requestId);
    let entered = false, binding: (ToolBinding & {validate: ValidateFunction}) | undefined, result: ToolResult;
    try {
      const declared = pkg.tools.find(item => item.name === tool);
      if (!declared) throw new RuntimeError('tool_not_declared', 'Application does not declare this tool');
      binding = this.#tools.get(tool);
      if (!binding) throw new RuntimeError('tool_unavailable', 'Application has no implementation of this tool');
      if (declared.permission !== binding.definition.permission || !!declared.side_effect !== !!binding.definition.side_effect) throw new RuntimeError('tool_contract', 'Package cannot weaken the host tool contract');
      if (!binding.validate(args) || !argumentsValidator(declared.parameters)(args)) throw new RuntimeError('invalid_tool_arguments', 'Arguments do not satisfy the tool schema');
      const request = ownedJson({operation: pkg.digest + ':' + tool, arguments: args, access: binding.access(args)});
      const registered = binding;
      result = await name(subject, request, async (user, call) => {
        const proof = this.options.permissions.authorize(user, call);
        await this.options.audit?.({kind: 'tool_authorized', action_id: actionId, tool, subject, package: pkg.digest,
          policy_revision: this.options.permissions.revision, access: request.access, arguments_sha256: jsonFingerprint(args)});
        return this.#execute(user, call, proof, registered, pkg.digest, actionId, cached, () => { entered = true; });
      });
    } catch (error) {
      const fault = error instanceof RuntimeError ? error : new RuntimeError('tool_failed', 'Tool handler failed');
      result = {action_id: actionId, tool, status: entered ? binding?.definition.side_effect ? 'uncertain' : 'failed' : 'denied', cached: false,
        error: {code: fault.code, message: fault.message}};
    }
    if (!cached) await this.#actions.finishAction(actionId, result);
    await this.options.audit?.({kind: 'tool_result', action_id: actionId, tool, subject, status: result.status, cached: result.cached,
      ...(result.error ? {error: result.error.code} : {})});
    return result;
  }
  async #execute<S, R>(subject: Named<S, string>, request: Named<R, AuthorizationRequest>, proof: Authorized<S, R>, binding: ToolBinding,
                      packageDigest: string, actionId: string, cached: ToolResult | undefined, entered: () => void): Promise<ToolResult> {
    if (subject.value !== this.options.subject()) throw new RuntimeError('permission_denied', 'The signed-in user changed before tool execution');
    if (request.value.operation !== packageDigest + ':' + binding.definition.name) throw new RuntimeError('invalid_authorization', 'Authorization is for a different tool');
    this.options.permissions.consume(subject, request, proof);
    if (cached) return {...cached, cached: true};
    entered();
    const value = await binding.handler(request.value.arguments, {action_id: actionId, package: packageDigest, subject: subject.value});
    return {action_id: actionId, tool: binding.definition.name, status: 'complete', cached: false, result: ownedJson(value, 'invalid_tool_output')};
  }
}
