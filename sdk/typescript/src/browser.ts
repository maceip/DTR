import semver from 'semver';
import {BrowserPackage, PackageOptions} from './browser-package.js';
import {BrowserState, fingerprint} from './browser-state.js';
import {NativeApplication, NativeHost, wireOptions} from './native-host.js';
import {WorkerBridge} from './worker-bridge.js';
import {generationOptions, prepareMessages} from './chat-common.js';
import {bytesFromBase64, decodeTensor, encodeTensor} from './tensors.js';
import {PermissionPolicy} from './permissions.js';
import {ToolGateway} from './tool-gateway.js';
import {fineTuneStub, modelCapabilities} from './training.js';
import type {FineTuneRequest, ModelCapabilities} from './types.js';
import {Application, ChatOptions, ChatResult, ClassificationResult, Execution, ImageInput, InferOptions, InferenceResult, Message, Model, PackageInfo, RuntimeError, RuntimeEvent, Selection, SendOptions, Session, TensorValue, Variant} from './types.js';

export interface BrowserOptions extends PackageOptions { workerUrl?: URL; wasmRoot?: string; maxModels?: number; stateName?: string; permissions?: PermissionPolicy; subject?: () => string; }
type Entry = {bridge: WorkerBridge; execution?: Execution; busy: boolean; used: number};
type Compatibility = {id: string; backend: string; compatible: boolean; reasons: string[]; runtime_version: string; execution_verified: boolean; last_execution?: Execution};

function budget(seconds: number, signal?: AbortSignal) {
  if (!Number.isFinite(seconds) || seconds <= 0 || seconds > 3600) throw new RuntimeError('invalid_timeout', 'Timeout must be between 0 and 3600 seconds');
  const deadline = performance.now() + seconds * 1000, timer = AbortSignal.timeout(Math.max(1, Math.ceil(seconds * 1000)));
  return {signal: signal ? AbortSignal.any([signal, timer]) : timer,
    remaining() { const remaining = (deadline - performance.now()) / 1000; if (remaining <= 0) throw new RuntimeError('inference_timeout', 'Request exhausted its time budget', undefined, true); return remaining; },
    error(error: unknown) { if (timer.aborted && !signal?.aborted) return new RuntimeError('inference_timeout', 'Request exhausted its time budget', undefined, true); return error; }};
}

export class BrowserRuntime {
  readonly state: BrowserState;
  readonly tools?: ToolGateway;
  private pool = new Map<string, Entry>();
  private observed = new Map<string, Execution>();
  constructor(readonly host: NativeHost, readonly options: BrowserOptions = {}) {
    if (!isSecureContext) throw new RuntimeError('secure_context_required', 'Use HTTPS or localhost for browser inference');
    if (!Number.isInteger(options.maxModels ?? 2) || (options.maxModels ?? 2) < 1) throw new RuntimeError('invalid_capacity', 'maxModels must be a positive integer');
    this.state = new BrowserState(options.stateName);
    if (options.permissions) {
      if (!options.subject) throw new RuntimeError('invalid_policy', 'Provide the app-authenticated subject callback');
      this.tools = new ToolGateway({permissions: options.permissions, subject: options.subject, actions: this.state,
        audit: async event => { await this.state.append('tool_permission', event); }});
    }
  }
  async application(digest: string): Promise<BrowserApplication> { return new BrowserApplication(this, await BrowserPackage.open(this.host, digest, this.options)); }
  device(): Record<string, unknown> { return {system: 'web', architecture: 'wasm32', location: 'browser', runtimes: {'@litertjs/core': '2.5.3', '@litert-lm/core': '0.18.1'}, webgpu_available: 'gpu' in navigator, webnn_available: 'ml' in navigator, npu_execution_verified: false}; }
  compatibility(pkg: BrowserPackage, model: Model): Compatibility[] {
    return model.variants.map(variant => {
      const reasons: string[] = [], target = variant.target, version = variant.engine === 'llm' ? '0.18.1' : '2.5.3';
      if (variant.backend !== 'litert') reasons.push('ExecuTorch requires an explicitly paired native host');
      if (!target.systems?.includes('web')) reasons.push('Artifact has no declared browser target');
      if (target.architectures?.length && !target.architectures.includes('wasm32')) reasons.push('Artifact does not target wasm32');
      if (target.soc) reasons.push('Browser cannot establish the required SoC identity');
      if (target.minimum_android_api) reasons.push('Android API requirements need an embedded Android host');
      if (target.firmware?.length || target.minimum_security_patch) reasons.push('Browser cannot establish firmware or security patch requirements');
      if (Object.keys(target.components || {}).length) reasons.push('Required native components cannot be established in the browser');
      if (target.runtime_version && !semver.satisfies(version, target.runtime_version.replace(/==/g, '=').replace(/,/g, ' '))) reasons.push(`Requires runtime ${target.runtime_version}; browser has ${version}`);
      const memory = (navigator as Navigator & {deviceMemory?: number}).deviceMemory;
      if (target.minimum_memory_mb && (!memory || memory * 1024 < target.minimum_memory_mb)) reasons.push('Browser cannot establish sufficient declared device memory');
      if (target.accelerator === 'gpu' && !('gpu' in navigator)) reasons.push('WebGPU is unavailable');
      if (target.accelerator === 'npu' && (variant.engine === 'llm' || !('ml' in navigator))) reasons.push('No compatible browser NPU API');
      if (variant.data_artifact || variant.supporting_artifacts?.length) reasons.push('External native model resources require a native host');
      if (variant.engine !== 'llm') for (const signature of Object.values(variant.signatures)) for (const spec of [...signature.inputs, ...signature.outputs]) {
        if (!['float32', 'int32', 'uint8'].includes(spec.dtype)) reasons.push(`LiteRT.js does not expose ${spec.dtype} I/O`);
        if (spec.shape.includes(-1)) reasons.push('Dynamic tensor resizing requires a native host with this LiteRT.js version');
      }
      const last = this.observed.get(this.key(pkg, variant));
      return {id: variant.id, backend: variant.backend, compatible: !reasons.length, reasons: [...new Set(reasons)], runtime_version: version, execution_verified: !!last, ...(last ? {last_execution: last} : {})};
    });
  }
  private key(pkg: BrowserPackage, variant: Variant): string { return pkg.digest + ':' + fingerprint(variant); }
  async acquire(pkg: BrowserPackage, modelName: string, options: Selection, requireTools = false): Promise<{entry: Entry; release: () => void; variant: Variant}> {
    const model = pkg.manifest.models[modelName];
    if (!model) throw new RuntimeError('unknown_model', `Unknown logical model ${modelName}`);
    if (options.backend === 'executorch') throw new RuntimeError('native_host_required', 'Select the paired native host to execute ExecuTorch');
    const compatibility = this.compatibility(pkg, model), fallback: unknown[] = [];
    const candidates = model.variants.filter((variant, index) => compatibility[index].compatible && (!requireTools || (variant.chat && variant.chat.tool_protocol !== 'none')) && (!options.variant || variant.id === options.variant) && (!options.backend || variant.backend === options.backend) && (!options.accelerator || (variant.target.accelerator || 'cpu') === options.accelerator))
      .sort((a, b) => Number((b.target.accelerator || 'cpu') === options.accelerator) - Number((a.target.accelerator || 'cpu') === options.accelerator) || (b.priority || 0) - (a.priority || 0));
    if (!candidates.length) throw new RuntimeError('no_compatible_variant', 'No browser-compatible variant matches this request; choose a native host or package a web artifact', compatibility);
    for (const variant of candidates) {
      const key = this.key(pkg, variant);
      let entry = this.pool.get(key);
      if (entry?.bridge.closed) { this.pool.delete(key); entry = undefined; }
      if (entry?.busy) throw new RuntimeError('model_busy', 'Model is serving another request', undefined, true);
      if (!entry) {
        if (this.pool.size >= (this.options.maxModels ?? 2)) {
          const idle = [...this.pool.entries()].filter(([, value]) => !value.busy).sort(([, a], [, b]) => a.used - b.used)[0];
          if (!idle) throw new RuntimeError('capacity_busy', 'All model workers are in use', undefined, true);
          idle[1].bridge.close(); this.pool.delete(idle[0]);
        }
        entry = {bridge: new WorkerBridge(this.options.workerUrl || new URL('./browser-worker.js', import.meta.url)), busy: true, used: performance.now()};
        this.pool.set(key, entry);
      } else entry.busy = true;
      const held = entry;
      try {
        if (!held.execution) {
          const artifact = await pkg.artifact(variant.artifact, options.signal);
          held.execution = await held.bridge.request<Execution>('load', {variant, model: artifact,
            wasmRoot: this.options.wasmRoot || new URL('./wasm', import.meta.url).href.replace(/\/$/, ''),
            execution: {backend: 'litert', engine: variant.engine, variant: variant.id, package: pkg.digest, model: modelName, location: 'browser', requested_accelerator: variant.target.accelerator || 'cpu', npu_execution_verified: false}}, 180, options.signal);
        }
        held.execution = {...held.execution, fallback: [...fallback], ...(options.accelerator && options.accelerator !== (variant.target.accelerator || 'cpu') ? {fallback_from_accelerator: options.accelerator} : {})};
        return {entry: held, variant, release: () => { held.busy = false; held.used = performance.now(); }};
      } catch (error) {
        held.bridge.close(); this.pool.delete(key);
        if (!options.allow_fallback || options.signal?.aborted) throw error;
        fallback.push({variant: variant.id, error: error instanceof RuntimeError ? {code: error.code, message: error.message} : String(error)});
      }
    }
    throw new RuntimeError('model_load_failed', 'All compatible variants failed to load', fallback);
  }
  async record(pkg: BrowserPackage, variant: Variant, kind: string, result: {execution: Execution}): Promise<Record<string, unknown>> {
    this.observed.set(this.key(pkg, variant), result.execution);
    return this.state.append(kind, {package: pkg.digest, execution: result.execution});
  }
  async close(): Promise<void> { for (const entry of this.pool.values()) entry.bridge.close(); this.pool.clear(); await this.state.close(); }
}

export class BrowserApplication implements Application {
  readonly digest: string;
  private native: NativeApplication;
  constructor(readonly runtime: BrowserRuntime, readonly package_: BrowserPackage) { this.digest = package_.digest; this.native = new NativeApplication(runtime.host, this.digest); }
  describe(): PackageInfo {
    const manifest = this.package_.manifest;
    return {name: manifest.name, version: manifest.version, digest: this.digest, models: manifest.models, tools: manifest.tools || [], permissions: manifest.permissions || [], agent: manifest.agent, signature: this.package_.signature,
      compatibility: Object.fromEntries(Object.entries(manifest.models).map(([name, model]) => [name, this.runtime.compatibility(this.package_, model)]))};
  }
  capabilities(model: string): ModelCapabilities {
    const definition = this.package_.manifest.models[model];
    if (!definition) throw new RuntimeError('unknown_model', 'Unknown logical model');
    return modelCapabilities(model, definition);
  }
  async fineTune(request: FineTuneRequest): Promise<never> {
    const definition = this.package_.manifest.models[request.model];
    if (!definition) throw new RuntimeError('unknown_model', 'Unknown logical model');
    return fineTuneStub(this.runtime.options.permissions, this.runtime.options.subject?.() || '', request, definition);
  }
  async infer(model: string, inputs: Record<string, TensorValue>, options: InferOptions = {}): Promise<InferenceResult> {
    const limit = budget(options.timeout_seconds ?? 60, options.signal);
    let reservation: Awaited<ReturnType<BrowserRuntime['acquire']>> | undefined;
    try {
      reservation = await this.runtime.acquire(this.package_, model, {...options, signal: limit.signal});
      const {entry, variant} = reservation;
      const result = await entry.bridge.request<InferenceResult>('infer', {inputs, options: wireOptions(options)}, limit.remaining(), limit.signal);
      result.execution = {...entry.execution!, ...result.execution, fallback: entry.execution!.fallback};
      result.receipt = await this.runtime.record(this.package_, variant, 'inference', result);
      return result;
    } catch (error) { throw limit.error(error); }
    finally { reservation?.release(); }
  }
  async *chatStream(model: string, messages: Message[], options: ChatOptions = {}): AsyncGenerator<RuntimeEvent> {
    const generation = generationOptions(options.generation);
    const limit = budget(generation.timeout_seconds, options.signal);
    let reservation: Awaited<ReturnType<BrowserRuntime['acquire']>> | undefined;
    try {
      reservation = await this.runtime.acquire(this.package_, model, {...options, signal: limit.signal}, !!options.tools?.length);
      const {entry, variant} = reservation;
      for await (const event of entry.bridge.events('chat', {messages, options: wireOptions(options)}, limit.remaining(), limit.signal)) {
        if ('execution' in event && event.execution) event.execution = {...entry.execution!, ...event.execution as Execution, fallback: entry.execution!.fallback};
        if (event.type === 'done') await this.runtime.record(this.package_, variant, 'generation', event);
        yield event;
      }
    } catch (error) { throw limit.error(error); }
    finally { reservation?.release(); }
  }
  async chat(model: string, messages: Message[], options: ChatOptions = {}): Promise<ChatResult> {
    for await (const event of this.chatStream(model, messages, options)) if (event.type === 'done') return event;
    throw new RuntimeError('incomplete_generation', 'Model did not complete generation');
  }
  async createSession(model: string, messages: Message[] = []): Promise<Session> {
    const definition = this.package_.manifest.models[model];
    if (!definition || definition.task !== 'chat') throw new RuntimeError('wrong_model_task', 'Choose a chat model');
    if (messages.length) prepareMessages(messages, {context_length: 1, modalities: ['text', 'image', 'audio'], tool_protocol: 'none'});
    return this.runtime.state.create(this.digest, model, messages);
  }
  async session(id: string): Promise<Session> { const session = await this.runtime.state.session(id); if (session.package !== this.digest) throw new RuntimeError('session_package', 'Session belongs to a different package'); return session; }
  invokeTool(name: string, arguments_: Record<string, unknown>, actionId: string): Promise<Record<string, unknown>> {
    if (this.runtime.tools) return this.runtime.tools.invoke({digest: this.digest, tools: this.package_.manifest.tools || []}, name, arguments_, actionId);
    return this.native.invokeTool(name, arguments_, actionId);
  }
  async recoverSession(id: string): Promise<Session> {
    await this.session(id);
    if (!navigator.locks) throw new RuntimeError('web_locks_required', 'Durable browser sessions require Web Locks');
    return navigator.locks.request('phaseone-session-' + id, {ifAvailable: true}, async lock => {
      if (!lock) throw new RuntimeError('session_busy', 'A page is still processing this session');
      return this.runtime.state.recover(id);
    });
  }
  async *sendStream(id: string, content: Message['content'], options: SendOptions = {}): AsyncGenerator<RuntimeEvent> {
    if (!navigator.locks) throw new RuntimeError('web_locks_required', 'Durable browser sessions require Web Locks');
    let unlock!: () => void, acquired!: (value: boolean) => void;
    const ready = new Promise<boolean>(resolve => { acquired = resolve; });
    const held = navigator.locks.request('phaseone-session-' + id, {ifAvailable: true}, lock => { acquired(!!lock); return lock ? new Promise<void>(resolve => { unlock = resolve; }) : undefined; });
    if (!await ready) { await held; throw new RuntimeError('session_busy', 'Another page is processing this session'); }
    try { yield* this.turn(id, content, options); } finally { unlock(); await held; }
  }
  private async *turn(id: string, content: Message['content'], options: SendOptions): AsyncGenerator<RuntimeEvent> {
    const session = await this.session(id), definition = options.agent ? this.package_.manifest.agent : undefined;
    if (options.agent && (!definition || definition.model !== session.model)) throw new RuntimeError('agent_not_configured', 'This session has no matching packaged agent');
    if (options.agent && options.tools?.length) throw new RuntimeError('agent_tools', 'Packaged agents use their declared tools');
    const generation = generationOptions(options.generation), turnId = options.turn_id || crypto.randomUUID();
    const begun = await this.runtime.state.begin(id, turnId, {content, options: {...wireOptions(options), turn_id: turnId}});
    if (begun.cached) { yield begun.cached; return; }
    const messages = structuredClone(begun.session.messages), actions: unknown[] = [];
    let committed = false;
    const deadline = performance.now() + generation.timeout_seconds * 1000;
    try {
      if (definition) {
        if (messages[0]?.role !== 'system') messages.unshift({role: 'system', content: definition.instruction});
        else if (!String(messages[0].content).startsWith(definition.instruction)) messages[0].content = definition.instruction + '\n' + String(messages[0].content);
      }
      messages.push({role: 'user', content});
      yield {type: 'turn', session_id: id, turn_id: turnId};
      const tools = definition ? (this.package_.manifest.tools || []).filter(tool => definition.tools.includes(tool.name)).map(({name, description, parameters}) => ({name, description, parameters})) : options.tools;
      for (let step = 0; step < (definition?.max_steps || 1); step++) {
        const remaining = (deadline - performance.now()) / 1000;
        if (remaining <= 0) throw new RuntimeError('inference_timeout', 'Agent turn exceeded its time budget');
        let result: ChatResult | undefined;
        for await (const event of this.chatStream(session.model, messages, {...options, tools, generation: {...generation, timeout_seconds: remaining}})) {
          if (event.type === 'done') result = event; else yield event;
        }
        if (!result) throw new RuntimeError('incomplete_generation', 'Model did not complete its response');
        messages.push(result.message);
        if (!definition || !result.message.tool_calls?.length) {
          const final = {...result, session_id: id, turn_id: turnId, actions, cached: false};
          await this.runtime.state.finish(id, turnId, messages, final); committed = true; yield final; return;
        }
        for (const [index, call] of result.message.tool_calls.entries()) {
          if (options.signal?.aborted) throw new RuntimeError('cancelled', 'Agent turn was cancelled');
          if (!definition.tools.includes(call.name)) throw new RuntimeError('tool_not_declared', 'Agent requested a tool outside its declared list');
          const actionId = `${turnId}:${step}:${index}`;
          yield {type: 'tool_call', step, action_id: actionId, call};
          // Choose one application-owned gateway; never fall back on a denial.
          const action = this.runtime.tools ? await this.invokeTool(call.name, call.arguments, actionId) : await this.native.host.request<Record<string, unknown>>(this.native.path + '/tools', {name: call.name, arguments: call.arguments, action_id: actionId},
            AbortSignal.any([...(options.signal ? [options.signal] : []), AbortSignal.timeout(Math.max(1, Math.round(deadline - performance.now())))]));
          actions.push(action);
          messages.push({role: 'tool', name: call.name, tool_call_id: call.id, content: JSON.stringify(Object.fromEntries(['action_id', 'status', 'result', 'error'].filter(key => key in action).map(key => [key, action[key]])))});
          yield {type: 'tool_result', step, action};
          if (action.status === 'uncertain') throw new RuntimeError('action_outcome_unknown', 'Inspect the paired-host action before continuing', {action_id: actionId});
        }
      }
      throw new RuntimeError('agent_step_limit', 'Agent reached its configured step limit', {actions});
    } catch (error) {
      if (!committed) { await this.runtime.state.finish(id, turnId, messages, undefined, {code: error instanceof RuntimeError ? error.code : 'turn_failed', message: String(error), actions}); committed = true; }
      throw error;
    } finally {
      if (!committed) await this.runtime.state.finish(id, turnId, messages, undefined, {code: 'cancelled', message: 'Consumer closed the stream', actions});
    }
  }
  async send(id: string, content: Message['content'], options: SendOptions = {}): Promise<ChatResult> {
    for await (const event of this.sendStream(id, content, options)) if (event.type === 'done') return event;
    throw new RuntimeError('incomplete_generation', 'Turn did not complete');
  }
  async classify(modelName: string, source: ImageInput, options: Selection & {top_k?: number} = {}): Promise<ClassificationResult> {
    const image = source instanceof Blob ? source : new Blob([typeof source === 'string' ? bytesFromBase64(source) as Uint8Array<ArrayBuffer> : source as Uint8Array<ArrayBuffer>]);
    const model = this.package_.manifest.models[modelName], config = model?.image;
    if (model?.task !== 'image.classification' || !config) throw new RuntimeError('wrong_model_task', 'Model has no image classification contract');
    if (image.size > 32 * 1024 * 1024) throw new RuntimeError('invalid_image', 'Image exceeds 32 MiB');
    const bitmap = await createImageBitmap(image, {imageOrientation: 'from-image'});
    let input: TensorValue;
    try {
      if (bitmap.width * bitmap.height > 40_000_000) throw new RuntimeError('invalid_image', 'Decoded image exceeds 40 megapixels');
      const canvas = new OffscreenCanvas(config.width, config.height), context = canvas.getContext('2d', {willReadFrequently: true})!;
      const resize = config.resize_shorter;
      if (resize) {
        const ratio = resize / Math.min(bitmap.width, bitmap.height), width = Math.floor(bitmap.width * ratio), height = Math.floor(bitmap.height * ratio);
        if (width < config.width || height < config.height) throw new RuntimeError('invalid_image', 'Resize dimensions are smaller than the requested crop');
        const resized = new OffscreenCanvas(width, height), resizeContext = resized.getContext('2d')!;
        resizeContext.drawImage(bitmap, 0, 0, width, height);
        context.drawImage(resized, Math.floor((width - config.width) / 2), Math.floor((height - config.height) / 2), config.width, config.height, 0, 0, config.width, config.height);
      } else context.drawImage(bitmap, 0, 0, config.width, config.height);
      const pixels = context.getImageData(0, 0, config.width, config.height).data, count = config.width * config.height, values = new Float32Array(count * 3), nchw = (config.layout || 'nchw') === 'nchw';
      for (let pixel = 0; pixel < count; pixel++) for (let channel = 0; channel < 3; channel++) values[nchw ? channel * count + pixel : pixel * 3 + channel] = (pixels[pixel * 4 + channel] * (config.scale ?? 1 / 255) - (config.mean?.[channel] || 0)) / (config.std?.[channel] ?? 1);
      input = encodeTensor(values, nchw ? [1, 3, config.height, config.width] : [1, config.height, config.width, 3]);
    } finally { bitmap.close(); }
    const result = await this.infer(modelName, {[config.input_name]: input}, options);
    const logits = decodeTensor(result.outputs[model.output_name || Object.keys(result.outputs)[0]]), maximum = Math.max(...logits), probabilities = Array.from(logits, value => Math.exp(value - maximum)), sum = probabilities.reduce((a, b) => a + b, 0);
    const labels: string[] = model.labels ? JSON.parse(await (await this.package_.artifact(model.labels, options.signal)).text()) : [];
    const topK = options.top_k ?? 5;
    if (!Number.isInteger(topK) || topK < 1 || topK > 100) throw new RuntimeError('invalid_top_k', 'top_k must be 1 to 100');
    return {...result, predictions: probabilities.map((value, index) => ({index, label: labels[index] || String(index), score: value / sum})).sort((a, b) => b.score - a.score).slice(0, topK)};
  }
}
