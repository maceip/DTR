import {Application, ChatOptions, ChatResult, ClassificationResult, ImageInput, InferOptions, InferenceResult, Manifest, Message, PackageInfo, RuntimeError, RuntimeEvent, Selection, SendOptions, Session, TensorValue} from './types.js';
import {base64FromBytes} from './tensors.js';
import type {FineTuneRequest, ModelCapabilities} from './types.js';

export class NativeHost {
  readonly url: string;
  constructor(url: string, private token: string) { this.url = url.replace(/\/$/, ''); }
  async response(path: string, init: RequestInit = {}): Promise<Response> {
    const headers = new Headers(init.headers);
    headers.set('Authorization', `Bearer ${this.token}`);
    if (init.body) headers.set('Content-Type', 'application/json');
    let response: Response;
    try { response = await fetch(this.url + path, {...init, headers}); }
    catch (error) {
      if (init.signal?.aborted) throw new RuntimeError('cancelled', 'Request was cancelled');
      throw new RuntimeError('host_unreachable', String(error), undefined, true);
    }
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      if (body.error) throw RuntimeError.from(body.error);
      throw new RuntimeError('host_request', `Host returned HTTP ${response.status}`, body);
    }
    return response;
  }
  async request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
    const response = await this.response(path, {method: body === undefined ? 'GET' : 'POST', body: body === undefined ? undefined : JSON.stringify(body), signal});
    const value = await response.json();
    if (value.error && !['denied', 'failed', 'uncertain'].includes(value.status)) throw RuntimeError.from(value.error);
    if (value.execution) value.execution.location = 'remote_host';
    return value as T;
  }
  async device(): Promise<Record<string, unknown>> { return {...await this.request<Record<string, unknown>>('/v1/device'), location: 'remote_host'}; }
  async packages(): Promise<PackageInfo[]> { return (await this.request<{packages: PackageInfo[]}>('/v1/packages')).packages; }
  async application(digest: string): Promise<NativeApplication> {
    await this.request(`/v1/packages/${encodeURIComponent(digest)}`);
    return new NativeApplication(this, digest);
  }
  async *events(path: string, body: Record<string, unknown>, signal?: AbortSignal): AsyncGenerator<RuntimeEvent> {
    const requestId = crypto.randomUUID();
    let completed = false;
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
    try {
      const response = await this.response(path, {method: 'POST', body: JSON.stringify({...body, stream: true}), headers: {'X-Request-ID': requestId}, signal});
      if (!response.body) throw new RuntimeError('host_protocol', 'Host returned an empty stream');
      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = '';
      for (;;) {
        const chunk = await reader.read();
        pending += decoder.decode(chunk.value, {stream: !chunk.done});
        if (pending.length > 4 * 1024 * 1024) throw new RuntimeError('host_protocol', 'Host emitted an oversized event');
        let newline;
        while ((newline = pending.indexOf('\n')) >= 0) {
          const line = pending.slice(0, newline); pending = pending.slice(newline + 1);
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          if (event.type === 'error') { completed = true; throw RuntimeError.from(event.error); }
          if (event.type === 'done') completed = true;
          if (event.execution) event.execution.location = 'remote_host';
          yield event as RuntimeEvent;
        }
        if (chunk.done) break;
      }
      if (pending.trim() || !completed) throw new RuntimeError('incomplete_generation', 'Host disconnected before completion', undefined, true);
    } catch (error) {
      if (signal?.aborted) throw new RuntimeError('cancelled', 'Request was cancelled');
      throw error;
    } finally {
      await reader?.cancel().catch(() => undefined);
      reader?.releaseLock();
      if (!completed) await this.request(`/v1/jobs/${requestId}/cancel`, {}, AbortSignal.timeout(3000)).catch(() => undefined);
    }
  }
}

export function wireOptions<T extends Selection>(options: T): Omit<T, 'signal'> { const {signal: _signal, ...wire} = options; return wire; }

export class NativeApplication implements Application {
  readonly path: string;
  constructor(readonly host: NativeHost, readonly digest: string) { this.path = `/v1/packages/${encodeURIComponent(digest)}`; }
  describe(): Promise<PackageInfo> { return this.host.request(this.path); }
  capabilities(model: string): Promise<ModelCapabilities> { return this.host.request(this.path + '/capabilities?model=' + encodeURIComponent(model)); }
  fineTune(request: FineTuneRequest): Promise<never> { return this.host.request(this.path + '/fine-tune', request); }
  manifest(): Promise<Manifest> { return this.host.request(this.path + '/manifest'); }
  infer(model: string, inputs: Record<string, TensorValue>, options: InferOptions = {}): Promise<InferenceResult> {
    for (const tensor of Object.values(inputs)) if (['int64', 'uint64'].includes(tensor.dtype) && tensor.data?.some(value => typeof value !== 'number' || !Number.isSafeInteger(value))) throw new RuntimeError('integer_precision', 'Encode 64-bit integers outside JavaScript safe-integer range as little-endian base64');
    return this.host.request(this.path + '/infer', {model, inputs, ...wireOptions(options)}, options.signal);
  }
  async classify(model: string, image: ImageInput, options: Selection & {top_k?: number} = {}): Promise<ClassificationResult> {
    const image_base64 = typeof image === 'string' ? image : base64FromBytes(image instanceof Blob ? new Uint8Array(await image.arrayBuffer()) : image);
    return this.host.request(this.path + '/classify', {model, image_base64, ...wireOptions(options)}, options.signal);
  }
  chat(model: string, messages: Message[], options: ChatOptions = {}): Promise<ChatResult> {
    return this.host.request(this.path + '/chat', {model, messages, ...wireOptions(options)}, options.signal);
  }
  async *chatStream(model: string, messages: Message[], options: ChatOptions = {}): AsyncGenerator<RuntimeEvent> {
    yield* this.host.events(this.path + '/chat', {model, messages, ...wireOptions(options)}, options.signal);
  }
  createSession(model: string, messages: Message[] = []): Promise<Session> { return this.host.request(this.path + '/sessions', {model, messages}); }
  session(sessionId: string): Promise<Session> { return this.host.request(`/v1/sessions/${encodeURIComponent(sessionId)}?package=${encodeURIComponent(this.digest)}`); }
  send(sessionId: string, content: Message['content'], options: SendOptions = {}): Promise<ChatResult> {
    return this.host.request(`/v1/sessions/${encodeURIComponent(sessionId)}/messages?package=${encodeURIComponent(this.digest)}`, {content, ...wireOptions(options), turn_id: options.turn_id || crypto.randomUUID()}, options.signal);
  }
  async *sendStream(sessionId: string, content: Message['content'], options: SendOptions = {}): AsyncGenerator<RuntimeEvent> {
    yield* this.host.events(`/v1/sessions/${encodeURIComponent(sessionId)}/messages?package=${encodeURIComponent(this.digest)}`, {content, ...wireOptions(options), turn_id: options.turn_id || crypto.randomUUID()}, options.signal);
  }
  invokeTool(name: string, arguments_: Record<string, unknown>, actionId: string): Promise<Record<string, unknown>> {
    return this.host.request(this.path + '/tools', {name, arguments: arguments_, action_id: actionId});
  }
  recoverSession(sessionId: string): Promise<Session> { return this.host.request(`/v1/sessions/${encodeURIComponent(sessionId)}/recover?package=${encodeURIComponent(this.digest)}`, {}); }
}
