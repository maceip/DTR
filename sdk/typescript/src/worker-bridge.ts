import {RuntimeError, RuntimeEvent} from './types.js';

/** Every native WASM engine runs in its own worker, so cancellation also interrupts synchronous kernels. */
export class WorkerBridge {
  private worker: Worker;
  private active?: (error: RuntimeError) => void;
  closed = false;
  // Upstream LiteRT's Emscripten loader uses importScripts. Bundle this entry
  // as a classic worker; a module worker cannot load those native WASM shims.
  constructor(url: URL) { this.worker = new Worker(url, {name: 'phaseone-inference'}); }
  close(error = new RuntimeError('cancelled', 'Inference worker was closed')): void {
    if (this.closed) return;
    this.closed = true;
    this.worker.terminate();
    this.active?.(error);
  }
  request<T>(operation: string, payload: unknown, timeout: number, signal?: AbortSignal, onEvent?: (event: RuntimeEvent) => void): Promise<T> {
    if (this.closed) return Promise.reject(new RuntimeError('worker_closed', 'Model worker is no longer available', undefined, true));
    if (this.active) return Promise.reject(new RuntimeError('model_busy', 'This model is processing another request', undefined, true));
    if (signal?.aborted) return Promise.reject(new RuntimeError('cancelled', 'Request was cancelled'));
    return new Promise((resolve, reject) => {
      const id = crypto.randomUUID();
      let settled = false;
      const finish = (error?: RuntimeError, value?: T) => {
        if (settled) return;
        settled = true; clearTimeout(timer); signal?.removeEventListener('abort', abort);
        this.worker.removeEventListener('message', receive); this.worker.removeEventListener('error', crashed);
        this.active = undefined;
        if (error) reject(error); else resolve(value!);
      };
      const receive = (message: MessageEvent) => {
        const result = message.data;
        if (result.id !== id) return;
        if (result.event) onEvent?.(result.event);
        else if (result.error) finish(RuntimeError.from(result.error));
        else finish(undefined, result.result);
      };
      const crashed = (event: ErrorEvent) => this.close(new RuntimeError('native_worker_failed', event.message || 'WASM worker crashed', undefined, true));
      const abort = () => this.close(new RuntimeError('cancelled', 'Request was cancelled'));
      const timer = setTimeout(() => this.close(new RuntimeError('inference_timeout', 'Inference exceeded its time budget', undefined, true)), timeout * 1000);
      this.active = error => finish(error);
      this.worker.addEventListener('message', receive); this.worker.addEventListener('error', crashed);
      signal?.addEventListener('abort', abort, {once: true});
      try { this.worker.postMessage({id, operation, payload}); }
      catch (error) { finish(new RuntimeError('request_encoding', String(error))); }
    });
  }
  async *events(operation: string, payload: unknown, timeout: number, signal?: AbortSignal): AsyncGenerator<RuntimeEvent> {
    const queue: RuntimeEvent[] = [];
    let wake: (() => void) | undefined, finished = false, failure: unknown;
    const run = this.request<RuntimeEvent>(operation, payload, timeout, signal, event => { queue.push(event); wake?.(); })
      .then(result => { queue.push(result); }, error => { failure = error; })
      .finally(() => { finished = true; wake?.(); });
    try {
      while (queue.length || !finished) {
        if (queue.length) yield queue.shift()!;
        else await new Promise<void>(resolve => { wake = resolve; });
      }
      if (failure) throw failure;
    } finally {
      if (!finished) this.close();
      await run;
    }
  }
}
