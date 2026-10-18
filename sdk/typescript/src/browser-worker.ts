/// <reference lib="webworker" />
import {loadLiteRt, loadAndCompile, Tensor, CompiledModel, TensorDetails} from '@litertjs/core';
import {loadLiteRtLm, Engine, Backend, SamplerType, type Message as NativeMessage} from '@litert-lm/core';
import {ChatOptions, ChatResult, Execution, InferOptions, InferenceResult, Message, RuntimeError, Signature, TensorSpec, TensorValue, Variant} from './types.js';
import {decodeTensor, dequantizeTensor, encodeTensor, prepareTensor} from './tensors.js';
import {completion, generationOptions, prepareMessages, StopStream} from './chat-common.js';

const scope = self as unknown as DedicatedWorkerGlobalScope;
let tensorModel: CompiledModel | undefined, languageModel: Engine | undefined;
let variant: Variant, execution: Execution, busy = false;
const dtypeSupported = new Set(['float32', 'int32', 'uint8']);
function wasmAssets(directory: string): void {
  // Emscripten otherwise derives its .wasm URL from the worker entry URL,
  // which differs from the vendored loader directory in this distribution.
  (scope as unknown as {Module: {locateFile: (name: string) => string}}).Module = {
    locateFile: name => new URL(name, directory).href,
  };
}

function metadata(specs: TensorSpec[], details: readonly TensorDetails[], side: string): void {
  if (specs.length !== details.length) throw new RuntimeError('model_contract', `Native ${side} count differs from the manifest`);
  for (const spec of specs) {
    const detail = details.find(value => value.name === (spec.native_name || spec.name));
    if (!detail || detail.dtype !== spec.dtype || detail.shape.length !== spec.shape.length || spec.shape.some((size, i) => size !== detail.shape[i])) throw new RuntimeError('model_contract', `Native ${side} metadata differs for ${spec.name}`, {declared: spec, actual: detail && {...detail, shape: [...detail.shape]}});
  }
}
async function load(payload: {variant: Variant; model: Blob; execution: Execution; wasmRoot: string}): Promise<Execution> {
  variant = payload.variant; execution = payload.execution;
  if (variant.backend !== 'litert') throw new RuntimeError('native_host_required', 'ExecuTorch requires an explicitly paired native host');
  const accelerator = variant.target.accelerator || 'cpu';
  if (variant.engine === 'llm') {
    if (!variant.chat) throw new RuntimeError('model_contract', 'Missing chat configuration');
    if (accelerator === 'npu') throw new RuntimeError('unsupported_accelerator', 'LiteRT-LM web does not expose an NPU backend');
    wasmAssets(payload.wasmRoot + '/litert-lm/');
    await loadLiteRtLm(payload.wasmRoot + '/litert-lm/');
    languageModel = await Engine.create({model: payload.model, backend: accelerator === 'gpu' ? Backend.GPU_ARTISAN : Backend.CPU,
      mainExecutorSettings: {maxNumTokens: variant.chat.context_length}, benchmarkEnabled: true});
    execution = {...execution, runtime_version: '0.18.1', backend_api: '@litert-lm/core', binding_compatibility: 'async-engine-settings; legacy-streaming-constructor', accelerator_evidence: 'engine_configuration; per-operator profiling not available'};
  } else {
    for (const signature of Object.values(variant.signatures)) for (const spec of [...signature.inputs, ...signature.outputs]) {
      if (!dtypeSupported.has(spec.dtype)) throw new RuntimeError('unsupported_dtype', `LiteRT.js 2.5.3 exposes float32, int32 and uint8 tensors; use a native host for ${spec.dtype}`);
      if (spec.shape.includes(-1)) throw new RuntimeError('unsupported_dynamic_shape', 'This LiteRT.js version does not expose dynamic tensor resizing; use a native host');
    }
    if (payload.model.size > 2 * 1024 ** 3) throw new RuntimeError('artifact_too_large', 'LiteRT.js tensor models must fit in a 2 GiB ArrayBuffer');
    wasmAssets(payload.wasmRoot + '/litert/');
    await loadLiteRt(payload.wasmRoot + '/litert/', {jspi: false, threads: false});
    tensorModel = await loadAndCompile(new Uint8Array(await payload.model.arrayBuffer()), {
      accelerator: accelerator === 'cpu' ? 'wasm' : accelerator === 'gpu' ? 'webgpu' : 'webnn',
      ...(accelerator === 'npu' ? {webNNOptions: {devicePreference: 'npu' as const}} : {}),
    });
    if (accelerator !== 'cpu' && !tensorModel.isFullyAccelerated) throw new RuntimeError('acceleration_unavailable', 'The requested accelerator did not accept the complete graph; choose CPU explicitly');
    for (const signature of Object.values(variant.signatures)) {
      const runner = tensorModel.signatures[signature.method];
      if (!runner) throw new RuntimeError('model_contract', `Native signature ${signature.method} is absent`);
      metadata(signature.inputs, runner.getInputDetails(), 'input'); metadata(signature.outputs, runner.getOutputDetails(), 'output');
    }
    execution = {...execution, runtime_version: '2.5.3', backend_api: '@litertjs/core', fully_accelerated: tensorModel.isFullyAccelerated,
      accelerator_evidence: accelerator === 'cpu' ? 'WASM CPU execution' : 'LiteRT isFullyAccelerated; physical NPU execution not profiled'};
  }
  return execution;
}
async function infer(payload: {inputs: Record<string, TensorValue>; options: InferOptions}): Promise<InferenceResult> {
  if (!tensorModel) throw new RuntimeError('wrong_model_task', 'This worker is not a tensor model');
  const name = payload.options.signature || 'default', signature: Signature = variant.signatures[name];
  if (!signature) throw new RuntimeError('unknown_signature', `Unknown logical signature ${name}`);
  if (Object.keys(payload.inputs).length !== signature.inputs.length || signature.inputs.some(spec => !Object.hasOwn(payload.inputs, spec.name))) throw new RuntimeError('input_names', 'Provide exactly the declared input names');
  const inputs: Record<string, Tensor> = {}, outputs: Record<string, TensorValue> = {};
  let nativeOutputs: Record<string, Tensor> = {};
  const started = performance.now();
  try {
    for (const spec of signature.inputs) {
      const value = prepareTensor(payload.inputs[spec.name], spec, payload.options.quantize !== false);
      inputs[spec.native_name || spec.name] = new Tensor(decodeTensor(value) as Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer> | Uint8Array<ArrayBuffer>, value.shape);
    }
    nativeOutputs = await tensorModel.signatures[signature.method].run(inputs);
    for (const spec of signature.outputs) {
      const native = nativeOutputs[spec.native_name || spec.name];
      if (!native) throw new RuntimeError('model_contract', `Native output ${spec.name} is missing`);
      const data = await native.data();
      const value = encodeTensor(data, [...native.type.layout.dimensions], native.type.dtype);
      if (value.dtype !== spec.dtype || value.shape.length !== spec.shape.length || value.shape.some((size, i) => size !== spec.shape[i])) throw new RuntimeError('model_contract', `Native output ${spec.name} has changed shape or dtype`);
      outputs[spec.name] = payload.options.dequantize !== false ? dequantizeTensor(value, spec) : value;
    }
    return {outputs, execution: {...execution, inference_ms: performance.now() - started, execution_verified: true}};
  } finally {
    const unique = new Set([...Object.values(inputs), ...Object.values(nativeOutputs)]);
    for (const tensor of unique) if (!tensor.deleted) tensor.delete();
  }
}
function nativeMessage(message: Message): NativeMessage {
  if (message.role === 'tool' && variant.chat?.tool_protocol === 'native') {
    let response: unknown = message.content;
    if (typeof response === 'string') { try { response = JSON.parse(response); } catch { /* Preserve ordinary text tool results. */ } }
    if (response && typeof response === 'object' && 'status' in response && response.status === 'complete' && 'result' in response) response = response.result;
    return {role: 'tool', content: [{type: 'tool_response', name: message.name, response}], tool_call_id: message.tool_call_id} as NativeMessage;
  }
  return {
    role: message.role,
    content: typeof message.content === 'string' ? message.content : message.content.map(part => part.type === 'text' ? {...part} : {type: part.type, data: part.data}),
    ...(message.tool_calls ? {tool_calls: message.tool_calls.map(call => ({type: 'function', id: call.id, function: {name: call.name, arguments: call.arguments}}))} : {}),
    ...(message.name ? {name: message.name} : {}), ...(message.tool_call_id ? {tool_call_id: message.tool_call_id} : {}),
  } as NativeMessage;
}
async function chat(id: string, payload: {messages: Message[]; options: ChatOptions}): Promise<ChatResult> {
  if (!languageModel || !variant.chat) throw new RuntimeError('wrong_model_task', 'This worker is not a conversation model');
  const options = generationOptions(payload.options.generation), config = variant.chat;
  const messages = prepareMessages(payload.messages, config, payload.options.tools);
  if (options.max_output_tokens >= config.context_length) throw new RuntimeError('context_length', 'Output budget must leave room for input in the model context');
  const conversation = await languageModel.createConversation({
    preface: {messages: messages.slice(0, -1).map(nativeMessage),
      ...(config.tool_protocol === 'native' ? {tools: (payload.options.tools || []).map(tool => ({type: 'function', function: tool}))} : {})},
    prefillPrefaceOnInit: true,
    sessionConfig: {maxOutputTokens: options.max_output_tokens,
      samplerParams: {type: options.temperature === 0 ? SamplerType.GREEDY : options.top_p !== undefined ? SamplerType.TOP_P : SamplerType.TOP_K,
        temperature: options.temperature, ...(options.top_p !== undefined ? {p: options.top_p} : {}), ...(options.top_k !== undefined ? {k: options.top_k} : {})}},
  });
  const started = performance.now(), stop = new StopStream(options.stop);
  let text = '';
  const emit = (value: string) => { if (value) { text += value; scope.postMessage({id, event: {type: 'delta', text: value}}); } };
  try {
    if (await conversation.getTokenCount() + options.max_output_tokens >= config.context_length) throw new RuntimeError('context_length', 'Conversation history leaves insufficient output space');
    scope.postMessage({id, event: {type: 'start', execution}});
    const reader = conversation.sendMessageStreaming(nativeMessage(messages.at(-1)!)).getReader();
    try {
      for (;;) {
        const chunk = await reader.read();
        if (chunk.done) break;
        const content = chunk.value.content;
        const delta = typeof content === 'string' ? content : (content || []).filter(part => part.type === 'text').map(part => part.text).join('');
        emit(stop.push(delta));
        if (stop.stopped) conversation.cancel();
      }
    } catch (error) { if (!stop.stopped) throw error; }
    finally { await reader.cancel().catch(() => undefined); reader.releaseLock(); }
    emit(stop.flush());
    const history = await conversation.getHistory(), last = history.at(-1);
    const calls = last?.role === 'assistant' ? last.tool_calls?.map(call => ({id: call.id || crypto.randomUUID(), name: call.function.name, arguments: call.function.arguments})) : undefined;
    const message = completion(text, config.tool_protocol, calls, !!payload.options.tools?.length);
    const benchmark = await conversation.getBenchmarkInfo();
    return {type: 'done', message, finish_reason: message.tool_calls?.length ? 'tool_calls' : stop.stopped ? 'stop' : benchmark.lastDecodeTokenCount >= options.max_output_tokens ? 'length' : 'stop',
      usage: {input_tokens: benchmark.lastPrefillTokenCount, output_tokens: benchmark.lastDecodeTokenCount},
      execution: {...execution, inference_ms: performance.now() - started, execution_verified: true}};
  } finally { await conversation.delete(); }
}
scope.onmessage = async ({data}) => {
  const {id, operation, payload} = data;
  if (busy) { scope.postMessage({id, error: {code: 'model_busy', message: 'Worker is already executing'}}); return; }
  busy = true;
  try {
    const result = operation === 'load' ? await load(payload) : operation === 'infer' ? await infer(payload) : operation === 'chat' ? await chat(id, payload) : undefined;
    if (result === undefined) throw new RuntimeError('worker_protocol', 'Unknown worker operation');
    scope.postMessage({id, result});
  } catch (error) {
    scope.postMessage({id, error: error instanceof RuntimeError ? {code: error.code, message: error.message, details: error.details, retryable: error.retryable} : {code: 'native_inference_failed', message: String(error), details: {stack: error instanceof Error ? error.stack : undefined}}});
  } finally { busy = false; }
};
