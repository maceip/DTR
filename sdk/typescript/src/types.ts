export type Backend = 'executorch' | 'litert';
export type DType = 'float16' | 'float32' | 'float64' | 'int8' | 'uint8' | 'int16' | 'uint16' | 'int32' | 'uint32' | 'int64' | 'uint64' | 'bool';
export interface TensorValue { dtype: DType; shape: number[]; data?: (number | boolean)[]; base64?: string; }
export interface Quantization { scales: number[]; zero_points: number[]; axis: number; }
export interface TensorSpec { name: string; dtype: DType; shape: number[]; native_name?: string; quantization?: Quantization; }
export interface Signature { method: string; inputs: TensorSpec[]; outputs: TensorSpec[]; }
export interface Artifact { path: string; size: number; sha256: string; media_type?: string; }
export interface ChatConfig {
  context_length: number; tokenizer?: string; tokenizer_config?: string; template?: string;
  prompt_format?: 'chatml' | 'llama3';
  modalities: ('text' | 'image' | 'audio')[]; tool_protocol: 'json' | 'native' | 'none';
  processor?: {config: string; image_token: string; image_splitting?: boolean; runner?: 'native' | 'program' | 'gemma4-ple'; vision_inputs?: string[]; image_fit?: 'processor' | 'square_pad'; image_size?: number}; max_images?: number;
  vision_accelerator?: 'cpu' | 'gpu' | 'npu'; audio_accelerator?: 'cpu' | 'gpu' | 'npu';
}
export interface Variant {
  id: string; backend: Backend; engine: 'program' | 'compiled' | 'interpreter' | 'llm'; artifact: string;
  data_artifact?: string; supporting_artifacts?: string[]; signatures: Record<string, Signature>; chat?: ChatConfig; priority?: number;
  target: { systems?: string[]; architectures?: string[]; accelerator?: 'cpu' | 'gpu' | 'npu'; delegate?: string; soc?: string; runtime_version?: string; minimum_memory_mb?: number;
    minimum_android_api?: number; components?: Record<string, string>; firmware?: string[]; minimum_security_patch?: string; };
}
export interface ImagePreprocess { input_name: string; width: number; height: number; layout?: 'nchw' | 'nhwc'; resize_shorter?: number; scale?: number; mean?: number[]; std?: number[]; }
export interface Model { task: 'tensor' | 'chat' | 'image.classification' | 'embedding'; variants: Variant[]; image?: ImagePreprocess; labels?: string; output_name?: string; }
export interface ToolDefinition { name: string; description: string; parameters: Record<string, unknown>; permission: string; side_effect?: boolean; }
export interface Manifest {
  schema_version: 1; name: string; version: string; description?: string; artifacts: Record<string, Artifact>; models: Record<string, Model>;
  tools?: ToolDefinition[]; permissions?: string[]; agent?: { model: string; instruction: string; tools: string[]; max_steps: number }; metadata?: Record<string, unknown>;
}
export interface ToolCall { id: string; name: string; arguments: Record<string, unknown>; }
export interface FineTuneRequest { model: string; dataset: string; objective?: 'function_calling' | 'chat'; method?: 'lora'; }
export interface ModelCapabilities {
  model: string;
  function_calling: {variants: {variant: string; protocol: 'json' | 'native' | 'none'; declared: boolean}[]; execution: 'application_tool_gateway'; quality_verified: false};
  fine_tuning: {available: false; status: 'not_implemented'; objectives: ('function_calling' | 'chat')[]};
}
export type ContentPart = { type: 'text'; text: string } | { type: 'image' | 'audio'; data: string; mime_type: string };
export interface Message { role: 'system' | 'user' | 'assistant' | 'tool'; content: string | ContentPart[]; tool_calls?: ToolCall[]; tool_call_id?: string; name?: string; }
export interface Generation { max_output_tokens?: number; temperature?: number; top_p?: number; top_k?: number; stop?: string[]; timeout_seconds?: number; }
export interface Selection { backend?: Backend; variant?: string; accelerator?: 'cpu' | 'gpu' | 'npu'; allow_fallback?: boolean; signal?: AbortSignal; }
export interface ChatOptions extends Selection { generation?: Generation; tools?: Omit<ToolDefinition, 'permission' | 'side_effect'>[]; }
export interface SendOptions extends ChatOptions { turn_id?: string; agent?: boolean; }
export interface InferOptions extends Selection { signature?: string; quantize?: boolean; dequantize?: boolean; timeout_seconds?: number; }
export interface Execution {
  backend: Backend; engine: string; variant: string; package: string; model: string; location: 'native_host' | 'remote_host' | 'browser';
  runtime_version?: string; requested_accelerator?: string; inference_ms?: number; execution_verified?: boolean; npu_execution_verified: boolean; [key: string]: unknown;
}
export interface InferenceResult { outputs: Record<string, TensorValue>; execution: Execution; receipt?: unknown; }
export type ImageInput = Blob | Uint8Array | string;
export interface ClassificationResult extends InferenceResult { predictions: {index: number; label: string; score: number}[]; }
export interface ChatResult { type: 'done'; message: Message; finish_reason: string; usage: Record<string, number | string>; execution: Execution; session_id?: string; turn_id?: string; cached?: boolean; actions?: unknown[]; }
export type RuntimeEvent = { type: 'start'; execution: Execution; request_id?: string } | { type: 'delta'; text: string; request_id?: string } | ChatResult | { type: 'turn' | 'tool_call' | 'tool_result'; [key: string]: unknown };
export interface Session { id: string; package: string; model: string; messages: Message[]; active_turn?: string | null; }
export interface PackageInfo { name: string; version: string; digest: string; description?: string; models: Record<string, Model>; tools: ToolDefinition[]; permissions: string[]; agent?: Manifest['agent']; compatibility?: Record<string, unknown[]>; signature: string; [key: string]: unknown; }

export class RuntimeError extends Error {
  constructor(public code: string, message: string, public details?: unknown, public retryable = false) { super(message); this.name = 'RuntimeError'; }
  static from(error: { code: string; message: string; details?: unknown; retryable?: boolean }): RuntimeError { return new RuntimeError(error.code, error.message, error.details, error.retryable); }
}

export interface Application {
  readonly digest: string;
  describe(): PackageInfo | Promise<PackageInfo>;
  capabilities(model: string): ModelCapabilities | Promise<ModelCapabilities>;
  fineTune(request: FineTuneRequest): Promise<never>;
  infer(model: string, inputs: Record<string, TensorValue>, options?: InferOptions): Promise<InferenceResult>;
  classify(model: string, image: ImageInput, options?: Selection & {top_k?: number}): Promise<ClassificationResult>;
  chat(model: string, messages: Message[], options?: ChatOptions): Promise<ChatResult>;
  chatStream(model: string, messages: Message[], options?: ChatOptions): AsyncGenerator<RuntimeEvent>;
  createSession(model: string, messages?: Message[]): Promise<Session>;
  session(sessionId: string): Promise<Session>;
  recoverSession(sessionId: string): Promise<Session>;
  invokeTool(name: string, arguments_: Record<string, unknown>, actionId: string): Promise<Record<string, unknown>>;
  send(sessionId: string, content: Message['content'], options?: SendOptions): Promise<ChatResult>;
  sendStream(sessionId: string, content: Message['content'], options?: SendOptions): AsyncGenerator<RuntimeEvent>;
}
