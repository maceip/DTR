import {ChatConfig, Generation, Message, RuntimeError, ToolCall, ToolDefinition} from './types.js';

export class StopStream {
  private pending = '';
  stopped = false;
  constructor(private stops: string[] = []) {}
  push(text: string): string {
    if (this.stopped) return '';
    this.pending += text;
    let position = this.pending.length;
    for (const stop of this.stops) { const index = this.pending.indexOf(stop); if (index >= 0) { position = Math.min(position, index); this.stopped = true; } }
    if (this.stopped) { const output = this.pending.slice(0, position); this.pending = ''; return output; }
    let hold = 0;
    for (const stop of this.stops) for (let size = 1; size < stop.length; size++) if (this.pending.endsWith(stop.slice(0, size))) hold = Math.max(hold, size);
    const output = this.pending.slice(0, this.pending.length - hold); this.pending = this.pending.slice(this.pending.length - hold); return output;
  }
  flush(): string { const output = this.stopped ? '' : this.pending; this.pending = ''; return output; }
}
export function generationOptions(value: Generation = {}): Required<Pick<Generation, 'max_output_tokens' | 'temperature' | 'timeout_seconds' | 'stop'>> & Generation {
  const options = {max_output_tokens: 128, temperature: 0, timeout_seconds: 120, stop: [], ...value};
  if (!Number.isInteger(options.max_output_tokens) || options.max_output_tokens < 1 || options.max_output_tokens > 32768 || !Number.isFinite(options.temperature) || options.temperature < 0 || options.temperature > 2 || !Number.isFinite(options.timeout_seconds) || options.timeout_seconds <= 0 || options.timeout_seconds > 3600 || options.stop.length > 16 || options.stop.some(x => typeof x !== 'string' || !x.length || x.length > 256)) throw new RuntimeError('invalid_generation', 'Invalid generation limits');
  if (options.top_p !== undefined && (!(options.top_p > 0) || options.top_p > 1)) throw new RuntimeError('invalid_generation', 'top_p must be in (0, 1]');
  if (options.top_k !== undefined && (!Number.isInteger(options.top_k) || options.top_k < 1 || options.top_k > 10000)) throw new RuntimeError('invalid_generation', 'top_k must be 1 to 10000');
  return options;
}
export function prepareMessages(messages: Message[], config: ChatConfig, tools: Omit<ToolDefinition, 'permission' | 'side_effect'>[] = []): Message[] {
  if (!messages.length || messages.length > 4096 || new TextEncoder().encode(JSON.stringify(messages)).length > 32 * 1024 * 1024) throw new RuntimeError('invalid_messages', 'Provide a bounded, nonempty conversation');
  const result = structuredClone(messages);
  for (const message of result) {
    if (!['system', 'user', 'assistant', 'tool'].includes(message.role)) throw new RuntimeError('invalid_messages', 'Unknown message role');
    if (typeof message.content !== 'string') {
      for (const part of message.content) {
        if (!config.modalities.includes(part.type)) throw new RuntimeError('unsupported_modality', `This model does not accept ${part.type}`);
      }
    }
    if (config.tool_protocol === 'json') {
      if (message.tool_calls) { message.content = JSON.stringify({tool_calls: message.tool_calls}); delete message.tool_calls; }
      if (message.role === 'tool') message.content = JSON.stringify({tool: message.name, tool_call_id: message.tool_call_id, result: message.content});
    }
  }
  if (tools.length && config.tool_protocol === 'none') throw new RuntimeError('unsupported_tools', 'This model does not declare a tool protocol');
  if (tools.length && config.tool_protocol === 'json') {
    const instruction = 'When you need a tool, reply only with JSON: {"tool_calls":[{"name":"tool_name","arguments":{}}]}. The host executes tools and returns results. Otherwise answer normally. Available tools: ' + JSON.stringify(tools);
    if (result[0].role === 'system' && typeof result[0].content === 'string') result[0].content += '\n' + instruction;
    else result.unshift({role: 'system', content: instruction});
  }
  return result;
}
export function completion(text: string, protocol: ChatConfig['tool_protocol'], nativeCalls?: ToolCall[], toolsEnabled = false): Message {
  let calls = nativeCalls;
  if (protocol === 'json' && toolsEnabled) {
    let envelope;
    try {
      const candidate = text.trim();
      envelope = JSON.parse(candidate.startsWith('```json\n') && candidate.endsWith('```') ? candidate.slice(8, -3).trim() : candidate);
    } catch { /* A non-JSON answer is ordinary assistant text. */ }
    if (envelope && Object.hasOwn(envelope, 'tool_calls')) {
      if (!Array.isArray(envelope.tool_calls) || envelope.tool_calls.length > 16) throw new RuntimeError('invalid_tool_call', 'Model emitted an invalid tool call list');
      calls = envelope.tool_calls.map((call: {id?: string; name: string; arguments: Record<string, unknown>}) => ({id: call.id || crypto.randomUUID(), name: call.name, arguments: call.arguments}));
      text = typeof envelope.content === 'string' ? envelope.content : '';
    }
  }
  if (calls?.length) {
    if (calls.length > 16 || calls.some(call => !/^[a-zA-Z][a-zA-Z0-9_.-]{0,95}$/.test(call.name) || !call.arguments || Array.isArray(call.arguments) || typeof call.arguments !== 'object')) throw new RuntimeError('invalid_tool_call', 'Model returned an invalid tool request');
    return {role: 'assistant', content: text, tool_calls: calls};
  }
  return {role: 'assistant', content: text};
}
