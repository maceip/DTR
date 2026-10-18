import {sha256} from '@noble/hashes/sha2.js';
import {bytesToHex} from '@noble/hashes/utils.js';
import {RuntimeError} from './types.js';

/** An immutable, bounded JSON snapshot shared by checks and execution. */
export function ownedJson<T>(value: T, code = 'invalid_tool_arguments'): T {
  const seen = new Set<object>();
  function check(item: unknown, depth: number): void {
    if (depth > 64) throw new RuntimeError(code, 'JSON nesting exceeds 64 levels');
    if (item === null || typeof item === 'string' || typeof item === 'boolean') return;
    if (typeof item === 'number' && Number.isFinite(item)) return;
    if (typeof item !== 'object' || seen.has(item)) throw new RuntimeError(code, 'Expected finite, acyclic JSON');
    if (!Array.isArray(item) && Object.getPrototypeOf(item) !== Object.prototype && Object.getPrototypeOf(item) !== null) throw new RuntimeError(code, 'Expected plain JSON objects');
    if (Array.isArray(item) && (Object.keys(item).length !== item.length || Object.keys(item).some((key, index) => key !== String(index)))) throw new RuntimeError(code, 'Sparse arrays and array properties are not JSON');
    seen.add(item);
    for (const child of Object.values(item)) check(child, depth + 1);
    seen.delete(item);
  }
  check(value, 0);
  if (new TextEncoder().encode(JSON.stringify(value)).length > 1024 * 1024) throw new RuntimeError(code, 'JSON exceeds 1 MiB');
  const copy = structuredClone(value);
  function freeze(item: unknown): void {
    if (item && typeof item === 'object') { Object.values(item).forEach(freeze); Object.freeze(item); }
  }
  freeze(copy);
  return copy;
}

export function jsonFingerprint(value: unknown): string {
  function stable(item: unknown): string {
    if (item === null || typeof item !== 'object') return JSON.stringify(item);
    if (Array.isArray(item)) return '[' + item.map(stable).join(',') + ']';
    return '{' + Object.entries(item).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => JSON.stringify(key) + ':' + stable(child)).join(',') + '}';
  }
  return bytesToHex(sha256(new TextEncoder().encode(stable(value))));
}
