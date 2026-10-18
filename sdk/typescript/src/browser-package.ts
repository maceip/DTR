import Ajv from 'ajv';
import {sha256} from '@noble/hashes/sha2.js';
import {bytesToHex} from '@noble/hashes/utils.js';
import schema from './package-schema.json';
import {NativeHost} from './native-host.js';
import {Manifest, RuntimeError} from './types.js';
import {bytesFromBase64} from './tensors.js';

const validate = new Ajv({strict: false, allErrors: true, useDefaults: true}).compile(schema);
export const hash = (value: Uint8Array): string => bytesToHex(sha256(value));
export interface PackageOptions { requireSignature?: boolean; trustedKeys?: string[]; cache?: boolean; maxArtifactBytes?: number; }

/** Verifies the same canonical bytes that the native host signs, without reserializing floats. */
export class BrowserPackage {
  private artifacts = new Map<string, Promise<Blob>>();
  private constructor(readonly host: NativeHost, readonly digest: string, readonly manifest: Manifest,
                      readonly signature: 'unsigned' | 'valid_untrusted' | 'trusted', private options: PackageOptions) {}
  static async open(host: NativeHost, digest: string, options: PackageOptions = {}): Promise<BrowserPackage> {
    if (!/^[a-f0-9]{64}$/.test(digest)) throw new RuntimeError('invalid_package', 'Package digest must be a SHA-256 hash');
    const response = await host.response(`/v1/packages/${digest}/manifest`);
    const blob = await response.blob();
    if (blob.size > 2 * 1024 * 1024) throw new RuntimeError('invalid_package', 'Manifest exceeds 2 MiB');
    const bytes = new Uint8Array(await blob.arrayBuffer());
    if (hash(bytes) !== digest) throw new RuntimeError('manifest_integrity', 'Manifest does not match the requested package digest');
    let manifest: Manifest;
    try { manifest = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes)); }
    catch { throw new RuntimeError('invalid_package', 'Manifest is not valid UTF-8 JSON'); }
    if (!validate(manifest)) throw new RuntimeError('invalid_package', 'Manifest does not satisfy schema version 1', validate.errors);
    const paths = new Set<string>();
    for (const artifact of Object.values(manifest.artifacts)) {
      const path = artifact.path.toLowerCase();
      if (!path || path.includes('\\') || path.includes('\0') || path.startsWith('/') || /^[a-z]:/.test(path) || path.split('/').some(part => !part || part === '.' || part === '..')) throw new RuntimeError('invalid_package', 'Artifact paths must be normalized relative paths');
      if (paths.has(path) || ['phaseone.json', 'phaseone.signature.json'].includes(path)) throw new RuntimeError('invalid_package', 'Artifact paths overlap or use a reserved filename');
      paths.add(path);
    }
    if (!Object.keys(manifest.models).length) throw new RuntimeError('invalid_package', 'An application needs at least one model');
    for (const model of Object.values(manifest.models)) {
      if (model.task === 'image.classification' && !model.image) throw new RuntimeError('invalid_package', 'Image classifiers require a preprocessing contract');
      if (model.image && ((model.image.mean || [0, 0, 0]).length !== 3 || (model.image.std || [1, 1, 1]).length !== 3 || (model.image.std || [1, 1, 1]).some(value => !(value > 0)))) throw new RuntimeError('invalid_package', 'RGB preprocessing requires three means and positive standard deviations');
      const ids = new Set<string>();
      for (const variant of model.variants) {
        variant.target ||= {};
        variant.signatures ||= {};
        if (ids.has(variant.id)) throw new RuntimeError('invalid_package', 'Duplicate variant ID');
        ids.add(variant.id);
        if ((variant.engine === 'llm') !== (model.task === 'chat') || (variant.engine === 'llm' && !variant.chat) || (variant.engine !== 'llm' && !Object.keys(variant.signatures || {}).length)) throw new RuntimeError('invalid_package', 'Variant engine and model task disagree');
        if ((variant.backend === 'litert' && variant.engine === 'program') || (variant.backend === 'executorch' && !['program', 'llm'].includes(variant.engine))) throw new RuntimeError('invalid_package', 'Engine does not belong to this backend');
        if (variant.backend === 'executorch' && variant.chat && !variant.chat.tokenizer) throw new RuntimeError('invalid_package', 'ExecuTorch chat requires a tokenizer');
        if (variant.chat?.runner === 'qnn') {
          if (variant.backend !== 'executorch' || variant.engine !== 'llm' || variant.target.accelerator !== 'npu'
              || variant.target.delegate !== 'QnnBackend' || variant.chat.modalities.length !== 1
              || variant.chat.modalities[0] !== 'text' || !variant.chat.decoder || !variant.chat.template
              || variant.chat.processor || variant.chat.tool_protocol === 'native') {
            throw new RuntimeError('invalid_package', 'QNN text runners require ExecuTorch/NPU/QnnBackend, a decoder and template');
          }
        } else if (variant.chat?.decoder != null) {
          throw new RuntimeError('invalid_package', 'The decoder field belongs to the QNN runner');
        }
        for (const signature of Object.values(variant.signatures || {})) for (const specs of [signature.inputs, signature.outputs]) {
          if (new Set(specs.map(spec => spec.name)).size !== specs.length) throw new RuntimeError('invalid_package', 'Tensor names must be unique');
          for (const spec of specs) {
            if (spec.shape.some(value => value < -1)) throw new RuntimeError('invalid_package', 'Tensor dimensions must be nonnegative or -1');
            if (spec.quantization && (spec.quantization.scales.some(value => !Number.isFinite(value) || value <= 0) || spec.quantization.scales.length !== spec.quantization.zero_points.length)) throw new RuntimeError('invalid_package', 'Quantization needs positive scales and matching zero points');
          }
        }
        for (const id of [variant.artifact, variant.data_artifact, ...(variant.supporting_artifacts || []), variant.chat?.tokenizer, variant.chat?.tokenizer_config, variant.chat?.template, model.labels]) {
          if (id && !Object.hasOwn(manifest.artifacts, id)) throw new RuntimeError('missing_artifact', `Manifest references missing artifact ${id}`);
        }
      }
    }
    const tools = manifest.tools || [], names = new Set(tools.map(tool => tool.name));
    if (names.size !== tools.length || tools.some(tool => !manifest.permissions?.includes(tool.permission))) throw new RuntimeError('invalid_package', 'Tool names must be unique and their permissions declared');
    if (manifest.agent && (manifest.models[manifest.agent.model]?.task !== 'chat' || manifest.agent.tools.some(name => !names.has(name)))) throw new RuntimeError('invalid_package', 'Agent references an undeclared chat model or tool');
    let status: BrowserPackage['signature'] = 'unsigned';
    try {
      const signature = await host.request<{algorithm: string; key_id: string; public_key: string; signature: string}>(`/v1/packages/${digest}/signature`);
      const publicBytes = bytesFromBase64(signature.public_key);
      if (signature.algorithm !== 'ed25519' || publicBytes.length !== 32 || hash(publicBytes) !== signature.key_id) throw new RuntimeError('invalid_signature', 'Invalid signing key');
      const key = await crypto.subtle.importKey('raw', publicBytes as Uint8Array<ArrayBuffer>, 'Ed25519', false, ['verify']);
      if (!await crypto.subtle.verify('Ed25519', key, bytesFromBase64(signature.signature) as Uint8Array<ArrayBuffer>, bytes)) throw new RuntimeError('invalid_signature', 'Package signature verification failed');
      status = (options.trustedKeys || []).some(value => value.toLowerCase() === bytesToHex(publicBytes)) ? 'trusted' : 'valid_untrusted';
    } catch (error) {
      if (!(error instanceof RuntimeError && error.code === 'signature_not_found')) {
        if (error instanceof RuntimeError) throw error;
        throw new RuntimeError('invalid_signature', `Cannot verify package signature: ${error}`);
      }
    }
    if (options.requireSignature && status !== 'trusted') throw new RuntimeError('untrusted_package', 'A signature from a browser-trusted key is required', {signature: status});
    return new BrowserPackage(host, digest, manifest, status, options);
  }
  async artifact(id: string, signal?: AbortSignal): Promise<Blob> {
    // A caller's cancellation must not cancel another caller's shared download.
    if (signal) return this.fetchArtifact(id, signal);
    let pending = this.artifacts.get(id);
    if (!pending) {
      pending = this.fetchArtifact(id).catch(error => { this.artifacts.delete(id); throw error; });
      this.artifacts.set(id, pending);
    }
    return pending;
  }
  private async fetchArtifact(id: string, signal?: AbortSignal): Promise<Blob> {
    const artifact = this.manifest.artifacts[id];
    if (!artifact) throw new RuntimeError('missing_artifact', `Unknown artifact ${id}`);
    const limit = this.options.maxArtifactBytes ?? 8 * 1024 ** 3;
    if (!Number.isSafeInteger(artifact.size) || artifact.size > limit) throw new RuntimeError('artifact_too_large', `Artifact exceeds the configured ${limit} byte limit`);
    const cache = this.options.cache !== false && typeof caches !== 'undefined' ? await caches.open('phaseone-artifacts-v1').catch(() => undefined) : undefined;
    const cacheKey = new URL(`/__phaseone_artifact__/${artifact.sha256}`, location.origin).href;
    const cached = await cache?.match(cacheKey);
    const response = cached || await this.host.response(`/v1/packages/${this.digest}/artifacts/${encodeURIComponent(id)}`, {signal});
    if (!response.body) throw new RuntimeError('artifact_integrity', 'Artifact response is empty');
    const reader = response.body.getReader(), digest = sha256.create(), chunks: BlobPart[] = [];
    let size = 0;
    try {
      for (;;) {
        if (signal?.aborted) throw new RuntimeError('cancelled', 'Artifact download was cancelled');
        const chunk = await reader.read();
        if (chunk.done) break;
        size += chunk.value.byteLength;
        if (size > artifact.size) throw new RuntimeError('artifact_integrity', 'Artifact is larger than its manifest declaration');
        digest.update(chunk.value);
        chunks.push(chunk.value as Uint8Array<ArrayBuffer>);
      }
      if (size !== artifact.size || bytesToHex(digest.digest()) !== artifact.sha256) {
        await cache?.delete(cacheKey);
        throw new RuntimeError('artifact_integrity', `Artifact checksum failed: ${id}`);
      }
      const result = new Blob(chunks, {type: artifact.media_type || 'application/octet-stream'});
      if (!cached && cache) await cache.put(cacheKey, new Response(result)).catch(() => undefined);
      return result;
    } finally { await reader.cancel().catch(() => undefined); reader.releaseLock(); }
  }
}
