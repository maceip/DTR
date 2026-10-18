import {readFile} from 'node:fs/promises';

/** Backport the async engine-settings binding used by the packaged WASM.
 * Upstream source: google-ai-edge/LiteRT-LM/js/packages/core/src/engine.ts.
 * Keep this narrow and fail the build if the pinned JS source changes.
 */
export const litertLmBindings = {
  name: 'litert-lm-0.18.1-binding-compatibility',
  setup(build) {
    build.onLoad({filter: /@litert-lm\/core\/dist\/engine\.js$/}, async ({path}) => {
      let contents = await readFile(path, 'utf8');
      const edits = [
        ['const wasmEngineSettings = wasm.EngineSettings.createDefault(modelAssets, { value: backend });',
         'const wasmEngineSettings = await wasm.EngineSettings.createDefault(modelAssets, { value: backend });'],
        ['engine = await wasm.Engine.createStreaming(wasmEngineSettings, inputPromptAsHint);',
         'engine = await (wasm.Engine.createStreaming || wasm.Engine.createLegacyEngine)(wasmEngineSettings, inputPromptAsHint);'],
      ];
      for (const [before, after] of edits) {
        if (contents.split(before).length !== 2) throw new Error('LiteRT-LM source changed; review and remove or update the binding compatibility patch.');
        contents = contents.replace(before, after);
      }
      return {contents, loader: 'js'};
    });
  },
};
