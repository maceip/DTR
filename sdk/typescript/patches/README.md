# LiteRT-LM 0.18.1 binding compatibility

The published JavaScript wrapper calls `EngineSettings.createDefault` without
awaiting it. The included WASM binding returns a promise; cleanup then obscures
the original error with `wasmEngineSettings.delete is not a function`.

The build applies two narrow source substitutions to the pinned wrapper:

- Await engine settings, as the current upstream implementation does.
- Accept the `createLegacyEngine` name for the legacy streaming constructor,
  retaining `createStreaming` for the older binding.

Reference: [upstream engine implementation](https://github.com/google-ai-edge/LiteRT-LM/blob/main/js/packages/core/src/engine.ts).

This does not replace inference, modify weights, suppress native failures, or
remove cleanup. The build fails if either expected source line changes. The
real-model browser check is `scripts/check_runtime_web_lm.mjs`.
