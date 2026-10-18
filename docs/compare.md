# Compare variants on your data

`dtr compare` answers two questions:
- What can this machine run?
- What does a quantization or engine change on my own inputs?

It runs every variant of one model that this machine can run, on the same
inputs, and ranks the results.

```sh
uv run dtr compare .work/chat --inputs prompts.jsonl
uv run dtr compare .work/vision --inputs images.jsonl --json
```

## Inputs

Inputs are a JSONL file with one object per line:

| Task | Line | Optional `expected` |
| --- | --- | --- |
| chat | `{"prompt": "..."}` or `{"messages": [...]}` | a string, or a list of strings, that a correct answer contains |
| image classification | `{"image": "relative/or/absolute.jpg"}` (relative to the JSONL file) | the correct label |

## What each column means

| Column | Meaning |
| --- | --- |
| accuracy | Share of inputs with `expected` that the variant got right. Matching ignores case and whitespace. |
| agreement | Similarity to the reference variant's output. Chat uses a character-sequence ratio; classification uses top-1 label match. |
| identical | Share of outputs exactly equal to the reference, after normalization |
| latency | Median time per input, excluding model load |
| tok/s | Output tokens divided by the engine's inference time. The same formula is used for every engine. |
| peak mem | Peak memory as reported by the engine; MLX reports it, other engines show `–` |
| load | Time to load the model into a fresh worker. For `local_server` this is only the connection, because the server already holds the model. |

Scoring ignores `<think>…</think>` reasoning blocks. Some engines return
them and others don't.

## Reference and ranking

- The reference is `--reference`. Without it, the reference is the runnable
  variant whose artifacts take the most bytes, usually the highest-precision
  build.
- With expected answers, variants rank by accuracy, then by latency. Without
  them, they rank by agreement with the reference, then by latency.
- Variants this machine can't run are listed as skipped, with the reasons.
- A variant that fails to load or run is listed with its error code; the
  other variants still run.
- Generation uses temperature 0. Each variant runs alone, unloaded between
  variants, so memory and load figures aren't mixed.

## Example

Run on an Apple M5 Max with Qwen3 0.6B and 8 short prompts that have expected
answers (`dtr compare … --max-tokens 48`):

| rank | variant | backend | accuracy | agreement | identical | latency | tok/s | peak mem | load |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | llama-q8 | local_server/server | 100% | 100% | 100% | 36 ms | 242 | – | 27 ms |
| 2 | mlx-8bit (reference) | mlx/gpu | 100% | 100% | 100% | 36 ms | 231 | 0.72 GB | 2817 ms |
| 3 | mlx-4bit | mlx/gpu | 88% | 68% | 25% | 39 ms | 304 | 0.44 GB | 2800 ms |

Below the table, the CLI prints each input with every variant's output. In
this run, that shows the 4-bit build answering "Red Planet" instead of "Mars",
and "The chemical symbol for gold is **Au**." when asked for the symbol only.
