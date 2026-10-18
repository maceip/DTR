#!/usr/bin/env python3
"""Export the pinned SmolLM2 Hugging Face checkpoint with ExecuTorch's LLM exporter.

Uses the upstream HF RoPE implementation so Q/K weights keep their original
layout. No checkpoint downloads or remote Python code run during export.
"""

import argparse
import json
from pathlib import Path


def export(source: Path, destination: Path, context: int):
    import torch
    from executorch.examples.models.llama.export_llama_lib import export_llama
    from executorch.extension.llm.export.config.llm_config import LlmConfig, ModelType
    from safetensors.torch import load_file

    source, destination = source.resolve(), destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    if config["model_type"] != "llama" or config.get("attention_bias"):
        raise ValueError("This exporter requires SmolLM2's Llama architecture without attention bias")
    weights = load_file(source / "model.safetensors")
    mapping = {
        "self_attn.q_proj.weight": "attention.wq.weight",
        "self_attn.k_proj.weight": "attention.wk.weight",
        "self_attn.v_proj.weight": "attention.wv.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
    }
    converted = {}
    for key, tensor in weights.items():
        if key == "model.embed_tokens.weight":
            name = "tok_embeddings.weight"
        elif key == "model.norm.weight":
            name = "norm.weight"
        elif key == "lm_head.weight":
            name = "output.weight"
        elif key.startswith("model.layers."):
            _, _, layer, suffix = key.split(".", 3)
            name = f"layers.{layer}.{mapping[suffix]}"
        else:
            raise ValueError(f"Unmapped checkpoint weight: {key}")
        converted[name] = tensor
    if config.get("tie_word_embeddings"):
        converted["output.weight"] = converted["tok_embeddings.weight"]
    checkpoint = destination / "checkpoint.pt"
    torch.save(converted, checkpoint)
    params = {
        "dim": config["hidden_size"], "hidden_dim": config["intermediate_size"],
        "n_layers": config["num_hidden_layers"], "n_heads": config["num_attention_heads"],
        "n_kv_heads": config["num_key_value_heads"], "vocab_size": config["vocab_size"],
        "norm_eps": config["rms_norm_eps"], "rope_theta": config["rope_theta"],
        "use_hf_rope": True,
    }
    params_path = destination / "params.json"
    params_path.write_text(json.dumps(params, indent=2) + "\n")
    options = LlmConfig()
    options.base.model_class = ModelType.smollm2
    options.base.checkpoint = str(checkpoint)
    options.base.params = str(params_path)
    options.base.tokenizer_path = str(source / "tokenizer.json")
    options.base.metadata = json.dumps({"get_bos_id": config["bos_token_id"],
                                       "get_eos_ids": [config["eos_token_id"]]})
    options.model.use_kv_cache = True
    # The stock native wheel does not register llama::custom_sdpa/update_cache.
    # Keep KV caching, expressed through portable/XNNPACK operators instead.
    options.model.use_sdpa_with_kv_cache = False
    options.export.max_seq_length = context
    options.export.max_context_length = context
    options.export.output_dir = str(destination)
    options.export.output_name = str(destination / "smollm2.pte")
    options.backend.xnnpack.enabled = True
    result = export_llama(options)
    print(json.dumps({"artifact": result, "context_length": context,
                      "checkpoint_source": json.loads((source / "source.json").read_text())}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--context", type=int, default=1024)
    args = parser.parse_args()
    export(args.source, args.destination, args.context)
