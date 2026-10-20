"""Hardware-aware model cache scanner, KV-cache physics calculator, and zero-copy package resolver."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .build import import_mlx
from .device import compatibility, discover
from .errors import RuntimeFault
from .package import MANIFEST, atomic_json, load_package, read_json

CONSUMER_PROFILES = {
    "macos-metal-pro": {
        "name": "Mac Apple Silicon (M-Series Pro/Max/Ultra)",
        "system": "darwin",
        "architecture": "arm64",
        "accelerator": "gpu",
        "delegate": "mlx.metal",
        "memory_mb": 65536,
        "inference_budget_mb": 49152,
        "preferred_backends": ["mlx", "litert", "executorch"],
    },
    "macos-metal-consumer": {
        "name": "Mac Apple Silicon (16GB Consumer Baseline)",
        "system": "darwin",
        "architecture": "arm64",
        "accelerator": "gpu",
        "delegate": "mlx.metal",
        "memory_mb": 16384,
        "inference_budget_mb": 10240,
        "preferred_backends": ["mlx", "litert", "executorch"],
    },
    "android-google-tensor": {
        "name": "Android Google Tensor (Pixel G5/G6 TPU)",
        "system": "android",
        "architecture": "arm64",
        "soc": "googletensorg6",
        "accelerator": "npu",
        "delegate": "litert.google-tensor",
        "memory_mb": 12288,
        "inference_budget_mb": 4096,
        "preferred_backends": ["litert", "executorch"],
    },
    "android-snapdragon-npu": {
        "name": "Android Qualcomm Snapdragon (8 Elite / HTP NPU)",
        "system": "android",
        "architecture": "arm64",
        "soc": "sm8750",
        "accelerator": "npu",
        "delegate": "QnnBackend",
        "memory_mb": 12288,
        "inference_budget_mb": 4096,
        "preferred_backends": ["executorch", "litert"],
    },
    "linux-qcs8275-npu": {
        "name": "Arduino Ventuno Q / Qualcomm QCS8275 (Hexagon HTP v75)",
        "system": "linux",
        "architecture": "arm64",
        "soc": "qualcommqcs8275",
        "accelerator": "npu",
        "delegate": "QnnBackend",
        "memory_mb": 8192,
        "inference_budget_mb": 4608,
        "preferred_backends": ["executorch", "litert"],
    },
}


def _extract_arch_params(config: dict[str, Any]) -> dict[str, Any]:
    text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    layers = text_cfg.get("num_hidden_layers") or config.get("num_hidden_layers") or 0
    heads = text_cfg.get("num_attention_heads") or config.get("num_attention_heads") or 0
    kv_heads = (
        text_cfg.get("num_key_value_heads")
        or config.get("num_key_value_heads")
        or heads
    )
    hidden = text_cfg.get("hidden_size") or config.get("hidden_size") or 0
    head_dim = (
        text_cfg.get("head_dim")
        or config.get("head_dim")
        or ((hidden // heads) if (hidden and heads) else 0)
    )
    max_pos = (
        text_cfg.get("max_position_embeddings")
        or config.get("max_position_embeddings")
        or 4096
    )
    quant = config.get("quantization") or text_cfg.get("quantization") or {}
    bits = quant.get("bits") if isinstance(quant, dict) else None
    # KV cache in fp16: 2 (K and V) * layers * kv_heads * head_dim * 2 bytes
    kv_bytes_per_token = int(2 * layers * kv_heads * head_dim * 2) if (layers and kv_heads and head_dim) else 0
    return {
        "model_type": config.get("model_type") or text_cfg.get("model_type"),
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": int(max_pos),
        "quantization_bits": bits,
        "kv_bytes_per_token": kv_bytes_per_token,
    }


def _extract_sampling_defaults(snapshot: Path, repo_id: str) -> dict[str, Any]:
    gen_path = snapshot / "generation_config.json"
    gen = {}
    if gen_path.is_file():
        try:
            gen = json.loads(gen_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            gen = {}
    tok_cfg_path = snapshot / "tokenizer_config.json"
    tok_cfg = {}
    if tok_cfg_path.is_file():
        try:
            tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            tok_cfg = {}
    template_text = ""
    if (snapshot / "chat_template.jinja").is_file():
        try:
            template_text = (snapshot / "chat_template.jinja").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
    elif isinstance(tok_cfg.get("chat_template"), str):
        template_text = tok_cfg["chat_template"]

    lower_id = repo_id.lower()
    has_chat_template = bool(template_text.strip())
    is_thinking = (
        ("thinking" in lower_id or "think" in lower_id or "deepseek-r1" in lower_id or "deepseek-v4" in lower_id)
        or ("<think>" in template_text and "instruct" not in lower_id)
    )
    temperature = gen.get("temperature")
    if temperature is None:
        temperature = 0.6 if is_thinking else 0.7 if gen.get("do_sample") else 0.0
    top_p = gen.get("top_p")
    top_k = gen.get("top_k")
    return {
        "temperature": float(temperature),
        "top_p": float(top_p) if top_p is not None else None,
        "top_k": int(top_k) if top_k is not None else None,
        "do_sample": bool(gen.get("do_sample", temperature > 0)),
        "is_thinking": is_thinking,
        "has_chat_template": has_chat_template,
        "source": "generation_config.json" if gen_path.is_file() and gen else "model_architecture_default",
    }


def compute_memory_recipe(
    *,
    weight_bytes: int,
    kv_bytes_per_token: int,
    max_position_embeddings: int,
    inference_budget_mb: int,
    is_thinking: bool = False,
) -> dict[str, Any]:
    weight_mb = round(weight_bytes / (1024 ** 2), 1)
    budget_bytes = inference_budget_mb * (1024 ** 2)
    # Reserve 15% overhead above raw weight bytes for runtime buffers and activations.
    required_base_bytes = int(weight_bytes * 1.15)
    remaining_for_kv = budget_bytes - required_base_bytes
    if remaining_for_kv <= 0:
        return {
            "fits_memory": False,
            "weight_mb": weight_mb,
            "inference_budget_mb": inference_budget_mb,
            "kv_mb_per_1k_tokens": round((kv_bytes_per_token * 1024) / (1024 ** 2), 2) if kv_bytes_per_token else None,
            "max_safe_context": 0,
            "recommended_context": 0,
            "recommended_max_output_tokens": 0,
            "reason": f"Requires ~{int(required_base_bytes / (1024 ** 2))} MiB base memory; budget is {inference_budget_mb} MiB",
        }
    if kv_bytes_per_token > 0:
        max_kv_tokens = int(remaining_for_kv // kv_bytes_per_token)
        max_safe_context = min(max_position_embeddings, max_kv_tokens)
        # Keep default interactive KV cache within 25% of total inference budget for low latency.
        comfortable_kv_tokens = int((budget_bytes * 0.25) // kv_bytes_per_token)
        target_ctx = min(max_safe_context, max(2048, comfortable_kv_tokens))
    else:
        max_safe_context = max_position_embeddings
        target_ctx = min(max_position_embeddings, 8192 if inference_budget_mb < 16384 else 32768)

    # Snap recommended_context to clean power-of-two tier <= target_ctx (capped at 65536 by default)
    tiers = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    valid_tiers = [t for t in tiers if t <= target_ctx]
    recommended_context = valid_tiers[-1] if valid_tiers else max(32, min(target_ctx, max_safe_context))
    if recommended_context < 512 and max_safe_context >= 512:
        recommended_context = 512

    base_out = 4096 if is_thinking else 1024
    if inference_budget_mb >= 32768 and is_thinking:
        base_out = 8192
    recommended_max_output = min(base_out, max(128, recommended_context // 2))
    return {
        "fits_memory": recommended_context >= 512,
        "weight_mb": weight_mb,
        "inference_budget_mb": inference_budget_mb,
        "kv_mb_per_1k_tokens": round((kv_bytes_per_token * 1024) / (1024 ** 2), 2) if kv_bytes_per_token else None,
        "max_safe_context": max_safe_context,
        "recommended_context": recommended_context,
        "recommended_max_output_tokens": recommended_max_output,
    }


def _infer_family(repo_id: str) -> str:
    lower = repo_id.lower()
    for prefix, family in (
        ("qwen3.8", "qwen3.8"),
        ("qwen3.6", "qwen3.6"),
        ("qwen3.5", "qwen3.5"),
        ("qwen3-coder", "qwen3-coder"),
        ("qwen3-reranker", "qwen3-reranker"),
        ("qwen3-tts", "qwen3-tts"),
        ("qwen3", "qwen3"),
        ("qwen2.5-coder", "qwen2.5-coder"),
        ("qwen2.5", "qwen2.5"),
        ("deepseek-v4", "deepseek"),
        ("deepseek-r1", "deepseek"),
        ("deepseek", "deepseek"),
        ("functiongemma", "functiongemma"),
        ("gemma-4", "gemma4"),
        ("gemma4", "gemma4"),
        ("gemma-3", "gemma3"),
        ("gemma-2", "gemma2"),
        ("glm-4", "glm4"),
        ("vibethinker", "vibethinker"),
        ("modernbert", "modernbert"),
        ("Kokoro".lower(), "kokoro-tts"),
    ):
        if prefix in lower:
            return family
    return repo_id.split("/")[-1].lower()


def _infer_quant_label(repo_id: str, arch_params: dict[str, Any], filenames: list[str]) -> str:
    bits = arch_params.get("quantization_bits")
    if bits:
        return f"{bits}bit"
    lower = repo_id.lower() + " " + " ".join(f.lower() for f in filenames)
    for token in ("4bit", "3bit", "8bit", "int4", "int8", "q4_k_m", "q8_0", "q4f16", "q4f32", "fp16", "bf16"):
        if token in lower:
            return token
    return "fp16/bf16"


def inspect_hf_cache_dir(model_dir: Path, inference_budget_mb: int) -> dict[str, Any]:
    parts = model_dir.name.split("--", 2)
    repo_id = f"{parts[1]}/{parts[2]}" if len(parts) == 3 else model_dir.name
    family = _infer_family(repo_id)
    snapshots_root = model_dir / "snapshots"
    if not snapshots_root.is_dir():
        return {
            "repo_id": repo_id,
            "family": family,
            "status": "incomplete",
            "reason": "Missing snapshots/ directory (download did not complete)",
            "path": str(model_dir),
        }
    snapshots = sorted([p for p in snapshots_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        return {
            "repo_id": repo_id,
            "family": family,
            "status": "incomplete",
            "reason": "Empty snapshots/ directory (0 downloaded commits)",
            "path": str(model_dir),
        }
    snap = snapshots[0]
    files = [p for p in snap.rglob("*") if p.is_file() or (p.is_symlink() and p.resolve().is_file())]
    rel_names = [p.relative_to(snap).as_posix() for p in files]

    # Check sharded safetensors index integrity if present
    index_file = snap / "model.safetensors.index.json"
    if index_file.is_file():
        try:
            idx = json.loads(index_file.read_text(encoding="utf-8"))
            expected_shards = sorted(set(idx.get("weight_map", {}).values()))
            missing_shards = [s for s in expected_shards if not (snap / s).is_file()]
            if missing_shards:
                return {
                    "repo_id": repo_id,
                    "family": family,
                    "status": "incomplete",
                    "reason": f"Missing {len(missing_shards)}/{len(expected_shards)} safetensor shards: {missing_shards[:3]}",
                    "path": str(snap),
                }
        except (OSError, ValueError) as exc:
            return {
                "repo_id": repo_id,
                "family": family,
                "status": "incomplete",
                "reason": f"Corrupt model.safetensors.index.json: {exc}",
                "path": str(snap),
            }

    weight_exts = (".safetensors", ".gguf", ".litertlm", ".pte", ".tflite", ".onnx", ".bin")
    weight_files = [p for p in files if p.name.lower().endswith(weight_exts)]
    if not weight_files:
        return {
            "repo_id": repo_id,
            "family": family,
            "status": "incomplete",
            "reason": f"No weight files ({', '.join(weight_exts)}) in snapshot; found only {rel_names[:5]}",
            "path": str(snap),
        }

    weight_bytes = sum(p.stat().st_size for p in weight_files)
    config = {}
    if (snap / "config.json").is_file():
        try:
            config = json.loads((snap / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
    arch = _extract_arch_params(config)
    sampling = _extract_sampling_defaults(snap, repo_id)
    quant_label = _infer_quant_label(repo_id, arch, rel_names)

    # Determine backend compatibility
    has_safetensors = any(p.name.endswith(".safetensors") for p in weight_files)
    has_litertlm = any(p.name.endswith(".litertlm") for p in weight_files)
    has_tflite = any(p.name.endswith(".tflite") for p in weight_files)
    has_gguf = any(p.name.endswith(".gguf") for p in weight_files)
    has_onnx = any(p.name.endswith(".onnx") for p in weight_files)
    has_tokenizer = (snap / "tokenizer.json").is_file() or (snap / "tokenizer.model").is_file()

    if has_litertlm:
        backend = "litert"
        format_kind = "litert-lm"
        runnable = True
    elif has_tflite:
        backend = "litert"
        format_kind = "tflite"
        runnable = True
    elif has_safetensors and (snap / "config.json").is_file() and has_tokenizer:
        # Check if this is a causal LM / chat model supported by mlx-lm with a valid chat template
        architectures = config.get("architectures") or []
        is_causal_lm = any(a.endswith(("ForCausalLM", "ForConditionalGeneration", "LMHeadModel")) for a in architectures)
        is_reranker_or_tts = any(x in repo_id.lower() for x in ("reranker", "tts", "whisper", "bert", "birefnet", "rmbg", "dinov3", "sam2", "colqwen"))
        has_template = sampling.get("has_chat_template", False)
        if is_causal_lm and not is_reranker_or_tts and has_template:
            backend = "mlx"
            format_kind = "mlx-safetensors" if repo_id.lower().startswith("mlx-community/") or arch.get("quantization_bits") else "hf-safetensors"
            runnable = True
        else:
            backend = "safetensors"
            format_kind = "base-checkpoint-no-chat-template" if (is_causal_lm and not has_template) else "safetensors-specialized"
            runnable = False
    elif has_gguf:
        backend = "local_server"
        format_kind = "gguf"
        runnable = False
    elif has_onnx:
        backend = "onnx"
        format_kind = "onnx"
        runnable = False
    else:
        backend = "unknown"
        format_kind = "weights"
        runnable = False

    memory_recipe = compute_memory_recipe(
        weight_bytes=weight_bytes,
        kv_bytes_per_token=arch["kv_bytes_per_token"],
        max_position_embeddings=arch["max_position_embeddings"],
        inference_budget_mb=inference_budget_mb,
        is_thinking=sampling["is_thinking"],
    )
    return {
        "repo_id": repo_id,
        "family": family,
        "status": "complete",
        "runnable": runnable,
        "backend": backend,
        "format": format_kind,
        "quantization": quant_label,
        "snapshot_path": str(snap),
        "architecture": arch,
        "sampling_defaults": sampling,
        "memory_recipe": memory_recipe,
    }


def scan_local_models(
    *,
    hf_cache_dir: Path | None = None,
    work_dirs: list[Path] | None = None,
    device: dict[str, Any] | None = None,
) -> dict[str, Any]:
    host = device or discover()
    budget_mb = host.get("inference_budget_mb") or host.get("memory_mb") or 8192
    hub = hf_cache_dir if hf_cache_dir is not None else (Path.home() / ".cache/huggingface/hub")

    runnable_models = []
    other_complete_models = []
    incomplete_downloads = []

    if hub.is_dir():
        for model_dir in sorted(hub.glob("models--*")):
            if not model_dir.is_dir():
                continue
            info = inspect_hf_cache_dir(model_dir, budget_mb)
            if info["status"] != "complete":
                incomplete_downloads.append(info)
            elif info.get("runnable"):
                runnable_models.append(info)
            else:
                other_complete_models.append(info)

    # Also scan local LocalKit packages (.work/*, .phaseone/packages/*)
    search_roots = work_dirs if work_dirs is not None else [Path(".work"), Path(".phaseone/packages")]
    seen_digests = set()
    packaged_models = []
    for root in search_roots:
        if not root.is_dir():
            continue
        candidates = [root] if (root / MANIFEST).is_file() else [p.parent for p in root.glob(f"*/{MANIFEST}")]
        for pkg_dir in sorted(candidates):
            try:
                pkg = load_package(pkg_dir, verify=False)
            except RuntimeFault:
                continue
            if pkg.digest in seen_digests:
                continue
            seen_digests.add(pkg.digest)
            for model_name, model_def in pkg.manifest.models.items():
                variants_summary = []
                for v in model_def.variants:
                    reasons = compatibility(v, host)
                    art = pkg.manifest.artifacts.get(v.artifact)
                    size_mb = round(art.size / (1024 ** 2), 1) if art else 0.0
                    variants_summary.append({
                        "id": v.id,
                        "backend": v.backend,
                        "accelerator": v.target.accelerator,
                        "delegate": v.target.delegate,
                        "soc": v.target.soc,
                        "context_length": v.chat.context_length if v.chat else None,
                        "weight_mb": size_mb,
                        "compatible_on_host": len(reasons) == 0,
                        "rejection_reasons": reasons,
                    })
                family = _infer_family(pkg.manifest.name + "-" + model_name)
                if any("gemma4" in a.path.lower() for a in pkg.manifest.artifacts.values()):
                    family = "gemma4"
                elif model_def.task == "image.classification":
                    family = "vision"
                gen_meta = pkg.manifest.metadata.get("generation_defaults", {}).get(model_name, {})
                packaged_models.append({
                    "package_name": pkg.manifest.name,
                    "package_path": str(pkg.root),
                    "package_digest": pkg.digest,
                    "model": model_name,
                    "family": family,
                    "task": model_def.task,
                    "generation_defaults": gen_meta,
                    "variants": variants_summary,
                })

    return {
        "host": {
            "soc": host.get("soc"),
            "system": host.get("system"),
            "architecture": host.get("architecture"),
            "memory_mb": host.get("memory_mb"),
            "available_memory_mb": host.get("available_memory_mb"),
            "inference_budget_mb": budget_mb,
        },
        "runnable_hf_models": runnable_models,
        "packaged_models": packaged_models,
        "other_cached_models": other_complete_models,
        "incomplete_hf_downloads": incomplete_downloads,
    }


def _score_hf_candidate(item: dict[str, Any], prefer_precision: str | None = None) -> tuple:
    recipe = item["memory_recipe"]
    fits = 1 if recipe["fits_memory"] else 0
    quant = item["quantization"]
    is_mlx_quant = 1 if quant in ("4bit", "8bit") else 0
    if prefer_precision == "full":
        is_mlx_quant = 1 if quant == "fp16/bf16" else 0
    # Prefer models that fit memory, match preferred quantization, and have smaller footprint for speed
    return (fits, is_mlx_quant, -recipe["weight_mb"])


def resolve_model(
    query: str | Path,
    *,
    state_dir: Path = Path(".phaseone"),
    device: dict[str, Any] | None = None,
    prefer_precision: str | None = None,
    include_all_family_variants: bool = False,
) -> dict[str, Any]:
    """Resolve a model name, family alias, or package path into a ready-to-run LocalKit package + recipe."""
    path_candidate = Path(query)
    if path_candidate.exists() and (
        path_candidate.is_dir() and (path_candidate / MANIFEST).is_file()
        or path_candidate.name == MANIFEST
        or path_candidate.suffix == ".phaseone"
    ):
        pkg = load_package(path_candidate, verify=False) if path_candidate.suffix != ".phaseone" else None
        models = list(pkg.manifest.models) if pkg else ["model"]
        gen_defaults = (pkg.manifest.metadata.get("generation_defaults", {}) if pkg else {})
        model_name = models[0]
        return {
            "package_path": path_candidate,
            "model": model_name,
            "generation_defaults": gen_defaults.get(model_name, {}),
            "resolved_from": "package_path",
        }

    host = device or discover()
    inventory = scan_local_models(device=host)
    q = str(query).strip().lower()

    # 1. Check packaged LocalKit models first if exact model/family/package match (and not requesting multi-variant family staging)
    if not include_all_family_variants:
        for pkg_entry in inventory["packaged_models"]:
            if q in (
                pkg_entry["package_name"].lower(),
                pkg_entry["model"].lower(),
                pkg_entry["family"].lower(),
                f"{pkg_entry['package_name']}:{pkg_entry['model']}".lower(),
            ):
                compatible = [v for v in pkg_entry["variants"] if v["compatible_on_host"]]
                if compatible:
                    gen_defaults = pkg_entry.get("generation_defaults") or {
                        "temperature": 0.0,
                        "max_output_tokens": 512,
                    }
                    return {
                        "package_path": Path(pkg_entry["package_path"]),
                        "model": pkg_entry["model"],
                        "repo_id": gen_defaults.get("repo_id"),
                        "family": pkg_entry["family"],
                        "quantization": gen_defaults.get("quantization"),
                        "generation_defaults": gen_defaults,
                        "resolved_from": f"localkit_package:{pkg_entry['package_name']}",
                        "selected_variant": compatible[0],
                        "all_variants": pkg_entry["variants"],
                    }

    # 2. Match runnable HF models by exact repo_id or family first, then substring
    exact_matches = [
        item for item in inventory["runnable_hf_models"]
        if q == item["repo_id"].lower()
        or q == item["family"].lower()
        or q == item["repo_id"].split("/")[-1].lower()
    ]
    matches = exact_matches or [
        item for item in inventory["runnable_hf_models"]
        if q in item["repo_id"].lower()
    ]
    if not matches and include_all_family_variants:
        # Fall back to packaged LocalKit models (e.g., gemma4 or vision)
        for pkg_entry in inventory["packaged_models"]:
            if q in (pkg_entry["package_name"].lower(), pkg_entry["model"].lower(), pkg_entry["family"].lower()):
                compatible = [v for v in pkg_entry["variants"] if v["compatible_on_host"]]
                if compatible:
                    gen_defaults = pkg_entry.get("generation_defaults") or {"temperature": 0.0, "max_output_tokens": 512}
                    return {
                        "package_path": Path(pkg_entry["package_path"]),
                        "model": pkg_entry["model"],
                        "repo_id": gen_defaults.get("repo_id"),
                        "family": pkg_entry["family"],
                        "generation_defaults": gen_defaults,
                        "resolved_from": f"localkit_package:{pkg_entry['package_name']}",
                        "selected_variant": compatible[0],
                        "all_variants": pkg_entry["variants"],
                    }

    if not matches:
        # Check if user asked for a model that exists in incomplete_hf_downloads or other_cached_models
        broken = [
            item for item in inventory["incomplete_hf_downloads"]
            if q in item["repo_id"].lower() or q == item["family"].lower()
        ]
        unsupported = [
            item for item in inventory["other_cached_models"]
            if q in item["repo_id"].lower() or q == item["family"].lower()
        ]
        available_families = sorted({
            *(m["family"] for m in inventory["runnable_hf_models"]),
            *(p["family"] for p in inventory["packaged_models"]),
        })
        if broken:
            raise RuntimeFault(
                "incomplete_model_cache",
                f"Model '{query}' matched {len(broken)} incomplete Hugging Face cache entry(s) on disk; weights were not fully downloaded.",
                details={
                    "incomplete_entries": broken,
                    "runnable_families_on_disk": available_families,
                },
            )
        if unsupported:
            raise RuntimeFault(
                "unsupported_model_format",
                f"Model '{query}' is cached in format '{unsupported[0]['format']}' ({unsupported[0]['repo_id']}), which requires an external runner ({unsupported[0]['backend']}).",
                details={
                    "matched_cached_models": unsupported,
                    "runnable_families_on_disk": available_families,
                },
            )
        raise RuntimeFault(
            "model_not_found",
            f"No local model or package matched '{query}'. Available families on this machine: {available_families}",
            details={"runnable_families_on_disk": available_families},
        )

    matches.sort(key=lambda m: _score_hf_candidate(m, prefer_precision=prefer_precision), reverse=True)
    selected = matches if include_all_family_variants else [matches[0]]
    primary = selected[0]
    if not primary["memory_recipe"]["fits_memory"]:
        raise RuntimeFault(
            "insufficient_memory",
            primary["memory_recipe"].get("reason", "Model exceeds host inference memory budget"),
            details={"model": primary["repo_id"], "recipe": primary["memory_recipe"]},
        )

    slug = re.sub(r"[^a-z0-9_.-]", "-", (primary["family"] if include_all_family_variants else primary["repo_id"]).lower()).strip("-")
    dest = state_dir.resolve() / "resolved" / slug
    model_key = re.sub(r"[^A-Za-z0-9_.-]", "-", primary["family"])
    if not model_key[0].isalpha():
        model_key = "m-" + model_key

    # Materialize zero-copy package with all selected variants and store auto-computed generation recipe
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(selected):
        quant_slug = re.sub(r"[^A-Za-z0-9_.-]", "-", item["quantization"])
        variant_id = f"mlx-{quant_slug}" if len(selected) > 1 else "mlx-gpu"
        ctx = item["memory_recipe"]["recommended_context"]
        import_mlx(
            Path(item["snapshot_path"]),
            dest,
            name=f"localkit-{slug[:40]}",
            version="1.0.0",
            model=model_key,
            variant_id=variant_id,
            chat={"context_length": ctx, "tool_protocol": "json"},
        )
        # Set priority so primary variant is highest priority
        raw = read_json((dest / MANIFEST).read_bytes())
        for v in raw["models"][model_key]["variants"]:
            if v["id"] == variant_id:
                v["priority"] = 100 - idx * 10
        raw.setdefault("metadata", {}).setdefault("generation_defaults", {})[model_key] = {
            **item["sampling_defaults"],
            "max_output_tokens": item["memory_recipe"]["recommended_max_output_tokens"],
            "recommended_context": ctx,
            "max_safe_context": item["memory_recipe"]["max_safe_context"],
            "weight_mb": item["memory_recipe"]["weight_mb"],
            "kv_mb_per_1k_tokens": item["memory_recipe"]["kv_mb_per_1k_tokens"],
            "repo_id": item["repo_id"],
            "quantization": item["quantization"],
        }
        atomic_json(dest / MANIFEST, raw)

    return {
        "package_path": dest,
        "model": model_key,
        "repo_id": primary["repo_id"],
        "family": primary["family"],
        "quantization": primary["quantization"],
        "generation_defaults": {
            **primary["sampling_defaults"],
            "max_output_tokens": primary["memory_recipe"]["recommended_max_output_tokens"],
            "recommended_context": primary["memory_recipe"]["recommended_context"],
            "max_safe_context": primary["memory_recipe"]["max_safe_context"],
            "weight_mb": primary["memory_recipe"]["weight_mb"],
            "kv_mb_per_1k_tokens": primary["memory_recipe"]["kv_mb_per_1k_tokens"],
        },
        "resolved_from": f"hf_cache:{primary['repo_id']}",
        "variants_staged": [m["repo_id"] for m in selected],
    }


def detect_project_targets(project_dir: Path) -> list[str]:
    targets = ["macos-metal-pro", "macos-metal-consumer"]
    if (
        (project_dir / "sdk/android").is_dir()
        or (project_dir / "android").is_dir()
        or list(project_dir.glob("**/AndroidManifest.xml"))
    ):
        targets.extend(["android-google-tensor", "android-snapdragon-npu"])
    if (project_dir / "scripts/deploy_runtime.py").is_file() or (project_dir / "docs/ventuno.md").is_file():
        targets.append("linux-qcs8275-npu")
    return targets


def build_cross_device_lock(
    query: str,
    *,
    project_dir: Path = Path("."),
    state_dir: Path = Path(".phaseone"),
    device: dict[str, Any] | None = None,
) -> dict[str, Any]:
    host = device or discover()
    resolved = resolve_model(query, state_dir=state_dir, device=host, include_all_family_variants=True)
    pkg = load_package(resolved["package_path"], verify=False)
    model_name = resolved["model"]
    family = resolved.get("family") or str(query)
    inventory = scan_local_models(device=host)

    # Gather all known local artifacts for this family across HF cache and LocalKit packages
    hf_matches = [m for m in inventory["runnable_hf_models"] if m["family"] == family]
    if not hf_matches:
        hf_matches = [m for m in inventory["runnable_hf_models"] if query.lower() in m["repo_id"].lower()]
    pkg_matches = [p for p in inventory["packaged_models"] if p["family"] == family]

    target_ids = detect_project_targets(project_dir)
    matrix = {}
    for target_id in target_ids:
        profile = CONSUMER_PROFILES[target_id]
        budget = profile["inference_budget_mb"]
        if profile["system"] == "darwin":
            # Evaluate HF MLX candidates against this Mac profile's memory budget
            candidates = []
            for m in hf_matches:
                arch = m["architecture"]
                weight_bytes = int(m["memory_recipe"]["weight_mb"] * (1024 ** 2))
                rec = compute_memory_recipe(
                    weight_bytes=weight_bytes,
                    kv_bytes_per_token=arch["kv_bytes_per_token"],
                    max_position_embeddings=arch["max_position_embeddings"],
                    inference_budget_mb=budget,
                    is_thinking=m["sampling_defaults"]["is_thinking"],
                )
                if rec["fits_memory"]:
                    candidates.append({
                        "repo_id": m["repo_id"],
                        "backend": "mlx",
                        "accelerator": "gpu",
                        "delegate": "mlx.metal",
                        "quantization": m["quantization"],
                        "weight_mb": rec["weight_mb"],
                        "recommended_context": rec["recommended_context"],
                        "max_safe_context": rec["max_safe_context"],
                        "recommended_max_output_tokens": rec["recommended_max_output_tokens"],
                        "sampling": m["sampling_defaults"],
                        "status": "verified_local_artifact",
                    })
            if candidates:
                # For consumer 16GB prefer 4bit; for Pro/Max 64GB+ allow 4bit (fastest) and list full precision
                candidates.sort(key=lambda c: (1 if c["quantization"] == "4bit" else 0, -c["weight_mb"]), reverse=True)
                matrix[target_id] = {"profile": profile["name"], "selected": candidates[0], "alternatives": candidates[1:]}
                continue

        # Check packaged LocalKit variants for Android / Linux / cross-platform
        matched_variants = []
        for p in pkg_matches:
            for v in p["variants"]:
                if v["weight_mb"] <= budget:
                    if profile["system"] == "linux" and v.get("delegate") == "QnnBackend":
                        matched_variants.append({**v, "package": p["package_path"], "status": "verified_local_artifact"})
                    elif v["backend"] in ("litert", "executorch") and v["accelerator"] == "cpu":
                        matched_variants.append({**v, "package": p["package_path"], "status": "verified_local_fallback"})
        if matched_variants:
            matrix[target_id] = {
                "profile": profile["name"],
                "selected": matched_variants[0],
                "alternatives": matched_variants[1:],
            }
        else:
            # Report exact required artifact specification for this target without fabricating an artifact
            matrix[target_id] = {
                "profile": profile["name"],
                "selected": None,
                "status": "artifact_export_or_download_required",
                "required_contract": {
                    "family": family,
                    "system": profile["system"],
                    "architecture": profile["architecture"],
                    "accelerator": profile["accelerator"],
                    "delegate": profile["delegate"],
                    "max_weight_mb": int(budget * 0.65),
                    "recommended_quantization": "int4",
                    "recommended_context": 4096 if budget <= 4608 else 8192,
                },
            }

    return {
        "model_query": query,
        "resolved_family": family,
        "local_package": str(resolved["package_path"]),
        "package_digest": pkg.digest,
        "host_device": {
            "soc": host.get("soc"),
            "memory_mb": host.get("memory_mb"),
            "inference_budget_mb": host.get("inference_budget_mb"),
        },
        "host_resolution": {
            "repo_id": resolved.get("repo_id"),
            "model": model_name,
            "generation_defaults": resolved["generation_defaults"],
        },
        "target_matrix": matrix,
    }
