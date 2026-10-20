"""Tests for hardware discovery, KV-cache physics, incomplete-cache diagnostics, and OpenAI API compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from phaseone.runtime import Runtime, RuntimeFault
from phaseone.runtime.api import create_app
from phaseone.runtime.device import discover
from phaseone.runtime.resolver import (
    build_cross_device_lock,
    compute_memory_recipe,
    resolve_model,
    scan_local_models,
)


def _create_hf_snapshot(
    hub: Path,
    repo_id: str,
    *,
    layers: int = 36,
    heads: int = 32,
    kv_heads: int = 8,
    head_dim: int = 128,
    max_pos: int = 262144,
    bits: int | None = 4,
    weight_bytes: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    include_template: bool = True,
) -> Path:
    folder_name = "models--" + repo_id.replace("/", "--")
    snap = hub / folder_name / "snapshots" / "abc123456789"
    snap.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": heads * head_dim,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": max_pos,
    }
    if bits is not None:
        config["quantization"] = {"group_size": 64, "bits": bits}
    (snap / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (snap / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k}),
        encoding="utf-8",
    )
    tok_cfg = {"bos_token": "<|endoftext|>", "eos_token": "<|im_end|>"}
    if include_template:
        tok_cfg["chat_template"] = "{% for m in messages %}{{ m.content }}{% endfor %}"
    (snap / "tokenizer_config.json").write_text(json.dumps(tok_cfg), encoding="utf-8")
    (snap / "tokenizer.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"\x00" * weight_bytes)
    return snap


def test_discover_reports_live_memory_budget_and_harnesses():
    host = discover()
    assert host["system"] in ("darwin", "linux", "windows", "android")
    assert host["memory_mb"] and host["memory_mb"] > 0
    assert host["inference_budget_mb"] and host["inference_budget_mb"] <= host["memory_mb"]
    assert "harnesses" in host
    if host["system"] == "darwin":
        assert host["soc"] is not None
        assert host["available_memory_mb"] is not None and host["available_memory_mb"] > 0


def test_kv_cache_physics_scales_context_to_device_memory_budget():
    # Qwen3-4B architecture: 36 layers, 8 kv_heads, 128 head_dim -> 147,456 bytes/token (~144 KiB/token)
    kv_bytes = 2 * 36 * 8 * 128 * 2
    assert kv_bytes == 147456

    # 1. On a 64GB+ Pro/Max machine (88,000 MiB budget) with 4-bit weights (2,158 MiB):
    pro = compute_memory_recipe(
        weight_bytes=2158 * 1024**2,
        kv_bytes_per_token=kv_bytes,
        max_position_embeddings=262144,
        inference_budget_mb=88000,
        is_thinking=False,
    )
    assert pro["fits_memory"] is True
    assert pro["max_safe_context"] == 262144
    assert pro["recommended_context"] == 65536
    assert pro["recommended_max_output_tokens"] == 1024

    # 2. On a 16GB consumer laptop (10,240 MiB budget) with bf16 weights (7,672 MiB):
    # 7672 * 1.15 = 8822.8 MiB base -> ~1,417 MiB remaining for KV -> ~10,075 max safe tokens
    consumer_bf16 = compute_memory_recipe(
        weight_bytes=7672 * 1024**2,
        kv_bytes_per_token=kv_bytes,
        max_position_embeddings=262144,
        inference_budget_mb=10240,
        is_thinking=False,
    )
    assert consumer_bf16["fits_memory"] is True
    assert 9500 <= consumer_bf16["max_safe_context"] <= 10500
    assert consumer_bf16["recommended_context"] == 8192

    # 3. On a 4GB mobile budget where 7.6GB bf16 weights cannot fit:
    oom = compute_memory_recipe(
        weight_bytes=7672 * 1024**2,
        kv_bytes_per_token=kv_bytes,
        max_position_embeddings=262144,
        inference_budget_mb=4096,
        is_thinking=False,
    )
    assert oom["fits_memory"] is False
    assert oom["recommended_context"] == 0


def test_scanner_detects_incomplete_downloads_and_base_checkpoints(tmp_path):
    hub = tmp_path / "hub"
    # 1. Valid 4-bit MLX instruct model
    _create_hf_snapshot(hub, "mlx-community/Qwen3-4B-Instruct-2507-4bit")

    # 2. Base checkpoint without chat_template (must not be marked runnable for chat)
    _create_hf_snapshot(hub, "Qwen/Qwen3.5-0.8B-Base", bits=None, include_template=False)

    # 3. Empty cache directory missing snapshots/ (like interrupted DeepSeek download)
    (hub / "models--mlx-community--DeepSeek-V4-Flash-4bit" / "blobs").mkdir(parents=True)

    # 4. Sharded model missing shards listed in model.safetensors.index.json
    sharded = hub / "models--Qwen--Qwen3.8-Flash-Next" / "snapshots" / "111"
    sharded.mkdir(parents=True)
    (sharded / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": "model-00001-of-00002.safetensors", "layer.1": "model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )

    # 5. README-only snapshot with no weights
    readme_only = hub / "models--litert-community--gemma-4-E4B-it-litert-lm" / "snapshots" / "222"
    readme_only.mkdir(parents=True)
    (readme_only / "README.md").write_text("# Gemma 4", encoding="utf-8")

    report = scan_local_models(hf_cache_dir=hub, work_dirs=[])
    runnable_ids = [m["repo_id"] for m in report["runnable_hf_models"]]
    assert runnable_ids == ["mlx-community/Qwen3-4B-Instruct-2507-4bit"]

    other_ids = [m["repo_id"] for m in report["other_cached_models"]]
    assert other_ids == ["Qwen/Qwen3.5-0.8B-Base"]

    incomplete = {m["repo_id"]: m["reason"] for m in report["incomplete_hf_downloads"]}
    assert "mlx-community/DeepSeek-V4-Flash-4bit" in incomplete
    assert "Missing snapshots/" in incomplete["mlx-community/DeepSeek-V4-Flash-4bit"]
    assert "Qwen/Qwen3.8-Flash-Next" in incomplete
    assert "Missing 2/2 safetensor shards" in incomplete["Qwen/Qwen3.8-Flash-Next"]
    assert "litert-community/gemma-4-E4B-it-litert-lm" in incomplete
    assert "No weight files" in incomplete["litert-community/gemma-4-E4B-it-litert-lm"]


def test_resolve_model_stages_zero_copy_hardlinks_and_preserves_sampling(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    snap = _create_hf_snapshot(
        hub,
        "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )
    monkeypatch.setattr(
        "phaseone.runtime.resolver.scan_local_models",
        lambda **kw: scan_local_models(hf_cache_dir=hub, work_dirs=[], **kw),
    )

    state_dir = tmp_path / "state"
    resolved = resolve_model("qwen3", state_dir=state_dir)
    assert resolved["repo_id"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert resolved["generation_defaults"]["temperature"] == 0.7
    assert resolved["generation_defaults"]["top_p"] == 0.8
    assert resolved["generation_defaults"]["top_k"] == 20

    # Verify zero-copy hardlink: staged weight file shares the exact inode with the HF snapshot
    staged_weights = list(Path(resolved["package_path"]).rglob("model.safetensors"))
    assert len(staged_weights) == 1
    assert staged_weights[0].stat().st_ino == (snap / "model.safetensors").stat().st_ino


def test_resolve_model_reports_incomplete_cache_honestly(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    (hub / "models--mlx-community--DeepSeek-V4-Flash-4bit" / "blobs").mkdir(parents=True)
    monkeypatch.setattr(
        "phaseone.runtime.resolver.scan_local_models",
        lambda **kw: scan_local_models(hf_cache_dir=hub, work_dirs=[], **kw),
    )

    with pytest.raises(RuntimeFault) as exc:
        resolve_model("deepseek", state_dir=tmp_path / "state")
    assert exc.value.code == "incomplete_model_cache"
    assert exc.value.details["incomplete_entries"][0]["repo_id"] == "mlx-community/DeepSeek-V4-Flash-4bit"


def test_build_cross_device_lock_creates_matrix(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    _create_hf_snapshot(hub, "mlx-community/Qwen3-4B-Instruct-2507-4bit", bits=4, weight_bytes=2048)
    _create_hf_snapshot(hub, "Qwen/Qwen3-4B-Instruct-2507", bits=None, weight_bytes=8192)
    monkeypatch.setattr(
        "phaseone.runtime.resolver.scan_local_models",
        lambda **kw: scan_local_models(hf_cache_dir=hub, work_dirs=[], **kw),
    )

    project = tmp_path / "app"
    (project / "android").mkdir(parents=True)
    lock = build_cross_device_lock("qwen3", project_dir=project, state_dir=tmp_path / "state")
    assert lock["resolved_family"] == "qwen3"
    assert "macos-metal-pro" in lock["target_matrix"]
    assert "macos-metal-consumer" in lock["target_matrix"]
    assert "android-google-tensor" in lock["target_matrix"]
    assert "android-snapdragon-npu" in lock["target_matrix"]
    assert lock["target_matrix"]["macos-metal-consumer"]["selected"]["quantization"] == "4bit"


def test_openai_compatible_models_and_chat_completions_endpoints(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    _create_hf_snapshot(
        hub,
        "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )
    monkeypatch.setattr(
        "phaseone.runtime.resolver.scan_local_models",
        lambda **kw: scan_local_models(hf_cache_dir=hub, work_dirs=[], **kw),
    )
    resolved = resolve_model("qwen3", state_dir=tmp_path / "state")

    token = "t" * 32
    with Runtime(tmp_path / "host") as runtime:
        app = runtime.load(resolved["package_path"])
        captured_gen = {}

        def fake_chat(model, messages, *, generation=None, **kwargs):
            captured_gen["model"] = model
            captured_gen["generation"] = generation.model_dump(mode="json")
            return {
                "type": "done",
                "message": {"role": "assistant", "content": "Hello from OpenAI endpoint", "tool_calls": []},
                "finish_reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 4},
                "execution": {"backend": "mlx", "variant": "mlx-gpu", "execution_verified": True},
                "receipt": {"seq": 1, "hash": "0" * 64},
            }

        monkeypatch.setattr(app, "chat", fake_chat)
        monkeypatch.setattr(runtime, "application", lambda digest: app)

        client = TestClient(create_app(runtime, token=token))
        headers = {"Authorization": f"Bearer {token}"}

        models_resp = client.get("/v1/models", headers=headers)
        assert models_resp.status_code == 200
        models_data = models_resp.json()["data"]
        assert any(m["id"] == "qwen3" for m in models_data)

        chat_resp = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "qwen3", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert chat_resp.status_code == 200
        body = chat_resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Hello from OpenAI endpoint"
        assert body["usage"]["total_tokens"] == 14
        # Verify package generation_defaults were automatically applied when caller omitted temperature & max_tokens
        assert captured_gen["generation"]["temperature"] == 0.7
        assert captured_gen["generation"]["top_p"] == 0.8
        assert captured_gen["generation"]["top_k"] == 20
        assert captured_gen["generation"]["max_output_tokens"] >= 1024
