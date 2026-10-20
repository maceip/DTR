"""Host capability discovery and explainable model selection."""

import hashlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from .errors import RuntimeFault
from .schema import Model, Variant

BACKEND_DISTRIBUTIONS = {
    ("executorch", "program"): "executorch",
    ("executorch", "llm"): "executorch",
    ("litert", "compiled"): "ai-edge-litert",
    ("litert", "interpreter"): "ai-edge-litert",
    ("litert", "llm"): "litert-lm-api",
    ("mlx", "llm"): "mlx-lm",
    # The model runs in a server the app does not ship; availability is checked at load.
    ("local_server", "llm"): None,
}


def canonical_soc(value: str | None) -> str | None:
    """Exact aliases only. MT6991 must never match MT69910 (or an unknown chip)."""
    if not value:
        return None
    key = re.sub(r"[^a-z0-9]", "", value.lower())
    return {
        "tensorg5": "googletensorg5", "tensorg6": "googletensorg6",
        "mt6991": "mediatekmt6991", "mt6989": "mediatekmt6989",
        "qcs8275": "qualcommqcs8275",
    }.get(key, key)


def canonical_arch(value: str) -> str:
    return {"aarch64": "arm64", "arm64-v8a": "arm64", "amd64": "x86_64"}.get(
        value.lower(), value.lower(),
    )


def runtime_build_id() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _darwin_memory_and_soc() -> tuple[str | None, int | None]:
    soc = None
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            soc = out.stdout.strip()
    except OSError:
        pass
    available_mb = None
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2, check=False)
        if vm.returncode == 0:
            page_match = re.search(r"page size of (\d+) bytes", vm.stdout)
            page_size = int(page_match.group(1)) if page_match else 16384
            reclaimable_pages = 0
            for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                match = re.search(rf"^{re.escape(key)}:\s+(\d+)\.", vm.stdout, re.MULTILINE)
                if match:
                    reclaimable_pages += int(match.group(1))
            if reclaimable_pages > 0:
                available_mb = (reclaimable_pages * page_size) // (1024 ** 2)
    except OSError:
        pass
    return soc, available_mb


def _linux_available_memory_mb() -> int | None:
    try:
        text = Path("/proc/meminfo").read_text()
        match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, re.MULTILINE)
        if match:
            return int(match.group(1)) // 1024
    except OSError:
        pass
    return None


def _discover_adb_devices() -> list[dict]:
    adb = shutil.which("adb")
    if not adb:
        return []
    try:
        out = subprocess.run([adb, "devices", "-l"], capture_output=True, text=True, timeout=2, check=False)
        if out.returncode != 0:
            return []
        devices = []
        for line in out.stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                meta = {"serial": parts[0]}
                for item in parts[2:]:
                    if ":" in item:
                        k, v = item.split(":", 1)
                        meta[k] = v
                devices.append(meta)
        return devices
    except (OSError, subprocess.TimeoutExpired):
        return []


def discover() -> dict:
    system = platform.system().lower()
    if system == "linux" and (os.environ.get("ANDROID_ROOT") or Path("/system/build.prop").exists()):
        system = "android"
    arch = {"aarch64": "arm64", "amd64": "x86_64"}.get(
        platform.machine().lower(), platform.machine().lower(),
    )
    versions = {}
    for distribution in sorted({*filter(None, BACKEND_DISTRIBUTIONS.values()), "transformers"}):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    try:
        memory_mb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 1024**2
    except (ValueError, AttributeError, OSError):
        memory_mb = None
    soc = None
    available_memory_mb = None
    if system == "darwin":
        soc, available_memory_mb = _darwin_memory_and_soc()
    else:
        available_memory_mb = _linux_available_memory_mb()
        for name in ("/sys/devices/soc0/machine", "/sys/devices/soc0/soc_id",
                     "/proc/device-tree/compatible"):
            try:
                raw = Path(name).read_text().replace("\x00", " ").strip()
                if raw:
                    soc = raw.lower()
                    break
            except (OSError, UnicodeError):
                pass
    components = {}
    if versions.get("executorch"):
        components["executorch.xnnpack"] = versions["executorch"]
    from .qnn_host import qnn_installation

    qnn = qnn_installation() if system == "linux" and arch == "arm64" else None
    if qnn:
        _, profile = qnn
        components["executorch.qnn_runner"] = profile["executorch_version"]
        components["qnn.htp"] = profile["qnn_version"]
    accelerator_info = None
    if system == "darwin" and arch == "arm64":
        try:
            components["mlx.metal"] = importlib.metadata.version("mlx-metal")
        except importlib.metadata.PackageNotFoundError:
            pass
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
                working_set_mb = int(info.get("max_recommended_working_set_size", 0)) // (1024 ** 2)
                max_buffer_mb = int(info.get("max_buffer_length", 0)) // (1024 ** 2)
                accelerator_info = {
                    "type": "metal_gpu",
                    "device_name": info.get("device_name") or soc,
                    "architecture": info.get("architecture"),
                    "max_recommended_working_set_mb": working_set_mb or None,
                    "max_buffer_length_mb": max_buffer_mb or None,
                }
                if not soc and info.get("device_name"):
                    soc = str(info["device_name"])
        except Exception:
            pass
    base_available = available_memory_mb or memory_mb
    inference_budget_mb = int(base_available * (0.85 if system == "darwin" else 0.80)) if base_available else None
    if accelerator_info and accelerator_info.get("max_recommended_working_set_mb") and inference_budget_mb:
        inference_budget_mb = min(inference_budget_mb, accelerator_info["max_recommended_working_set_mb"])
    harnesses = {
        "mlx": versions.get("mlx-lm") or ("cli:" + shutil.which("mlx_lm.generate") if shutil.which("mlx_lm.generate") else None),
        "litert_lm": versions.get("litert-lm-api"),
        "litert": versions.get("ai-edge-litert"),
        "executorch": versions.get("executorch"),
        "qnn_runner": components.get("executorch.qnn_runner"),
        "llama_server": shutil.which("llama-server"),
        "ollama": shutil.which("ollama"),
    }
    return {"system": system, "architecture": arch, "os_version": platform.release(),
            "memory_mb": memory_mb, "available_memory_mb": available_memory_mb,
            "inference_budget_mb": inference_budget_mb, "soc": soc,
            "accelerator": accelerator_info, "runtimes": versions,
            "components": components, "harnesses": harnesses,
            "connected_devices": _discover_adb_devices(),
            "phaseone_version": "0.3.0",
            "runtime_build": runtime_build_id(),
            "python": platform.python_version(), "location": "native_host"}


def compatibility(variant: Variant, host: dict) -> list[str]:
    reasons = []
    target = variant.target
    if target.systems and host["system"] not in target.systems:
        reasons.append(f"Requires OS {target.systems}; host is {host['system']}")
    architectures = [canonical_arch(x) for x in target.architectures]
    if architectures and canonical_arch(host["architecture"]) not in architectures:
        reasons.append(f"Requires architecture {architectures}; host is {host['architecture']}")
    if target.soc and canonical_soc(target.soc) != canonical_soc(host.get("soc")):
        reasons.append(f"Requires detected SoC {target.soc}; detected {host.get('soc') or 'unknown'}")
    distribution = BACKEND_DISTRIBUTIONS[(variant.backend, variant.engine)]
    qnn_runner = bool(variant.chat and variant.chat.runner == "qnn")
    installed = (host.get("components", {}).get("executorch.qnn_runner") if qnn_runner
                 else host["runtimes"].get(distribution) if distribution else None)
    if qnn_runner and (host["system"] != "linux" or canonical_arch(host["architecture"]) != "arm64"):
        reasons.append("The installed QNN runner requires Linux ARM64")
    if distribution and not installed:
        reasons.append("Install the LocalKit QNN host runtime" if qnn_runner
                       else f"Install {distribution} for {variant.backend}/{variant.engine}")
    elif distribution and target.runtime_version:
        try:
            if installed not in SpecifierSet(target.runtime_version):
                reasons.append(f"Requires {distribution}{target.runtime_version}; installed {installed}")
        except InvalidSpecifier:
            reasons.append(f"Invalid runtime version requirement: {target.runtime_version}")
    if target.minimum_memory_mb:
        memory = host.get("memory_mb")
        if memory is None or memory < target.minimum_memory_mb:
            reasons.append(f"Requires at least {target.minimum_memory_mb} MiB physical memory")
    if target.minimum_android_api and (host.get("android_api") or 0) < target.minimum_android_api:
        reasons.append(f"Requires Android API {target.minimum_android_api}")
    if target.firmware and host.get("firmware") not in target.firmware:
        reasons.append("Firmware is not in this variant's qualified allowlist")
    if target.minimum_security_patch and (host.get("security_patch") or "") < target.minimum_security_patch:
        reasons.append(f"Requires security patch {target.minimum_security_patch} or later")
    for component, requirement in target.components.items():
        version = host.get("components", {}).get(component)
        try:
            if not version or version not in SpecifierSet(requirement):
                reasons.append(f"Requires native component {component}{requirement}; installed {version}")
        except InvalidSpecifier:
            reasons.append(f"Invalid native component version requirement: {component}{requirement}")
    if variant.engine == "interpreter" and target.accelerator != "cpu":
        reasons.append("Interpreter variants support CPU; use a compiled variant for GPU/NPU")
    if (variant.chat and variant.chat.processor and variant.chat.processor.runner != "gemma4-ple"
            and not host["runtimes"].get("transformers")):
        reasons.append("Install phaseone-runtime[vision] for this packaged multimodal processor")
    return reasons


def candidates(model: Model, host: dict, *, backend: str | None = None,
               variant_id: str | None = None, accelerator: str | None = None,
               modalities: set[str] | None = None, image_count: int = 0,
               require_tools: bool = False) -> tuple[list, list]:
    available, rejected = [], []
    for variant in sorted(model.variants, key=lambda item: -item.priority):
        reasons = compatibility(variant, host)
        if backend and backend != variant.backend:
            reasons.append(f"Requested backend {backend}")
        if variant_id and variant_id != variant.id:
            reasons.append(f"Requested variant {variant_id}")
        if accelerator and accelerator != variant.target.accelerator:
            reasons.append(f"Requested accelerator {accelerator}")
        if modalities and (variant.chat is None or modalities - set(variant.chat.modalities)):
            reasons.append(f"Request requires modalities {sorted(modalities)}")
        if variant.chat and image_count > variant.chat.max_images:
            reasons.append(f"Request contains {image_count} images; variant supports {variant.chat.max_images}")
        if require_tools and (not variant.chat or variant.chat.tool_protocol == "none"):
            reasons.append("Request requires function calling")
        if reasons:
            rejected.append({"variant": variant.id, "reasons": reasons})
        else:
            available.append(variant)
    if not available:
        raise RuntimeFault("no_compatible_variant", "No application variant matches this host and request",
                           details={"host": host, "variants": rejected})
    return available, rejected
