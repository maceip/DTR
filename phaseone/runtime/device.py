"""Host capability discovery and explainable model selection."""

import hashlib
import importlib.metadata
import os
import platform
import re
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


def discover() -> dict:
    system = platform.system().lower()
    if system == "linux" and (os.environ.get("ANDROID_ROOT") or Path("/system/build.prop").exists()):
        system = "android"
    arch = {"aarch64": "arm64", "amd64": "x86_64"}.get(
        platform.machine().lower(), platform.machine().lower(),
    )
    versions = {}
    for distribution in sorted({*BACKEND_DISTRIBUTIONS.values(), "transformers"}):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    try:
        memory_mb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 1024**2
    except (ValueError, AttributeError, OSError):
        memory_mb = None
    soc = None
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
    return {"system": system, "architecture": arch, "os_version": platform.release(),
            "memory_mb": memory_mb, "soc": soc, "runtimes": versions,
            "components": components, "phaseone_version": "0.3.0",
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
                 else host["runtimes"].get(distribution))
    if qnn_runner and (host["system"] != "linux" or canonical_arch(host["architecture"]) != "arm64"):
        reasons.append("The installed QNN runner requires Linux ARM64")
    if not installed:
        reasons.append("Install the DTR QNN host runtime" if qnn_runner
                       else f"Install {distribution} for {variant.backend}/{variant.engine}")
    elif target.runtime_version:
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
