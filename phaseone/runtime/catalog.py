"""A data-only release catalog shared by desktop and embedded Android hosts.

Catalogs describe model data and the native components an application must already
contain. They cannot install native code. Selection is separate from qualification:
a candidate becomes active only after its bundled task checks pass on the host.
"""

import base64
import hashlib
import math
import re
from typing import Annotated, Literal

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic import Field, PrivateAttr, field_validator, model_validator

from .device import BACKEND_DISTRIBUTIONS, canonical_arch, canonical_soc
from .errors import RuntimeFault
from .package import canonical, canonical_source, read_json
from .schema import Contract, Manifest, Name, Variant, relative_path

CATALOG_LIMIT = 8 * 1024**2
NATIVE_SUFFIXES = (".so", ".dll", ".dylib", ".dex", ".jar", ".aar", ".apk", ".py", ".js", ".exe", ".wasm")


def version_requirement(value: str) -> str:
    # A deliberately small, identical grammar in Kotlin and Python, not two
    # subtly different interpretations of PEP440, Maven, and npm ranges.
    if not re.fullmatch(r"(?:==|!=|>=|<=|>|<)\d+(?:\.\d+){0,2}(?:,(?:==|!=|>=|<=|>|<)\d+(?:\.\d+){0,2})*", value):
        raise ValueError("Use comma-separated numeric version comparisons, e.g. >=2.3.0,<3")
    return value


class TensorFixture(Contract):
    artifact: Name
    dtype: Literal["float32"] = "float32"
    shape: list[int] = Field(min_length=1, max_length=12)

    @field_validator("shape")
    @classmethod
    def bounded_shape(cls, shape):
        import math
        if any(d <= 0 for d in shape) or math.prod(shape) > 16_000_000:
            raise ValueError("Qualification tensors must have positive, bounded dimensions")
        return shape


class TensorCheck(Contract):
    kind: Literal["tensor"] = "tensor"
    name: Name
    signature: Name = "default"
    inputs: dict[Name, TensorFixture]
    outputs: dict[Name, TensorFixture]
    absolute_tolerance: float = Field(default=0.001, ge=0, le=1)

    @model_validator(mode="after")
    def nonempty(self):
        if not self.inputs or not self.outputs:
            raise ValueError("Qualification requires both inputs and reference outputs")
        return self

    def artifact_ids(self):
        return {t.artifact for t in [*self.inputs.values(), *self.outputs.values()]}


class ChatCheckMessage(Contract):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=65536)
    images: list[Name] = Field(default_factory=list, max_length=16)


class ChatCheck(Contract):
    kind: Literal["chat"] = "chat"
    name: Name
    messages: list[ChatCheckMessage] = Field(min_length=1, max_length=64)
    max_output_tokens: int = Field(default=64, ge=1, le=512)
    contains: list[str] = Field(default_factory=list, max_length=32)
    json_fields: dict[str, str | int | float | bool] = {}
    json_choices: dict[str, list[str | int | float | bool]] = {}

    @model_validator(mode="after")
    def expectations(self):
        if self.messages[-1].role != "user" or not (self.contains or self.json_fields or self.json_choices):
            raise ValueError("Chat qualification needs a final user message and concrete expectations")
        if any(not value.strip() for value in self.contains):
            raise ValueError("Chat expectations cannot be empty strings")
        if any(not choices for choices in self.json_choices.values()):
            raise ValueError("JSON choices must declare at least one accepted value")
        return self

    def artifact_ids(self):
        return {image for message in self.messages for image in message.images}


QualificationCheck = Annotated[TensorCheck | ChatCheck, Field(discriminator="kind")]


class Release(Contract):
    id: Name
    sequence: int = Field(ge=1)
    capability: Name
    contract: Name
    lineage: str = Field(min_length=1, max_length=512)
    minimum_runtime: str = Field(default="0.3.0", pattern=r"^\d+\.\d+\.\d+$")
    manifest: Manifest
    model: Name
    base: str
    checks: list[QualificationCheck] = Field(min_length=1, max_length=64)
    _source_digest: str | None = PrivateAttr(default=None)

    @field_validator("base")
    @classmethod
    def base_path(cls, value):
        return relative_path(value.rstrip("/")) + "/"

    @model_validator(mode="after")
    def valid(self):
        if self.model not in self.manifest.models:
            raise ValueError("Release model is absent from the package")
        for artifact in self.manifest.artifacts.values():
            if artifact.path.lower().endswith(NATIVE_SUFFIXES) or ".so." in artifact.path.lower():
                raise ValueError("Model releases cannot distribute executable native/application code")
            if artifact.path.casefold() in ("catalog.json", "activation.json"):
                raise ValueError("Release metadata paths are reserved")
        for check in self.checks:
            if check.artifact_ids() - self.manifest.artifacts.keys():
                raise ValueError("Qualification references undeclared artifacts")
            if isinstance(check, TensorCheck):
                for fixture in [*check.inputs.values(), *check.outputs.values()]:
                    if self.manifest.artifacts[fixture.artifact].size != math.prod(fixture.shape) * 4:
                        raise ValueError("Qualification artifact size must match its float32 tensor shape")
        for variant in self.manifest.models[self.model].variants:
            if variant.target.runtime_version:
                version_requirement(variant.target.runtime_version)
            for requirement in variant.target.components.values():
                version_requirement(requirement)
            if variant.target.accelerator != "cpu" and not variant.target.components:
                raise ValueError("Accelerated releases must name their required native components")
            if variant.target.accelerator == "npu" and not variant.target.soc:
                raise ValueError("NPU releases require an exact target SoC")
            for check in self.checks:
                if isinstance(check, ChatCheck):
                    if variant.engine != "llm":
                        raise ValueError("Chat qualification requires a chat model")
                    if check.artifact_ids() and "image" not in variant.chat.modalities:
                        raise ValueError("Image qualification requires an image-capable variant")
                    continue
                signature = variant.signatures.get(check.signature)
                if signature is None:
                    raise ValueError("Every variant must implement the qualification signature")
                for fixtures, specs in ((check.inputs, signature.inputs), (check.outputs, signature.outputs)):
                    if set(fixtures) != {s.name for s in specs}:
                        raise ValueError("Qualification tensor names must match every variant")
                    for spec in specs:
                        item = fixtures[spec.name]
                        if item.dtype != spec.dtype or any(a != b and b != -1 for a, b in zip(item.shape, spec.shape, strict=True)):
                            raise ValueError("Qualification tensor contracts must match every variant")
        return self

    def artifact_ids(self, variant: Variant) -> set[str]:
        result = variant.artifact_ids()
        model = self.manifest.models[self.model]
        if model.labels:
            result.add(model.labels)
        for check in self.checks:
            result |= check.artifact_ids()
        return result

    @property
    def digest(self):
        """Release identity: SHA-256 of the release JSON as published, shared with Android.

        A release decoded from catalog bytes hashes that JSON exactly, including
        number spelling and omitted defaults. A release built in Python hashes the
        expanded form that `signed_catalog` publishes.
        """
        if self._source_digest:
            return self._source_digest
        return hashlib.sha256(canonical_source(self.model_dump(mode="json", exclude_none=True))).hexdigest()


class Catalog(Contract):
    schema_version: Literal[1] = 1
    name: Name
    releases: list[Release] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def unique(self):
        if len({r.id for r in self.releases}) != len(self.releases):
            raise ValueError("Release IDs are immutable and unique")
        keys = [(r.capability, r.sequence) for r in self.releases]
        if len(set(keys)) != len(keys):
            raise ValueError("Sequence numbers are unique within a capability")
        return self


class ReleasePolicy(Contract):
    contract: Name
    allowed_lineages: list[str] = Field(min_length=1)
    accelerators: list[Literal["cpu", "gpu", "npu"]] = Field(default_factory=lambda: ["cpu"], min_length=1)
    backend: Literal["executorch", "litert"] | None = None
    allow_fallback: bool = False


def rejection_reasons(variant: Variant, device: dict) -> list[dict]:
    target = variant.target
    reasons = []

    def reject(code, message, app_update=False):
        reasons.append({"code": code, "message": message, "app_update": app_update})

    if target.systems and device["system"] not in target.systems:
        reject("os", "Operating system does not match")
    if target.architectures and canonical_arch(device["architecture"]) not in {canonical_arch(v) for v in target.architectures}:
        reject("abi", "Native architecture does not match")
    if target.soc and canonical_soc(target.soc) != canonical_soc(device.get("soc")):
        reject("soc", "Exact SoC does not match")
    if target.minimum_android_api > (device.get("android_api") or 0):
        reject("android_api", "Android API level is too old")
    if target.minimum_memory_mb > (device.get("memory_mb") or 0):
        reject("memory", "Insufficient physical memory")
    if target.firmware and device.get("firmware") not in target.firmware:
        reject("firmware", "Firmware is not qualified for this variant")
    if target.minimum_security_patch and (device.get("security_patch") or "") < target.minimum_security_patch:
        reject("security_patch", "Security patch is too old")
    distribution = BACKEND_DISTRIBUTIONS[(variant.backend, variant.engine)]
    installed = device.get("runtimes", {}).get(distribution)
    if not installed or (target.runtime_version and installed not in SpecifierSet(target.runtime_version)):
        reject("engine", f"Requires {distribution}{target.runtime_version or ''}; installed {installed}", True)
    for name, requirement in target.components.items():
        installed = device.get("components", {}).get(name)
        if not installed or installed not in SpecifierSet(requirement):
            reject("component", f"Requires {name}{requirement}; installed {installed}", True)
    supported = device.get("engines")
    if supported is not None and f"{variant.backend}/{variant.engine}" not in supported:
        reject("adapter", "This application does not include the required adapter", True)
    return reasons


def qualification_fingerprint(device: dict) -> str:
    """A model must be rechecked when SDK code, native engines or firmware changes."""
    values = {name: device.get(name) or "" for name in (
        "system", "firmware", "security_patch", "phaseone_version", "runtime_build",
    )}
    values.update({"architecture": canonical_arch(device["architecture"]),
                   "soc": canonical_soc(device.get("soc")) or "",
                   "android_api": device.get("android_api") or 0,
                   "runtimes": {k: v for k, v in device.get("runtimes", {}).items() if v},
                   "components": {k: v for k, v in device.get("components", {}).items() if v}})
    return hashlib.sha256(canonical(values)).hexdigest()


def resolve(catalog: Catalog, capability: str, device: dict, policy: ReleasePolicy,
            *, active: dict | None = None, excluded: set[tuple[str, str]] | None = None) -> dict:
    rejected = []
    chosen = None
    needs_app = False
    releases = sorted((r for r in catalog.releases if r.capability == capability), key=lambda r: -r.sequence)
    catalog_sequence = max((r.sequence for r in releases
                            if r.contract == policy.contract and r.lineage in policy.allowed_lineages), default=0)
    for release in releases:
        for variant in sorted(release.manifest.models[release.model].variants, key=lambda v: (-v.priority, v.id)):
            reasons = rejection_reasons(variant, device)
            if (release.id, variant.id) in (excluded or set()):
                reasons.append({"code": "qualification", "message": "Candidate failed during this update", "app_update": False})
            if Version(device.get("phaseone_version", "0")) < Version(release.minimum_runtime):
                reasons.append({"code": "runtime", "message": "PHASEONE SDK update required", "app_update": True})
            for failed, message in (
                (release.contract != policy.contract, "Task contract does not match"),
                (release.lineage not in policy.allowed_lineages, "Checkpoint lineage is not allowed by the app"),
                (variant.target.accelerator not in policy.accelerators, "Accelerator is not allowed by the app"),
                (policy.backend is not None and variant.backend != policy.backend, "Backend is not requested"),
            ):
                if failed:
                    reasons.append({"code": "policy", "message": message, "app_update": False})
            if reasons:
                rejected.append({"release": release.id, "variant": variant.id, "reasons": reasons})
                needs_app |= all(r["app_update"] for r in reasons)
                continue
            if chosen is None:
                ids = sorted(release.artifact_ids(variant))
                chosen = {"release": release.id, "sequence": release.sequence, "variant": variant.id,
                          "catalog_sequence": catalog_sequence,
                          "backend": variant.backend, "accelerator": variant.target.accelerator,
                          "artifacts": ids, "download_bytes": sum(release.manifest.artifacts[a].size for a in ids),
                          "qualification": "required", "model": release.model}
    status = "no_compatible_release"
    if chosen:
        status = "up_to_date" if active and (chosen["release"], chosen["variant"]) == (active.get("release"), active.get("variant")) else "model_update_available"
        if status == "up_to_date" and active.get("qualification_fingerprint") != qualification_fingerprint(device):
            status = "requalification_required"
        if active and catalog_sequence < active.get("catalog_sequence", active.get("sequence", 0)):
            # An old feed is different from choosing an older CPU variant from a
            # current feed after policy/compatibility or qualification rejects NPU.
            status = "stale_catalog"
            chosen = None
    if needs_app and status != "stale_catalog":
        status = "app_update_required" if chosen is None or status in ("up_to_date", "requalification_required") else status
    return {"status": status, "selection": chosen, "app_update_available": needs_app, "rejected": rejected}


def signed_catalog(catalog: Catalog, private_key: bytes) -> bytes:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.from_private_bytes(private_key)
    public = key.public_key().public_bytes_raw()
    payload = canonical(catalog.model_dump(mode="json", exclude_none=True))
    return canonical({"format": "phaseone.catalog.signed.v1", "key_id": hashlib.sha256(public).hexdigest(),
                      "payload": base64.b64encode(payload).decode(), "signature": base64.b64encode(key.sign(payload)).decode()})


def decode_catalog(data: bytes, *, trusted_keys: dict[str, bytes] | None = None,
                   allow_unsigned=False) -> Catalog:
    if len(data) > CATALOG_LIMIT:
        raise RuntimeFault("invalid_catalog", "Catalog exceeds its size limit")
    try:
        raw = read_json(data)
        source = data
        if raw.get("format") == "phaseone.catalog.signed.v1":
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            public = (trusted_keys or {}).get(raw["key_id"])
            if public is None or hashlib.sha256(public).hexdigest() != raw["key_id"]:
                raise RuntimeFault("untrusted_catalog", "Catalog signing key is not trusted by this app")
            source = base64.b64decode(raw["payload"], validate=True)
            Ed25519PublicKey.from_public_bytes(public).verify(base64.b64decode(raw["signature"], validate=True), source)
            raw = read_json(source)
        elif not allow_unsigned:
            raise RuntimeFault("untrusted_catalog", "A signed release catalog is required")
        catalog = Catalog.model_validate(raw)
        spelled = read_json(source, keep_number_text=True)["releases"]
        for release, entry in zip(catalog.releases, spelled, strict=True):
            # Set only when the published JSON differs from the model's expanded
            # form, so equal releases with the same identity still compare equal.
            digest = hashlib.sha256(canonical_source(entry)).hexdigest()
            if digest != release.digest:
                release._source_digest = digest
        return catalog
    except RuntimeFault:
        raise
    except Exception as exc:
        raise RuntimeFault("invalid_catalog", "Catalog or signature validation failed") from exc
