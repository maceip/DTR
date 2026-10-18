"""Content-verified packages, detached signatures, and atomic local installation."""

import base64
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .errors import RuntimeFault
from .schema import Manifest, relative_path

MANIFEST = "phaseone.json"
SIGNATURE = "phaseone.signature.json"
MAX_MANIFEST_BYTES = 2 * 1024**2


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class JsonFloat(float):
    """A JSON number that keeps its source spelling, e.g. `3e-3` or `2.50`."""

    def __new__(cls, text):
        value = super().__new__(cls, text)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite JSON number: {text}")
        value.text = text
        return value


class JsonInt(int):
    """A JSON integer that keeps its source spelling, e.g. `-0`."""

    def __new__(cls, text):
        value = super().__new__(cls, text)
        value.text = text
        return value


def read_json(data: bytes, *, keep_number_text: bool = False):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    numbers = {"parse_float": JsonFloat, "parse_int": JsonInt} if keep_number_text else {}
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                      **numbers)


def canonical_source(value) -> bytes:
    """Release-identity JSON, byte-identical to the Android SDK's `canonical()`.

    Numbers parsed with `keep_number_text` keep their source spelling. Keys sort
    by UTF-16 code unit and U+2028/U+2029 are escaped, matching Kotlin string
    ordering and Gson output.
    """
    def encode(item) -> str:
        if isinstance(item, (JsonFloat, JsonInt)):
            return item.text
        if item is None or isinstance(item, (bool, int, float)):
            return json.dumps(item, allow_nan=False)
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=False).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        if isinstance(item, dict):
            keys = sorted(item, key=lambda key: key.encode("utf-16-be", "surrogatepass"))
            return "{" + ",".join(encode(key) + ":" + encode(item[key]) for key in keys) + "}"
        if isinstance(item, (list, tuple)):
            return "[" + ",".join(encode(entry) for entry in item) + "]"
        raise TypeError(f"Not a JSON value: {type(item).__name__}")

    return encode(value).encode("utf-8")


def atomic_json(path: Path, value, *, private: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".phaseone-", dir=path.parent)
    try:
        if private:
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as target:
            target.write(canonical(value) + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def contained_file(root: Path, name: str) -> Path:
    relative_path(name)
    root = root.resolve()
    candidate = root / name
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            raise RuntimeFault("unsafe_path", f"Package symlinks are not permitted: {name}")
        cursor = cursor.parent
    if not candidate.resolve().is_relative_to(root) or not candidate.is_file():
        raise RuntimeFault("missing_artifact", f"Missing package file: {name}")
    return candidate


@dataclass(frozen=True)
class Package:
    root: Path
    manifest: Manifest
    digest: str
    signature_status: str
    signer: str | None = None

    def artifact(self, name: str) -> Path:
        try:
            return contained_file(self.root, self.manifest.artifacts[name].path)
        except KeyError as exc:
            raise RuntimeFault("missing_artifact", f"Unknown artifact ID: {name}") from exc

    def describe(self) -> dict:
        return {
            "name": self.manifest.name, "version": self.manifest.version,
            "description": self.manifest.description,
            "digest": self.digest, "signature": self.signature_status, "signer": self.signer,
            "models": {name: model.model_dump(mode="json")
                       for name, model in self.manifest.models.items()},
            "permissions": self.manifest.permissions,
            "tools": [tool.model_dump(mode="json") for tool in self.manifest.tools],
            "agent": self.manifest.agent.model_dump(mode="json") if self.manifest.agent else None,
        }


def load_package(path: str | Path, *, trust_store: Path | None = None,
                 require_signature: bool = False, verify: bool = True, _signing: bool = False) -> Package:
    root = Path(path).resolve()
    if root.name == MANIFEST and root.is_file():
        root = root.parent
    try:
        manifest_path = contained_file(root, MANIFEST)
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("Manifest is too large")
        raw = read_json(manifest_path.read_bytes())
        manifest = Manifest.model_validate(raw)
        payload = canonical(raw)
    except RuntimeFault:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeFault("invalid_package", f"Invalid application manifest: {exc}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    status, signer = "unsigned", None
    signature_path = root / SIGNATURE
    if signature_path.exists() and not _signing:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            sig_path = contained_file(root, SIGNATURE)
            if sig_path.stat().st_size > 16384:
                raise ValueError("Signature is too large")
            sig = read_json(sig_path.read_bytes())
            if set(sig) != {"algorithm", "key_id", "public_key", "signature"}:
                raise ValueError("Invalid signature fields")
            if sig["algorithm"] != "ed25519":
                raise ValueError("Unsupported signature algorithm")
            public_bytes = base64.b64decode(sig["public_key"], validate=True)
            signer = hashlib.sha256(public_bytes).hexdigest()
            if signer != sig["key_id"]:
                raise ValueError("Signing key ID does not match public key")
            Ed25519PublicKey.from_public_bytes(public_bytes).verify(
                base64.b64decode(sig["signature"], validate=True), payload,
            )
            status = "valid_untrusted"
            if trust_store:
                key_file = trust_store / f"{signer}.pub"
                if key_file.is_file() and key_file.read_bytes().strip() == public_bytes.hex().encode():
                    status = "trusted"
        except (OSError, ValueError, KeyError, InvalidSignature) as exc:
            raise RuntimeFault("invalid_signature", "Application signature verification failed") from exc
    if require_signature and status != "trusted":
        raise RuntimeFault("untrusted_package", "A signature from a locally trusted key is required",
                           details={"signature": status, "signer": signer})
    if verify:
        for name, artifact in manifest.artifacts.items():
            item = contained_file(root, artifact.path)
            if item.stat().st_size != artifact.size or digest_file(item) != artifact.sha256:
                raise RuntimeFault("artifact_integrity", f"Artifact integrity check failed: {name}",
                                   details={"path": artifact.path})
    return Package(root, manifest, digest, status, signer)


def create_key(path: Path, *, trust_store: Path | None = None) -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if path.exists():
        raise RuntimeFault("key_exists", f"Refusing to replace an existing signing key: {path}")
    key = Ed25519PrivateKey.generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                            serialization.NoEncryption())
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as target:
        target.write(raw)
    public = key.public_key().public_bytes_raw()
    key_id = hashlib.sha256(public).hexdigest()
    if trust_store:
        trust_store.mkdir(parents=True, exist_ok=True)
        (trust_store / f"{key_id}.pub").write_text(public.hex() + "\n")
    return key_id


def sign_package(root: Path, key_path: Path) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    # Re-signing an edited package must replace its stale signature. Artifact
    # validation still runs before the new signature is written atomically.
    package = load_package(root, _signing=True)
    key = Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
    public = key.public_key().public_bytes_raw()
    key_id = hashlib.sha256(public).hexdigest()
    # Sign the original JSON, including all extension metadata and artifact hashes.
    payload = canonical(read_json((package.root / MANIFEST).read_bytes()))
    atomic_json(package.root / SIGNATURE, {
        "algorithm": "ed25519", "key_id": key_id,
        "public_key": base64.b64encode(public).decode(),
        "signature": base64.b64encode(key.sign(payload)).decode(),
    })
    return key_id


def pack(root: Path, destination: Path) -> dict:
    package = load_package(root)
    names = [MANIFEST, *(a.path for a in package.manifest.artifacts.values())]
    if (package.root / SIGNATURE).is_file():
        names.append(SIGNATURE)
    if destination.resolve().is_relative_to(package.root):
        raise RuntimeFault("unsafe_path", "Write the archive outside its application directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name in sorted(names):
                archive.write(contained_file(package.root, name), arcname=name)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), "package_digest": package.digest,
            "sha256": digest_file(destination), "size": destination.stat().st_size}


def install(source: Path, store: Path, *, trust_store: Path | None = None,
            require_signature: bool = False, max_bytes: int = 64 * 1024**3) -> Package:
    """Install an immutable copy; never execute directly from a mutable download."""
    store.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=store) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        if source.is_dir() or source.name == MANIFEST:
            package = load_package(source, trust_store=trust_store,
                                   require_signature=require_signature)
            if sum(a.size for a in package.manifest.artifacts.values()) > max_bytes:
                raise RuntimeFault("package_too_large", "Application exceeds installation size limit")
            names = [MANIFEST, *(a.path for a in package.manifest.artifacts.values())]
            if (package.root / SIGNATURE).is_file():
                names.append(SIGNATURE)
            for name in names:
                destination = staging / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(contained_file(package.root, name), destination)
        else:
            try:
                with zipfile.ZipFile(source) as archive:
                    infos = archive.infolist()
                    names = [info.filename for info in infos]
                    if len(names) != len(set(names)) or MANIFEST not in names:
                        raise ValueError("Archive has duplicate paths or no manifest")
                    if archive.getinfo(MANIFEST).file_size > MAX_MANIFEST_BYTES:
                        raise ValueError("Oversized manifest")
                    manifest = Manifest.model_validate(read_json(archive.read(MANIFEST)))
                    allowed = {MANIFEST, SIGNATURE, *(a.path for a in manifest.artifacts.values())}
                    total = 0
                    for info in infos:
                        relative_path(info.filename.rstrip("/"))
                        if info.is_dir():
                            continue
                        if info.filename not in allowed or stat.S_ISLNK(info.external_attr >> 16):
                            raise ValueError(f"Unexpected archive entry: {info.filename}")
                        if info.filename == SIGNATURE and info.file_size > 16384:
                            raise ValueError("Oversized signature")
                        total += info.file_size
                        if total > max_bytes:
                            raise ValueError("Archive exceeds installation size limit")
                        destination = staging / info.filename
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as reader, destination.open("xb") as writer:
                            shutil.copyfileobj(reader, writer, length=4 * 1024**2)
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                raise RuntimeFault("invalid_archive", f"Cannot install application: {exc}") from exc
        package = load_package(staging, trust_store=trust_store, require_signature=require_signature)
        destination = store / package.digest
        if destination.exists():
            cached = load_package(destination, trust_store=trust_store)
            if cached.digest != package.digest:
                raise RuntimeFault("artifact_integrity", "Cached application no longer matches its content address")
            # The detached signature is not part of the content address. A later
            # signed distribution may upgrade a previously unsigned installation;
            # reinstalling an unsigned copy must never erase that trust evidence.
            ranks = {"unsigned": 0, "valid_untrusted": 1, "trusted": 2}
            if ranks[package.signature_status] > ranks[cached.signature_status]:
                atomic_json(destination / SIGNATURE, read_json((staging / SIGNATURE).read_bytes()))
            return load_package(destination, trust_store=trust_store,
                                require_signature=require_signature)
        # Atomic rename keeps incomplete installs out of the model resolver.
        os.replace(staging, destination)
        return load_package(destination, trust_store=trust_store,
                            require_signature=require_signature)
