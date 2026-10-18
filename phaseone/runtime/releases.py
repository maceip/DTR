"""Selective data delivery, on-device qualification, activation and rollback."""

import hashlib
import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import httpx
import numpy as np

from .catalog import (
    CATALOG_LIMIT,
    ChatCheck,
    ReleasePolicy,
    decode_catalog,
    qualification_fingerprint,
    resolve,
)
from .errors import RuntimeFault
from .package import (
    Package,
    atomic_json,
    canonical,
    contained_file,
    digest_file,
    load_package,
    read_json,
)


def _json_equal(actual, expected):
    # JSON booleans are not numeric 0/1; Python's equality otherwise conflates them.
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    return actual == expected


def allowed_url(url: str, *, local=False) -> None:
    parsed = urlparse(url)
    if parsed.username or parsed.password or parsed.fragment:
        raise RuntimeFault("invalid_source", "Catalog URLs cannot contain credentials or fragments")
    if parsed.scheme == "https" and parsed.hostname:
        return
    if local and parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        return
    if local and parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        return
    raise RuntimeFault("invalid_source", "Use HTTPS; local development explicitly permits file/loopback URLs")


class ReleaseManager:
    """One catalog lifecycle for the existing Python runtime and Android SDK.

    All APIs are synchronous; use the caller's background executor for downloads
    and qualification. Readers keep the last complete activation during an update.
    """

    def __init__(self, runtime, *, root: Path | None = None,
                 trusted_keys: dict[str, bytes] | None = None, local_development=False,
                 max_download_bytes=16 * 1024**3):
        self.runtime = runtime
        self.root = Path(root or runtime.root / "releases").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.trusted_keys = trusted_keys or {}
        self.local_development = local_development
        self.allow_unsigned = local_development and not runtime.require_signature
        self.max_download_bytes = max_download_bytes

    @contextmanager
    def _write_lock(self):
        # SQLite supplies a cross-process lock on Windows as well as POSIX.
        with sqlite3.connect(self.root / "lock.sqlite3", timeout=300) as db:
            db.execute("BEGIN IMMEDIATE")
            yield

    def _read(self, url, limit):
        allowed_url(url, local=self.local_development)
        parsed = urlparse(url)
        if parsed.scheme == "file":
            from urllib.parse import unquote
            with Path(unquote(parsed.path)).open("rb") as source:
                data = source.read(limit + 1)
        else:
            with httpx.Client(timeout=60, follow_redirects=False) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    data = bytearray()
                    for chunk in response.iter_bytes():
                        data.extend(chunk)
                        if len(data) > limit:
                            break
                    data = bytes(data)
        if len(data) > limit:
            raise RuntimeFault("download_limit", "Download exceeds declared size limit")
        return data

    def catalog(self, url):
        data = self._read(url, CATALOG_LIMIT)
        return decode_catalog(data, trusted_keys=self.trusted_keys,
                              allow_unsigned=self.allow_unsigned), data

    def state(self, capability: str) -> dict:
        from .schema import relative_path
        relative_path(capability)
        if "/" in capability:
            raise RuntimeFault("invalid_capability", "A capability is a single identifier")
        path = self.root / "active" / f"{capability}.json"
        return read_json(path.read_bytes()) if path.exists() else {"active": None, "previous": None}

    def check(self, url: str, capability: str, policy: ReleasePolicy) -> dict:
        catalog, _ = self.catalog(url)
        return resolve(catalog, capability, self.runtime.device, policy,
                       active=self.state(capability)["active"])

    def _cache(self, url, artifact):
        cache = self.root / "objects" / artifact.sha256
        if cache.is_file() and cache.stat().st_size == artifact.size and digest_file(cache) == artifact.sha256:
            return cache, 0
        cache.parent.mkdir(parents=True, exist_ok=True)
        allowed_url(url, local=self.local_development)
        if artifact.size > self.max_download_bytes:
            raise RuntimeFault("download_limit", "Artifact exceeds configured download budget")
        fd, name = tempfile.mkstemp(prefix=".download-", dir=cache.parent)
        size = 0
        sha = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as target:
                def write(chunks):
                    nonlocal size
                    for chunk in chunks:
                        size += len(chunk)
                        if size > artifact.size:
                            raise RuntimeFault("artifact_integrity", "Artifact exceeds its declared size")
                        sha.update(chunk)
                        target.write(chunk)
                if urlparse(url).scheme == "file":
                    from urllib.parse import unquote
                    with Path(unquote(urlparse(url).path)).open("rb") as source:
                        write(iter(lambda: source.read(1024**2), b""))
                else:
                    with httpx.Client(timeout=120, follow_redirects=False) as client:
                        with client.stream("GET", url) as response:
                            response.raise_for_status()
                            write(response.iter_bytes(1024**2))
                target.flush()
                os.fsync(target.fileno())
            if size != artifact.size or sha.hexdigest() != artifact.sha256:
                raise RuntimeFault("artifact_integrity", "Artifact size or SHA-256 does not match the catalog")
            os.replace(name, cache)
            return cache, size
        finally:
            Path(name).unlink(missing_ok=True)

    @staticmethod
    def _projection(release, variant):
        manifest = release.manifest.model_dump(mode="json", exclude_none=True)
        model = manifest["models"][release.model]
        model["variants"] = [variant.model_dump(mode="json", exclude_none=True)]
        manifest["models"] = {release.model: model}
        manifest["artifacts"] = {a: manifest["artifacts"][a] for a in sorted(release.artifact_ids(variant))}
        if manifest.get("agent", {}).get("model") != release.model:
            manifest.pop("agent", None)
        manifest["metadata"] = {**manifest.get("metadata", {}), "release": release.id,
                                "contract": release.contract, "lineage": release.lineage,
                                "source_release_digest": release.digest}
        return manifest

    def _application(self, directory, release, variant):
        from .host import Application
        package = load_package(directory)
        expected = self._projection(release, variant)
        if canonical(read_json((directory / "phaseone.json").read_bytes())) != canonical(expected):
            raise RuntimeFault("artifact_integrity", "Installed projection differs from its catalog")
        source = contained_file(directory, "catalog.json").read_bytes()
        decode_catalog(source, trusted_keys=self.trusted_keys, allow_unsigned=self.allow_unsigned)
        signature = "trusted_catalog" if read_json(source).get("format") == "phaseone.catalog.signed.v1" else "development_catalog"
        package = Package(package.root, package.manifest, package.digest, signature)
        return Application(self.runtime, package)

    def _qualify(self, app, release, variant):
        from .host import Application, Runtime

        # Qualification has its own worker pool. Closing a temporary model must
        # not interrupt an application's in-flight generation on the old release.
        with Runtime(self.root / "qualification-state" / release.digest, max_loaded_models=1) as runtime:
            runtime.device = self.runtime.device
            return self._run_checks(Application(runtime, app.package), release, variant)

    def _run_checks(self, app, release, variant):
        from .tensors import decode_tensor
        checks = []
        for check in release.checks:
            if isinstance(check, ChatCheck):
                import base64
                import json
                import mimetypes

                messages = []
                for message in check.messages:
                    content = message.content
                    if message.images:
                        content = [{"type": "text", "text": message.content}, *[
                            {"type": "image", "mime_type": mimetypes.guess_type(app.package.artifact(name).name)[0] or "image/png",
                             "data": base64.b64encode(app.package.artifact(name).read_bytes()).decode()}
                            for name in message.images]]
                    messages.append({"role": message.role, "content": content})
                result = app.chat(release.model, messages, variant=variant.id, allow_fallback=False,
                                  generation={"max_output_tokens": check.max_output_tokens, "temperature": 0, "timeout_seconds": 240})
                output = result["message"]["content"]
                passed = isinstance(output, str) and all(text.casefold() in output.casefold() for text in check.contains)
                if check.json_fields or check.json_choices:
                    try:
                        parsed = json.loads(output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
                        passed &= all(_json_equal(parsed.get(key), value) for key, value in check.json_fields.items())
                        passed &= all(any(_json_equal(parsed.get(key), value) for value in values) for key, values in check.json_choices.items())
                    except (ValueError, AttributeError):
                        passed = False
                checks.append({"name": check.name, "passed": passed, "output": output, "execution": result["execution"]})
                if not passed:
                    raise RuntimeFault("qualification_failed", "Chat task checks failed; active release is unchanged", details={"checks": checks})
                continue
            inputs = {name: np.fromfile(app.package.artifact(f.artifact), dtype="<f4").reshape(f.shape)
                      for name, f in check.inputs.items()}
            result = app.infer(release.model, inputs, signature=check.signature, variant=variant.id,
                               allow_fallback=False)
            errors = {}
            for name, fixture in check.outputs.items():
                expected = np.fromfile(app.package.artifact(fixture.artifact), dtype="<f4").reshape(fixture.shape)
                actual = decode_tensor(result["outputs"][name])
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise RuntimeFault("qualification_failed", f"Invalid output in {check.name}/{name}")
                errors[name] = float(np.max(np.abs(actual - expected)))
            passed = all(value <= check.absolute_tolerance for value in errors.values())
            checks.append({"name": check.name, "passed": passed, "max_absolute_error": errors,
                           "execution": result["execution"]})
            if not passed:
                raise RuntimeFault("qualification_failed", "Task reference checks failed; active release is unchanged",
                                   details={"checks": checks})
        return checks

    def update(self, url: str, capability: str, policy: ReleasePolicy) -> dict:
        with self._write_lock():
            catalog, catalog_data = self.catalog(url)
            state = self.state(capability)
            excluded, failures = set(), []
            for _ in range(16):
                plan = resolve(catalog, capability, self.runtime.device, policy,
                               active=state["active"], excluded=excluded)
                selection = plan["selection"]
                if plan["status"] not in ("model_update_available", "requalification_required") or selection is None:
                    if failures:
                        if plan["status"] == "up_to_date":
                            return {**plan, "status": "retained", "activation": state["active"], "fallback": failures}
                        raise RuntimeFault("no_qualified_release", "No allowed candidate passed; active release is unchanged",
                                           details={"attempts": failures, "plan": plan})
                    return plan
                try:
                    return self._activate(url, capability, catalog, catalog_data, state, plan, failures)
                except RuntimeFault as failure:
                    if not policy.allow_fallback or failure.code in {"release_conflict", "untrusted_catalog", "invalid_catalog", "activation_failed"}:
                        raise
                    failures.append({"release": selection["release"], "variant": selection["variant"],
                                     "error": {"code": failure.code, "message": str(failure)}})
                    excluded.add((selection["release"], selection["variant"]))
            raise RuntimeFault("no_qualified_release", "Candidate attempt limit reached; active release is unchanged",
                               details={"attempts": failures})

    def _activate(self, url, capability, catalog, catalog_data, state, plan, failures):
        selection = plan["selection"]
        release = next(r for r in catalog.releases if r.id == selection["release"])
        variant = next(v for v in release.manifest.models[release.model].variants if v.id == selection["variant"])
        # Even a trusted publisher cannot quietly redefine a previously seen ID.
        identity = self.root / "identities" / f"{release.id}.json"
        if identity.exists() and read_json(identity.read_bytes())["digest"] != release.digest:
            raise RuntimeFault("release_conflict", "A release ID was reused with different model data")
        if selection["download_bytes"] > self.max_download_bytes:
            raise RuntimeFault("download_limit", "Selected dependency closure exceeds the download budget")
        downloaded, artifacts = 0, []
        destination = self.root / "installed" / release.digest / variant.id
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "package"
            staging.mkdir()
            for name in selection["artifacts"]:
                artifact = release.manifest.artifacts[name]
                source_url = urljoin(url, quote(release.base + artifact.path, safe="/"))
                cached, size = self._cache(source_url, artifact)
                downloaded += size
                target = staging / artifact.path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached, target)
                artifacts.append({"id": name, "sha256": artifact.sha256, "downloaded_bytes": size})
            atomic_json(staging / "phaseone.json", self._projection(release, variant))
            (staging / "catalog.json").write_bytes(catalog_data)
            app = self._application(staging, release, variant)
            checks = self._qualify(app, release, variant)
            record = {**selection, "digest": release.digest, "contract": release.contract,
                      "lineage": release.lineage, "qualification": "passed", "checks": checks,
                      "downloaded_bytes": downloaded, "artifact_receipts": artifacts,
                      "catalog_signature": app.package.signature_status, "fallback": failures,
                      "qualification_fingerprint": qualification_fingerprint(self.runtime.device),
                      "activated_at": time.time(), "device": self.runtime.device}
            atomic_json(staging / "activation.json", record)
            if destination.exists():
                # Previously qualified objects are immutable. Verify them before
                # reusing; never remove a release a concurrent reader may use.
                self._application(destination, release, variant)
            else:
                os.replace(staging, destination)
            atomic_json(identity, {"digest": release.digest})
            atomic_json(self.root / "active" / f"{capability}.json",
                        {"active": record, "previous": state["previous"] if plan["status"] == "requalification_required" else state["active"]})
        return {"status": "activated", "activation": record, "rejected": plan["rejected"]}

    def _open_record(self, record):
        if not record:
            raise RuntimeFault("no_active_release", "Install a qualified model release first")
        directory = self.root / "installed" / record["digest"] / record["variant"]
        catalog = decode_catalog(contained_file(directory, "catalog.json").read_bytes(),
                                 trusted_keys=self.trusted_keys, allow_unsigned=self.allow_unsigned)
        release = next((r for r in catalog.releases if r.id == record["release"]), None)
        if release is None or release.digest != record["digest"]:
            raise RuntimeFault("artifact_integrity", "Installed release has no matching catalog entry")
        variant = next(v for v in release.manifest.models[release.model].variants if v.id == record["variant"])
        from .catalog import rejection_reasons
        reasons = rejection_reasons(variant, self.runtime.device)
        from packaging.version import Version
        if Version(self.runtime.device.get("phaseone_version", "0")) < Version(release.minimum_runtime):
            reasons.append({"code": "runtime", "message": "PHASEONE SDK update required", "app_update": True})
        if reasons:
            raise RuntimeFault("release_incompatible", "The active release no longer matches this host", details={"reasons": reasons})
        return self._application(directory, release, variant), release, variant

    def open(self, capability: str):
        current = self.state(capability)["active"]
        if current and current.get("qualification_fingerprint") != qualification_fingerprint(self.runtime.device):
            self.requalify(capability)
        app = self._open_record(self.state(capability)["active"])[0]
        self.runtime.packages[app.package.digest] = app.package
        return app

    def requalify(self, capability: str) -> dict:
        """Recheck installed data after an SDK/firmware update, without a network fetch."""
        with self._write_lock():
            state = self.state(capability)
            app, release, variant = self._open_record(state["active"])
            checks = self._qualify(app, release, variant)
            record = {**state["active"], "checks": checks, "device": self.runtime.device,
                      "qualification_fingerprint": qualification_fingerprint(self.runtime.device),
                      "activated_at": time.time()}
            atomic_json(self.root / "active" / f"{capability}.json", {"active": record, "previous": state["previous"]})
            return {"status": "requalified", "activation": record}

    def rollback(self, capability: str) -> dict:
        with self._write_lock():
            state = self.state(capability)
            if state["previous"] is None:
                raise RuntimeFault("no_previous_release", "No previously qualified release is available")
            app, release, variant = self._open_record(state["previous"])
            checks = self._qualify(app, release, variant)
            record = {**state["previous"], "checks": checks, "activated_at": time.time(),
                      "device": self.runtime.device,
                      "catalog_sequence": max(state["previous"].get("catalog_sequence", state["previous"]["sequence"]),
                                              state["active"].get("catalog_sequence", state["active"]["sequence"])),
                      "qualification_fingerprint": qualification_fingerprint(self.runtime.device)}
            atomic_json(self.root / "active" / f"{capability}.json",
                        {"active": record, "previous": state["active"]})
            return {"status": "rolled_back", "activation": record}
