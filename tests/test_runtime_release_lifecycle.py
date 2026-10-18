"""Release download, activation, rollback and requalification without native engines.

Qualification is stubbed at `_run_checks`; everything else (catalog signature,
artifact transfer, cache, projection, activation records) is the real code path.
Native execution of the same lifecycle is covered by check_runtime_releases.py.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from phaseone.runtime import Runtime, RuntimeFault
from phaseone.runtime.catalog import Catalog, ReleasePolicy, signed_catalog
from phaseone.runtime.releases import ReleaseManager

ROOT = Path(__file__).resolve().parents[1]
CASE = json.loads((ROOT / "sdk/contracts/resolver-cases.json").read_text())["cases"][0]
POLICY = ReleasePolicy.model_validate(CASE["policy"])


class Feed:
    """A signed file:// catalog whose artifacts are real files with real hashes."""

    def __init__(self, root: Path):
        self.root = root
        self.key = Ed25519PrivateKey.generate()
        self.public = self.key.public_key().public_bytes_raw()
        self.count = 0

    def release(self, release_id, sequence, weights=b"weights"):
        data = copy.deepcopy(CASE["catalog"]["releases"][0])
        data.update(id=release_id, sequence=sequence, base=f"{release_id}-{weights.decode()}/")
        for name, artifact in data["manifest"]["artifacts"].items():
            if name.startswith("qualification."):
                content = bytes(artifact["size"])  # Size is pinned by the float32 tensor shape.
            elif name == "vision.labels":
                content = json.dumps([f"{release_id}-label"]).encode()
            else:
                content = b"%s:%s:%s" % (release_id.encode(), name.encode(), weights)
            path = self.root / data["base"] / artifact["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            artifact.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
        return data

    def publish(self, *releases) -> str:
        self.count += 1
        catalog = Catalog.model_validate({"name": "lifecycle-tests", "releases": list(releases)})
        path = self.root / f"catalog-{self.count}.json"
        path.write_bytes(signed_catalog(catalog, self.key.private_bytes_raw()))
        return path.as_uri()

    def artifact(self, release, name) -> Path:
        return self.root / release["base"] / release["manifest"]["artifacts"][name]["path"]


@pytest.fixture
def feed(tmp_path):
    return Feed(tmp_path / "feed")


@pytest.fixture
def qualification(monkeypatch):
    """Records qualification runs; release IDs in `failing` fail their checks."""
    runs, failing = [], set()

    def run_checks(self, app, release, variant):
        runs.append((release.id, variant.id))
        if release.id in failing:
            raise RuntimeFault("qualification_failed", "Stubbed check failed")
        return [{"name": "stub", "passed": True}]

    monkeypatch.setattr(ReleaseManager, "_run_checks", run_checks)
    return runs, failing


@pytest.fixture
def runtime(tmp_path, feed):
    key_id = hashlib.sha256(feed.public).hexdigest()
    with Runtime(tmp_path / "state", catalog_keys={key_id: feed.public}, local_development=True) as runtime:
        runtime.device = copy.deepcopy(CASE["device"])
        yield runtime


def active(runtime):
    record = runtime.releases.state("vision")["active"]
    return record and (record["release"], record["variant"])


def test_update_activates_and_rollback_restores_the_previous_release(runtime, feed, qualification):
    v1 = feed.release("vision-1", 1)
    first = runtime.releases.update(feed.publish(v1), "vision", POLICY)
    assert first["status"] == "activated"
    assert active(runtime) == ("vision-1", "executorch-cpu")
    assert first["activation"]["downloaded_bytes"] > 0

    url = feed.publish(v1, feed.release("vision-2", 2))
    assert runtime.releases.update(url, "vision", POLICY)["status"] == "activated"
    state = runtime.releases.state("vision")
    assert (state["active"]["release"], state["previous"]["release"]) == ("vision-2", "vision-1")
    assert runtime.open("vision").package.manifest.metadata["release"] == "vision-2"

    rolled = runtime.releases.rollback("vision")
    state = runtime.releases.state("vision")
    assert rolled["status"] == "rolled_back"
    assert (state["active"]["release"], state["previous"]["release"]) == ("vision-1", "vision-2")
    assert runtime.open("vision").package.manifest.metadata["release"] == "vision-1"
    # Rollback requalifies the older release on this device before activating it.
    assert qualification[0] == [("vision-1", "executorch-cpu"), ("vision-2", "executorch-cpu"),
                                ("vision-1", "executorch-cpu")]

    # Reactivating installed data reuses verified cache objects instead of downloading.
    again = runtime.releases.update(url, "vision", POLICY)
    assert again["status"] == "activated" and active(runtime) == ("vision-2", "executorch-cpu")
    assert again["activation"]["downloaded_bytes"] == 0


def test_failed_qualification_keeps_the_working_release(runtime, feed, qualification):
    _, failing = qualification
    v1 = feed.release("vision-1", 1)
    runtime.releases.update(feed.publish(v1), "vision", POLICY)
    before = runtime.releases.state("vision")
    failing.add("vision-2")
    url = feed.publish(v1, feed.release("vision-2", 2))

    with pytest.raises(RuntimeFault) as error:
        runtime.releases.update(url, "vision", POLICY)
    assert error.value.code == "qualification_failed"
    assert runtime.releases.state("vision") == before

    # Explicit fallback tries both vision-2 variants, then keeps the active release.
    retained = runtime.releases.update(url, "vision", POLICY.model_copy(update={"allow_fallback": True}))
    assert retained["status"] == "retained"
    assert [(f["release"], f["variant"]) for f in retained["fallback"]] == [
        ("vision-2", "executorch-cpu"), ("vision-2", "litert-cpu")]
    assert runtime.releases.state("vision") == before
    assert runtime.open("vision").package.manifest.metadata["release"] == "vision-1"


@pytest.mark.parametrize("tamper", ["replaced", "extended"])
def test_served_bytes_must_match_the_signed_catalog(runtime, feed, qualification, tamper):
    v1 = feed.release("vision-1", 1)
    url = feed.publish(v1)
    weights = feed.artifact(v1, "vision.executorch")
    data = weights.read_bytes()
    weights.write_bytes(bytes(len(data)) if tamper == "replaced" else data + b"!")

    with pytest.raises(RuntimeFault) as error:
        runtime.releases.update(url, "vision", POLICY)
    assert error.value.code == "artifact_integrity"
    assert active(runtime) is None
    assert qualification[0] == []
    digest = v1["manifest"]["artifacts"]["vision.executorch"]["sha256"]
    assert not (runtime.releases.root / "objects" / digest).exists()


def test_corrupted_cache_object_is_downloaded_again(runtime, feed, qualification):
    v1 = feed.release("vision-1", 1)
    runtime.releases.update(feed.publish(v1), "vision", POLICY)
    # Both releases share byte-identical qualification fixtures, so one cache object.
    shared = v1["manifest"]["artifacts"]["qualification.input"]
    cached = runtime.releases.root / "objects" / shared["sha256"]
    cached.write_bytes(b"\xff" * shared["size"])

    result = runtime.releases.update(feed.publish(v1, feed.release("vision-2", 2)), "vision", POLICY)
    receipts = {r["id"]: r["downloaded_bytes"] for r in result["activation"]["artifact_receipts"]}
    assert receipts["qualification.input"] == shared["size"]
    assert receipts["qualification.output"] == 0
    assert hashlib.sha256(cached.read_bytes()).hexdigest() == shared["sha256"]


def test_reused_release_id_with_different_data_is_a_conflict(runtime, feed, qualification):
    v1 = feed.release("vision-1", 1)
    runtime.releases.update(feed.publish(v1), "vision", POLICY)
    runtime.releases.update(feed.publish(v1, feed.release("vision-2", 2)), "vision", POLICY)
    before = runtime.releases.state("vision")
    redefined = feed.release("vision-1", 3, weights=b"different")

    with pytest.raises(RuntimeFault) as error:
        runtime.releases.update(feed.publish(redefined), "vision",
                                POLICY.model_copy(update={"allow_fallback": True}))
    assert error.value.code == "release_conflict"
    assert runtime.releases.state("vision") == before


def test_older_catalog_cannot_downgrade_the_active_release(runtime, feed, qualification):
    v1 = feed.release("vision-1", 1)
    old = feed.publish(v1)
    runtime.releases.update(feed.publish(v1, feed.release("vision-2", 2)), "vision", POLICY)
    before = runtime.releases.state("vision")

    assert runtime.releases.update(old, "vision", POLICY)["status"] == "stale_catalog"
    assert runtime.releases.state("vision") == before


def test_untrusted_catalog_key_is_rejected_before_download(runtime, feed, qualification, tmp_path):
    other = Feed(tmp_path / "other")
    url = other.publish(other.release("vision-1", 1))

    with pytest.raises(RuntimeFault) as error:
        runtime.releases.update(url, "vision", POLICY)
    assert error.value.code == "untrusted_catalog"
    assert not (runtime.releases.root / "objects").exists()


def test_open_requalifies_after_a_device_runtime_change(runtime, feed, qualification):
    runtime.releases.update(feed.publish(feed.release("vision-1", 1)), "vision", POLICY)
    fingerprint = runtime.releases.state("vision")["active"]["qualification_fingerprint"]
    runtime.open("vision")
    assert len(qualification[0]) == 1

    runtime.device = {**runtime.device, "runtime_build": "sdk-upgrade"}
    runtime.open("vision")
    assert len(qualification[0]) == 2
    assert runtime.releases.state("vision")["active"]["qualification_fingerprint"] != fingerprint

    qualification[1].add("vision-1")
    runtime.device = {**runtime.device, "runtime_build": "broken-sdk-upgrade"}
    with pytest.raises(RuntimeFault) as error:
        runtime.open("vision")
    assert error.value.code == "qualification_failed"
