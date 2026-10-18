"""Shared resolver vectors and hostile/invalid model-delivery boundaries.

Native execution checks live in check_runtime_releases.py and the Android
instrumented suite; the stubbed-qualification lifecycle is in
test_runtime_release_lifecycle.py. These tests exercise contract boundaries.
"""

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from phaseone.runtime.catalog import (
    Catalog,
    ReleasePolicy,
    decode_catalog,
    qualification_fingerprint,
    resolve,
    signed_catalog,
)
from phaseone.runtime.errors import RuntimeFault
from phaseone.runtime.package import canonical
from phaseone.runtime.releases import _json_equal, allowed_url

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "sdk/contracts/resolver-cases.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_shared_android_desktop_resolver_vectors(case):
    assert qualification_fingerprint(case["device"]) == case["expected"]["device_fingerprint"]
    catalog = Catalog.model_validate(case["catalog"])
    assert {r.id: r.digest for r in catalog.releases} == case["expected"]["release_digests"]
    result = resolve(catalog, case["capability"],
                     case["device"], ReleasePolicy.model_validate(case["policy"]), active=case["active"])
    assert result["status"] == case["expected"]["status"]
    assert (result["selection"]["variant"] if result["selection"] else None) == case["expected"]["variant"]
    assert result["app_update_available"] == case["expected"]["app_update_available"]


def test_catalog_signature_requires_an_application_trust_root():
    catalog = Catalog.model_validate(CASES[0]["catalog"])
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes_raw()
    signed = signed_catalog(catalog, key.private_bytes_raw())
    assert decode_catalog(signed, trusted_keys={hashlib.sha256(public).hexdigest(): public}) == catalog
    with pytest.raises(RuntimeFault, match="not trusted"):
        decode_catalog(signed)
    raw = json.loads(signed)
    payload = json.loads(base64.b64decode(raw["payload"]))
    payload["releases"][0]["manifest"]["artifacts"]["vision.executorch"]["sha256"] = "0" * 64
    raw["payload"] = base64.b64encode(canonical(payload)).decode()
    with pytest.raises(RuntimeFault) as error:
        decode_catalog(canonical(raw), trusted_keys={hashlib.sha256(public).hexdigest(): public})
    assert error.value.code == "invalid_catalog"


def test_unsigned_and_duplicate_catalogs_are_rejected_by_default():
    data = canonical(CASES[0]["catalog"])
    with pytest.raises(RuntimeFault) as error:
        decode_catalog(data)
    assert error.value.code == "untrusted_catalog"
    assert decode_catalog(data, allow_unsigned=True).name == "phaseone-vision"
    duplicate = data.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1', 1)
    with pytest.raises(RuntimeFault):
        decode_catalog(duplicate, allow_unsigned=True)


@pytest.mark.parametrize("path", ["plugin.so", "lib.so.2", "plugin.dll", "extension.py", "module.dex", "plugin.aar", "../model.pte", "catalog.json", "activation.json"])
def test_catalog_cannot_ship_native_code_or_overwrite_metadata(path):
    data = copy.deepcopy(CASES[0]["catalog"])
    data["releases"][0]["manifest"]["artifacts"]["vision.executorch"]["path"] = path
    with pytest.raises(ValidationError):
        Catalog.model_validate(data)


def test_variant_qualification_contracts_must_agree():
    data = copy.deepcopy(CASES[0]["catalog"])
    data["releases"][0]["manifest"]["models"]["vision"]["variants"][1]["signatures"]["default"]["outputs"][0]["shape"] = [1, 1001]
    with pytest.raises(ValidationError):
        Catalog.model_validate(data)


def test_qualification_reference_bytes_must_match_tensor_shape():
    data = copy.deepcopy(CASES[0]["catalog"])
    data["releases"][0]["manifest"]["artifacts"]["qualification.output"]["size"] += 4
    with pytest.raises(ValidationError, match="artifact size"):
        Catalog.model_validate(data)


def test_semantic_boolean_expectations_do_not_accept_numeric_substitutes():
    assert not _json_equal(1, True)
    assert not _json_equal(False, 0)
    assert _json_equal(1.0, 1)


def test_fallback_cannot_relax_application_lineage_or_accelerator_policy():
    case = CASES[0]
    catalog = Catalog.model_validate(case["catalog"])
    policy = ReleasePolicy.model_validate({**case["policy"], "backend": "executorch", "allow_fallback": True})
    result = resolve(catalog, case["capability"], case["device"], policy,
                     excluded={("vision-1", "executorch-cpu")})
    assert result["selection"] is None
    assert result["status"] == "no_compatible_release"


def test_npu_candidates_require_native_components_not_just_a_chip_name():
    data = copy.deepcopy(CASES[0]["catalog"])
    data["releases"][0]["manifest"]["models"]["vision"]["variants"][0]["target"]["accelerator"] = "npu"
    with pytest.raises(ValidationError, match="required native components"):
        Catalog.model_validate(data)


def test_required_artifact_closure_excludes_other_engine_weights():
    release = Catalog.model_validate(CASES[0]["catalog"]).releases[0]
    variant = release.manifest.models["vision"].variants[0]
    closure = release.artifact_ids(variant)
    assert closure == {"vision.executorch", "vision.labels", "qualification.input", "qualification.output"}


@pytest.mark.parametrize("url", ["http://models.example/model.json", "file:///tmp/models.json", "http://127.0.0.1/models.json", "https://user:password@example.com/models.json"])
def test_production_catalog_transport(url):
    with pytest.raises(RuntimeFault):
        allowed_url(url)


def test_local_development_is_explicit_and_does_not_allow_arbitrary_cleartext():
    allowed_url("https://models.example/catalog.json")
    allowed_url("http://127.0.0.1:8930/catalog.json", local=True)
    allowed_url("file:///tmp/catalog.json", local=True)
    with pytest.raises(RuntimeFault):
        allowed_url("http://127.0.0.1.evil.example/catalog.json", local=True)
