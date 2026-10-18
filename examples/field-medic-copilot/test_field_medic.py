import importlib.util
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location("app_field_medic", Path(__file__).parent / "app.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FieldMedicService = _mod.FieldMedicService


def test_field_medic_casualty_triage_flow(tmp_path):
    medic = FieldMedicService(state_dir=tmp_path)
    result = medic.evaluate_casualty("PT-1042")

    assert result["patient_id"] == "PT-1042"
    assert result["triage"]["triage_category"] == "YELLOW_DELAYED"
    assert "Fentanyl" in result["dose"]["entry"]["medication"]
    assert "receipt" in result


def test_field_medic_phi_leak_denial(tmp_path):
    medic = FieldMedicService(state_dir=tmp_path)
    leak_res = medic.test_unauthorized_phi_leak()

    assert leak_res["status"] == "denied"
    assert leak_res["error"]["code"] == "permission_denied"


def test_field_medic_remote_revocation(tmp_path):
    medic = FieldMedicService(state_dir=tmp_path)

    # Dispatch remotely revokes paramedic chart write permissions
    medic.policy.replace({
        "version": 1,
        "rules": [
            {"subject": "paramedic:responder", "resource": "medical:wound", "action": "read", "effect": "allow"},
            # patient:chart write revoked!
        ]
    })

    # Subsequent charting must be denied
    res = medic.runtime.tool_gateway.invoke(
        medic.package,
        "record_administered_dose",
        {"patient_id": "PT-1042", "medication": "Morphine", "dosage_mg": 4.0},
        action_id="revoked-dose-attempt",
    )
    assert res["status"] == "denied"
    assert res["error"]["code"] == "permission_denied"


def test_field_medic_ota_policy():
    medic = FieldMedicService()
    policy = medic.get_ota_clinical_policy()
    assert policy.contract == "trauma-triage.v1"
    assert len(policy.allowed_lineages) >= 1
