import importlib.util
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location("app_mobile", Path(__file__).parent / "app.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MobileAssistantService = _mod.MobileAssistantService


def test_mobile_assistant_normal_flow(tmp_path):
    service = MobileAssistantService(state_dir=tmp_path)

    # 1. Scan photo
    photo_res = service.scan_user_photo("sample_dog_001.jpg")
    assert photo_res["status"] == "complete"
    assert "dog" in photo_res["result"]["detected_objects"]

    # 2. Access note
    note_res = service.access_note("vacation")
    assert note_res["status"] == "complete"
    assert "Flight" in note_res["result"]["content"]

    # 3. Save new note
    save_res = service.save_note("meeting", "Review hackathon deliverables at 12:00")
    assert save_res["status"] == "complete"
    assert service.user_notes["meeting"] == "Review hackathon deliverables at 12:00"


def test_mobile_assistant_unauthorized_access(tmp_path):
    service = MobileAssistantService(state_dir=tmp_path)

    # Contacts access is unauthorized under the mobile user policy
    contacts_res = service.test_unauthorized_contacts_access()
    assert contacts_res["status"] == "denied"
    assert contacts_res["error"] is not None
    assert contacts_res["error"]["code"] == "permission_denied"


def test_mobile_assistant_permission_revocation(tmp_path):
    service = MobileAssistantService(state_dir=tmp_path)

    # First access succeeds
    assert service.access_note("vacation")["status"] == "complete"

    # User revokes notes permission in app settings
    service.policy.replace({
        "version": 1,
        "rules": [
            {"subject": "mobile:user", "resource": "photo:scan", "action": "read", "effect": "allow"},
            # notes:personal is removed!
        ]
    })

    # Subsequent access must be denied immediately
    revoked_res = service.access_note("vacation")
    assert revoked_res["status"] == "denied"
    assert revoked_res["error"]["code"] == "permission_denied"


def test_mobile_release_policy_configuration(tmp_path):
    service = MobileAssistantService(state_dir=tmp_path)
    policy = service.get_release_policy()
    assert policy.contract == "imagenet1k.rgb224.v1"
    assert len(policy.allowed_lineages) >= 1
