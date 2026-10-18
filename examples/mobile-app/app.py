"""Mobile Edge AI Assistant: On-Device Vision, OTA Releases & Tool Gateway.

Demonstrates mobile client integration with DTR runtime, showcasing on-device
local inference, safe OTA release qualification/updates, and fine-grained
tool permissions for smartphone and wearable applications.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dtr import PermissionPolicy, ReleasePolicy, Runtime, RuntimeFault
from phaseone.runtime.schema import ToolDefinition

ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mobile-assistant")


class MobileAssistantService:
    """Mobile edge assistant managing local AI models, OTA updates, and permissions."""

    def __init__(
        self,
        policy_path: Path | None = None,
        runtime: Runtime | None = None,
        subject: str = "mobile:user",
        state_dir: Path | None = None,
    ) -> None:
        policy_file = policy_path or (ROOT / "policy.json")
        policy_doc = json.loads(policy_file.read_text(encoding="utf-8"))
        self.policy = PermissionPolicy(policy_doc)
        self.subject = subject
        self.runtime = runtime or Runtime(
            state_dir=state_dir or (ROOT / ".state"),
            permissions=self.policy,
            subject=self.subject,
        )
        self.user_notes: dict[str, str] = {
            "vacation": "Flight leaves at 08:30 on Friday.",
            "groceries": "Oat milk, coffee beans, dark chocolate.",
        }
        self.turn_count = 0
        self._register_mobile_tools()

    def _register_mobile_tools(self) -> None:
        """Register host mobile tools with strict resource bindings."""
        # 1. Photo Scanner Tool
        scan_def = {
            "name": "scan_photo",
            "description": "Scan and analyze an image from the mobile photo library",
            "permission": "photo.read",
            "parameters": {
                "type": "object",
                "properties": {"photo_id": {"type": "string"}},
                "required": ["photo_id"],
                "additionalProperties": False,
            },
        }

        def scan_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
            return {
                "photo_id": args["photo_id"],
                "detected_objects": ["dog", "park", "tennis ball"],
                "confidence": 0.94,
            }

        self.runtime.tool_gateway.register(
            scan_def,
            scan_handler,
            access=lambda args: [{"resource": "photo:scan", "action": "read"}],
        )

        # 2. Read Personal Note Tool
        read_note_def = {
            "name": "read_personal_note",
            "description": "Read a personal note from the device note vault",
            "permission": "notes.read",
            "parameters": {
                "type": "object",
                "properties": {"note_key": {"type": "string"}},
                "required": ["note_key"],
                "additionalProperties": False,
            },
        }

        def read_note_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
            key = args["note_key"]
            if key not in self.user_notes:
                raise RuntimeFault("not_found", f"Note '{key}' not found")
            return {"note_key": key, "content": self.user_notes[key]}

        self.runtime.tool_gateway.register(
            read_note_def,
            read_note_handler,
            access=lambda args: [{"resource": "notes:personal", "action": "read"}],
        )

        # 3. Save Personal Note Tool
        save_note_def = {
            "name": "save_personal_note",
            "description": "Write or update a personal note in the device vault",
            "permission": "notes.write",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_key": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["note_key", "content"],
                "additionalProperties": False,
            },
            "side_effect": True,
        }

        def save_note_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
            self.user_notes[args["note_key"]] = args["content"]
            return {"status": "saved", "note_key": args["note_key"]}

        self.runtime.tool_gateway.register(
            save_note_def,
            save_note_handler,
            access=lambda args: [{"resource": "notes:personal", "action": "write"}],
        )

        # 4. Contacts Exfiltration Tool (Denied by default mobile policy!)
        contacts_def = {
            "name": "export_contacts",
            "description": "Export the user's global address book",
            "permission": "contacts.read",
            "parameters": {
                "type": "object",
                "properties": {"format": {"type": "string"}},
                "additionalProperties": False,
            },
        }

        def contacts_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
            raise RuntimeError("CRITICAL: Unauthorized address book access!")

        self.runtime.tool_gateway.register(
            contacts_def,
            contacts_handler,
            access=lambda args: [{"resource": "contacts:global", "action": "read"}],
        )

        self.tool_definitions = [scan_def, read_note_def, save_note_def, contacts_def]
        self.package = SimpleNamespace(
            digest="mobile-assistant-v1",
            name="mobile-assistant",
            manifest=SimpleNamespace(
                tools=[ToolDefinition(**d) for d in self.tool_definitions]
            ),
        )

    def scan_user_photo(self, photo_id: str, action_id: str | None = None) -> dict[str, Any]:
        """Request photo scan tool invocation through the permission gateway."""
        import uuid
        aid = action_id or f"mobile-photo-{uuid.uuid4().hex[:8]}"
        logger.info("Executing photo scan for photo_id=%s...", photo_id)
        result = self.runtime.tool_gateway.invoke(
            self.package,
            "scan_photo",
            {"photo_id": photo_id},
            action_id=aid,
        )
        return result

    def access_note(self, note_key: str, action_id: str | None = None) -> dict[str, Any]:
        """Read a personal note via tool gateway."""
        import uuid
        aid = action_id or f"mobile-note-{uuid.uuid4().hex[:8]}"
        logger.info("Accessing personal note note_key=%s...", note_key)
        result = self.runtime.tool_gateway.invoke(
            self.package,
            "read_personal_note",
            {"note_key": note_key},
            action_id=aid,
        )
        return result

    def save_note(self, note_key: str, content: str, action_id: str | None = None) -> dict[str, Any]:
        """Save a personal note via tool gateway."""
        import uuid
        aid = action_id or f"mobile-save-{uuid.uuid4().hex[:8]}"
        logger.info("Saving personal note note_key=%s...", note_key)
        result = self.runtime.tool_gateway.invoke(
            self.package,
            "save_personal_note",
            {"note_key": note_key, "content": content},
            action_id=aid,
        )
        return result

    def test_unauthorized_contacts_access(self, action_id: str | None = None) -> dict[str, Any]:
        """Verify that unauthorized address book access is denied."""
        import uuid
        aid = action_id or f"mobile-contacts-{uuid.uuid4().hex[:8]}"
        logger.info("Testing unauthorized contacts export...")
        result = self.runtime.tool_gateway.invoke(
            self.package,
            "export_contacts",
            {},
            action_id=aid,
        )
        return {"status": result["status"], "error": result.get("error")}

    def get_release_policy(self) -> ReleasePolicy:
        """Create mobile release policy enforcing signed updates and qualification."""
        return ReleasePolicy(
            contract="imagenet1k.rgb224.v1",
            allowed_lineages=["torchvision.mobilenet-v3-small.v2", "torchvision.mobilenet-v3-small.v1"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mobile Edge AI Assistant Example")
    parser.add_argument("--photo", default="img_0492.jpg", help="Sample photo ID")
    args = parser.parse_args()

    service = MobileAssistantService()
    logger.info("Initialized Mobile Edge Assistant for subject '%s'", service.subject)

    # 1. Photo scan
    scan = service.scan_user_photo(args.photo)
    logger.info("Photo scan result: %s", scan["result"])

    # 2. Note access
    note = service.access_note("vacation")
    logger.info("Note read result: %s", note["result"])

    # 3. Unauthorized access check
    contacts = service.test_unauthorized_contacts_access()
    logger.info("Contacts access attempt status: %s (Denied as expected)", contacts["status"])
    assert contacts["status"] == "denied", "Security boundary failure!"

    # 4. Release Policy inspect
    policy = service.get_release_policy()
    logger.info("Configured Mobile OTA Release Policy: contract=%s lineages=%s", policy.contract, policy.allowed_lineages)


if __name__ == "__main__":
    main()
