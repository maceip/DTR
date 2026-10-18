"""Desktop AI Agent & Developer Tool Studio: Multi-Backend Inference & Sandboxed Tools.

Demonstrates embedding DTR runtime on desktop environments (macOS, Linux, Windows),
featuring sandboxed workspace tools, policy-based least-privilege permissions,
and unified model inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dtr import PermissionPolicy, Runtime, RuntimeFault
from phaseone.runtime.schema import ToolDefinition

ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("desktop-studio")


class DesktopToolStudio:
    """Desktop assistant providing sandboxed development tools and multi-engine AI."""

    def __init__(
        self,
        workspace_dir: Path | None = None,
        policy_path: Path | None = None,
        runtime: Runtime | None = None,
        subject: str = "desktop:developer",
        state_dir: Path | None = None,
    ) -> None:
        self.workspace = workspace_dir or (ROOT / "workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
        policy_file = policy_path or (ROOT / "policy.json")
        policy_doc = json.loads(policy_file.read_text(encoding="utf-8"))
        self.policy = PermissionPolicy(policy_doc)
        self.subject = subject
        self.runtime = runtime or Runtime(
            state_dir=state_dir or (ROOT / ".state"),
            permissions=self.policy,
            subject=self.subject,
            workspace=self.workspace,
        )
        self._register_desktop_tools()

    def _register_desktop_tools(self) -> None:
        """Register host desktop tools with scoped resource bindings."""
        # 1. Read Workspace File
        read_file_def = {
            "name": "read_workspace_file",
            "description": "Read a text document within the sandboxed project workspace",
            "permission": "workspace.read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        }

        def read_file_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
            rel_path = args["path"].lstrip("/")
            target = (self.workspace / rel_path).resolve()
            # Ensure path does not escape workspace directory
            if not str(target).startswith(str(self.workspace.resolve())):
                raise RuntimeFault("workspace_scope", f"Path '{args['path']}' escapes workspace boundary")
            if not target.exists():
                raise RuntimeFault("not_found", f"File '{args['path']}' does not exist")
            return {"path": args["path"], "content": target.read_text(encoding="utf-8")}

        self.runtime.tool_gateway.register(
            read_file_def,
            read_file_handler,
            access=lambda args: [{"resource": "workspace:file", "action": "read"}],
        )

        # 2. Write Workspace Patch
        write_file_def = {
            "name": "write_workspace_patch",
            "description": "Write or update a file patch inside the sandboxed project workspace",
            "permission": "workspace.write",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            "side_effect": True,
        }

        def write_file_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
            rel_path = args["path"].lstrip("/")
            target = (self.workspace / rel_path).resolve()
            if not str(target).startswith(str(self.workspace.resolve())):
                raise RuntimeFault("workspace_scope", f"Path '{args['path']}' escapes workspace boundary")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args["content"], encoding="utf-8")
            return {"path": args["path"], "bytes_written": len(args["content"].encode("utf-8")), "status": "written"}

        self.runtime.tool_gateway.register(
            write_file_def,
            write_file_handler,
            access=lambda args: [{"resource": "workspace:file", "action": "write"}],
        )

        # 3. System Device Telemetry
        device_def = {
            "name": "get_device_telemetry",
            "description": "Query host workstation hardware and platform telemetry",
            "permission": "system.read",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }

        def device_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
            dev = self.runtime.device
            return {
                "os": dev.get("os", "unknown") if isinstance(dev, dict) else getattr(dev, "os", "unknown"),
                "arch": dev.get("arch", "unknown") if isinstance(dev, dict) else getattr(dev, "arch", "unknown"),
                "model": dev.get("model", "unknown") if isinstance(dev, dict) else getattr(dev, "model", "unknown"),
                "soc": dev.get("soc", "unknown") if isinstance(dev, dict) else getattr(dev, "soc", "unknown"),
            }

        self.runtime.tool_gateway.register(
            device_def,
            device_handler,
            access=lambda args: [{"resource": "system:device", "action": "read"}],
        )

        # 4. Arbitrary Shell Execution (Forbidden by default policy!)
        shell_def = {
            "name": "execute_arbitrary_shell",
            "description": "Execute arbitrary shell commands on host machine",
            "permission": "system.execute",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
            "side_effect": True,
        }

        def shell_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
            raise RuntimeError("CRITICAL: Arbitrary host shell execution was attempted!")

        self.runtime.tool_gateway.register(
            shell_def,
            shell_handler,
            access=lambda args: [{"resource": "system:shell", "action": "execute"}],
        )

        self.tool_definitions = [read_file_def, write_file_def, device_def, shell_def]
        self.package = SimpleNamespace(
            digest="desktop-studio-v1",
            name="desktop-studio",
            manifest=SimpleNamespace(
                tools=[ToolDefinition(**d) for d in self.tool_definitions]
            ),
        )

    def write_file(self, rel_path: str, content: str, action_id: str | None = None) -> dict[str, Any]:
        """Write content to a file inside the sandboxed workspace."""
        aid = action_id or f"desktop-write-{uuid.uuid4().hex[:8]}"
        logger.info("Writing workspace file '%s'...", rel_path)
        return self.runtime.tool_gateway.invoke(
            self.package,
            "write_workspace_patch",
            {"path": rel_path, "content": content},
            action_id=aid,
        )

    def read_file(self, rel_path: str, action_id: str | None = None) -> dict[str, Any]:
        """Read content from a workspace file."""
        aid = action_id or f"desktop-read-{uuid.uuid4().hex[:8]}"
        logger.info("Reading workspace file '%s'...", rel_path)
        return self.runtime.tool_gateway.invoke(
            self.package,
            "read_workspace_file",
            {"path": rel_path},
            action_id=aid,
        )

    def get_system_telemetry(self, action_id: str | None = None) -> dict[str, Any]:
        """Query host workstation device info."""
        aid = action_id or f"desktop-device-{uuid.uuid4().hex[:8]}"
        return self.runtime.tool_gateway.invoke(
            self.package,
            "get_device_telemetry",
            {},
            action_id=aid,
        )

    def test_unauthorized_shell(self, action_id: str | None = None) -> dict[str, Any]:
        """Attempt arbitrary host command execution (should be blocked by policy)."""
        aid = action_id or f"desktop-shell-{uuid.uuid4().hex[:8]}"
        logger.info("Testing unauthorized shell execution attempt...")
        result = self.runtime.tool_gateway.invoke(
            self.package,
            "execute_arbitrary_shell",
            {"command": "rm -rf /"},
            action_id=aid,
        )
        return {"status": result["status"], "error": result.get("error")}

    def test_path_traversal(self, escape_path: str = "../../../etc/passwd", action_id: str | None = None) -> dict[str, Any]:
        """Attempt workspace sandbox traversal."""
        aid = action_id or f"desktop-escape-{uuid.uuid4().hex[:8]}"
        logger.info("Testing workspace path traversal with '%s'...", escape_path)
        result = self.runtime.tool_gateway.invoke(
            self.package,
            "read_workspace_file",
            {"path": escape_path},
            action_id=aid,
        )
        return {"status": result["status"], "error": result.get("error")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop AI Agent & Tool Studio")
    parser.add_argument("--file", default="demo.txt", help="Sample workspace filename")
    parser.add_argument("--content", default="Hello from DTR Desktop Studio!\n", help="File content")
    args = parser.parse_args()

    studio = DesktopToolStudio()
    logger.info("Initialized Desktop Studio (Workspace: %s, Subject: %s)", studio.workspace, studio.subject)

    # 1. Write file
    write_res = studio.write_file(args.file, args.content)
    logger.info("Write result: %s", write_res["result"])

    # 2. Read file
    read_res = studio.read_file(args.file)
    logger.info("Read result: %s", read_res["result"])

    # 3. System info
    telem = studio.get_system_telemetry()
    logger.info("Workstation telemetry: %s", telem["result"])

    # 4. Unauthorized shell check
    shell_res = studio.test_unauthorized_shell()
    logger.info("Shell execution status: %s (Blocked as expected)", shell_res["status"])
    assert shell_res["status"] == "denied", "Security policy failed to block shell!"

    # 5. Path traversal check
    traversal_res = studio.test_path_traversal()
    logger.info("Traversal status: %s (Sandbox contained)", traversal_res["status"])
    assert traversal_res["status"] == "failed" and traversal_res["error"]["code"] == "workspace_scope", "Sandbox escape succeeded!"

    logger.info("All desktop studio security and execution checks passed.")


if __name__ == "__main__":
    main()
