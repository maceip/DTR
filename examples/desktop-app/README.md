# Desktop AI Agent & Developer Tool Studio Example

A local desktop application demonstrating embedded **DTR Runtime** on macOS, Linux, and Windows workstations.

## Overview

Desktop and engineering workflows require running local neural models with full access to development tools, while maintaining absolute security against accidental data loss, path traversal, or unverified script execution.

This example demonstrates how desktop software embeds DTR to:
1. **Unify Multi-Engine Inference**: Execute models across portable ExecuTorch (CPU/NPU) and LiteRT without rebuilding the UI or application logic.
2. **Sandbox Workspace Tools**: Provide AI agents with safe file read and patch writing capabilities locked strictly to the project directory (`workspace:`).
3. **Trap Traversal & Escalation**: Path traversal attempts (such as `../../etc/passwd` or symlink escapes) are rejected by the runtime before disk access occurs.
4. **Enforce Least-Privilege Policy**: System execution tools (like arbitrary shell execution) are denied by default under the developer workstation policy (`policy.json`).

## Project Contents

- `app.py`: Standalone Desktop Tool Studio CLI and runtime runner demonstrating sandboxed workspace operations and policy enforcement.
- `policy.json`: Declarative desktop developer workstation policy.
- `test_desktop_app.py`: Automated pytest test suite covering workspace reads/writes, path containment, telemetry queries, and dynamic permission revocation.

## Running the Application

```sh
# Run desktop tool studio actions
uv run python examples/desktop-app/app.py

# Run test suite
uv run pytest examples/desktop-app/test_desktop_app.py
```

## Policy Configuration (`policy.json`)

```json
{
  "version": 1,
  "rules": [
    {"subject": "desktop:developer", "resource": "workspace:file", "action": "read", "effect": "allow"},
    {"subject": "desktop:developer", "resource": "workspace:file", "action": "write", "effect": "allow"},
    {"subject": "desktop:developer", "resource": "system:device", "action": "read", "effect": "allow"}
  ]
}
```
Attempting to invoke an unauthorized tool (e.g. `execute_arbitrary_shell` targeting `system:shell`) is blocked at the gateway level, returning `permission_denied` and signing an auditable refusal receipt.
