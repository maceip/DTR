# Airgapped CAD Auditor: Confidential Engineering & ITAR Sandbox

An air-gapped desktop engineering assistant running locally on developer workstations (Apple Silicon macOS, Snapdragon X Elite Windows, Linux).

## Value Proposition

Aerospace and defense contractors (Lockheed Martin, Boeing, Northrop Grumman) and semiconductor design houses work under strict International Traffic in Arms Regulations (ITAR) and export controls. Proprietary Verilog/VHDL code, analog schematic netlists, and PCB layouts can never be transmitted to external cloud LLM APIs.

However, running unconstrained open-source agent harnesses locally creates severe security liabilities: prompt-injection or tool hallucinations can read sensitive private keys (`~/.ssh/id_rsa`), exfiltrate intellectual property, or execute destructive shell commands (`rm -rf`).

**Airgapped CAD Auditor** provides a mathematically enforced solution:
1. **Multi-Engine Desktop AI**: Runs local engineering models across Apple Silicon CPU/Metal or Snapdragon X Elite NPU via unified DTR interfaces without recompiling.
2. **Strict Directory Sandbox**: DTR’s `Workspace` tool abstraction uses POSIX directory descriptors and canonical path verification to ensure that tools can read and write files strictly inside the designated `cad_workspace/` project directory.
3. **Path Traversal Escape Trapping**: Any attempt to read parent paths (e.g. `../../../../etc/shadow` or symlink escapes) is caught at the DTR gateway boundary, throwing `workspace_scope` and aborting before file access occurs.
4. **No Privilege Escalation**: Host shell execution (`system:shell -> execute`) is blocked by default under the workstation policy (`policy.json`). Every permitted tool action logs a cryptographic receipt for compliance audits.

## Running the Application

```sh
# Run CAD schematic audit and netlist patch
uv run python examples/airgapped-cad-auditor/app.py --file "antenna_pll_v2.v"

# Run automated test suite
uv run pytest examples/airgapped-cad-auditor/test_cad_auditor.py
```

## Policy Configuration (`policy.json`)

```json
{
  "version": 1,
  "rules": [
    {"subject": "engineer:cad", "resource": "workspace:cad", "action": "read", "effect": "allow"},
    {"subject": "engineer:cad", "resource": "workspace:cad", "action": "write", "effect": "allow"},
    {"subject": "engineer:cad", "resource": "system:workstation", "action": "read", "effect": "allow"}
  ]
}
```
Tool calls requesting external shell execution or accessing unapproved resources fail with `permission_denied` and generate an auditable non-repudiation receipt.
