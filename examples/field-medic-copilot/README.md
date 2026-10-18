# Field Medic Co-Pilot: Offline Trauma Triage & Safe OTA Releases

An offline emergency medical assistant running on mobile tablets and ruggedized phones (Qualcomm Snapdragon 8 Elite / Android).

## Value Proposition

First responders and search-and-rescue teams routinely operate in zero-connectivity environments (earthquake disaster zones, remote wilderness, underground tunnels, maritime transit). Paramedics need on-device image analysis (burn depth grading, pupil acuity, triage category) paired with local electronic patient health records. 

However, medical clinical guidelines and drug dosage protocols update frequently. In standard mobile architectures, deploying an updated neural model requires a weeks-long App Store review, risking deployment of untested weights that can crash during an emergency.

**Field Medic Co-Pilot** demonstrates how DTR solves this:
1. **100% Offline Clinical Vision**: Uses on-device hardware accelerators (LiteRT / ExecuTorch NPU) to evaluate trauma injuries in **<15 ms** with zero network connectivity.
2. **Safe Over-The-Air (OTA) Model Releases**: When an ambulance connects to hospital Wi-Fi, DTR’s `ReleasePolicy` downloads candidate model packages and executes **on-device qualification checks** against clinical test cases. If a candidate degrades below the contract accuracy threshold, DTR rejects the release and **keeps the working release active**.
3. **HIPAA Least-Privilege Gate**: The AI assistant can read `patient:vitals` and chart administered medications, but cannot exfiltrate unredacted Protected Health Information (PHI) over untrusted networks (`network:cloud -> export` is blocked).
4. **Instant Remote Revocation**: If a tablet is stolen or an EMS responder’s credentials expire, dispatch broadcasts a policy update that revokes local tools within 1 ms.

## Running the Application

```sh
# Run offline triage assessment
uv run python examples/field-medic-copilot/app.py --patient "PT-1042"

# Run automated test suite
uv run pytest examples/field-medic-copilot/test_field_medic.py
```

## Policy Configuration (`policy.json`)

```json
{
  "version": 1,
  "rules": [
    {"subject": "paramedic:responder", "resource": "medical:wound", "action": "read", "effect": "allow"},
    {"subject": "paramedic:responder", "resource": "patient:vitals", "action": "read", "effect": "allow"},
    {"subject": "paramedic:responder", "resource": "patient:chart", "action": "write", "effect": "allow"}
  ]
}
```
Attempting to export unredacted patient data outside the paramedic boundary fails with `permission_denied` before any external network connection is established.
