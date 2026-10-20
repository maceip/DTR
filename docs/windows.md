# LocalKit on Windows

Windows is a development and deployment target. This checkout has **not been
qualified on a Windows machine**; successful macOS or Linux execution is not
Windows evidence. The Python application API exists, but Windows packaging,
private-file handling, native workers, and release lifecycle need verification.

## Planned native integration

Use Windows ML where its device and OS requirements are met. Its execution
provider catalog can acquire supported native CPU/GPU/NPU implementations while
the app retains the same LocalKit capability interface. LocalKit Model CDN should deliver
the matching model and assets. These are planned LocalKit adapters, not implemented
Windows functionality in this checkout.

[Windows ML's managed providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/windows-ml-eps-vs-bring-your-own)
require Windows 11 24H2+ for catalog acquisition and depend on supported hardware
and drivers. CPU and DirectML have a different availability boundary. Compatible
provider updates arrive through Windows Update; applications cannot force a new
provider version to install. A bundled fallback may be appropriate for offline
or managed machines.

The reference ARM64 laptop target is Snapdragon X Elite / SC8380XP, Windows ARM64. An
ordinary Windows x64 laptop is useful for installation, CPU, packaging, and
lifecycle tests, but cannot establish Qualcomm NPU execution. The current
ExecuTorch/LiteRT Python package choices do not establish a native Windows ARM64
wheel path; that target needs its native runtime integration.

## Acceptance checks before claiming support

1. Install the SDK and open a compatible model on a real Windows host.
2. Run the same task contract as the macOS/Android examples and record the actual
   execution provider and accelerator, including fallback behavior.
3. Exercise model download, activation, offline opening, and rollback.
4. Validate a compatible provider update without changing the consuming app.
5. Repeat the NPU path on the actual Snapdragon laptop.

Model and engine updates should run at application lifecycle boundaries. LocalKit
does not require an always-running Windows service.
