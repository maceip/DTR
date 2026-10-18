#!/usr/bin/env python3
"""Run real embedded inference on an explicitly selected physical Android phone."""

import argparse
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ai.phaseone.demo"
RUNNER = "ai.phaseone.demo.test/androidx.test.runner.AndroidJUnitRunner"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--suite", choices=["vision", "chat", "gate"], default="vision")
    parser.add_argument("--accelerator", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--expected-release")
    args = parser.parse_args()
    locks = ROOT / ".work/android-verification-locks"
    locks.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(args.serial.encode()).hexdigest()[:16]
    with sqlite3.connect(locks / f"{identity}.sqlite3", timeout=0.2) as lock:
        try:
            lock.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as failure:
            raise RuntimeError("Another PHASEONE check is active on this phone; run its suites sequentially") from failure
        run(args)


def run(args):
    adb = ["adb", "-s", args.serial]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def command(*parts, timeout=120):
        result = subprocess.run([*adb, *parts], capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr).strip())
        return result.stdout

    if args.serial.startswith("emulator-") or command("shell", "getprop", "ro.kernel.qemu").strip() == "1":
        raise RuntimeError("This check requires a physical phone; an emulator is not device evidence")
    if command("get-state").strip() != "device":
        raise RuntimeError("Unlock the phone and authorize this Mac's USB debugging connection")
    if args.install:
        for apk in ("debug/demo-debug.apk", "androidTest/debug/demo-debug-androidTest.apk"):
            path = ROOT / "sdk/android/demo/build/outputs/apk" / apk
            command("install", "-r", str(path))
    command("reverse", "tcp:8930", "tcp:8930")

    def instrument(test, name, *options):
        remote = f"files/evidence/{name}.json"
        command("shell", "run-as", PACKAGE, "rm", "-f", remote)
        result = command("shell", "am", "instrument", "-w", "-r", "-e", "class", test,
                         *options, RUNNER, timeout=1200)
        (output / f"{name}.instrumentation.txt").write_text(result)
        if "OK (" not in result or "FAILURES!!!" in result or "Process crashed" in result:
            raise RuntimeError(f"{test} did not finish successfully; inspect {output / (name + '.instrumentation.txt')}")
        raw = command("exec-out", "run-as", PACKAGE, "cat", remote)
        report = json.loads(raw)
        (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        if not report.get("passed") or "FAILURES!!!" in result or "Process crashed" in result:
            raise RuntimeError(f"{test} failed; inspect {output / (name + '.json')}")

    if args.suite == "vision":
        options = ["-e", "expectedRelease", args.expected_release] if args.expected_release else []
        instrument("ai.phaseone.demo.RuntimeDeviceTest#localEnginesAndReleaseLifecycle", "runtime-android", *options)
        command("reverse", "--remove", "tcp:8930")
        try:
            instrument("ai.phaseone.demo.RuntimeDeviceTest#offlineLocalInference", "runtime-android-offline")
        finally:
            command("reverse", "tcp:8930", "tcp:8930")
    elif args.suite == "chat":
        instrument("ai.phaseone.demo.ChatDeviceTest#nativeTextConversations", "runtime-android-chat")
    else:
        instrument("ai.phaseone.demo.ChatDeviceTest#gateScreenQualification", f"runtime-android-gate-{args.accelerator}",
                   "-e", "accelerator", args.accelerator)
    print(json.dumps({"passed": True, "suite": args.suite, "output": str(output)}))


if __name__ == "__main__":
    main()
