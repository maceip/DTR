#!/usr/bin/env python3
"""Prove both runtime backends execute on the explicitly selected physical board."""

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from phaseone.runtime import RemoteRuntime, Runtime
from phaseone.runtime.package import load_package
from phaseone.runtime.tensors import decode_tensor

ROOT = Path(__file__).resolve().parents[1]


def check(args):
    if args.serial.startswith(('emulator-', '-')) or any(c.isspace() for c in args.serial):
        raise SystemExit('Select the physical VENTUNO serial, not an emulator.')
    adb = ['adb', '-s', args.serial]
    subprocess.run(adb + ['get-state'], check=True, capture_output=True, text=True)
    identity = subprocess.check_output(adb + ['shell', 'uname -s; uname -m; cat /etc/os-release'], text=True)
    if not identity.startswith('Linux\naarch64\n') or 'ID=ubuntu' not in identity:
        raise SystemExit('This check requires the physical Ubuntu ARM64 VENTUNO.')
    mappings = subprocess.check_output(['adb', 'forward', '--list'], text=True)
    if [args.serial, f'tcp:{args.port}', 'tcp:8920'] not in [line.split() for line in mappings.splitlines()]:
        raise SystemExit('The selected local port is not forwarded to this board. Run deploy_runtime.py --start.')
    photo = args.image.read_bytes()
    local_package = load_package(args.package)
    with Runtime(ROOT / '.work/board-reference') as reference_runtime:
        reference = reference_runtime.load(args.package).classify('vision', photo, backend='executorch', accelerator='cpu')
        expected = decode_tensor(reference['outputs']['logits'])
    with RemoteRuntime(f'http://127.0.0.1:{args.port}', args.token_file.read_text().strip()) as runtime:
        device = runtime.device()
        if (device['system'], device['architecture']) != ('linux', 'arm64'):
            raise SystemExit('The paired HTTP host does not identify as Linux ARM64.')
        packages = [item for item in runtime.packages() if item['digest'] == local_package.digest]
        if len(packages) != 1:
            raise SystemExit('Deploy the same vision package selected with --package before this check.')
        app = runtime.application(packages[0]['digest'])
        results, arrays = {}, {}
        for backend in ('executorch', 'litert'):
            result = app.classify('vision', photo, backend=backend, accelerator='cpu')
            execution = result['execution']
            assert execution['execution_verified'] and execution['backend'] == backend
            assert execution['location'] == 'remote_host'
            assert result['predictions'][0]['label'] == 'Samoyed', 'Use the included sample-dog.jpg'
            arrays[backend] = decode_tensor(result['outputs']['logits'])
            results[backend] = {'execution': execution, 'predictions': result['predictions']}
        np.testing.assert_allclose(arrays['litert'], arrays['executorch'], rtol=1e-3, atol=2e-4)
        np.testing.assert_allclose(arrays['executorch'], expected, rtol=1e-3, atol=2e-4)
        evidence = {
            'passed': True, 'checked_at': datetime.datetime.now(datetime.UTC).isoformat(),
            'serial': args.serial, 'device': device, 'package': app.digest,
            'image_sha256': hashlib.sha256(photo).hexdigest(), 'results': results,
            'maximum_absolute_error': float(np.max(np.abs(arrays['litert'] - arrays['executorch']))),
            'mac_reference_maximum_absolute_error': float(np.max(np.abs(arrays['executorch'] - expected))),
            'reference_execution': reference['execution'],
            'npu_execution_verified': False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'passed': True, 'evidence': str(args.output), 'serial': args.serial}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', default='1050156610')
    parser.add_argument('--port', type=int, default=8921)
    parser.add_argument('--token-file', type=Path, default=ROOT / '.state/runtime-board.token')
    parser.add_argument('--package', type=Path, default=ROOT / '.work/vision')
    parser.add_argument('--image', type=Path, default=ROOT / 'examples/vision/sample-dog.jpg')
    parser.add_argument('--output', type=Path, default=ROOT / '.work/runtime-board.json')
    check(parser.parse_args())
