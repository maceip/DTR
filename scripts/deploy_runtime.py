#!/usr/bin/env python3
"""Deploy the runtime into its own directory on an explicitly selected VENTUNO.

The existing hardware gate and its services are separate. No emulator is an
acceptable deployment target. Installation uses a user-owned virtual environment.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from phaseone.runtime.package import load_package, pack

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--package', action='append', type=Path)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--forward-port', type=int, default=8921)
    parser.add_argument('--grant', action='append', default=[])
    args = parser.parse_args()
    if args.serial.startswith(('emulator-', '-')) or any(character.isspace() for character in args.serial):
        raise SystemExit('Select the physical VENTUNO serial, not an emulator.')
    if not 1024 <= args.forward_port <= 65535:
        raise SystemExit('Choose an unprivileged local forwarding port.')
    sources = args.package or [ROOT / '.work/vision']
    packages = [load_package(source) for source in sources]
    if len({package.manifest.name for package in packages}) != len(packages):
        raise SystemExit('Choose only one version of each named application.')
    adb = ['adb', '-s', args.serial]

    def remote(script, **kwargs):
        return subprocess.run(adb + ['shell', 'sh -lc ' + shlex.quote(script)], check=True, **kwargs)

    identity = remote('uname -s; uname -m; cat /etc/os-release', capture_output=True, text=True).stdout
    if not identity.startswith('Linux\naarch64\n') or 'ID=ubuntu' not in identity:
        raise SystemExit('Expected the Ubuntu ARM64 VENTUNO board; deployment was not attempted.')
    with tempfile.TemporaryDirectory(prefix='phaseone-deploy-') as temporary:
        staging = Path(temporary)
        archive = staging / 'runtime.tar'
        package_archives = []
        for source, package in zip(sources, packages, strict=True):
            print(f'Packing {package.manifest.name}...', file=sys.stderr, flush=True)
            target = staging / (package.manifest.name + '.phaseone')
            pack(source, target)
            package_archives.append(target)
        print('Preparing the board transfer...', file=sys.stderr, flush=True)
        # USB transfer is faster than compressing large model weights on the host.
        with tarfile.open(archive, 'w') as tar:
            for name in ('phaseone', 'dtr', 'pyproject.toml', 'uv.lock', 'README.md', 'docs',
                         'scripts/setup_runtime_board.sh', 'scripts/run_runtime_board.sh'):
                tar.add(ROOT / name, arcname=name, filter=lambda item: None if '__pycache__' in item.name else item)
            for item in package_archives:
                tar.add(item, arcname='apps/' + item.name)
        print(f'Transferring {archive.stat().st_size / 1024**2:.0f} MiB to {args.serial}...',
              file=sys.stderr, flush=True)
        subprocess.run(adb + ['push', str(archive), '/tmp/phaseone-runtime.tar'], check=True)
        remote('set -e\nmkdir -p "$HOME/phaseone-runtime"\ntar -xf /tmp/phaseone-runtime.tar -C "$HOME/phaseone-runtime"\nrm /tmp/phaseone-runtime.tar')
    if args.install:
        print('Installing the board runtime...', file=sys.stderr, flush=True)
        needs_vision = any(variant.chat and variant.chat.processor and variant.chat.processor.runner != 'gemma4-ple' for package in packages
                           for model in package.manifest.models.values() for variant in model.variants)
        remote('cd "$HOME/phaseone-runtime" && bash scripts/setup_runtime_board.sh' + (' --vision' if needs_vision else ''))
    if args.start:
        package_args = ' '.join('--package ' + shlex.quote('apps/' + package.manifest.name + '.phaseone') for package in packages)
        grants = ' '.join('--grant ' + shlex.quote(permission) for permission in args.grant)
        # Refuse to signal any PID unless Linux confirms its cwd and command.
        stop = '''import os, signal, time
from pathlib import Path
pid_file = Path('.state/host.pid')
if pid_file.exists():
    pid = int(pid_file.read_text())
    process = Path('/proc') / str(pid)
    if process.exists():
        command = (process / 'cmdline').read_bytes()
        if (process / 'cwd').resolve() != Path.cwd() or b'phaseone.runtime.cli' not in command or b'serve' not in command:
            raise SystemExit('PID no longer belongs to this runtime; refusing to stop it')
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            if not process.exists():
                break
            time.sleep(0.1)
        else:
            raise SystemExit('Previous runtime did not stop; inspect its log before restarting')
    pid_file.unlink()
'''
        remote('set -e\ncd "$HOME/phaseone-runtime"\nmkdir -p .state\n.venv/bin/python -c ' + shlex.quote(stop))
        remote('set -e\ncd "$HOME/phaseone-runtime"\n'
               'nohup bash scripts/run_runtime_board.sh serve --state-dir .state --host 127.0.0.1 --port 8920 '
               + package_args + ' ' + grants + ' > .state/host.log 2>&1 < /dev/null &\necho $! > .state/host.pid')
    forwards = subprocess.check_output(adb + ['forward', '--list'], text=True)
    mapping = [args.serial, f'tcp:{args.forward_port}', 'tcp:8920']
    if mapping not in [line.split() for line in forwards.splitlines()]:
        subprocess.run(adb + ['forward', '--no-rebind', f'tcp:{args.forward_port}', 'tcp:8920'], check=True)
    result = {'deployed': '~/phaseone-runtime', 'url': f'http://127.0.0.1:{args.forward_port}',
              'packages': [package.digest for package in packages], 'execution_verified': False}
    if args.start:
        import httpx

        deadline = time.monotonic() + 90
        while True:
            try:
                health = httpx.get(result['url'] + '/health', timeout=3).json()
                if health.get('product') == 'PHASEONE Runtime':
                    break
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise SystemExit('Board host did not start; inspect ~/phaseone-runtime/.state/host.log on the board.')
            time.sleep(1)
        token = remote('cat "$HOME/phaseone-runtime/.state/host.token"', capture_output=True).stdout.strip()
        destination = ROOT / '.state/runtime-board.token'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'wb') as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(token + b'\n')
        result['pairing_token_file'] = str(destination)
        # A live health check is not proof of model execution.
        result['host_health_verified'] = True
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
