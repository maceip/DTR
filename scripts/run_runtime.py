#!/usr/bin/env python3
"""Serve the prepared applications, or report the already running paired host."""

from pathlib import Path

import httpx

from phaseone.runtime.cli import main

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:8920'


def run():
    token = ROOT / '.work/runtime-host/host.token'
    try:
        health = httpx.get(URL + '/health', timeout=2)
    except httpx.ConnectError:
        health = None
    if health is not None:
        if health.is_success and health.json().get('product') == 'PHASEONE Runtime' and token.exists():
            response = httpx.get(URL + '/v1/packages', headers={'Authorization': 'Bearer ' + token.read_text().strip()})
            response.raise_for_status()
            print('PHASEONE Runtime is already running at ' + URL)
            print('Loaded applications: ' + ', '.join(item['name'] for item in response.json()['packages']))
            return
        raise SystemExit('Port 8920 is occupied by another service. Choose a different port with phaseone-runtime serve.')
    packages = [ROOT / '.work' / name for name in ('vision-package', 'chat-package', 'web-chat-package', 'runtime-fixture')]
    packages = [path for path in packages if (path / 'phaseone.json').is_file()]
    if not packages:
        raise SystemExit('No prepared applications found. Run make runtime-build or follow docs/RUNTIME.md.')
    arguments = ['serve', '--state-dir', str(token.parent), '--port', '8920', '--grant', 'device.read']
    for package in packages:
        arguments.extend(['--package', str(package)])
    main(arguments)


if __name__ == '__main__':
    run()
