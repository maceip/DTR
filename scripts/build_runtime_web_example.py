#!/usr/bin/env python3
"""Package the upstream web-compatible Gemma artifact for local LiteRT-LM.

This is an additional web example. The native SmolLM package remains the
two-backend conversation example; its native artifact is not web compatible.
"""
import argparse
import json
import shutil
from pathlib import Path

from phaseone.runtime.build import record
from phaseone.runtime.package import atomic_json, load_package


def build(source: Path, destination: Path):
    provenance = json.loads((source.parent / 'source.json').read_text())
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copyfile(source, target)
    artifact = record(target, destination)
    expected = next(item for item in provenance['files'] if item['path'] == source.name)
    if artifact['sha256'] != expected['sha256']:
        raise ValueError('Web model does not match its pinned download receipt')
    manifest = {
        'schema_version': 1, 'name': 'web-assistant', 'version': '1.0.0',
        'description': 'Gemma 4 E2B running locally through LiteRT-LM WebGPU',
        'artifacts': {'gemma': artifact},
        'models': {'assistant': {'task': 'chat', 'variants': [{
            'id': 'litert-web-gpu', 'backend': 'litert', 'engine': 'llm', 'artifact': 'gemma',
            'target': {'systems': ['web'], 'architectures': ['wasm32'], 'accelerator': 'gpu',
                       'runtime_version': '>=0.18.1,<0.19'},
            'chat': {'context_length': 4096, 'modalities': ['text'], 'tool_protocol': 'native'},
        }]}},
        'permissions': ['device.read'],
        'tools': [{'name': 'device_info', 'description': 'Read the actual paired host device and installed inference engines.',
                   'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                   'permission': 'device.read', 'side_effect': False}],
        'agent': {'model': 'assistant', 'instruction': 'Answer concisely. Use device_info before reporting facts about the paired device; do not guess its operating system.',
                  'tools': ['device_info'], 'max_steps': 4},
        'metadata': {'source': provenance, 'license': 'Gemma Terms of Use',
                     'license_url': 'https://ai.google.dev/gemma/terms',
                     'support_boundary': 'Upstream LiteRT-LM JS 0.18.1 supports this web-specific artifact; native .litertlm files are not interchangeable.'},
    }
    atomic_json(destination / 'phaseone.json', manifest)
    package = load_package(destination)
    print(json.dumps({'package': package.manifest.name, 'digest': package.digest, 'artifact_bytes': artifact['size']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('.work/gemma-web/gemma-4-E2B-it-web.litertlm'))
    parser.add_argument('--output', type=Path, default=Path('.work/web-chat-package'))
    args = parser.parse_args()
    build(args.source, args.output)
