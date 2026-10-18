#!/usr/bin/env python3
"""Exercise packaging, signing, import, execution and benchmark CLI paths with real models."""

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from phaseone.runtime.package import digest_file, load_package
from phaseone.runtime.tensors import decode_tensor, encode_tensor, preprocess_image

ROOT = Path(__file__).resolve().parents[1]


def check(source, output):
    source = source.resolve()
    prepared = load_package(source)
    build_metadata = prepared.manifest.metadata['builds']
    assert all(item['source_sha256'] == digest_file(ROOT / 'examples/vision/model.py') for item in build_metadata.values())
    photo = (ROOT / 'examples/vision/sample-dog.jpg').read_bytes()
    tensor = encode_tensor(preprocess_image(photo, prepared.manifest.models['vision'].image))
    checks = []
    with tempfile.TemporaryDirectory(prefix='workflow-', dir=ROOT / '.work') as temporary:
        work = Path(temporary)
        state = work / 'state'

        def cli(*arguments):
            result = subprocess.run([sys.executable, '-m', 'phaseone.runtime.cli', *map(str, arguments),
                                     '--state-dir', str(state)], capture_output=True, text=True, timeout=240, cwd=ROOT)
            if result.returncode:
                raise RuntimeError(f'CLI {arguments[0]} failed: {result.stderr}')
            return json.loads(result.stdout)

        package = work / 'source'
        shutil.copytree(source, package)
        key = work / 'package.key'
        cli('keygen', key, '--trust')
        cli('sign', package, '--key', key)
        archive = work / 'application.phaseone'
        packed = cli('pack', package, archive)
        for command in ('inspect', 'verify', 'install'):
            result = cli(command, archive, '--require-signature')
            assert result['digest'] == packed['package_digest'] and result['signature'] == 'trusted'
        checks.append({'kind': 'signed_archive_workflow', 'commands': ['keygen', 'sign', 'pack', 'inspect', 'verify', 'install'],
                       'package': packed['package_digest']})
        inputs = work / 'inputs.json'
        inputs.write_text(json.dumps({'image': tensor}))
        arrays = {}
        for backend in ('executorch', 'litert'):
            result = cli('run', archive, '--model', 'vision', '--backend', backend, '--inputs', inputs, '--require-signature')
            arrays[backend] = decode_tensor(result['outputs']['logits'])
            assert result['execution']['execution_verified'] and result['execution']['backend'] == backend
            prediction = cli('classify', archive, '--model', 'vision', '--backend', backend,
                             '--image', ROOT / 'examples/vision/sample-dog.jpg', '--require-signature')
            assert prediction['predictions'][0]['label'] == 'Samoyed'
            benchmark = cli('bench', archive, '--model', 'vision', '--backend', backend, '--inputs', inputs,
                            '--repeat', '2', '--warmup', '1', '--require-signature')
            assert benchmark['iterations'] == 2 and benchmark['last_execution']['backend'] == backend
            checks.append({'kind': 'run_classify_bench', 'backend': backend, 'execution': result['execution'],
                           'top_label': prediction['predictions'][0]['label'], 'benchmark': benchmark})
            imported = work / ('import-' + backend)
            cli('import', prepared.artifact('vision.' + backend), imported, '--backend', backend,
                '--name', 'imported-' + backend, '--model', 'vision')
            imported_package = load_package(imported)
            signature = imported_package.manifest.models['vision'].variants[0].signatures['default']
            imported_inputs = work / ('inputs-' + backend + '.json')
            imported_inputs.write_text(json.dumps({signature.inputs[0].name: tensor}))
            imported_result = cli('run', imported, '--model', 'vision', '--backend', backend, '--inputs', imported_inputs)
            np.testing.assert_allclose(decode_tensor(imported_result['outputs'][signature.outputs[0].name]), arrays[backend])
            checks.append({'kind': 'import_native_metadata_and_execute', 'backend': backend, 'execution': imported_result['execution']})
        np.testing.assert_allclose(arrays['litert'], arrays['executorch'], rtol=1e-3, atol=2e-4)
        receipt_check = cli('receipts', '--verify')
        assert receipt_check['valid']
        checks.append({'kind': 'receipts', **receipt_check})
        generated_schema = cli('schema')
        assert generated_schema == json.loads((ROOT / 'sdk/typescript/src/package-schema.json').read_text())
        checks.append({'kind': 'python_typescript_schema_matches', 'passed': True})
    evidence = {'passed': True, 'checked_at': datetime.datetime.now(datetime.UTC).isoformat(),
                'builds': build_metadata, 'checks': checks}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'passed': True, 'checks': len(checks), 'evidence': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=ROOT / '.work/workflow-vision')
    parser.add_argument('--output', type=Path, default=ROOT / 'evidence/runtime-workflow.json')
    arguments = parser.parse_args()
    check(arguments.package, arguments.output)
