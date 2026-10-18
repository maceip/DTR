#!/usr/bin/env python3
"""Compare a real image classifier across both engines using identical pixels."""
import argparse
import json
from pathlib import Path

import numpy as np

from phaseone.runtime import Runtime
from phaseone.runtime.tensors import decode_tensor, encode_tensor, preprocess_image


def check(package: Path, image: Path, output: Path):
    with Runtime(Path('.work/vision-contract')) as runtime:
        app = runtime.load(package)
        model = app.package.manifest.models['vision']
        prepared = preprocess_image(image.read_bytes(), model.image)
        # Also used by the browser test to separate engine numerics from the
        # browser canvas versus Pillow resize implementation.
        (output.parent / 'vision-input.json').write_text(json.dumps(encode_tensor(prepared)) + '\n')
        results, arrays = {}, {}
        for backend in ('executorch', 'litert'):
            result = app.classify('vision', image.read_bytes(), backend=backend)
            arrays[backend] = decode_tensor(result['outputs']['logits'])
            results[backend] = {'execution': result['execution'], 'predictions': result['predictions']}
        np.testing.assert_allclose(arrays['litert'], arrays['executorch'], rtol=1e-3, atol=2e-4)
        assert results['executorch']['predictions'][0]['index'] == results['litert']['predictions'][0]['index']
        assert results['executorch']['predictions'][0]['label'] == 'Samoyed', 'Use the documented sample-dog.jpg for this task-level check'
        evidence = {'passed': True, 'package': app.package.digest, 'image': image.name,
                    'maximum_absolute_error': float(np.max(np.abs(arrays['litert'] - arrays['executorch']))), 'results': results}
        output.write_text(json.dumps(evidence, indent=2) + '\n')
        (output.parent / 'vision-expected.json').write_text(json.dumps(encode_tensor(arrays['executorch'])) + '\n')
        print(json.dumps({'passed': True, 'top_label': results['executorch']['predictions'][0]['label'], 'maximum_absolute_error': evidence['maximum_absolute_error']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path('.work/vision-package'))
    parser.add_argument('--image', type=Path, default=Path('examples/vision/sample-dog.jpg'))
    parser.add_argument('--output', type=Path, default=Path('evidence/runtime-vision.json'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    check(args.package, args.image, args.output)
