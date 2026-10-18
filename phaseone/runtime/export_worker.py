"""Standalone converter entry point, invoked only by an explicit developer build.

Intentionally uses no PHASEONE imports: converter environments have conflicting
PyTorch requirements and do not need the application host installed.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path


def export(source, factory, backend, destination, report):
    import numpy as np
    import torch

    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("phaseone_developer_model", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    definition = getattr(module, factory)()
    if isinstance(definition, tuple):
        model, arguments = definition
        definition = {"signatures": {"default": {"model": model,
                      "inputs": {f"input_{i}": value for i, value in enumerate(arguments)}}}}
    signatures = definition["signatures"]
    contracts, expected, examples, exports = {}, {}, {}, {}
    converter = None
    if backend == "litert":
        import litert_torch

        converter = litert_torch
    for name, item in signatures.items():
        model = item["model"].eval()
        inputs = item["inputs"]
        arguments = tuple(inputs.values())
        with torch.inference_mode():
            output = model(*arguments)
        if isinstance(output, torch.Tensor):
            output = (output,)
        if not isinstance(output, (list, tuple)) or not all(isinstance(value, torch.Tensor) for value in output):
            raise ValueError("Factory models must return a tensor or flat list/tuple of tensors")
        names = item.get("outputs", [f"output_{i}" for i in range(len(output))])
        if len(names) != len(output):
            raise ValueError("Output names must match the model's output count")
        native = "forward" if name == "default" and backend == "executorch" else name
        contracts[name] = {"method": native,
            "inputs": [{"name": key, "dtype": value.numpy().dtype.name, "shape": list(value.shape),
                         **({"native_name": f"args_{i}"} if backend == "litert" else {})}
                       for i, (key, value) in enumerate(inputs.items())],
            "outputs": [{"name": key, "dtype": value.numpy().dtype.name, "shape": list(value.shape),
                          **({"native_name": f"output_{i}"} if backend == "litert" else {})}
                        for i, (key, value) in enumerate(zip(names, output, strict=True))]}
        expected[name] = [value.detach().numpy().copy() for value in output]
        examples[name] = arguments
        dynamic = item.get("dynamic_shapes")
        if dynamic:
            # Source authors supply the same explicit public shape contract for
            # every backend; graph constraints alone do not name logical tensors.
            overrides = item.get("tensor_contract")
            if overrides is None:
                raise ValueError("Dynamic exports require tensor_contract inputs/outputs")
            for direction in ("inputs", "outputs"):
                for tensor, declared in zip(contracts[name][direction], overrides[direction], strict=True):
                    tensor.update(declared)
        if backend == "executorch":
            exports[native] = torch.export.export(model, arguments, dynamic_shapes=dynamic)
        else:
            converter = converter.signature(native, model, arguments, dynamic_shapes=dynamic)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if backend == "executorch":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from executorch.runtime import Runtime

        program = to_edge_transform_and_lower(exports, partitioner=[XnnpackPartitioner()]).to_executorch()
        destination.write_bytes(program.buffer)
        loaded = Runtime.get().load_program(str(destination))
        actual = {name: [v.numpy() for v in loaded.load_method(contract["method"]).execute(examples[name])]
                  for name, contract in contracts.items()}
        exporter = importlib.metadata.version("executorch")
    else:
        from ai_edge_litert.interpreter import Interpreter

        converted = converter.convert()
        converted.export(str(destination))
        loaded = Interpreter(model_path=str(destination))
        loaded.allocate_tensors()
        actual = {}
        for name, contract in contracts.items():
            runner = loaded.get_signature_runner(contract["method"])
            result = runner(**{f"args_{i}": value.numpy() for i, value in enumerate(examples[name])})
            actual[name] = [result[f"output_{i}"] for i in range(len(expected[name]))]
            del runner
        exporter = importlib.metadata.version("litert-torch")
    validation = []
    for name in signatures:
        for index, (reference, output) in enumerate(zip(expected[name], actual[name], strict=True)):
            np.testing.assert_allclose(output, reference, rtol=1e-3, atol=1e-4)
            difference = np.abs(output.astype(np.float64) - reference.astype(np.float64))
            validation.append({"signature": name, "output": index,
                               "max_absolute_error": float(difference.max()) if difference.size else 0})
    metadata = {key: definition[key] for key in ("task", "image", "labels", "output_name", "metadata") if key in definition}
    report.write_text(json.dumps({"signatures": contracts, "definition": metadata,
                                  "exporter": exporter, "torch": torch.__version__,
                                  "sample_validation": validation}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--factory", default="build")
    parser.add_argument("--backend", choices=["executorch", "litert"], required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    export(args.source.resolve(), args.factory, args.backend, args.destination.resolve(), args.report.resolve())
