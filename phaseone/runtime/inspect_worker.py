"""Read native tensor metadata in a disposable process."""

import argparse
import json
import re
from pathlib import Path


def name(value, default):
    return value if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,95}", value) else default


def inspect(artifact, backend, report, data=None):
    import importlib.metadata

    import numpy as np

    contracts = {}
    if backend == "executorch":
        from executorch.runtime import Runtime

        from phaseone.runtime.backends.executorch import tensor_info

        program = Runtime.get().load_program(str(artifact), data_path=str(data) if data else None)
        for index, method in enumerate(sorted(program.method_names)):
            metadata = program.metadata(method)
            try:
                inputs = [tensor_info(metadata.input_tensor_meta(i)) for i in range(metadata.num_inputs())]
                outputs = [tensor_info(metadata.output_tensor_meta(i)) for i in range(metadata.num_outputs())]
            except RuntimeError:
                continue  # Constant metadata methods have non-tensor results.
            logical = "default" if method == "forward" else name(method, f"method_{index}")
            contracts[logical] = {"method": method,
                                  "inputs": [{"name": f"input_{i}", **value} for i, value in enumerate(inputs)],
                                  "outputs": [{"name": f"output_{i}", **value} for i, value in enumerate(outputs)]}
        version = importlib.metadata.version("executorch")
    else:
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(artifact))
        interpreter.allocate_tensors()
        signatures = interpreter.get_signature_list()
        for index, method in enumerate(signatures or {"default": {}}):
            if signatures:
                runner = interpreter.get_signature_runner(method)
                inputs, outputs = runner.get_input_details(), runner.get_output_details()
                del runner
            else:
                inputs = {value["name"]: value for value in interpreter.get_input_details()}
                outputs = {value["name"]: value for value in interpreter.get_output_details()}
            tensor_contracts = []
            for direction, tensors in (("input", inputs), ("output", outputs)):
                values = []
                for i, (native_name, detail) in enumerate(tensors.items()):
                    value = {"name": name(native_name, f"{direction}_{i}"), "native_name": native_name,
                             "dtype": np.dtype(detail["dtype"]).name,
                             "shape": [int(d) for d in detail.get("shape_signature", detail["shape"])]}
                    quant = detail.get("quantization_parameters", {})
                    if len(quant.get("scales", [])):
                        value["quantization"] = {"scales": quant["scales"].tolist(), "zero_points": quant["zero_points"].tolist(), "axis": int(quant["quantized_dimension"])}
                    values.append(value)
                tensor_contracts.append(values)
            logical = "default" if len(signatures) <= 1 else name(method, f"signature_{index}")
            contracts[logical] = {"method": method, "inputs": tensor_contracts[0], "outputs": tensor_contracts[1]}
        version = importlib.metadata.version("ai-edge-litert")
    report.write_text(json.dumps({"signatures": contracts, "runtime_version": version}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("backend", choices=["executorch", "litert"])
    parser.add_argument("report", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    inspect(args.artifact, args.backend, args.report, args.data)
