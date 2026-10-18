#!/usr/bin/env python3
"""Build real, small multi-signature artifacts for native integration tests.

The useful application example is built separately. This deterministic graph
exercises names, multiple methods and backend switching without a large model.
"""

import argparse
import hashlib
import json
from pathlib import Path


def build(destination: Path, *, quantized=False, dynamic=False):
    import flatbuffers
    import torch
    from ai_edge_litert import schema_py_generated as fb
    from executorch.exir import to_edge

    destination.mkdir(parents=True, exist_ok=True)

    class Add(torch.nn.Module):
        def forward(self, left, right):
            return left + right

    class Subtract(torch.nn.Module):
        def forward(self, left, right):
            return left - right

    shape = (2 if dynamic else 1, 4)
    dtype = torch.int8 if quantized else torch.float32
    arguments = (torch.ones(shape, dtype=dtype), torch.ones(shape, dtype=dtype))
    dynamic_shapes = ({0: torch.export.Dim("batch", min=1, max=8)},) * 2 if dynamic else None
    programs = {name: torch.export.export(module, arguments, dynamic_shapes=dynamic_shapes)
                for name, module in (("forward", Add()), ("difference", Subtract()))}
    exported = to_edge(programs).to_executorch()
    (destination / "fixture.pte").write_bytes(exported.buffer)

    model = fb.ModelT()
    model.version = 3
    model.description = b"PHASEONE native adapter contract fixture"
    model.buffers = [fb.BufferT()]
    model.operatorCodes, model.subgraphs, model.signatureDefs = [], [], []
    for index, (key, operation) in enumerate((("sum", fb.BuiltinOperator.ADD),
                                             ("difference", fb.BuiltinOperator.SUB))):
        opcode = fb.OperatorCodeT()
        opcode.builtinCode = operation
        opcode.deprecatedBuiltinCode = operation
        opcode.version = 2 if quantized else 1
        model.operatorCodes.append(opcode)
        subgraph = fb.SubGraphT()
        subgraph.name = key.encode()
        subgraph.tensors = []
        for name in ("a", "b", "result"):
            tensor = fb.TensorT()
            tensor.name = name.encode()
            tensor.type = fb.TensorType.INT8 if quantized else fb.TensorType.FLOAT32
            tensor.shape = list(shape)
            tensor.shapeSignature = [-1, 4] if dynamic else list(shape)
            if quantized:
                quant = fb.QuantizationParametersT()
                quant.scale = [.25]
                quant.zeroPoint = [0]
                tensor.quantization = quant
            tensor.hasRank = True
            tensor.buffer = 0
            subgraph.tensors.append(tensor)
        operator = fb.OperatorT()
        operator.opcodeIndex = index
        operator.inputs, operator.outputs = [0, 1], [2]
        if index == 0:
            operator.builtinOptionsType = fb.BuiltinOptions.AddOptions
            operator.builtinOptions = fb.AddOptionsT()
        else:
            operator.builtinOptionsType = fb.BuiltinOptions.SubOptions
            operator.builtinOptions = fb.SubOptionsT()
        subgraph.operators = [operator]
        subgraph.inputs, subgraph.outputs = [0, 1], [2]
        model.subgraphs.append(subgraph)
        signature = fb.SignatureDefT()
        signature.signatureKey = key.encode()
        signature.subgraphIndex = index
        mappings = []
        for tensor_index, name in enumerate(("a", "b", "result")):
            mapping = fb.TensorMapT()
            mapping.name = name.encode()
            mapping.tensorIndex = tensor_index
            mappings.append(mapping)
        signature.inputs, signature.outputs = mappings[:2], mappings[2:]
        model.signatureDefs.append(signature)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    (destination / "fixture.tflite").write_bytes(builder.Output())

    variants = []
    for backend, engine, artifact, priority in (
        ("executorch", "program", "et", 30),
        ("litert", "compiled", "lite", 20),
        ("litert", "interpreter", "lite", 10),
    ):
        signatures = {}
        for logical, native in (("default", "forward" if backend == "executorch" else "sum"),
                                 ("difference", "difference")):
            signatures[logical] = {
                "method": native,
                "inputs": [{"name": name, "native_name": actual, "dtype": "float32", "shape": [1, 4]}
                           for name, actual in (("left", "a"), ("right", "b"))],
                "outputs": [{"name": "output", "native_name": "result", "dtype": "float32", "shape": [1, 4]}],
            }
            for tensor in signatures[logical]["inputs"] + signatures[logical]["outputs"]:
                tensor["shape"] = [-1, 4] if dynamic else list(shape)
                if quantized:
                    tensor["dtype"] = "int8"
                    tensor["quantization"] = {"scales": [.25], "zero_points": [0], "axis": 0}
        variants.append({"id": f"{backend}-{engine}", "backend": backend, "engine": engine,
                         "artifact": artifact, "target": {"accelerator": "cpu"},
                         "priority": priority, "signatures": signatures})
    if not quantized and not dynamic:
        import copy
        web = copy.deepcopy(variants[1])
        web.update(id="litert-web-cpu", priority=0,
                   target={"systems": ["web"], "architectures": ["wasm32"],
                           "accelerator": "cpu", "runtime_version": ">=2.5.3,<3"})
        variants.append(web)
    artifacts = {}
    for name, filename in (("et", "fixture.pte"), ("lite", "fixture.tflite")):
        data = (destination / filename).read_bytes()
        artifacts[name] = {"path": filename, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"schema_version": 1, "name": "runtime-fixture", "version": "1.0.0",
                "description": "Real native multi-signature contract fixture",
                "artifacts": artifacts, "models": {"calculator": {"task": "tensor", "variants": variants}}}
    (destination / "phaseone.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(build(parser.parse_args().destination))
