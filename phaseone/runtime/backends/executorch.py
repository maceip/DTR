"""ExecuTorch program execution with native metadata validation."""

import importlib.metadata

import numpy as np

from ..errors import RuntimeFault

_DTYPES = {
    "Byte": "uint8", "Char": "int8", "Short": "int16", "Int": "int32",
    "Long": "int64", "Half": "float16", "Float": "float32", "Double": "float64",
    "Bool": "bool", "UInt16": "uint16", "UInt32": "uint32", "UInt64": "uint64",
}
# TensorInfo.dtype() returns the ScalarType numeric value in the native wheel.
_SCALAR_TYPES = {0: "Byte", 1: "Char", 2: "Short", 3: "Int", 4: "Long", 5: "Half",
                 6: "Float", 7: "Double", 11: "Bool", 27: "UInt16", 28: "UInt32", 29: "UInt64"}


def read_attribute(value):
    return value() if callable(value) else value


def tensor_info(metadata):
    dtype = read_attribute(metadata.dtype)
    name = _SCALAR_TYPES.get(dtype, getattr(dtype, "name", str(dtype).split(".")[-1]))
    if name not in _DTYPES:
        raise RuntimeFault("unsupported_dtype", f"ExecuTorch tensor dtype {name} is not supported")
    return {"shape": list(read_attribute(metadata.sizes)), "dtype": _DTYPES[name]}


def check_contract(spec, actual):
    if spec.dtype != actual["dtype"]:
        raise RuntimeFault("artifact_contract", f"Tensor {spec.name} dtype disagrees with the artifact",
                           details={"declared": spec.dtype, "native": actual["dtype"]})
    if len(spec.shape) != len(actual["shape"]) or any(
        declared != -1 and native != -1 and declared != native
        for declared, native in zip(spec.shape, actual["shape"], strict=True)
    ):
        raise RuntimeFault("artifact_contract", f"Tensor {spec.name} shape disagrees with the artifact",
                           details={"declared": spec.shape, "native": actual["shape"]})


class ExecuTorchTensor:
    def __init__(self, package, variant):
        import torch
        from executorch.runtime import Runtime

        self.torch = torch
        self.variant = variant
        runtime = Runtime.get()
        delegates = list(runtime.backend_registry.registered_backend_names)
        if variant.target.delegate and variant.target.delegate not in delegates:
            raise RuntimeFault("delegate_unavailable",
                               f"ExecuTorch build has no {variant.target.delegate}",
                               details={"registered_delegates": delegates})
        if variant.target.accelerator != "cpu" and not variant.target.delegate:
            raise RuntimeFault("invalid_target", "Accelerated ExecuTorch variants must name their delegate")
        self.program = runtime.load_program(
            str(package.artifact(variant.artifact)),
            data_path=str(package.artifact(variant.data_artifact)) if variant.data_artifact else None,
        )
        self.native = {}
        self.methods = {}
        for name, signature in variant.signatures.items():
            if signature.method not in self.program.method_names:
                raise RuntimeFault("signature_not_found", f"No ExecuTorch method {signature.method}",
                                   details={"methods": sorted(self.program.method_names)})
            metadata = self.program.metadata(signature.method)
            if metadata.num_inputs() != len(signature.inputs) or metadata.num_outputs() != len(signature.outputs):
                raise RuntimeFault("artifact_contract", f"Method {signature.method} tensor count disagrees with manifest")
            inputs = [tensor_info(metadata.input_tensor_meta(i)) for i in range(metadata.num_inputs())]
            outputs = [tensor_info(metadata.output_tensor_meta(i)) for i in range(metadata.num_outputs())]
            for spec, actual in zip(signature.inputs + signature.outputs, inputs + outputs, strict=True):
                check_contract(spec, actual)
            self.native[name] = {"method": signature.method, "inputs": inputs, "outputs": outputs}
            self.methods[name] = self.program.load_method(signature.method)
        self.info = {
            "backend": "executorch", "engine": "program",
            "runtime_version": importlib.metadata.version("executorch"),
            "requested_accelerator": variant.target.accelerator,
            "delegate": variant.target.delegate or "portable_cpu",
            "registered_delegates": delegates,
            "acceleration_evidence": "cpu_execution" if variant.target.accelerator == "cpu" else "registered_delegate; profiling_required",
            "npu_execution_verified": False,
            "signatures": self.native,
        }

    def infer(self, signature_name, inputs):
        signature = self.variant.signatures[signature_name]
        method = self.methods[signature_name]
        arguments = [self.torch.from_numpy(np.array(inputs[item.name], copy=True))
                     for item in signature.inputs]
        with self.torch.inference_mode():
            results = method.execute(arguments)
        if len(results) != len(signature.outputs):
            raise RuntimeFault("output_contract", "ExecuTorch returned an unexpected output count")
        return {spec.name: value.detach().cpu().numpy().copy()
                for spec, value in zip(signature.outputs, results, strict=True)}

    def close(self):
        self.methods = {}
        self.program = None
