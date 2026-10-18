"""LiteRT CompiledModel and Interpreter execution behind one tensor contract."""

import importlib.metadata

import numpy as np

from ..errors import RuntimeFault
from .executorch import check_contract


def detail_info(detail):
    return {"shape": [int(x) for x in detail.get("shape_signature", detail["shape"])],
            "dtype": np.dtype(detail["dtype"]).name}


class LiteRTTensor:
    def __init__(self, package, variant):
        self.variant = variant
        self.compiled = variant.engine == "compiled"
        if self.compiled:
            from ai_edge_litert.compiled_model import CompiledModel
            from ai_edge_litert.hardware_accelerator import HardwareAccelerator

            accelerator = getattr(HardwareAccelerator, variant.target.accelerator.upper())
            self.model = CompiledModel.from_file(str(package.artifact(variant.artifact)),
                                                hardware_accel=accelerator)
            self.signatures = self.model.get_signature_list()
            fully_accelerated = self.model.is_fully_accelerated()
            if variant.target.accelerator != "cpu" and not fully_accelerated:
                self.model.close()
                raise RuntimeFault("acceleration_unavailable",
                                   "LiteRT did not fully accelerate the requested variant; choose an explicit CPU variant")
        else:
            from ai_edge_litert.interpreter import Interpreter

            if variant.target.accelerator != "cpu":
                raise RuntimeFault("unsupported_accelerator", "Interpreter variants require CPU")
            self.model = Interpreter(model_path=str(package.artifact(variant.artifact)))
            self.model.allocate_tensors()
            self.signatures = self.model.get_signature_list()
            fully_accelerated = False
        self.details = {}
        for name, signature in variant.signatures.items():
            if self.signatures:
                if signature.method not in self.signatures:
                    raise RuntimeFault("signature_not_found", f"No LiteRT signature {signature.method}",
                                       details={"signatures": list(self.signatures)})
                if self.compiled:
                    native_inputs = self.model.get_input_tensor_details(signature.method)
                    native_outputs = self.model.get_output_tensor_details(signature.method)
                else:
                    runner = self.model.get_signature_runner(signature.method)
                    native_inputs, native_outputs = runner.get_input_details(), runner.get_output_details()
                    del runner
            elif self.compiled:
                # CompiledModel normally supplies a default signature even for a
                # model without SignatureDefs. Do not guess nonexistent names.
                raise RuntimeFault("signature_not_found", "Compiled model has no executable signature")
            else:
                native_inputs = {x["name"]: x for x in self.model.get_input_details()}
                native_outputs = {x["name"]: x for x in self.model.get_output_details()}
            if len(native_inputs) != len(signature.inputs) or len(native_outputs) != len(signature.outputs):
                raise RuntimeFault("artifact_contract", "LiteRT tensor count disagrees with manifest")
            mapped = []
            for declared, native in ((signature.inputs, native_inputs), (signature.outputs, native_outputs)):
                mapping = {}
                for spec in declared:
                    native_name = spec.native_name or spec.name
                    if native_name not in native:
                        raise RuntimeFault("artifact_contract", f"No LiteRT tensor named {native_name}",
                                           details={"available": list(native)})
                    detail = native[native_name]
                    check_contract(spec, detail_info(detail))
                    quant = detail.get("quantization_parameters", {})
                    scales = np.asarray(quant.get("scales", []))
                    if scales.size:
                        if spec.quantization is None:
                            raise RuntimeFault("artifact_contract", f"Quantized tensor {spec.name} needs quantization metadata")
                        if not np.allclose(scales, spec.quantization.scales) or not np.array_equal(
                            quant["zero_points"], spec.quantization.zero_points,
                        ) or int(quant.get("quantized_dimension", 0)) != spec.quantization.axis:
                            raise RuntimeFault("artifact_contract", f"Quantization metadata mismatch for {spec.name}")
                    mapping[spec.name] = {"native_name": native_name, "detail": detail}
                mapped.append(mapping)
            self.details[name] = {"inputs": mapped[0], "outputs": mapped[1]}
        self.info = {
            "backend": "litert", "engine": variant.engine,
            "runtime_version": importlib.metadata.version("ai-edge-litert"),
            "requested_accelerator": variant.target.accelerator,
            "fully_accelerated": fully_accelerated,
            "acceleration_evidence": "compiled_model_report" if self.compiled else "cpu_execution",
            "npu_execution_verified": False,
            "signatures": list(variant.signatures),
        }

    def infer(self, signature_name, inputs):
        signature = self.variant.signatures[signature_name]
        details = self.details[signature_name]
        if self.compiled:
            index = self.model.get_signature_index(signature.method)
            for spec in signature.inputs:
                name = details["inputs"][spec.name]["native_name"]
                native = self.model.get_input_tensor_details(signature.method)[name]
                if tuple(native["shape"]) != inputs[spec.name].shape:
                    self.model.resize_input_tensor_by_name(signature.method, name, list(inputs[spec.name].shape))
            input_buffers = self.model.create_input_buffers(index)
            output_buffers = []
            try:
                output_buffers = self.model.create_output_buffers(index)
                # Native signature order may differ from manifest order.
                native_order = self.signatures[signature.method]["inputs"]
                by_name = {details["inputs"][spec.name]["native_name"]: inputs[spec.name]
                           for spec in signature.inputs}
                for name, buffer in zip(native_order, input_buffers, strict=True):
                    buffer.write(by_name[name])
                self.model.run_by_index(index, input_buffers, output_buffers)
                native_outputs = self.model.get_output_tensor_details(signature.method)
                values = {}
                for name, buffer in zip(self.signatures[signature.method]["outputs"], output_buffers, strict=True):
                    detail = native_outputs[name]
                    values[name] = buffer.read(int(np.prod(detail["shape"])), np.dtype(detail["dtype"])).reshape(detail["shape"]).copy()
                return {spec.name: values[details["outputs"][spec.name]["native_name"]]
                        for spec in signature.outputs}
            finally:
                for buffer in input_buffers + output_buffers:
                    buffer.destroy()
        if self.signatures:
            runner = self.model.get_signature_runner(signature.method)
            try:
                result = runner(**{details["inputs"][spec.name]["native_name"]: inputs[spec.name]
                                   for spec in signature.inputs})
                return {spec.name: result[details["outputs"][spec.name]["native_name"]].copy()
                        for spec in signature.outputs}
            finally:
                del runner
        for spec in signature.inputs:
            index = details["inputs"][spec.name]["detail"]["index"]
            self.model.resize_tensor_input(index, list(inputs[spec.name].shape), strict=True)
        self.model.allocate_tensors()
        for spec in signature.inputs:
            self.model.set_tensor(details["inputs"][spec.name]["detail"]["index"], inputs[spec.name])
        self.model.invoke()
        return {spec.name: self.model.get_tensor(details["outputs"][spec.name]["detail"]["index"])
                for spec in signature.outputs}

    def close(self):
        if self.compiled:
            self.model.close()
        self.model = None
