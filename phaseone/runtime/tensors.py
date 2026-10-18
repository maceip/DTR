"""One tensor wire format and identical validation for both engines."""

import base64
import io
import math

import numpy as np

from .errors import RuntimeFault
from .schema import ImagePreprocess, TensorSpec, TensorValue

MAX_TENSOR_BYTES = 256 * 1024**2


def contiguous(array: np.ndarray) -> np.ndarray:
    # ascontiguousarray promotes rank-zero tensors to rank one. Preserve scalars.
    return np.ascontiguousarray(array).reshape(array.shape)


def decode_tensor(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        result = value
    elif hasattr(value, "detach") and hasattr(value, "numpy"):
        result = value.detach().cpu().numpy()
    elif isinstance(value, (TensorValue, dict)):
        try:
            tensor = value if isinstance(value, TensorValue) else TensorValue.model_validate(value)
            dtype = np.dtype(tensor.dtype).newbyteorder("<")
            nbytes = math.prod(tensor.shape) * dtype.itemsize
            if nbytes > MAX_TENSOR_BYTES:
                raise ValueError("Tensor exceeds the byte limit")
            if tensor.base64 is not None:
                if len(tensor.base64) > (MAX_TENSOR_BYTES + 2) * 4 // 3 + 4:
                    raise ValueError("Encoded tensor exceeds the byte limit")
                raw = base64.b64decode(tensor.base64, validate=True)
                if len(raw) != nbytes:
                    raise ValueError("Tensor byte length does not match its shape and dtype")
                result = np.frombuffer(raw, dtype=dtype).reshape(tensor.shape).copy()
            else:
                raw = np.asarray(tensor.data)
                result = checked_cast(raw, dtype).reshape(tensor.shape)
        except (ValueError, TypeError, OverflowError) as exc:
            raise RuntimeFault("invalid_tensor", str(exc)) from exc
    else:
        try:
            result = np.asarray(value)
        except (ValueError, TypeError) as exc:
            raise RuntimeFault("invalid_tensor", str(exc)) from exc
    if result.dtype.kind not in "biuf" or result.nbytes > MAX_TENSOR_BYTES:
        raise RuntimeFault("invalid_tensor", "Unsupported dtype or tensor exceeds byte limit")
    if not np.isfinite(result).all():
        raise RuntimeFault("invalid_tensor", "Non-finite tensor values are not permitted")
    return contiguous(result)


def checked_cast(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
        raise ValueError("Tensor values must be finite numeric values")
    if dtype.kind in "iu":
        limits = np.iinfo(dtype)
        upper = value >= int(limits.max) + 1 if value.dtype.kind == "f" else value > limits.max
        if value.size and (np.any(value < limits.min) or np.any(upper)):
            raise ValueError(f"Values are outside the range of {dtype}")
        if value.dtype.kind == "f" and np.any(value != np.floor(value)):
            raise ValueError("Fractional values cannot be silently converted to integers")
    if dtype.kind == "b" and np.any((value != 0) & (value != 1)):
        raise ValueError("Boolean tensors accept only true/false or 0/1")
    with np.errstate(over="ignore", invalid="ignore"):
        output = value.astype(dtype, copy=False)
    if not np.isfinite(output).all():
        raise ValueError(f"Values overflow {dtype}")
    return output


def quantization_arrays(spec: TensorSpec, rank: int):
    quant = spec.quantization
    scales = np.asarray(quant.scales, dtype=np.float32)
    zeros = np.asarray(quant.zero_points, dtype=np.int64)
    if scales.size > 1:
        if not -rank <= quant.axis < rank:
            raise RuntimeFault("invalid_quantization", "Quantization axis is outside tensor rank")
        shape = [1] * rank
        shape[quant.axis] = scales.size
        scales, zeros = scales.reshape(shape), zeros.reshape(shape)
    else:
        scales, zeros = scales.reshape(()), zeros.reshape(())
    return scales, zeros


def prepare_input(value, spec: TensorSpec, *, quantize: bool = True) -> np.ndarray:
    array = decode_tensor(value)
    if len(array.shape) != len(spec.shape) or any(
        expected != -1 and actual != expected
        for actual, expected in zip(array.shape, spec.shape, strict=True)
    ):
        raise RuntimeFault("shape_mismatch", f"Input {spec.name} has an incompatible shape",
                           details={"expected": spec.shape, "actual": list(array.shape)})
    dtype = np.dtype(spec.dtype)
    if spec.quantization and quantize and array.dtype.kind == "f" and dtype.kind in "iu":
        scales, zeros = quantization_arrays(spec, array.ndim)
        limits = np.iinfo(dtype)
        try:
            array = np.clip(np.rint(array / scales) + zeros, limits.min, limits.max)
        except ValueError as exc:
            raise RuntimeFault("invalid_quantization", str(exc)) from exc
    try:
        return contiguous(checked_cast(array, dtype))
    except (ValueError, TypeError, OverflowError) as exc:
        raise RuntimeFault("dtype_mismatch", f"Input {spec.name}: {exc}") from exc


def prepare_output(array, spec: TensorSpec, *, dequantize: bool = True) -> np.ndarray:
    array = decode_tensor(array)
    if len(array.shape) != len(spec.shape) or any(
        expected != -1 and actual != expected
        for actual, expected in zip(array.shape, spec.shape, strict=True)
    ):
        raise RuntimeFault("output_contract", f"Output {spec.name} violates its declared shape",
                           details={"expected": spec.shape, "actual": list(array.shape)})
    if array.dtype != np.dtype(spec.dtype):
        raise RuntimeFault("output_contract", f"Output {spec.name} violates its declared dtype",
                           details={"expected": spec.dtype, "actual": str(array.dtype)})
    if spec.quantization and dequantize:
        scales, zeros = quantization_arrays(spec, array.ndim)
        array = (array.astype(np.float32) - zeros) * scales
        array = array.astype(np.float32)
    return contiguous(array)


def encode_tensor(array, *, binary_threshold: int = 1024) -> dict:
    array = decode_tensor(array)
    tensor = {"dtype": array.dtype.name, "shape": list(array.shape)}
    # JSON numbers lose integer precision in JavaScript above 2**53-1. Keep
    # those values in the little-endian wire encoding even for scalar tensors.
    wide_integer = array.dtype.kind in "iu" and array.dtype.itemsize == 8 and (
        bool(np.any(array > 2**53 - 1)) or (array.dtype.kind == "i" and bool(np.any(array < -(2**53 - 1))))
    )
    if array.size <= binary_threshold and not wide_integer:
        tensor["data"] = array.reshape(-1).tolist()
    else:
        little = array.astype(array.dtype.newbyteorder("<"), copy=False)
        tensor["base64"] = base64.b64encode(little.tobytes()).decode()
    return tensor


def preprocess_image(data: bytes, spec: ImagePreprocess) -> np.ndarray:
    from PIL import Image, ImageOps

    if len(data) > 32 * 1024**2:
        raise RuntimeFault("invalid_image", "Image exceeds 32 MiB")
    try:
        with Image.open(io.BytesIO(data)) as original:
            if original.width * original.height > 40_000_000:
                raise ValueError("Decoded image is too large")
            image = ImageOps.exif_transpose(original).convert("RGB")
            if spec.resize_shorter:
                scale = spec.resize_shorter / min(image.size)
                image = image.resize((int(image.width * scale), int(image.height * scale)),
                                     Image.Resampling.BILINEAR)
                left, top = (image.width - spec.width) // 2, (image.height - spec.height) // 2
                if left < 0 or top < 0:
                    raise ValueError("Resize dimensions are smaller than the requested crop")
                image = image.crop((left, top, left + spec.width, top + spec.height))
            else:
                image = image.resize((spec.width, spec.height), Image.Resampling.BILINEAR)
            result = np.asarray(image, dtype=np.float32) * spec.scale
        result = (result - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(
            spec.std, dtype=np.float32,
        )
        if spec.layout == "nchw":
            result = result.transpose(2, 0, 1)
        return np.ascontiguousarray(result[None])
    except (ValueError, OSError) as exc:
        raise RuntimeFault("invalid_image", str(exc)) from exc
