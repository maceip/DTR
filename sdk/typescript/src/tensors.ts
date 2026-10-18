import {DType, Quantization, RuntimeError, TensorSpec, TensorValue} from './types.js';

export type NumericArray = Float32Array | Float64Array | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array;
const constructors = {float32: Float32Array, float64: Float64Array, int8: Int8Array, uint8: Uint8Array, int16: Int16Array, uint16: Uint16Array, int32: Int32Array, uint32: Uint32Array, bool: Uint8Array} as const;
const limits: Partial<Record<DType, [number, number]>> = {int8: [-128, 127], uint8: [0, 255], int16: [-32768, 32767], uint16: [0, 65535], int32: [-2147483648, 2147483647], uint32: [0, 4294967295], bool: [0, 1]};
const MAX_BYTES = 256 * 1024 * 1024;
export function bytesFromBase64(value: string): Uint8Array {
  if (value.length > (MAX_BYTES / 3 + 1) * 4 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value)) throw new RuntimeError('invalid_tensor', 'Invalid or oversized base64 tensor');
  const raw = atob(value); return Uint8Array.from(raw, character => character.charCodeAt(0));
}
export function base64FromBytes(value: Uint8Array): string {
  const chunks = [];
  for (let i = 0; i < value.length; i += 32768) chunks.push(String.fromCharCode(...value.subarray(i, i + 32768)));
  return btoa(chunks.join(''));
}
export function shapeSize(shape: number[]): number {
  if (shape.length > 12 || shape.some(x => !Number.isSafeInteger(x) || x < 0)) throw new RuntimeError('invalid_tensor', 'Invalid tensor shape');
  const size = shape.reduce((a, b) => a * b, 1);
  if (!Number.isSafeInteger(size) || size > 64_000_000) throw new RuntimeError('invalid_tensor', 'Tensor exceeds size limit');
  return size;
}
export function decodeTensor(value: TensorValue): NumericArray {
  const Constructor = constructors[value.dtype as keyof typeof constructors];
  if (!Constructor) throw new RuntimeError('unsupported_dtype', `Browser numeric adapter does not expose ${value.dtype}; use a native host`);
  const count = shapeSize(value.shape), length = count * Constructor.BYTES_PER_ELEMENT;
  if (length > MAX_BYTES || (value.data === undefined) === (value.base64 === undefined)) throw new RuntimeError('invalid_tensor', 'Provide exactly one bounded tensor encoding');
  let array: NumericArray;
  if (value.base64 !== undefined) {
    const bytes = bytesFromBase64(value.base64);
    if (bytes.byteLength !== length) throw new RuntimeError('invalid_tensor', 'Tensor byte length does not match shape');
    array = new Constructor(bytes.buffer as ArrayBuffer);
  } else {
    const data = value.data!;
    if (data.length !== count) throw new RuntimeError('invalid_tensor', 'Tensor value count does not match shape');
    const values = data.map(item => typeof item === 'boolean' ? Number(item) : item);
    const range = limits[value.dtype];
    for (const item of values) {
      if (!Number.isFinite(item) || (range && (!Number.isInteger(item) || item < range[0] || item > range[1]))) throw new RuntimeError('invalid_tensor', 'Tensor has invalid values or integer overflow');
    }
    array = new Constructor(values);
  }
  if (Array.from(array).some(x => !Number.isFinite(x))) throw new RuntimeError('invalid_tensor', 'Tensor values must be finite');
  return array;
}
export function encodeTensor(array: NumericArray, shape: number[], dtype?: DType): TensorValue {
  if (array.length !== shapeSize(shape)) throw new RuntimeError('invalid_tensor', 'Shape and array length differ');
  const inferred = dtype || Object.entries(constructors).find(([, constructor]) => array.constructor === constructor)?.[0] as DType;
  if (!inferred) throw new RuntimeError('unsupported_dtype', 'Unrecognized typed array');
  return {dtype: inferred, shape, ...(array.length <= 1024 ? {data: Array.from(array)} : {base64: base64FromBytes(new Uint8Array(array.buffer, array.byteOffset, array.byteLength))})};
}
function channel(index: number, shape: number[], quant: Quantization): number {
  if (quant.scales.length === 1) return 0;
  const axis = quant.axis < 0 ? quant.axis + shape.length : quant.axis;
  if (axis < 0 || axis >= shape.length || shape[axis] !== quant.scales.length) throw new RuntimeError('invalid_quantization', 'Quantization axis does not match shape');
  const stride = shape.slice(axis + 1).reduce((a, b) => a * b, 1);
  return Math.floor(index / stride) % shape[axis];
}
function roundEven(value: number): number {
  const floor = Math.floor(value), fraction = value - floor;
  return fraction === .5 ? (floor % 2 ? floor + 1 : floor) : Math.round(value);
}
export function prepareTensor(value: TensorValue, spec: TensorSpec, quantize = true): TensorValue {
  if (value.shape.length !== spec.shape.length || spec.shape.some((x, i) => x !== -1 && x !== value.shape[i])) throw new RuntimeError('shape_mismatch', `Input ${spec.name} has an incompatible shape`);
  const array = decodeTensor(value), Constructor = constructors[spec.dtype as keyof typeof constructors];
  if (!Constructor) throw new RuntimeError('unsupported_dtype', `Use a native host for ${spec.dtype}`);
  const output = new Constructor(array.length), range = limits[spec.dtype];
  for (let i = 0; i < array.length; i++) {
    let number = array[i];
    if (spec.quantization && quantize && value.dtype.startsWith('float') && range) {
      const q = spec.quantization, c = channel(i, value.shape, q);
      if (!(q.scales[c] > 0) || !Number.isFinite(q.scales[c])) throw new RuntimeError('invalid_quantization', 'Quantization scale must be positive');
      number = Math.min(range[1], Math.max(range[0], roundEven(number / q.scales[c]) + q.zero_points[c]));
    }
    if (range && (!Number.isInteger(number) || number < range[0] || number > range[1])) throw new RuntimeError('dtype_mismatch', 'Integer conversion would lose or overflow values');
    output[i] = number;
    if (!Number.isFinite(output[i])) throw new RuntimeError('dtype_mismatch', 'Value overflows output dtype');
  }
  return encodeTensor(output, value.shape, spec.dtype);
}
export function dequantizeTensor(value: TensorValue, spec: TensorSpec): TensorValue {
  if (!spec.quantization) return value;
  const input = decodeTensor(value), output = new Float32Array(input.length), q = spec.quantization;
  for (let i = 0; i < input.length; i++) { const c = channel(i, value.shape, q); output[i] = (input[i] - q.zero_points[c]) * q.scales[c]; }
  return encodeTensor(output, value.shape);
}
