"""PHASEONE v1 package and invocation contracts, shared by every backend."""

import math
import re
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Name = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,95}$")]
BackendName = Literal["executorch", "litert", "mlx", "local_server"]
DType = Literal[
    "float16", "float32", "float64", "int8", "uint8", "int16", "uint16",
    "int32", "uint32", "int64", "uint64", "bool",
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value or "\\" in value or "\x00" in value or path.is_absolute()
        or any(part in (".", "..", "") for part in value.split("/"))
        or re.match(r"^[A-Za-z]:", value)
    ):
        raise ValueError("Expected a normalized, relative package path")
    return value


class Artifact(Contract):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0, le=256 * 1024**3)
    media_type: str = "application/octet-stream"

    _path = field_validator("path")(relative_path)


class Quantization(Contract):
    scales: list[float] = Field(min_length=1)
    zero_points: list[int] = Field(min_length=1)
    axis: int = 0

    @model_validator(mode="after")
    def valid(self):
        if any(x <= 0 for x in self.scales) or len(self.scales) != len(self.zero_points):
            raise ValueError("Quantization requires positive scales and matching zero points")
        return self


class TensorSpec(Contract):
    name: Name
    dtype: DType
    shape: list[int] = Field(max_length=12)
    native_name: str | None = None
    quantization: Quantization | None = None

    @field_validator("shape")
    @classmethod
    def dimensions(cls, shape):
        if any(d < -1 for d in shape):
            raise ValueError("Dimensions must be non-negative, or -1 for dynamic dimensions")
        return shape


class Signature(Contract):
    method: str = "forward"
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]

    @model_validator(mode="after")
    def unique_names(self):
        for items in (self.inputs, self.outputs):
            if len({item.name for item in items}) != len(items):
                raise ValueError("Tensor names must be unique in a signature")
        return self


class Target(Contract):
    systems: list[Literal["darwin", "linux", "windows", "android", "ios", "web"]] = []
    architectures: list[str] = []
    accelerator: Literal["cpu", "gpu", "npu"] = "cpu"
    delegate: str | None = None
    soc: str | None = None
    runtime_version: str | None = None
    minimum_memory_mb: int = Field(default=0, ge=0)
    minimum_android_api: int = Field(default=0, ge=0, le=1000)
    components: dict[Name, str] = {}
    firmware: list[str] = []
    minimum_security_patch: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class ChatProcessor(Contract):
    """A hash-covered processor profile; packages cannot supply executable code."""

    config: Name
    image_token: str = Field(min_length=1, max_length=256)
    image_splitting: bool | None = None
    runner: Literal["native", "program", "gemma4-ple"] = "native"
    vision_inputs: list[Name] = Field(default_factory=lambda: ["pixel_values"], min_length=1, max_length=4)
    image_fit: Literal["processor", "square_pad"] = "processor"
    image_size: int = Field(default=512, ge=32, le=2048)

    @field_validator("vision_inputs")
    @classmethod
    def unique_vision_inputs(cls, names):
        if len(set(names)) != len(names):
            raise ValueError("Vision inputs must be unique and ordered as in the compiled graph")
        return names


class ChatConfig(Contract):
    context_length: int = Field(ge=32, le=2_000_000)
    tokenizer: str | None = None
    template: str | None = None
    tokenizer_config: str | None = None
    modalities: list[Literal["text", "image", "audio"]] = ["text"]
    tool_protocol: Literal["json", "native", "none"] = "json"
    num_bos: int = Field(default=0, ge=0, le=2)
    num_eos: int = Field(default=0, ge=0, le=2)
    processor: ChatProcessor | None = None
    max_images: int = Field(default=1, ge=1, le=16)
    vision_accelerator: Literal["cpu", "gpu", "npu"] | None = None
    audio_accelerator: Literal["cpu", "gpu", "npu"] | None = None
    prompt_format: Literal["chatml", "llama3"] | None = None
    runner: Literal["native", "qnn"] = "native"
    decoder: Literal["gemma3", "gemma4"] | None = None


class Variant(Contract):
    id: Name
    backend: BackendName
    engine: Literal["program", "compiled", "interpreter", "llm"]
    artifact: str
    data_artifact: str | None = None
    supporting_artifacts: list[str] = []
    target: Target = Field(default_factory=Target)
    signatures: dict[str, Signature] = {}
    chat: ChatConfig | None = None
    priority: int = 0

    @model_validator(mode="after")
    def engine_matches(self):
        if self.backend == "executorch" and self.engine not in ("program", "llm"):
            raise ValueError("ExecuTorch uses program or llm engines")
        if self.backend == "litert" and self.engine == "program":
            raise ValueError("LiteRT uses compiled, interpreter, or llm engines")
        if self.backend in ("mlx", "local_server") and self.engine != "llm":
            raise ValueError("MLX and local server variants use the llm engine")
        if self.engine == "llm" and self.chat is None:
            raise ValueError("An LLM variant requires a chat contract")
        if self.engine != "llm" and not self.signatures:
            raise ValueError("A tensor variant requires at least one signature")
        if self.backend == "executorch" and self.chat and not self.chat.tokenizer:
            raise ValueError("ExecuTorch chat requires a packaged tokenizer")
        if self.chat and self.chat.runner == "qnn":
            if (self.backend != "executorch" or self.engine != "llm" or self.target.accelerator != "npu"
                    or self.target.delegate != "QnnBackend" or self.chat.modalities != ["text"]
                    or not self.chat.decoder or not self.chat.template or self.chat.processor
                    or self.chat.tool_protocol == "native"):
                raise ValueError("QNN text runners require ExecuTorch/NPU/QnnBackend, a decoder and template")
        elif self.chat and self.chat.decoder is not None:
            raise ValueError("The decoder field belongs to the QNN runner")
        return self

    def artifact_ids(self) -> set[str]:
        ids = {self.artifact, *self.supporting_artifacts}
        if self.data_artifact:
            ids.add(self.data_artifact)
        if self.chat:
            ids.update(x for x in (
                self.chat.tokenizer, self.chat.template, self.chat.tokenizer_config,
            ) if x)
            if self.chat.processor:
                ids.add(self.chat.processor.config)
        return ids


class ImagePreprocess(Contract):
    input_name: Name
    height: int = Field(ge=1, le=8192)
    width: int = Field(ge=1, le=8192)
    layout: Literal["nchw", "nhwc"] = "nchw"
    resize_shorter: int | None = Field(default=None, ge=1, le=16384)
    scale: float = 1.0 / 255.0
    mean: list[float] = [0.0, 0.0, 0.0]
    std: list[float] = [1.0, 1.0, 1.0]

    @model_validator(mode="after")
    def channels(self):
        if len(self.mean) != 3 or len(self.std) != 3 or any(x <= 0 for x in self.std):
            raise ValueError("RGB preprocessing needs three means and positive standard deviations")
        return self


class Model(Contract):
    task: Literal["tensor", "chat", "image.classification", "embedding"]
    variants: list[Variant] = Field(min_length=1)
    image: ImagePreprocess | None = None
    labels: str | None = None
    output_name: str | None = None

    @model_validator(mode="after")
    def consistent(self):
        if len({v.id for v in self.variants}) != len(self.variants):
            raise ValueError("Variant IDs must be unique within a model")
        if any((v.engine == "llm") != (self.task == "chat") for v in self.variants):
            raise ValueError("Chat models require LLM variants; other tasks require tensor variants")
        if self.task == "image.classification" and self.image is None:
            raise ValueError("Image classification requires an explicit preprocessing contract")
        return self


class ToolDefinition(Contract):
    name: Name
    description: str = Field(max_length=4096)
    parameters: dict[str, Any]
    permission: Name
    side_effect: bool = False


class AgentDefinition(Contract):
    model: Name
    instruction: str = Field(max_length=65536)
    tools: list[Name] = []
    max_steps: int = Field(default=6, ge=1, le=32)


class Manifest(Contract):
    schema_version: Literal[1] = 1
    name: Name
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$")
    description: str = Field(default="", max_length=8192)
    artifacts: dict[Name, Artifact]
    models: dict[Name, Model]
    tools: list[ToolDefinition] = []
    permissions: list[Name] = []
    agent: AgentDefinition | None = None
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def references(self):
        if not self.models:
            raise ValueError("An application needs at least one model")
        paths = [a.path for a in self.artifacts.values()]
        if len({path.casefold() for path in paths}) != len(paths):
            raise ValueError("Artifact paths must be unique, including on case-insensitive filesystems")
        if any(path.casefold() in ("phaseone.json", "phaseone.signature.json") for path in paths):
            raise ValueError("Artifacts cannot replace package metadata files")
        for model in self.models.values():
            ids = {model.labels} if model.labels else set()
            for variant in model.variants:
                ids |= variant.artifact_ids()
            missing = ids - self.artifacts.keys()
            if missing:
                raise ValueError(f"Unknown artifacts: {sorted(missing)}")
        names = [tool.name for tool in self.tools]
        if len(set(names)) != len(names):
            raise ValueError("Tool names must be unique")
        if any(t.permission not in self.permissions for t in self.tools):
            raise ValueError("Every tool permission must be declared by the application")
        if self.agent:
            if self.agent.model not in self.models or self.models[self.agent.model].task != "chat":
                raise ValueError("Agent model must refer to a chat model")
            if set(self.agent.tools) - set(names):
                raise ValueError("Agent references undeclared tools")
        return self


class TensorValue(Contract):
    dtype: DType
    shape: list[int] = Field(max_length=12)
    data: list[Any] | None = None
    base64: str | None = None

    @model_validator(mode="after")
    def encoding(self):
        if (self.data is None) == (self.base64 is None):
            raise ValueError("Provide exactly one tensor encoding: data or base64")
        if any(x < 0 for x in self.shape) or math.prod(self.shape) > 64_000_000:
            raise ValueError("Invalid or oversized tensor shape")
        return self


class ContentPart(Contract):
    type: Literal["text", "image", "audio"]
    text: str | None = None
    data: str | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def content(self):
        if self.type == "text" and (self.text is None or self.data is not None):
            raise ValueError("Text content requires text only")
        if self.type != "text" and (self.data is None or self.mime_type is None or self.text is not None):
            raise ValueError("Media content requires base64 data and MIME type")
        return self


class ToolCall(Contract):
    id: str = Field(min_length=1, max_length=128)
    name: Name
    arguments: dict[str, Any]


class Message(Contract):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] = ""
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None
    name: str | None = None


class Generation(Contract):
    max_output_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1, le=10000)
    stop: list[str] = Field(default_factory=list, max_length=16)
    timeout_seconds: float = Field(default=120, gt=0, le=3600)

    @field_validator("stop")
    @classmethod
    def nonempty_stops(cls, values):
        if any(not x or len(x) > 256 for x in values):
            raise ValueError("Stop sequences must contain 1 to 256 characters")
        return values
