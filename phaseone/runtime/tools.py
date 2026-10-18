"""Host-owned tools: schemas, permissions, scope and durable action identities."""

import hashlib
import json
import os
import stat
import threading
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator

from .errors import RuntimeFault
from .package import canonical
from .schema import ToolDefinition, relative_path


def validator(schema):
    """No remote references: validating model arguments must never fetch a URL."""
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("$ref", "$dynamicRef") and (not isinstance(item, str) or not item.startswith("#")):
                    raise RuntimeFault("invalid_tool_schema", "Tool schemas may reference only local definitions")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise RuntimeFault("invalid_tool_schema", str(exc)) from exc
    if schema.get("type") != "object":
        raise RuntimeFault("invalid_tool_schema", "Tool arguments must be a JSON object")
    return Draft202012Validator(schema)


@dataclass(frozen=True)
class ToolContext:
    action_id: str
    package_digest: str
    cancellation: threading.Event | None = None
    subject: str = "local"
    authorization: dict | None = None

    def check_cancelled(self):
        if self.cancellation and self.cancellation.is_set():
            raise RuntimeFault("cancelled", "Tool invocation was cancelled")


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    handler: Callable
    arguments: Draft202012Validator
    access: Callable | None = None


class ToolGateway:
    def __init__(self, runtime):
        self.runtime = runtime
        self.tools: dict[str, RegisteredTool] = {}
        self.lock = threading.RLock()

    def register(self, definition: ToolDefinition | dict, handler: Callable, *, access: Callable | None = None):
        definition = ToolDefinition.model_validate(definition).model_copy(deep=True)
        registered = RegisteredTool(definition, handler, validator(definition.parameters), access)
        with self.lock:
            if definition.name in self.tools:
                raise RuntimeFault("tool_exists", f"Tool already registered: {definition.name}")
            self.tools[definition.name] = registered

    def descriptions(self, package, names=None):
        definitions = {tool.name: tool for tool in package.manifest.tools}
        names = names if names is not None else list(definitions)
        result = []
        for name in names:
            if name not in definitions:
                raise RuntimeFault("tool_not_declared", f"Application does not declare {name}")
            tool = definitions[name]
            validator(tool.parameters)
            result.append({"name": name, "description": tool.description, "parameters": tool.parameters})
        return result

    def invoke(self, package, name: str, arguments: dict, *, action_id: str, cancellation=None):
        if not isinstance(action_id, str) or not 1 <= len(action_id) <= 256:
            raise RuntimeFault("invalid_action_id", "An action ID of 1 to 256 characters is required")
        try:
            encoded = canonical(arguments)
        except (TypeError, ValueError) as exc:
            raise RuntimeFault("invalid_tool_arguments", "Arguments must be finite JSON values") from exc
        if not isinstance(arguments, dict) or len(encoded) > 1024**2:
            raise RuntimeFault("invalid_tool_arguments", "Tool arguments must be a JSON object no larger than 1 MiB")
        # Take an owned snapshot before policy evaluation or handler entry.
        arguments = json.loads(encoded)
        subject = self.runtime.subject
        request = {"package": package.digest, "subject": subject, "tool": name, "arguments": arguments}
        begun = self.runtime.state.begin_action(action_id, request)
        entered, registered, authorization = False, None, None
        try:
            declared = next((t for t in package.manifest.tools if t.name == name), None)
            if declared is None:
                raise RuntimeFault("tool_not_declared", f"Application does not declare {name}")
            with self.lock:
                registered = self.tools.get(name)
            if registered is None:
                raise RuntimeFault("tool_unavailable", f"Host has no implementation of {name}")
            actual = registered.definition
            if actual.permission != declared.permission or actual.side_effect != declared.side_effect:
                raise RuntimeFault("tool_contract", "Package cannot weaken a host tool's permission or side-effect declaration")
            if self.runtime.permissions is None and actual.permission not in self.runtime.granted_permissions:
                raise RuntimeFault("permission_denied", f"Host has not granted {actual.permission}",
                                   details={"permission": actual.permission})
            for check in (registered.arguments, validator(declared.parameters)):
                error = next(check.iter_errors(arguments), None)
                if error:
                    raise RuntimeFault("invalid_tool_arguments", error.message,
                                       details={"path": list(error.absolute_path)})
            policy = self.runtime.permissions
            if policy is not None and registered.access is None:
                raise RuntimeFault("tool_scope_missing", "Host tool has no resource binding")
            scope = (policy.authorize(subject, registered.access(json.loads(encoded)))
                     if policy is not None else nullcontext(None))
            with scope as authorization:
                # Re-check permission even for completed actions: a retry must
                # not disclose a saved result after consent has been revoked.
                context = ToolContext(action_id, package.digest, cancellation, subject, authorization)
                context.check_cancelled()
                if begun["cached"]:
                    result = {**begun["result"], "cached": True}
                else:
                    entered = True
                    value = registered.handler(arguments, context)
                    if len(canonical(value)) > 1024**2:
                        raise RuntimeFault("tool_output_too_large", "Tool result exceeds 1 MiB")
                    result = {"action_id": action_id, "tool": name, "status": "complete", "result": value, "cached": False}
        except Exception as exc:
            error = exc if isinstance(exc, RuntimeFault) else RuntimeFault("tool_failed", f"{type(exc).__name__}: {exc}")
            status = "uncertain" if entered and registered.definition.side_effect else "failed" if entered else "denied"
            result = {"action_id": action_id, "tool": name, "status": status, "error": error.as_dict(), "cached": False}
        # A crash before this commit leaves a pending action; its outcome is
        # unknown and a retry cannot blindly execute the operation again.
        if not begun["cached"]:
            self.runtime.state.finish_action(action_id, result, result["status"])
        receipt = self.runtime.state.receipt("tool_action", {
            "action_id": action_id, "package": package.digest, "tool": name, "status": result["status"],
            "arguments_sha256": hashlib.sha256(encoded).hexdigest(),
            "subject": subject, "authorization": authorization,
        })
        return {**result, "receipt": receipt}


class Workspace:
    """Filesystem scope using directory descriptors and no-follow path traversal.

    POSIX host implementation. No shell, network or arbitrary absolute paths.
    Write is atomic and may only replace an existing file when explicitly asked.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def parent(self, name):
        try:
            relative_path(name)
        except ValueError as exc:
            raise RuntimeFault("workspace_scope", "Expected a relative workspace path") from exc
        if os.name != "posix":
            raise RuntimeFault("unsupported_tool_platform", "Scoped filesystem tools currently require a POSIX host")
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            parts = name.split("/")
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            yield descriptor, parts[-1]
        except OSError as exc:
            raise RuntimeFault("workspace_io", str(exc)) from exc
        finally:
            os.close(descriptor)

    def read(self, args, context):
        context.check_cancelled()
        with self.parent(args["path"]) as (directory, name):
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            with os.fdopen(descriptor, "rb") as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise RuntimeFault("workspace_scope", "Only regular files can be read")
                data = source.read(512 * 1024 + 1)
                if len(data) > 512 * 1024:
                    raise RuntimeFault("tool_output_too_large", "Text file exceeds 512 KiB")
        try:
            return {"path": args["path"], "text": data.decode("utf-8")}
        except UnicodeDecodeError as exc:
            raise RuntimeFault("invalid_text", "File is not UTF-8 text") from exc

    def write(self, args, context):
        data = args["text"].encode("utf-8")
        if len(data) > 512 * 1024:
            raise RuntimeFault("invalid_tool_arguments", "Text exceeds 512 KiB")
        context.check_cancelled()
        with self.parent(args["path"]) as (directory, name):
            temporary = ".phaseone-write-" + hashlib.sha256(context.action_id.encode()).hexdigest()
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            try:
                with os.fdopen(fd, "wb") as destination:
                    destination.write(data)
                    destination.flush()
                    os.fsync(destination.fileno())
                context.check_cancelled()
                if args.get("replace", False):
                    os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
                else:
                    os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
                os.fsync(directory)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
        return {"path": args["path"], "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def register_builtin_tools(runtime, *, workspace: Path | None = None):
    gateway = runtime.tool_gateway
    gateway.register({"name": "device_info", "description": "Report the device and installed inference runtimes.",
                      "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                      "permission": "device.read"}, lambda args, context: runtime.device,
                     access=lambda args: [{"resource": "device:local", "action": "read"}])
    if workspace is not None:
        files = Workspace(workspace)
        path = {"type": "string", "minLength": 1, "maxLength": 1024}
        gateway.register({"name": "read_text", "description": "Read a UTF-8 file inside the configured workspace.",
                          "parameters": {"type": "object", "properties": {"path": path}, "required": ["path"], "additionalProperties": False},
                          "permission": "files.read"}, files.read,
                         access=lambda args: [{"resource": "workspace:" + relative_path(args["path"]), "action": "read"}])
        gateway.register({"name": "write_text", "description": "Write a UTF-8 file inside the configured workspace.",
                          "parameters": {"type": "object", "properties": {"path": path, "text": {"type": "string", "maxLength": 524288},
                                                                                "replace": {"type": "boolean", "default": False}},
                                         "required": ["path", "text"], "additionalProperties": False},
                          "permission": "files.write", "side_effect": True}, files.write,
                         access=lambda args: [{"resource": "workspace:" + relative_path(args["path"]), "action": "write"}])
