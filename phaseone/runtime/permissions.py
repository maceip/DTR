"""Application-owned resource permissions, independent of model or engine.

Resource IDs are opaque, exact matches. There are no wildcard grants and no
implicit escalation from read/write to export, execute, or training.
"""

import threading
from contextlib import contextmanager
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import RuntimeFault

Identity = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$")]
ResourceIdentity = Annotated[str, Field(min_length=1, max_length=1024,
                                       pattern=r"^[^\x00-\x1f\x7f*?]+$")]
Action = Literal["read", "write", "execute", "export", "train"]


class ResourceAccess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    resource: ResourceIdentity
    action: Action


class PermissionRule(ResourceAccess):
    subject: Identity
    effect: Literal["allow", "deny"]


class PermissionDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1]
    rules: list[PermissionRule] = Field(max_length=4096)

    @field_validator("version", mode="before")
    @classmethod
    def numeric_version(cls, value):
        if isinstance(value, bool):
            raise ValueError("Policy version must be a number")
        return value


class PermissionPolicy:
    """Trusted app configuration; never populate this from model output.

    Supply rules from the app's existing authentication/consent system. Updating
    rules revokes access on the next invocation, including cached tool results.
    The synchronous authorization scope serializes updates against tool entry;
    it cannot undo an operation that has already started.
    """

    def __init__(self, document: PermissionDocument | dict):
        self._lock = threading.RLock()
        self._revision = 0
        self.replace(document)

    @property
    def revision(self):
        with self._lock:
            return self._revision

    def replace(self, document: PermissionDocument | dict):
        document = PermissionDocument.model_validate(document)
        with self._lock:
            self._rules = tuple(document.rules)
            self._revision += 1

    def allows(self, subject: str, access: ResourceAccess | dict) -> bool:
        access = ResourceAccess.model_validate(access)
        with self._lock:
            matches = [rule for rule in self._rules
                       if rule.subject == subject and rule.resource == access.resource
                       and rule.action == access.action]
            return any(rule.effect == "allow" for rule in matches) and not any(
                rule.effect == "deny" for rule in matches)

    @contextmanager
    def authorize(self, subject: str, requirements):
        accesses = tuple(ResourceAccess.model_validate(item) for item in requirements)
        if not accesses or len(accesses) > 128:
            raise RuntimeFault("tool_scope_missing", "Declare 1 to 128 resource operations")
        with self._lock:
            for access in accesses:
                if not self.allows(subject, access):
                    raise RuntimeFault("permission_denied", "Resource operation is not permitted",
                                       details={"subject": subject, **access.model_dump(),
                                                "policy_revision": self._revision})
            yield {"subject": subject, "policy_revision": self._revision,
                   "access": [item.model_dump() for item in accesses]}
