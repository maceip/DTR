"""Stable, transport-independent runtime failures."""

from typing import Any


class RuntimeFault(Exception):
    def __init__(self, code: str, message: str, *, details: Any = None, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.details = details
        self.retryable = retryable

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": str(self),
            "details": self.details,
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "RuntimeFault":
        return cls(
            value["code"], value["message"],
            details=value.get("details"), retryable=value.get("retryable", False),
        )
