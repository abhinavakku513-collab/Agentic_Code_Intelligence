"""Typed errors (docs/spec/06 §1). Every error carries the HTTP status the API surface must map it to."""

from __future__ import annotations


class AcisError(Exception):
    """Base class for every error the engine raises on purpose."""

    http_status: int = 500
    code: str = "acis_error"

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = dict(context)

    def __str__(self) -> str:
        """The class name is part of the text so that callers (and `tests/robustness`) can match on the type."""
        return f"{type(self).__name__}: {self.message}"

    def to_problem(self) -> dict[str, object]:
        """RFC 7807 `application/problem+json` body (docs/spec/06 §2)."""
        return {
            "type": f"about:acis/{self.code}",
            "title": self.code,
            "status": self.http_status,
            "detail": self.message,
            **self.context,
        }


class InvalidInput(AcisError):
    http_status = 400
    code = "invalid_input"


class NotFound(AcisError):
    http_status = 404
    code = "not_found"


class IndexRequired(AcisError):
    http_status = 409
    code = "index_required"


class SnapshotInvalid(AcisError):
    http_status = 409
    code = "snapshot_invalid"


class VersionConflict(AcisError):
    http_status = 409
    code = "version_conflict"


class ResourceLimit(AcisError):
    http_status = 413
    code = "resource_limit"


class NotReady(AcisError):
    http_status = 503
    code = "not_ready"


class StrictViolation(AcisError):
    """Raised in strict mode (official runs) whenever a degradation would otherwise be recorded (INV-7)."""

    http_status = 500
    code = "strict_violation"


class SealedDataAccess(AcisError):
    """Raised when TEST labels are reached outside `acis.eval.final` (INV-8). Never returned over the API."""

    http_status = 500
    code = "sealed_data_access"


__all__ = [
    "AcisError",
    "IndexRequired",
    "InvalidInput",
    "NotFound",
    "NotReady",
    "ResourceLimit",
    "SealedDataAccess",
    "SnapshotInvalid",
    "StrictViolation",
    "VersionConflict",
]
