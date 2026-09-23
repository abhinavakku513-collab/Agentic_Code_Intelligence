"""Structured logging (docs/spec/06 §7). JSON lines, no query text and no code by default.

Privacy rule: an event may carry hashes, lengths and timings. Raw query text or snippet bodies are logged only when
`ACIS_LOG_TEXT=1` is set explicitly by a human, and even then they are truncated.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Any

import structlog

_CONFIGURED = False
TEXT_ALLOWED = "ACIS_LOG_TEXT"
_TEXT_CAP = 200


def _redact(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop or truncate anything that could carry user text or code."""
    allow_text = os.environ.get(TEXT_ALLOWED) == "1"
    for key in list(event_dict):
        if key in ("query", "text", "body", "source", "snippet", "code"):
            value = event_dict.pop(key)
            if allow_text:
                event_dict[f"{key}_redacted"] = str(value)[:_TEXT_CAP]
            else:
                event_dict[f"{key}_len"] = len(str(value))
    return event_dict


class _CurrentStderr:
    """A stand-in for `sys.stderr` that resolves it at write time, and never raises at the caller.

    structlog binds the stream object when logging is configured. If that object is later replaced or closed —
    which pytest's capture does routinely, and a daemonised run can do too — every subsequent log call raises
    `ValueError: I/O operation on closed file` **inside whatever was being logged**. Observability must not be
    able to break the thing it observes, so the stream is looked up per write and a failed write is dropped.
    """

    def write(self, data: str) -> int:
        try:
            return sys.stderr.write(data)
        except (ValueError, OSError):
            return 0

    def flush(self) -> None:
        with contextlib.suppress(ValueError, OSError):
            sys.stderr.flush()


def configure(level: str | int | None = None, *, json_lines: bool = True) -> None:
    """Idempotent logging setup. Safe to call from the CLI, the API and tests."""
    global _CONFIGURED  # noqa: PLW0603 — module-level idempotence flag
    lvl = level if level is not None else os.environ.get("ACIS_LOG_LEVEL", "INFO")
    numeric = logging.getLevelName(lvl.upper()) if isinstance(lvl, str) else lvl
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=numeric, force=True)
    renderer = structlog.processors.JSONRenderer() if json_lines else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact,
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(_CurrentStderr()),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str = "acis") -> Any:
    if not _CONFIGURED:
        configure()
    return structlog.get_logger(name)


__all__ = ["TEXT_ALLOWED", "configure", "get_logger"]
