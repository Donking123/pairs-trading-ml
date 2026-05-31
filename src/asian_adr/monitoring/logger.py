"""Structured JSON logging (architecture §11.1).

Uses ``structlog`` when available to emit JSON logs (shipped to Loki via
Promtail in production); otherwise falls back to a small stdlib-backed adapter
with the same ``logger.info("event", key=value)`` call style, so application
code never needs to branch on whether structlog is installed.

Imports only the standard library; ``structlog`` is optional.
"""

from __future__ import annotations

import json
import logging
from typing import Any

__all__ = ["configure_logging", "get_logger"]

_configured = False


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure process-wide structured logging (idempotent)."""
    global _configured
    if _configured:
        return
    log_level = getattr(logging, level.upper(), logging.INFO)
    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                (
                    structlog.processors.JSONRenderer()
                    if json_output
                    else structlog.dev.ConsoleRenderer()
                ),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            cache_logger_on_first_use=True,
        )
    except ImportError:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        )
    _configured = True


class _StdlibBoundLogger:
    """structlog-compatible adapter over a stdlib logger.

    Supports ``log.info("event", key=value)`` and ``log.bind(**ctx)`` so call
    sites work identically whether or not structlog is installed.
    """

    def __init__(self, logger: logging.Logger, context: dict[str, Any] | None = None) -> None:
        self._logger = logger
        self._context = context or {}

    def bind(self, **kwargs: Any) -> "_StdlibBoundLogger":
        return _StdlibBoundLogger(self._logger, {**self._context, **kwargs})

    def _emit(self, level: int, event: str, **kwargs: Any) -> None:
        payload = {**self._context, **kwargs}
        msg = event if not payload else f"{event} {json.dumps(payload, default=str)}"
        self._logger.log(level, msg)

    def debug(self, event: str, **kw: Any) -> None:
        self._emit(logging.DEBUG, event, **kw)

    def info(self, event: str, **kw: Any) -> None:
        self._emit(logging.INFO, event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._emit(logging.WARNING, event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._emit(logging.ERROR, event, **kw)

    def critical(self, event: str, **kw: Any) -> None:
        self._emit(logging.CRITICAL, event, **kw)


def get_logger(name: str | None = None) -> Any:
    """Return a structured logger (structlog if available, else stdlib adapter)."""
    try:
        import structlog

        return structlog.get_logger(name)
    except ImportError:
        return _StdlibBoundLogger(logging.getLogger(name or "asian_adr"))
