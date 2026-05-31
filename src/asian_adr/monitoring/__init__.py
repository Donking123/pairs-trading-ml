"""Observability: structured logging, metrics, tracing, alerting, health.

* :mod:`logger`   — structured JSON logging (structlog, stdlib fallback).
* :mod:`metrics`  — Prometheus metrics + a bus :class:`MetricsCollector`.
* :mod:`tracing`  — OpenTelemetry tracer (no-op fallback).
* :mod:`alerting` — :class:`AlertEngine` + dispatcher (log / Slack / PagerDuty).
* :mod:`health`   — liveness / readiness checks and HTTP server.

All external observability backends are optional and imported lazily, so this
package is import-safe and the system runs with or without them. Imports only
``core`` and ``event_bus``.
"""

from __future__ import annotations

from .alerting import (
    Alert,
    AlertDispatcher,
    AlertEngine,
    AlertSeverity,
    LogChannel,
    PagerDutyChannel,
    SlackChannel,
)
from .health import HealthCheck, HealthServer, HealthStatus
from .logger import configure_logging, get_logger
from .metrics import MetricsCollector, start_metrics_server
from .tracing import configure_tracing, get_tracer, traced

__all__ = [
    "configure_logging",
    "get_logger",
    "MetricsCollector",
    "start_metrics_server",
    "configure_tracing",
    "get_tracer",
    "traced",
    "Alert",
    "AlertSeverity",
    "AlertEngine",
    "AlertDispatcher",
    "LogChannel",
    "SlackChannel",
    "PagerDutyChannel",
    "HealthCheck",
    "HealthServer",
    "HealthStatus",
]
