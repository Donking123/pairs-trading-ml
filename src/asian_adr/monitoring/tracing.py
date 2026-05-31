"""OpenTelemetry tracing setup (architecture §11).

Configures an OTLP tracer when ``opentelemetry`` is installed; otherwise returns
a no-op tracer whose spans are inert context managers, so ``with traced(...)``
works everywhere without branching.

Imports only the standard library; ``opentelemetry`` is optional.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

__all__ = ["configure_tracing", "get_tracer", "traced", "HAVE_OTEL"]

try:  # pragma: no cover - env-dependent
    from opentelemetry import trace as _otel_trace

    HAVE_OTEL = True
except ImportError:
    _otel_trace = None
    HAVE_OTEL = False


class _NoOpSpan:
    def set_attribute(self, *a, **k) -> None: ...
    def record_exception(self, *a, **k) -> None: ...
    def __enter__(self) -> "_NoOpSpan":
        return self
    def __exit__(self, *exc) -> bool:
        return False


class _NoOpTracer:
    def start_as_current_span(self, name: str, **kwargs) -> _NoOpSpan:
        return _NoOpSpan()


_tracer: Any = _NoOpTracer()


def configure_tracing(
    service_name: str = "asian-adr",
    *,
    endpoint: str | None = None,
) -> None:
    """Configure the OTLP tracer if OpenTelemetry is available (idempotent)."""
    global _tracer
    if not HAVE_OTEL:
        logger.info("opentelemetry not installed; tracing is a no-op")
        return
    try:  # pragma: no cover - requires otel
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        _otel_trace.set_tracer_provider(provider)
        _tracer = _otel_trace.get_tracer(service_name)
    except Exception as exc:  # pragma: no cover
        logger.warning("tracing setup failed (%s); using no-op tracer", exc)
        _tracer = _NoOpTracer()


def get_tracer(name: str | None = None) -> Any:
    if HAVE_OTEL and not isinstance(_tracer, _NoOpTracer):  # pragma: no cover
        return _otel_trace.get_tracer(name or "asian-adr")
    return _tracer


@contextmanager
def traced(name: str, **attributes: Any) -> Iterator[Any]:
    """Span context manager — a real span with OTel, else a no-op."""
    span_cm = get_tracer().start_as_current_span(name)
    with span_cm as span:
        for key, value in attributes.items():
            try:
                span.set_attribute(key, value)
            except Exception:  # pragma: no cover
                pass
        yield span
