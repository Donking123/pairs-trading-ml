"""Health and readiness checks (architecture §1.6, §3.13).

:class:`HealthCheck` aggregates named component checks and tracks the system
lifecycle state (``STARTING`` → ``READY`` per the recovery sequence).
:class:`HealthServer` exposes ``/health`` (liveness) and ``/ready`` (readiness)
over ``aiohttp`` when available.

Imports only the standard library; ``aiohttp`` is optional.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

__all__ = ["HealthStatus", "HealthCheck", "HealthServer"]

CheckFn = Callable[[], "bool | Awaitable[bool]"]


class HealthStatus(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPING = "stopping"


class HealthCheck:
    """Tracks lifecycle state and runs registered component checks."""

    def __init__(self) -> None:
        self._status = HealthStatus.STARTING
        self._checks: dict[str, CheckFn] = {}

    @property
    def status(self) -> HealthStatus:
        return self._status

    def set_status(self, status: HealthStatus) -> None:
        logger.info("health status: %s → %s", self._status.value, status.value)
        self._status = status

    def mark_ready(self) -> None:
        self.set_status(HealthStatus.READY)

    def register(self, name: str, check: CheckFn) -> None:
        self._checks[name] = check

    async def run_checks(self) -> dict[str, bool]:
        import inspect

        results: dict[str, bool] = {}
        for name, check in self._checks.items():
            try:
                outcome = check()
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                results[name] = bool(outcome)
            except Exception:  # pragma: no cover - defensive
                logger.exception("health check %s failed", name)
                results[name] = False
        return results

    async def is_live(self) -> bool:
        return self._status != HealthStatus.STOPPING

    async def is_ready(self) -> bool:
        if self._status != HealthStatus.READY:
            return False
        results = await self.run_checks()
        return all(results.values())


class HealthServer:
    """Serves ``/health`` and ``/ready`` over aiohttp (optional dependency)."""

    def __init__(self, health: HealthCheck, *, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._health = health
        self._host = host
        self._port = port
        self._runner = None

    async def start(self) -> None:  # pragma: no cover - requires aiohttp
        try:
            from aiohttp import web
        except ImportError as exc:
            raise RuntimeError("HealthServer requires 'aiohttp' (uv add aiohttp).") from exc

        async def health_handler(_request):
            live = await self._health.is_live()
            return web.json_response(
                {"status": self._health.status.value, "live": live},
                status=200 if live else 503,
            )

        async def ready_handler(_request):
            ready = await self._health.is_ready()
            checks = await self._health.run_checks()
            return web.json_response(
                {"ready": ready, "status": self._health.status.value, "checks": checks},
                status=200 if ready else 503,
            )

        app = web.Application()
        app.router.add_get("/health", health_handler)
        app.router.add_get("/ready", ready_handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        logger.info("health server on %s:%d", self._host, self._port)

    async def stop(self) -> None:  # pragma: no cover - requires aiohttp
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
