"""Runner configuration loaded from TOML (with safe defaults).

A single :class:`RunnerConfig` backs all three runners. Every field has a
sensible default, so the runners work without any config file present;
``from_toml`` overlays whatever a TOML file provides. Builders translate it
into the layer-specific configs (:class:`RiskConfig`, ``BacktestConfig``).

Uses only the standard library, ``asian_adr.risk``, and ``asian_adr.backtest``.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from asian_adr.backtest import BacktestConfig, EquitiesCostModel
from asian_adr.risk import RiskConfig

logger = logging.getLogger(__name__)

__all__ = ["RunnerConfig"]

_DATA = "datastream/data/parquet"
_PAIRS = "datastream/config/pairs/asian_adr_pairs.json"


@dataclass
class RunnerConfig:
    """Cross-runner configuration."""

    environment: str = "development"

    # Data sources
    pairs_json: str = _PAIRS
    adr_prices: str = f"{_DATA}/adr/adr_prices.parquet"
    global_prices: str = f"{_DATA}/global/global_prices.parquet"
    fx_rates: str = f"{_DATA}/fx/fx_rates.parquet"
    backtest_out_dir: str = "datastream/data/backtest"

    # Strategy parameter overrides (empty → use each pair's registered values)
    k0: float | None = None
    kc: float | None = None
    estimation_days: int | None = None
    holding_days: int | None = None

    # Sizing / risk
    notional_per_leg: float = 100_000.0
    aum_usd: float = 1_000_000.0
    max_open_pairs: int = 20

    # Live connectivity (Phase 3/5 — used once connectors exist)
    database_url: str = "postgresql://localhost:5432/asian_adr"
    ib_tws_port: int = 7497
    recording_path: str = "datastream/data/recordings/session.jsonl"

    # -- loading -------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | Path | None) -> "RunnerConfig":
        cfg = cls()
        if path is None:
            return cfg
        p = Path(path)
        if not p.exists():
            logger.warning("config file %s not found; using defaults", p)
            return cfg
        data = tomllib.loads(p.read_text())
        # Accept either flat keys or sectioned ([data], [risk], [live]) tables.
        flat: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
        known = {f.name for f in cls.__dataclass_fields__.values()}
        for k, v in flat.items():
            if k in known:
                setattr(cfg, k, v)
        return cfg

    # -- builders ------------------------------------------------------------

    def param_overrides(self) -> dict:
        ov: dict = {}
        if self.k0 is not None:
            ov["k0"] = self.k0
        if self.kc is not None:
            ov["kc"] = self.kc
        if self.estimation_days is not None:
            ov["estimation_days"] = self.estimation_days
        if self.holding_days is not None:
            ov["holding_days"] = self.holding_days
        return ov

    def risk_config(self) -> RiskConfig:
        return RiskConfig(
            aum_usd=Decimal(str(self.aum_usd)),
            max_open_pairs=self.max_open_pairs,
            max_notional_per_leg=Decimal(str(self.notional_per_leg)),
            entry_notional_usd=Decimal(str(self.notional_per_leg)),
        )

    def backtest_config(self, extra_overrides: dict | None = None) -> BacktestConfig:
        overrides = self.param_overrides()
        if extra_overrides:
            overrides.update(extra_overrides)
        return BacktestConfig(
            risk_config=self.risk_config(),
            cost_model=EquitiesCostModel(),
            notional_per_leg=Decimal(str(self.notional_per_leg)),
            param_overrides=overrides,
        )
