"""Per-country concentration limit.

Caps the share of gross notional any single country may carry. The country is
derived from the underlying exchange so that the multiple Datastream mnemonics
for one market (e.g. ``TKS``/``TSE`` → Japan) bucket together.
"""

from __future__ import annotations

from asian_adr.core import HongSusmelSignalEvent

from .base import AbstractRiskRule, RiskContext, RiskRuleResult

__all__ = ["CountryConcentrationRule", "EXCHANGE_COUNTRY", "country_of"]


# Datastream exchange mnemonic → country (architecture §3.4 exchange set).
EXCHANGE_COUNTRY: dict[str, str] = {
    "TKS": "JP", "TSE": "JP",
    "HKG": "HK",
    "KRX": "KR",
    "BOM": "IN", "BSE": "IN", "NSE": "IN",
    "ASX": "AU",
    "SES": "SG",
    "TAI": "TW",
    "IDX": "ID",
    "PHS": "PH",
    "KLS": "MY",
}


def country_of(exchange: str) -> str:
    """Map an exchange mnemonic to a country code (falls back to the mnemonic)."""
    return EXCHANGE_COUNTRY.get(exchange, exchange)


class CountryConcentrationRule(AbstractRiskRule):
    """Caps per-country notional at a fraction of gross *capacity*.

    The limit is measured against ``max_gross_notional`` (the portfolio's gross
    capacity), not the current realised gross. Measuring against current gross
    would block the first trade in any country — it is trivially 100% of a
    one-position book — which is not the intent of a diversification limit.
    """

    name = "country_concentration"

    def evaluate(
        self, signal: HongSusmelSignalEvent | None, ctx: RiskContext
    ) -> RiskRuleResult:
        capacity = ctx.config.max_gross_notional
        if capacity <= 0:
            return RiskRuleResult.ok(self.name)
        projected_country = ctx.country_notional_usd + ctx.proposed_notional
        concentration = projected_country / capacity
        if concentration > ctx.config.max_country_concentration_pct:
            return RiskRuleResult.block(
                self.name,
                f"country {ctx.country} exposure {concentration:.1%} of capacity "
                f"exceeds limit {ctx.config.max_country_concentration_pct:.1%}",
            )
        return RiskRuleResult.ok(self.name)
