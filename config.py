"""
config.py  —  All parameters in one place. Edit this before running anything else.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path("/Users/deepakgarrepalli/Desktop/dev/MQF/MQF/Term 3/Quantitative Trading Strategies")
DATA_RAW  = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
DATA_RES  = BASE_DIR / "data" / "results"

for d in [DATA_RAW, DATA_PROC, DATA_RES]:
    d.mkdir(parents=True, exist_ok=True)

# ── WRDS ───────────────────────────────────────────────────────────────────────
WRDS_USERNAME = "deepakg2025"

# ── Date range ─────────────────────────────────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE   = "2023-12-31"

# ── Universe ───────────────────────────────────────────────────────────────────
UNIVERSE           = "sp500"
MIN_DOLLAR_VOLUME  = 1_000_000   # $1M average daily dollar volume minimum

# ── Fama-French 5 factors + Momentum ──────────────────────────────────────────
# Pulled from WRDS ff.fivefactors_daily + ff.factors_daily.
# These 6 factors are orthogonal by construction — unlike the sector ETFs
# (XLF/XLK/SPY) which are highly correlated and made Ridge betas noisy.
FF5_FACTORS = {
    "mktrf": "Market",
    "smb":   "Size",
    "hml":   "Value",
    "rmw":   "Profitability",
    "cma":   "Investment",
    "umd":   "Momentum",
}
FACTOR_NAMES = list(FF5_FACTORS.values())

# ── Rolling window ─────────────────────────────────────────────────────────────
FORMATION_DAYS  = 756   # 3 years: paper-faithful formation window
TRADING_DAYS    = 63    # 3 months: enough for pairs with half-life up to ~30d to revert
ROLL_STEP_DAYS  = 21    # roll forward 1 month each iteration (3 overlapping cohorts)
MIN_OBS_FRAC    = 0.60  # require ≥60% non-NaN days in formation window

# ── Regression ────────────────────────────────────────────────────────────────
# Ridge regression (RidgeCV) handles multicollinearity among correlated factor
# ETFs (XLF, XLK, SPY etc.). L2 penalty shrinks and stabilises beta estimates.
# λ is cross-validated per stock per window from RIDGE_ALPHAS in each script.

# ── Clustering ─────────────────────────────────────────────────────────────────
# FIX: lowered from 0.6 → 0.4 to produce smaller, tighter clusters.
# Previous clusters averaged 20-35 stocks — too large, generating too many
# spurious candidate pairs. Target: 8-15 stocks per cluster.
CLUSTER_DISTANCE_THRESHOLD = 0.15
MIN_CLUSTER_SIZE            = 5
MAX_CLUSTER_SIZE            = 30   # skip giant catch-all clusters (too many spurious pairs)

# ── Cointegration ──────────────────────────────────────────────────────────────
# FIX: tightened from 0.05 → 0.01 (Bonferroni-style correction).
# At 0.05 with 15k candidates, ~750 pairs pass by pure chance with no real
# mean-reversion. At 0.01 this drops to ~150, keeping only robust pairs.
COINT_PVALUE_THRESHOLD = 0.01

# Half-life filter: only keep pairs whose spread reverts within this range.
# Too short (<5 days) = microstructure noise, too long (>60 days) = slow
# mean-reversion that routinely hits stop-loss before converging.
HALFLIFE_MIN_DAYS = 5
HALFLIFE_MAX_DAYS = 60

# ── Trading signal ─────────────────────────────────────────────────────────────
ENTRY_ZSCORE    = 2.0
EXIT_ZSCORE     = 0.0    # exit when spread crosses mean

# FIX: widened stop-loss from 4.0 → 5.0.
# 95.8% of trades were hitting stop-loss at ±4σ — the spread was routinely
# drifting past ±4σ before reverting. ±5σ gives more room while still
# protecting against genuine pair breakdown.
STOPLOSS_ZSCORE = 5.0

# ── Transaction costs (IBKR Pro Fixed pricing, S&P 500 stocks) ────────────────
# Source: interactivebrokers.com/en/pricing/commissions-stocks.php
#         interactivebrokers.com/en/pricing/short-sale-cost.php
#
# Commission:  $0.005/share, min $1.00 per order, max 1% of trade value
#              For a $10,000 leg (~133 shares of avg $75 stock): $1.00 = 1 bps
#              Both legs (long + short) = $2.00 = 2 bps of $10,000 notional
#
# Bid-ask spread: ~1.5 bps one-way for S&P 500 large-caps (crossing spread)
#                 Both legs: 3 bps round-trip
#
# Total round-trip: 2 bps commission + 3 bps spread ≈ 5 bps
# (we use 5 bps; the original 10 bps was 2× too conservative for IBKR)
#
# Short borrow: IBKR General Collateral rate = 0.25% per year (25 bps)
# Source: interactivebrokers.com/campus/traders-insight/securities/
#         short-selling/the-risks-of-shorting-series-part-ii-borrow-fees/
# (S&P 500 stocks are GC; hard-to-borrow stocks can be 1-10%+ but excluded here)
ROUND_TRIP_BPS   = 5     # was 10; updated to IBKR Pro rate (2 bps commission + 3 bps spread)
SHORT_BORROW_BPS = 25    # was 35; updated to IBKR GC rate of 0.25%/year

# ── Parallelisation ────────────────────────────────────────────────────────────
N_JOBS = 4   # threading backend used in 04 — safe on Mac, no spawn overhead
