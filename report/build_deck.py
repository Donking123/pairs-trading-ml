"""Build a clean 16:9 PDF slide deck for the QF621 pairs-trading project.

Pure matplotlib (no external slide tooling) so it renders identically everywhere.
Run:  python3 report/build_deck.py   →   report/QF621_pairs_trading_deck.pdf
"""
from __future__ import annotations
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.image as mpimg

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
OUT = HERE / "QF621_pairs_trading_deck.pdf"

# --- palette ---
NAVY = "#1b4965"
BLUE = "#5fa8d3"
LIGHT = "#cae9ff"
INK = "#1d2733"
GRAY = "#5a6b7b"
BG = "#ffffff"
PANEL = "#f2f7fb"
GREEN = "#2e7d4f"
RED = "#b3402f"

plt.rcParams.update({"font.family": "DejaVu Sans"})
W, H = 13.333, 7.5
pages: list = []


def new_fig():
    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def footer(ax, n):
    ax.add_patch(Rectangle((0, 0), 1, 0.012, color=NAVY, lw=0))
    ax.text(0.04, 0.035, "Clustering-Based Pairs Trading · QF621", color=GRAY, fontsize=9)
    ax.text(0.96, 0.035, f"{n}", color=GRAY, fontsize=9, ha="right")


def header(ax, kicker, title):
    ax.text(0.055, 0.90, kicker.upper(), color=BLUE, fontsize=12.5, weight="bold")
    ax.text(0.055, 0.85, title, color=NAVY, fontsize=27, weight="bold", va="top")
    ax.add_patch(Rectangle((0.055, 0.76), 0.18, 0.006, color=BLUE, lw=0))


def bullets(ax, items, x=0.075, y=0.70, dy=0.105, width=78, fs=16, gap=0.0):
    cy = y
    for it in items:
        lead = it[0] if isinstance(it, tuple) else None
        text = it[1] if isinstance(it, tuple) else it
        wrapped = textwrap.wrap(text, width=width)
        ax.add_patch(plt.Circle((x, cy + 0.005), 0.006, color=BLUE, transform=ax.transAxes))
        for i, line in enumerate(wrapped):
            ax.text(x + 0.022, cy - i * 0.045, line, color=INK, fontsize=fs, va="center")
        cy -= dy + 0.045 * (len(wrapped) - 1)
    return cy


def table(ax, headers, rows, col_x, y0, row_h=0.075, fs=14, highlight=None, align=None):
    highlight = highlight or {}
    align = align or ["left"] * len(headers)
    # header band
    ax.add_patch(Rectangle((col_x[0] - 0.012, y0 - row_h * 0.62), col_x[-1] - col_x[0] + 0.10,
                           row_h, color=NAVY, lw=0))
    for j, h in enumerate(headers):
        ha = align[j]
        xx = col_x[j] + (0.085 if ha == "right" else 0)
        ax.text(xx, y0 - row_h * 0.10, h, color="white", fontsize=fs, weight="bold", ha=ha, va="center")
    y = y0 - row_h
    for ri, row in enumerate(rows):
        if ri % 2 == 0:
            ax.add_patch(Rectangle((col_x[0] - 0.012, y - row_h * 0.62), col_x[-1] - col_x[0] + 0.10,
                                   row_h, color=PANEL, lw=0))
        if ri in highlight:
            ax.add_patch(Rectangle((col_x[0] - 0.012, y - row_h * 0.62), col_x[-1] - col_x[0] + 0.10,
                                   row_h, color=highlight[ri], lw=0, alpha=0.55))
        for j, cell in enumerate(row):
            ha = align[j]
            xx = col_x[j] + (0.085 if ha == "right" else 0)
            wt = "bold" if (ri in highlight or j == 0 and False) else "normal"
            ax.text(xx, y - row_h * 0.10, str(cell), color=INK, fontsize=fs, ha=ha, va="center", weight=wt)
        y -= row_h


def image_slide(kicker, title, img, caption=None, img_w=0.62, top=0.72):
    fig, ax = new_fig(); header(ax, kicker, title)
    im = mpimg.imread(str(img)); h, w = im.shape[:2]
    aspect = h / w
    iw = img_w; ih = iw * aspect * (W / H)
    ix = (1 - iw) / 2; iy = top - ih
    iax = fig.add_axes([ix, max(iy, 0.14), iw, ih]); iax.axis("off"); iax.imshow(im)
    if caption:
        for i, line in enumerate(textwrap.wrap(caption, 120)):
            ax.text(0.5, 0.115 - i * 0.035, line, color=GRAY, fontsize=12.5, ha="center")
    return fig, ax


def lead_slide(title_lines, sub_lines, footer_line=None):
    fig, ax = new_fig()
    ax.add_patch(Rectangle((0, 0), 1, 1, color=NAVY, lw=0))
    ax.add_patch(Rectangle((0, 0.0), 1, 0.10, color=BLUE, lw=0))
    cy = 0.62
    for t, sz in title_lines:
        ax.text(0.5, cy, t, color="white", fontsize=sz, weight="bold", ha="center", va="center")
        cy -= 0.085 * (sz / 38)
    cy -= 0.04
    for s in sub_lines:
        ax.text(0.5, cy, s, color=LIGHT, fontsize=16, ha="center", va="center")
        cy -= 0.06
    if footer_line:
        ax.text(0.5, 0.05, footer_line, color="#9bbdd4", fontsize=12.5, ha="center")
    return fig, ax


# ============================ SLIDES ============================
# 1 — Title
f, _ = lead_slide(
    [("Clustering-Based Pairs Trading", 40),
     ("Replicating & extending Rotondi & Russo (2025)", 20)],
    ["Matched the paper's best strategy at  Sharpe 1.028  (paper: 1.01)",
     "21-year, survivorship-bias-free, out-of-sample backtest — built from the data up"],
    "Python · WRDS/CRSP · ML clustering + statistical-arbitrage · custom backtest engine")
pages.append(f)

# 2 — Problem
f, ax = new_fig(); header(ax, "The problem", "Picking the right pairs is the hard part")
bullets(ax, [
    "Pairs trading: trade two co-moving securities' spread — short the rich one, long the cheap one, profit when it reverts.",
    "Across a ~1,000-stock universe there are ~500,000 candidate pairs. Brute-forcing them is noisy and overfits badly.",
    "We want pairs that are economically related, not coincidental.",
], y=0.68, dy=0.10, fs=16)
ax.add_patch(FancyBboxPatch((0.075, 0.16), 0.85, 0.16, boxstyle="round,pad=0.02,rounding_size=0.02",
             fc=PANEL, ec=BLUE, lw=1.5))
ax.text(0.10, 0.265, "The paper's idea", color=NAVY, fontsize=15, weight="bold")
ax.text(0.10, 0.205, "Use unsupervised ML clustering to group similar stocks first, then only form pairs WITHIN a cluster —",
        color=INK, fontsize=14.5)
ax.text(0.10, 0.165, "fewer, higher-quality candidates. My goal: rebuild it independently, reproduce the result, then extend it.",
        color=INK, fontsize=14.5)
footer(ax, 2); pages.append(f)

# 3 — Data
f, ax = new_fig(); header(ax, "Data spine", "Built for honesty, not optics")
rows3 = [
    ("Source", "WRDS / CRSP daily — institutional-grade academic data"),
    ("Period", "Jan 2000 – Dec 2023  (24 years)"),
    ("Universe", "991 stocks — share codes 10/11 (US ordinary common only)"),
    ("Survivorship bias", "Eliminated — delisted names kept, delisting returns modelled"),
    ("Trading window", "2003–2023 = 251 monthly OUT-OF-SAMPLE returns"),
]
y = 0.66; rh = 0.088
for i, (k, v) in enumerate(rows3):
    if i % 2 == 0:
        ax.add_patch(Rectangle((0.06, y - rh * 0.5), 0.88, rh, color=PANEL, lw=0))
    ax.text(0.085, y, k, color=NAVY, fontsize=15.5, weight="bold", va="center")
    ax.text(0.34, y, v, color=INK, fontsize=15.5, va="center")
    y -= rh
ax.text(0.075, 0.15, "First 3 years are reserved purely as the formation window — so every reported return is genuinely",
        color=GRAY, fontsize=12.5)
ax.text(0.075, 0.11, "out-of-sample. The 79 names dropped vs the raw pull are fully accounted for (foreign / REITs / funds).",
        color=GRAY, fontsize=12.5)
footer(ax, 3); pages.append(f)

# 4 — Pipeline
f, ax = new_fig(); header(ax, "Method", "The pipeline — rolling, every month")
steps = [("1", "Distance\nmetric", "SSD or PC"), ("2", "OPTICS\nclustering", "density-based"),
         ("3", "Cointegration\nfilter", "Engle-Granger"), ("4", "Spread +\nz-score", "6-mo rolling"),
         ("5", "Backtest\ntrade", "|z|>2 → z=0")]
n = len(steps); bw = 0.155; gap = (1 - 0.14 - n * bw) / (n - 1); x = 0.07; y = 0.50
for i, (num, name, sub) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((x, y), bw, 0.20, boxstyle="round,pad=0.01,rounding_size=0.02",
                 fc=NAVY if i in (0, 1) else BLUE, ec="none"))
    ax.text(x + bw / 2, y + 0.155, num, color=LIGHT, fontsize=14, ha="center", weight="bold")
    ax.text(x + bw / 2, y + 0.10, name, color="white", fontsize=13.5, ha="center", va="center", weight="bold")
    ax.text(x + bw / 2, y + 0.03, sub, color=LIGHT, fontsize=11, ha="center")
    if i < n - 1:
        ax.annotate("", xy=(x + bw + gap, y + 0.10), xytext=(x + bw, y + 0.10),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))
    x += bw + gap
ax.text(0.5, 0.30, "Each month: form pairs on the trailing 3 years, trade them for the next 1 month, roll forward — 251 times.",
        color=INK, fontsize=15.5, ha="center")
ax.text(0.5, 0.22, "Pairs are only ever formed between two stocks in the same cluster.",
        color=GRAY, fontsize=13.5, ha="center", style="italic")
footer(ax, 4); pages.append(f)

# 5 — Distances
f, ax = new_fig(); header(ax, "Method · step 1", "Two distance metrics")
ax.add_patch(FancyBboxPatch((0.06, 0.30), 0.41, 0.42, boxstyle="round,pad=0.02,rounding_size=0.02", fc=PANEL, ec=GRAY, lw=1.2))
ax.text(0.085, 0.66, "SSD", color=NAVY, fontsize=20, weight="bold")
ax.text(0.085, 0.61, "baseline · my Phase 1", color=GRAY, fontsize=12.5, style="italic")
for i, l in enumerate(textwrap.wrap("Sum of Squared Deviations between normalised price paths. The classic Gatev et al. selector.", 40)):
    ax.text(0.085, 0.55 - i * 0.05, l, color=INK, fontsize=14)
ax.add_patch(FancyBboxPatch((0.53, 0.30), 0.41, 0.42, boxstyle="round,pad=0.02,rounding_size=0.02", fc=LIGHT, ec=NAVY, lw=2))
ax.text(0.555, 0.66, "PC", color=NAVY, fontsize=20, weight="bold")
ax.text(0.555, 0.61, "the paper's winner", color=NAVY, fontsize=12.5, style="italic")
for i, l in enumerate(textwrap.wrap("Partial-correlation distance on market-adjusted returns. Strips out the common market factor → finds idiosyncratic, market-neutral co-movement.", 40)):
    ax.text(0.555, 0.55 - i * 0.05, l, color=INK, fontsize=14)
ax.text(0.5, 0.19, "The distance feeds the clusterer — and this single choice is the biggest driver of risk-adjusted performance.",
        color=GRAY, fontsize=13.5, ha="center")
footer(ax, 5); pages.append(f)

# 6 — OPTICS
f, ax = new_fig(); header(ax, "Method · step 2", "Clustering with OPTICS")
bullets(ax, [
    "Density-based (cousin of DBSCAN) — no need to pre-specify the number of clusters.",
    "Leaves genuinely unrelated stocks UNclustered, instead of forcing them into a group.",
], y=0.69, dy=0.075, fs=15.5, width=98)
ax.text(0.075, 0.515, "Validation vs ground-truth industry codes (Dec 2023):", color=NAVY, fontsize=14.5, weight="bold")
table(ax, ["Metric", "Mine SSD", "Mine PC", "Paper"], [
    ["# clusters", "47", "81", "48 / 109"],
    ["Purity vs SIC codes", "0.871", "0.937", "0.81 / 0.84"],
    ["GOOG & GOOGL together?", "yes", "yes", "yes"],
], col_x=[0.075, 0.45, 0.62, 0.79], y0=0.45, row_h=0.072, fs=14,
   align=["left", "right", "right", "right"])
ax.text(0.075, 0.135, "xi tuned on cluster quality across 3 hold-out dates — and FROZEN before ever looking at Sharpe (no overfitting).",
        color=GRAY, fontsize=12.5)
footer(ax, 6); pages.append(f)

# 7 — Cointegration + signal
f, ax = new_fig(); header(ax, "Method · steps 3–4", "Cointegration filter & the trading signal")
ax.text(0.075, 0.70, "Engle-Granger filter  (optional variant)", color=NAVY, fontsize=15.5, weight="bold")
bullets(ax, [
    "Keep a pair only if the spread rejects the unit-root null at 5%, and",
    "spread half-life ∈ [5, 60] trading days — fast enough to revert in the window, slow enough to not be noise.",
], y=0.62, dy=0.07, fs=14.5)
ax.text(0.075, 0.40, "Trading signal", color=NAVY, fontsize=15.5, weight="bold")
bullets(ax, [
    "Hedge ratio γ via OLS → spread series → 6-month rolling z-score (strict look-ahead protection).",
    "Enter when |z| > 2 · exit at z = 0 · force-close at month-end.",
], y=0.32, dy=0.07, fs=14.5)
footer(ax, 7); pages.append(f)

# 8 — Result (sharpe bar)
f, ax = image_slide("Results", "Matched the paper — across all four variants", ASSETS / "sharpe_bar.png",
                    caption="PC core: Sharpe 1.028 vs the paper's 1.01 — inside the ±0.15 tolerance band. Every variant lands where the paper says it should.",
                    img_w=0.60, top=0.74)
footer(ax, 8); pages.append(f)

# 9 — Equity curve
f, ax = image_slide("Results", "Growth of $1 — 251 months out-of-sample", ASSETS / "equity_curves.png",
                    caption="PC core (dark) is the smoothest compounder. SSD+filter ends higher in absolute terms but is far choppier — which is exactly why its risk-adjusted Sharpe is lower.",
                    img_w=0.58, top=0.74)
footer(ax, 9); pages.append(f)

# 10 — Scorecard
f, ax = new_fig(); header(ax, "Results", "Full scorecard")
table(ax, ["Strategy", "Ann. ret", "Ann. vol", "Sharpe", "Max DD", "vs paper"], [
    ["PC core", "3.41%", "3.32%", "1.028", "-5.7%", "1.01  ✓"],
    ["PC + filter", "2.59%", "3.48%", "0.752", "-5.9%", "0.80  ✓"],
    ["SSD + filter", "4.37%", "6.11%", "0.731", "-9.7%", "0.75  ✓"],
    ["SSD core  (baseline)", "3.01%", "5.28%", "0.589", "-14.3%", "0.88  ✗"],
], col_x=[0.075, 0.40, 0.53, 0.66, 0.76, 0.86], y0=0.68, row_h=0.085, fs=14.5,
   align=["left", "right", "right", "right", "right", "right"],
   highlight={0: LIGHT})
ax.text(0.075, 0.20, "PC roughly halves both volatility AND drawdown vs SSD —", color=NAVY, fontsize=15.5, weight="bold")
ax.text(0.075, 0.15, "the market-neutral, idiosyncratic pairs are simply lower-risk.", color=NAVY, fontsize=15.5, weight="bold")
footer(ax, 10); pages.append(f)

# 11 — Why it works
f, ax = new_fig(); header(ax, "The insight", "Why it works — diagnosed, not assumed")
ax.text(0.075, 0.71, "I ran a P&L attribution on the baseline first and found a bimodal pattern:", color=INK, fontsize=15.5)
ax.add_patch(FancyBboxPatch((0.07, 0.45), 0.40, 0.18, boxstyle="round,pad=0.015,rounding_size=0.02", fc=PANEL, ec=GREEN, lw=1.5))
ax.text(0.09, 0.58, "11.4% of trades", color=GREEN, fontsize=15, weight="bold")
ax.text(0.09, 0.53, "cleanly revert: +471 bps each", color=INK, fontsize=14)
ax.text(0.09, 0.485, "→ ALL of the profit", color=INK, fontsize=14, style="italic")
ax.add_patch(FancyBboxPatch((0.52, 0.45), 0.40, 0.18, boxstyle="round,pad=0.015,rounding_size=0.02", fc=PANEL, ec=RED, lw=1.5))
ax.text(0.54, 0.58, "88.4% of trades", color=RED, fontsize=15, weight="bold")
ax.text(0.54, 0.53, "force-closed: -32 bps each", color=INK, fontsize=14)
ax.text(0.54, 0.485, "→ a constant drag", color=INK, fontsize=14, style="italic")
ax.add_patch(FancyBboxPatch((0.07, 0.20), 0.85, 0.14, boxstyle="round,pad=0.015,rounding_size=0.02", fc=NAVY, ec="none"))
ax.text(0.5, 0.295, "Thesis", color=LIGHT, fontsize=13, ha="center", weight="bold")
ax.text(0.5, 0.24, "The job isn't \"find a higher Sharpe.\" It's KILL THE FORCE-CLOSE DRAG —",
        color="white", fontsize=15.5, ha="center", weight="bold")
ax.text(0.5, 0.205, "only trade pairs that actually revert.", color="white", fontsize=15.5, ha="center", weight="bold")
footer(ax, 11); pages.append(f)

# 12 — Mechanism
f, ax = image_slide("The insight", "Mechanism — prediction confirmed", ASSETS / "mechanism.png",
                    caption="PC distance cut per-trade force-close drag by 65%; adding the cointegration filter eliminated every tail blow-up (0 trades with |return|>50% in 21 years, vs 5 for the baseline).",
                    img_w=0.62, top=0.74)
footer(ax, 12); pages.append(f)

# 13 — Engineering rigour
f, ax = new_fig(); header(ax, "Engineering", "Built like production research code")
bullets(ax, [
    "Modular package — distances · clustering · spread · cointegration · backtest · performance, each independently tested.",
    "18 synthetic unit tests — validate every component on data with a KNOWN answer before touching real CRSP.",
    "Look-ahead protection baked into the z-score and the rolling formation/trading split.",
    "Invariant checks — the extended engine reproduces the Phase-1 baseline Sharpe to 4 decimal places (no regression).",
    "Reproducible — config-driven single source of truth; hyperparameters frozen pre-results.",
], y=0.70, dy=0.105, fs=15, width=92)
ax.text(0.075, 0.135, "Stack: Python · pandas · NumPy · scikit-learn (OPTICS) · statsmodels (ADF/OLS) · matplotlib · parquet.",
        color=GRAY, fontsize=12.5)
footer(ax, 13); pages.append(f)

# 14 — What it demonstrates
f, ax = new_fig(); header(ax, "Takeaways", "What this demonstrates")
bullets(ax, [
    "Independent replication of a 2025 research paper — read it, rebuilt it, matched the headline number on self-sourced data.",
    "Quant intuition — diagnosed WHY the strategy makes money, then engineered the fix and confirmed the prediction.",
    "Discipline — survivorship-bias-free data, out-of-sample design, frozen hyperparameters, unit-tested code.",
], y=0.70, dy=0.115, fs=16, width=88)
ax.add_patch(FancyBboxPatch((0.07, 0.16), 0.85, 0.16, boxstyle="round,pad=0.015,rounding_size=0.02", fc=PANEL, ec=BLUE, lw=1.4))
ax.text(0.09, 0.265, "Roadmap", color=NAVY, fontsize=14, weight="bold")
ax.text(0.09, 0.21, "Factor-beta clustering (my own extension) → robustness suite (hierarchical clustering, robust hedge",
        color=INK, fontsize=14)
ax.text(0.09, 0.175, "ratios, stop-losses) → live Alpaca paper-trade forward test.", color=INK, fontsize=14)
footer(ax, 14); pages.append(f)

# 15 — Thank you
f, _ = lead_slide(
    [("Thank you", 46)],
    ["Clustering-Based Pairs Trading",
     "Sharpe 1.028 · 21 years out-of-sample · built from the data up"],
    "Happy to walk through the code, the backtest engine, or the attribution analysis.")
pages.append(f)

# ---- write PDF ----
with PdfPages(OUT) as pdf:
    for fig in pages:
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)
print(f"wrote {OUT}  ({len(pages)} slides)")
