"""Build the supplementary slide deck — Phases 2.5–6.

Same visual language as build_deck.py (palette, helpers, layout).
Run:  python3 report/build_supplementary_deck.py
  →   report/QF621_supplementary_deck.pdf
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
OUT = HERE / "QF621_supplementary_deck.pdf"

# --- palette (same as main deck) ---
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
    ax.text(0.04, 0.035, "Clustering-Based Pairs Trading · QF621 — Supplementary", color=GRAY, fontsize=9)
    ax.text(0.96, 0.035, f"{n}", color=GRAY, fontsize=9, ha="right")


def header(ax, kicker, title):
    ax.text(0.055, 0.90, kicker.upper(), color=BLUE, fontsize=12.5, weight="bold")
    ax.text(0.055, 0.85, title, color=NAVY, fontsize=27, weight="bold", va="top")
    ax.add_patch(Rectangle((0.055, 0.76), 0.18, 0.006, color=BLUE, lw=0))


def bullets(ax, items, x=0.075, y=0.70, dy=0.105, width=78, fs=16, gap=0.0):
    cy = y
    for it in items:
        text = it[1] if isinstance(it, tuple) else it
        wrapped = textwrap.wrap(text, width=width)
        ax.add_patch(plt.Circle((x, cy + 0.005), 0.006, color=BLUE, transform=ax.transAxes))
        for i, line in enumerate(wrapped):
            ax.text(x + 0.022, cy - i * 0.045, line, color=INK, fontsize=fs, va="center")
        cy -= dy + 0.045 * (len(wrapped) - 1)
    return cy


def table(ax, headers, rows, col_x, y0, row_h=0.075, fs=14, highlight=None, align=None, band_w=None):
    highlight = highlight or {}
    align = align or ["left"] * len(headers)
    w = band_w if band_w is not None else col_x[-1] - col_x[0] + 0.10
    ax.add_patch(Rectangle((col_x[0] - 0.012, y0 - row_h * 0.62), w,
                           row_h, color=NAVY, lw=0))
    for j, h in enumerate(headers):
        ha = align[j]
        xx = col_x[j] + (0.085 if ha == "right" else 0)
        ax.text(xx, y0 - row_h * 0.10, h, color="white", fontsize=fs, weight="bold", ha=ha, va="center")
    y = y0 - row_h
    for ri, row in enumerate(rows):
        if ri % 2 == 0:
            ax.add_patch(Rectangle((col_x[0] - 0.012, y - row_h * 0.62), w,
                                   row_h, color=PANEL, lw=0))
        if ri in highlight:
            ax.add_patch(Rectangle((col_x[0] - 0.012, y - row_h * 0.62), w,
                                   row_h, color=highlight[ri], lw=0, alpha=0.55))
        for j, cell in enumerate(row):
            ha = align[j]
            xx = col_x[j] + (0.085 if ha == "right" else 0)
            wt = "bold" if ri in highlight else "normal"
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

# S1 — Title
f, _ = lead_slide(
    [("Supplementary Slides", 40),
     ("Phases 2.5 – 6: Extensions, Robustness, Realism & Honesty Tests", 18)],
    ["Factor-beta extension · robustness testing · transaction costs · out-of-sample",
     "Position carry-over · implementation-correctness review · bootstrap CIs · dispersion"],
    "These slides continue from the main deck (Phases 0–2)")
pages.append(f)

# S2 — Phase 2.5: Factor-Beta Extension
pg = 2
f, ax = new_fig(); header(ax, "Phase 2.5 · Original Extension", "Factor-beta clustering — a structurally different signal")
bullets(ax, [
    "Cluster stocks by shared RISK-FACTOR EXPOSURES, not return co-movement.",
    "Ridge-regress each stock's returns on 18 factors (6 FF style + 12 FF industry) → 18-d beta vector per stock.",
    "Distance = standardised Euclidean between beta vectors. Stocks with similar factor loadings cluster together.",
], y=0.70, dy=0.095, fs=15.5, width=90)
ax.add_patch(FancyBboxPatch((0.07, 0.24), 0.85, 0.16, boxstyle="round,pad=0.015,rounding_size=0.02",
             fc=PANEL, ec=BLUE, lw=1.5))
ax.text(0.09, 0.345, "Results", color=NAVY, fontsize=14, weight="bold")
ax.text(0.09, 0.295, "Factor core Sharpe 1.013 (comparable to PC 1.028). Factor + cointegration filter: 0.858.",
        color=INK, fontsize=14.5)
ax.text(0.09, 0.255, "An independent metric reaching ~1.0 is strong evidence the edge is real, not a PC-specific artefact.",
        color=INK, fontsize=14.5)
footer(ax, pg); pages.append(f)

# S3 — Phase 3: Robustness Heatmap
pg = 3
f, ax = image_slide("Phase 3 · Robustness", "What breaks when you swap components?",
                    ASSETS / "robustness_heatmap.png",
                    caption="Clustering algorithm is LOAD-BEARING: HDBSCAN/hierarchical dilute PC badly (0.485-0.623). Factor-beta is sturdier — holds 0.991 under hierarchical where PC breaks. Hedge ratio & sizing are secondary (±0.05).",
                    img_w=0.62, top=0.72)
footer(ax, pg); pages.append(f)

# S4 — Phase 3: Robustness summary
pg = 4
f, ax = new_fig(); header(ax, "Phase 3 · Robustness", "What we learned — and what it means for deployment")
ax.text(0.075, 0.71, "8-cell grid:  {PC, Factor} × {HDBSCAN, Hierarchical, RLM hedge, Z-weight}",
        color=NAVY, fontsize=15, weight="bold")
table(ax, ["", "PC", "Factor"], [
    ["Core (OPTICS, OLS, equal)", "1.028", "1.013"],
    ["HDBSCAN", "0.623", "0.738"],
    ["Hierarchical", "0.485", "0.991"],
    ["RLM hedge ratio", "1.046", "1.033"],
    ["Z-weight sizing", "1.011", "1.060"],
], col_x=[0.075, 0.62, 0.78], y0=0.64, row_h=0.072, fs=14,
   align=["left", "right", "right"])
ax.add_patch(FancyBboxPatch((0.07, 0.14), 0.85, 0.14, boxstyle="round,pad=0.015,rounding_size=0.02",
             fc=PANEL, ec=BLUE, lw=1.4))
ax.text(0.09, 0.235, "Robustness bands:  PC 0.485 – 1.046  ·  Factor 0.615 – 1.060", color=NAVY, fontsize=14, weight="bold")
ax.text(0.09, 0.175, "Factor-beta is the more robust metric in-sample (tighter band, survives the harshest perturbation).",
        color=INK, fontsize=14)
footer(ax, pg); pages.append(f)

# S5 — Phase 4a: Realism
pg = 5
f, ax = image_slide("Phase 4a · Realism", "From frictionless to deployable — the cost drag",
                    ASSETS / "realism_waterfall.png",
                    caption="Realistic frictions (actual CRSP bid/ask + 35 bps borrow + 3.5σ stop) roughly halve Sharpe to ~0.57. Passive execution (+0.21) is the #1 lever; dropping the stop (+0.11-0.14) is #2.",
                    img_w=0.56, top=0.72)
footer(ax, pg); pages.append(f)

# S6 — Phase 4a: Cost levers
pg = 6
f, ax = new_fig(); header(ax, "Phase 4a · Realism", "Which levers recover the edge?")
table(ax, ["Lever", "PC Sharpe", "Effect"], [
    ["Baseline (full realism)", "0.572", "—"],
    ["+ Passive execution", "0.782", "+0.210  (#1)"],
    ["+ Drop the 3.5σ stop", "0.782", "+0.11–0.14  (#2)"],
    ["+ Cointegration filter", "worse", "sheds more alpha than turnover saved"],
], col_x=[0.075, 0.50, 0.68], y0=0.68, row_h=0.080, fs=14.5,
   align=["left", "right", "left"],
   highlight={1: LIGHT, 2: LIGHT})
bullets(ax, [
    "The stop-loss is NET-NEGATIVE: extra round-trips cost more than the tail protection they buy.",
    "Passive execution (limit orders, half the spread) is realistic for a monthly strategy and recovers most of the gap.",
], y=0.25, dy=0.085, fs=15, width=90)
footer(ax, pg); pages.append(f)

# S7 — Phase 4b: Lookahead
pg = 7
f, ax = new_fig(); header(ax, "Phase 4b · Integrity", "Lookahead-bias audit — 6/6 PASS")
bullets(ax, [
    "Black-box test: run full period (2003–2023) and truncated (2003–cut date). Compare daily position vectors on overlap.",
    "If IDENTICAL → no lookahead. If they differ → a decision on day T used data from after day T.",
], y=0.70, dy=0.085, fs=15.5, width=95)
table(ax, ["Metric", "Cut date", "Overlap days", "Mismatches", "Result"], [
    ["PC", "2017-12-31", "3,776", "0", "PASS"],
    ["PC", "2013-12-31", "2,768", "0", "PASS"],
    ["PC", "2009-12-31", "1,762", "0", "PASS"],
    ["Factor", "2017-12-31", "3,776", "0", "PASS"],
    ["Factor", "2013-12-31", "2,768", "0", "PASS"],
    ["Factor", "2009-12-31", "1,762", "0", "PASS"],
], col_x=[0.075, 0.26, 0.44, 0.61, 0.78], y0=0.48, row_h=0.055, fs=12.5,
   align=["left", "left", "right", "right", "left"])
footer(ax, pg); pages.append(f)

# S8 — Phase 4c: OOS
pg = 8
f, ax = image_slide("Phase 4c · Out-of-Sample", "The real test — 23 months of unseen 2024–2025 data",
                    ASSETS / "oos_bar.png",
                    caption="PC generalises (0.858 ≈ IS 1.028). Factor collapses (0.117 ≈ noise) — sturdier in-sample but weaker OOS. Both weak in trending 2024, recovered in 2025 (regime dependence). Honest deployable Sharpe: ~0.5–0.8.",
                    img_w=0.50, top=0.72)
footer(ax, pg); pages.append(f)

# S9 — Phase 5: Carry-over
pg = 9
f, ax = new_fig(); header(ax, "Phase 5 · Carry-Over", "Hold positions over month-end instead of force-closing")
ax.text(0.075, 0.72, "Motivation: 88% of trades force-close at month-end at a mean loss.", color=NAVY, fontsize=15.5, weight="bold")
bullets(ax, [
    "Freeze hedge ratio (γ) at entry — never re-fit mid-trade.",
    "MAX_CARRY_MONTHS = 3, tied to the 60-day half-life ceiling.",
    "Force-close a carried pair if the cluster no longer endorses it.",
], y=0.63, dy=0.07, fs=14.5, width=92)
ax.text(0.075, 0.40, "Key finding: carry is a COST lever, not a signal lever.", color=NAVY, fontsize=15.5, weight="bold")
table(ax, ["", "PC (frictionless)", "PC (net of cost)"], [
    ["No-carry", "1.028", "0.855"],
    ["Carry", "1.019", "0.900"],
    ["Delta", "-0.009", "+0.045"],
], col_x=[0.075, 0.45, 0.68], y0=0.35, row_h=0.060, fs=14,
   align=["left", "right", "right"],
   highlight={2: LIGHT})
ax.text(0.075, 0.095, "Carry cut trades by 39% → turnover savings pay off net of costs. Moved PC from 0.572 → 0.900.",
        color=GRAY, fontsize=13)
footer(ax, pg); pages.append(f)

# S10 — Phase 5: Grid & drawdown
pg = 10
f, ax = image_slide("Phase 5 · Carry-Over", "Carry is metric-dependent — but drawdown improves everywhere",
                    ASSETS / "carry_deltas.png",
                    caption="SSD gains most (+0.097/+0.077 — worst baseline, slowest reversions). PC gains net-of-cost only (+0.045). Factor HURTS (drifting betas). But drawdown improves for ALL three metrics.",
                    img_w=0.55, top=0.72)
footer(ax, pg); pages.append(f)

# S11 — Phase 5: OOS verdict
pg = 11
f, ax = new_fig(); header(ax, "Phase 5 · Carry-Over", "OOS verdict: carry does NOT generalise")
table(ax, ["Metric", "IS carry (E)", "OOS carry", "OOS no-carry (Ph4)"], [
    ["PC", "1.019", "0.434", "0.858"],
    ["Factor", "0.929", "-0.016", "0.117"],
    ["SSD", "0.686", "0.071", "—"],
], col_x=[0.075, 0.35, 0.55, 0.73], y0=0.68, row_h=0.080, fs=15,
   align=["left", "right", "right", "right"],
   highlight={0: "#f5c6cb"})
bullets(ax, [
    "PC carry OOS (0.434) ≈ HALF the no-carry OOS (0.858). The in-sample +0.045 net gain did NOT survive.",
    "2024–2025 was strongly trending, low-dispersion — adverse for mean-reversion, worst for long holds. Carry doubled hold time (20→50 days).",
    "Verdict: do NOT ship carry as a default. Passive execution remains a separately validated lever.",
], y=0.37, dy=0.080, fs=14.5, width=95)
footer(ax, pg); pages.append(f)

# S12 — Phase 6: Title
pg = 12
f, ax = new_fig(); header(ax, "Phase 6 · Correctness", "Correctness review — six fixes, one at a time")
ax.text(0.075, 0.72, "A line-by-line code review found six flaws. None are lookahead bias (6/6 audit stands).",
        color=INK, fontsize=15.5)
table(ax, ["ID", "Fix", "What was wrong"], [
    ["D6.1", "MacKinnon p-values", "Engle-Granger: anti-conservative raw-ADF on residuals"],
    ["D6.2", "Delisting fix", "Code-map shifted one CRSP bucket; dlret not compounded"],
    ["D6.3", "Stop cooldown", "Stop-out re-entered same position next day (churn)"],
    ["D6.5", "Execution delay", "Fills at signal close (can't observe close and trade at it)"],
], col_x=[0.075, 0.17, 0.40], y0=0.62, row_h=0.072, fs=13,
   align=["left", "left", "left"], band_w=0.87)
ax.add_patch(FancyBboxPatch((0.07, 0.16), 0.85, 0.12, boxstyle="round,pad=0.015,rounding_size=0.02",
             fc=PANEL, ec=NAVY, lw=1.4))
ax.text(0.09, 0.24, "Discipline: every fix behind its own flag, defaults bit-identical, re-run one-at-a-time.",
        color=NAVY, fontsize=14, weight="bold")
ax.text(0.09, 0.19, "No flags tuned on Sharpe — these are CORRECTNESS changes. Whatever they do, we report.",
        color=INK, fontsize=14)
footer(ax, pg); pages.append(f)

# S13 — Phase 6: Ladder
pg = 13
f, ax = image_slide("Phase 6 · Correctness", "Corrections ladder — what changed?",
                    ASSETS / "corrections_ladder.png",
                    caption="Headline (pc_core) robust to delisting fix (1.028→1.007) and survives honest fills (0.882, 86% genuine). The cointegration filter was an artifact (0.752→0.299). Stop still net-negative with churn removed.",
                    img_w=0.65, top=0.72)
footer(ax, pg); pages.append(f)

# S14 — Phase 6: MacKinnon finding
pg = 14
f, ax = new_fig(); header(ax, "Phase 6 · Finding 1", "The cointegration filter was a statistical artifact")
bullets(ax, [
    "Raw ADF on estimated residuals is anti-conservative — effective threshold ~12-15%, not 5%.",
    "Correct MacKinnon test: pass rate drops 26% → 11%, Sharpe collapses 0.752 → 0.299.",
], y=0.70, dy=0.08, fs=15, width=92)
ax.text(0.075, 0.50, "Trade-level decomposition of the filtered baseline:", color=NAVY, fontsize=14.5, weight="bold")
table(ax, ["Subset", "Trades", "Mean ret (bps)", "Hit rate"], [
    ["MacKinnon-passes", "3,937", "+12.1", "53.6%"],
    ["Raw-ADF-only", "4,989", "+19.3", "53.5%"],
], col_x=[0.075, 0.38, 0.55, 0.72], y0=0.44, row_h=0.068, fs=14,
   align=["left", "right", "right", "right"])
ax.add_patch(FancyBboxPatch((0.07, 0.11), 0.85, 0.12, boxstyle="round,pad=0.015,rounding_size=0.02",
             fc=PANEL, ec=RED, lw=1.5))
ax.text(0.09, 0.195, "Stronger cointegration evidence selects WORSE trades. The filter's only effect is shrinking N.",
        color=RED, fontsize=14, weight="bold")
ax.text(0.09, 0.14, "Dose-response: no filter 1.028 → weak filter 0.752 → correct filter 0.299. Edge = short-horizon reversion.",
        color=INK, fontsize=14)
footer(ax, pg); pages.append(f)

# S15 — Phase 6: Bootstrap CIs
pg = 15
f, ax = image_slide("Phase 6 · Error Bars", "Bootstrap 95% CIs — what the numbers actually mean",
                    ASSETS / "bootstrap_forest.png",
                    caption="OOS PC (0.858) consistent with IS (P(OOS>=IS)=33%) — generalisation strengthened. Factor CI = [-0.52, +1.12] — inconclusive. Execution delay (Δ=0.145, CI [0.02, 0.31]) significant but edge survives.",
                    img_w=0.60, top=0.72)
footer(ax, pg); pages.append(f)

# S16 — Phase 6: Dispersion
pg = 16
f, ax = image_slide("Phase 6 · Regime", "Dispersion drives returns — but the timing signal is weak",
                    ASSETS / "dispersion_quartiles.png",
                    caption="Contemporaneous: cleanly monotone (0.73→1.56) — the strategy eats dislocations. Prior-month: non-monotone — only Q4 is reliable. Correct advice: SCALE UP in high dispersion, don't gate. GFC (10% of months) = 28% of total return.",
                    img_w=0.60, top=0.72)
footer(ax, pg); pages.append(f)

# S17 — Updated scoreboard
pg = 17
f, ax = new_fig(); header(ax, "Updated Picture", "The honest, fully-corrected scoreboard")
table(ax, ["Stage", "PC Sharpe", "Notes"], [
    ["In-sample, frictionless (signal-close fill)", "1.028", "paper-match headline"],
    ["In-sample, frictionless (honest fill, +1d)", "0.882", "86% genuine signal"],
    ["Net of costs (marketable)", "0.572", "realistic worst case"],
    ["Net + passive + no stop", "~0.78", "best defensible IS operating point"],
    ["Net + passive + carry", "0.900", "IS only — does NOT survive OOS"],
    ["Lookahead audit", "PASS 6/6", "temporally honest"],
    ["Out-of-sample 2024–25 (no carry)", "0.858", "PC generalises"],
    ["Out-of-sample factor-beta", "0.117", "CI includes ~1.0 — inconclusive"],
], col_x=[0.075, 0.56, 0.72], y0=0.68, row_h=0.060, fs=13,
   align=["left", "right", "left"],
   highlight={0: LIGHT, 6: LIGHT})
ax.text(0.075, 0.095, "Honest deployable Sharpe: ~0.5–0.8, regime-dependent — strongest in turbulent, high-dispersion markets.",
        color=NAVY, fontsize=14, weight="bold")
footer(ax, pg); pages.append(f)

# S18 — Conclusions
pg = 18
f, ax = new_fig(); header(ax, "Conclusions", "What Phases 2.5–6 add to the story")
bullets(ax, [
    "Factor-beta (Phase 2.5) independently corroborates ~1.0 Sharpe — the edge is real, not a PC artefact.",
    "Clustering algorithm is load-bearing (Phase 3); OPTICS conservatism is a feature, not a tuning choice.",
    "Costs roughly halve the Sharpe; passive execution & no stop recover most of it (Phase 4).",
    "PC generalises OOS (0.858); factor does not but 23 months is inconclusive (Phases 4 + 6).",
    "Carry-over works in-sample (+0.045 net) but fails OOS — do not ship as default (Phase 5).",
    "Cointegration filter was a statistical artifact of using the wrong p-value distribution (Phase 6).",
], y=0.70, dy=0.075, fs=13.5, width=100)
ax.text(0.075, 0.095, "~86% of the frictionless headline is genuine signal. The edge is real, dislocation-driven, and regime-dependent.",
        color=NAVY, fontsize=14, weight="bold")
footer(ax, pg); pages.append(f)

# S19 — Thank you
f, _ = lead_slide(
    [("Thank you", 46)],
    ["Supplementary slides: Phases 2.5 – 6",
     "Factor-beta · robustness · realism · OOS · carry-over · correctness · error bars · dispersion"],
    "Happy to walk through any phase, the code, or the specific findings.")
pages.append(f)

# ---- write PDF ----
with PdfPages(OUT) as pdf:
    for fig in pages:
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)
print(f"wrote {OUT}  ({len(pages)} slides)")
