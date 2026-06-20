ADR dislocation alpha:
When ADR creation/redemption is blocked (shelf registration maxed, regulatory cap), the arbitrage mechanism that keeps ADR ≈ underlying breaks down. TSMC ADR (TSM) can trade at 10-20% premium to the TWD-converted underlying with no clean arbitrage path. That spread is real and persistent — not mean-reverting in the traditional sense.

Why cointegration catches it:
If fungibility is blocked, the ADR and underlying decouple. Cointegration test on the pair fails — spread is non-stationary. The stationarity filter would correctly discard it. But the reason it fails is mechanical, not fundamental — once fungibility restores, the spread closes violently.

The filter you're describing:
Check ADR creation/redemption status before including any ADR pair. Sources: Citibank/BNY Mellon ADR databases, SEC shelf registration filings. If creation is suspended → mechanically exclude from universe regardless of cointegration result.

FX considerations by jurisdiction:
- JPY/USD: clean, liquid, near-zero stamp duty — manageable
- HKD/USD: 0.1% stamp duty each way on HK leg — eats into round-trip margin meaningfully
- TWD: capital controls, offshore convertibility limits — harder
- Overnight FX risk: if you hold a position overnight, the FX move between close and open hits your P&L on the foreign leg

The Sharpe claim is plausible — ADR dislocations when fungibility is blocked are structural mispricings with known resolution triggers, not the ambiguous "will it revert?" of standard pairs. Much cleaner signal. Worth a separate filter layer: cointegration pass and fungibility confirmed.