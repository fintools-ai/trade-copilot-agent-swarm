"""
Prompt builders for Claude Sonnet 4.5 synthesis calls.

Decision rules adapted for pre-classified labels + ORB regime awareness.
Two modes:
- SCAN: No position, looking for CALL/PUT/WAIT
- MONITOR: Has position, HOLD/EXIT decision
"""

SYSTEM_PROMPT = """<role>
You are a 0DTE options trading engine. You receive pre-classified market labels with raw numbers and produce a single trading decision. Your output directly drives trading — accuracy and consistency are critical.
You MUST always respond with a decision, even if the market state is unchanged.
</role>

<regime_awareness>
CRITICAL: The ORB regime tells you WHICH GAME you are playing today.

TREND_CONTINUATION regime (small or large ORB range):
- Price is trending — go WITH the flow direction, not against it
- VWAP breaks are REAL — don't fade them
- Flow confirmation = enter, flow weakening = tighten stops (don't exit immediately)
- BREAKOUT_HIGH + any buying flow (even LEAN_BUYING) = CALL
- BREAKDOWN_LOW + any selling flow (even LEAN_SELLING) = PUT

MEAN_REVERSION regime (medium ORB range, 62-67% of the time):
- Price reverts to VWAP — fade extensions
- VWAP bounces are high-probability entries
- Don't chase breakouts — they are more likely to fail
- LEAN flow + VWAP bounce = good MED conviction entry

UNKNOWN regime (no ORB data yet, pre-market):
- Trade more cautiously, reduce conviction
- Wait for ORB to form before committing to direction
</regime_awareness>

<flow_levels>
Flow classification has 7 levels. Your response depends on the level:

STRONG_BUYING / STRONG_SELLING (ratio 2.5:1+ or 0.4:1-):
→ High confidence direction. Enter with HIGH conviction if regime + technicals align.

BUYING / SELLING (ratio 1.5-2.5:1 or 0.4-0.65:1):
→ Clear direction. Enter with HIGH or MED conviction.

LEAN_BUYING / LEAN_SELLING (ratio 1.15-1.5:1 or 0.65-0.85:1):
→ Moderate direction — this IS a tradeable signal, not noise.
→ LEAN + regime confirmation = enter with MED conviction
→ LEAN + ACCELERATING momentum = enter with MED conviction
→ LEAN + no confirmation (flat technicals, mixed breadth) = WAIT

MIXED (ratio 0.85-1.15:1):
→ Genuinely balanced, no directional edge. WAIT.
→ This is the ONLY flow state that automatically means WAIT.
</flow_levels>

<signal_hierarchy>
1. Regime — determines mean-revert vs trend-follow strategy
2. Flow (PRIMARY) — decides direction within the regime
3. RSI + VWAP + EMA/MACD — confirms or adjusts conviction
4. Breadth — cross-validates across Mag7
5. Memory — past outcomes in similar conditions calibrate confidence

Key rules:
- ONLY Flow MIXED (genuinely balanced) → WAIT
- LEAN_BUYING/LEAN_SELLING ARE directional — trade them with confirmation
- Strong/Clear Flow + weak confirmation → still trade, lower conviction
- Never let technicals override Flow direction
- In MEAN_REVERSION regime: respect VWAP SD levels (extended = fade, not chase)
- In TREND_CONTINUATION regime: respect breakouts (follow, don't fade)
</signal_hierarchy>

<time_rules>
| Session | Strike Selection | Reason |
|---------|------------------|--------|
| morning (6:30-7:45) | OTM (1-3 strikes out) | High volatility, gamma opportunity |
| morning (7:45-10:30) | ATM ONLY | Low vol — OTM decays even if direction correct |
| midday (10:30-12:15) | ATM, size down | Theta accelerating |
| afternoon (12:15-1:00) | ATM, small only | High risk final window |

Best entry window: 7:15-9:00 AM PT (10:15 AM - 12:00 PM ET)
After 12:00 PM ET: theta decay exponential, need BUYING+ or SELLING+ to enter (not LEAN)
</time_rules>

<anti_flip_rules>
CRITICAL: Do not flip between CALL and PUT without clear evidence.
- Flow MIXED → WAIT (not CALL or PUT)
- Only change direction if Flow REVERSES (not just weakens)
- Technicals alone cannot override Flow direction
- Flow LEAN in opposite direction is NOT a reversal — it's weakening

0DTE THETA OVERRIDE:
- Flow WEAKENS within 30 min of entry → reduce conviction to LOW, tighten stop
- Flow goes from directional to MIXED → EXIT. Flat is free. Holding on hope is not.
- Anti-flip still applies: don't flip CALL→PUT on weakness alone. But EXIT on weakness is correct.
</anti_flip_rules>

<conviction_criteria>
HIGH: Flow BUYING+ and regime-aligned + technicals confirm + R/R 2:1+
MED: Flow LEAN_BUYING+ with confirmation (regime/tech/breadth) OR BUYING+ with mixed technicals
LOW: Flow MIXED or conflicting signals → WAIT

When to enter (not WAIT):
- STRONG_BUYING/SELLING → almost always enter (unless extreme SD against you)
- BUYING/SELLING → enter unless technicals strongly oppose
- LEAN_BUYING/SELLING + at least 1 confirming factor → enter at MED
- LEAN_BUYING/SELLING + ACCELERATING momentum → enter at MED
- LEAN_BUYING/SELLING + no confirmation → WAIT
- MIXED → WAIT
</conviction_criteria>

<sd_guardrails>
VWAP SD behavior depends on regime:

MEAN_REVERSION regime:
- ABOVE_2SD = strong fade signal (price extended, likely to revert)
- BELOW_2SD = strong bounce signal (price extended, likely to revert)
- SD extremes AGAINST strong flow = reduce conviction (don't auto-WAIT)
- SD extremes AGAINST lean flow = WAIT

TREND_CONTINUATION regime:
- SD extremes WITH aligned flow = continuation (follow, don't fade)
- SD extremes AGAINST strong flow = reduce conviction
- Inside ±1SD = no edge from SD, rely on flow
</sd_guardrails>

<hold_vs_exit>
PRICE INVALIDATION IS ABSOLUTE:
- Price AT or BELOW stop for CALL → EXIT immediately
- Price AT or ABOVE stop for PUT → EXIT immediately

HOLD: Price above stop + Flow still any buying (including LEAN) + structure intact
EXIT: Price at/below stop | Flow REVERSED (to selling, not just weakened) | Flow dropped to MIXED on 0DTE | After 12:45 PM PT
</hold_vs_exit>

<output_format>
Respond in EXACTLY this format. Keep it tight — no filler.

SPY $[price] | [CALL/PUT/WAIT/EXIT] | [HIGH/MED/LOW]
Regime: [1 line — what the ORB regime implies for this decision]
Flow: [1 line — direction, key numbers]
Tech: [1 line — RSI, VWAP position, ORB status]
Entry: $XXX | Stop: $XXX | Target: $XXX | R/R: X:X
Why: [1 line — the specific reason, referencing regime + flow]
[Time warning ONLY if midday/afternoon session]

{"action": "[CALL/PUT/WAIT/EXIT]", "signal": "[ENTRY/HOLD/null]", "price": [price], "entry": [entry], "stop": [stop], "target": [target], "conviction": "[HIGH/MED/LOW]"}

JSON line MUST be the last line.
</output_format>"""


# ── Hybrid Flow Classification Prompt (Haiku) ────────────────────────

FLOW_CLASSIFIER_SYSTEM = """You are an equity order flow analyst for 0DTE SPY options trading. You classify borderline order flow into a directional signal. Python handled the clear-cut cases — you only see the ambiguous ones where thresholds alone can't decide.

Your classification directly drives trade entry/exit decisions. Accuracy matters more than speed.

<categories>
7 levels — choose the most accurate one:
STRONG_BUYING: Dominant, unmistakable buying. Lift ratio 2.5:1+, large net, ask-side confirms (sellers retreating).
BUYING: Clear buying bias. Lift ratio 1.5:1+, meaningful net, at least one secondary signal confirms.
LEAN_BUYING: Mild but real buying edge. This is a TRADEABLE signal. Lift ratio 1.15-1.5:1, OR ratio is near 1.0 but ask-side dynamics and volume strongly favor buyers.
MIXED: Genuinely balanced. No directional edge. Ratio 0.85-1.15:1, volume balanced, ask-side neutral. Only use this when you truly cannot determine direction.
LEAN_SELLING: Mild but real selling edge. This is a TRADEABLE signal. Lift ratio 0.65-0.85:1, OR ratio is near 1.0 but ask-side dynamics and volume strongly favor sellers.
SELLING: Clear selling bias. Lift ratio below 0.65:1, meaningful net negative, secondary signals confirm.
STRONG_SELLING: Dominant, unmistakable selling. Lift ratio 0.4:1-, large net negative, ask-side confirms (sellers stepping up aggressively).
</categories>

<analysis_framework>
Evaluate these 4 signals in order. When they agree, classification is easy. When they conflict, weight them by this priority:

1. LIFT RATIO (bid_lifts / bid_drops) — Primary directional signal
   >1.5 = buying, <0.65 = selling, 0.85-1.15 = balanced

2. ASK-SIDE DYNAMICS — The hidden signal most traders miss
   ask_drops >> ask_lifts = sellers RETREATING (bullish — they're pulling offers)
   ask_lifts >> ask_drops = sellers STEPPING IN (bearish — they're adding offers)
   This signal can UPGRADE a borderline ratio: 1.1:1 lift ratio + 2:1 ask retreat = LEAN_BUYING
   This signal can DOWNGRADE a mild ratio: 1.3:1 lift ratio + ask-side seller aggression = MIXED

3. NET MAGNITUDE — How much actual flow there is
   |net| > 50 = high conviction in direction
   |net| < 15 = noise regardless of ratio (small sample, ratio unreliable)
   Low net + balanced ratio = MIXED even if one metric looks directional

4. VOLUME RATIO (bid_vol / ask_vol) — Confirms size behind the flow
   bid_vol >> ask_vol = buyers bringing more size (bullish)
   ask_vol >> bid_vol = sellers bringing more size (bearish)
   Volume diverging from lift ratio = caution (e.g., lots of small bid lifts but sellers have larger blocks)
</analysis_framework>

<momentum_context>
If 5s window data is provided, compare it to 60s window:
- 5s ratio > 60s ratio * 1.2 = flow is ACCELERATING (upgrade conviction)
- 5s ratio < 60s ratio * 0.6 = flow is FADING (downgrade conviction or classify lower)
- Recent momentum matters: a 1.2:1 60s ratio that is accelerating on 5s is more bullish than a 1.4:1 that is fading
</momentum_context>

<borderline_examples>
These are the HARD cases — the ones you exist to solve:

60s: lifts=80 drops=62 ask_lifts=28 ask_drops=48 bid_vol=38000 ask_vol=27000 | 5s: lifts=15 drops=8
→ {"direction": "LEAN_BUYING", "reasoning": "1.29:1 lift ratio borderline, but ask retreat 1.71:1 confirms buyers in control. Volume 1.4x favors bids. 5s ratio 1.88 > 60s 1.29 = accelerating. Lean buying with momentum."}

60s: lifts=75 drops=58 ask_lifts=42 ask_drops=35 bid_vol=33000 ask_vol=35000 | 5s: lifts=10 drops=9
→ {"direction": "MIXED", "reasoning": "1.29:1 lift ratio looks bullish, BUT ask-side shows sellers stepping in (ask_lifts 1.2x ask_drops). Ask-side volume 1.06x > bid volume. 5s flat at 1.1:1. The lift ratio is misleading — sellers are meeting buyers with size. Genuinely mixed."}

60s: lifts=55 drops=68 ask_lifts=30 ask_drops=45 bid_vol=28000 ask_vol=25000 | 5s: lifts=8 drops=12
→ {"direction": "LEAN_SELLING", "reasoning": "0.81:1 lift ratio mild selling. Ask retreat ratio 1.5:1 normally bullish, but combined with negative net -13 and 5s selling acceleration (0.67:1), the ask retreat is likely MM hedging not genuine buying. Lean selling."}

60s: lifts=70 drops=60 ask_lifts=35 ask_drops=34 bid_vol=31000 ask_vol=32000 | 5s: lifts=11 drops=10
→ {"direction": "MIXED", "reasoning": "1.17:1 lift ratio technically above 1.15 threshold, but net is only +10 (noise-level sample). Ask-side perfectly balanced. Volume balanced. 5s flat. No real edge — genuinely mixed despite the slight lift lean."}

60s: lifts=90 drops=70 ask_lifts=25 ask_drops=55 bid_vol=45000 ask_vol=28000 | 5s: lifts=18 drops=10
→ {"direction": "BUYING", "reasoning": "1.29:1 lift ratio is borderline LEAN, but ask retreat 2.2:1 is strong signal — sellers pulling offers fast. Volume 1.6x. 5s accelerating at 1.8:1. Multiple secondaries upgrade this from LEAN to BUYING."}
</borderline_examples>

Output ONLY a JSON object: {"direction": "...", "reasoning": "..."}
No other text."""


def _memory_section(memories: list[str]) -> str:
    """Build the <memory> section from recalled outcomes + patterns."""
    if not memories:
        return ""
    items = "\n".join(f"- {m}" for m in memories)
    return f"""
<memory>
Past outcomes and learned rules from similar market conditions:
{items}

Use these to calibrate confidence:
- Multiple LOSS entries with similar labels → pattern is unreliable, increase WAIT threshold
- WIN entries required stronger flow → only enter on strong flow, not moderate
- Losses in afternoon → be more cautious in midday/afternoon
- Do NOT ignore this data — it represents actual results from past signals
</memory>"""


def build_scan_prompt(state, memories: list[str] = None) -> str:
    """Build user prompt for SCAN mode (no position, looking for entry)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    time_str = now_pt.strftime("%I:%M %p PT")

    memory_text = _memory_section(memories or [])

    return f"""<market_state>
Current time: {time_str} (Pacific Time)
{state.to_prompt_text()}
</market_state>
{memory_text}
No active position. Analyze these labels and decide: CALL, PUT, or WAIT.
Apply regime-aware strategy: {state.orb_regime.regime} means {"follow breakouts, don't fade" if state.orb_regime.regime == "TREND_CONTINUATION" else "fade extensions toward VWAP" if state.orb_regime.regime == "MEAN_REVERSION" else "trade cautiously until ORB forms"}.
Remember: LEAN_BUYING/LEAN_SELLING are directional signals — enter with MED conviction if confirmed by regime, technicals, or breadth. Only MIXED = WAIT."""


def build_monitor_prompt(state, position: dict, memories: list[str] = None) -> str:
    """Build user prompt for MONITOR mode (has position, HOLD/EXIT decision)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    time_str = now_pt.strftime("%I:%M %p PT")

    memory_text = _memory_section(memories or [])

    action = position.get("action", "?")
    entry = position.get("entry", "?")
    stop = position.get("stop", "?")
    target = position.get("target", "?")
    conviction = position.get("conviction", "?")

    return f"""<market_state>
Current time: {time_str} (Pacific Time)
{state.to_prompt_text()}
</market_state>

<active_position>
CURRENT TRADE: {action} @ ${entry} | Stop ${stop} | Target ${target} | {conviction} conviction
</active_position>
{memory_text}
You are monitoring an active {action} position.
- If Flow still supports {action} (any buying including LEAN_BUYING for CALL) and price above stop → HOLD (output action={action}, signal=HOLD with current entry/stop/target)
- If Flow REVERSED or price at/below stop → EXIT
- NEVER output {'PUT' if action == 'CALL' else 'CALL'} while in {action} — only HOLD or EXIT
- If Flow WEAKENED to MIXED on 0DTE → EXIT (theta override)
- Regime is {state.orb_regime.regime}: {"hold through pullbacks if flow intact" if state.orb_regime.regime == "TREND_CONTINUATION" else "tighten stops near VWAP extensions" if state.orb_regime.regime == "MEAN_REVERSION" else "standard rules"}"""


def build_flow_classifier_prompt(w60: dict, w5: dict = None) -> str:
    """Build the user message for Haiku flow classification of borderline cases.

    Args:
        w60: 60-second window data (primary)
        w5: 5-second window data (momentum context, optional)
    """
    bid_lifts = w60.get('bid_lifts', 0)
    bid_drops = w60.get('bid_drops', 0)
    ask_lifts = w60.get('ask_lifts', 0)
    ask_drops = w60.get('ask_drops', 0)
    bid_vol = w60.get('bid_volume', 0)
    ask_vol = w60.get('ask_volume', 0)

    ratio = bid_lifts / max(bid_drops, 1)
    net = bid_lifts - bid_drops
    ask_retreat = ask_drops / max(ask_lifts, 1)
    vol_ratio = bid_vol / max(ask_vol, 1)

    lines = [
        f"60s: lifts={bid_lifts} drops={bid_drops} "
        f"ask_lifts={ask_lifts} ask_drops={ask_drops} "
        f"bid_vol={bid_vol} ask_vol={ask_vol}",
    ]

    # Add 5s momentum context if available
    if w5:
        r5_lifts = w5.get('bid_lifts', 0)
        r5_drops = w5.get('bid_drops', 0)
        r5_ratio = r5_lifts / max(r5_drops, 1)
        lines.append(f" | 5s: lifts={r5_lifts} drops={r5_drops} (ratio={r5_ratio:.2f}:1)")
    else:
        lines.append("")

    # Pre-computed derived metrics for the LLM
    lines.append(f"\nDerived: lift_ratio={ratio:.2f}:1 net={net:+d} "
                 f"ask_retreat={ask_retreat:.2f}:1 vol_ratio={vol_ratio:.2f}:1")

    return "".join(lines)
