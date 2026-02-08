# How We Give LLMs Long-Term Memory for Trading

## The Problem

Every time our OI analysis runs, the LLM starts from zero. It sees today's open interest data — 45,000 contracts stacked at the $580 call wall — and makes a judgment call. But it has no idea if that wall appeared today or has been building for a week.

A human trader would think: "That put wall at $570 has been growing for 3 days straight. That's institutional accumulation, not a one-off hedge."

Without memory, the LLM can't do this. It treats every day as day one.

## The Memory Loop

The core idea: every analysis both **reads from** and **writes to** memory, creating a compounding feedback loop. Each cycle makes the next one smarter.

```
                         ┌──────────────────────────────────┐
                         │         MEMORY STORE              │
                         │                                  │
                         │  Semantic Facts    Episodes       │
                         │  (compressed)      (full context) │
                         │                                  │
                         └──────┬───────────────────▲───────┘
                                │                   │
                         recall()              store_episode()
                         5 most relevant       after each
                         facts per ticker      analysis completes
                                │                   │
       ┌────────────────────────▼───────────────────┴────────────────────────┐
       │                                                                     │
       │                        DAILY ANALYSIS LOOP                          │
       │                                                                     │
       │   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐ │
       │   │  Collect   │    │  Compute  │    │    LLM    │    │   Store   │ │
       │   │  raw OI    │───▶│  deltas   │───▶│  analyzes │───▶│  results  │ │
       │   │  + market  │    │  vs prev  │    │  raw data │    │  + store  │ │
       │   │  data      │    │  day      │    │  + delta  │    │  episode  │ │
       │   │           │    │           │    │  + memory  │    │  to memory│ │
       │   └───────────┘    └───────────┘    └───────────┘    └───────────┘ │
       │                                                                     │
       └─────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ runs daily
                                        ▼

       Day 1:  LLM sees raw data only              → stores first episode
       Day 2:  LLM sees raw data + 1 day of facts  → stores second episode
       Day 5:  LLM sees raw data + 4 days of facts → stores fifth episode
       Day 20: LLM sees raw data + rich history     → stores with full context
                                          │
                                          ▼
                                 THE SYSTEM COMPOUNDS
                                 More days = richer facts
                                 Richer facts = better analysis
                                 Better analysis = better episodes stored
                                 Better episodes = richer facts extracted
```

**What compounds over time:**

| Day | What the LLM knows | Example insight |
|-----|---------------------|-----------------|
| 1 | Today's raw data only | "There's a call wall at $580" |
| 3 | 2 days of pattern history | "That call wall has been growing — sustained accumulation" |
| 7 | Trend direction + past accuracy | "Bullish streak since Monday. Last 2 call targets hit." |
| 14 | Regime shifts + structural levels | "$570 is a structural floor — tested 4x and held. $590 resistance broke on day 10." |
| 30 | Full institutional behavior model | "Institutions rotate from calls to puts when VIX > 20. Current setup matches the Feb 8 reversal pattern." |

The memory service auto-expires events after 30 days. Older than that, OI patterns are stale — institutions roll positions, market regimes shift. This keeps the facts fresh and relevant.

**The key property**: the LLM doesn't just accumulate data — it accumulates *understanding*. Semantic memory extracts facts like "the $570 put wall has held 4 times" from multiple raw episodes. The LLM never sees the raw history — just the compressed wisdom.

## The Solution: Two Memory Layers

We split memory into two layers based on access pattern.

### Layer 1: Semantic Memory (Long-Term, Cross-Session)

After each daily analysis, we store what the LLM found as an **episode**:

```
OI Analysis for SPY on 2025-02-06:
Direction: CALL | Confidence: 78%
Confluence: aligned (short-term bullish, long-term bullish)
Thesis: Institutions building call positions above $580 with sustained
        accumulation across 30 and 90 DTE timeframes
Key Strikes:
  - $580 (call_wall): 45,000 OI, 5d change: +12,000
  - $570 (put_wall): 38,000 OI, 5d change: +8,000
Trade: Buy Call entry $582.50 stop $580.00 target $585.00 R/R 2.5:1
Market Regime: low_vol | Fear: low
Risks: VIX expansion could invalidate call wall
```

The memory service ingests this and automatically extracts **semantic facts** — compressed, searchable knowledge. Over multiple days of episodes, the system builds up a factual understanding per ticker.

Before the next analysis run, we **recall** facts for the ticker being analyzed:

```python
facts = recall("SPY")
# Returns:
# [
#   "SPY has shown consistent bullish OI bias with call accumulation above $580 for 4 consecutive days",
#   "Put wall at $570 held as reliable support across 3 recent analyses with growing OI",
#   "Short-term and long-term timeframes have been aligned bullish since Feb 3",
#   "Previous $585 target was reached on 2 of last 3 bullish calls",
#   "Confidence averaged 72% during current bullish streak"
# ]
```

These get injected into the LLM prompt:

```
# HISTORICAL CONTEXT (from past analyses)
These are facts extracted from your previous analyses of SPY.
Use them to identify TRENDS — is today confirming or contradicting the pattern?
Do NOT blindly repeat past conclusions. Compare past vs present data.

- SPY has shown consistent bullish OI bias with call accumulation above $580 for 4 consecutive days
- Put wall at $570 held as reliable support across 3 recent analyses with growing OI
- Short-term and long-term timeframes have been aligned bullish since Feb 3
- Previous $585 target was reached on 2 of last 3 bullish calls
- Confidence averaged 72% during current bullish streak
```

Now the LLM can reason: "The $580 call wall has 45K OI today vs 33K yesterday, and memory says it's been growing for 4 days. This is sustained institutional accumulation, not noise. High conviction."

**Why not just dump 5 days of raw OI into the prompt?** Token cost. Raw OI JSON for one ticker across 4 DTEs is ~10-20K tokens per day. Five days = 50-100K tokens per ticker. Semantic memory compresses that into ~5 facts, maybe 200 tokens. Same information density, 99% fewer tokens.

The anti-anchoring instruction is critical. Without "Do NOT blindly repeat past conclusions", the LLM just rubber-stamps its previous analysis. The instruction forces genuine comparison: past facts vs today's raw data.

### Layer 2: Redis Cache (Fast, Same-Day)

Redis handles everything that needs to be fast and ephemeral:

```
oi:SPY:30DTE:2025-02-06     ← Today's raw OI (tomorrow's delta calc needs this)
oi:delta:SPY:30DTE:2025-02-06  ← Day-over-day OI changes
oi:analysis:SPY:2025-02-06     ← Today's LLM result
oi:results                     ← Full results for UI
oi:status                      ← Pipeline progress (for live progress bar)
oi:events (pub/sub)            ← SSE stream to browser
```

7-day TTL. Redis doesn't learn anything — it's just fast storage for today's operational data.

### How They Work Together

```
Day 1 (cold start):
  Raw OI data → LLM analyzes (no history) → Store episode to memory
                                             Store raw data to Redis

Day 3:
  Raw OI data → Delta vs Redis Day 2
              → Recall 2 days of facts from memory
              → LLM sees: today's data + delta + "put wall at $570 held for 2 days"
              → Store episode → Store to Redis

Day 7:
  Raw OI data → Delta vs Redis Day 6
              → Recall 6 days of accumulated facts
              → LLM sees: today's data + delta + "bullish streak since Day 1,
                 $585 target hit 3x, put wall at $570 is institutional floor"
              → Store episode → Store to Redis
```

The system genuinely gets smarter each day. By day 7, it knows which levels are structural (institutions keep defending them) vs noise (appeared once, disappeared). It knows its own track record — which setups worked and which didn't.

## Memory Design

Two strategies run inside the memory store:

```
┌───────────────────────────────────────────────┐
│              MEMORY STORE                      │
│                                                │
│  ┌─────────────────────┐  ┌────────────────┐  │
│  │   Semantic Memory    │  │    Episodic    │  │
│  │                     │  │    Memory      │  │
│  │  Auto-extracted     │  │               │  │
│  │  facts per ticker:  │  │  Full daily   │  │
│  │                     │  │  episodes     │  │
│  │  "NVDA call wall    │  │  with all     │  │
│  │   at $140 grew      │  │  context:     │  │
│  │   +15K over 3 days" │  │  OI levels,   │  │
│  │                     │  │  thesis,      │  │
│  │  "META put/call     │  │  trades,      │  │
│  │   ratio shifted     │  │  market       │  │
│  │   bearish on Feb 4" │  │  regime       │  │
│  │                     │  │               │  │
│  └─────────────────────┘  └────────────────┘  │
│                                                │
│  Namespaces: /facts/{ticker}/                  │
│              /episodes/{ticker}/                │
│  Expiry: 30 days                               │
└───────────────────────────────────────────────┘
```

- **Semantic** = distilled facts. Searchable via embedding similarity. What the LLM reads before each analysis.
- **Episodic** = full context snapshots. The raw material from which facts are extracted. Useful for debugging or replaying past reasoning.

Events expire after 30 days. OI patterns older than a month are stale — institutional positions roll, market regimes shift.

## Concrete Example: SPY Over 5 Days

### Day 1 — Monday

**Raw data**: Call OI at $580 = 33,000. Put OI at $570 = 30,000.
**Memory**: Empty (first run).
**LLM output**: "Moderate bullish bias, $580 call wall forming. Confidence: 62%. Confluence: aligned."

Episode stored. Semantic memory extracts: *"SPY showing early bullish positioning with call accumulation at $580"*

### Day 2 — Tuesday

**Raw data**: Call OI at $580 = 38,000 (+5K). Put OI at $570 = 34,000 (+4K).
**Recall returns**: `["SPY showing early bullish positioning with call accumulation at $580"]`
**LLM output**: "Both call and put walls growing. Institutions adding on both sides — could be strangle positioning. Confidence: 55%. Confluence: divergent."

Memory updates. New fact: *"SPY OI growing on both call and put sides, possible strangle/straddle positioning"*

### Day 3 — Wednesday

**Raw data**: Call OI at $580 = 45,000 (+7K). Put OI at $570 = 35,000 (+1K).
**Recall returns**:
```
- SPY showing early bullish positioning with call accumulation at $580
- SPY OI growing on both call and put sides, possible strangle/straddle positioning
```
**LLM output**: "Call side accelerating (+7K) while put side flat (+1K). Resolving from strangle to directional bullish. The $580 wall has grown from 33K to 45K in 3 days — this is sustained accumulation. Confidence: 74%. Confluence: aligned."

The LLM connected the dots. Without memory, day 3 would look identical to day 1 — "there's a call wall at $580." With memory, the LLM knows this wall has been *building* and the put side stalled, so the thesis shifts from "maybe strangle" to "directional bullish."

### Day 4 — Thursday

**Raw data**: Call OI at $580 = 44,000 (-1K). Put OI at $575 = 28,000 (new level).
**Recall returns**:
```
- SPY call wall at $580 grew from 33K to 45K over 3 days (sustained accumulation)
- Put side growth stalled on Day 3, resolving strangle into directional bullish
- Short and long-term timeframes aligned bullish since Day 1
```
**LLM output**: "Call wall at $580 lost 1K OI — first decline after 3 days of growth. New put accumulation at $575 (closer to price). Institutions may be taking profits on calls while opening downside protection. Confidence: 48%. Caution warranted."

Memory prevented a false bullish signal. Without history, the LLM would see 44K at $580 and think "big call wall = bullish." With memory, it knows 44K is actually *down from 45K* after a sustained build — the trend just broke.

### Day 5 — Friday

**Raw data**: Call OI at $580 = 40,000 (-4K). Put OI at $575 = 42,000 (+14K).
**Recall returns**:
```
- SPY call wall at $580 peaked at 45K on Day 3, now declining (profit-taking)
- New put accumulation at $575 appeared Day 4 (institutional downside protection)
- Previous bullish thesis held for 3 days before reversing
- Confidence dropped from 74% to 48% as call wall weakened
```
**LLM output**: "Regime change. Call wall at $580 unwinding (-5K in 2 days). Put wall at $575 exploded to 42K. Institutions flipped from directional bullish to downside protection. Direction: PUT. Confidence: 71%. The speed of put accumulation ($575 from 0 to 42K in 2 days) signals urgency."

Five days of memory turned a simple "there are puts at $575" into a nuanced narrative: institutions built calls for 3 days, took profits, then aggressively rotated into puts. That story is invisible without memory.

## How This Feeds Into 0DTE Trading

The OI memory isn't just for daily analysis — it feeds the real-time 0DTE trading agent through two pathways:

**1. Daily levels via Redis**
After OI analysis, critical levels get written to a fast-access key. The 0DTE coordinator reads these for stop/target placement during the trading day.

**2. Pattern context via memory recall**
The 0DTE coordinator can call `recall("SPY")` to understand the bigger picture: are institutions positioned bullish or bearish? Is the current regime stable or transitioning? This gives a fast 8-12 second trading loop deep institutional context without running the full OI pipeline.

---

## OI Analysis System

### What It Does

Analyzes open interest patterns across 18 tickers and 4 expiration timeframes (30/50/60/90 DTE) to identify institutional positioning and generate directional trade ideas with entry, stop, target, and risk/reward.

OI is settled end-of-day — it represents where institutions have committed capital. Unlike price action (noisy, full of fake moves), OI shows you where the real money is.

### The Pipeline

```
Phase 1          Phase 2          Phase 3          Phase 4          Phase 5
Collection  ───▶ Market      ───▶ Delta       ───▶ LLM         ───▶ Clustering
                 Context          Calculation       Analysis

18 tickers       VIX regime       vs previous       1 call per       Bullish /
x 4 DTEs         fear level       day's data        ticker with      Bearish /
via MCP          P/C ratio        store to Redis    ALL 4 DTEs       Unclear
                                                    + memory
                                                    recall           Top 5 picks
```

Progress streams to the UI in real-time via SSE.

### Why One LLM Call Per Ticker

Previous version: **88 separate LLM calls** (18 tickers x 4 DTEs + extras). Each call only saw one DTE in isolation — the LLM analyzing 30 DTE had no idea what 90 DTE looked like.

Current version: **1 call per ticker** with all 4 DTEs visible:

```
### 30 DTE — Short-term / Gamma
{raw OI data, delta changes, technicals}

### 50 DTE — Medium-term / Swing
{raw OI data, delta changes, technicals}

### 60 DTE — Medium-term / Swing
{raw OI data, delta changes, technicals}

### 90 DTE — Long-term / Institutional
{raw OI data, delta changes, technicals}

### VIX Market Context
{regime, fear level, P/C ratio}

### Historical Context (from memory)
- [recalled facts from past analyses]
```

Now the LLM reasons about **term structure confluence**:

| Timeframe | DTE | Reveals |
|-----------|-----|---------|
| Short-term | 30 | Momentum / gamma. MM delta hedging noise lives here. |
| Medium-term | 50-60 | Swing setups. More deliberate, less noise. |
| Long-term | 90 | Institutional hedging / strategic positioning. |

When short and long term agree = **aligned** = high conviction.
When they disagree = **divergent** = lower conviction, but informative (institutions hedging against retail momentum?).

### What the LLM Returns

~15 fields. Down from 60+ in the previous version. Less output = less hallucination.

```json
{
  "direction": "CALL",
  "confidence": 78,
  "thesis": "Institutions accumulating calls above $580 across all timeframes.
             Put wall at $570 has held for 4 days — structural support.
             Short and long-term aligned bullish.",
  "term_structure": {
    "short_term": { "bias": "bullish", "key_strike": 580, "key_oi": 45000 },
    "long_term":  { "bias": "bullish", "key_strike": 600, "key_oi": 120000 }
  },
  "key_strikes": [
    { "strike": 580, "type": "call_wall", "oi": 45000, "change_5d": "+12000" },
    { "strike": 570, "type": "put_wall",  "oi": 38000, "change_5d": "+8000"  }
  ],
  "trade": {
    "instrument": "Buy Call",
    "entry": 582.50,
    "stop": 580.00,
    "target": 585.00,
    "expiry_dte": 30,
    "risk_reward": "2.5:1",
    "current_price": 582.30
  },
  "risks": [
    "VIX expansion could invalidate call wall",
    "Large positions at $580 may be unwinding, not accumulating"
  ],
  "confluence": "aligned"
}
```

### Anti-Bias Rules in the Prompt

The LLM has explicit instructions to fight confirmation bias:

1. **Not every OI pattern is actionable** — most are noise
2. **Put walls can break, call resistance can fail** — levels are probabilistic
3. **Large positions may be unwinding, not accumulating** — size alone isn't signal
4. **Hedging != directional prediction** — MM delta hedging creates false signals
5. **Conflicting timeframes = lower confidence** — don't force a thesis

### Raw Data In, Analysis Out

Design principle: feed the LLM raw API data as-is. No pre-processing, no filtering, no "let me clean this up first." The MCP servers return well-structured JSON. The LLM is better at finding patterns in complete data than we are at deciding what's "important" to keep.

```
MCP OI Server              MCP Market Data Server
     │                              │
     │  strikes, OI, P/C            │  technicals, volume,
     │  ratios, volumes             │  indicators, price
     │                              │
     └──────────────┬───────────────┘
                    │
                    ▼
            ┌─────────────────────────┐
            │  LLM sees EVERYTHING:   │
            │  4 DTEs of raw OI       │
            │  + raw market data      │
            │  + VIX context          │
            │  + recalled memory      │
            └────────────┬────────────┘
                         │
                         ▼
            Structured JSON output
```

### Clustering & Market Bias

After all 18 tickers are analyzed, the clustering engine groups them into bullish (CALL) / bearish (PUT) / unclear. Market bias is computed from the ratio:

- \>65% bullish = market bullish
- <35% bullish = market bearish
- else = mixed

Top 5 tickers by confidence are extracted as **high conviction picks**.

### UI

The OI analysis runs from a Bloomberg-style terminal as a dedicated tab:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DESK │ DATA │ OI ANALYSIS                                              │
├──────────────┬──────────────────────────────┬───────────────────────────┤
│              │                              │                           │
│  TICKER LIST │   SELECTED TICKER DETAIL     │   MARKET OVERVIEW         │
│              │                              │                           │
│  SPY  78%  A │   Term Structure             │   [Run Analysis]          │
│  NVDA 72%  A │   ┌────┬────┬────┬────┐      │   ████████████░░ 72%     │
│  AAPL 68%  D │   │30  │50  │60  │90  │      │   Analyzing META (8/18)  │
│  META 65%  A │   │bull│bull│neut│bull│      │                           │
│  TSLA 61%  A │   └────┴────┴────┴────┘      │   Signals                │
│  AMZN 58%  D │                              │   BULL ████████░░ 11      │
│  ...        │   "Institutions building      │   BEAR ████░░░░░░  5      │
│              │    call positions above       │   UNCLEAR ██░░░░░░  2     │
│  Sorted by   │    $580..."                  │                           │
│  confidence  │                              │   VIX: low_vol / low fear │
│              │   Key Strikes                │                           │
│  A = Aligned │   $580 call_wall  45K +12K   │   High Conviction         │
│  D = Divergent│  $570 put_wall   38K  +8K   │   1. SPY  CALL 78%       │
│              │                              │   2. NVDA CALL 72%        │
│              │   Trade: Buy Call $582.50    │   3. AAPL PUT  68%        │
│              │   Stop $580 → Target $585   │   4. META CALL 65%        │
│              │   R/R: 2.5:1                │   5. TSLA CALL 61%        │
└──────────────┴──────────────────────────────┴───────────────────────────┘
```

Click "Run Analysis" and watch it stream progress in real-time as each ticker gets collected and analyzed. Results auto-populate when complete.
