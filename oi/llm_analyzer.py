"""
LLM Analyzer - Per-DTE focused analysis + synthesis
Each DTE gets its own Bedrock call with timeframe-specific framing.
A synthesis call compares per-DTE results to produce the overall verdict.
"""

import asyncio
import json
import time
import logging
import boto3
from datetime import datetime
from config.settings import AWS_REGION, BEDROCK_MODEL_ID, OI_BEDROCK_CONCURRENCY

logger = logging.getLogger(__name__)

# Timeframe-specific framing for per-DTE prompts
DTE_FRAMING = {
    30: {
        "label": "SHORT-TERM (30 DTE)",
        "focus": "momentum/gamma positioning",
        "questions": (
            "Are market makers net short gamma at key strikes? "
            "Is there a gamma squeeze setup where a move through a high-OI strike forces MM delta hedging? "
            "What does the put/call skew at this expiry tell us about near-term fear vs greed? "
            "Are there signs of retail speculation (heavy OTM call buying) vs institutional hedging?"
        ),
    },
    50: {
        "label": "MEDIUM-TERM (50 DTE)",
        "focus": "swing positioning",
        "questions": (
            "Are institutions rolling positions into this expiry from shorter-dated options? "
            "Is there accumulation at specific strikes suggesting a directional swing bet? "
            "What does the P/C ratio trend suggest about medium-term sentiment? "
            "Are there signs of spread structures (verticals, calendars) being built?"
        ),
    },
    60: {
        "label": "MEDIUM-TERM (60 DTE)",
        "focus": "institutional setups",
        "questions": (
            "Are large block trades building walls at specific strikes? "
            "Is there protective put buying (portfolio hedging) or aggressive call selling (overwriting)? "
            "What's the institutional thesis implied by the OI distribution? "
            "Are there collar structures (simultaneous put buying + call selling) visible?"
        ),
    },
    90: {
        "label": "LONG-TERM (90 DTE)",
        "focus": "strategic/hedging positioning",
        "questions": (
            "Is this hedging activity (protective puts, covered calls) or directional accumulation? "
            "Are collar structures visible that reveal institutional price range expectations? "
            "What does persistent OI at these strikes tell us about where institutions expect the stock in 3 months? "
            "Is there evidence of LEAPS rolling or new strategic positions being established?"
        ),
    },
}


class LLMAnalyzer:
    def __init__(self):
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
        self.model_id = BEDROCK_MODEL_ID
        self._semaphore = asyncio.Semaphore(OI_BEDROCK_CONCURRENCY)

    # ── Per-DTE Analysis ──────────────────────────────────────────────

    def analyze_dte(self, ticker, dte, dte_data, market_context=None):
        """
        Focused LLM analysis of ONE DTE for one ticker.
        No historical context — that's reserved for synthesis.
        """
        try:
            prompt = self._build_dte_prompt(ticker, dte, dte_data, market_context)
            logger.info(f"{ticker} {dte}DTE: prompt ~{len(prompt)} chars")

            t0 = time.time()
            response = self._call_bedrock(prompt, max_tokens=2000)
            dur = time.time() - t0

            analysis = self._parse_dte_response(response, dte)
            analysis["dte"] = dte

            if analysis.get("status") == "error":
                logger.warning(f"{ticker} {dte}DTE: parse failed - {analysis.get('error', '?')} ({dur:.1f}s)")
            else:
                logger.info(f"{ticker} {dte}DTE: {analysis.get('bias', '?')} {analysis.get('confidence', '?')}% ({dur:.1f}s)")

            return analysis
        except Exception as e:
            logger.error(f"{ticker} {dte}DTE: analysis failed - {e}")
            return {"dte": dte, "status": "error", "error": str(e)}

    async def _analyze_dte_async(self, ticker, dte, dte_data, market_context):
        """Async wrapper — runs sync analyze_dte in thread pool with semaphore"""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.analyze_dte, ticker, dte, dte_data, market_context
            )

    def _build_dte_prompt(self, ticker, dte, dte_data, market_context):
        """Build focused prompt for a single DTE — raw data passed as-is"""
        framing = DTE_FRAMING.get(dte, {
            "label": f"{dte} DTE",
            "focus": "options positioning",
            "questions": "What does the OI distribution tell us about positioning at this timeframe?",
        })

        oi_json = json.dumps(dte_data.get('oi_data'), indent=2) if dte_data.get('oi_data') else 'No data'
        delta_json = json.dumps(dte_data.get('delta'), indent=2) if dte_data.get('delta') else 'No delta'
        market_json = json.dumps(dte_data.get('market_data'), indent=2) if dte_data.get('market_data') else 'No data'
        vix_json = json.dumps(market_context, indent=2) if market_context else 'Not available'

        return f"""You are a quantitative options analyst. Analyze the {framing['label']} open interest data for {ticker} and determine what smart money is doing at this expiration.

# YOUR TASK
{framing['questions']}

# STEP-BY-STEP ANALYSIS PROCESS (follow this order for accuracy)

## Step 1: Assess Data Quality
- Is total OI meaningful (>1000 contracts) or thin/illiquid?
- Are delta changes available and significant, or is this baseline data?
- Does the technical data confirm or contradict the OI story?
Rate as A (high quality, all sources) / B (adequate) / C (partial/thin) / D (insufficient — lower confidence)

## Step 2: Identify the OI Structure
- P/C ratio and what it implies (< 0.7 = call-heavy bullish, > 1.3 = put-heavy bearish)
- Max pain level and distance from current price
- Where are the OI walls? (strikes with >10,000 OI)
- Are walls clustered at round numbers (retail) or odd strikes (institutional)?

## Step 3: Read the Delta (Day-over-Day Changes)
This is where ACCURACY comes from — static OI is stale, CHANGES reveal intent:
- Rising OI = NEW positions being opened (directional signal)
- Falling OI = positions being CLOSED/unwound (momentum fading)
- Large single-strike jumps (>5,000) = institutional block trades
- P/C ratio shift direction = sentiment shift

## Step 4: Decode WHO is Positioning
- Heavy OTM call accumulation = retail speculation or institutional upside bets
- ITM put accumulation = hedging (protective puts)
- ATM straddle/strangle = volatility bets, not directional
- Simultaneous call selling + put buying at same strikes = collar = institutional hedging

## Step 5: Assess Gamma & Dealer Impact
- MMs short gamma at high-OI call strikes → price through that strike forces delta hedging that ACCELERATES the move
- MMs short gamma at high-OI put strikes → hedge by selling stock as price drops
- Max pain acts as magnet near expiry
- At {dte} DTE: {"gamma effects are STRONG — dealers actively hedge, walls are real constraints" if dte <= 30 else "gamma effects are moderate — walls are softer, more about positioning intent" if dte <= 60 else "gamma effects are weak at this distance — focus on positioning INTENT not mechanical levels"}

## Step 6: Form Your Thesis
4-6 sentences explaining:
1. WHAT: Cite specific strikes, OI levels, P/C ratio, delta changes from the data
2. WHO: Who is behind this positioning (block size, strike selection, structure)
3. WHY: The mechanical reason they're positioned this way
4. IMPLICATION: What this means for {ticker} price over the next {dte} days

BAD: "Bullish positioning detected with strong call OI."
GOOD: "The $580 call strike has 45,000 OI (+12,000 change), suggesting institutional accumulation above the current $575 price. MMs who sold these calls are short gamma — a move above $580 forces delta hedging that accelerates the rally. The P/C ratio of 0.65 confirms call-dominant flow. However, the $590 call wall (30,000 OI) caps upside, creating a $580-$590 range."

# ANTI-BIAS RULES
1. Most OI is noise from hedging, rolling, and spread legs — not directional
2. High OI at a strike ≠ "wall". It could be one leg of a spread
3. Large positions may be UNWINDING, not accumulating — check delta direction
4. If signals conflict or data is thin, confidence MUST be low (<50%)

# CONFIDENCE CALIBRATION
- 80-95%: Multiple confirming signals, quality data, clear mechanical thesis
- 65-79%: Good evidence, 2+ signals align, minor uncertainties
- 50-64%: Modest edge, some conflicting signals
- 35-49%: Weak/thin data, conflicting signals
- <35%: Insufficient data — say "neutral"
Most real-world accuracy is 55-65%. Don't inflate.

# DATA FOR {ticker} — {dte} DTE

## Open Interest Data:
{oi_json}

## Delta Changes (vs yesterday):
{delta_json}

## Technical/Market Data:
{market_json}

## Market Context (VIX):
{vix_json}

# OUTPUT FORMAT
Return ONLY valid JSON. No text before or after.

{{
  "dte": {dte},
  "bias": "bullish or bearish or neutral",
  "confidence": 65,
  "thesis": "4-6 sentence thesis following WHAT/WHO/WHY/IMPLICATION structure",
  "key_strike": 580,
  "key_strike_type": "call_wall or put_wall or max_pain",
  "key_oi": 45000,
  "pcr": 0.85,
  "max_pain": 575,
  "notable_flow": "describe unusual activity or empty string if nothing notable",
  "data_quality": "A or B or C or D",
  "smart_money_read": "BULLISH_ACCUMULATION or BEARISH_DISTRIBUTION or HEDGING or UNWINDING or MIXED",
  "gamma_risk": "squeeze_up or squeeze_down or neutral or unknown",
  "key_strikes": [
    {{"strike": 580, "type": "call_wall or put_wall or max_pain", "oi": 45000, "change_5d": "+12000 or N/A", "interpretation": "mechanical meaning for price action"}}
  ]
}}"""

    def _parse_dte_response(self, response_text, dte):
        """Parse per-DTE LLM JSON response"""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No valid JSON found")

            analysis = json.loads(response_text[json_start:json_end])

            for field in ["bias", "confidence", "thesis"]:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")

            analysis["confidence"] = int(str(analysis["confidence"]).replace('%', '').split('.')[0])
            return analysis
        except Exception as e:
            return {
                "dte": dte,
                "status": "error",
                "error": f"Failed to parse: {str(e)}",
                "raw_response": response_text[:500],
            }

    # ── Synthesis ─────────────────────────────────────────────────────

    def synthesize_ticker(self, ticker, dte_analyses, market_context=None, historical_context=None, sentiment_context=None):
        """
        Synthesis LLM call: compare per-DTE structured results → overall verdict + trade.
        Receives compact structured JSON, not raw OI data. Small prompt.
        """
        try:
            prompt = self._build_synthesis_prompt(ticker, dte_analyses, market_context, historical_context, sentiment_context)
            n_dtes = len(dte_analyses)
            n_facts = len(historical_context) if historical_context else 0
            n_sentiment = len(sentiment_context) if sentiment_context else 0
            logger.info(f"{ticker}: synthesis prompt ~{len(prompt)} chars, {n_dtes} DTEs, {n_facts} memory facts, {n_sentiment} sentiment facts")

            t0 = time.time()
            response = self._call_bedrock(prompt, max_tokens=4000)
            dur = time.time() - t0

            analysis = self._parse_synthesis_response(response)
            analysis["ticker"] = ticker
            analysis["analysis_timestamp"] = datetime.now().isoformat()

            if analysis.get("status") == "error":
                logger.warning(f"{ticker}: synthesis parse failed - {analysis.get('error', '?')} ({dur:.1f}s)")
            else:
                logger.info(f"{ticker}: SYNTHESIS {analysis.get('direction', '?')} {analysis.get('confidence', '?')}% {analysis.get('confluence', '?')} ({dur:.1f}s)")

            return analysis
        except Exception as e:
            logger.error(f"{ticker}: synthesis failed - {e}")
            return {
                "ticker": ticker,
                "status": "error",
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
            }

    def _build_synthesis_prompt(self, ticker, dte_analyses, market_context, historical_context=None, sentiment_context=None):
        """Build synthesis prompt from per-DTE structured results"""
        dte_block = json.dumps(dte_analyses, indent=2)
        vix_json = json.dumps(market_context, indent=2) if market_context else 'Not available'

        return f"""You are a quantitative options analyst. You have already analyzed each DTE individually for {ticker}. Now synthesize the term structure to produce a final trade recommendation.

# STEP-BY-STEP SYNTHESIS (follow this order for accuracy)

## Step 1: Compare DTE Biases
Look at each DTE's bias, confidence, and data quality:
- If 3-4 DTEs agree on direction with >60% confidence → "aligned", HIGH overall confidence
- If short-term (30) disagrees with long-term (90) → "divergent", investigate why
- If data quality is D at any DTE, DISCOUNT that DTE's bias — don't let bad data drive the conclusion
- Weight long-term DTEs (60-90) more than short-term (30) for directional conviction — institutions drive longer-dated positioning

## Step 2: Cross-Reference Smart Money Signals
- ACCUMULATION at multiple DTEs = strong directional conviction
- HEDGING at 90 DTE + ACCUMULATION at 30 DTE = collar structure = neutral-to-bearish despite short-term call flow
- UNWINDING across DTEs = positioning is CLOSING, not a new signal — reduce confidence
- MIXED at most DTEs = genuinely uncertain — say so, don't force a direction

## Step 3: Assess Gamma vs Positioning Alignment
- Short-term gamma risk (squeeze_up/down at 30 DTE) shows the MECHANICAL pressure
- Long-term positioning shows the STRATEGIC intent
- When both align (gamma squeeze up + long-term bullish accumulation) = highest conviction
- When they conflict (gamma squeeze down but long-term bullish) = short-term pain, longer-term recovery

## Step 4: Select Trade Strategy
ONLY use these 3 types based on the combined evidence:
- **Buy Call**: Accumulation across DTEs + gamma squeeze potential + aligned term structure
- **Buy Put**: Distribution across DTEs + bearish flow + protective hedging at long DTEs
- **Put Credit Spread**: Strong put walls at long DTEs creating support + neutral-to-bullish bias + high IV environment (premium selling edge)

## Step 5: Form Synthesis Thesis
Your thesis MUST be 4-6 sentences that SYNTHESIZE, not summarize:
1. What the term structure reveals when viewed as a whole (reference specific DTE biases + confidence)
2. Where institutional consensus exists and where it breaks down
3. The key level or strike that determines direction (cite the number)
4. What would INVALIDATE this trade — the specific level or signal that means you're wrong

BAD: "30 DTE is bullish, 60 DTE is bullish, 90 DTE is bullish, so we recommend calls."
GOOD: "Institutions are accumulating calls across all timeframes: 30 DTE shows $580 call wall (+12K 5-day change), while 90 DTE confirms strategic positioning with $600 calls at 120K OI. The P/C ratio declining from 1.1 to 0.7 across DTEs signals progressive bullish rotation. The gamma squeeze setup at 30 DTE ($580 flip point) could catalyze a move that aligns with the 90 DTE target of $600. Invalidation: a close below $570 max pain would break the thesis."

# ACCURACY RULES
1. If DTEs strongly disagree, confidence MUST be <50% and confluence = "divergent"
2. Don't force a trade — if data is genuinely mixed, say confidence 35-45% and acknowledge it
3. A DTE with data_quality D should not drive the overall direction
4. If only 1-2 DTEs have data, reduce confidence proportionally
5. Don't repeat the per-DTE theses — SYNTHESIZE what the combined picture means
6. success_probability should be realistic: 55-65% is typical for good setups

# CONFIDENCE CALIBRATION
- 80-95%: All DTEs aligned, quality data (A/B), clear gamma catalyst, strong accumulation
- 65-79%: Most DTEs agree, adequate data, reasonable mechanical thesis
- 50-64%: Some agreement but conflicting signals or thin data at key DTEs
- 35-49%: Significant disagreement or poor data quality
- <35%: Highly mixed or insufficient data — trade not recommended

# PER-DTE ANALYSIS RESULTS
{dte_block}

# Market Context (VIX):
{vix_json}

{self._build_memory_section(ticker, historical_context)}

{self._build_sentiment_section(ticker, sentiment_context)}

# OUTPUT FORMAT
Return ONLY valid JSON. No text before or after.

RULES:
- direction must be exactly "CALL" or "PUT"
- confidence must be integer 0-100
- confluence must be exactly "aligned" or "divergent"
- instrument must be exactly "Buy Call", "Buy Put", or "Put Credit Spread"
- All price fields are STOCK PRICES (not option premiums), plain numbers without $ signs
- key_strikes: 3-6 most important strikes across ALL DTEs with interpretations
- trade.expiry_dte: choose the BEST expiry for the trade based on your analysis
- success_probability: realistic integer 0-100 (separate from confidence)

{{
  "direction": "CALL or PUT",
  "confidence": 75,
  "success_probability": 65,
  "thesis": "4-6 sentence synthesis thesis (see requirements above)",
  "confluence": "aligned or divergent",
  "pattern_type": "institutional_accumulation or distribution or gamma_squeeze_setup or protective_hedging or mixed",
  "smart_money_summary": "1-2 sentence summary of what institutions are doing across timeframes",
  "term_structure": {{
    "short_term": {{"bias": "bullish or bearish or neutral", "key_strike": 580, "key_oi": 45000, "note": "brief note on what this means"}},
    "long_term": {{"bias": "bullish or bearish or neutral", "key_strike": 600, "key_oi": 120000, "note": "brief note on what this means"}}
  }},
  "key_strikes": [
    {{"strike": 580, "type": "call_wall or put_wall or max_pain", "oi": 45000, "change_5d": "+12000 or N/A", "interpretation": "what this strike means"}}
  ],
  "trade": {{
    "instrument": "Buy Call or Buy Put or Put Credit Spread",
    "entry": 582.50,
    "stop": 580.00,
    "target": 585.00,
    "expiry_dte": 30,
    "risk_reward": "2.5:1",
    "current_price": 582.30,
    "entry_triggers": ["trigger1", "trigger2"],
    "exit_strategy": "when and how to exit"
  }},
  "risk_management": {{
    "primary_risks": ["risk1", "risk2", "risk3"],
    "hedge_strategy": "how to hedge if wrong",
    "volatility_considerations": "IV impact on the trade",
    "position_adjustments": "when and how to adjust"
  }},
  "data_quality": {{
    "oi_quality": "High or Medium or Low",
    "delta_reliability": "Strong or Weak or Baseline",
    "technical_confluence": "Confirming or Neutral or Conflicting",
    "overall_score": "A or B or C or D"
  }}
}}"""

    def _parse_synthesis_response(self, response_text):
        """Parse synthesis LLM JSON response (same format as legacy)"""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No valid JSON found in response")

            analysis = json.loads(response_text[json_start:json_end])

            for field in ["direction", "confidence", "thesis", "trade"]:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")

            analysis["confidence"] = int(str(analysis["confidence"]).replace('%', '').split('.')[0])
            return analysis
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse synthesis: {str(e)}",
                "raw_response": response_text[:500],
            }

    # ── Full Orchestrator ─────────────────────────────────────────────

    async def analyze_ticker_full(self, ticker, all_dte_data, market_context=None, historical_context=None, sentiment_context=None):
        """
        Full per-DTE analysis pipeline:
        1. Run per-DTE LLM calls in parallel (asyncio.gather)
        2. Collect results, filter errors
        3. Run synthesis LLM call with valid DTE results + memory
        4. Return combined result: {...synthesis_fields, dte_analyses: [...]}
        """
        try:
            # Step 1: Per-DTE analysis in parallel
            dte_tasks = []
            dte_labels = []
            for dte in sorted(all_dte_data.keys(), key=lambda x: int(x)):
                data = all_dte_data[dte]
                if data.get("oi_data"):
                    dte_tasks.append(self._analyze_dte_async(ticker, int(dte), data, market_context))
                    dte_labels.append(dte)

            if not dte_tasks:
                return {
                    "ticker": ticker,
                    "status": "error",
                    "error": "No OI data available for any DTE",
                    "analysis_timestamp": datetime.now().isoformat(),
                }

            logger.info(f"{ticker}: starting {len(dte_tasks)} per-DTE analyses ({', '.join(str(d) for d in dte_labels)} DTE)")
            t0 = time.time()
            dte_results = await asyncio.gather(*dte_tasks, return_exceptions=True)

            # Step 2: Collect valid results
            valid_analyses = []
            for i, result in enumerate(dte_results):
                if isinstance(result, Exception):
                    logger.warning(f"{ticker} {dte_labels[i]}DTE: EXCEPTION - {result}")
                elif result.get("status") == "error":
                    logger.warning(f"{ticker} {dte_labels[i]}DTE: ERROR - {result.get('error', '?')}")
                else:
                    valid_analyses.append(result)

            dte_dur = time.time() - t0
            logger.info(f"{ticker}: {len(valid_analyses)}/{len(dte_tasks)} DTE analyses succeeded ({dte_dur:.1f}s)")

            if not valid_analyses:
                return {
                    "ticker": ticker,
                    "status": "error",
                    "error": "All DTE analyses failed",
                    "analysis_timestamp": datetime.now().isoformat(),
                }

            # Step 3: Synthesis
            logger.info(f"{ticker}: synthesizing {len(valid_analyses)} DTE results...")
            synthesis = self.synthesize_ticker(ticker, valid_analyses, market_context, historical_context, sentiment_context)

            if synthesis.get("status") == "error":
                # Synthesis failed but we still have per-DTE results — return what we have
                synthesis["dte_analyses"] = valid_analyses
                return synthesis

            # Step 4: Combine
            synthesis["dte_analyses"] = valid_analyses
            synthesis["ticker"] = ticker
            synthesis["analysis_timestamp"] = datetime.now().isoformat()
            return synthesis

        except Exception as e:
            logger.error(f"{ticker}: analyze_ticker_full failed - {e}")
            return {
                "ticker": ticker,
                "status": "error",
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
            }

    # ── Legacy (single-call) ──────────────────────────────────────────

    def analyze_ticker_legacy(self, ticker, all_dte_data, market_context=None, historical_context=None):
        """Legacy: ONE LLM call with ALL DTEs combined (fallback)"""
        try:
            prompt = self._build_legacy_prompt(ticker, all_dte_data, market_context, historical_context)
            n_dtes = len(all_dte_data)
            n_facts = len(historical_context) if historical_context else 0
            logger.info(f"{ticker}: legacy prompt ~{len(prompt)} chars, {n_dtes} DTEs, {n_facts} memory facts")

            t0 = time.time()
            response = self._call_bedrock(prompt)
            bedrock_dur = time.time() - t0
            logger.info(f"{ticker}: Bedrock call {bedrock_dur:.1f}s")

            analysis = self._parse_synthesis_response(response)
            analysis["ticker"] = ticker
            analysis["analysis_timestamp"] = datetime.now().isoformat()

            if analysis.get("status") == "error":
                logger.warning(f"{ticker}: parse failed - {analysis.get('error', 'unknown')}")
            else:
                logger.info(f"{ticker}: {analysis.get('direction', '?')} {analysis.get('confidence', '?')}% conf, {analysis.get('confluence', '?')}")

            return analysis
        except Exception as e:
            logger.error(f"{ticker}: LLM analysis failed - {e}")
            return {
                "ticker": ticker,
                "status": "error",
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
            }

    def _build_legacy_prompt(self, ticker, all_dte_data, market_context, historical_context=None):
        """Build legacy prompt with all DTEs for one ticker"""
        dte_sections = []
        for dte in sorted(all_dte_data.keys(), key=lambda x: int(x)):
            data = all_dte_data[dte]
            section = f"""### {dte} DTE

#### Open Interest Data:
{json.dumps(data.get('oi_data'), indent=2) if data.get('oi_data') else 'No data'}

#### Delta Changes (vs yesterday):
{json.dumps(data.get('delta'), indent=2) if data.get('delta') else 'No delta'}

#### Technical/Market Data:
{json.dumps(data.get('market_data'), indent=2) if data.get('market_data') else 'No data'}"""
            dte_sections.append(section)

        dte_block = "\n\n".join(dte_sections)

        return f"""You are a professional options trader analyzing institutional open interest patterns for {ticker}.
You have data across MULTIPLE expiration timeframes (DTEs). Analyze the FULL TERM STRUCTURE.

# CRITICAL INSTRUCTIONS
- Analyze ALL DTEs together to see the full picture
- Short-term (30 DTE) = momentum/gamma plays
- Medium-term (50-60 DTE) = swing setups
- Long-term (90 DTE) = institutional hedging/strategic positioning
- When short and long term agree = HIGH conviction (aligned)
- When they disagree = investigate why (divergent)

# ANTI-BIAS RULES
1. Not every OI pattern is actionable — most are noise
2. Put walls can break, call resistance can fail
3. Large positions may be unwinding, not accumulating
4. Hedging activity != directional prediction
5. If signals conflict across timeframes, lower your confidence

# RAW DATA FOR {ticker}

{dte_block}

## Market Context (VIX):
{json.dumps(market_context, indent=2) if market_context else 'Not available'}

{self._build_memory_section(ticker, historical_context)}

# OUTPUT FORMAT
Return ONLY valid JSON with this exact structure. No text before or after.

CRITICAL FORMATTING RULES:
- ALL price fields must be STOCK PRICES (not option premiums), plain numbers without $ signs
- confidence must be integer 0-100
- direction must be exactly "CALL" or "PUT"
- confluence must be exactly "aligned" or "divergent"
- key_strikes array should have 3-6 most important strikes

{{
  "direction": "CALL or PUT",
  "confidence": 75,
  "thesis": "2-3 sentence institutional thesis explaining what smart money is doing and why",
  "term_structure": {{
    "short_term": {{"bias": "bullish or bearish or neutral", "key_strike": 580, "key_oi": 45000, "note": "brief note"}},
    "long_term": {{"bias": "bullish or bearish or neutral", "key_strike": 600, "key_oi": 120000, "note": "brief note"}}
  }},
  "key_strikes": [
    {{"strike": 580, "type": "call_wall or put_wall or max_pain", "oi": 45000, "change_5d": "+12000 or N/A"}}
  ],
  "trade": {{
    "instrument": "Buy Call or Buy Put or Put Credit Spread",
    "entry": 582.50,
    "stop": 580.00,
    "target": 585.00,
    "expiry_dte": 30,
    "risk_reward": "2.5:1",
    "current_price": 582.30
  }},
  "risks": ["risk1", "risk2"],
  "confluence": "aligned or divergent"
}}"""

    # ── Shared Helpers ────────────────────────────────────────────────

    def _build_memory_section(self, ticker, historical_context):
        """Build historical context section from Bedrock Memory facts"""
        if not historical_context:
            return ""

        facts = "\n".join(f"- {fact}" for fact in historical_context)
        return f"""# PAST ANALYSES FOR {ticker}
Each line is a daily snapshot: direction, confidence, key walls, price levels, per-DTE bias.
Compare today vs past days to answer:
- Are the same walls showing up repeatedly? (persistent = high conviction level)
- Is direction consistent or reversing? (trend vs noise)
- Are walls growing or shrinking? (building = real, fading = unwinding)
- Did price respect past key levels?

{facts}
"""

    def _build_sentiment_section(self, ticker, sentiment_context):
        """Build social media sentiment section from market_sentiment memory"""
        if not sentiment_context:
            return ""

        facts = "\n".join(f"- {fact}" for fact in sentiment_context)
        return f"""# SOCIAL MEDIA SENTIMENT FOR {ticker}
Crowd sentiment from X/Twitter posts
This is a SECONDARY signal — use it to confirm or question OI-based thesis, not to override it.
- Crowd bullish + OI bullish = higher conviction
- Crowd bearish + OI bullish = potential contrarian setup (smart money vs crowd)
- Crowd consensus is often wrong at extremes — extreme bullish sentiment can be a fade signal

{facts}
"""

    def _call_bedrock(self, prompt, max_tokens=4000):
        """Call AWS Bedrock"""
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
        )

        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
