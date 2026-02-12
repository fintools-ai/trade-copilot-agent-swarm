"""
End-of-day pattern extraction — how the system learns.

Retrieves today's trade outcomes, calls Claude to extract rules/anti-patterns,
stores extracted patterns in /facts/patterns/SPY/ for next-day recall.
"""

import json
import logging
import asyncio

import boto3

from config.settings import AWS_REGION, ENGINE_MODEL_ID, ENGINE_MAX_TOKENS
from trading_engine.memory import MemoryStore
from redis_stream import publish_event

logger = logging.getLogger(__name__)

CONSOLIDATION_PROMPT = """You are analyzing today's 0DTE trade outcomes to extract reusable patterns.

<outcomes>
{outcomes}
</outcomes>

Extract patterns from these trade outcomes. For each pattern:

1. RULES (what works):
   "IF [classified conditions] THEN [action] — [wins]W/[losses]L ([win_pct]%)"
   Example: "IF Flow STRONG_BUYING + RSI OVERSOLD + morning THEN CALL — 4W/1L (80%)"

2. ANTI-PATTERNS (what to avoid):
   "AVOID [classified conditions] — [losses]L/[total] ([loss_pct]%)"
   Example: "AVOID MIXED flow + OVERBOUGHT + midday — 1W/4L (20% win)"

3. TIME PATTERNS:
   Any time-of-day tendencies in win/loss rates.

4. CONVICTION CALIBRATION:
   Did HIGH conviction trades actually win more? Should thresholds change?

Output ONLY the extracted patterns, one per line. No explanations. Format each as a single concise line.
If fewer than 2 trades, output "INSUFFICIENT_DATA" only."""


async def consolidate(memory: MemoryStore):
    """
    End-of-day consolidation: extract patterns from today's trades.

    1. Retrieve today's outcomes from AgentCore
    2. If < 2 trades, skip
    3. Call Claude to extract patterns
    4. Store patterns in /facts/patterns/SPY/
    5. Publish summary to UI
    """
    outcomes = await asyncio.get_event_loop().run_in_executor(None, memory.get_todays_outcomes)

    if len(outcomes) < 2:
        print(f"[v2] Consolidation: only {len(outcomes)} trades today, skipping")
        publish_event("ENGINE_STATUS", f"Consolidation skipped — {len(outcomes)} trades (need 2+)")
        return

    print(f"[v2] Consolidation: analyzing {len(outcomes)} trades...")
    outcomes_text = "\n\n".join(f"Trade {i+1}:\n{o}" for i, o in enumerate(outcomes))

    # Call Claude for pattern extraction
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": ENGINE_MAX_TOKENS * 2,
        "messages": [
            {"role": "user", "content": CONSOLIDATION_PROMPT.format(outcomes=outcomes_text)}
        ],
    })

    try:
        response = bedrock.invoke_model(
            modelId=ENGINE_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]
    except Exception as e:
        logger.error(f"Consolidation Claude call failed: {e}")
        print(f"[v2] Consolidation error: {e}")
        return

    if "INSUFFICIENT_DATA" in text:
        print("[v2] Consolidation: insufficient data for pattern extraction")
        return

    # Parse patterns (one per line)
    patterns = [line.strip() for line in text.strip().split("\n") if line.strip() and not line.startswith("#")]

    if not patterns:
        print("[v2] Consolidation: no patterns extracted")
        return

    # Store in AgentCore
    await asyncio.get_event_loop().run_in_executor(None, memory.store_patterns, patterns)

    # Publish summary
    summary = f"Consolidated {len(outcomes)} trades → {len(patterns)} patterns:\n" + \
              "\n".join(f"  - {p[:100]}" for p in patterns[:5])
    publish_event("ENGINE_STATUS", summary)
    print(f"[v2] Consolidation complete: {len(patterns)} patterns stored")
    print(summary)
