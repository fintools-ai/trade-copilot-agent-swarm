"""
OI Bedrock Memory - Long-term memory for OI analysis using AWS Bedrock AgentCore
Stores analysis episodes, retrieves historical context for LLM enrichment

Uses two clients:
- bedrock-agentcore-control (boto3): create/list memory stores
- bedrock_agentcore.memory.MemoryClient: create events, retrieve records
"""

import json
import time
import logging
import boto3
from datetime import datetime
from bedrock_agentcore.memory import MemoryClient
from config.settings import AWS_REGION

logger = logging.getLogger(__name__)

# Memory store name — created once via setup_memory()
MEMORY_NAME = "oi-analysis"
EVENT_EXPIRY_DAYS = 30

_memory_client = None
_control_client = None
_memory_id = None


def _get_memory_client():
    """Data plane client — create events, retrieve records"""
    global _memory_client
    if _memory_client is None:
        _memory_client = MemoryClient(region_name=AWS_REGION)
    return _memory_client


def _get_control_client():
    """Control plane client — create/list memory stores"""
    global _control_client
    if _control_client is None:
        _control_client = boto3.client("bedrock-agentcore-control", region_name=AWS_REGION)
    return _control_client


def _get_memory_id():
    """Get or cache the memory ID"""
    global _memory_id
    if _memory_id is not None:
        return _memory_id

    control = _get_control_client()

    # List memories to find ours
    try:
        response = control.list_memories()
        for mem in response.get("memories", []):
            if mem.get("name") == MEMORY_NAME:
                _memory_id = mem["id"]
                return _memory_id
    except Exception as e:
        logger.warning(f"Bedrock Memory list failed: {e}")

    # Not found — create it
    _memory_id = setup_memory()
    return _memory_id


def setup_memory():
    """One-time setup: create the memory store with semantic + episodic strategies"""
    client = _get_memory_client()

    result = client.create_memory_and_wait(
        name=MEMORY_NAME,
        description="OI pattern analysis memory - stores daily analysis episodes and extracted facts",
        strategies=[
            {"semanticMemoryStrategy": {
                "name": "oi-facts",
                "description": "Key OI facts: levels, walls, bias, patterns",
                "namespaces": ["/facts/{actorId}/"],
            }},
            {"episodicMemoryStrategy": {
                "name": "oi-episodes",
                "description": "Full daily analysis episodes with conditions and outcomes",
                "namespaces": ["/episodes/{actorId}/"],
            }},
        ],
        event_expiry_days=EVENT_EXPIRY_DAYS,
    )

    memory_id = result["memoryId"]
    logger.info(f"Created Bedrock Memory store: {memory_id}")
    return memory_id


def store_episode(ticker, analysis, market_context=None):
    """
    Store an analysis episode after LLM completes.
    Bedrock automatically extracts facts into semantic memory
    and stores the full episode in episodic memory.
    """
    if analysis.get("status") == "error":
        return

    today = datetime.now().strftime("%Y-%m-%d")
    memory_id = _get_memory_id()
    client = _get_memory_client()

    trade = analysis.get("trade", {})
    term = analysis.get("term_structure", {})
    short_term = term.get("short_term", {})
    long_term = term.get("long_term", {})
    key_strikes = analysis.get("key_strikes", [])

    # Format key strikes for readability
    strikes_text = "\n".join(
        f"  - ${s.get('strike')} ({s.get('type')}): {s.get('oi', 0):,} OI, 5d change: {s.get('change_5d', 'N/A')}"
        for s in key_strikes
    )

    # Build context string
    regime = market_context.get("regime", "unknown") if market_context else "unknown"
    fear = market_context.get("fear_level", "unknown") if market_context else "unknown"

    content = f"""OI Analysis for {ticker} on {today}:
Direction: {analysis.get('direction', 'N/A')} | Confidence: {analysis.get('confidence', 0)}%
Confluence: {analysis.get('confluence', 'N/A')} (short-term {short_term.get('bias', '?')}, long-term {long_term.get('bias', '?')})
Thesis: {analysis.get('thesis', 'N/A')}
Key Strikes:
{strikes_text}
Trade: {trade.get('instrument', 'N/A')} entry ${trade.get('entry', '?')} stop ${trade.get('stop', '?')} target ${trade.get('target', '?')} R/R {trade.get('risk_reward', '?')}
Current Price: ${trade.get('current_price', '?')}
Market Regime: {regime} | Fear: {fear}
Risks: {', '.join(analysis.get('risks', []))}"""

    try:
        t0 = time.time()
        client.create_event(
            memory_id=memory_id,
            actor_id=f"/ticker/{ticker}",
            session_id=f"analysis-{today}",
            event_timestamp=int(datetime.now().timestamp() * 1000),
            payload=[{
                "Conversational": {
                    "content": content,
                    "role": "assistant"
                }
            }]
        )
        logger.info(f"{ticker}: stored episode ({analysis.get('direction', '?')} {analysis.get('confidence', '?')}%) ({time.time()-t0:.1f}s)")
    except Exception as e:
        logger.warning(f"{ticker}: store_episode failed - {e}")


def recall(ticker, query=None):
    """
    Retrieve historical context for a ticker from Bedrock Memory.
    Returns list of relevant fact strings for LLM prompt injection.
    """
    if query is None:
        query = f"OI patterns, key levels, institutional positioning, past accuracy for {ticker}"

    memory_id = _get_memory_id()
    client = _get_memory_client()

    facts = []

    try:
        t0 = time.time()
        result = client.retrieve_memory_records(
            memory_id=memory_id,
            namespace=f"/facts//ticker/{ticker}/",
            search_criteria={
                "searchQuery": query,
                "topK": 5,
            }
        )

        for record in result.get("memoryRecords", []):
            content = record.get("content", {}).get("value", "")
            if content:
                facts.append(content)

        dur = time.time() - t0
        if facts:
            logger.info(f"{ticker}: recalled {len(facts)} facts ({dur:.1f}s)")
        else:
            logger.info(f"{ticker}: no facts found ({dur:.1f}s)")

    except Exception as e:
        logger.warning(f"{ticker}: recall failed - {e}")

    return facts
