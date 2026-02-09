"""
OI Bedrock Memory - Long-term memory for OI analysis using AWS Bedrock AgentCore

Uses MemoryClient (snake_case params) for all operations:
- create_memory_and_wait: one-time setup
- list_memories: find existing memory by ID prefix
- create_event: store analysis episodes (messages as tuples)
- retrieve_memories: search semantic memory for facts
"""

import time
import logging
import boto3
from datetime import datetime
from bedrock_agentcore.memory import MemoryClient
from config.settings import AWS_REGION

logger = logging.getLogger(__name__)

MEMORY_NAME = "oi_analysis"
EVENT_EXPIRY_DAYS = 30

_memory_client = None
_control_client = None
_memory_id = None


def _get_memory_client():
    """MemoryClient — snake_case params, handles events + retrieval"""
    global _memory_client
    if _memory_client is None:
        _memory_client = MemoryClient(region_name=AWS_REGION)
    return _memory_client


def _get_control_client():
    """boto3 control plane — camelCase, for get_memory to resolve name→ID"""
    global _control_client
    if _control_client is None:
        _control_client = boto3.client("bedrock-agentcore-control", region_name=AWS_REGION)
    return _control_client


def _get_memory_id():
    """
    Get or cache the memory ID.

    Strategy:
    1. list_memories via MemoryClient (returns dicts with 'id' key, NO 'name')
    2. For each, call get_memory via boto3 control plane to check the name
    3. If not found, create a new memory store
    4. If "already exists" error, retry listing
    """
    global _memory_id
    if _memory_id is not None:
        return _memory_id

    client = _get_memory_client()
    control = _get_control_client()

    # Step 1: List memories and resolve by name
    try:
        memories = client.list_memories()
        logger.info(f"list_memories returned {len(memories)} entries")

        for mem in memories:
            mid = mem.get("id", "")
            if not mid:
                continue

            # list_memories doesn't return 'name', so check via get_memory
            # But first try ID prefix match (IDs often start with the name)
            if mid.startswith(MEMORY_NAME):
                _memory_id = mid
                logger.info(f"Found memory by ID prefix: {_memory_id}")
                return _memory_id

            # Try get_memory to check the actual name
            try:
                detail = control.get_memory(memoryId=mid)
                mem_detail = detail.get("memory", detail)
                name = mem_detail.get("name", "")
                if name == MEMORY_NAME:
                    _memory_id = mid
                    logger.info(f"Found memory by name lookup: {_memory_id}")
                    return _memory_id
            except Exception:
                pass  # get_memory failed for this entry, skip

    except Exception as e:
        logger.warning(f"list_memories failed: {e}")

    # Step 2: Not found — create
    try:
        _memory_id = _create_memory()
        return _memory_id
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.warning(f"Memory '{MEMORY_NAME}' already exists, retrying list...")
            # Retry listing — race condition or list was stale
            try:
                memories = client.list_memories()
                for mem in memories:
                    mid = mem.get("id", "")
                    if mid and mid.startswith(MEMORY_NAME):
                        _memory_id = mid
                        logger.info(f"Found memory on retry: {_memory_id}")
                        return _memory_id
            except Exception as e2:
                logger.error(f"Retry list failed: {e2}")
        else:
            logger.error(f"Failed to create memory: {e}")

    return _memory_id


def _create_memory():
    """One-time setup: create the memory store"""
    client = _get_memory_client()

    result = client.create_memory_and_wait(
        name=MEMORY_NAME,
        description="OI pattern analysis memory - stores daily analysis episodes and extracted facts",
        strategies=[
            {"semanticMemoryStrategy": {
                "name": "oi_facts",
                "description": "Key OI facts: levels, walls, bias, patterns",
                "namespaces": ["/facts/{actorId}/"],
            }},
            {"summaryMemoryStrategy": {
                "name": "oi_summaries",
                "description": "Rolling summaries of daily OI analysis episodes",
                "namespaces": ["/summaries/{actorId}/{sessionId}/"],
            }},
        ],
        event_expiry_days=EVENT_EXPIRY_DAYS,
    )

    # MemoryClient returns "id", not "memoryId"
    memory_id = result.get("id") or result.get("memoryId", "")
    logger.info(f"Created Bedrock Memory store: {memory_id}")
    return memory_id


def store_episode(ticker, analysis, market_context=None):
    """
    Store compact structured snapshot for multi-day comparison.
    Bedrock auto-extracts facts — compact format makes extraction cleaner.
    """
    if analysis.get("status") == "error":
        return

    today = datetime.now().strftime("%Y-%m-%d")
    memory_id = _get_memory_id()
    if not memory_id:
        logger.warning(f"{ticker}: skipping store_episode — no memory ID")
        return

    client = _get_memory_client()

    direction = analysis.get("direction", "N/A")
    confidence = analysis.get("confidence", 0)
    confluence = analysis.get("confluence", "N/A")
    trade = analysis.get("trade", {})
    key_strikes = analysis.get("key_strikes", [])

    # Compact wall summary: "$580 call 45K, $570 put 38K"
    walls = ", ".join(
        f"${s.get('strike')} {s.get('type', '?')} {s.get('oi', 0):,}"
        for s in key_strikes[:5]
    )

    # Per-DTE one-liner: "30DTE:bullish 80% | 60DTE:neutral 55%"
    dte_analyses = analysis.get("dte_analyses", [])
    dte_line = " | ".join(
        f"{da.get('dte', '?')}DTE:{da.get('bias', '?')} {da.get('confidence', 0)}%"
        for da in dte_analyses
    )

    maxpain = ""
    if key_strikes:
        mp = next((s for s in key_strikes if s.get("type") == "max_pain"), None)
        if mp:
            maxpain = f" maxpain=${mp.get('strike')}"

    content = (
        f"{ticker} | {today} | {direction} {confidence}% | {confluence}\n"
        f"walls: {walls or 'none'}{maxpain}\n"
        f"price: ${trade.get('current_price', '?')} entry=${trade.get('entry', '?')} stop=${trade.get('stop', '?')} target=${trade.get('target', '?')}\n"
    )

    if dte_line:
        content += f"{dte_line}\n"

    try:
        t0 = time.time()
        client.create_event(
            memory_id=memory_id,
            actor_id=f"ticker/{ticker}",
            session_id=f"analysis-{today}",
            messages=[(content, "ASSISTANT")],
        )
        logger.info(f"{ticker}: stored episode ({direction} {confidence}%) ({time.time()-t0:.1f}s)")
    except Exception as e:
        logger.warning(f"{ticker}: store_episode failed - {e}")


def recall(ticker, query=None):
    """
    Retrieve historical context from Bedrock Memory semantic store.
    Uses MemoryClient.retrieve_memories (snake_case, returns list of dicts).
    """
    if query is None:
        query = f"{ticker} direction, confidence, key strike walls, price levels, per-DTE bias"

    memory_id = _get_memory_id()
    if not memory_id:
        logger.warning(f"{ticker}: skipping recall — no memory ID")
        return []

    client = _get_memory_client()
    facts = []

    try:
        t0 = time.time()
        # MemoryClient.retrieve_memories: returns List[Dict]
        records = client.retrieve_memories(
            memory_id=memory_id,
            namespace=f"/summaries/ticker/{ticker}/",
            query=query,
            top_k=5,
        )

        for record in records:
            # Records have 'content' dict with 'text' key
            content = record.get("content", {})
            if isinstance(content, dict):
                text = content.get("text", "")
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)
            if text:
                facts.append(text)

        dur = time.time() - t0
        if facts:
            logger.info(f"{ticker}: recalled {len(facts)} facts ({dur:.1f}s)")
        else:
            logger.info(f"{ticker}: no facts found ({dur:.1f}s)")

    except Exception as e:
        logger.warning(f"{ticker}: recall failed - {e}")

    return facts
