"""
0DTE Trade Outcome Memory - Learn from past trade results via Bedrock AgentCore

Hook API (called from zero_dte_agent.py):
- before_swarm(query_with_context, has_position) → enriched query with past outcomes
- after_signal(signal) → captures ENTRY, records EXIT outcome (async)

Internal:
- capture_entry() → Redis snapshot
- record_outcome() → Bedrock Memory episode (background thread)
- recall_outcomes() → semantic search for similar past outcomes

Uses same MemoryClient patterns as oi/memory.py (snake_case params, tuples for messages).
"""

import json
import time
import logging
import threading
import boto3
from datetime import datetime
from zoneinfo import ZoneInfo
from bedrock_agentcore.memory import MemoryClient
from config.settings import AWS_REGION

logger = logging.getLogger(__name__)

MEMORY_NAME = "zero_dte_outcomes_v2"
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
    2. For each, try ID prefix match, then get_memory for name check
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
        logger.info(f"outcome_memory: list_memories returned {len(memories)} entries")

        for mem in memories:
            mid = mem.get("id", "")
            if not mid:
                continue

            if mid.startswith(MEMORY_NAME):
                _memory_id = mid
                logger.info(f"outcome_memory: found by ID prefix: {_memory_id}")
                return _memory_id

            try:
                detail = control.get_memory(memoryId=mid)
                mem_detail = detail.get("memory", detail)
                name = mem_detail.get("name", "")
                if name == MEMORY_NAME:
                    _memory_id = mid
                    logger.info(f"outcome_memory: found by name lookup: {_memory_id}")
                    return _memory_id
            except Exception:
                pass

    except Exception as e:
        logger.warning(f"outcome_memory: list_memories failed: {e}")

    # Step 2: Not found — create
    try:
        _memory_id = _create_memory()
        return _memory_id
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.warning(f"outcome_memory: '{MEMORY_NAME}' already exists, retrying list...")
            try:
                memories = client.list_memories()
                for mem in memories:
                    mid = mem.get("id", "")
                    if mid and mid.startswith(MEMORY_NAME):
                        _memory_id = mid
                        logger.info(f"outcome_memory: found on retry: {_memory_id}")
                        return _memory_id
            except Exception as e2:
                logger.error(f"outcome_memory: retry list failed: {e2}")
        else:
            logger.error(f"outcome_memory: failed to create memory: {e}")

    return _memory_id


def _create_memory():
    """One-time setup: create the memory store"""
    client = _get_memory_client()

    result = client.create_memory_and_wait(
        name=MEMORY_NAME,
        description="0DTE trade outcome memory - stores win/loss episodes with flow conditions for learning",
        strategies=[
            {"semanticMemoryStrategy": {
                "name": "trade_outcomes",
                "description": "0DTE trade outcomes: win/loss, flow conditions, exit reason",
                "namespaces": ["/facts/{actorId}/"],
            }},
            {"summaryMemoryStrategy": {
                "name": "trade_summaries",
                "description": "Rolling summaries of 0DTE trade outcomes per session",
                "namespaces": ["/summaries/{actorId}/{sessionId}/"],
            }},
        ],
        event_expiry_days=EVENT_EXPIRY_DAYS,
    )

    memory_id = result.get("id") or result.get("memoryId", "")
    logger.info(f"outcome_memory: created Bedrock Memory store: {memory_id}")
    return memory_id


# ── Entry Capture (Redis, temporary) ──────────────────────────────────

def capture_entry(signal, flow_text, reasoning=""):
    """Store entry snapshot in Redis — read back on EXIT to compute outcome."""
    from redis_stream import get_stream

    now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    snapshot = {
        "action": signal.get("action"),
        "conviction": signal.get("conviction"),
        "entry_price": signal.get("entry"),
        "stop": signal.get("stop"),
        "target": signal.get("target"),
        "entry_time": now_pt.strftime("%H:%M"),
        "flow": (flow_text or "")[:800],
        "reasoning": (reasoning or "")[:1000],
        "period": "morning" if now_pt.hour < 10 else "midday" if now_pt.hour < 12 else "afternoon",
    }

    get_stream().redis.setex("zero_dte:entry_snapshot", 3600, json.dumps(snapshot))
    logger.info(f"outcome: captured entry {signal.get('action')} ${signal.get('entry')} {signal.get('conviction')}")
    print(f"\n{'='*60}")
    print(f"OUTCOME MEMORY — ENTRY CAPTURED")
    print(f"{'='*60}")
    print(f"Action: {snapshot['action']} | Conviction: {snapshot['conviction']}")
    print(f"Entry: ${snapshot['entry_price']} | Stop: ${snapshot.get('stop')} | Target: ${snapshot.get('target')}")
    print(f"Time: {snapshot['entry_time']} ({snapshot['period']})")
    print(f"Flow: {(snapshot['flow'] or 'N/A')[:150]}...")
    print(f"Reasoning: {(snapshot['reasoning'] or 'N/A')[:150]}...")
    print(f"{'='*60}\n")


# ── Outcome Recording (Bedrock Memory, async) ────────────────────────

def record_outcome(exit_signal):
    """
    Compute outcome and store in Bedrock Memory.
    Called in a background thread — never blocks the main loop.
    """
    from redis_stream import get_stream

    try:
        raw = get_stream().redis.get("zero_dte:entry_snapshot")
        if not raw:
            logger.warning("outcome: no entry snapshot found, skipping")
            return

        entry = json.loads(raw)
        exit_price = exit_signal.get("price", 0)
        entry_price = entry.get("entry_price", 0)
        action = entry.get("action", "")

        if not exit_price or not entry_price or action not in ("CALL", "PUT"):
            logger.warning(f"outcome: incomplete data, skipping (action={action}, entry={entry_price}, exit={exit_price})")
            return

        # Compute result
        if action == "CALL":
            result = "WIN" if exit_price > entry_price else "LOSS"
        else:
            result = "WIN" if exit_price < entry_price else "LOSS"

        now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
        exit_time = now_pt.strftime("%H:%M")
        today = now_pt.strftime("%Y-%m-%d")

        # Episode: WHY was this decided + WHAT actually happened
        content = (
            f"OUTCOME: {result} — decided {action} but price went {'with' if result == 'WIN' else 'against'} the trade\n"
            f"Decision: {action} with {entry.get('conviction')} conviction at {entry.get('entry_time')} ({entry.get('period')})\n"
            f"Entry: ${entry_price} → Exit: ${exit_price} at {exit_time} on {today}\n"
            f"\n"
            f"COORDINATOR REASONING AT ENTRY:\n"
            f"{entry.get('reasoning', 'N/A')}\n"
            f"\n"
            f"ORDER FLOW AT ENTRY:\n"
            f"{entry.get('flow', 'N/A')}\n"
        )

        memory_id = _get_memory_id()
        if not memory_id:
            logger.warning("outcome: no memory ID, skipping store")
            return

        t0 = time.time()
        _get_memory_client().create_event(
            memory_id=memory_id,
            actor_id="trader/SPY",
            session_id=f"trades-{today}",
            messages=[(content, "USER")],
        )

        # Clear entry snapshot
        get_stream().redis.delete("zero_dte:entry_snapshot")

        dur = time.time() - t0
        logger.info(f"outcome: stored {result} {action} ${entry_price}→${exit_price} ({dur:.1f}s)")
        print(f"\n{'='*60}")
        print(f"OUTCOME MEMORY — EPISODE STORED")
        print(f"{'='*60}")
        print(f"Result: {result}")
        print(f"Trade: {action} ${entry_price} → ${exit_price}")
        print(f"Time: {entry.get('entry_time')} → {exit_time} ({entry.get('period')})")
        print(f"Conviction: {entry.get('conviction')}")
        print(f"Bedrock write: {dur:.1f}s")
        print(f"\nStored episode:")
        print(f"---")
        print(content[:500])
        print(f"---")
        print(f"{'='*60}\n")

    except Exception as e:
        logger.warning(f"outcome: record_outcome failed - {e}")


# ── Outcome Recall (Bedrock Memory, synchronous) ─────────────────────

def recall_outcomes(flow_text, top_k=5):
    """
    Recall past trade outcomes matching current flow conditions.
    Called synchronously — only when flat (no position), before ENTRY decisions.
    """
    memory_id = _get_memory_id()
    if not memory_id:
        logger.warning("outcome: no memory ID, skipping recall")
        return []

    query = f"trade outcomes when flow was {flow_text[:200]}"
    facts = []

    try:
        t0 = time.time()
        records = _get_memory_client().retrieve_memories(
            memory_id=memory_id,
            namespace="/summaries/trader/SPY/",
            query=query,
            top_k=top_k,
        )

        for record in records:
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
            logger.info(f"outcome: recalled {len(facts)} past outcomes ({dur:.1f}s)")
            print(f"\n{'='*60}")
            print(f" OUTCOME MEMORY — RECALLED {len(facts)} PAST OUTCOMES")
            print(f"{'='*60}")
            print(f"Query: {query}")
            print(f"Bedrock read: {dur:.1f}s")
            for i, fact in enumerate(facts, 1):
                print(f"\n--- Outcome {i} ---")
                print(fact[:300])
            print(f"\n{'='*60}\n")
        else:
            logger.info(f"outcome: no past outcomes found ({dur:.1f}s)")
            print(f"\n OUTCOME MEMORY — no past outcomes found ({dur:.1f}s)\n")

    except Exception as e:
        logger.warning(f"outcome: recall failed - {e}")

    return facts


# ── Hook API (called from zero_dte_agent.py) ──────────────────────────

def _get_latest_flow_text():
    """Get the most recent order flow text from Redis history."""
    from redis_stream import get_stream
    try:
        stream = get_stream()
        events = stream.redis.lrange("zero_dte:history", 0, 10)
        for event_json in events:
            event = json.loads(event_json)
            if event.get("type") == "FLOW_RESPONSE":
                return event.get("content", "")[:800]
    except Exception:
        pass
    return ""


def before_swarm(query_with_context, has_position):
    """
    Pre-swarm hook: recall past outcomes when flat (no position).
    Returns enriched query_with_context.
    """
    if has_position:
        return query_with_context

    try:
        flow_text = _get_latest_flow_text()
        if flow_text:
            outcomes = recall_outcomes(flow_text)
            if outcomes:
                outcomes_str = "\n".join(f"- {o}" for o in outcomes)
                query_with_context += f"\n\n[PAST TRADE OUTCOMES in similar conditions:\n{outcomes_str}]"
                logger.info(f"outcome: injected {len(outcomes)} past outcomes into context")
    except Exception as e:
        logger.warning(f"outcome: before_swarm failed - {e}")

    return query_with_context


def after_signal(signal, response_text=""):
    """
    Post-signal hook: capture ENTRY snapshot (with reasoning), record EXIT outcome (async).
    """
    if not signal:
        return

    # On ENTRY — capture snapshot + coordinator reasoning to Redis
    if signal.get("signal") == "ENTRY":
        try:
            flow_text = _get_latest_flow_text()
            capture_entry(signal, flow_text or "", reasoning=response_text)
        except Exception as e:
            logger.warning(f"outcome: entry capture failed - {e}")

    # On EXIT — record outcome in background thread
    if signal.get("signal") == "EXIT" or signal.get("action") == "EXIT":
        threading.Thread(target=record_outcome, args=(signal,), daemon=True).start()
