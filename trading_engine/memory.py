"""
Bedrock AgentCore v2 memory — the learning brain.

Uses MemoryClient SDK (snake_case) for retrieval + boto3 data plane for writes.
Separate store from v1 (zero_dte_outcomes_v2).

Two namespaces (both semanticMemoryStrategy → both use /facts/ prefix):
- /facts/trader/SPY/         — Trade outcomes + missed opportunities
- /facts/patterns/SPY/       — Extracted rules from daily consolidation

CRITICAL DESIGN DECISION:
- WRITES use batch_create_memory_records (boto3 data plane, camelCase)
  Records are IMMEDIATELY searchable — no 60s async extraction delay.
- READS use retrieve_memories (MemoryClient SDK, snake_case)
  Semantic search over records in the target namespace.
- create_event is NOT used — it triggers async extraction that takes 60+ seconds,
  meaning records were never available for the next cycle (10s interval).

Hash-cached recall: only queries AgentCore when classified labels actually change.
"""

import json
import time
import logging
import threading
import uuid
import boto3
from datetime import datetime
from zoneinfo import ZoneInfo

from bedrock_agentcore.memory import MemoryClient
from config.settings import AWS_REGION, ENGINE_MEMORY_NAME

logger = logging.getLogger(__name__)


class MemoryStore:
    """Encapsulates all Bedrock AgentCore state — no module-level globals."""

    def __init__(self):
        self._memory_client = None
        self._control_client = None
        self._data_client = None
        self._memory_id = None
        self._last_recall_hash = None
        self._last_recall_result: list[str] = []

    @property
    def memory_client(self):
        """MemoryClient SDK — snake_case params, for retrieval."""
        if self._memory_client is None:
            self._memory_client = MemoryClient(region_name=AWS_REGION)
        return self._memory_client

    @property
    def control_client(self):
        """boto3 control plane — camelCase, for get_memory to resolve name→ID."""
        if self._control_client is None:
            self._control_client = boto3.client("bedrock-agentcore-control", region_name=AWS_REGION)
        return self._control_client

    @property
    def data_client(self):
        """boto3 data plane — camelCase, for batch_create_memory_records (immediate writes)."""
        if self._data_client is None:
            self._data_client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)
        return self._data_client

    def get_or_create_memory(self) -> str:
        """
        Find or create the 'zero_dte_v2' memory store. Returns memoryId.

        Uses create_or_get_memory (idempotent) — returns existing if name matches,
        creates if not found. Falls back to list + name lookup if that fails.
        """
        if self._memory_id is not None:
            return self._memory_id

        client = self.memory_client

        # Try idempotent create_or_get_memory first
        try:
            result = client.create_or_get_memory(
                name=ENGINE_MEMORY_NAME,
                description="0DTE v2 trading engine — trade outcomes + learned patterns",
                strategies=[
                    {"semanticMemoryStrategy": {
                        "name": "trade_memory",
                        "description": "Trade outcomes, missed opportunities, and learned patterns",
                        "namespaces": ["/facts/{actorId}/", "/facts/patterns/{actorId}/"],
                    }},
                ],
                event_expiry_days=30,
            )
            self._memory_id = result.get("id") or result.get("memoryId", "")
            logger.info(f"v2 memory: store ready: {self._memory_id}")
            print(f"[v2] Memory store ready: {self._memory_id[:20]}...")
            return self._memory_id

        except Exception as e:
            logger.warning(f"v2 memory: create_or_get_memory failed: {e}, falling back to list")

        # Fallback: list + name lookup via control plane
        try:
            control = self.control_client
            memories = client.list_memories()
            logger.info(f"v2 memory: list_memories returned {len(memories)} entries")

            for mem in memories:
                mid = mem.get("id", "")
                if not mid:
                    continue
                if mid.startswith(ENGINE_MEMORY_NAME):
                    self._memory_id = mid
                    logger.info(f"v2 memory: found by ID prefix: {self._memory_id}")
                    return self._memory_id
                try:
                    detail = control.get_memory(memoryId=mid)
                    mem_detail = detail.get("memory", detail)
                    name = mem_detail.get("name", "")
                    if name == ENGINE_MEMORY_NAME:
                        self._memory_id = mid
                        logger.info(f"v2 memory: found by name lookup: {self._memory_id}")
                        return self._memory_id
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"v2 memory: fallback list failed: {e}")

        return self._memory_id

    def _write_record(self, namespace: str, content: str, record_id: str = None):
        """
        Write a memory record DIRECTLY via batch_create_memory_records.

        Records are immediately searchable — no async extraction delay.
        This bypasses the create_event → extraction pipeline which takes 60+ seconds.
        """
        memory_id = self.get_or_create_memory()
        if not memory_id:
            logger.warning("v2 memory: no memory_id, skipping write")
            return

        if not record_id:
            record_id = str(uuid.uuid4())

        try:
            t0 = time.time()
            self.data_client.batch_create_memory_records(
                memoryId=memory_id,
                records=[{
                    "requestIdentifier": record_id,
                    "namespaces": [namespace],
                    "content": {"text": content},
                    "timestamp": datetime.now(ZoneInfo("America/Los_Angeles")),
                }],
            )
            dur = time.time() - t0
            logger.info(f"v2 memory: wrote record to {namespace} ({dur:.1f}s)")
            return dur
        except Exception as e:
            logger.error(f"v2 memory: batch_create_memory_records failed: {e}")
            # Fallback: try create_event (async, but at least stores the data)
            try:
                logger.info("v2 memory: falling back to create_event")
                today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
                # Extract actor_id from namespace: /facts/trader/SPY/ → trader/SPY
                parts = namespace.strip("/").split("/")
                actor_id = "/".join(parts[1:]) if len(parts) > 1 else "trader/SPY"
                self.memory_client.create_event(
                    memory_id=memory_id,
                    actor_id=actor_id,
                    session_id=f"trades-{today}",
                    messages=[(content, "ASSISTANT")],
                )
                logger.info("v2 memory: fallback create_event succeeded (async extraction, 60s+ delay)")
            except Exception as e2:
                logger.error(f"v2 memory: fallback create_event also failed: {e2}")
            return None

    def _extract_text(self, record: dict) -> str:
        """Extract text content from a memory record (handles both dict and str content)."""
        content = record.get("content", {})
        if isinstance(content, dict):
            return content.get("text", "")
        elif isinstance(content, str):
            return content
        return str(content) if content else ""

    def recall(self, market_state) -> list[str]:
        """
        Hash-cached recall: only query AgentCore when classified labels change.
        Returns list of text strings (past outcomes + learned rules).
        """
        current_hash = market_state.labels_hash()
        if current_hash == self._last_recall_hash and self._last_recall_result:
            return self._last_recall_result

        memory_id = self.get_or_create_memory()
        if not memory_id:
            return []

        client = self.memory_client
        search_text = market_state.to_search_text()
        results = []

        # Search trade outcomes
        try:
            t0 = time.time()
            records = client.retrieve_memories(
                memory_id=memory_id,
                namespace="/facts/trader/SPY/",
                query=search_text,
                top_k=5,
            )
            for record in records:
                text = self._extract_text(record)
                if text:
                    results.append(text)
            dur = time.time() - t0
            logger.info(f"v2 memory: recalled {len(results)} outcomes ({dur:.1f}s)")
        except Exception as e:
            logger.warning(f"v2 memory: outcome recall failed: {e}")

        # Search learned patterns
        # Namespace: /facts/patterns/SPY/ (strategy /facts/patterns/{actorId}/ with actorId=SPY)
        try:
            t0 = time.time()
            records = client.retrieve_memories(
                memory_id=memory_id,
                namespace="/facts/patterns/SPY/",
                query=search_text,
                top_k=10,
            )
            for record in records:
                text = self._extract_text(record)
                if text:
                    results.append(f"[LEARNED RULE] {text}")
            dur = time.time() - t0
            logger.info(f"v2 memory: recalled {len(results)} total incl patterns ({dur:.1f}s)")
        except Exception as e:
            logger.warning(f"v2 memory: pattern recall failed: {e}")

        self._last_recall_hash = current_hash
        self._last_recall_result = results

        if results:
            print(f"\n{'='*60}")
            print(f" V2 MEMORY — RECALLED {len(results)} ITEMS")
            print(f"{'='*60}")
            for i, r in enumerate(results[:5], 1):
                print(f"  {i}. {r[:200]}")
            print(f"{'='*60}\n")

        return results

    def record_entry(self, signal: dict, market_state):
        """Snapshot entry state to Redis — read back on EXIT to compute outcome."""
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

        now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
        snapshot = {
            "action": signal.get("action"),
            "conviction": signal.get("conviction"),
            "entry_price": signal.get("entry"),
            "stop": signal.get("stop"),
            "target": signal.get("target"),
            "entry_time": now_pt.strftime("%H:%M"),
            "session": market_state.session,
            "labels": market_state.to_prompt_text(),
            "labels_hash": market_state.labels_hash(),
        }

        r.setex("zero_dte_v2:entry_snapshot", 3600, json.dumps(snapshot))
        logger.info(f"v2 memory: entry captured {signal.get('action')} ${signal.get('entry')}")
        print(f"\n{'='*60}")
        print(f" V2 ENTRY CAPTURED")
        print(f"{'='*60}")
        print(f"  {snapshot['action']} @ ${snapshot['entry_price']} | {snapshot['conviction']}")
        print(f"  Labels: {snapshot['labels'][:150]}")
        print(f"{'='*60}\n")

    def record_outcome(self, exit_signal: dict):
        """
        Compute WIN/LOSS, write DIRECTLY to AgentCore (immediately searchable).
        Uses batch_create_memory_records (boto3 data plane) — NOT create_event.
        """
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

        try:
            raw = r.get("zero_dte_v2:entry_snapshot")
            if not raw:
                logger.warning("v2 memory: no entry snapshot, skipping outcome")
                return

            entry = json.loads(raw)
            exit_price = exit_signal.get("price", 0)
            entry_price = entry.get("entry_price", 0)
            action = entry.get("action", "")

            if not exit_price or not entry_price or action not in ("CALL", "PUT"):
                return

            result = "WIN" if (
                (action == "CALL" and exit_price > entry_price) or
                (action == "PUT" and exit_price < entry_price)
            ) else "LOSS"

            now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
            today = now_pt.strftime("%Y-%m-%d")

            pnl = abs(exit_price - entry_price)
            pnl_sign = "+" if result == "WIN" else "-"
            content = (
                f"OUTCOME: {result} {pnl_sign}${pnl:.2f} | {action} | "
                f"{entry.get('conviction')} conviction | {entry.get('session')}\n"
                f"{entry.get('labels', '')}\n"
                f"Entry: ${entry_price} at {entry.get('entry_time')} → "
                f"Exit: ${exit_price} at {now_pt.strftime('%H:%M')} on {today}"
            )

            record_id = f"outcome-{action}-{today}-{now_pt.strftime('%H%M')}"
            dur = self._write_record("/facts/trader/SPY/", content, record_id)

            # Clear entry snapshot
            r.delete("zero_dte_v2:entry_snapshot")

            dur_str = f"{dur:.1f}s" if dur else "fallback"
            logger.info(f"v2 memory: stored {result} {action} ${entry_price}→${exit_price} ({dur_str})")
            print(f"\n{'='*60}")
            print(f" V2 OUTCOME STORED: {result}")
            print(f"{'='*60}")
            print(f"  {action} ${entry_price} → ${exit_price} ({pnl_sign}${pnl:.2f})")
            print(f"  Bedrock write: {dur_str}")
            print(f"{'='*60}\n")

        except Exception as e:
            logger.warning(f"v2 memory: record_outcome failed: {e}")

    def record_outcome_async(self, exit_signal: dict):
        """Fire-and-forget outcome recording in background thread."""
        threading.Thread(target=self.record_outcome, args=(exit_signal,), daemon=True).start()

    def record_wait_outcome(self, wait_snap: dict, current_price: float, missed_action: str, move: float):
        """
        Store missed opportunity DIRECTLY in AgentCore (immediately searchable).

        Content format:
        MISSED_OPPORTUNITY: WAIT → should have been CALL | Flow LEAN_BUYING 1.3:1 | ...
        Price at WAIT: $582.00 → moved to $583.50 (+$1.50)
        """
        try:
            now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
            today = now_pt.strftime("%Y-%m-%d")
            abs_move = abs(move)
            direction_word = "up" if move > 0 else "down"
            wait_price = wait_snap.get("price", 0)

            content = (
                f"MISSED_OPPORTUNITY: WAIT → should have been {missed_action} | "
                f"Flow {wait_snap.get('flow_direction')} {wait_snap.get('flow_ratio')}:1 | "
                f"{wait_snap.get('session')}\n"
                f"{wait_snap.get('labels', '')}\n"
                f"Price at WAIT: ${wait_price:.2f} at {wait_snap.get('time')} → "
                f"moved {direction_word} ${abs_move:.2f} to ${current_price:.2f} on {today}"
            )

            record_id = f"missed-{missed_action}-{today}-{now_pt.strftime('%H%M%S')}"
            dur = self._write_record("/facts/trader/SPY/", content, record_id)

            dur_str = f"{dur:.1f}s" if dur else "fallback"
            logger.info(f"v2 memory: stored MISSED {missed_action} +${abs_move:.2f} ({dur_str})")
            print(f"\n{'='*60}")
            print(f" V2 MISSED OPPORTUNITY STORED")
            print(f"{'='*60}")
            print(f"  WAIT → should have been {missed_action}")
            print(f"  ${wait_price:.2f} → ${current_price:.2f} ({direction_word} ${abs_move:.2f})")
            print(f"  Bedrock write: {dur_str}")
            print(f"{'='*60}\n")

        except Exception as e:
            logger.warning(f"v2 memory: record_wait_outcome failed: {e}")

    def record_wait_outcome_async(self, wait_snap: dict, current_price: float, missed_action: str, move: float):
        """Fire-and-forget missed opportunity recording in background thread."""
        threading.Thread(
            target=self.record_wait_outcome,
            args=(wait_snap, current_price, missed_action, move),
            daemon=True,
        ).start()

    def store_patterns(self, patterns: list[str]):
        """
        Store extracted patterns from daily consolidation.
        Writes DIRECTLY to /facts/patterns/SPY/ (immediately searchable).

        actor_id="SPY" so namespace resolves to /facts/patterns/SPY/
        (NOT actor_id="patterns/SPY" which would create /facts/patterns/patterns/SPY/).
        """
        today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")

        stored = 0
        for i, pattern in enumerate(patterns):
            record_id = f"pattern-{today}-{i}"
            dur = self._write_record("/facts/patterns/SPY/", pattern, record_id)
            if dur is not None:
                stored += 1

        if stored:
            logger.info(f"v2 memory: stored {stored}/{len(patterns)} patterns")
            print(f"\n{'='*60}")
            print(f" V2 PATTERNS STORED: {stored} rules")
            print(f"{'='*60}")
            for p in patterns[:5]:
                print(f"  - {p[:120]}")
            print(f"{'='*60}\n")

    def get_todays_outcomes(self) -> list[str]:
        """Retrieve today's trade outcomes from AgentCore for consolidation."""
        memory_id = self.get_or_create_memory()
        if not memory_id:
            return []

        client = self.memory_client
        today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")

        try:
            records = client.retrieve_memories(
                memory_id=memory_id,
                namespace="/facts/trader/SPY/",
                query=f"OUTCOME trade on {today}",
                top_k=20,
            )
            outcomes = []
            for record in records:
                text = self._extract_text(record)
                if text and today in text:
                    outcomes.append(text)
            return outcomes
        except Exception as e:
            logger.warning(f"v2 memory: get_todays_outcomes failed: {e}")
            return []

    def verify_memory(self) -> dict:
        """
        Diagnostic: check what records actually exist in each namespace.
        Call this to debug recall issues.
        """
        memory_id = self.get_or_create_memory()
        if not memory_id:
            return {"error": "no memory_id"}

        result = {"memory_id": memory_id, "namespaces": {}}

        for ns in ["/facts/trader/SPY/", "/facts/patterns/SPY/"]:
            try:
                records = self.data_client.list_memory_records(
                    memoryId=memory_id,
                    namespace=ns,
                )
                summaries = records.get("memoryRecordSummaries", [])
                result["namespaces"][ns] = {
                    "count": len(summaries),
                    "samples": [
                        s.get("content", "")[:100] for s in summaries[:3]
                    ],
                }
            except Exception as e:
                result["namespaces"][ns] = {"error": str(e)}

        return result
