"""
Bedrock AgentCore v2 memory — the learning brain.

Separate store from v1 (zero_dte_outcomes_v2). Uses boto3 data plane (camelCase).

Two namespaces:
- /facts/trader/SPY/  — Episodic: individual trade outcomes with classified labels
- /facts/patterns/SPY/ — Semantic: extracted rules/anti-patterns from daily consolidation

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

from config.settings import AWS_REGION, ENGINE_MEMORY_NAME

logger = logging.getLogger(__name__)


class MemoryStore:
    """Encapsulates all Bedrock AgentCore state — no module-level globals."""

    def __init__(self):
        self._data_client = None
        self._control_client = None
        self._memory_id = None
        self._last_recall_hash = None
        self._last_recall_result: list[str] = []

    @property
    def data_client(self):
        """boto3 data plane — camelCase params."""
        if self._data_client is None:
            self._data_client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)
        return self._data_client

    @property
    def control_client(self):
        """boto3 control plane — for create/list/get memory stores."""
        if self._control_client is None:
            self._control_client = boto3.client("bedrock-agentcore-control", region_name=AWS_REGION)
        return self._control_client

    def get_or_create_memory(self) -> str:
        """
        Find or create the 'zero_dte_v2' memory store. Returns memoryId.

        Strategy:
        1. List memories via control plane
        2. Match by name
        3. If not found, create with semantic strategies
        """
        if self._memory_id is not None:
            return self._memory_id

        control = self.control_client

        # Step 1: List and find by name
        try:
            response = control.list_memories()
            memories = response.get("memories", response.get("memorySummaries", []))

            for mem in memories:
                name = mem.get("name", "")
                mid = mem.get("memoryId", mem.get("id", ""))
                if name == ENGINE_MEMORY_NAME and mid:
                    self._memory_id = mid
                    logger.info(f"v2 memory: found existing store: {self._memory_id}")
                    return self._memory_id
        except Exception as e:
            logger.warning(f"v2 memory: list_memories failed: {e}")

        # Step 2: Create
        try:
            response = control.create_memory(
                name=ENGINE_MEMORY_NAME,
                description="0DTE v2 trading engine — trade outcomes + learned patterns",
                strategies=[
                    {"semanticMemoryStrategy": {
                        "name": "trade_outcomes",
                        "description": "Individual 0DTE trade outcomes with classified market labels",
                        "namespaces": ["/facts/{actorId}/"],
                    }},
                    {"semanticMemoryStrategy": {
                        "name": "learned_patterns",
                        "description": "Extracted trading rules and anti-patterns from daily consolidation",
                        "namespaces": ["/facts/patterns/{actorId}/"],
                    }},
                ],
                eventExpiryDays=30,
            )
            self._memory_id = response.get("memoryId", response.get("id", ""))
            logger.info(f"v2 memory: created store: {self._memory_id}")

            # Wait for ACTIVE status
            for _ in range(30):
                time.sleep(2)
                status_resp = control.get_memory(memoryId=self._memory_id)
                mem_detail = status_resp.get("memory", status_resp)
                status = mem_detail.get("status", "")
                if status == "ACTIVE":
                    logger.info("v2 memory: store is ACTIVE")
                    break
            return self._memory_id

        except Exception as e:
            if "already exists" in str(e).lower() or "conflict" in str(e).lower():
                logger.warning("v2 memory: store already exists, retrying list...")
                try:
                    response = control.list_memories()
                    memories = response.get("memories", response.get("memorySummaries", []))
                    for mem in memories:
                        if mem.get("name") == ENGINE_MEMORY_NAME:
                            self._memory_id = mem.get("memoryId", mem.get("id", ""))
                            return self._memory_id
                except Exception as e2:
                    logger.error(f"v2 memory: retry list failed: {e2}")
            else:
                logger.error(f"v2 memory: create failed: {e}")

        return self._memory_id

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

        client = self.data_client
        search_text = market_state.to_search_text()
        results = []

        # Search trade outcomes
        try:
            t0 = time.time()
            response = client.retrieve_memory_records(
                memoryId=memory_id,
                namespace="/facts/trader/SPY/",
                searchCriteria={
                    "searchQuery": search_text,
                    "topK": 5,
                },
            )
            for record in response.get("memoryRecordSummaries", []):
                text = record.get("content", "")
                if isinstance(text, dict):
                    text = text.get("text", "")
                if text:
                    results.append(text)
            logger.info(f"v2 memory: recalled {len(results)} outcomes ({time.time()-t0:.1f}s)")
        except Exception as e:
            logger.warning(f"v2 memory: outcome recall failed: {e}")

        # Search learned patterns
        try:
            t0 = time.time()
            response = client.retrieve_memory_records(
                memoryId=memory_id,
                namespace="/facts/patterns/SPY/",
                searchCriteria={
                    "searchQuery": search_text,
                    "topK": 10,
                },
            )
            for record in response.get("memoryRecordSummaries", []):
                text = record.get("content", "")
                if isinstance(text, dict):
                    text = text.get("text", "")
                if text:
                    results.append(f"[LEARNED RULE] {text}")
            logger.info(f"v2 memory: recalled patterns ({time.time()-t0:.1f}s)")
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
        Compute WIN/LOSS, store to AgentCore with classified labels.
        Runs in background thread — never blocks the main loop.
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

            # Content with classified labels (not raw numbers)
            pnl = abs(exit_price - entry_price)
            pnl_sign = "+" if result == "WIN" else "-"
            content = (
                f"OUTCOME: {result} {pnl_sign}${pnl:.2f} | {action} | "
                f"{entry.get('conviction')} conviction | {entry.get('session')}\n"
                f"{entry.get('labels', '')}\n"
                f"Entry: ${entry_price} at {entry.get('entry_time')} → "
                f"Exit: ${exit_price} at {now_pt.strftime('%H:%M')} on {today}"
            )

            memory_id = self.get_or_create_memory()
            if not memory_id:
                return

            client = self.data_client
            t0 = time.time()

            client.create_event(
                memoryId=memory_id,
                actorId="trader/SPY",
                sessionId=f"trades-{today}",
                eventTimestamp=now_pt,
                payload=[{"content": {"text": content}}],
                metadata={"result": result, "action": action, "session": entry.get("session", "")},
            )

            # Clear entry snapshot
            r.delete("zero_dte_v2:entry_snapshot")

            dur = time.time() - t0
            logger.info(f"v2 memory: stored {result} {action} ${entry_price}→${exit_price} ({dur:.1f}s)")
            print(f"\n{'='*60}")
            print(f" V2 OUTCOME STORED: {result}")
            print(f"{'='*60}")
            print(f"  {action} ${entry_price} → ${exit_price} ({pnl_sign}${pnl:.2f})")
            print(f"  Bedrock write: {dur:.1f}s")
            print(f"{'='*60}\n")

        except Exception as e:
            logger.warning(f"v2 memory: record_outcome failed: {e}")

    def record_outcome_async(self, exit_signal: dict):
        """Fire-and-forget outcome recording in background thread."""
        threading.Thread(target=self.record_outcome, args=(exit_signal,), daemon=True).start()

    def store_patterns(self, patterns: list[str]):
        """Store extracted patterns from daily consolidation into /facts/patterns/SPY/."""
        memory_id = self.get_or_create_memory()
        if not memory_id:
            return

        client = self.data_client
        today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")

        records = []
        for i, pattern in enumerate(patterns):
            records.append({
                "requestIdentifier": f"pattern-{today}-{i}-{uuid.uuid4().hex[:8]}",
                "namespaces": ["/facts/patterns/SPY/"],
                "content": {"text": pattern},
                "timestamp": datetime.now(ZoneInfo("America/Los_Angeles")),
            })

        if not records:
            return

        try:
            t0 = time.time()
            client.batch_create_memory_records(
                memoryId=memory_id,
                records=records,
            )
            dur = time.time() - t0
            logger.info(f"v2 memory: stored {len(records)} patterns ({dur:.1f}s)")
            print(f"\n{'='*60}")
            print(f" V2 PATTERNS STORED: {len(records)} rules")
            print(f"{'='*60}")
            for p in patterns[:5]:
                print(f"  - {p[:120]}")
            print(f"{'='*60}\n")
        except Exception as e:
            logger.error(f"v2 memory: store_patterns failed: {e}")

    def get_todays_outcomes(self) -> list[str]:
        """Retrieve today's trade outcomes from AgentCore for consolidation."""
        memory_id = self.get_or_create_memory()
        if not memory_id:
            return []

        client = self.data_client
        today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")

        try:
            response = client.retrieve_memory_records(
                memoryId=memory_id,
                namespace="/facts/trader/SPY/",
                searchCriteria={
                    "searchQuery": f"OUTCOME trade on {today}",
                    "topK": 20,
                },
            )
            outcomes = []
            for record in response.get("memoryRecordSummaries", []):
                text = record.get("content", "")
                if isinstance(text, dict):
                    text = text.get("text", "")
                if text and today in text:
                    outcomes.append(text)
            return outcomes
        except Exception as e:
            logger.warning(f"v2 memory: get_todays_outcomes failed: {e}")
            return []
