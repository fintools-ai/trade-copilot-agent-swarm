"""
Bedrock AgentCore v2 memory — the learning brain.

Uses MemoryClient SDK (snake_case) — same proven pattern as oi/memory.py.
Separate store from v1 (zero_dte_outcomes_v2).

Two namespaces (both semanticMemoryStrategy → both use /facts/ prefix):
- /facts/trader/SPY/         — Trade outcomes with classified labels
- /facts/patterns/SPY/       — Extracted rules/anti-patterns from daily consolidation

Hash-cached recall: only queries AgentCore when classified labels actually change.
"""

import json
import time
import logging
import threading
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
        self._memory_id = None
        self._last_recall_hash = None
        self._last_recall_result: list[str] = []

    @property
    def memory_client(self):
        """MemoryClient SDK — snake_case params, handles events + retrieval."""
        if self._memory_client is None:
            self._memory_client = MemoryClient(region_name=AWS_REGION)
        return self._memory_client

    @property
    def control_client(self):
        """boto3 control plane — camelCase, for get_memory to resolve name→ID."""
        if self._control_client is None:
            self._control_client = boto3.client("bedrock-agentcore-control", region_name=AWS_REGION)
        return self._control_client

    def get_or_create_memory(self) -> str:
        """
        Find or create the 'zero_dte_v2' memory store. Returns memoryId.

        Strategy (matches oi/memory.py):
        1. list_memories via MemoryClient (returns dicts with 'id' key, NO 'name')
        2. For each, try ID prefix match, then get_memory for name check
        3. If not found, create with create_memory_and_wait
        4. If "already exists" error, retry listing
        """
        if self._memory_id is not None:
            return self._memory_id

        client = self.memory_client
        control = self.control_client

        # Step 1: List memories and resolve by name
        try:
            memories = client.list_memories()
            logger.info(f"v2 memory: list_memories returned {len(memories)} entries")

            for mem in memories:
                mid = mem.get("id", "")
                if not mid:
                    continue

                # ID prefix match (IDs often start with the name)
                if mid.startswith(ENGINE_MEMORY_NAME):
                    self._memory_id = mid
                    logger.info(f"v2 memory: found by ID prefix: {self._memory_id}")
                    return self._memory_id

                # Try get_memory via control plane to check actual name
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
            logger.warning(f"v2 memory: list_memories failed: {e}")

        # Step 2: Not found — create
        try:
            result = client.create_memory_and_wait(
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
                event_expiry_days=30,
            )
            # MemoryClient returns "id", not "memoryId"
            self._memory_id = result.get("id") or result.get("memoryId", "")
            logger.info(f"v2 memory: created store: {self._memory_id}")
            return self._memory_id

        except Exception as e:
            if "already exists" in str(e).lower():
                logger.warning("v2 memory: store already exists, retrying list...")
                try:
                    memories = client.list_memories()
                    for mem in memories:
                        mid = mem.get("id", "")
                        if mid and mid.startswith(ENGINE_MEMORY_NAME):
                            self._memory_id = mid
                            logger.info(f"v2 memory: found on retry: {self._memory_id}")
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

        client = self.memory_client
        search_text = market_state.to_search_text()
        results = []

        # Search trade outcomes (facts namespace — matches semanticMemoryStrategy config)
        try:
            t0 = time.time()
            records = client.retrieve_memories(
                memory_id=memory_id,
                namespace="/facts/trader/SPY/",
                query=search_text,
                top_k=5,
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
                    results.append(text)
            logger.info(f"v2 memory: recalled {len(results)} outcomes ({time.time()-t0:.1f}s)")
        except Exception as e:
            logger.warning(f"v2 memory: outcome recall failed: {e}")

        # Search learned patterns (facts/patterns namespace — matches 2nd semanticMemoryStrategy)
        try:
            t0 = time.time()
            records = client.retrieve_memories(
                memory_id=memory_id,
                namespace="/facts/patterns/SPY/",
                query=search_text,
                top_k=10,
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
        Uses MemoryClient.create_event (snake_case, messages as tuples).
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

            t0 = time.time()

            # MemoryClient: snake_case, messages as tuples
            self.memory_client.create_event(
                memory_id=memory_id,
                actor_id="trader/SPY",
                session_id=f"trades-{today}",
                messages=[(content, "ASSISTANT")],
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

    def record_wait_outcome(self, wait_snap: dict, current_price: float, missed_action: str, move: float):
        """
        Store missed opportunity in AgentCore — teaches the LLM what happens when it WAITs
        on directional flow.

        Content format:
        MISSED_OPPORTUNITY: WAIT → should have been CALL | Flow LEAN_BUYING 1.3:1 | ...
        Price at WAIT: $582.00 → moved to $583.50 (+$1.50) in ~30s
        """
        try:
            memory_id = self.get_or_create_memory()
            if not memory_id:
                return

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

            t0 = time.time()
            self.memory_client.create_event(
                memory_id=memory_id,
                actor_id="trader/SPY",
                session_id=f"trades-{today}",
                messages=[(content, "ASSISTANT")],
            )
            dur = time.time() - t0

            logger.info(f"v2 memory: stored MISSED {missed_action} +${abs_move:.2f} ({dur:.1f}s)")
            print(f"\n{'='*60}")
            print(f" V2 MISSED OPPORTUNITY STORED")
            print(f"{'='*60}")
            print(f"  WAIT → should have been {missed_action}")
            print(f"  ${wait_price:.2f} → ${current_price:.2f} ({direction_word} ${abs_move:.2f})")
            print(f"  Bedrock write: {dur:.1f}s")
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
        Uses create_event per pattern (MemoryClient auto-extracts into /facts/).
        """
        memory_id = self.get_or_create_memory()
        if not memory_id:
            return

        client = self.memory_client
        today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")

        stored = 0
        for i, pattern in enumerate(patterns):
            try:
                client.create_event(
                    memory_id=memory_id,
                    actor_id="patterns/SPY",
                    session_id=f"consolidation-{today}",
                    messages=[(pattern, "ASSISTANT")],
                )
                stored += 1
            except Exception as e:
                logger.warning(f"v2 memory: store pattern {i} failed: {e}")

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
                content = record.get("content", {})
                if isinstance(content, dict):
                    text = content.get("text", "")
                elif isinstance(content, str):
                    text = content
                else:
                    text = str(content)
                if text and today in text:
                    outcomes.append(text)
            return outcomes
        except Exception as e:
            logger.warning(f"v2 memory: get_todays_outcomes failed: {e}")
            return []
