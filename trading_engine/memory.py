"""
Bedrock AgentCore v2 memory — the learning brain.

WRITES: batch_create_memory_records (boto3 data plane, camelCase)
  - Records stored EXACTLY as written — no LLM extraction or rewriting
  - Immediately searchable via semantic embeddings
  - Each trade outcome / missed opportunity / post-trade lesson is a separate record

READS: retrieve_memories (MemoryClient SDK, snake_case)
  - Semantic search over records in /facts/trader/SPY/
  - Hash-cached: only queries AgentCore when classified labels change

Namespace: /facts/trader/SPY/ — all record types in one namespace
  - OUTCOME: WIN/LOSS trade results with flow metrics and labels
  - MISSED_OPPORTUNITY: WAITs where price moved in flow direction
  - POST_EXIT_ANALYSIS: 5-min post-exit LLM analysis (early/good/neutral)
  - POST_WAIT_ANALYSIS: 5-min post-wait LLM analysis (missed/good/neutral)
"""

import json
import time
import logging
import threading
import uuid
import boto3
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from bedrock_agentcore.memory import MemoryClient
from config.settings import AWS_REGION, ENGINE_MEMORY_NAME

logger = logging.getLogger(__name__)

# ── Structured Header Schema ──────────────────────────────────────────

VALID_TYPES = {"OUTCOME", "MISSED_OPPORTUNITY", "POST_EXIT_ANALYSIS", "POST_WAIT_ANALYSIS"}
VALID_RESULTS = {"WIN", "LOSS"}
VALID_ACTIONS = {"CALL", "PUT"}
VALID_CONVICTIONS = {"HIGH", "MED", "LOW"}
VALID_REGIMES = {"TREND_CONTINUATION", "MEAN_REVERSION", "UNKNOWN"}
VALID_SESSIONS = {"morning", "midday", "afternoon"}
VALID_FLOW_DIRS = {
    "STRONG_BUYING", "BUYING", "LEAN_BUYING", "MIXED",
    "LEAN_SELLING", "SELLING", "STRONG_SELLING",
}
VALID_EXIT_VERDICTS = {"EARLY_EXIT", "GOOD_EXIT", "NEUTRAL"}
VALID_WAIT_VERDICTS = {"MISSED_OPPORTUNITY", "GOOD_WAIT", "NEUTRAL"}

# Enum fields per record type — for validation
TYPE_ENUMS = {
    "OUTCOME": {"result": VALID_RESULTS, "action": VALID_ACTIONS, "conviction": VALID_CONVICTIONS,
                "regime": VALID_REGIMES, "session": VALID_SESSIONS, "flow_dir": VALID_FLOW_DIRS},
    "MISSED_OPPORTUNITY": {"missed_action": VALID_ACTIONS, "regime": VALID_REGIMES,
                           "session": VALID_SESSIONS, "flow_dir": VALID_FLOW_DIRS},
    "POST_EXIT_ANALYSIS": {"verdict": VALID_EXIT_VERDICTS, "action": VALID_ACTIONS,
                           "regime": VALID_REGIMES, "session": VALID_SESSIONS, "flow_dir": VALID_FLOW_DIRS},
    "POST_WAIT_ANALYSIS": {"verdict": VALID_WAIT_VERDICTS, "regime": VALID_REGIMES,
                           "session": VALID_SESSIONS, "flow_dir": VALID_FLOW_DIRS},
}

# Adjacency maps for soft scoring
FLOW_ADJACENCY = {
    "STRONG_BUYING": {"BUYING"},
    "BUYING": {"STRONG_BUYING", "LEAN_BUYING"},
    "LEAN_BUYING": {"BUYING", "MIXED"},
    "MIXED": {"LEAN_BUYING", "LEAN_SELLING"},
    "LEAN_SELLING": {"MIXED", "SELLING"},
    "SELLING": {"LEAN_SELLING", "STRONG_SELLING"},
    "STRONG_SELLING": {"SELLING"},
}

SESSION_ADJACENCY = {
    "morning": {"midday"},
    "midday": {"morning", "afternoon"},
    "afternoon": {"midday"},
}


@dataclass
class MemoryMetrics:
    records_written: int = 0
    records_rejected: int = 0
    recalls_total: int = 0
    recalls_filtered_out: int = 0
    wins: int = 0
    losses: int = 0
    missed: int = 0


class MemoryStore:
    """Encapsulates all Bedrock AgentCore state — no module-level globals."""

    def __init__(self):
        self._memory_client = None
        self._control_client = None
        self._data_client = None
        self._memory_id = None
        self._last_recall_hash = None
        self._last_recall_result: list[str] = []
        self._written_ids: set = set()
        self._metrics = MemoryMetrics()

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
                        "namespaces": ["/facts/{actorId}/"],
                    }},
                ],
                event_expiry_days=365,
            )
            self._memory_id = result.get("id") or result.get("memoryId", "")
            logger.info("[MEMORY] store ready: %s", self._memory_id[:20])
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
        Write directly via batch_create_memory_records (boto3 data plane).
        Records are immediately searchable — no async extraction, no LLM rewriting.
        Content is stored exactly as written. Validates structured headers before writing.
        """
        memory_id = self.get_or_create_memory()
        if not memory_id:
            logger.warning("[MEMORY] no memory_id, skipping write")
            return

        if not record_id:
            record_id = str(uuid.uuid4())

        # Validate structured header (if present)
        ok, reason = self._validate_record(content, record_id)
        if not ok:
            self._metrics.records_rejected += 1
            logger.warning("[MEMORY] REJECTED: %s (record_id=%s)", reason, record_id)
            return None

        try:
            t0 = time.time()
            self.data_client.batch_create_memory_records(
                memoryId=memory_id,
                records=[{
                    'requestIdentifier': record_id,
                    'namespaces': [namespace],
                    'content': {'text': content},
                    'timestamp': datetime.now(ZoneInfo("America/Los_Angeles")),
                }],
            )
            dur = time.time() - t0
            self._written_ids.add(record_id)
            self._metrics.records_written += 1
            logger.info("[MEMORY] wrote record to %s (%.1fs, immediate)", namespace, dur)
            return dur
        except Exception as e:
            logger.error("[MEMORY] batch_create_memory_records failed: %s", e)
            return None

    def _state_to_nlp(self, market_state) -> str:
        """Convert a MarketState to natural language for semantic search matching."""
        f = market_state.flow
        t = market_state.tech
        orb = market_state.orb_regime
        b = market_state.breadth

        # Flow
        flow_dir = f.direction.lower().replace("_", " ")
        vol_ratio = round(f.bid_vol_60 / max(f.ask_vol_60, 1), 2)
        flow_text = (
            f"{flow_dir} flow with {f.ratio}:1 ratio "
            f"({f.bid_lifts_60} lifts vs {f.bid_drops_60} drops, volume ratio {vol_ratio}:1), "
            f"{f.momentum.lower()} momentum"
        )

        # RSI
        rsi_text = f"RSI {t.rsi_state.lower()} at {t.rsi_value}"

        # VWAP
        vwap_map = {
            "ABOVE_2SD": "price extended above VWAP by 2 standard deviations",
            "ABOVE_1SD": "price above VWAP by 1 standard deviation",
            "AT_VWAP": "price near VWAP",
            "BELOW_1SD": "price below VWAP by 1 standard deviation",
            "BELOW_2SD": "price extended below VWAP by 2 standard deviations",
        }
        vwap_text = vwap_map.get(t.vwap_position, t.vwap_position)

        # Regime
        regime_text = orb.regime.lower().replace("_", " ")

        # Breadth
        breadth_text = f"Mag7 breadth {b.bias.lower().replace('_', ' ')} with {b.aligned_count}/7 aligned"

        return (
            f"{flow_text}. "
            f"{rsi_text}, {vwap_text}. "
            f"In {regime_text} regime during {market_state.session} session. "
            f"{breadth_text}."
        )

    def _flow_metrics_to_nlp(self, flow_metrics: dict) -> str:
        """Convert a stored flow_metrics dict to natural language."""
        ratio = flow_metrics.get("ratio", 0)
        lifts = flow_metrics.get("bid_lifts", 0)
        drops = flow_metrics.get("bid_drops", 0)
        vol_ratio = flow_metrics.get("vol_ratio", 0)

        if ratio >= 2.5:
            direction = "strong buying"
        elif ratio >= 1.5:
            direction = "buying"
        elif ratio >= 1.1:
            direction = "lean buying"
        elif ratio <= 0.4:
            direction = "strong selling"
        elif ratio <= 0.65:
            direction = "selling"
        elif ratio <= 0.9:
            direction = "lean selling"
        else:
            direction = "mixed"

        return (
            f"{direction} flow with {ratio}:1 ratio "
            f"({lifts} lifts vs {drops} drops, volume ratio {vol_ratio}:1)"
        )

    def _extract_text(self, record: dict) -> str:
        """Extract text content from a memory record (handles both dict and str content)."""
        content = record.get("content", {})
        if isinstance(content, dict):
            return content.get("text", "")
        elif isinstance(content, str):
            return content
        return str(content) if content else ""

    def _build_recall_queries(self, market_state) -> list[tuple[str, str]]:
        """Build targeted NLP queries from current market state.

        Returns list of (query_text, label) tuples for multi-query recall.
        Each query targets a different type of insight.
        """
        flow_dir = market_state.flow.direction
        session = market_state.session
        regime = market_state.orb_regime.regime

        # Determine implied direction from flow
        if "BUYING" in flow_dir:
            action = "CALL"
            flow_desc = f"{flow_dir.lower().replace('_', ' ')} flow"
        elif "SELLING" in flow_dir:
            action = "PUT"
            flow_desc = f"{flow_dir.lower().replace('_', ' ')} flow"
        else:
            action = None
            flow_desc = "mixed flow"

        queries = []

        if action:
            # Query 1: Past trade outcomes with similar entry conditions
            queries.append((
                f"{action} trade entries with {flow_desc} in {session} session {regime.lower().replace('_', ' ')} regime",
                "outcomes",
            ))
            # Query 2: Missed opportunities from WAITing on similar flow
            queries.append((
                f"missed opportunity WAIT with {flow_desc} in {session} that should have been {action}",
                "missed",
            ))
            # Query 3: Exit timing lessons for similar positions
            queries.append((
                f"post exit analysis lesson {action} position with {flow_desc} exit timing",
                "lessons",
            ))
        else:
            # MIXED flow — search for times MIXED was right vs wrong
            queries.append((
                f"WAIT on mixed flow in {session} session outcome",
                "outcomes",
            ))
            queries.append((
                f"post wait analysis mixed flow {session} good wait or missed opportunity",
                "missed",
            ))

        return queries

    def recall(self, market_state) -> list[str]:
        """
        Multi-query recall: 3 targeted NLP queries for diverse memory types.
        Hash-cached — only queries AgentCore when classified labels change.
        After semantic search, applies hard filters (regime, session) and soft scoring (flow, recency).
        Returns list of text strings (outcomes + missed opportunities + lessons).
        """
        current_hash = market_state.labels_hash()
        if current_hash == self._last_recall_hash and self._last_recall_result:
            return self._last_recall_result

        memory_id = self.get_or_create_memory()
        if not memory_id:
            return []

        self._metrics.recalls_total += 1

        client = self.memory_client
        queries = self._build_recall_queries(market_state)
        raw_results = []
        seen = set()  # Deduplicate across queries

        logger.info("[MEMORY] recall: %d queries for regime=%s session=%s flow=%s",
                    len(queries), market_state.orb_regime.regime, market_state.session,
                    market_state.flow.direction)

        MIN_RELEVANCE_SCORE = 0.3  # Drop records below this cosine similarity

        t0 = time.time()
        for query_text, label in queries:
            logger.info("[MEMORY]   [%s] %s", label, query_text)
            try:
                from config.settings import ENGINE_MEMORY_MAX_RESULTS
                records = client.retrieve_memories(
                    memory_id=memory_id,
                    namespace="/facts/trader/SPY/",
                    query=query_text,
                    top_k=ENGINE_MEMORY_MAX_RESULTS,
                )
                count = 0
                skipped = 0
                for record in records:
                    score = record.get("score", 1.0)
                    if score < MIN_RELEVANCE_SCORE:
                        skipped += 1
                        continue
                    text = self._extract_text(record)
                    if text and text not in seen:
                        seen.add(text)
                        raw_results.append(text)
                        count += 1
                logger.info("[MEMORY]   [%s] %d unique records (dropped %d below %.1f)",
                           label, count, skipped, MIN_RELEVANCE_SCORE)
            except Exception as e:
                logger.warning("[MEMORY]   [%s] failed: %s", label, e)

        dur = time.time() - t0
        pre_filter_count = len(raw_results)

        # Apply structured header filters
        results = self._filter_records(
            raw_results,
            regime=market_state.orb_regime.regime,
            session=market_state.session,
            flow_dir=market_state.flow.direction,
        )

        filtered_out = pre_filter_count - len(results)
        logger.info("[MEMORY] Recalled %d memories in %.1fs (raw=%d, filtered_out=%d)",
                    len(results), dur, pre_filter_count, filtered_out)

        self._last_recall_hash = current_hash
        self._last_recall_result = results

        if results:
            # Count memory types (check NLP body, not header)
            outcomes = sum(1 for r in results if "OUTCOME:" in r and "POST_" not in r)
            missed = sum(1 for r in results if "MISSED_OPPORTUNITY:" in r)
            post_exit = sum(1 for r in results if "POST_EXIT_ANALYSIS:" in r)
            post_wait = sum(1 for r in results if "POST_WAIT_ANALYSIS:" in r)

            logger.info("[MEMORY] Outcomes=%d Missed=%d PostExit=%d PostWait=%d",
                        outcomes, missed, post_exit, post_wait)
            for i, r in enumerate(results[:5], 1):
                logger.info("[MEMORY]   [%d] %s", i, r[:200])
            if len(results) > 5:
                logger.info("[MEMORY]   ... and %d more", len(results) - 5)

        return results

    def record_entry(self, signal: dict, market_state):
        """Snapshot entry state to Redis — read back on EXIT to compute outcome."""
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

        now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))

        # Extract raw order flow metrics for learning
        flow_metrics = {
            "bid_lifts": market_state.flow.bid_lifts_60,
            "bid_drops": market_state.flow.bid_drops_60,
            "ratio": round(market_state.flow.ratio, 2),
            "net": market_state.flow.net,
            "bid_vol": market_state.flow.bid_vol_60,
            "ask_vol": market_state.flow.ask_vol_60,
            "vol_ratio": round(market_state.flow.bid_vol_60 / max(market_state.flow.ask_vol_60, 1), 2),
        }

        snapshot = {
            "action": signal.get("action"),
            "conviction": signal.get("conviction"),
            "entry_price": signal.get("entry"),
            "stop": signal.get("stop"),
            "target": signal.get("target"),
            "entry_time": now_pt.strftime("%H:%M"),
            "session": market_state.session,
            "regime": market_state.orb_regime.regime,
            "labels": market_state.to_prompt_text(),
            "labels_hash": market_state.labels_hash(),
            "flow_metrics": flow_metrics,
            "entry_nlp": self._state_to_nlp(market_state),
        }

        r.setex("zero_dte_v2:entry_snapshot", 3600, json.dumps(snapshot))
        logger.info("[MEMORY] entry captured %s $%s | %s | regime=%s",
                    signal.get('action'), signal.get('entry'), signal.get('conviction'),
                    market_state.orb_regime.regime)
        logger.info("[MEMORY]   Flow: %s:1 (lifts=%d, drops=%d, vol=%s:1)",
                    flow_metrics['ratio'], flow_metrics['bid_lifts'],
                    flow_metrics['bid_drops'], flow_metrics['vol_ratio'])

    def _extract_exit_reason(self, response_text: str) -> str:
        """Extract the 'Why:' line from agent response for outcome recording."""
        for line in response_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("Why:"):
                return stripped[4:].strip()
        return ""

    def record_outcome(self, exit_signal: dict, exit_state=None, response_text: str = ""):
        """
        Compute WIN/LOSS with full context — entry AND exit conditions.
        Prepends structured JSON header, then NLP body for semantic search.
        Writes directly via batch_create_memory_records (immediately searchable).
        """
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

        try:
            raw = r.get("zero_dte_v2:entry_snapshot")
            if not raw:
                logger.warning("[MEMORY] no entry snapshot, skipping outcome")
                return

            entry = json.loads(raw)

            # Handle price - could be float or dict with 'current' key
            exit_price_raw = exit_signal.get("price", 0)
            if isinstance(exit_price_raw, dict):
                exit_price = exit_price_raw.get("current", 0)
            else:
                exit_price = exit_price_raw

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
            signed_pnl = pnl if result == "WIN" else -pnl

            # Track wins/losses
            if result == "WIN":
                self._metrics.wins += 1
            else:
                self._metrics.losses += 1

            # Calculate hold duration
            entry_time = entry.get("entry_time", "")
            exit_time = now_pt.strftime("%H:%M")
            duration = ""
            if entry_time:
                try:
                    e_h, e_m = map(int, entry_time.split(":"))
                    x_h, x_m = map(int, exit_time.split(":"))
                    mins = (x_h * 60 + x_m) - (e_h * 60 + e_m)
                    duration = f" ({mins}min hold)"
                except (ValueError, TypeError):
                    pass

            # NLP-friendly entry context
            entry_nlp = entry.get("entry_nlp", "")
            if not entry_nlp:
                # Fallback for snapshots taken before NLP was added
                flow_metrics = entry.get("flow_metrics", {})
                entry_nlp = f"{self._flow_metrics_to_nlp(flow_metrics)} during {entry.get('session', 'unknown')} session."

            # NLP-friendly exit context
            exit_nlp = ""
            if exit_state:
                exit_nlp = f"Exited when conditions shifted to {self._state_to_nlp(exit_state)}"

            # Agent's exit reasoning
            exit_reason = self._extract_exit_reason(response_text)

            # Derive structured fields from entry snapshot
            conviction = entry.get("conviction", "MED")
            session = entry.get("session", "morning")
            regime = entry.get("regime", "UNKNOWN")
            flow_metrics = entry.get("flow_metrics", {})
            flow_ratio = flow_metrics.get("ratio", 1.0)
            flow_dir = self._ratio_to_flow_dir(flow_ratio)

            # Build structured header (line 1)
            header = self._build_header(
                type="OUTCOME", result=result, action=action, conviction=conviction,
                regime=regime, session=session, flow_dir=flow_dir,
                flow_ratio=round(flow_ratio, 2), pnl=round(signed_pnl, 2), date=today,
            )

            # Build NLP body (drives semantic search)
            body_lines = [
                f"OUTCOME: {result} {pnl_sign}${pnl:.2f} | {action} | "
                f"{conviction} conviction | {session}",
                f"Entered on {entry_nlp}",
            ]
            if exit_nlp:
                body_lines.append(exit_nlp)
            if exit_reason:
                body_lines.append(f"Exit reason: {exit_reason}")
            body_lines.append(
                f"Held for {duration.strip(' ()')  if duration else '?'} from "
                f"${entry_price:.2f} to ${exit_price:.2f} on {today}."
            )

            content = header + "\n" + "\n".join(body_lines)

            record_id = f"outcome-{action}-{today}-{now_pt.strftime('%H%M')}"
            dur = self._write_record("/facts/trader/SPY/", content, record_id)

            # Clear entry snapshot
            r.delete("zero_dte_v2:entry_snapshot")

            dur_str = f"{dur:.1f}s" if dur else "rejected"
            logger.info("[MEMORY] stored %s %s $%.2f->$%.2f (%s%s) regime=%s (%s)",
                        result, action, entry_price, exit_price, pnl_sign, f"${pnl:.2f}",
                        regime, dur_str)

        except Exception as e:
            logger.warning("[MEMORY] record_outcome failed: %s", e)

    def record_outcome_async(self, exit_signal: dict, exit_state=None, response_text: str = ""):
        """Fire-and-forget outcome recording in background thread."""
        threading.Thread(
            target=self.record_outcome,
            args=(exit_signal, exit_state, response_text),
            daemon=True,
        ).start()

    def record_wait_outcome(self, wait_snap: dict, current_price: float, missed_action: str, move: float):
        """
        Store missed opportunity DIRECTLY in AgentCore (immediately searchable).
        Prepends structured JSON header, then NLP body for semantic search.
        """
        try:
            now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
            today = now_pt.strftime("%Y-%m-%d")
            abs_move = abs(move)
            direction_word = "up" if move > 0 else "down"
            wait_price = wait_snap.get("price", 0)

            # Derive structured fields from wait snapshot
            session = wait_snap.get("session", "morning")
            regime = wait_snap.get("regime", "UNKNOWN")
            flow_direction = wait_snap.get("flow_direction", "MIXED")
            flow_ratio = wait_snap.get("flow_ratio", 1.0)
            flow_dir = self._ratio_to_flow_dir(flow_ratio) if flow_direction == "MIXED" else flow_direction

            self._metrics.missed += 1

            # Build structured header (line 1)
            header = self._build_header(
                type="MISSED_OPPORTUNITY", missed_action=missed_action,
                regime=regime, session=session, flow_dir=flow_dir,
                flow_ratio=round(flow_ratio, 2), move=round(abs_move, 2), date=today,
            )

            # Use pre-computed NLP if available, else build from raw fields
            wait_nlp = wait_snap.get("wait_nlp", "")
            if not wait_nlp:
                flow_dir_text = flow_direction.lower().replace("_", " ")
                wait_nlp = f"{flow_dir_text} flow with {flow_ratio}:1 ratio during {session} session"

            # Build NLP body (drives semantic search)
            body = (
                f"MISSED_OPPORTUNITY: WAIT should have been {missed_action} | {session}\n"
                f"Said WAIT at ${wait_price:.2f} with {wait_nlp}.\n"
                f"Price moved {direction_word} ${abs_move:.2f} to ${current_price:.2f} on {today} at {wait_snap.get('time', '?')}. "
                f"Should have entered {missed_action}."
            )

            content = header + "\n" + body

            record_id = f"missed-{missed_action}-{today}-{now_pt.strftime('%H%M%S')}"
            dur = self._write_record("/facts/trader/SPY/", content, record_id)

            dur_str = f"{dur:.1f}s" if dur else "rejected"
            logger.info("[MEMORY] stored MISSED %s +$%.2f regime=%s (%s)",
                        missed_action, abs_move, regime, dur_str)

        except Exception as e:
            logger.warning("[MEMORY] record_wait_outcome failed: %s", e)

    def record_wait_outcome_async(self, wait_snap: dict, current_price: float, missed_action: str, move: float):
        """Fire-and-forget missed opportunity recording in background thread."""
        threading.Thread(
            target=self.record_wait_outcome,
            args=(wait_snap, current_price, missed_action, move),
            daemon=True,
        ).start()

    # ── Structured Header Helpers ─────────────────────────────────────

    def _build_header(self, **fields) -> str:
        """Build a JSON header line for a memory record. Always includes v=1."""
        header = {"v": 1, **fields}
        return json.dumps(header)

    def _parse_header(self, text: str) -> dict | None:
        """Parse structured header from first line. Returns None for legacy records."""
        if not text:
            return None
        first_line = text.split("\n", 1)[0]
        try:
            header = json.loads(first_line)
            if isinstance(header, dict) and "v" in header:
                return header
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _validate_record(self, content: str, record_id: str) -> tuple[bool, str]:
        """Validate a new record before writing. Returns (ok, reason)."""
        header = self._parse_header(content)
        if header is None:
            return True, ""  # Legacy record — skip validation

        record_type = header.get("type")
        if record_type not in VALID_TYPES:
            return False, f"unknown type: {record_type}"

        # Validate enum fields for this type
        enum_spec = TYPE_ENUMS.get(record_type, {})
        for field_name, valid_values in enum_spec.items():
            val = header.get(field_name)
            if val is not None and val not in valid_values:
                return False, f"invalid {field_name}: {val}"

        # Minimum move/pnl thresholds
        if record_type == "OUTCOME":
            pnl = abs(header.get("pnl", 0))
            if pnl < 0.05:
                return False, f"pnl too small: ${pnl:.2f}"
        elif record_type == "MISSED_OPPORTUNITY":
            move = abs(header.get("move", 0))
            if move < 0.30:
                return False, f"move too small: ${move:.2f}"

        # Flow ratio sanity
        flow_ratio = header.get("flow_ratio")
        if flow_ratio is not None and (flow_ratio <= 0 or flow_ratio >= 50):
            return False, f"flow_ratio out of range: {flow_ratio}"

        # Dedup within session
        if record_id and record_id in self._written_ids:
            return False, f"duplicate record_id: {record_id}"

        return True, ""

    def _ratio_to_flow_dir(self, ratio: float) -> str:
        """Convert a flow ratio to a flow direction label (same thresholds as classifier)."""
        if ratio >= 2.5:
            return "STRONG_BUYING"
        elif ratio >= 1.5:
            return "BUYING"
        elif ratio >= 1.10:
            return "LEAN_BUYING"
        elif ratio <= 0.4:
            return "STRONG_SELLING"
        elif ratio <= 0.65:
            return "SELLING"
        elif ratio <= 0.90:
            return "LEAN_SELLING"
        else:
            return "MIXED"

    def _filter_records(self, records: list[str], regime: str, session: str, flow_dir: str) -> list[str]:
        """Post-filter recalled records by regime/session, score by flow and recency."""
        today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
        scored = []
        filtered_out = 0

        for text in records:
            header = self._parse_header(text)

            if header is None:
                # Legacy record — pass through with penalty
                scored.append((0.7, text))
                continue

            # Hard filter: regime must match
            rec_regime = header.get("regime", "")
            if rec_regime and rec_regime != regime:
                filtered_out += 1
                continue

            # Hard filter: session must match or be adjacent
            rec_session = header.get("session", "")
            if rec_session and rec_session != session and rec_session not in SESSION_ADJACENCY.get(session, set()):
                filtered_out += 1
                continue

            # Soft scoring
            score = 1.0

            # Flow adjacency scoring
            rec_flow = header.get("flow_dir", "")
            if rec_flow:
                if rec_flow == flow_dir:
                    score *= 1.2
                elif rec_flow in FLOW_ADJACENCY.get(flow_dir, set()):
                    score *= 1.0
                else:
                    score *= 0.7

            # Recency boost — today's records
            rec_date = header.get("date", "")
            if rec_date == today:
                score *= 1.3

            scored.append((score, text))

        self._metrics.recalls_filtered_out += filtered_out

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored]

    def get_metrics(self) -> dict:
        """Return session metrics as a dict."""
        m = self._metrics
        return {
            "records_written": m.records_written,
            "records_rejected": m.records_rejected,
            "recalls_total": m.recalls_total,
            "recalls_filtered_out": m.recalls_filtered_out,
            "wins": m.wins,
            "losses": m.losses,
            "missed": m.missed,
        }

    def verify_memory(self) -> dict:
        """
        Diagnostic: check what records actually exist in each namespace.
        Call this to debug recall issues.
        """
        memory_id = self.get_or_create_memory()
        if not memory_id:
            return {"error": "no memory_id"}

        result = {"memory_id": memory_id, "namespaces": {}}

        for ns in ["/facts/trader/SPY/"]:
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
