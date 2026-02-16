"""
Trading Engine v2 — Code classifies, Claude synthesizes, Memory learns.

Single async loop: Fetch → Classify → Recall → Synthesize → Act

Entry point: TradingEngine().run()
"""

import json
import time
import asyncio
import logging
import subprocess
import sys
import signal
from datetime import datetime
from zoneinfo import ZoneInfo

import boto3
import redis
import requests
from strands import Agent
from strands.agent.conversation_manager import SummarizingConversationManager

from config.settings import (
    AWS_REGION,
    ENGINE_MODEL_ID,
    ENGINE_SCAN_INTERVAL,
    ENGINE_MONITOR_INTERVAL,
    CLASSIFIER_MODEL_ID,
    CLASSIFIER_ALWAYS_LLM,
)
from trading_engine.classifier import classify_all, classify_flow, _get_window, MarketState, FLOW_DIRECTIONS
from trading_engine.prompts import (
    SYSTEM_PROMPT, ENGINE_SUMMARIZATION_PROMPT,
    build_scan_prompt, build_monitor_prompt,
    FLOW_CLASSIFIER_SYSTEM, build_flow_classifier_prompt,
)
from trading_engine.tools import create_engine_tools
from trading_engine.memory import MemoryStore
from trading_engine.trade_logger import TradeLogger
from trading_engine.post_exit_analyzer import PostExitAnalyzer
from redis_stream import publish_event

logger = logging.getLogger(__name__)

ORDER_FLOW_URL = "http://localhost:8300/flow/all"
REDIS_KEY_SPY = "market:spy:data"
REDIS_KEY_MAG7 = "market:mag7:data"


class TradingEngine:
    """Single-loop 0DTE trading engine with learning memory."""

    def __init__(self):
        self.redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        self.bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        self.memory = MemoryStore()
        self.trade_logger = TradeLogger()
        self.post_exit_analyzer = PostExitAnalyzer(self.redis, self.bedrock, self.memory)
        self.position = None  # Active position dict or None
        self.cycle_count = 0
        self.poller_proc = None
        self._shutdown = asyncio.Event()

        # Strands Agent state
        self.agent = None  # Created once, persists all day with conversation history
        self._current_state = None  # Set before each agent call (MarketState)
        self._current_memories = None  # Set before each agent call (list[str])
        self._wait_streak = 0  # Consecutive WAIT count

    async def run(self):
        """Main loop: start poller, run cycles until market close, then consolidate."""
        pt_tz = ZoneInfo("America/Los_Angeles")

        # Wire up shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown.set)

        # Start market poller subprocess — inherit stdout so logs are visible
        self.poller_proc = subprocess.Popen(
            [sys.executable, "market_poller.py"],
        )
        logger.info("[STARTUP] Market poller started (pid %s)", self.poller_proc.pid)

        # Initialize memory store (background, don't block)
        asyncio.get_event_loop().run_in_executor(None, self.memory.get_or_create_memory)

        # Create Strands Agent (persistent conversation history)
        self._create_agent()

        # Wait for first poller data
        logger.info("[STARTUP] Waiting for market data...")
        for _ in range(30):
            if self.redis.exists(REDIS_KEY_SPY):
                break
            await asyncio.sleep(1)

        logger.info("[STARTUP] Agent created — model: %s, summarizing context manager", ENGINE_MODEL_ID)
        publish_event("ENGINE_STATUS", "v2 engine started", {"version": "v2", "agent": True})

        try:
            while not self._shutdown.is_set():
                # Stop after market close
                now_pt = datetime.now(pt_tz)
                if now_pt.hour >= 14:
                    logger.info("[STARTUP] Market closed (1PM PT) — stopping")
                    break

                # Wait for market open
                if now_pt.hour < 6 or (now_pt.hour == 6 and now_pt.minute < 30):
                    open_time = now_pt.replace(hour=6, minute=30, second=0)
                    wait_secs = (open_time - now_pt).total_seconds()
                    logger.info("[STARTUP] Market opens in %.0fmin, waiting...", wait_secs/60)
                    try:
                        await asyncio.wait_for(self._shutdown.wait(), timeout=min(wait_secs, 60))
                    except asyncio.TimeoutError:
                        pass
                    continue

                await self.cycle()

                interval = ENGINE_MONITOR_INTERVAL if self.position else ENGINE_SCAN_INTERVAL
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass

        except Exception as e:
            logger.error("[ERROR] Engine fatal: %s", e, exc_info=True)
        finally:
            self._cleanup()

    def _cleanup(self):
        """Stop poller and clean up."""
        if self.poller_proc:
            self.poller_proc.terminate()
            try:
                self.poller_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.poller_proc.kill()
            logger.info("[STARTUP] Market poller stopped")

    async def cycle(self):
        """Single engine cycle: Fetch → Classify → Recall → Synthesize → Act"""
        self.cycle_count += 1
        t0 = time.time()

        # 1. FETCH (parallel)
        flow_data, spy_data, mag7_data = await self._fetch_data()
        if not spy_data:
            logger.warning("[FETCH] No SPY data available, skipping cycle")
            return
        logger.info("[FETCH] flow=%s spy=%s mag7=%s",
                    "OK" if flow_data else "EMPTY",
                    "OK" if spy_data else "EMPTY",
                    "OK" if mag7_data else "EMPTY")

        # 0. EVALUATE WAIT OUTCOMES — did we miss a trade last cycle?
        current_price = spy_data.get("price", 0)
        if current_price:
            self._evaluate_wait_outcome(current_price)

        # 2. CLASSIFY (~1ms Python, +600ms if borderline → Haiku)
        state = classify_all(flow_data, spy_data, mag7_data)

        # Hybrid: Haiku classifies flow (always if flag set, otherwise only borderline)
        if CLASSIFIER_ALWAYS_LLM or state.flow.borderline:
            llm_direction = await self._classify_flow_llm(flow_data, "SPY")
            if llm_direction and llm_direction != state.flow.direction:
                logger.info("[HAIKU] Reclassified SPY flow: %s -> %s", state.flow.direction, llm_direction)
                state.flow.direction = llm_direction
                state.flow.borderline = False

        # Log classified labels
        haiku_tag = " [HAIKU]" if CLASSIFIER_ALWAYS_LLM or (not state.flow.borderline and state.flow.direction in ("LEAN_BUYING", "LEAN_SELLING", "BUYING", "SELLING") and 0.4 < state.flow.ratio < 2.5) else ""
        logger.info("═══ CYCLE #%d | %s ═══════════════════════════════════", self.cycle_count, state.timestamp)
        logger.info("[CLASSIFY] Regime: %s (ORB $%s = %.3f%%)", state.orb_regime.regime, state.orb_regime.range_dollars, state.orb_regime.range_pct)
        logger.info("[CLASSIFY] Flow: %s %.2f:1 net=%d %s%s", state.flow.direction, state.flow.ratio, state.flow.net, state.flow.momentum, haiku_tag)
        logger.info("[CLASSIFY] RSI: %s (%.1f) | VWAP: %s | EMA: %s", state.tech.rsi_state, state.tech.rsi_value, state.tech.vwap_position, state.tech.ema_cross)
        logger.info("[CLASSIFY] Breadth: %s (%d/7)%s", state.breadth.bias, state.breadth.aligned_count,
                    f" | Divergent: {', '.join(state.breadth.divergent_tickers)}" if state.breadth.divergent_tickers else "")

        # 3. RECALL (hash-cached)
        memories = await asyncio.get_event_loop().run_in_executor(None, self.memory.recall, state)

        # 4. STORE STATE + MEMORIES on engine instance (tools read these)
        self._current_state = state
        self._current_memories = memories
        
        # Log memory results
        if memories:
            has_post_exit = any("POST_EXIT_ANALYSIS" in m for m in memories)
            has_post_wait = any("POST_WAIT_ANALYSIS" in m for m in memories)
            has_missed = any("MISSED_OPPORTUNITY" in m for m in memories)
            has_outcome = any("OUTCOME" in m for m in memories)

            types = []
            if has_outcome: types.append("OUTCOME")
            if has_missed: types.append("MISSED")
            if has_post_exit: types.append("POST_EXIT")
            if has_post_wait: types.append("POST_WAIT")

            logger.info("[MEMORY] Recalled %d memories (types: %s) query=%s",
                        len(memories), ", ".join(types) or "none", state.to_search_text())
            for i, m in enumerate(memories, 1):
                logger.debug("[MEMORY]   %d. %s", i, m[:120])
        else:
            logger.info("[MEMORY] No memories recalled for current labels")

        # 5. BUILD PROMPT (data inline + wait streak warning)
        mode = "monitor" if self.position else "scan"
        if self.position:
            user_prompt = build_monitor_prompt(state, self.position, memories)
        else:
            user_prompt = build_scan_prompt(state, memories, wait_streak=self._wait_streak)

        logger.info("[AGENT] Calling %s | mode=%s | memories=%d | wait_streak=%d",
                    ENGINE_MODEL_ID, mode, len(memories or []), self._wait_streak)

        # 6. SYNTHESIZE (Strands Agent with conversation history)
        query_start_ts = time.time()
        response_text = await self._invoke_agent(user_prompt)
        latency = time.time() - query_start_ts

        if not response_text:
            logger.error("[AGENT] Empty response from agent after %.1fs", latency)
            return

        logger.info("[AGENT] Response (%.1fs):\n%s", latency, response_text)

        # 7. PARSE SIGNAL
        signal_data = self._parse_signal(response_text)
        signal_data["latency"] = round(latency, 1)

        action = signal_data.get("action")
        conviction = signal_data.get("conviction", "")

        # Check if parse fell back to default WAIT
        if signal_data.get("signal") is None and action == "WAIT" and conviction == "LOW":
            logger.warning("[SIGNAL] Parse fell back to default WAIT — no valid JSON found in response")

        # Log if decision was influenced by memory
        if memories and action != "WAIT":
            if "memory" in response_text.lower() or "learned" in response_text.lower() or "past" in response_text.lower():
                logger.info("[MEMORY] Decision %s %s was influenced by recalled memories", action, conviction)

        # 8. TRACK WAIT STREAK
        if signal_data.get("action") == "WAIT":
            self._wait_streak += 1
        else:
            self._wait_streak = 0

        # 9. ACT
        self._act(signal_data, state, response_text)

        logger.info("[SIGNAL] %s %s | entry=%s stop=%s target=%s | price=%s",
                    action, conviction,
                    signal_data.get("entry", "-"), signal_data.get("stop", "-"),
                    signal_data.get("target", "-"), signal_data.get("price", "-"))

        # 10. PUBLISH TO UI
        total = time.time() - t0

        # Structured events for v2 engine dashboard
        publish_event("V2_STATE", "", self._state_to_dict(state, spy_data))
        publish_event("V2_MEMORY", "", {
            "items": memories or [],
            "count": len(memories or []),
            "labels_hash": state.labels_hash(),
            "search_query": state.to_search_text(),
            "cache_hit": (state.labels_hash() == self.memory._last_recall_hash) if memories else False,
            "store_id": self.memory._memory_id or "",
        })
        if self.position:
            publish_event("V2_POSITION", "", {
                "active": True, **self.position,
            })

        # Legacy events (v1 terminal compatibility)
        publish_event(
            "AGENT_QUESTION",
            f"[v2] {mode} cycle #{self.cycle_count}",
            {"query_start_ts": query_start_ts},
        )
        publish_event(
            "SWARM_RESPONSE",
            response_text + f"\n\n---\n*[v2 Engine | {latency:.1f}s]*",
            {**signal_data, "cycle": self.cycle_count, "total_time": round(total, 1), "wait_streak": self._wait_streak},
        )

        streak_tag = f" | streak={self._wait_streak}" if self._wait_streak >= 2 else ""
        logger.info("═══ CYCLE #%d DONE | %.1fs total | %.1fs LLM%s ═══════════════",
                    self.cycle_count, total, latency, streak_tag)

    async def _fetch_data(self) -> tuple[dict, dict, dict]:
        """Fetch order flow, SPY data, and Mag7 data in parallel."""
        loop = asyncio.get_event_loop()

        async def fetch_flow():
            try:
                resp = await loop.run_in_executor(
                    None, lambda: requests.get(ORDER_FLOW_URL, timeout=5)
                )
                return resp.json() if resp.status_code == 200 else {}
            except Exception as e:
                logger.error("[FETCH] Order flow fetch failed: %s", e)
                return {}

        async def fetch_redis(key):
            try:
                raw = await loop.run_in_executor(None, self.redis.get, key)
                return json.loads(raw) if raw else {}
            except Exception as e:
                logger.error("[FETCH] Redis fetch %s failed: %s", key, e)
                return {}

        flow_data, spy_data, mag7_data = await asyncio.gather(
            fetch_flow(),
            fetch_redis(REDIS_KEY_SPY),
            fetch_redis(REDIS_KEY_MAG7),
        )

        return flow_data, spy_data, mag7_data

    def _create_agent(self):
        """Create the persistent Strands Agent with tools and summarizing context manager."""
        tools = create_engine_tools(self)
        self.agent = Agent(
            model=ENGINE_MODEL_ID,
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            conversation_manager=SummarizingConversationManager(
                summary_ratio=0.4,
                preserve_recent_messages=10,
                summarization_system_prompt=ENGINE_SUMMARIZATION_PROMPT,
            ),
        )
        logger.info("[STARTUP] Strands Agent created — model: %s, summarizing (ratio=0.4, preserve=10)", ENGINE_MODEL_ID)

    async def _invoke_agent(self, user_prompt: str) -> str:
        """Call the Strands Agent from the async event loop via run_in_executor.

        The agent has conversation history from previous cycles and tools
        to access market state, memory, and position. Data is also provided
        inline in the prompt so the agent usually makes 0 tool calls (single LLM round-trip).
        """
        loop = asyncio.get_event_loop()

        def _call():
            result = self.agent(user_prompt)
            return str(result)

        try:
            return await loop.run_in_executor(None, _call)
        except Exception as e:
            logger.error("[AGENT] LLM call failed: %s", e, exc_info=True)
            return ""

    async def _classify_flow_llm(self, flow_data: dict, symbol: str) -> str:
        """
        Tier 2: Call Haiku to classify borderline flow via Converse API with prompt caching.
        Returns direction string or None on failure.
        """
        loop = asyncio.get_event_loop()
        ticker_data = flow_data.get(symbol, {})
        w60 = _get_window(ticker_data, "60s")
        w5 = _get_window(ticker_data, "5s")

        def _invoke():
            user_text = build_flow_classifier_prompt(w60, w5)
            logger.info("[HAIKU] Classifying %s flow — input: %s", symbol, user_text.replace("\n", " "))
            response = self.bedrock.converse(
                modelId=CLASSIFIER_MODEL_ID,
                system=[
                    {"text": FLOW_CLASSIFIER_SYSTEM},
                    {"cachePoint": {"type": "default"}},
                ],
                messages=[
                    {"role": "user", "content": [{"text": user_text}]}
                ],
                outputConfig={
                    "textFormat": {
                        "type": "json_schema",
                        "structure": {
                            "jsonSchema": {
                                "schema": json.dumps({
                                    "type": "object",
                                    "properties": {
                                        "direction": {
                                            "type": "string",
                                            "enum": FLOW_DIRECTIONS,
                                        },
                                        "reasoning": {"type": "string"},
                                    },
                                    "required": ["direction", "reasoning"],
                                    "additionalProperties": False,
                                }),
                                "name": "flow_classification",
                                "description": "Classify order flow direction",
                            }
                        }
                    }
                },
                inferenceConfig={"maxTokens": 2000, "temperature": 0},
            )
            raw_text = response["output"]["message"]["content"][0]["text"]
            logger.info("[HAIKU] Raw response: %s", raw_text)
            parsed = json.loads(raw_text)
            return parsed["direction"]

        try:
            t0 = time.time()
            direction = await loop.run_in_executor(None, _invoke)
            dur = time.time() - t0
            logger.info("[HAIKU] Classified %s flow in %.1fs: %s", symbol, dur, direction)
            return direction
        except json.JSONDecodeError as e:
            logger.error("[HAIKU] JSON parse failed: %s", e)
            return None
        except Exception as e:
            logger.error("[HAIKU] Classification failed: %s", e, exc_info=True)
            return None

    def _state_to_dict(self, state: MarketState, spy_data: dict = None) -> dict:
        """Convert MarketState to a flat dict for SSE publishing."""
        d = {
            "price": state.price,
            "timestamp": state.timestamp,
            "session": state.session,
            # ORB Regime
            "regime": state.orb_regime.regime,
            "regime_confidence": state.orb_regime.confidence,
            "orb_range": state.orb_regime.range_dollars,
            "orb_pct": state.orb_regime.range_pct,
            "orb_direction": state.orb_regime.direction,
            # Flow
            "flow_direction": state.flow.direction,
            "flow_ratio": state.flow.ratio,
            "flow_net": state.flow.net,
            "flow_momentum": state.flow.momentum,
            "flow_borderline": state.flow.borderline,
            "flow_bid_lifts": state.flow.bid_lifts_60,
            "flow_bid_drops": state.flow.bid_drops_60,
            "flow_ask_lifts": state.flow.ask_lifts_60,
            "flow_ask_drops": state.flow.ask_drops_60,
            "flow_bid_vol": state.flow.bid_vol_60,
            "flow_ask_vol": state.flow.ask_vol_60,
            # Technicals
            "rsi_state": state.tech.rsi_state,
            "rsi_value": state.tech.rsi_value,
            "vwap_position": state.tech.vwap_position,
            "price_vs_vwap": state.tech.price_vs_vwap,
            "ema_cross": state.tech.ema_cross,
            "macd_state": state.tech.macd_state,
            "macd_histogram": state.tech.macd_histogram,
            "orb_status": state.tech.orb_status,
            # Breadth
            "breadth_bias": state.breadth.bias,
            "breadth_aligned": state.breadth.aligned_count,
            "breadth_divergent": state.breadth.divergent_tickers,
            "breadth_tickers": [
                {
                    "symbol": t.symbol,
                    "price_dir": t.price_direction,
                    "change_pct": t.change_pct,
                    "flow_dir": t.flow.direction,
                    "flow_ratio": t.flow.ratio,
                    "flow_net": t.flow.net,
                    "aligned": t.aligned,
                    "divergent": t.divergent,
                }
                for t in state.breadth.per_ticker
            ],
        }
        if spy_data:
            d["vwap_raw"] = spy_data.get("vwap", 0)
            d["vwap_plus_1"] = spy_data.get("vwap_plus_1", 0)
            d["vwap_plus_2"] = spy_data.get("vwap_plus_2", 0)
            d["vwap_minus_1"] = spy_data.get("vwap_minus_1", 0)
            d["vwap_minus_2"] = spy_data.get("vwap_minus_2", 0)
            d["ema_9"] = spy_data.get("ema_9", 0)
            d["ema_21"] = spy_data.get("ema_21", 0)
            d["orb_high"] = spy_data.get("orb_high", 0)
            d["orb_low"] = spy_data.get("orb_low", 0)
        return d

    def _parse_signal(self, response_text: str) -> dict:
        """Extract JSON signal from the last line of response."""
        lines = response_text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("```"):
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict) and "action" in parsed:
                    if "direction" not in parsed:
                        parsed["direction"] = parsed["action"]
                    logger.info("[SIGNAL] Parsed OK: %s", json.dumps(parsed))
                    return parsed
            except (json.JSONDecodeError, ValueError):
                continue
        # Log the last 3 lines so we can debug what the LLM actually returned
        tail = lines[-3:] if len(lines) >= 3 else lines
        logger.error("[SIGNAL] No valid JSON found in response. Last lines: %s", tail)
        return {"action": "WAIT", "signal": None, "conviction": "LOW"}

    def _act(self, signal_data: dict, state: MarketState, response_text: str):
        """Handle position tracking, entry capture, and outcome recording."""
        action = signal_data.get("action", "")
        sig = signal_data.get("signal", "")

        # ENTRY — new position
        if sig == "ENTRY" and action in ("CALL", "PUT") and not self.position:
            self.position = {
                "action": action,
                "entry": signal_data.get("entry"),
                "stop": signal_data.get("stop"),
                "target": signal_data.get("target"),
                "conviction": signal_data.get("conviction"),
                "entry_time": state.timestamp,
            }
            self.memory.record_entry(signal_data, state)
            self.trade_logger.log_entry(signal_data, state)  # Log to JSONL
            publish_event("V2_POSITION", "", {"active": True, "event": "ENTRY", **self.position})
            logger.info("[ACTION] >>> ENTERED %s @ $%s | stop=$%s target=$%s | %s",
                        action, signal_data.get("entry"), signal_data.get("stop"),
                        signal_data.get("target"), signal_data.get("conviction"))

        # EXIT — close position
        elif action == "EXIT" and self.position:
            signal_data["price"] = signal_data.get("price", state.price)
            
            # Get entry snapshot for trade logger and post-exit analysis
            entry_snapshot = self.redis.get("zero_dte_v2:entry_snapshot")
            if entry_snapshot:
                entry_data = json.loads(entry_snapshot)
                self.trade_logger.log_exit(signal_data, entry_data)  # Log to JSONL
                market_labels = {
                    "regime": state.orb_regime.regime,
                    "session": state.session,
                    "flow_dir": state.flow.direction,
                    "flow_ratio": state.flow.ratio,
                    "action": self.position["action"],
                }
                self.post_exit_analyzer.start_tracking(signal_data, entry_data, market_labels=market_labels)
            
            publish_event("V2_POSITION", "", {
                "active": False, "event": "EXIT",
                "exit_price": signal_data.get("price"),
                **self.position,
            })
            self.memory.record_outcome_async(signal_data, state, response_text)
            logger.info("[ACTION] >>> EXITED %s @ $%s", self.position["action"], signal_data.get("price"))
            logger.info("[POST] Post-exit tracking started (5min)")
            self.position = None

        # HOLD — update conviction if changed
        elif sig == "HOLD" and self.position:
            new_conv = signal_data.get("conviction")
            if new_conv and new_conv != self.position.get("conviction"):
                logger.info("[ACTION] Conviction changed: %s -> %s", self.position["conviction"], new_conv)
                self.position["conviction"] = new_conv

        # WAIT — snapshot for missed opportunity tracking
        elif action == "WAIT" and not self.position:
            flow_dir = state.flow.direction
            logger.info("[ACTION] WAIT | Flow: %s", flow_dir)
            logger.info("[POST] Post-WAIT tracking started")
            self._snapshot_wait(state)
            # Start post-WAIT tracking for all WAIT signals
            market_labels = {
                "regime": state.orb_regime.regime,
                "session": state.session,
                "flow_dir": state.flow.direction,
                "flow_ratio": state.flow.ratio,
            }
            self.post_exit_analyzer.start_wait_tracking(
                wait_signal={"price": signal_data.get("price", 0)},
                market_state={"flow": flow_dir},
                market_labels=market_labels,
            )

    # ── WAIT outcome tracking ─────────────────────────────────

    WAIT_SNAPSHOT_KEY = "zero_dte_v2:wait_snapshot"
    WAIT_MOVE_THRESHOLD = 0.50  # $0.50 SPY move = meaningful missed opportunity

    def _snapshot_wait(self, state: MarketState):
        """Capture price + labels when LLM says WAIT on directional flow."""
        now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
        snapshot = {
            "price": state.price,
            "flow_direction": state.flow.direction,
            "flow_ratio": state.flow.ratio,
            "labels": state.to_prompt_text(),
            "labels_hash": state.labels_hash(),
            "session": state.session,
            "regime": state.orb_regime.regime,
            "time": now_pt.strftime("%H:%M"),
            "ts": time.time(),
            "wait_nlp": self.memory._state_to_nlp(state),
        }
        # Only keep the most recent WAIT — overwrite previous
        self.redis.setex(self.WAIT_SNAPSHOT_KEY, 300, json.dumps(snapshot))

    def _evaluate_wait_outcome(self, current_price: float):
        """
        Check if a previous WAIT was a missed opportunity.
        If price moved > threshold in the direction flow suggested, store as MISSED_OPPORTUNITY.
        """
        raw = self.redis.get(self.WAIT_SNAPSHOT_KEY)
        if not raw:
            return

        snap = json.loads(raw)
        wait_price = snap.get("price", 0)
        if not wait_price:
            return

        # Handle current_price - could be float or dict
        if isinstance(current_price, dict):
            current_price = current_price.get('current', 0)
            
        move = current_price - wait_price
        abs_move = abs(move)

        # Not enough movement to matter
        if abs_move < self.WAIT_MOVE_THRESHOLD:
            return

        flow_dir = snap.get("flow_direction", "")
        # Determine if this was a missed opportunity
        # BUYING flow + price went UP = missed CALL
        # SELLING flow + price went DOWN = missed PUT
        buying_flow = flow_dir in ("STRONG_BUYING", "BUYING", "LEAN_BUYING")
        selling_flow = flow_dir in ("STRONG_SELLING", "SELLING", "LEAN_SELLING")

        if buying_flow and move > 0:
            missed_action = "CALL"
        elif selling_flow and move < 0:
            missed_action = "PUT"
        else:
            # Flow was wrong direction or market moved against flow — not a miss
            self.redis.delete(self.WAIT_SNAPSHOT_KEY)
            return

        # Store missed opportunity in memory
        self.memory.record_wait_outcome_async(snap, current_price, missed_action, move)
        self.redis.delete(self.WAIT_SNAPSHOT_KEY)

        direction_word = "up" if move > 0 else "down"
        logger.warning("[POST] MISSED %s: WAIT at $%.2f, price moved %s $%.2f to $%.2f",
                       missed_action, wait_price, direction_word, abs_move, current_price)
