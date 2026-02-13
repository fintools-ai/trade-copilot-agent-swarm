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
from strands.agent.conversation_manager import SlidingWindowConversationManager

from config.settings import (
    AWS_REGION,
    ENGINE_MODEL_ID,
    ENGINE_SCAN_INTERVAL,
    ENGINE_MONITOR_INTERVAL,
    ENGINE_CONVERSATION_WINDOW,
    CLASSIFIER_MODEL_ID,
    CLASSIFIER_ALWAYS_LLM,
)
from trading_engine.classifier import classify_all, classify_flow, _get_window, MarketState
from trading_engine.prompts import (
    SYSTEM_PROMPT, build_scan_prompt, build_monitor_prompt,
    FLOW_CLASSIFIER_SYSTEM, build_flow_classifier_prompt,
)
from trading_engine.tools import create_engine_tools
from trading_engine.memory import MemoryStore
from trading_engine.consolidator import consolidate
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
        logger.info(f"Market poller started (pid {self.poller_proc.pid})")
        print(f"[v2] Market poller started (pid {self.poller_proc.pid})")

        # Initialize memory store (background, don't block)
        asyncio.get_event_loop().run_in_executor(None, self.memory.get_or_create_memory)

        # Create Strands Agent (persistent conversation history)
        self._create_agent()

        # Wait for first poller data
        print("[v2] Waiting for market data...")
        for _ in range(30):
            if self.redis.exists(REDIS_KEY_SPY):
                break
            await asyncio.sleep(1)

        print(f"[v2] Agent created — model: {ENGINE_MODEL_ID}, window: {ENGINE_CONVERSATION_WINDOW}")
        publish_event("ENGINE_STATUS", "v2 engine started", {"version": "v2", "agent": True})

        try:
            while not self._shutdown.is_set():
                # Stop after market close
                now_pt = datetime.now(pt_tz)
                if now_pt.hour >= 13:
                    print("[v2] Market closed (1PM PT) — stopping")
                    break

                # Wait for market open
                if now_pt.hour < 6 or (now_pt.hour == 6 and now_pt.minute < 30):
                    open_time = now_pt.replace(hour=6, minute=30, second=0)
                    wait_secs = (open_time - now_pt).total_seconds()
                    print(f"[v2] Market opens in {wait_secs/60:.0f}min, waiting...")
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
            logger.error(f"Engine error: {e}")
            print(f"[v2] Engine error: {e}")
        finally:
            # End of day: consolidate
            try:
                print("[v2] Running end-of-day consolidation...")
                await consolidate(self.memory)
            except Exception as e:
                logger.warning(f"Consolidation error: {e}")

            self._cleanup()

    def _cleanup(self):
        """Stop poller and clean up."""
        if self.poller_proc:
            self.poller_proc.terminate()
            try:
                self.poller_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.poller_proc.kill()
            print("[v2] Market poller stopped")

    async def cycle(self):
        """Single engine cycle: Fetch → Classify → Recall → Synthesize → Act"""
        self.cycle_count += 1
        t0 = time.time()

        # 1. FETCH (parallel)
        flow_data, spy_data, mag7_data = await self._fetch_data()
        if not spy_data:
            logger.warning("No SPY data available, skipping cycle")
            return

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
                print(f"  [hybrid] Haiku reclassified SPY flow: {state.flow.direction} → {llm_direction}")
                state.flow.direction = llm_direction
                state.flow.borderline = False

        # Print classified labels
        print(f"\n[v2] Cycle #{self.cycle_count} — {state.timestamp}")
        print(f"  Regime: {state.orb_regime.regime} (ORB ${state.orb_regime.range_dollars} = {state.orb_regime.range_pct:.3f}%)")
        haiku_tag = " [LLM]" if CLASSIFIER_ALWAYS_LLM or (not state.flow.borderline and state.flow.direction in ("LEAN_BUYING", "LEAN_SELLING", "BUYING", "SELLING") and 0.4 < state.flow.ratio < 2.5) else ""
        print(f"  Flow: {state.flow.direction} {state.flow.ratio}:1 net={state.flow.net} {state.flow.momentum}{haiku_tag}")
        print(f"  RSI: {state.tech.rsi_state} ({state.tech.rsi_value}) | VWAP: {state.tech.vwap_position}")
        print(f"  Breadth: {state.breadth.bias} ({state.breadth.aligned_count}/7)")
        if state.breadth.divergent_tickers:
            print(f"  Divergent: {', '.join(state.breadth.divergent_tickers)}")

        # 3. RECALL (hash-cached)
        memories = await asyncio.get_event_loop().run_in_executor(None, self.memory.recall, state)

        # 4. STORE STATE + MEMORIES on engine instance (tools read these)
        self._current_state = state
        self._current_memories = memories

        # 5. BUILD PROMPT (data inline + wait streak warning)
        if self.position:
            user_prompt = build_monitor_prompt(state, self.position, memories)
        else:
            user_prompt = build_scan_prompt(state, memories, wait_streak=self._wait_streak)

        # 6. SYNTHESIZE (Strands Agent with conversation history)
        query_start_ts = time.time()
        response_text = await self._invoke_agent(user_prompt)
        latency = time.time() - query_start_ts

        if not response_text:
            return

        # 7. PARSE SIGNAL
        signal_data = self._parse_signal(response_text)
        signal_data["latency"] = round(latency, 1)

        # 8. TRACK WAIT STREAK
        if signal_data.get("action") == "WAIT":
            self._wait_streak += 1
        else:
            self._wait_streak = 0

        # 9. ACT
        self._act(signal_data, state, response_text)

        # 10. PUBLISH TO UI
        mode = "monitor" if self.position else "scan"
        total = time.time() - t0

        # Structured events for v2 engine dashboard
        publish_event("V2_STATE", "", self._state_to_dict(state))
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
        print(f"  Signal: {signal_data.get('action', '?')} | {signal_data.get('conviction', '?')} | {latency:.1f}s LLM | {total:.1f}s total{streak_tag}")

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
                logger.warning(f"Order flow fetch failed: {e}")
                return {}

        async def fetch_redis(key):
            try:
                raw = await loop.run_in_executor(None, self.redis.get, key)
                return json.loads(raw) if raw else {}
            except Exception as e:
                logger.warning(f"Redis fetch {key} failed: {e}")
                return {}

        flow_data, spy_data, mag7_data = await asyncio.gather(
            fetch_flow(),
            fetch_redis(REDIS_KEY_SPY),
            fetch_redis(REDIS_KEY_MAG7),
        )

        return flow_data, spy_data, mag7_data

    def _create_agent(self):
        """Create the persistent Strands Agent with tools and sliding window history."""
        tools = create_engine_tools(self)
        self.agent = Agent(
            model=ENGINE_MODEL_ID,
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            conversation_manager=SlidingWindowConversationManager(
                window_size=ENGINE_CONVERSATION_WINDOW,
                should_truncate_results=True,
            ),
        )
        logger.info(f"Strands Agent created — model: {ENGINE_MODEL_ID}, window: {ENGINE_CONVERSATION_WINDOW}")

    async def _invoke_agent(self, user_prompt: str) -> str:
        """Call the Strands Agent from the async event loop via run_in_executor.

        The agent has conversation history from previous cycles and tools
        to access market state, memory, and position. Data is also provided
        inline in the prompt so the agent usually makes 0 tool calls (single LLM round-trip).
        """
        loop = asyncio.get_event_loop()

        def _call():
            result = self.agent(user_prompt)
            # Strands Agent returns an AgentResult; extract the text
            return str(result)

        try:
            return await loop.run_in_executor(None, _call)
        except Exception as e:
            logger.error(f"Agent call failed: {e}")
            print(f"  [v2] Agent error: {e}")
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
            response = self.bedrock.converse(
                modelId=CLASSIFIER_MODEL_ID,
                system=[
                    {"text": FLOW_CLASSIFIER_SYSTEM},
                    {"cachePoint": {"type": "default"}},
                ],
                messages=[
                    {"role": "user", "content": [{"text": user_text}]}
                ],
                inferenceConfig={"maxTokens": 200, "temperature": 0},
            )
            text = response["output"]["message"]["content"][0]["text"]
            parsed = json.loads(text)
            return parsed.get("direction", "")

        try:
            t0 = time.time()
            direction = await loop.run_in_executor(None, _invoke)
            dur = time.time() - t0
            print(f"  [hybrid] Haiku classified {symbol} flow in {dur:.1f}s: {direction}")
            return direction
        except Exception as e:
            logger.warning(f"Haiku flow classification failed: {e}")
            return None

    def _state_to_dict(self, state: MarketState) -> dict:
        """Convert MarketState to a flat dict for SSE publishing."""
        return {
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
                    return parsed
            except (json.JSONDecodeError, ValueError):
                continue
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
            publish_event("V2_POSITION", "", {"active": True, "event": "ENTRY", **self.position})
            print(f"  >>> ENTERED {action} @ ${signal_data.get('entry')}")

        # EXIT — close position
        elif action == "EXIT" and self.position:
            signal_data["price"] = signal_data.get("price", state.price)
            publish_event("V2_POSITION", "", {
                "active": False, "event": "EXIT",
                "exit_price": signal_data.get("price"),
                **self.position,
            })
            self.memory.record_outcome_async(signal_data)
            print(f"  >>> EXITED {self.position['action']} @ ${signal_data.get('price')}")
            self.position = None

        # HOLD — update conviction if changed
        elif sig == "HOLD" and self.position:
            new_conv = signal_data.get("conviction")
            if new_conv and new_conv != self.position.get("conviction"):
                print(f"  >>> Conviction: {self.position['conviction']} → {new_conv}")
                self.position["conviction"] = new_conv

        # WAIT — snapshot for missed opportunity tracking (only when flow is directional)
        elif action == "WAIT" and not self.position:
            flow_dir = state.flow.direction
            if flow_dir not in ("MIXED",):
                self._snapshot_wait(state)

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
            "time": now_pt.strftime("%H:%M"),
            "ts": time.time(),
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
        print(f"  [memory] MISSED {missed_action}: WAIT at ${wait_price:.2f}, price moved {direction_word} ${abs_move:.2f} to ${current_price:.2f}")
