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

from config.settings import (
    AWS_REGION,
    ENGINE_MODEL_ID,
    ENGINE_SCAN_INTERVAL,
    ENGINE_MONITOR_INTERVAL,
    ENGINE_MAX_TOKENS,
    CLASSIFIER_MODEL_ID,
)
from trading_engine.classifier import classify_all, classify_flow, _get_window, MarketState
from trading_engine.prompts import (
    SYSTEM_PROMPT, build_scan_prompt, build_monitor_prompt,
    FLOW_CLASSIFIER_SYSTEM, build_flow_classifier_prompt,
)
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

        # Wait for first poller data
        print("[v2] Waiting for market data...")
        for _ in range(30):
            if self.redis.exists(REDIS_KEY_SPY):
                break
            await asyncio.sleep(1)

        print(f"[v2] Engine started — model: {ENGINE_MODEL_ID}")
        publish_event("ENGINE_STATUS", "v2 engine started", {"version": "v2"})

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

        # 2. CLASSIFY (~1ms Python, +600ms if borderline → Haiku)
        state = classify_all(flow_data, spy_data, mag7_data)

        # Hybrid: if SPY flow is borderline, let Haiku reclassify
        if state.flow.borderline:
            llm_direction = await self._classify_flow_llm(flow_data, "SPY")
            if llm_direction and llm_direction != state.flow.direction:
                print(f"  [hybrid] Haiku reclassified SPY flow: {state.flow.direction} → {llm_direction}")
                state.flow.direction = llm_direction
                state.flow.borderline = False

        # Print classified labels
        print(f"\n[v2] Cycle #{self.cycle_count} — {state.timestamp}")
        print(f"  Regime: {state.orb_regime.regime} (ORB ${state.orb_regime.range_dollars} = {state.orb_regime.range_pct:.3f}%)")
        haiku_tag = " [LLM-classified]" if not state.flow.borderline and state.flow.direction in ("LEAN_BUYING", "LEAN_SELLING", "BUYING", "SELLING") and 0.4 < state.flow.ratio < 2.5 else ""
        print(f"  Flow: {state.flow.direction} {state.flow.ratio}:1 net={state.flow.net} {state.flow.momentum}{haiku_tag}")
        print(f"  RSI: {state.tech.rsi_state} ({state.tech.rsi_value}) | VWAP: {state.tech.vwap_position}")
        print(f"  Breadth: {state.breadth.bias} ({state.breadth.aligned_count}/7)")
        if state.breadth.divergent_tickers:
            print(f"  Divergent: {', '.join(state.breadth.divergent_tickers)}")

        # 3. RECALL (hash-cached)
        memories = await asyncio.get_event_loop().run_in_executor(None, self.memory.recall, state)

        # 4. BUILD PROMPT
        if self.position:
            user_prompt = build_monitor_prompt(state, self.position, memories)
        else:
            user_prompt = build_scan_prompt(state, memories)

        # 5. SYNTHESIZE (Claude Sonnet 4.5)
        query_start_ts = time.time()
        response_text = await self._call_claude(user_prompt)
        latency = time.time() - query_start_ts

        if not response_text:
            return

        # 6. PARSE SIGNAL
        signal_data = self._parse_signal(response_text)
        signal_data["latency"] = round(latency, 1)

        # 7. ACT
        self._act(signal_data, state, response_text)

        # 8. PUBLISH TO UI
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
            {**signal_data, "cycle": self.cycle_count, "total_time": round(total, 1)},
        )

        print(f"  Signal: {signal_data.get('action', '?')} | {signal_data.get('conviction', '?')} | {latency:.1f}s LLM | {total:.1f}s total")

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

    async def _call_claude(self, user_prompt: str) -> str:
        """Call Claude Sonnet 4.5 via Bedrock invoke_model."""
        loop = asyncio.get_event_loop()

        def _invoke():
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": ENGINE_MAX_TOKENS,
                "temperature": 0,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
            })

            response = self.bedrock.invoke_model(
                modelId=ENGINE_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )

            result = json.loads(response["body"].read())
            return result["content"][0]["text"]

        try:
            return await loop.run_in_executor(None, _invoke)
        except Exception as e:
            logger.error(f"Claude call failed: {e}")
            print(f"  [v2] Claude error: {e}")
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
