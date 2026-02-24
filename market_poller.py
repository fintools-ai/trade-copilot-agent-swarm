"""
Background Market Data Poller — Redis Cache for MCP Tools

Keeps one persistent MCP subprocess alive, polls SPY technicals + Mag7 quotes
on intervals, writes to Redis. LLM tools read from Redis (~1ms) instead of
spawning MCP per-call (3-5s).

Usage:
    python market_poller.py
    python market_poller.py --spy 5 --mag7 10
"""

import json
import time
import asyncio
import signal
import logging
import argparse
from datetime import datetime

import redis

from tools.fast_0dte_tools import (
    create_twelvedata_mcp,
    _parse,
    _calc_orb,
    _calc_vwap_sd,
    _calc_trama,
)
from config.settings import POLLER_SPY_INTERVAL, POLLER_MAG7_INTERVAL

logger = logging.getLogger(__name__)

# Redis keys
REDIS_KEY_SPY = "market:spy:data"
REDIS_KEY_MAG7 = "market:mag7:data"
REDIS_KEY_STATUS = "market:poller:status"
REDIS_TTL = 30  # seconds


def _get_redis():
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


async def poll_spy(mcp, r):
    """Poll SPY technicals via MCP, write to Redis."""
    symbol = "SPY"
    t0 = time.time()

    try:
        fetch_time = datetime.now()
        data = {
            "symbol": symbol,
            "timestamp": fetch_time.strftime("%Y-%m-%d %H:%M:%S"),
            "fetch_ts": fetch_time.timestamp(),
        }

        # Only fetch quote for fast 1s updates
        quote = await mcp.call_tool_async(
            tool_use_id=f"q_{symbol}",
            name="GetQuote",
            arguments={"params": {"symbol": symbol}},
        )

        # Process Quote
        if not isinstance(quote, Exception) and quote and quote.get("status") == "success":
            q = _parse(quote)
            if q:
                data["price"] = {
                    "current": float(q.get("close", 0)),
                    "open": float(q.get("open", 0)),
                    "high": float(q.get("high", 0)),
                    "low": float(q.get("low", 0)),
                    "prev_close": float(q.get("previous_close", 0)),
                    "change": float(q.get("change", 0)),
                    "change_pct": float(q.get("percent_change", 0)),
                }
                data["volume"] = {
                    "current": int(q.get("volume", 0)),
                    "average": int(q.get("average_volume", 0)),
                }
                if data["volume"]["average"] > 0:
                    data["volume"]["ratio"] = round(
                        data["volume"]["current"] / data["volume"]["average"], 2
                    )

        # Get existing data from Redis to preserve technicals
        existing = r.get(REDIS_KEY_SPY)
        if existing:
            existing_data = json.loads(existing)
            # Preserve technical indicators from previous poll
            for key in ["rsi", "vwap", "price_vs_vwap", "ema_9", "ema_21", "macd", "orb_high", "orb_low", "orb_range", "orb_direction", "trama"]:
                if key in existing_data:
                    data[key] = existing_data[key]

        # Write to Redis
        r.set(REDIS_KEY_SPY, json.dumps(data), ex=REDIS_TTL)

        # Publish price tick for UI updates
        price_data = data.get("price", {})
        if price_data.get("current"):
            tick = json.dumps({
                "type": "V2_PRICE_TICK",
                "timestamp": data.get("timestamp", ""),
                "signal": {
                    "price": price_data["current"],
                    "change": price_data.get("change", 0),
                    "change_pct": price_data.get("change_pct", 0),
                },
            })
            r.publish("zero_dte:events", tick)
        r.set(REDIS_KEY_SPY, json.dumps(data), ex=REDIS_TTL)

        elapsed = time.time() - t0
        price_data = data.get("price", {})
        price = price_data.get("current", "?")
        logger.info("[SPY] Price poll: $%s in %.1fs", price, elapsed)

    except Exception as e:
        logger.error("[SPY] Price poll failed: %s", e)


async def poll_spy_technicals(mcp, r):
    """Poll SPY technical indicators (slower, run every 5-10s)."""
    symbol = "SPY"
    t0 = time.time()

    try:
        rsi, vwap, ema9, ema21, macd, ts = await asyncio.gather(
            mcp.call_tool_async(
                tool_use_id=f"rsi_{symbol}",
                name="GetTimeSeriesRsi",
                arguments={"params": {"symbol": symbol, "interval": "5min", "time_period": 14}},
            ),
            mcp.call_tool_async(
                tool_use_id=f"vwap_{symbol}",
                name="GetTimeSeriesVwap",
                arguments={"params": {"symbol": symbol, "interval": "5min"}},
            ),
            mcp.call_tool_async(
                tool_use_id=f"ema9_{symbol}",
                name="GetTimeSeriesEma",
                arguments={"params": {"symbol": symbol, "interval": "5min", "time_period": 9}},
            ),
            mcp.call_tool_async(
                tool_use_id=f"ema21_{symbol}",
                name="GetTimeSeriesEma",
                arguments={"params": {"symbol": symbol, "interval": "5min", "time_period": 21}},
            ),
            mcp.call_tool_async(
                tool_use_id=f"macd_{symbol}",
                name="GetTimeSeriesMacd",
                arguments={"params": {"symbol": symbol, "interval": "5min"}},
            ),
            mcp.call_tool_async(
                tool_use_id=f"ts_{symbol}",
                name="GetTimeSeries",
                arguments={"params": {"symbol": symbol, "interval": "5min", "outputsize": 50}},
            ),
            return_exceptions=True,
        )

        # Get existing data
        existing = r.get(REDIS_KEY_SPY)
        if not existing:
            return
        
        data = json.loads(existing)

        # Process RSI
        if not isinstance(rsi, Exception) and rsi and rsi.get("status") == "success":
            r_data = _parse(rsi)
            if r_data and "values" in r_data and r_data["values"]:
                data["rsi"] = round(float(r_data["values"][0].get("rsi", 0)), 1)

        # Process VWAP
        if not isinstance(vwap, Exception) and vwap and vwap.get("status") == "success":
            v = _parse(vwap)
            if v and "values" in v and v["values"]:
                vwap_val = float(v["values"][0].get("vwap", 0))
                data["vwap"] = round(vwap_val, 2)
                if data.get("price", {}).get("current"):
                    data["price_vs_vwap"] = round(data["price"]["current"] - vwap_val, 2)

        # Process EMA 9
        if not isinstance(ema9, Exception) and ema9 and ema9.get("status") == "success":
            e = _parse(ema9)
            if e and "values" in e and e["values"]:
                data["ema_9"] = round(float(e["values"][0].get("ema", 0)), 2)

        # Process EMA 21
        if not isinstance(ema21, Exception) and ema21 and ema21.get("status") == "success":
            e = _parse(ema21)
            if e and "values" in e and e["values"]:
                data["ema_21"] = round(float(e["values"][0].get("ema", 0)), 2)

        # Process MACD
        if not isinstance(macd, Exception) and macd and macd.get("status") == "success":
            m = _parse(macd)
            if m and "values" in m and m["values"]:
                mv = m["values"][0]
                data["macd"] = {
                    "macd": round(float(mv.get("macd", 0)), 3),
                    "signal": round(float(mv.get("macd_signal", 0)), 3),
                    "histogram": round(float(mv.get("macd_hist", 0)), 3),
                }

        # Process Time Series (ORB, TRAMA, VWAP SD)
        if not isinstance(ts, Exception) and ts and ts.get("status") == "success":
            t = _parse(ts)
            if t and "values" in t:
                logger.info("[SPY] Processing time series: %d candles available", len(t["values"]))
                
                orb = _calc_orb(t["values"])
                if orb:
                    data["orb_high"] = orb.get("high", 0)
                    data["orb_low"] = orb.get("low", 0)
                    data["orb_range"] = orb.get("range", 0)
                    data["orb_direction"] = orb.get("direction", "")
                    logger.info("[SPY] ORB calculated: high=$%.2f low=$%.2f range=$%.2f", 
                               orb.get("high", 0), orb.get("low", 0), orb.get("range", 0))
                else:
                    logger.warning("[SPY] ORB calculation returned None (need 6+ candles from today)")

                trama = _calc_trama(t["values"])
                if trama:
                    data["trama"] = trama

                if data.get("vwap") and data.get("price", {}).get("current"):
                    vwap_sd = _calc_vwap_sd(
                        t["values"], data["vwap"], data["price"]["current"]
                    )
                    if vwap_sd:
                        data["vwap_sd"] = vwap_sd["sd"]
                        data["vwap_position"] = vwap_sd["position"]
                        data["vwap_plus_1"] = vwap_sd["plus_1"]
                        data["vwap_plus_2"] = vwap_sd["plus_2"]
                        data["vwap_minus_1"] = vwap_sd["minus_1"]
                        data["vwap_minus_2"] = vwap_sd["minus_2"]
            else:
                logger.warning("[SPY] Time series parse failed or no values")
        else:
            logger.warning("[SPY] Time series fetch failed or returned error")

        # Write updated data back to Redis
        r.set(REDIS_KEY_SPY, json.dumps(data), ex=REDIS_TTL)

        elapsed = time.time() - t0
        logger.info("[SPY] Technicals updated in %.1fs", elapsed)

    except Exception as e:
        logger.error("[SPY] Technicals fetch failed: %s", e)


async def poll_mag7(mcp, r):
    """Poll Mag7 quotes via MCP, write to Redis."""
    symbols = ["SPY", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    t0 = time.time()

    try:
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fetch_ts": datetime.now().timestamp(),
            "symbols": {},
            "summary": {"bullish": 0, "bearish": 0, "neutral": 0},
        }

        results = await asyncio.gather(
            *[
                mcp.call_tool_async(
                    tool_use_id=f"q_{sym}",
                    name="GetQuote",
                    arguments={"params": {"symbol": sym}},
                )
                for sym in symbols
            ],
            return_exceptions=True,
        )

        for sym, result in zip(symbols, results):
            if isinstance(result, Exception):
                data["symbols"][sym] = {"error": str(result)}
                continue

            if result and result.get("status") == "success":
                q = _parse(result)
                if q:
                    pct = float(q.get("percent_change", 0))

                    if pct > 0.15:
                        data["summary"]["bullish"] += 1
                    elif pct < -0.15:
                        data["summary"]["bearish"] += 1
                    else:
                        data["summary"]["neutral"] += 1

                    data["symbols"][sym] = {
                        "price": float(q.get("close", 0)),
                        "change": float(q.get("change", 0)),
                        "change_pct": round(pct, 2),
                    }

        # Write to Redis
        json_str = json.dumps(data, indent=2)
        r.set(REDIS_KEY_MAG7, json_str, ex=REDIS_TTL)

        elapsed = time.time() - t0
        logger.info("[MAG7] Polled %d symbols in %.1fs", len(data["symbols"]), elapsed)

    except Exception as e:
        logger.error("[MAG7] Poll failed: %s", e)


async def _loop(fn, interval, mcp, r, shutdown_event):
    """Generic poll loop: call fn, sleep remainder of interval, skip-on-overlap."""
    while not shutdown_event.is_set():
        t0 = time.time()
        await fn(mcp, r)
        elapsed = time.time() - t0

        # Sleep for remainder of interval, or skip if poll took longer
        sleep_time = max(0, interval - elapsed)
        if sleep_time > 0:
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=sleep_time)
            except asyncio.TimeoutError:
                pass  # Normal: timeout means we should poll again
        else:
            logger.warning("[POLLER] %s took %.1fs > %ss interval, skipping sleep", fn.__name__, elapsed, interval)


async def _heartbeat(r, shutdown_event):
    """Update poller status key every 5s."""
    while not shutdown_event.is_set():
        status = json.dumps({"alive": True, "last_tick": time.time()})
        r.set(REDIS_KEY_STATUS, status, ex=15)
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass


async def run(spy_interval: int, mag7_interval: int):
    """Main poller loop. Opens MCP client once, runs poll loops concurrently."""
    r = _get_redis()
    r.ping()
    logger.info("[POLLER] Redis connected")

    shutdown_event = asyncio.Event()

    # Wire up signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    logger.info("[POLLER] Starting: SPY quote every %ss, SPY technicals every 10s, Mag7 every %ss", spy_interval, mag7_interval)

    with create_twelvedata_mcp() as mcp:
        logger.info("[MCP] Client connected (persistent)")

        await asyncio.gather(
            _loop(poll_spy, spy_interval, mcp, r, shutdown_event),
            _loop(poll_spy_technicals, 10, mcp, r, shutdown_event),  # Technicals every 10s
            _loop(poll_mag7, mag7_interval, mcp, r, shutdown_event),
            _heartbeat(r, shutdown_event),
        )

    # Cleanup status on exit
    r.delete(REDIS_KEY_STATUS)
    logger.info("[POLLER] Stopped")


def main():
    parser = argparse.ArgumentParser(description="Background market data poller")
    parser.add_argument("--spy", type=int, default=POLLER_SPY_INTERVAL, help="SPY poll interval (seconds)")
    parser.add_argument("--mag7", type=int, default=POLLER_MAG7_INTERVAL, help="Mag7 poll interval (seconds)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(run(args.spy, args.mag7))


if __name__ == "__main__":
    main()
