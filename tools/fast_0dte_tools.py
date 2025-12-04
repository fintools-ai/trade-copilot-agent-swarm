"""
Fast 0DTE Tools - Raw data for LLM reasoning
Uses Twelve Data MCP to fetch market data. Returns structured JSON.
NO bias calculation - let the LLM (desk trader agent) interpret and decide.

Two tools:
1. fast_spy_check - Price, RSI, VWAP, EMA, MACD, ORB for SPY
2. fast_mag7_scan - Quick scan of Mag7 for breadth confirmation
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional
from strands import tool
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient
from config.settings import TWELVE_DATA_API_KEY
from redis_stream import publish_event

logger = logging.getLogger(__name__)

# Twelve Data API key from environment
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", TWELVE_DATA_API_KEY)


def create_twelvedata_mcp():
    """Create Twelve Data MCP client with extended endpoints (-n 100)"""
    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx",
                args=["mcp-server-twelve-data", "-k", TWELVEDATA_API_KEY, "-n", "100"]
            )
        )
    )


@tool
async def fast_spy_check() -> str:
    """
    Fast SPY data fetch for 0DTE trading.
    Returns RAW market data as JSON - LLM interprets and decides.

    Data returned:
    - price: current, open, high, low, prev_close, change, change_pct
    - volume: current, average, ratio
    - rsi: 14-period on 5min (values 0-100)
    - vwap: volume-weighted average price
    - price_vs_vwap: delta (positive = above VWAP)
    - ema_9: fast EMA for trend
    - ema_21: slow EMA for trend
    - macd: macd line, signal line, histogram
    - orb: opening range high/low/range (first 30min)
    - trama: Trend Regularity Adaptive MA (LLM interprets)
      - value: current TRAMA level
      - price_vs_trama: positive = above TRAMA, negative = below
      - trend_strength: 0-1, higher = more new HH/LL = stronger trend

    Use for: Quick market read, trend confirmation, entry validation.
    """
    symbol = "SPY"

    try:
        data = {
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with create_twelvedata_mcp() as mcp:
            # 1. Quote - price, volume, change
            quote = await mcp.call_tool_async(
                tool_use_id=f"q_{symbol}",
                name="GetQuote",
                arguments={"params": {"symbol": symbol}}
            )
            if quote and quote.get("status") == "success":
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

            # 2. RSI (14 period, 5min)
            rsi = await mcp.call_tool_async(
                tool_use_id=f"rsi_{symbol}",
                name="GetTimeSeriesRsi",
                arguments={"params": {"symbol": symbol, "interval": "5min", "time_period": 14}}
            )
            if rsi and rsi.get("status") == "success":
                r = _parse(rsi)
                if r and "values" in r and r["values"]:
                    data["rsi"] = round(float(r["values"][0].get("rsi", 0)), 1)

            # 3. VWAP
            vwap = await mcp.call_tool_async(
                tool_use_id=f"vwap_{symbol}",
                name="GetTimeSeriesVwap",
                arguments={"params": {"symbol": symbol, "interval": "5min"}}
            )
            if vwap and vwap.get("status") == "success":
                v = _parse(vwap)
                if v and "values" in v and v["values"]:
                    vwap_val = float(v["values"][0].get("vwap", 0))
                    data["vwap"] = round(vwap_val, 2)
                    if data.get("price", {}).get("current"):
                        data["price_vs_vwap"] = round(
                            data["price"]["current"] - vwap_val, 2
                        )

            # 4. EMA 9 (fast)
            ema9 = await mcp.call_tool_async(
                tool_use_id=f"ema9_{symbol}",
                name="GetTimeSeriesEma",
                arguments={"params": {"symbol": symbol, "interval": "5min", "time_period": 9}}
            )
            if ema9 and ema9.get("status") == "success":
                e = _parse(ema9)
                if e and "values" in e and e["values"]:
                    data["ema_9"] = round(float(e["values"][0].get("ema", 0)), 2)

            # 5. EMA 21 (slow)
            ema21 = await mcp.call_tool_async(
                tool_use_id=f"ema21_{symbol}",
                name="GetTimeSeriesEma",
                arguments={"params": {"symbol": symbol, "interval": "5min", "time_period": 21}}
            )
            if ema21 and ema21.get("status") == "success":
                e = _parse(ema21)
                if e and "values" in e and e["values"]:
                    data["ema_21"] = round(float(e["values"][0].get("ema", 0)), 2)

            # 6. MACD
            macd = await mcp.call_tool_async(
                tool_use_id=f"macd_{symbol}",
                name="GetTimeSeriesMacd",
                arguments={"params": {"symbol": symbol, "interval": "5min"}}
            )
            if macd and macd.get("status") == "success":
                m = _parse(macd)
                if m and "values" in m and m["values"]:
                    mv = m["values"][0]
                    data["macd"] = {
                        "macd": round(float(mv.get("macd", 0)), 3),
                        "signal": round(float(mv.get("macd_signal", 0)), 3),
                        "histogram": round(float(mv.get("macd_hist", 0)), 3),
                    }

            # 7. Time series for ORB and TRAMA calculation
            ts = await mcp.call_tool_async(
                tool_use_id=f"ts_{symbol}",
                name="GetTimeSeries",
                arguments={"params": {"symbol": symbol, "interval": "5min", "outputsize": 50}}
            )
            if ts and ts.get("status") == "success":
                t = _parse(ts)
                if t and "values" in t:
                    # ORB calculation
                    orb = _calc_orb(t["values"])
                    if orb:
                        data["orb"] = orb

                    # TRAMA calculation (Trend Regularity Adaptive MA)
                    trama = _calc_trama(t["values"])
                    if trama:
                        data["trama"] = trama

        # Publish structured market data to UI (no parsing needed)
        _publish_market_data(data)

        return json.dumps(data, indent=2)

    except Exception as e:
        logger.error(f"fast_spy_check error: {e}")
        return json.dumps({"error": str(e)})


@tool
async def fast_mag7_scan() -> str:
    """
    Quick scan of SPY + Mag7 for market breadth.
    Returns RAW data as JSON - LLM interprets alignment/divergence.

    Data returned:
    - symbols: price, change, change_pct for each
    - summary: count of bullish (>0.15%), bearish (<-0.15%), neutral

    Use for: Cross-validation, breadth confirmation, divergence detection.
    """
    symbols = ["SPY", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    try:
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbols": {},
            "summary": {
                "bullish": 0,
                "bearish": 0,
                "neutral": 0,
            }
        }

        with create_twelvedata_mcp() as mcp:
            for sym in symbols:
                try:
                    quote = await mcp.call_tool_async(
                        tool_use_id=f"q_{sym}",
                        name="GetQuote",
                        arguments={"params": {"symbol": sym}}
                    )
                    if quote and quote.get("status") == "success":
                        q = _parse(quote)
                        if q:
                            pct = float(q.get("percent_change", 0))

                            # Categorize for summary
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
                except Exception as e:
                    data["symbols"][sym] = {"error": str(e)}

        return json.dumps(data, indent=2)

    except Exception as e:
        logger.error(f"fast_mag7_scan error: {e}")
        return json.dumps({"error": str(e)})


def _parse(result: Dict) -> Optional[Dict]:
    """Parse MCP result - extract JSON from content[0].text"""
    try:
        if result.get("content"):
            c = result["content"]
            if isinstance(c, list) and c:
                return json.loads(c[0].get("text", "{}"))
    except:
        pass
    return None


def _calc_orb(values: list) -> Optional[Dict]:
    """
    Calculate Opening Range Breakout levels from 5min candles.
    ORB = first 30 minutes (6 x 5min candles) high/low.
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        today_candles = [v for v in values if v.get("datetime", "").startswith(today)]
        today_candles.sort(key=lambda x: x.get("datetime", ""))

        if len(today_candles) < 6:
            return None

        orb_candles = today_candles[:6]  # First 30 min
        orb_high = max(float(c.get("high", 0)) for c in orb_candles)
        orb_low = min(float(c.get("low", 0)) for c in orb_candles)

        return {
            "high": round(orb_high, 2),
            "low": round(orb_low, 2),
            "range": round(orb_high - orb_low, 2),
        }
    except:
        return None


def _calc_trama(values: list, length: int = 14) -> Optional[Dict]:
    """
    Calculate Trend Regularity Adaptive Moving Average (TRAMA).

    TRAMA adapts based on how many new highs/lows are being made:
    - More new HH/LL = tighter (closer to price) = trend is strong
    - Fewer new HH/LL = looser (smoother) = trend exhausting or chop

    Args:
        values: List of candle dicts with 'high', 'low', 'close', 'datetime'
        length: Lookback period (default 14)

    Returns:
        Dict with trama value and trend_strength (0-1)
    """
    try:
        # Sort by datetime (oldest first for calculation)
        sorted_values = sorted(values, key=lambda x: x.get("datetime", ""))

        if len(sorted_values) < length + 5:
            return None

        # Extract prices
        closes = [float(v.get("close", 0)) for v in sorted_values]
        highs = [float(v.get("high", 0)) for v in sorted_values]
        lows = [float(v.get("low", 0)) for v in sorted_values]

        n = len(closes)

        # Track new highest highs and lowest lows
        hh = [0] * n  # 1 if new highest high
        ll = [0] * n  # 1 if new lowest low

        for i in range(length, n):
            # Current rolling high/low
            current_high = max(highs[i-length+1:i+1])
            current_low = min(lows[i-length+1:i+1])

            # Previous rolling high/low
            prev_high = max(highs[i-length:i])
            prev_low = min(lows[i-length:i])

            # New highest high?
            if current_high > prev_high:
                hh[i] = 1
            # New lowest low?
            if current_low < prev_low:
                ll[i] = 1

        # Trend regularity: 1 if new high OR new low
        tr = [max(hh[i], ll[i]) for i in range(n)]

        # Calculate trend coefficient: SMA of tr, then squared
        tc = [0.0] * n
        for i in range(length, n):
            tr_sum = sum(tr[i-length+1:i+1])
            tr_avg = tr_sum / length
            tc[i] = tr_avg ** 2  # Squared for smoothing

        # Adaptive moving average
        trama = [0.0] * n
        trama[0] = closes[0]
        for i in range(1, n):
            trama[i] = trama[i-1] + tc[i] * (closes[i] - trama[i-1])

        # Get current values
        current_trama = trama[-1]
        current_price = closes[-1]
        current_tc = tc[-1]  # Trend coefficient (0-1, higher = stronger trend)

        # Price vs TRAMA
        price_vs_trama = current_price - current_trama

        return {
            "value": round(current_trama, 2),
            "price_vs_trama": round(price_vs_trama, 2),
            "trend_strength": round(current_tc, 3),  # 0-1, higher = more new HH/LL being made
        }
    except Exception as e:
        logger.error(f"TRAMA calculation error: {e}")
        return None


def _publish_market_data(data: Dict) -> None:
    """
    Publish structured market data to UI via Redis.
    UI receives this directly - no text parsing needed.
    """
    try:
        # Extract fields for UI market data history
        price_data = data.get("price", {})
        orb_data = data.get("orb", {})

        market_data = {
            "time": data.get("timestamp", ""),
            "price": price_data.get("current", 0),
            "vwap": data.get("vwap", 0),
            "rsi": data.get("rsi", 0),
            "orb": f"{orb_data.get('low', 0):.2f}-{orb_data.get('high', 0):.2f}" if orb_data else "--",
            "ema_9": data.get("ema_9", 0),
            "ema_21": data.get("ema_21", 0),
            "macd_hist": data.get("macd", {}).get("histogram", 0),
            "price_vs_vwap": data.get("price_vs_vwap", 0),
            "trama": data.get("trama", {}).get("value", 0),
            "trend_strength": data.get("trama", {}).get("trend_strength", 0),
        }

        # Publish as MARKET_DATA event type
        publish_event("MARKET_DATA", "", market_data)
        logger.debug(f"Published market data: SPY ${market_data['price']:.2f}")
    except Exception as e:
        logger.error(f"Failed to publish market data: {e}")