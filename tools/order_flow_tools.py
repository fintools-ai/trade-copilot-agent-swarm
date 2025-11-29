"""
Order Flow Tools for Trade Copilot Agent Swarm

Connects to order-flow-server which provides real-time quote data.
Server does math (counts, averages). LLM does interpretation.
"""

import requests
import json
import logging
from strands import tool

from config.settings import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

# Order flow server base URL (no /api suffix)
ORDER_FLOW_BASE_URL = "http://localhost:8300"


@tool
async def equity_order_flow_tool(ticker: str) -> str:
    """
    Get real-time order flow data for 0DTE trading decisions.

    Use when asked about:
     - Recent buying/selling pressure
     - Bid/ask dynamics and imbalances
     - Order flow patterns
     - Cross-market confirmation signals

    Args:
        ticker: Primary ticker to analyze (e.g., SPY)

    Returns:
        Order flow metrics for SPY + Mag7 tickers with:
        - bid_lifts/bid_drops: How many times bid price moved up/down
        - ask_lifts/ask_drops: How many times ask price moved up/down
        - bid_size_avg/ask_size_avg: Average size on each side
        - Windows: 5s (scalping), 15s (momentum), 60s (trend)

        Interpretation guide:
        - More bid_lifts than drops = buyers stepping up (bullish)
        - More ask_drops than lifts = sellers backing off (bullish)
        - Higher bid_size_avg than ask = more support (bullish)
        - Opposite patterns = bearish
    """
    try:
        # Get all tickers in one request
        url = f"{ORDER_FLOW_BASE_URL}/flow/all"
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)

        if response.status_code == 200:
            all_data = response.json()

            # Extract primary ticker
            ticker = ticker.upper()
            primary_data = all_data.get(ticker)

            if not primary_data:
                return json.dumps({
                    "error": f"Ticker {ticker} not tracked",
                    "available": list(all_data.keys())
                })

            return json.dumps({
                "primary_ticker": ticker,
                "primary_data": primary_data,
                "cross_validation": all_data
            }, indent=2)
        else:
            logger.warning(f"HTTP {response.status_code}: {response.text}")
            return json.dumps({"error": f"HTTP {response.status_code}"})

    except requests.exceptions.RequestException as e:
        logger.error(f"Order flow request failed: {str(e)}")
        return json.dumps({"error": f"Connection failed: {str(e)}"})