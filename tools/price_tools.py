"""
Price Tools for Trade Copilot Agent Swarm
Lightweight tool to fetch current price using Twelve Data API
"""

import json
import logging
from datetime import datetime
from strands import tool
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient
from config.settings import TWELVE_DATA_API_KEY

logger = logging.getLogger(__name__)


def _create_twelvedata_mcp():
    """Create Twelve Data MCP client"""
    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx",
                args=["mcp-server-twelve-data", "-k", TWELVE_DATA_API_KEY, "-n", "10"]
            )
        )
    )


@tool
async def get_current_price(ticker: str) -> str:
    """
    Gets the current price for a ticker symbol.

    Use this tool to:
    - Get the current/latest price for any stock
    - Determine ATM (at-the-money) strike level
    - Know current price relative to key OI levels

    Args:
        ticker: Stock ticker symbol (e.g., SPY, AAPL, NVDA)

    Returns:
        JSON with current price data:
        - price: current price
        - change: dollar change from previous close
        - change_pct: percentage change
        - timestamp: when the data was fetched
    """
    try:
        with _create_twelvedata_mcp() as mcp:
            quote = await mcp.call_tool_async(
                tool_use_id=f"quote_{ticker}",
                name="GetQuote",
                arguments={"params": {"symbol": ticker.upper()}}
            )

            if quote and quote.get("status") == "success" and quote.get("content"):
                content = quote["content"]
                if isinstance(content, list) and content:
                    q = json.loads(content[0].get("text", "{}"))

                    result = {
                        "ticker": ticker.upper(),
                        "price": float(q.get("close", 0)),
                        "change": float(q.get("change", 0)),
                        "change_pct": float(q.get("percent_change", 0)),
                        "open": float(q.get("open", 0)),
                        "high": float(q.get("high", 0)),
                        "low": float(q.get("low", 0)),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    return json.dumps(result, indent=2)

            return json.dumps({
                "error": f"Could not fetch price for {ticker}",
                "ticker": ticker.upper(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    except Exception as e:
        logger.error(f"Error fetching price for {ticker}: {e}")
        return json.dumps({
            "error": str(e),
            "ticker": ticker.upper(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })