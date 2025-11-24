"""
Open Interest Tools for Trade Copilot Agent Swarm
Uses Strands MCP client for mcp-openinterest-server integration
"""

import json
import asyncio
import logging
from typing import Dict, Any
from strands import tool
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

from config.settings import MCP_OI_EXECUTABLE

logger = logging.getLogger(__name__)

# Lock to prevent concurrent MCP client access (fixes race condition)
_mcp_lock = asyncio.Lock()

# Initialize MCP client for open interest server
open_interest_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command=MCP_OI_EXECUTABLE,
            args=[]
        )
    )
)


@tool
async def analyze_open_interest_tool(
    ticker: str,
    days: int = 30,
    target_dte: int = 30,
    include_news: bool = True
) -> str:
    """
    Analyzes open interest data for options trading to capture market breadth and institutional positioning.

    CRITICAL FOR 0DTE TRADING: Open Interest reveals:
    - Institutional positioning and smart money flow
    - PUT/CALL ratio and sentiment
    - Strike concentrations (support/resistance magnets)
    - Max pain levels
    - Gamma exposure and squeeze potential
    - Multi-day OI pattern trends

    Use when asked about:
    - Market breadth and overall sentiment
    - Institutional positioning across strikes
    - Options support/resistance levels
    - Max pain calculations
    - PUT/CALL ratios and changes
    - Unusual options activity clusters
    - Multi-ticker OI pattern analysis
    - Smart money positioning

    Args:
        ticker: Stock ticker symbol (e.g., SPY, AAPL, QQQ)
        days: Number of days to analyze for pattern detection (default: 30)
        target_dte: Target days to expiration for analysis (default: 30)
        include_news: Include news context in analysis (default: True)

    Returns:
        Open interest analysis with:
        - PUT/CALL ratios and shifts
        - Strike-level OI concentrations
        - Max pain calculations
        - Large OI increases (institutional blocks)
        - Historical pattern trends
        - News context (if requested)
    """
    try:
        # Use lock to prevent concurrent access to MCP client
        async with _mcp_lock:
            with open_interest_mcp:
                result = await open_interest_mcp.call_tool_async(
                    tool_use_id=f"oi_{ticker}_{target_dte}",
                    name="analyze_open_interest",
                    arguments={
                        "ticker": ticker,
                        "days": days,
                        "target_dte": target_dte,
                        "include_news": include_news
                    }
                )

                if result and result.get("status") == "success" and result.get("content"):
                    # Extract text from MCP response
                    content = result["content"][0]["text"]
                    return content
                else:
                    error_msg = f"No open interest data available for {ticker}"
                    logger.error(error_msg)
                    return json.dumps({"error": error_msg})

    except Exception as e:
        error_msg = f"Error fetching open interest for {ticker}: {str(e)}"
        logger.exception(error_msg)
        return json.dumps({"error": error_msg})


@tool
async def analyze_multi_ticker_oi_breadth(
    tickers: list[str],
    days: int = 30,
    target_dte: int = 30
) -> str:
    """
    Analyzes open interest across multiple tickers to capture market breadth.

    MARKET BREADTH ANALYSIS: Critical for understanding:
    - Sector-wide sentiment shifts
    - Broad market positioning vs individual stock divergence
    - Correlation of institutional flows across mega-caps
    - Market regime detection (risk-on vs risk-off)

    Use when asked about:
    - Overall market sentiment and breadth
    - Sector rotation signals
    - Divergence between stocks
    - Broad institutional positioning

    Args:
        tickers: List of ticker symbols (e.g., ["SPY", "AAPL", "MSFT", "NVDA"])
        days: Number of days for pattern analysis (default: 30)
        target_dte: Target DTE for analysis (default: 30)

    Returns:
        Aggregated open interest analysis across all tickers with breadth metrics
    """
    try:
        results = {}
        errors = []

        # Use lock to prevent concurrent access to MCP client
        async with _mcp_lock:
            with open_interest_mcp:
                # Fetch OI data for all tickers
                for ticker in tickers:
                    try:
                        logger.info(f"Fetching OI breadth data for {ticker}")
                        result = await open_interest_mcp.call_tool_async(
                            tool_use_id=f"oi_breadth_{ticker}_{target_dte}",
                            name="analyze_open_interest",
                            arguments={
                                "ticker": ticker,
                                "days": days,
                                "target_dte": target_dte,
                                "include_news": False  # Skip news for breadth analysis
                            }
                        )

                        if result and result.get("status") == "success" and result.get("content"):
                            content = result["content"][0]["text"]
                            results[ticker] = json.loads(content)
                        else:
                            errors.append(f"{ticker}: No data available")

                    except Exception as e:
                        logger.error(f"Error fetching OI for {ticker}: {str(e)}")
                        errors.append(f"{ticker}: {str(e)}")

        # Format breadth analysis response
        if not results:
            return json.dumps({
                "error": "Failed to retrieve OI data for any ticker",
                "errors": errors
            })

        breadth_response = {
            "market_breadth_analysis": {
                "tickers_analyzed": list(results.keys()),
                "failed_tickers": errors,
                "individual_oi_data": results,
                "breadth_summary": {
                    "total_tickers": len(tickers),
                    "successful": len(results),
                    "failed": len(errors)
                }
            },
            "usage_note": """
            Analyze PUT/CALL ratios across all tickers for market breadth:
            - High PUT bias across multiple tickers = broad bearish sentiment
            - Divergence (some bullish, some bearish) = stock-specific plays
            - Consistent CALL bias = strong bullish momentum
            - Look for OI concentration patterns across mega-caps
            """
        }

        return json.dumps(breadth_response, indent=2)

    except Exception as e:
        error_msg = f"Error in multi-ticker OI breadth analysis: {str(e)}"
        logger.exception(error_msg)
        return json.dumps({"error": error_msg})