"""
Financial Data Tools for Trade Copilot Agent Swarm
Uses Strands MCP client for mcp-market-data-server integration
"""

import json
import logging
from typing import Dict, Any
from strands import tool
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

logger = logging.getLogger(__name__)

# Initialize MCP client for market data server
market_data_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="mcp-market-data-server",
            args=[]
        )
    )
)


@tool
async def financial_volume_profile_tool(symbol: str) -> str:
    """
    Retrieves Volume Profile structure (POC, VAH, VAL, Nodes) for granular timeframes (1m, 5m).

    Use when asked about:
    - Volume-based support/resistance levels
    - High/low volume concentration areas
    - Point of Control (POC) for different timeframes
    - Value Area High/Low (VAH/VAL)

    Args:
        symbol: Stock ticker symbol (e.g., SPY, AAPL, QQQ)

    Returns:
        Volume profile analysis with POC, VAH, VAL, and high/low volume nodes
    """
    try:
        with market_data_mcp:
            result = await market_data_mcp.call_tool(
                "financial_volume_profile_tool",
                {"symbol": symbol}
            )

            if result and "content" in result and result["content"]:
                # Extract text from MCP response
                content = result["content"][0]["text"]
                return content
            else:
                error_msg = f"No volume profile data available for {symbol}"
                logger.error(error_msg)
                return json.dumps({"error": error_msg})

    except Exception as e:
        error_msg = f"Error fetching volume profile for {symbol}: {str(e)}"
        logger.exception(error_msg)
        return json.dumps({"error": error_msg})


@tool
async def financial_technical_analysis_tool(symbol: str) -> str:
    """
    Retrieves comprehensive Technical Analysis indicators (SMA, RSI, MACD, ATR, VWAP) for timeframes (1m, 5m, 1d).

    Use when asked about:
    - Trend strength and momentum
    - Moving averages (SMA, EMA)
    - Oscillators (RSI, MACD, Stochastic)
    - Volatility (ATR)
    - Volume indicators (VWAP, OBV)

    Args:
        symbol: Stock ticker symbol (e.g., SPY, NVDA, TSLA)

    Returns:
        Technical indicators across multiple timeframes with trend analysis
    """
    try:
        with market_data_mcp:
            result = await market_data_mcp.call_tool(
                "financial_technical_analysis_tool",
                {"symbol": symbol}
            )

            if result and "content" in result and result["content"]:
                content = result["content"][0]["text"]
                return content
            else:
                error_msg = f"No technical analysis data available for {symbol}"
                logger.error(error_msg)
                return json.dumps({"error": error_msg})

    except Exception as e:
        error_msg = f"Error fetching technical analysis for {symbol}: {str(e)}"
        logger.exception(error_msg)
        return json.dumps({"error": error_msg})


@tool
async def financial_technical_zones_tool(symbol: str) -> str:
    """
    Retrieves calculated support/resistance price zones from Volume Profile and volatility for timeframes (1m, 5m).

    Use when asked about:
    - Precise entry/exit levels
    - Support and resistance zones
    - Stop-loss placement
    - High-probability price levels

    Args:
        symbol: Stock ticker symbol (e.g., SPY, AAPL)

    Returns:
        Support/resistance zones with strength and confidence levels
    """
    try:
        with market_data_mcp:
            result = await market_data_mcp.call_tool(
                "financial_technical_zones_tool",
                {"symbol": symbol}
            )

            if result and "content" in result and result["content"]:
                content = result["content"][0]["text"]
                return content
            else:
                error_msg = f"No technical zones data available for {symbol}"
                logger.error(error_msg)
                return json.dumps({"error": error_msg})

    except Exception as e:
        error_msg = f"Error fetching technical zones for {symbol}: {str(e)}"
        logger.exception(error_msg)
        return json.dumps({"error": error_msg})


@tool
async def financial_orb_analysis_tool(symbol: str) -> str:
    """
    Analyzes Opening Range Breakout (ORB) levels for multiple timeframes (5, 15, 30 minutes).

    Use when asked about:
    - 0DTE trading strategies
    - Intraday breakout levels
    - Opening range high/low
    - Breakout confirmation with volume
    - Extension targets

    Args:
        symbol: Stock ticker symbol (e.g., SPY, QQQ)

    Returns:
        ORB analysis with breakout status, volume confirmation, and extension targets
    """
    try:
        with market_data_mcp:
            result = await market_data_mcp.call_tool(
                "financial_orb_analysis_tool",
                {"symbol": symbol}
            )

            if result and "content" in result and result["content"]:
                content = result["content"][0]["text"]
                return content
            else:
                error_msg = f"No ORB analysis data available for {symbol}"
                logger.error(error_msg)
                return json.dumps({"error": error_msg})

    except Exception as e:
        error_msg = f"Error fetching ORB analysis for {symbol}: {str(e)}"
        logger.exception(error_msg)
        return json.dumps({"error": error_msg})


@tool
async def financial_fvg_analysis_tool(symbol: str) -> str:
    """
    Analyzes Fair Value Gaps (FVGs) across multiple timeframes (1m, 5m, 15m).

    Use when asked about:
    - Price imbalances and gaps
    - Mean reversion opportunities
    - Unfilled gaps as support/resistance
    - 3-candle pattern gaps
    - Gap fill probability

    Args:
        symbol: Stock ticker symbol (e.g., NVDA, TSLA)

    Returns:
        FVG analysis with gap levels, fill status, volume data, and nearest gaps
    """
    try:
        with market_data_mcp:
            result = await market_data_mcp.call_tool(
                "financial_fvg_analysis_tool",
                {"symbol": symbol}
            )

            if result and "content" in result and result["content"]:
                content = result["content"][0]["text"]
                return content
            else:
                error_msg = f"No FVG analysis data available for {symbol}"
                logger.error(error_msg)
                return json.dumps({"error": error_msg})

    except Exception as e:
        error_msg = f"Error fetching FVG analysis for {symbol}: {str(e)}"
        logger.exception(error_msg)
        return json.dumps({"error": error_msg})