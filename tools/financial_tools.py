"""
Financial Tools for Trade Copilot Agent Swarm
Uses separate MCP client instances to avoid parallel execution conflicts
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, Any
from strands import tool
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

from config.settings import MCP_MARKET_DATA_EXECUTABLE

logger = logging.getLogger(__name__)

def create_mcp_client():
    """Create a new MCP client instance with connection pooling"""
    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command=MCP_MARKET_DATA_EXECUTABLE,
                args=[]
            )
        )
    )

@tool
async def financial_volume_profile_tool(symbol: str) -> str:
    """Analyzes volume profile to identify key price levels."""
    try:
        with create_mcp_client() as mcp:
            result = await mcp.call_tool_async(
                tool_use_id=f"volume_profile_{symbol}",
                name="financial_volume_profile_tool",
                arguments={"symbol": symbol}
            )

            if result and result.get("status") == "success" and result.get("content"):
                return result["content"][0]["text"]
            else:
                return json.dumps({"error": f"No volume profile data available for {symbol}"})

    except Exception as e:
        logger.error(f"Error fetching volume profile for {symbol}: {str(e)}")
        return json.dumps({"error": str(e)})

@tool
async def financial_technical_analysis_tool(symbol: str) -> str:
    """Performs technical analysis using multiple indicators."""
    try:
        with create_mcp_client() as mcp:
            result = await mcp.call_tool_async(
                tool_use_id=f"technical_analysis_{symbol}",
                name="financial_technical_analysis_tool",
                arguments={"symbol": symbol}
            )

            if result and result.get("status") == "success" and result.get("content"):
                return result["content"][0]["text"]
            else:
                return json.dumps({"error": f"No technical analysis data available for {symbol}"})

    except Exception as e:
        logger.error(f"Error fetching technical analysis for {symbol}: {str(e)}")
        return json.dumps({"error": str(e)})

@tool
async def financial_technical_zones_tool(symbol: str) -> str:
    """Identifies support and resistance zones."""
    try:
        with create_mcp_client() as mcp:
            result = await mcp.call_tool_async(
                tool_use_id=f"technical_zones_{symbol}",
                name="financial_technical_zones_tool",
                arguments={"symbol": symbol}
            )

            if result and result.get("status") == "success" and result.get("content"):
                return result["content"][0]["text"]
            else:
                return json.dumps({"error": f"No technical zones data available for {symbol}"})

    except Exception as e:
        logger.error(f"Error fetching technical zones for {symbol}: {str(e)}")
        return json.dumps({"error": str(e)})

@tool
async def financial_orb_analysis_tool(symbol: str) -> str:
    """Analyzes Opening Range Breakout levels."""
    try:
        with create_mcp_client() as mcp:
            result = await mcp.call_tool_async(
                tool_use_id=f"orb_analysis_{symbol}",
                name="financial_orb_analysis_tool",
                arguments={"symbol": symbol}
            )

            if result and result.get("status") == "success" and result.get("content"):
                return result["content"][0]["text"]
            else:
                return json.dumps({"error": f"No ORB analysis data available for {symbol}"})

    except Exception as e:
        logger.error(f"Error fetching ORB analysis for {symbol}: {str(e)}")
        return json.dumps({"error": str(e)})

@tool
async def financial_fvg_analysis_tool(symbol: str) -> str:
    """Detects and analyzes Fair Value Gaps."""
    try:
        with create_mcp_client() as mcp:
            result = await mcp.call_tool_async(
                tool_use_id=f"fvg_analysis_{symbol}",
                name="financial_fvg_analysis_tool",
                arguments={"symbol": symbol}
            )

            if result and result.get("status") == "success" and result.get("content"):
                return result["content"][0]["text"]
            else:
                return json.dumps({"error": f"No FVG analysis data available for {symbol}"})

    except Exception as e:
        logger.error(f"Error fetching FVG analysis for {symbol}: {str(e)}")
        return json.dumps({"error": str(e)})
