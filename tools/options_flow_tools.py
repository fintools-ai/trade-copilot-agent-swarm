"""
Options Flow Tools for Trade Copilot Agent Swarm
Uses Strands @tool decorator for HTTP-based options flow server
"""

import requests
import logging
from typing import List, Dict, Any
from strands import tool

from config.settings import (
    OPTIONS_FLOW_SERVER_URL,
    DEFAULT_TIMEOUT
)

logger = logging.getLogger(__name__)


@tool
async def options_order_flow_tool(ticker: str) -> str:
    """
    Gets options order flow data monitoring for the ticker.

    Use when asked about:
    - Options sweeps and unusual activity
    - PUT vs CALL bias
    - Large block trades
    - Options chain activity
    - Strike-specific flows

    Args:
        ticker: Stock ticker symbol (e.g., SPY, QQQ, AAPL)

    Returns:
        Options order flow analysis with sweeps, blocks, and directional bias
    """
    try:
        url = f"{OPTIONS_FLOW_SERVER_URL}/options_flow"
        params = {"ticker": ticker}

        logger.info(f"Fetching options flow data for {ticker}")

        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)

        if response.status_code == 200:
            return response.text
        else:
            error_msg = f"Error getting options flow data: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    except requests.exceptions.Timeout:
        error_msg = f"Timeout fetching options flow data for {ticker}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Exception getting options flow data for {ticker}: {str(e)}"
        logger.exception(error_msg)
        return f"Error: {error_msg}"


@tool
async def options_monitoring_tool(
    ticker: str,
    expiration: int,
    strike_range: List[float],
    include_both_types: bool = True
) -> str:
    """
    Configures monitoring for specific option strikes to track order flow activity.

    Use when the user wants to:
    - Monitor specific strikes or strike ranges
    - Track PUT/CALL activity for specific levels
    - Set up targeted options surveillance

    Args:
        ticker: Stock ticker symbol (e.g., SPY, QQQ)
        expiration: Expiration date in YYYYMMDD format (e.g., 20250115)
        strike_range: Array of strike prices to monitor (e.g., [580, 585, 590])
        include_both_types: Whether to monitor both PUT and CALL types (default: True)

    Returns:
        Configuration status and monitoring setup confirmation
    """
    try:
        url = f"{OPTIONS_FLOW_SERVER_URL}/options_monitoring"

        configurations = [{
            "expiration": expiration,
            "strike_range": strike_range,
            "include_both_types": include_both_types
        }]

        payload = {
            "ticker": ticker,
            "configurations": configurations
        }

        headers = {"Content-Type": "application/json"}

        logger.info(f"Configuring options monitoring for {ticker} - exp: {expiration}, strikes: {strike_range}")

        response = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)

        if response.status_code == 200:
            return f"""Successfully configured monitoring for {ticker}
Expiration: {expiration}
Strike Range: {strike_range}
Monitoring: {'Both PUTs and CALLs' if include_both_types else 'Single type'}
Status: Active"""
        else:
            error_msg = f"Error configuring options monitoring: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    except requests.exceptions.Timeout:
        error_msg = f"Timeout configuring options monitoring for {ticker}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Exception configuring options monitoring for {ticker}: {str(e)}"
        logger.exception(error_msg)
        return f"Error: {error_msg}"