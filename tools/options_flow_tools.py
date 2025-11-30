"""
Options Flow Tools for Trade Copilot Agent Swarm
Uses Strands @tool decorator for HTTP-based options flow server
"""

import json
import requests
import logging
from typing import List
from strands import tool

from config.settings import (
    OPTIONS_FLOW_SERVER_URL,
    DEFAULT_TIMEOUT
)

logger = logging.getLogger(__name__)


@tool
async def options_order_flow_tool(ticker: str) -> str:
    """
    Gets real-time options order flow data for the ticker.

    Use when asked about:
    - Options activity and flow
    - PUT vs CALL pressure
    - Strike-specific options activity
    - Options order flow signals

    Args:
        ticker: Stock ticker symbol (e.g., SPY, QQQ, AAPL)

    Returns:
        Options order flow data with per-strike metrics:
        - bid_lifts/bid_drops: How many times bid price moved up/down
        - ask_lifts/ask_drops: How many times ask price moved up/down
        - bid_volume/ask_volume: Accumulated depth on each side
        - Windows: 5s (scalping), 15s (momentum), 60s (trend)

        Interpretation:
        - call.bid_lifts > call.bid_drops = Call buyers stepping up (BULLISH)
        - put.bid_lifts > put.bid_drops = Put buyers stepping up (BEARISH)
        - High activity at specific strike = Smart money interest
    """
    try:
        url = f"{OPTIONS_FLOW_SERVER_URL}/flow"
        params = {"ticker": ticker.upper()}

        logger.info(f"Fetching options flow data for {ticker}")

        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)

        if response.status_code == 200:
            return response.text
        elif response.status_code == 202:
            # Waiting for data
            data = response.json()
            return f"Waiting for data: {data.get('message', 'Connecting...')}"
        elif response.status_code == 404:
            return f"No options data for {ticker}. Use options_subscribe_tool first to subscribe to strikes."
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
async def options_subscribe_tool(
    ticker: str,
    expiration: int,
    strikes: List[float]
) -> str:
    """
    Subscribes to options contracts for real-time flow monitoring.
    Automatically monitors both CALL and PUT for each strike.

    Use when:
    - Setting up options monitoring for a ticker
    - Configuring which strikes to track
    - Starting a new trading session

    Args:
        ticker: Stock ticker symbol (e.g., SPY, QQQ)
        expiration: Expiration date in YYYYMMDD format (e.g., 20250115)
        strikes: List of strike prices to monitor (e.g., [580, 585, 590])

    Returns:
        Subscription status with list of contracts being monitored
    """
    try:
        url = f"{OPTIONS_FLOW_SERVER_URL}/subscribe"

        payload = {
            "ticker": ticker.upper(),
            "expiration": expiration,
            "strikes": strikes
        }

        headers = {"Content-Type": "application/json"}

        logger.info(f"Subscribing to options for {ticker} - exp: {expiration}, strikes: {strikes}")

        response = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            contracts = data.get("contracts", [])
            return f"""Successfully subscribed to options flow for {ticker}
Expiration: {expiration}
Strikes: {strikes}
Contracts: {len(contracts)} (CALL + PUT for each strike)
Status: Active - data will be available in a few seconds"""
        else:
            error_msg = f"Error subscribing to options: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    except requests.exceptions.Timeout:
        error_msg = f"Timeout subscribing to options for {ticker}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Exception subscribing to options for {ticker}: {str(e)}"
        logger.exception(error_msg)
        return f"Error: {error_msg}"
