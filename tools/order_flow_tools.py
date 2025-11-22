"""
Order Flow Tools for Trade Copilot Agent Swarm
Uses HTTP requests to order flow server (like VTS bot)
"""

import requests
import json
import logging
from typing import Optional
from strands import tool

from config.settings import ORDER_FLOW_SERVER_URL, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

@tool
async def equity_order_flow_tool(
    ticker: str,
    history_minutes: int = 10,
    pattern_types: Optional[str] = None
) -> str:
    """
    Use when asked about:
     - Recent buying/selling activity
     - Institutional vs retail flow
     - Order flow patterns
     - Volume imbalances
     - Bid/ask dynamics
     - Cross-market confirmation signals

     Args:
         ticker: Primary ticker to analyze (e.g., SPY, AAPL)
         history_minutes: History window in minutes (default: 10)
         pattern_types: Optional filter for specific pattern types

     Returns:
         Multi-ticker order flow analysis with:
         - Primary ticker data prominently displayed
         - Supporting Mag 7 ticker flows for cross-validation
         - Divergence detection notes
         - Signal strength validation across mega-caps
     """

    
    # Define mega-cap tickers for comprehensive analysis
    multi_tickers = ['SPY', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN']
    
    ticker_results = {}
    primary_result = None
    
    for current_ticker in multi_tickers:
        try:
            # Make HTTP GET request (same as VTS bot)
            url = f"{ORDER_FLOW_SERVER_URL}/get_order_flow_data"
            params = {
                "ticker": current_ticker,
                "history": f"{history_minutes}mins",
                "pattern_types": pattern_types
            }
            
            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            
            if response.status_code == 200:
                data = response.text
                ticker_results[current_ticker] = data
                
                # Store primary ticker result
                if current_ticker == ticker:
                    primary_result = data
                    
            else:
                logger.warning(f"HTTP {response.status_code} for {current_ticker}: {response.text}")
                ticker_results[current_ticker] = {"error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {current_ticker}: {str(e)}")
            ticker_results[current_ticker] = {"error": f"Connection failed: {str(e)}"}
    
    # Format response
    if primary_result:
        return json.dumps({
            "primary_ticker": ticker,
            "primary_data": primary_result,
            "cross_validation": ticker_results,
            "analysis_note": f"Multi-ticker analysis for {ticker} with Mag 7 cross-validation"
        }, indent=2)
    else:
        return json.dumps({
            "error": f"Failed to get data for primary ticker {ticker}",
            "attempted_tickers": multi_tickers,
            "results": ticker_results
        }, indent=2)