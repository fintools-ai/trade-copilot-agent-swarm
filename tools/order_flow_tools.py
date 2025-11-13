"""
Order Flow Tools for Trade Copilot Agent Swarm
Uses Strands @tool decorator for HTTP-based order flow server
"""

import requests
import logging
from typing import Optional
from strands import tool

from config.settings import (
    ORDER_FLOW_SERVER_URL,
    DEFAULT_TIMEOUT
)

logger = logging.getLogger(__name__)


@tool
async def equity_order_flow_tool(
    ticker: str,
    history_minutes: int = 10,
    pattern_types: Optional[str] = None
) -> str:
    """
    Gets real-time equity order flow data for Mag 7 mega-caps with multi-ticker analysis.

    IMPORTANT: Always fetches order flow for all Mag 7 tickers (SPY, AAPL, MSFT, NVDA, GOOGL, AMZN)
    to enable cross-validation and divergence detection critical for 0DTE trading.

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
    # Define mega-cap tickers for comprehensive 0DTE analysis
    # These represent ~35% of SPY weight and drive 55-65% of daily volatility
    multi_tickers = ['SPY', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN']

    ticker_results = {}
    primary_result = None

    # Always analyze all Mag 7 tickers regardless of what was requested
    for t in multi_tickers:
        try:
            url = f"{ORDER_FLOW_SERVER_URL}/get_order_flow_data"
            params = {
                "ticker": t,
                "history": f"{history_minutes}mins"
            }

            if pattern_types:
                params["pattern_types"] = pattern_types

            logger.info(f"Fetching order flow data for {t}")

            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)

            if response.status_code == 200:
                ticker_results[t] = response.text
                # Keep the originally requested ticker as primary
                if t == ticker:
                    primary_result = response.text
            else:
                logger.error(f"Error getting order flow for {t}: {response.status_code}")

        except Exception as e:
            logger.error(f"Error fetching order flow for {t}: {str(e)}")

    # Return error if no data retrieved
    if not ticker_results:
        return "Error: Failed to retrieve any order flow data"

    # Format response with primary ticker prominently and supporting data
    if primary_result:
        import json
        supporting_data = {k: v for k, v in ticker_results.items() if k != ticker}

        return f"""<multi_ticker_order_flow_analysis primary="{ticker}">
<analysis_context>
Multi-ticker order flow monitoring active for enhanced 0DTE trading signals.
Tracking {len(ticker_results)} mega-caps that drive 55-65% of SPY daily volatility.
Primary ticker: {ticker} | Supporting analysis: {', '.join(supporting_data.keys())}
</analysis_context>

<primary_ticker_data ticker="{ticker}">
{primary_result}
</primary_ticker_data>

<supporting_tickers>
{json.dumps(supporting_data, indent=2)}
</supporting_tickers>

<usage_note>
Cross-examine {ticker} signals with mega-cap flows above. Look for:
- Confirming weakness/strength across multiple mega-caps
- Divergence signals when individual stocks contradict primary ticker flow
- Cross-validation of momentum shifts across major market constituents
- Signal strength validation through multi-ticker consensus
</usage_note>
</multi_ticker_order_flow_analysis>"""

    # Fallback if primary ticker not available
    import json
    return f"""<multi_ticker_order_flow_analysis requested="{ticker}" status="fallback">
<analysis_context>
Primary ticker {ticker} unavailable. Showing available mega-cap order flows for market context.
Cross-examine flows across all available tickers for comprehensive analysis.
</analysis_context>

<available_ticker_data>
{json.dumps(ticker_results, indent=2)}
</available_ticker_data>
</multi_ticker_order_flow_analysis>"""