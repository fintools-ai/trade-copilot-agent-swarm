"""
Standalone invocation of Market Breadth Agent
Analyzes open interest and identifies key levels
"""

import sys
import os
import asyncio

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agents.market_breadth_agent import create_market_breadth_agent


async def run_market_breadth(ticker: str = "SPY"):
    """
    Run Market Breadth Agent standalone

    Args:
        ticker: Ticker to analyze (default: SPY)
    """

    print(f"\n{'='*60}")
    print(f"📊 MARKET BREADTH AGENT - {ticker}")
    print(f"{'='*60}\n")

    agent = create_market_breadth_agent()

    prompt = f"""Analyze open interest breadth for {ticker} and identify key levels for 1DTE trading today.

Provide:
- Max Pain level
- Put Wall (support)
- Call Wall (resistance)
- Current price context
- Trading implications"""

    response = await agent.invoke_async(prompt)

    print(response.message["content"][0]["text"])
    print(f"\n{'='*60}\n")

    return response


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    asyncio.run(run_market_breadth(ticker))