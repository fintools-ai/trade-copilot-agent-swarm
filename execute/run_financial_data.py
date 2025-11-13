"""
Standalone invocation of Financial Data Agent
Analyzes technical indicators, volume profile, ORB, and FVG
"""

import sys
sys.path.insert(0, '/Users/sayantan/Documents/Workspace/trade-copilot-agent-swarm')

from agents.financial_data_agent import create_financial_data_agent


def run_financial_data(ticker: str = "SPY"):
    """
    Run Financial Data Agent standalone

    Args:
        ticker: Ticker to analyze (default: SPY)
    """

    print(f"\n{'='*60}")
    print(f"📊 FINANCIAL DATA AGENT - {ticker}")
    print(f"{'='*60}\n")

    agent = create_financial_data_agent()

    prompt = f"""Perform technical analysis for {ticker} for intraday trading.

Analyze:
- Volume Profile (POC, VAH, VAL)
- Technical Indicators (RSI, MACD, MAs)
- Support/Resistance Zones
- Opening Range Breakout (ORB)
- Fair Value Gaps (FVG)

Provide technical bias and key intraday levels."""

    response = agent.run(prompt)

    print(response.content[0].text)
    print(f"\n{'='*60}\n")

    return response


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    run_financial_data(ticker)