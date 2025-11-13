"""
Standalone invocation of Order Flow Agent
Analyzes multi-ticker equity order flow and institutional patterns
"""

import sys
sys.path.insert(0, '/Users/sayantan/Documents/Workspace/trade-copilot-agent-swarm')

from agents.order_flow_agent import create_order_flow_agent


def run_order_flow(ticker: str = "SPY"):
    """
    Run Order Flow Agent standalone

    Args:
        ticker: Primary ticker to analyze (default: SPY)
    """

    print(f"\n{'='*60}")
    print(f"💹 ORDER FLOW AGENT - {ticker}")
    print(f"{'='*60}\n")

    agent = create_order_flow_agent()

    prompt = f"""Analyze order flow for {ticker} and Mag 7 tickers to detect institutional patterns.

Analyze:
- Buy/Sell imbalances
- Institutional activity (absorption, stacking, sweeps)
- Volume confirmation
- Cross-ticker validation
- Key levels from order flow

Provide intraday bias and key levels."""

    response = agent.run(prompt)

    print(response.content[0].text)
    print(f"\n{'='*60}\n")

    return response


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    run_order_flow(ticker)