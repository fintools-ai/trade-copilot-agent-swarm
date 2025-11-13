"""
Standalone invocation of Options Flow Agent
Analyzes options sweeps, blocks, and PUT/CALL bias
"""

import sys
sys.path.insert(0, '/Users/sayantan/Documents/Workspace/trade-copilot-agent-swarm')

from agents.options_flow_agent import create_options_flow_agent


def run_options_flow(ticker: str = "SPY"):
    """
    Run Options Flow Agent standalone

    Args:
        ticker: Ticker to analyze (default: SPY)
    """

    print(f"\n{'='*60}")
    print(f"📈 OPTIONS FLOW AGENT - {ticker}")
    print(f"{'='*60}\n")

    agent = create_options_flow_agent()

    prompt = f"""Analyze options flow for {ticker} to identify smart money positioning.

Analyze:
- Options sweeps (CALL vs PUT)
- Block trades (institutional size)
- Unusual activity
- PUT/CALL ratio and bias
- Premium flow direction
- Target strikes from smart money

Provide directional bias and conviction."""

    response = agent.run(prompt)

    print(response.content[0].text)
    print(f"\n{'='*60}\n")

    return response


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    run_options_flow(ticker)