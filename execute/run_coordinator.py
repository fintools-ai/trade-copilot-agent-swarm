"""
Standalone invocation of Coordinator Agent
Synthesizes all specialist insights into dual CALL/PUT recommendations
"""

import sys
sys.path.insert(0, '/Users/sayantan/Documents/Workspace/trade-copilot-agent-swarm')

from agents.coordinator_agent import create_coordinator_agent


def run_coordinator(ticker: str = "SPY"):
    """
    Run Coordinator Agent standalone

    Args:
        ticker: Ticker to analyze (default: SPY)
    """

    print(f"\n{'='*60}")
    print(f"🎯 COORDINATOR AGENT - {ticker}")
    print(f"{'='*60}\n")

    agent = create_coordinator_agent()

    prompt = f"""Synthesize all agent insights and provide dual 0DTE recommendations for {ticker}.

Note: This requires all specialist agents to have run first and cached their analysis:
- Market Breadth Agent (oi_breadth_data)
- Order Flow Agent (order_flow_analysis)
- Options Flow Agent (options_flow_analysis)
- Financial Data Agent (financial_data_analysis)

Provide TWO separate recommendations:
1. 0DTE CALL recommendation with conviction score (HIGH/MEDIUM/LOW)
2. 0DTE PUT recommendation with conviction score (HIGH/MEDIUM/LOW)

Include:
- Bullish/Bearish signals from each agent
- Agent alignment count
- Entry/Target/Stop levels
- Strike recommendations
- Risk/Reward ratios
- Final recommendation (which setup is best)"""

    response = agent.run(prompt)

    print(response.content[0].text)
    print(f"\n{'='*60}\n")

    return response


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    run_coordinator(ticker)