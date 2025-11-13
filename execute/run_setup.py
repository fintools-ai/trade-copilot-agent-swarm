"""
Standalone invocation of Setup Agent
Configures options monitoring based on OI levels
"""

import sys
sys.path.insert(0, '/Users/sayantan/Documents/Workspace/trade-copilot-agent-swarm')

from agents.setup_agent import create_setup_agent


def run_setup(ticker: str = "SPY"):
    """
    Run Setup Agent standalone

    Args:
        ticker: Ticker to configure monitoring for (default: SPY)
    """

    print(f"\n{'='*60}")
    print(f"⚙️  SETUP AGENT - {ticker}")
    print(f"{'='*60}\n")

    agent = create_setup_agent()

    prompt = f"""Configure options monitoring for {ticker} based on OI key levels.

Note: This requires Market Breadth Agent to have run first and cached OI data.
If OI data is not in cache, request Market Breadth Agent to run first.

Configure monitoring for:
- Core strikes (ATM, Max Pain, Put Wall, Call Wall)
- Range strikes (between walls)
- Extension strikes (breakout levels)"""

    response = agent.run(prompt)

    print(response.content[0].text)
    print(f"\n{'='*60}\n")

    return response


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    run_setup(ticker)