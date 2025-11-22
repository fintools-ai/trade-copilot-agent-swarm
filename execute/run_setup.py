"""
Standalone invocation of Setup Agent
Configures options monitoring based on OI levels
"""

import sys
import os
import asyncio

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agents.setup_agent import create_setup_agent


async def run_setup(ticker: str = "SPY"):
    """
    Run Setup Agent standalone

    Args:
        ticker: Ticker to configure monitoring for (default: SPY)
    """

    print(f"\n{'='*60}")
    print(f"⚙️  SETUP AGENT - {ticker}")
    print(f"{'='*60}\n")

    agent = create_setup_agent()

    prompt = f"""Monitor for {ticker} for range of 670 to 675 for today."""

    response = await agent.invoke_async(prompt)

    print(response.message["content"][0]["text"])
    print(f"\n{'='*60}\n")

    return response


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    asyncio.run(run_setup(ticker))