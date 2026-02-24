"""
Zero-DTE v2 — Learning Trading Engine

Code classifies, Claude synthesizes, Memory learns.

Architecture:
1. Python classifies raw data → categorical labels (~1ms)
2. One Claude Sonnet 4.5 call per cycle (~2s)
3. Bedrock AgentCore stores outcomes, recalls similar situations, extracts patterns daily

Usage:
    python zero_dte_v2.py
"""

import asyncio
import logging

from trading_engine.engine import TradingEngine


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\033[95m")  # Purple
    print("""
████████╗██████╗  █████╗ ██████╗ ███████╗    ██╗      ██████╗  ██████╗ ██████╗ 
╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝    ██║     ██╔═══██╗██╔═══██╗██╔══██╗
   ██║   ██████╔╝███████║██║  ██║█████╗      ██║     ██║   ██║██║   ██║██████╔╝
   ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝      ██║     ██║   ██║██║   ██║██╔═══╝ 
   ██║   ██║  ██║██║  ██║██████╔╝███████╗    ███████╗╚██████╔╝╚██████╔╝██║     
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝    ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     
                                                                                 
        Code classifies ··· Claude synthesizes ··· Memory learns
    """)
    print("\033[0m")  # Reset

    engine = TradingEngine()
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
