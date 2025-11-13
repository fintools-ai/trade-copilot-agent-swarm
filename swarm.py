"""
Trade Copilot Agent Swarm - Conversational Trading Assistant
Multi-agent system for 0DTE/1DTE options trading recommendations using Strands Graph

Usage:
    from swarm import TradingSwarm

    swarm = TradingSwarm()
    swarm.ask("What does SPY look like now?")
    swarm.ask("Should I trade NVDA today?")
    swarm.ask("Give me CALL and PUT recommendations for SPY")
"""

import re
from datetime import datetime
from pathlib import Path
from strands.multiagent.graph import Graph, GraphBuilder, GraphNode
from strands.session.file_session_manager import FileSessionManager

from agents.market_breadth_agent import create_market_breadth_agent
from agents.setup_agent import create_setup_agent
from agents.order_flow_agent import create_order_flow_agent
from agents.options_flow_agent import create_options_flow_agent
from agents.financial_data_agent import create_financial_data_agent
from agents.coordinator_agent import create_coordinator_agent


class TradingSwarm:
    """
    Conversational trading swarm that orchestrates multiple specialist agents
    to provide 0DTE/1DTE trading recommendations

    Architecture:
        MarketBreadth → Setup → [OrderFlow, OptionsFlow, FinancialData] → Coordinator
                                      (parallel execution)

    Session Management:
        - Uses FileSessionManager to persist conversation history and agent state
        - Default session_id is trading date (e.g., "trading-2025-01-15")
        - Sessions stored in ./sessions/ directory
        - All agent conversations and analysis results are cached
    """

    def __init__(self, session_id: str = None, storage_dir: str = None):
        """
        Initialize the trading swarm graph with session management

        Args:
            session_id: Unique session identifier (default: "trading-YYYY-MM-DD")
            storage_dir: Directory to store sessions (default: "./sessions")
        """

        # Set default session_id based on trading date
        if session_id is None:
            trading_date = datetime.now().strftime("%Y-%m-%d")
            session_id = f"trading-{trading_date}"

        # Set default storage directory
        if storage_dir is None:
            storage_dir = str(Path(__file__).parent / "sessions")

        # Create session manager
        self.session_manager = FileSessionManager(
            session_id=session_id,
            storage_dir=storage_dir
        )

        self.session_id = session_id
        self.graph = self._build_graph()

        print(f"💾 Session: {session_id}")
        print(f"📁 Storage: {storage_dir}\n")

    def _build_graph(self) -> Graph:
        """
        Build the multi-agent graph with Strands GraphBuilder and session management

        Returns:
            Configured Strands Graph with FileSessionManager
        """

        # Initialize all agents (without session managers - only Graph gets it)
        market_breadth_agent = create_market_breadth_agent()
        setup_agent = create_setup_agent()
        order_flow_agent = create_order_flow_agent()
        options_flow_agent = create_options_flow_agent()
        financial_data_agent = create_financial_data_agent()
        coordinator_agent = create_coordinator_agent()

        # Build the graph
        builder = GraphBuilder()

        # STEP 1: Market Breadth - Analyzes OI, caches key levels
        builder.add_node(
            "market_breadth",
            GraphNode(
                name="market_breadth",
                agent=market_breadth_agent,
                timeout=60.0
            )
        )

        # STEP 2: Setup - Configures monitoring based on OI
        builder.add_node(
            "setup",
            GraphNode(
                name="setup",
                agent=setup_agent,
                timeout=30.0
            )
        )

        # STEP 3a: Order Flow Specialist
        builder.add_node(
            "order_flow",
            GraphNode(
                name="order_flow",
                agent=order_flow_agent,
                timeout=60.0
            )
        )

        # STEP 3b: Options Flow Specialist
        builder.add_node(
            "options_flow",
            GraphNode(
                name="options_flow",
                agent=options_flow_agent,
                timeout=60.0
            )
        )

        # STEP 3c: Financial Data Specialist
        builder.add_node(
            "financial_data",
            GraphNode(
                name="financial_data",
                agent=financial_data_agent,
                timeout=60.0
            )
        )

        # STEP 4: Coordinator - Synthesizes everything
        builder.add_node(
            "coordinator",
            GraphNode(
                name="coordinator",
                agent=coordinator_agent,
                timeout=60.0
            )
        )

        # Define edges (execution dependencies)
        # Sequential: market_breadth → setup
        builder.add_edge("market_breadth", "setup")

        # Parallel: setup → all three specialists
        builder.add_edge("setup", "order_flow")
        builder.add_edge("setup", "options_flow")
        builder.add_edge("setup", "financial_data")

        # Convergence: all specialists → coordinator
        builder.add_edge("order_flow", "coordinator")
        builder.add_edge("options_flow", "coordinator")
        builder.add_edge("financial_data", "coordinator")

        # Build graph with session manager
        return builder.build(session_manager=self.session_manager)

    def ask(self, query: str) -> str:
        """
        Ask the trading swarm a question about trading opportunities

        Examples:
            "What does SPY look like now?"
            "Should I trade NVDA today?"
            "Give me CALL and PUT recommendations for AAPL"
            "Analyze SPY for 0DTE trading"

        Args:
            query: Natural language question about trading

        Returns:
            Final recommendation from the Coordinator Agent
        """

        # Extract ticker from query (defaults to SPY)
        ticker = self._extract_ticker(query)
        trading_date = datetime.now().strftime("%Y-%m-%d")

        # Build comprehensive prompt for the graph
        graph_prompt = f"""USER QUERY: {query}

TICKER: {ticker}
DATE: {trading_date}

INSTRUCTIONS FOR EACH AGENT:

[MARKET BREADTH AGENT]
Analyze open interest breadth for {ticker} and identify key levels (max pain, put wall, call wall) for 1DTE trading today. Cache this data for the session.

[SETUP AGENT]
Configure options monitoring for {ticker} based on the OI key levels identified by the Market Breadth Agent.

[ORDER FLOW AGENT]
Analyze order flow for {ticker} and Mag 7 tickers to detect institutional patterns, volume imbalances, and intraday trading signals.

[OPTIONS FLOW AGENT]
Analyze options flow for {ticker} to identify smart money positioning, sweeps, blocks, and PUT/CALL bias.

[FINANCIAL DATA AGENT]
Perform technical analysis for {ticker} including volume profile, technical indicators, ORB (Opening Range Breakout), and FVG (Fair Value Gaps).

[COORDINATOR AGENT]
Synthesize all specialist insights and provide TWO separate 0DTE recommendations:
1. CALL recommendation with conviction score (HIGH/MEDIUM/LOW)
2. PUT recommendation with conviction score (HIGH/MEDIUM/LOW)

Cross-validate signals across all 4 agents, identify the best setup, and provide actionable entry/exit/stop levels."""

        # Print execution header
        print(f"\n{'='*70}")
        print(f"🤖 TRADE COPILOT AGENT SWARM")
        print(f"{'='*70}")
        print(f"Query: {query}")
        print(f"Ticker: {ticker}")
        print(f"Date: {trading_date}")
        print(f"{'='*70}\n")

        print("🔄 Executing multi-agent workflow...\n")
        print("  [1/6] Market Breadth Agent - Analyzing open interest")
        print("  [2/6] Setup Agent - Configuring monitoring")
        print("  [3/6] Order Flow Agent - Detecting institutional patterns")
        print("  [4/6] Options Flow Agent - Tracking smart money")
        print("  [5/6] Financial Data Agent - Technical analysis")
        print("  [6/6] Coordinator Agent - Synthesizing recommendations\n")

        # Execute the graph
        result = self.graph(graph_prompt)

        print(f"{'='*70}")
        print(f"✅ ANALYSIS COMPLETE")
        print(f"{'='*70}\n")

        # Extract coordinator's final recommendation
        coordinator_response = result.node_results.get("coordinator")

        if coordinator_response and coordinator_response.content:
            final_recommendation = coordinator_response.content[0].text
        else:
            final_recommendation = "No recommendation generated"

        # Print metrics
        print(f"📊 Execution Metrics:")
        print(f"   • Total Tokens: {result.total_tokens:,}")
        print(f"   • Latency: {result.latency:.2f}s")
        print(f"   • Agents Executed: {len(result.node_results)}/6")
        print(f"\n{'='*70}\n")

        return final_recommendation

    def _extract_ticker(self, query: str) -> str:
        """
        Extract ticker symbol from user query

        Args:
            query: User's natural language query

        Returns:
            Ticker symbol (defaults to SPY)
        """

        # Common tickers to look for
        tickers = ['SPY', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META']

        # Search for tickers in query (case insensitive)
        query_upper = query.upper()
        for ticker in tickers:
            if ticker in query_upper:
                return ticker

        # Default to SPY
        return "SPY"


def main():
    """
    Main entry point for interactive usage

    Usage:
        python swarm.py
    """

    print("\n" + "="*70)
    print("🤖 TRADE COPILOT AGENT SWARM - Conversational Trading Assistant")
    print("="*70)
    print("\nInitializing multi-agent system...")

    swarm = TradingSwarm()

    print("✓ System ready!\n")
    print("Example queries:")
    print("  • What does SPY look like now?")
    print("  • Should I trade NVDA today?")
    print("  • Give me CALL and PUT recommendations for AAPL")
    print("\n" + "="*70 + "\n")

    # Interactive loop
    while True:
        try:
            user_query = input("You: ").strip()

            if not user_query:
                continue

            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye! Happy trading!\n")
                break

            # Get recommendation from swarm
            recommendation = swarm.ask(user_query)

            # Print final recommendation
            print("="*70)
            print("📋 FINAL RECOMMENDATION")
            print("="*70)
            print(recommendation)
            print("="*70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy trading!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


if __name__ == "__main__":
    main()