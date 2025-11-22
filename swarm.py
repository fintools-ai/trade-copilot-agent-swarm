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
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from agents.market_breadth_agent import create_market_breadth_agent
from agents.setup_agent import create_setup_agent
from agents.order_flow_agent import create_order_flow_agent
from agents.options_flow_agent import create_options_flow_agent
from agents.financial_data_agent import create_financial_data_agent
from agents.coordinator_agent import create_coordinator_agent

console = Console()


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
        #if session_id is None:
        #    trading_date = datetime.now().strftime("%Y-%m-%d")
        #    session_id = f"trading-{trading_date}"

        # Set default storage directory
        #if storage_dir is None:
        #    storage_dir = str(Path(__file__).parent / "sessions")

        # Create session manager
        #self.session_manager = FileSessionManager(
        #    session_id=session_id,
        #    storage_dir=storage_dir
        #)

        self.session_id = session_id
        self.graph = self._build_graph()

        console.print(Panel.fit(
            f"[bold green]Trade Copilot Agent Swarm Ready[/bold green]\n"
            f"[cyan]Agents:[/cyan] 6-Agent Multi-Specialist System\n"
            f"[yellow]Flow:[/yellow] MarketBreadth → Setup → [OrderFlow, OptionsFlow, FinancialData] → Coordinator\n"
            f"[blue]Session:[/blue] {session_id or 'No session'}\n"
            f"[magenta]Architecture:[/magenta] Sequential + Parallel + Synthesis",
            title="[bold]Trading System[/bold]",
            border_style="green"
        ))

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
        builder.add_node(market_breadth_agent, "market_breadth")

        # STEP 2: Setup - Configures monitoring based on OI
        builder.add_node(setup_agent, "setup")

        # STEP 3a: Order Flow Specialist
        builder.add_node(order_flow_agent, "order_flow")

        # STEP 3b: Options Flow Specialist
        builder.add_node(options_flow_agent, "options_flow")

        # STEP 3c: Financial Data Specialist
        builder.add_node(financial_data_agent, "financial_data")

        # STEP 4: Coordinator - Synthesizes everything
        builder.add_node(coordinator_agent, "coordinator")

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

        # Build graph with session manager to preserve OI cache
        return builder.build()

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

        console.print(Panel(
            f"[bold yellow]Query:[/bold yellow] {query}\n"
            f"[cyan]Ticker:[/cyan] {ticker}\n"
            f"[green]Date:[/green] {trading_date}\n"
            f"[magenta]Expected time:[/magenta] ~25-30s (OI cached) or ~55-60s (first run)",
            title="[bold]Analysis Starting[/bold]",
            border_style="yellow"
        ))

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

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Executing 6-agent workflow...", total=None)
            
            # Execute the graph
            result = self.graph(graph_prompt)
            
            progress.update(task, description="[green]Analysis Complete!")

        # Extract coordinator's final recommendation - check NodeResult structure
        if hasattr(result, 'results') and result.results:
            coordinator_response = result.results.get("coordinator")
            if coordinator_response:
                # Check all possible attributes
                for attr in ['content', 'message', 'output', 'result', 'data', 'response']:
                    if hasattr(coordinator_response, attr):
                        value = getattr(coordinator_response, attr)
                        if value:
                            if isinstance(value, list) and len(value) > 0:
                                if hasattr(value[0], 'text'):
                                    final_recommendation = value[0].text
                                    break
                                else:
                                    final_recommendation = str(value[0])
                                    break
                            elif isinstance(value, dict):
                                if 'content' in value:
                                    final_recommendation = value['content'][0]['text'] if isinstance(value['content'], list) else value['content']
                                    break
                                elif 'text' in value:
                                    final_recommendation = value['text']
                                    break
                            elif isinstance(value, str):
                                final_recommendation = value
                                break
                            else:
                                final_recommendation = str(value)
                                break
                else:
                    final_recommendation = "Could not extract coordinator result - unknown NodeResult structure"
            else:
                final_recommendation = "No coordinator response found"
        else:
            final_recommendation = "No results found"

        # Show execution metrics if available
        metrics_text = f"[blue]Agents Executed:[/blue] {len(result.results) if hasattr(result, 'results') and result.results else 0}/6"
        
        if hasattr(result, 'total_tokens'):
            metrics_text = f"[green]Total Tokens:[/green] {result.total_tokens:,}\n" + metrics_text
        
        if hasattr(result, 'latency'):
            metrics_text = f"[yellow]Latency:[/yellow] {result.latency:.2f}s\n" + metrics_text

        console.print(Panel.fit(
            metrics_text,
            title="[bold]Execution Metrics[/bold]",
            border_style="blue"
        ))

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

    console.print(Panel.fit(
        "[bold cyan]TRADE COPILOT AGENT SWARM - Interactive Mode[/bold cyan]\n"
        "[green]6-Agent System:[/green] Market Breadth + Setup + Order Flow + Options Flow + Financial Data + Coordinator\n"
        "[yellow]Sequential + Parallel Analysis with Intelligent Synthesis[/yellow]\n\n"
        "[bold]Commands:[/bold]\n"
        "  • What does SPY look like now?\n"
        "  • Should I trade NVDA today?\n"
        "  • Give me CALL and PUT recommendations for AAPL\n"
        "  • Type 'quit' to exit",
        title="[bold]Welcome[/bold]",
        border_style="blue"
    ))

    console.print("[cyan]Initializing multi-agent system...[/cyan]")
    swarm = TradingSwarm()

    # Interactive loop
    while True:
        try:
            user_query = console.input("\n[bold green]Your request:[/bold green] ").strip()

            if not user_query:
                continue

            if user_query.lower() in ['exit', 'quit', 'bye']:
                console.print("[bold red]👋 Goodbye! Happy trading![/bold red]")
                break

            # Get recommendation from swarm
            recommendation = swarm.ask(user_query)

            # Try to render as markdown, fallback to plain text
            try:
                console.print(Panel(
                    Markdown(recommendation),
                    title="[bold green]📋 FINAL RECOMMENDATION[/bold green]",
                    border_style="green"
                ))
            except Exception:
                # Fallback to plain text if markdown fails
                console.print(Panel(
                    recommendation,
                    title="[bold green]📋 FINAL RECOMMENDATION[/bold green]",
                    border_style="green"
                ))

        except KeyboardInterrupt:
            console.print("\n[bold red]👋 Goodbye! Happy trading![/bold red]")
            break
        except Exception as e:
            console.print(f"[bold red]❌ Error:[/bold red] {e}")
            continue


if __name__ == "__main__":
    main()