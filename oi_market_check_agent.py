#!/usr/bin/env python3
"""
OI Market Check Agent - Simple test swarm for quick market analysis
Combines Market Breadth (OI) + Financial Data agents for fast insights
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from strands.multiagent.graph import Graph, GraphBuilder
from strands.session.file_session_manager import FileSessionManager
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from agents.market_breadth_agent import create_market_breadth_agent
from agents.financial_data_agent import create_financial_data_agent
from agents.coordinator_agent import create_coordinator_agent

console = Console()


class OIMarketCheckAgent:
    """Simple 3-agent swarm: Market Breadth + Financial Data + Coordinator"""
    
    def __init__(self, session_id: str = None):
        if session_id is None:
            session_id = f"oi_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create session manager
        storage_dir = str(Path(__file__).parent / "sessions")
        self.session_manager = FileSessionManager(
            session_id=session_id,
            storage_dir=storage_dir
        )
        
        self.session_id = session_id
        self.graph = self._build_graph()
        
        console.print(Panel.fit(
            f"[bold green]OI Market Check Agent Ready[/bold green]\n"
            f"[cyan]Agents:[/cyan] Market Breadth + Financial Data + Coordinator\n"
            f"[yellow]Flow:[/yellow] [Market Breadth, Financial Data] → Coordinator\n"
            f"[blue]Session:[/blue] {session_id}\n"
            f"[magenta]Storage:[/magenta] {storage_dir}",
            title="[bold]Agent System[/bold]",
            border_style="green"
        ))
    
    def _build_graph(self) -> Graph:
        """Build 3-agent graph with coordinator"""
        
        # Create agents
        market_breadth_agent = create_market_breadth_agent()
        financial_data_agent = create_financial_data_agent()
        coordinator_agent = create_coordinator_agent()
        
        # Build graph
        builder = GraphBuilder()
        
        # Add nodes
        builder.add_node(market_breadth_agent, "market_breadth")
        builder.add_node(financial_data_agent, "financial_data")
        builder.add_node(coordinator_agent, "coordinator")
        
        # Parallel: market_breadth and financial_data run together
        # Then coordinator synthesizes both results
        builder.add_edge("market_breadth", "coordinator")
        builder.add_edge("financial_data", "coordinator")
        
        # Build without session manager for now
        return builder.build()
    
    async def analyze_async(self, query: str) -> str:
        """Async analysis"""
        
        console.print(Panel(
            f"[bold yellow]Query:[/bold yellow] {query}\n"
            f"[cyan]Expected time:[/cyan] ~25-30s\n"
            f"[green]Flow:[/green] Market Breadth + Financial Data → Coordinator",
            title="[bold]Analysis Starting[/bold]",
            border_style="yellow"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running parallel analysis...", total=None)
            
            # Execute graph
            result = await self.graph.invoke_async(query)
            
            progress.update(task, description="[green]Analysis Complete!")
        
        # Debug: Show what we actually got
        console.print(f"[dim]Result type: {type(result)}[/dim]")
        console.print(f"[dim]Has results attr: {hasattr(result, 'results')}[/dim]")
        
        if hasattr(result, 'results'):
            console.print(f"[dim]Results keys: {list(result.results.keys()) if result.results else 'None'}[/dim]")
            for key, value in (result.results.items() if result.results else []):
                console.print(f"[dim]{key}: {type(value)}[/dim]")
                console.print(f"[dim]  - attributes: {[attr for attr in dir(value) if not attr.startswith('_')]}[/dim]")
                if hasattr(value, 'message'):
                    console.print(f"[dim]  - message type: {type(value.message)}[/dim]")
                    if hasattr(value.message, 'content'):
                        console.print(f"[dim]  - message.content: {type(value.message.content)}[/dim]")
        
        # Extract coordinator's final result - check NodeResult structure
        if hasattr(result, 'results') and result.results:
            coordinator_response = result.results.get("coordinator")
            if coordinator_response:
                # Check all possible attributes
                for attr in ['content', 'message', 'output', 'result', 'data', 'response']:
                    if hasattr(coordinator_response, attr):
                        value = getattr(coordinator_response, attr)
                        console.print(f"[dim]Found {attr}: {type(value)}[/dim]")
                        if value:
                            if isinstance(value, list) and len(value) > 0:
                                if hasattr(value[0], 'text'):
                                    return value[0].text
                                else:
                                    return str(value[0])
                            elif isinstance(value, dict):
                                if 'content' in value:
                                    return value['content'][0]['text'] if isinstance(value['content'], list) else value['content']
                                elif 'text' in value:
                                    return value['text']
                            elif isinstance(value, str):
                                return value
                            else:
                                return str(value)
        
        return "Could not extract coordinator result - unknown NodeResult structure"
    
    def analyze(self, query: str) -> str:
        """Sync wrapper"""
        import asyncio
        return asyncio.run(self.analyze_async(query))


def main():
    """Interactive mode"""
    agent = OIMarketCheckAgent()
    
    console.print(Panel.fit(
        "[bold cyan]OI MARKET CHECK AGENT - Interactive Mode[/bold cyan]\n"
        "[green]3-Agent System:[/green] Market Breadth + Financial Data + Coordinator\n"
        "[yellow]Parallel Analysis with Intelligent Synthesis[/yellow]\n\n"
        "[bold]Commands:[/bold]\n"
        "  • Type your analysis request (e.g., 'Analyze SPY for 0DTE trading')\n"
        "  • Type 'quit' to exit",
        title="[bold]Welcome[/bold]",
        border_style="blue"
    ))
    
    while True:
        try:
            query = console.input("\n[bold green]Your request:[/bold green] ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                console.print("[bold red]Goodbye![/bold red]")
                break
            
            if not query:
                continue
            
            result = agent.analyze(query)
            
            # Try to render as markdown, fallback to plain text
            try:
                console.print(Panel(
                    Markdown(result),
                    title="[bold green]FINAL ANALYSIS[/bold green]",
                    border_style="green"
                ))
            except Exception:
                # Fallback to plain text if markdown fails
                console.print(Panel(
                    result,
                    title="[bold green]FINAL ANALYSIS[/bold green]",
                    border_style="green"
                ))
            
        except KeyboardInterrupt:
            console.print("\n[bold red]Goodbye![/bold red]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
