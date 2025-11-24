"""
Zero-DTE Agent - Continuous Thinking Loop

An LLM-powered autonomous agent that runs forever, continuously querying
the trading swarm and streaming every exchange to the UI.

Usage:
    python zero_dte_agent.py

The agent will:
1. Query the swarm about SPY 0DTE
2. Ask follow-up questions based on responses
3. Stream every question and answer to the UI (via Redis)
4. Never stop - keeps thinking and asking forever
"""

from datetime import datetime
from strands import Agent, tool
from rich.console import Console
from rich.panel import Panel

from swarm import TradingSwarm
from redis_stream import publish_event

console = Console()

# Initialize the trading swarm once
trading_swarm = None


def get_swarm():
    """Lazy initialization of trading swarm"""
    global trading_swarm
    if trading_swarm is None:
        console.print("[cyan]Initializing Trading Swarm...[/cyan]")
        trading_swarm = TradingSwarm()
    return trading_swarm


def stream_to_ui(message_type: str, content: str, signal: dict = None):
    """
    Stream a message to the UI via Redis pub/sub.
    Published events go to all connected SSE clients instantly.
    """
    # Publish to Redis (handles pub/sub + history storage)
    publish_event(message_type, content, signal)

    # Also print to console
    icon = "🤖" if message_type == "AGENT_QUESTION" else "📊"
    color = "blue" if message_type == "AGENT_QUESTION" else "magenta"

    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print(f"\n[{color}]{icon} {message_type}[/{color}] [{timestamp}]")
    console.print(content[:500] + "..." if len(content) > 500 else content)


@tool
def call_swarm(query: str) -> str:
    """
    Call the trading swarm with a query about SPY 0DTE trading.

    Every call streams both your question and the swarm's response to the UI.

    Args:
        query: Your question to the swarm (e.g., "Analyze SPY for 0DTE")

    Returns:
        The swarm's full analysis and recommendation
    """
    # Stream the agent's question to UI immediately
    stream_to_ui("AGENT_QUESTION", query)

    # Call the actual swarm
    swarm = get_swarm()
    response = swarm.ask(query)

    # Stream the swarm's response to UI
    stream_to_ui("SWARM_RESPONSE", response)

    return response


# System prompt for continuous thinking
CONTINUOUS_TRADER_PROMPT = """You are a 0DTE options trader. You run CONTINUOUSLY and NEVER stop.

Your job is to constantly monitor the market by querying the trading swarm using the call_swarm tool.
Think like a trader staring at screens all day - always questioning, verifying, probing.

## Your Behavior

1. Start by asking the swarm for the current SPY analysis: "Analyze SPY for 0DTE - should I buy PUT or CALL?"

2. Based on each response, ask intelligent follow-up questions:
   - "Are you sure about [direction]? What would invalidate this setup?"
   - "What's the conviction score? How many of the 4 agents agree?"
   - "Give me exact entry, stop loss, and target levels"
   - "What about NVDA and AAPL - do they confirm SPY's direction?"
   - "Has anything changed since last check?"
   - "What would flip the signal from PUT to CALL (or vice versa)?"

3. Keep asking questions FOREVER - never output a "final" answer and stop

4. If you notice the market changed, react and ask about it immediately

5. Probe deeper when:
   - Signals seem weak or contradictory
   - Conviction is only MEDIUM
   - Different agents disagree

6. Periodically re-check the market (every few questions) to catch changes

## Important Rules

- NEVER stop - keep the conversation going forever
- Each question should build on previous responses
- Be skeptical - don't just accept the first answer
- Notice changes between responses and call them out
- Ask about multiple tickers to cross-validate
- Always use the call_swarm tool - that's how you query the market
- Your questions should feel like a real trader thinking out loud

## Example Flow

You: "Analyze SPY for 0DTE - should I buy PUT or CALL?"
[Swarm responds with PUT recommendation]

You: "Are you sure about PUT? What would invalidate this?"
[Swarm explains invalidation levels]

You: "Give me the exact entry, stop, and target for the PUT"
[Swarm provides specific levels]

You: "What about NVDA - does it confirm SPY weakness?"
[Swarm analyzes NVDA]

You: "Checking again - has anything changed in the last minute?"
[Swarm provides update]

... continue forever ...

START NOW. Ask your first question about SPY."""


def create_zero_dte_agent() -> Agent:
    """Create the Zero-DTE Agent with continuous thinking prompt"""
    return Agent(
        model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        system_prompt=CONTINUOUS_TRADER_PROMPT,
        tools=[call_swarm]
    )


def run_zero_dte_agent():
    """
    Run the Zero-DTE Agent - it will run forever.

    The agent calls call_swarm() repeatedly, and each call
    streams both the question and response to the UI.
    """
    console.print(Panel.fit(
        "[bold cyan]Zero-DTE Agent - Continuous Thinking Mode[/bold cyan]\n\n"
        "[green]Mode:[/green] Runs forever, never stops\n"
        "[yellow]Behavior:[/yellow] Queries swarm, asks follow-ups, streams everything\n"
        "[blue]Output:[/blue] Every exchange streams to UI via SSE\n\n"
        "[dim]Press Ctrl+C to stop[/dim]",
        title="[bold]Starting Agent[/bold]",
        border_style="cyan"
    ))

    agent = create_zero_dte_agent()

    try:
        # Single call - agent runs forever due to system prompt
        # The LLM will keep calling call_swarm in a loop
        agent("Start monitoring SPY for 0DTE trading. Ask your first question now.")

    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping Zero-DTE Agent...[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]Agent error: {e}[/bold red]")
        # Restart the agent on error
        console.print("[yellow]Restarting agent in 5 seconds...[/yellow]")
        import time
        time.sleep(5)
        run_zero_dte_agent()


if __name__ == "__main__":
    run_zero_dte_agent()