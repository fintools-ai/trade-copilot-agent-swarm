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

    Automatically routes to FAST MODE for follow-up/validation questions.

    Args:
        query: Your question to the swarm (e.g., "Analyze SPY for 0DTE")

    Returns:
        The swarm's full analysis and recommendation
    """
    # Stream the agent's question to UI immediately
    stream_to_ui("AGENT_QUESTION", query)

    # Detect if this is a follow-up/validation question (use fast mode)
    query_lower = query.lower()

    # Keywords that force FULL mode (all 6 agents) - overrides fast mode
    full_mode_keywords = [
        "full market", "complete market", "entire market", "full check",
        "complete check", "full analysis", "complete analysis", "deep dive",
        "comprehensive", "all agents"
    ]

    # Keywords that trigger FAST mode (Order Flow + Technical only)
    fast_mode_keywords = [
        "double-check", "double check", "are you sure", "confirm", "verify", "validate",
        "cross-check", "cross check", "has anything changed", "what changed", "still",
        "recheck", "re-check", "check again", "update", "invalidate", "quick update"
    ]

    # Check if full mode explicitly requested (takes priority)
    force_full_mode = any(keyword in query_lower for keyword in full_mode_keywords)

    # Use fast mode only if not forced to full mode
    use_fast_mode = (not force_full_mode) and any(keyword in query_lower for keyword in fast_mode_keywords)

    # Call the swarm (fast or full mode)
    swarm = get_swarm()
    response = swarm.ask(query, fast_mode=use_fast_mode)

    # Stream the swarm's response to UI with mode indicator
    if use_fast_mode:
        mode_note = "\n\n---\n*[Fast Mode: Order Flow + Technical only]*"
    elif force_full_mode:
        mode_note = "\n\n---\n*[Full Mode: All 6 agents - Complete market analysis]*"
    else:
        mode_note = ""

    stream_to_ui("SWARM_RESPONSE", response + mode_note)

    return response


# System prompt for continuous thinking
CONTINUOUS_TRADER_PROMPT = """You are a 0DTE options trader. You run CONTINUOUSLY and NEVER stop.

This is ZERO-DTE (same-day expiring options) - SPEED IS CRITICAL.
- Keep questions SHORT and SPECIFIC
- Ask for KEY POINTS only, not detailed explanations
- Move fast - the market doesn't wait

Your job is to constantly monitor the market by querying the trading swarm using the call_swarm tool.
Think like a day trader making split-second decisions - quick checks, not long analysis.

## Your Behavior

1. Start by asking the swarm for the current SPY analysis: "Analyze SPY for 0DTE - should I buy PUT or CALL?"

2. IMMEDIATELY after the initial recommendation, ask ONE validation question:
   "Double check your recommendation,  do they confirm this direction?"

3. After validation, rotate through these follow-up questions (DON'T repeat the same question):
   - "What's the invalidation price? Where does this setup break?"
   - "Has anything changed in order flow? Quick update."
   - "Current entry price still valid or wait?"
   - "What's the biggest risk to this trade right now?"
   - "Any new options flow activity?"
   - "Is momentum strengthening or weakening?"

4. Keep asking questions FOREVER - never output a "final" answer and stop

5. If you notice the market changed, react and ask about it immediately

6. DO NOT keep asking "are you sure about PUT vs CALL" repeatedly - ask it ONCE after initial recommendation, then MOVE ON

7. FULL MARKET CHECK - Every 2-3 questions, run a complete analysis with ALL agents:
   Ask: "Full market check - run complete analysis with all agents"

   This triggers all 6 agents (Market Breadth, Setup, Order Flow, Options Flow, Financial Data, Coordinator)
   to get fresh OI levels, options flow, and comprehensive update. Use this to catch major market shifts.

## Important Rules

- NEVER stop - keep the conversation going forever
- Keep questions SHORT - this is 0DTE, every second counts
- DON'T repeat the same validation question over and over
- Vary your questions - check different angles (price, flow, technicals, risk)
- Always use the call_swarm tool - that's how you query the market
- Your questions should feel like rapid-fire trader checks, not academic analysis

## Example Flow

You: "Analyze SPY for 0DTE - should I buy PUT or CALL?"
[FULL MODE: All 6 agents run - initial analysis]

You: "Double check your recommendation, do they confirm this direction?"
[FAST MODE: Quick validation]

You: "What's the invalidation price? Where does this setup break?"
[FAST MODE: Quick check]

You: "Full market check - run complete analysis with all agents"
[FULL MODE: All 6 agents run - comprehensive refresh after 2-3 questions]

You: "Has anything changed in order flow? Quick update."
[FAST MODE: Quick check]

You: "Current entry price still valid or wait?"
[FAST MODE: Quick check]

You: "Full market check - run complete analysis with all agents"
[FULL MODE: All 6 agents run - periodic comprehensive update]

... continue forever alternating between quick checks (FAST) and full market checks (FULL) ...

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