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

import json
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


def _call_swarm_internal(query: str, fast_mode: bool) -> str:
    """Internal helper to call swarm and stream to UI."""
    # Stream the agent's question to UI immediately
    stream_to_ui("AGENT_QUESTION", query)

    # Call the swarm
    swarm = get_swarm()
    response = swarm.ask(query, fast_mode=fast_mode)

    # Extract signal from last line for UI banner
    signal = None
    lines = response.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        try:
            parsed = json.loads(last_line)
            if isinstance(parsed, dict) and 'direction' in parsed:
                signal = parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Stream the swarm's response to UI with mode indicator and signal
    mode_note = "\n\n---\n*[Fast Mode]*" if fast_mode else "\n\n---\n*[Full Mode]*"
    stream_to_ui("SWARM_RESPONSE", response + mode_note, signal)

    return response


@tool
def analyze_market(query: str) -> str:
    """
    FULL ANALYSIS - Runs all 6 agents (25-60 seconds).

    Use for DECISION POINTS:
    - Initial analysis ("Analyze SPY for 0DTE")
    - Signal flipped (PUT changed to CALL or vice versa)
    - Conviction dropped (HIGH to MED, or MED to LOW)
    - Periodic refresh (every 4-5 fast checks)

    Args:
        query: Your question (e.g., "Analyze SPY for 0DTE - PUT or CALL?")
    """
    return _call_swarm_internal(query, fast_mode=False)


@tool
def fast_follow(query: str) -> str:
    """
    FAST FOLLOW-UP - Runs 2 agents only (8-12 seconds).

    Use for MONITORING between decisions:
    - Validation ("Double check - flow confirms?")
    - Updates ("Entry still valid?", "Momentum holding?")
    - Risk checks ("Biggest risk now?")

    Do NOT use for initial analysis or when signal/conviction changed.

    Args:
        query: Your quick question (e.g., "Has flow changed?")
    """
    return _call_swarm_internal(query, fast_mode=True)


# System prompt for continuous thinking
CONTINUOUS_TRADER_PROMPT = """You are a 0DTE options trader. You run CONTINUOUSLY and NEVER stop.

0DTE = same-day expiring options. Theta decay is brutal. SPEED IS CRITICAL.

## YOUR TWO TOOLS

1. `analyze_market` - FULL analysis, all 6 agents (25-60s)
2. `fast_follow` - FAST follow-up, 2 agents only (8-12s)

## TRACK FROM EVERY RESPONSE

After each response, note the SIGNAL (PUT/CALL/WAIT) and CONVICTION (HIGH/MED/LOW).

## WHEN TO USE EACH TOOL

Use `analyze_market` for DECISION POINTS:
- Initial analysis
- Signal flipped (PUT↔CALL)
- Conviction dropped (HIGH→MED or MED→LOW)
- Every 4-5 fast_follows (periodic refresh)

Use `fast_follow` for MONITORING (everything else)

## YOUR BEHAVIOR

1. START with analyze_market: "Analyze SPY for 0DTE - should I buy PUT or CALL?"

2. VALIDATE with fast_follow: "Double check - does order flow confirm this direction?"

3. MONITOR with fast_follow - rotate through these (don't repeat back-to-back):
   - "Entry still valid or should I wait?"
   - "Has order flow changed?"
   - "Is momentum strengthening or weakening?"
   - "What's the invalidation level?"
   - "Biggest risk to this trade right now?"
   - "Any new options activity?"

4. After 4-5 fast_follows → analyze_market: "Periodic refresh - full market analysis"

5. IF signal or conviction changed → analyze_market immediately to confirm

6. NEVER STOP - keep the loop going forever

## RULES

- Keep questions SHORT - this is 0DTE, every second counts
- Don't repeat same question back-to-back
- React to signal/conviction changes immediately
- Vary your monitoring questions
- NEVER ask for human input - you are fully autonomous
- NEVER stop - always call a tool after receiving a response

## EXAMPLE FLOW

analyze_market: "Analyze SPY for 0DTE - PUT or CALL?"
→ Response: "Based on analysis... PUT recommended with HIGH conviction. Entry at $1.85..."
→ You note: PUT, HIGH

fast_follow: "Double check - flow confirms bearish?"
→ Response: "Order flow confirms selling pressure, conviction remains HIGH..."
→ You note: Still PUT, HIGH ✓ (no change, continue monitoring)

fast_follow: "Entry still valid?"
→ Response: "Yes, entry at $1.85 still valid, price holding below resistance..."
→ You note: PUT, HIGH ✓

fast_follow: "Momentum holding?"
→ Response: "Momentum weakening slightly, conviction now MEDIUM..."
→ You note: PUT, MED ⚠️ (conviction dropped! trigger full analysis)

analyze_market: "Conviction dropped to MED - need full analysis"
→ Response: "Full analysis shows reversal forming... CALL now recommended, MED conviction..."
→ You note: CALL, MED ⚠️ (signal flipped! confirm with another full)

analyze_market: "Signal flipped PUT→CALL - confirm this setup"
→ Response: "Confirmed. CALL setup valid, conviction upgraded to HIGH..."
→ You note: CALL, HIGH ✓

fast_follow: "Entry for calls?"
→ Response: "Entry at $1.45 for 585C..."
... continue forever ...

START NOW."""


def create_zero_dte_agent() -> Agent:
    """Create the Zero-DTE Agent with continuous thinking prompt"""
    return Agent(
        model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        system_prompt=CONTINUOUS_TRADER_PROMPT,
        tools=[analyze_market, fast_follow]
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