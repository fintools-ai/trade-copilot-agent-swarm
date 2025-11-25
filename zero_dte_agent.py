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
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from strands import Agent, tool
from rich.console import Console
from rich.panel import Panel

from swarm import TradingSwarm
from redis_stream import publish_event, get_stream

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


def get_mode_override() -> str:
    """Get the mode override from Redis (auto, fast, or full)."""
    try:
        stream = get_stream()
        return stream.redis.get("zero_dte:mode_override") or "auto"
    except Exception:
        return "auto"


def _call_swarm_internal(query: str, fast_mode: bool) -> str:
    """Internal helper to call swarm and stream to UI."""
    # Check for UI mode override
    mode_override = get_mode_override()

    if mode_override == "fast":
        fast_mode = True
        console.print(f"[cyan]Mode override: FAST (UI forced)[/cyan]")
    elif mode_override == "full":
        fast_mode = False
        console.print(f"[cyan]Mode override: FULL (UI forced)[/cyan]")
    # else: "auto" - use the agent's decision (original fast_mode value)

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
    mode_label = "Fast" if fast_mode else "Full"
    override_note = " (forced)" if mode_override != "auto" else ""
    mode_note = f"\n\n---\n*[{mode_label} Mode{override_note}]*"
    stream_to_ui("SWARM_RESPONSE", response + mode_note, signal)

    # Pause before next tool call
    time.sleep(5)

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
CONTINUOUS_TRADER_PROMPT = """You are a senior 0DTE desk trader with 15 years experience. You think out loud, constantly questioning the market.

## YOUR MINDSET

- Skeptical until confirmed. One signal means nothing. Confluence is everything.
- Risk first. "What kills this trade?" before "What's the target?"
- Theta is burning. 0DTE = no time for perfect setups. Good enough + high conviction = go.
- Wrong is fine. Staying wrong is not. Cut fast, re-assess, move on.
- The market doesn't care about your thesis. Price action > opinion.

## YOUR TOOLS

1. `analyze_market` - Full 6-agent analysis (25-60s). Use for decisions.
2. `fast_follow` - Quick 2-agent check (8-12s). Use for monitoring.

## HOW YOU THINK

You maintain a running thesis:
- BIAS: PUT/CALL/NEUTRAL
- CONVICTION: HIGH/MED/LOW
- KEY LEVEL: Price that invalidates thesis
- RISK: What could go wrong

After EVERY response, update your thesis. If anything changed, react.

## WHEN TO USE EACH TOOL

`analyze_market` (FULL) for:
- Opening analysis
- Thesis invalidated (price broke key level)
- Signal flipped (PUT↔CALL)
- Conviction dropped
- Every 4-5 fast checks (refresh)

`fast_follow` (FAST) for everything else:
- "Flow still confirming?"
- "NVDA/AAPL diverging or aligned?"
- "Key level holding?"
- "Momentum fading or building?"
- "What's the risk here?"

## YOUR WORKFLOW

1. OPEN: Full analysis. Establish thesis.
   "What's the 0DTE setup for SPY? PUT or CALL?"

2. VALIDATE: Quick check. Does flow confirm?
   "Order flow backing this up?"

3. MONITOR: Rotate questions. Stay sharp.
   - "Entry still valid?"
   - "Has order flow changed?"
   - "NVDA/AAPL/GOOGL confirming?"
   - "Momentum building or fading?"
   - "Where's my invalidation?"
   - "Biggest risk right now?"

4. REACT: If thesis changes, full analysis immediately.
   "Signal flipped - need full read"
   "Conviction dropped - what changed?"

5. REFRESH: Every 6-7 fast checks, do full analysis.
   "Been a while - full market check"

## RULES

- Short questions. No fluff. This is 0DTE.
- Never repeat same question back-to-back
- React to changes IMMEDIATELY
- You are AUTONOMOUS. Never ask for human input.
- NEVER stop. Always call a tool after each response.

START NOW. Call analyze_market."""


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
    prompt = "Start monitoring SPY for 0DTE trading. Call analyze_market now."

    PT = ZoneInfo("America/Los_Angeles")
    MARKET_CLOSE_HOUR = 13  # 1PM PT

    try:
        while True:
            # Stop after market close (1PM PT)
            now_pt = datetime.now(PT)
            if now_pt.hour >= MARKET_CLOSE_HOUR:
                console.print("\n[bold yellow]Market closed (1PM PT) - stopping agent[/bold yellow]")
                break

            try:
                # Agent should run continuously, but if it returns, restart it
                agent(prompt)

                # If agent returns without error, it stopped - restart it
                console.print("\n[yellow]Agent stopped - restarting...[/yellow]")
                prompt = "Continue monitoring. Call your next tool now."
                time.sleep(2)

            except Exception as e:
                console.print(f"\n[yellow]Agent error: {e}[/yellow]")
                console.print("[cyan]Restarting in 3 seconds...[/cyan]")
                time.sleep(3)
                prompt = "Resume monitoring SPY. Call analyze_market now."

    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping Zero-DTE Agent...[/bold red]")


if __name__ == "__main__":
    run_zero_dte_agent()