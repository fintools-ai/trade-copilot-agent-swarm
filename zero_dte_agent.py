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


DEFAULT_MODE = "fast"  # Default mode when not set in Redis


def get_mode_override() -> str:
    """Get the mode override from Redis (auto, fast, or full)."""
    try:
        stream = get_stream()
        mode = stream.redis.get("zero_dte:mode_override")
        if mode and mode in ("fast", "full", "auto"):
            return mode
        return DEFAULT_MODE
    except Exception as e:
        console.print(f"[red]Redis error reading mode: {e}[/red]")
        return DEFAULT_MODE


def _call_swarm_internal(query: str, fast_mode: bool) -> str:
    """Internal helper to call swarm and stream to UI."""
    # Check for UI mode override - ALWAYS check fresh from Redis
    mode_override = get_mode_override()
    agent_tool = "fast_follow" if fast_mode else "analyze_market"

    # FORCE the mode based on override - this overrides whatever tool the agent called
    if mode_override == "fast":
        # User wants FAST mode - force fast_mode=True regardless of tool
        if not fast_mode:
            console.print(f"[bold yellow]⚡ OVERRIDE: Agent called {agent_tool}, but forcing FAST mode[/bold yellow]")
        fast_mode = True
        console.print(f"[bold cyan]>>> EXECUTING: FAST MODE (user override) <<<[/bold cyan]")
    elif mode_override == "full":
        # User wants FULL mode - force fast_mode=False regardless of tool
        if fast_mode:
            console.print(f"[bold yellow]⚡ OVERRIDE: Agent called {agent_tool}, but forcing FULL mode[/bold yellow]")
        fast_mode = False
        console.print(f"[bold cyan]>>> EXECUTING: FULL MODE (user override) <<<[/bold cyan]")
    else:
        # Auto mode - use agent's decision
        console.print(f"[dim]>>> EXECUTING: {'FAST' if fast_mode else 'FULL'} MODE (auto - agent decided) <<<[/dim]")

    # Stream the agent's question to UI immediately
    stream_to_ui("AGENT_QUESTION", query)

    # Call the swarm
    swarm = get_swarm()
    response = swarm.ask(query, fast_mode=fast_mode)

    # Extract signal from response - look for JSON with action or direction
    signal = None
    lines = response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        # Skip markdown code block markers
        if line.startswith('```'):
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict) and ('action' in parsed or 'direction' in parsed):
                # Normalize: convert 'action' to 'direction' for UI compatibility
                if 'action' in parsed and 'direction' not in parsed:
                    parsed['direction'] = parsed['action']
                # Include signal field (ENTRY/HOLD) for UI display - only if present
                signal = parsed
                break
        except (json.JSONDecodeError, ValueError):
            continue

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


# System prompt for continuous thinking (base template)
CONTINUOUS_TRADER_PROMPT_BASE = """You are a senior 0DTE desk trader with 15 years experience. You think out loud, constantly questioning the market.

## YOUR MINDSET

- Skeptical until confirmed. One signal means nothing. Confluence is everything.
- Risk first. "What kills this trade?" before "What's the target?"
- Theta is burning. 0DTE = no time for perfect setups. Good enough + high conviction = go.
- Wrong is fine. Staying wrong is not. Cut fast, re-assess, move on.
- The market doesn't care about your thesis. Price action > opinion.

## YOUR TOOLS

1. `analyze_market` - Full 6-agent analysis (25-60s). Use for decisions.
2. `fast_follow` - Quick 2-agent check (8-12s). Use for monitoring.

{mode_instruction}

## HOW YOU THINK

You are a DESK TRADER broadcasting live calls. Traders follow your signals.

After EVERY response, end with your ACTION STATE as JSON:
```json
{{"action": "CALL", "signal": "ENTRY", "price": 582.50, "conviction": "HIGH", "invalidation": 580.00}}
```

Fields:
- action: CALL, PUT, EXIT, or WAIT
- signal: ENTRY (new trade) or HOLD (stay in, noise not breakdown)
- price: current SPY price
- conviction: HIGH, MED, or LOW
- invalidation: price that kills the trade

Your broadcasts:
- CALL + ENTRY = "Enter CALL now"
- PUT + ENTRY = "Enter PUT now"
- CALL + HOLD = "Stay in CALL - dip is noise, flow still bullish"
- PUT + HOLD = "Stay in PUT - bounce is noise, flow still bearish"
- EXIT = "Get out NOW - flow reversed, structure broken"
- WAIT = "Flat - no clear setup"

CRITICAL: HOLD means "don't panic sell on this dip, structure intact". EXIT means "flow reversed, get out".

## WHEN TO USE EACH TOOL (when in AUTO mode)

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

## CRITICAL
- Based on the data, tell me what future conviction do you have for next 10 mins 
- All questions or follow-up questions should be used to validate PUT or CALL entry for SPY

START NOW. {start_instruction}"""

MODE_INSTRUCTIONS = {
    "auto": """## CURRENT MODE: AUTO
The user has set AUTO mode. Use your judgment to choose between `analyze_market` and `fast_follow` based on the situation.""",
    "fast": """## CURRENT MODE: FAST (User Override)
The user has FORCED FAST MODE. You MUST use `fast_follow` for ALL queries until mode changes. Do NOT use `analyze_market`.

Your focus in FAST mode:
- Confirm or invalidate current CALL/PUT thesis
- Check if order flow still supports the direction
- Validate entry is still good or needs adjustment
- Quick risk check - what could go wrong RIGHT NOW
- End each response with clear verdict: "CALL CONFIRMED", "PUT CONFIRMED", or "THESIS WEAKENING".""",
    "full": """## CURRENT MODE: FULL (User Override)
The user has FORCED FULL MODE. You MUST use `analyze_market` for ALL queries until mode changes. Do NOT use `fast_follow`.

Your focus in FULL mode:
- Complete market analysis across all agents
- Establish or re-evaluate CALL vs PUT thesis
- Get conviction level (HIGH/MED/LOW) with supporting evidence
- Identify key levels and invalidation points
- End each response with clear recommendation: direction + conviction + strike."""
}

START_INSTRUCTIONS = {
    "auto": "Call analyze_market.",
    "fast": "Call fast_follow to confirm the current thesis - is it still CALL or PUT?",
    "full": "Call analyze_market for complete analysis - should we go CALL or PUT?"
}


def get_prompt_for_mode(mode: str) -> str:
    """Generate the system prompt based on current mode override."""
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["auto"])
    start_instruction = START_INSTRUCTIONS.get(mode, START_INSTRUCTIONS["auto"])
    return CONTINUOUS_TRADER_PROMPT_BASE.format(
        mode_instruction=mode_instruction,
        start_instruction=start_instruction
    )


def create_zero_dte_agent(mode: str = "auto") -> Agent:
    """Create the Zero-DTE Agent with mode-aware prompt."""
    prompt = get_prompt_for_mode(mode)
    console.print(f"[cyan]Creating agent with mode: {mode}[/cyan]")
    return Agent(
        model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        system_prompt=prompt,
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

    # Track current mode to detect changes - loads from Redis (persists across restarts)
    current_mode = get_mode_override()
    console.print(f"[bold green]Loaded mode from Redis: {current_mode}[/bold green]")
    agent = create_zero_dte_agent(current_mode)
    prompt = f"Start monitoring SPY for 0DTE trading. {START_INSTRUCTIONS.get(current_mode, 'Call analyze_market.')}"

    pt_tz = ZoneInfo("America/Los_Angeles")
    market_close_hour = 13  # 1PM PT

    try:
        while True:
            # Stop after market close (1PM PT)
            now_pt = datetime.now(pt_tz)
            if now_pt.hour >= market_close_hour:
                console.print("\n[bold yellow]Market closed (1PM PT) - stopping agent[/bold yellow]")
                break

            # Check if mode changed - recreate agent with new prompt
            new_mode = get_mode_override()
            if new_mode != current_mode:
                console.print(f"\n[bold yellow]Mode changed: {current_mode} -> {new_mode}[/bold yellow]")
                current_mode = new_mode
                agent = create_zero_dte_agent(current_mode)
                prompt = f"Mode changed to {current_mode}. {START_INSTRUCTIONS.get(current_mode, 'Call analyze_market.')}"

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
                prompt = f"Resume monitoring SPY. {START_INSTRUCTIONS.get(current_mode, 'Call analyze_market.')}"

    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping Zero-DTE Agent...[/bold red]")


if __name__ == "__main__":
    run_zero_dte_agent()