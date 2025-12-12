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
from strands.agent.conversation_manager import SlidingWindowConversationManager
from rich.console import Console
from rich.panel import Panel

from swarm import TradingSwarm
from redis_stream import publish_event, get_stream
from utils.token_tracker import TokenTracker

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


def get_position_context() -> str:
    """Get the custom position/question from Redis (if set by user)."""
    try:
        stream = get_stream()
        position = stream.redis.get("zero_dte:position")
        return position.strip() if position else ""
    except Exception as e:
        console.print(f"[red]Redis error reading position: {e}[/red]")
        return ""


def was_just_exited() -> bool:
    """
    Check if the most recent signal was an EXIT (manual or system).
    Used to tell the agent "position just closed, scan for new setup".
    """
    try:
        stream = get_stream()
        events = stream.redis.lrange("zero_dte:history", 0, 5)

        for event_json in events:
            try:
                event = json.loads(event_json)
                if event.get("signal") and event.get("type") == "SWARM_RESPONSE":
                    sig = event["signal"]
                    signal_type = sig.get("signal")
                    action = sig.get("action")
                    # First SWARM_RESPONSE we find - check if it's EXIT
                    if signal_type == "EXIT" or action == "EXIT":
                        return True
                    else:
                        return False  # Most recent signal is not EXIT
            except (json.JSONDecodeError, KeyError):
                continue
        return False
    except Exception:
        return False


def get_last_recommendation() -> dict:
    """
    Get the last active position and current state from Redis history.

    Returns BOTH:
    1. Original ENTRY params (entry/stop/target) for risk management
    2. Current state (HOLD/EXIT) and conviction

    Logic:
    - Find the most recent ENTRY to get trade params
    - Find the most recent signal to get current state
    - If EXIT found before ENTRY → position closed, return {}
    """
    try:
        stream = get_stream()
        events = stream.redis.lrange("zero_dte:history", 0, 25)

        entry_data = None
        current_state = None

        for event_json in events:
            try:
                event = json.loads(event_json)
                if event.get("signal") and event.get("type") == "SWARM_RESPONSE":
                    sig = event["signal"]
                    signal_type = sig.get("signal")
                    action = sig.get("action") or sig.get("direction")

                    # Capture most recent signal as current state (first one we see)
                    if current_state is None:
                        current_state = {
                            "current_signal": signal_type,
                            "current_conviction": sig.get("conviction"),
                            "current_price": sig.get("price"),
                            "last_update": event.get("timestamp")
                        }

                    # EXIT found before ENTRY → position closed
                    if entry_data is None and (signal_type == "EXIT" or action == "EXIT"):
                        return {}

                    # Found ENTRY → capture trade params
                    if signal_type == "ENTRY" and entry_data is None:
                        entry_data = {
                            "action": action,
                            "entry": sig.get("entry"),
                            "stop": sig.get("stop"),
                            "target": sig.get("target"),
                            "entry_time": event.get("timestamp")
                        }
                        break

            except (json.JSONDecodeError, KeyError):
                continue

        # Combine entry + current state if we have both
        if entry_data and current_state:
            return {**entry_data, **current_state}

        return {}
    except Exception as e:
        console.print(f"[red]Redis error reading last recommendation: {e}[/red]")
        return {}


def _call_swarm_internal(query: str, fast_mode: bool) -> str:
    """Internal helper to call swarm and stream to UI."""
    # Inject fresh timestamp into every query (agents no longer have static timestamps)
    pt_tz = ZoneInfo("America/Los_Angeles")
    now = datetime.now(pt_tz)
    current_time_full = now.strftime("%Y-%m-%d %H:%M:%S PT")
    # Market hours: 6:30 AM - 1:00 PM PT
    market_status = 'OPEN' if (now.hour == 6 and now.minute >= 30) or (7 <= now.hour < 13) else 'CLOSED'
    time_context = f"\n\n[CURRENT TIME: {current_time_full} | Market: {market_status}]"

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

    # Get current trade context (original ENTRY + latest state)
    last_rec = get_last_recommendation()
    if last_rec and last_rec.get("action"):
        # Natural language - like a trader would say it
        prev_context = f"\n\n[CURRENT TRADE: {last_rec['action']} @ ${last_rec.get('entry')} | Stop ${last_rec.get('stop')} | Target ${last_rec.get('target')} — {last_rec.get('current_signal')} with {last_rec.get('current_conviction')} conviction]"
        query_with_context = query + time_context + prev_context
        console.print(f"[dim]Trade: {last_rec['action']} @ ${last_rec.get('entry')} — {last_rec.get('current_signal')} {last_rec.get('current_conviction')}[/dim]")
    elif was_just_exited():
        # Position was just closed - tell agent to scan for new setup
        exit_context = "\n\n[POSITION CLOSED - scanning for new entry setup. Look for fresh CALL or PUT opportunity.]"
        query_with_context = query + time_context + exit_context
        console.print(f"[dim]Status: Position closed, scanning for new trade[/dim]")
    else:
        query_with_context = query + time_context

    # Stream the agent's question to UI immediately (without context noise)
    # Include query_start_ts for latency calculation
    query_start_ts = time.time()
    stream_to_ui("AGENT_QUESTION", query, {"query_start_ts": query_start_ts})

    # Call the swarm with context
    swarm = get_swarm()
    try:
        response = swarm.ask(query_with_context, fast_mode=fast_mode)
    except Exception as e:
        error_msg = str(e)
        # Publish error to UI
        error_signal = {"action": "ERROR", "conviction": "HIGH", "error": error_msg[:200], "query_start_ts": query_start_ts}
        stream_to_ui("SWARM_ERROR", f"⚠️ SWARM ERROR: {error_msg}", error_signal)
        console.print(f"[bold red]Swarm error: {error_msg}[/bold red]")
        raise  # Re-raise so the outer loop can handle restart

    # Calculate latency
    response_end_ts = time.time()
    latency = response_end_ts - query_start_ts

    # Track token usage from swarm
    try:
        token_usage = swarm.get_last_token_usage()
        if token_usage and token_usage.get('total', {}).get('input', 0) > 0:
            mode_label = "fast" if fast_mode else "full"
            tracker = TokenTracker(mode=mode_label, model="haiku-4.5")

            # Record per-agent tokens
            for agent_name, agent_data in token_usage.get('agents', {}).items():
                tracker.record(agent_name, agent_data.get('input', 0), agent_data.get('output', 0))

            # Save to Redis + JSONL
            tracker.finish()
            console.print(f"[dim]Tokens: {token_usage['total']['input']:,} in / {token_usage['total']['output']:,} out[/dim]")
    except Exception as e:
        console.print(f"[dim]Token tracking error: {e}[/dim]")

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
                # Add latency for UI display
                parsed['latency'] = round(latency, 1)
                signal = parsed
                break
        except (json.JSONDecodeError, ValueError):
            continue

    # If no signal parsed, create minimal one with latency
    if signal is None:
        signal = {"latency": round(latency, 1)}

    # Stream the swarm's response to UI with mode indicator and signal
    mode_label = "Fast" if fast_mode else "Full"
    override_note = " (forced)" if mode_override != "auto" else ""
    mode_note = f"\n\n---\n*[{mode_label} Mode{override_note}]*"
    stream_to_ui("SWARM_RESPONSE", response + mode_note, signal)

    console.print(f"[dim]Latency: {latency:.1f}s[/dim]")

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
{{"action": "CALL", "signal": "ENTRY", "price": 582.50, "entry": 582.50, "stop": 580.00, "target": 585.00, "conviction": "HIGH"}}
```

Fields:
- action: CALL, PUT, EXIT, or WAIT
- signal: ENTRY (new trade) or HOLD (stay in, noise not breakdown)
- price: current SPY price
- entry: entry price level
- stop: stop loss level (invalidation)
- target: profit target level
- conviction: HIGH, MED, or LOW

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

# Position validation prompt - used when user sets a custom question
POSITION_VALIDATION_PROMPT = """You are a senior 0DTE desk trader helping validate a specific position/question.

## USER'S QUESTION/POSITION
{position}

## YOUR MISSION
The user has a specific question or position they need validated. Your ONLY job is to continuously monitor and answer THIS question.

## YOUR TOOLS
1. `analyze_market` - Full 6-agent analysis (25-60s). Use for deep validation.
2. `fast_follow` - Quick 2-agent check (8-12s). Use for monitoring.

{mode_instruction}

## HOW TO RESPOND

Every response must directly address the user's question. Structure:

1. CURRENT STATE: What's the market doing RIGHT NOW?
2. VALIDATION: Does current flow/technicals support their position?
3. VERDICT: Clear answer - HOLD, CUT, or ADJUST
4. RISK: What could invalidate this in next 5-10 mins?

After EVERY response, end with your ACTION STATE as JSON:
```json
{{"action": "CALL", "signal": "HOLD", "price": 582.50, "entry": 581.00, "stop": 579.50, "target": 584.00, "conviction": "HIGH"}}
```

Fields:
- action: CALL, PUT, HOLD, or EXIT (what position type they have or should exit)
- signal: HOLD (stay in position) or EXIT (cut now)
- price: current SPY price
- entry: their entry price (from their question, or current price if new)
- stop: stop loss level (where to cut)
- target: profit target level
- conviction: HIGH, MED, or LOW

## YOUR WORKFLOW

1. FIRST: Full analysis to understand their position context
2. THEN: Quick checks every 30-60s to validate
3. REACT: If thesis breaks, tell them IMMEDIATELY to cut

## RULES

- EVERY response must answer their question
- Be DIRECT: "HOLD - flow supports" or "CUT NOW - flow reversed"
- You are validating THEIR position, not finding new trades
- NEVER stop monitoring until user clears the position
- Always call a tool after each response

START NOW. Analyze the market and answer their question."""


def get_prompt_for_position(position: str, mode: str) -> str:
    """Generate the system prompt for position validation mode."""
    pt_tz = ZoneInfo("America/Los_Angeles")
    now = datetime.now(pt_tz)
    current_time_full = now.strftime("%Y-%m-%d %H:%M:%S PT")

    # Market hours: 6:30 AM - 1:00 PM PT
    market_status = 'OPEN' if (now.hour == 6 and now.minute >= 30) or (7 <= now.hour < 13) else 'CLOSED'
    timestamp_header = f"""<current_time>
Current Time: {current_time_full}
Market Session: {market_status}
</current_time>

"""

    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["auto"])
    prompt = POSITION_VALIDATION_PROMPT.format(
        position=position,
        mode_instruction=mode_instruction
    )
    return timestamp_header + prompt


def get_prompt_for_mode(mode: str) -> str:
    """Generate the system prompt based on current mode override."""
    pt_tz = ZoneInfo("America/Los_Angeles")
    now = datetime.now(pt_tz)
    current_time_full = now.strftime("%Y-%m-%d %H:%M:%S PT")

    # Market hours: 6:30 AM - 1:00 PM PT
    market_status = 'OPEN' if (now.hour == 6 and now.minute >= 30) or (7 <= now.hour < 13) else 'CLOSED'
    timestamp_header = f"""<current_time>
Current Time: {current_time_full}
Market Session: {market_status}
</current_time>

"""

    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["auto"])
    start_instruction = START_INSTRUCTIONS.get(mode, START_INSTRUCTIONS["auto"])
    base_prompt = CONTINUOUS_TRADER_PROMPT_BASE.format(
        mode_instruction=mode_instruction,
        start_instruction=start_instruction
    )
    return timestamp_header + base_prompt


def create_zero_dte_agent(mode: str = "auto", position: str = "") -> Agent:
    """Create the Zero-DTE Agent with mode-aware or position-aware prompt."""
    if position:
        prompt = get_prompt_for_position(position, mode)
        console.print(f"[cyan]Creating agent in VALIDATION mode: {mode}[/cyan]")
        console.print(f"[green]Position: {position[:60]}...[/green]" if len(position) > 60 else f"[green]Position: {position}[/green]")
    else:
        prompt = get_prompt_for_mode(mode)
        console.print(f"[cyan]Creating agent in SCANNING mode: {mode}[/cyan]")

    conversation_manager = SlidingWindowConversationManager(
        window_size=5,
        should_truncate_results=False
    )

    return Agent(
        model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        system_prompt=prompt,
        tools=[analyze_market, fast_follow],
        conversation_manager=conversation_manager
    )


def run_zero_dte_agent():
    """
    Run the Zero-DTE Agent - it will run forever.

    The agent calls call_swarm() repeatedly, and each call
    streams both the question and response to the UI.

    Modes:
    - SCANNING: Looking for new 0DTE setups (default)
    - VALIDATING: Monitoring a specific position/question set by user
    """
    console.print(Panel.fit(
        "[bold cyan]Zero-DTE Agent - Continuous Thinking Mode[/bold cyan]\n\n"
        "[green]Mode:[/green] Runs forever, never stops\n"
        "[yellow]Behavior:[/yellow] Queries swarm, asks follow-ups, streams everything\n"
        "[blue]Output:[/blue] Every exchange streams to UI via SSE\n\n"
        "[magenta]Position Mode:[/magenta] Set a custom question in the UI to switch from\n"
        "                scanning to validating your specific position.\n\n"
        "[dim]Press Ctrl+C to stop[/dim]",
        title="[bold]Starting Agent[/bold]",
        border_style="cyan"
    ))

    # Track current state - loads from Redis (persists across restarts)
    current_mode = get_mode_override()
    current_position = get_position_context()

    console.print(f"[bold green]Loaded mode from Redis: {current_mode}[/bold green]")
    if current_position:
        console.print(f"[bold green]Loaded position from Redis: {current_position[:50]}...[/bold green]")

    agent = create_zero_dte_agent(current_mode, current_position)

    # Set initial prompt based on whether we have a position
    if current_position:
        prompt = f"User has a position/question: '{current_position}'. Analyze the market and validate their position now."
    else:
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

            # Check for changes in mode or position
            new_mode = get_mode_override()
            new_position = get_position_context()

            # Detect if we need to recreate the agent
            need_recreate = False
            mode_changed = new_mode != current_mode
            position_changed = new_position != current_position

            if mode_changed:
                console.print(f"\n[bold yellow]Mode changed: {current_mode} -> {new_mode}[/bold yellow]")
                current_mode = new_mode
                need_recreate = True

            if position_changed:
                if new_position and not current_position:
                    console.print(f"\n[bold green]>>> POSITION SET - Switching to VALIDATION mode <<<[/bold green]")
                    console.print(f"[green]Question: {new_position}[/green]")
                elif not new_position and current_position:
                    console.print(f"\n[bold yellow]>>> POSITION CLEARED - Returning to SCANNING mode <<<[/bold yellow]")
                else:
                    console.print(f"\n[bold yellow]Position updated[/bold yellow]")
                current_position = new_position
                need_recreate = True

            if need_recreate:
                agent = create_zero_dte_agent(current_mode, current_position)
                if current_position:
                    prompt = f"User updated their position/question: '{current_position}'. Analyze and validate now."
                else:
                    prompt = f"Returned to scanning mode. {START_INSTRUCTIONS.get(current_mode, 'Call analyze_market.')}"

            try:
                # Agent should run continuously, but if it returns, restart it
                result = agent(prompt)

                # Track outer agent tokens (the orchestrating Haiku agent)
                if hasattr(result, 'metrics') and hasattr(result.metrics, 'accumulated_usage'):
                    usage = result.metrics.accumulated_usage
                    outer_input = usage.get('inputTokens', 0)
                    outer_output = usage.get('outputTokens', 0)
                    if outer_input > 0 or outer_output > 0:
                        tracker = TokenTracker(mode=current_mode, model="haiku-4.5")
                        tracker.record("zero_dte_agent", outer_input, outer_output)
                        tracker.finish()
                        console.print(f"[dim]Outer agent tokens: {outer_input:,} in / {outer_output:,} out[/dim]")

                # If agent returns without error, it stopped - restart it
                console.print("\n[yellow]Agent stopped - restarting...[/yellow]")
                if current_position:
                    prompt = f"Continue validating the user's position: '{current_position}'. Call your next tool now."
                else:
                    prompt = "Continue monitoring. Call your next tool now."
                time.sleep(2)

            except Exception as e:
                error_msg = str(e)
                console.print(f"\n[yellow]Agent error: {error_msg}[/yellow]")
                console.print("[cyan]Restarting in 3 seconds...[/cyan]")

                # Publish error to UI so user sees it
                error_signal = {"action": "ERROR", "conviction": "HIGH", "error": error_msg[:200]}
                stream_to_ui("SWARM_ERROR", f"⚠️ ERROR: {error_msg[:300]}", error_signal)

                time.sleep(3)
                if current_position:
                    prompt = f"Resume validating position: '{current_position}'. Call your tool now."
                else:
                    prompt = f"Resume monitoring SPY. {START_INSTRUCTIONS.get(current_mode, 'Call analyze_market.')}"

    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping Zero-DTE Agent...[/bold red]")


if __name__ == "__main__":
    run_zero_dte_agent()