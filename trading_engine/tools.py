"""
Strands @tool definitions for the v2 trading engine agent.

Three tools that close over the engine instance, giving the agent access to:
- Current classified market state
- Recalled memories (past outcomes, missed opportunities, learned rules)
- Active position status
"""

from strands import tool


def create_engine_tools(engine):
    """
    Create tool functions that close over the engine instance.

    Returns a list of @tool-decorated callables for the Strands Agent.
    """

    @tool
    def get_market_state() -> str:
        """Fetch and return current classified market labels for SPY.
        Call this FIRST every cycle to see what the market is doing.

        Returns pre-classified labels with raw numbers:
        - Flow direction + ratio + momentum
        - RSI, VWAP position, EMA/MACD, ORB status
        - ORB regime (TREND_CONTINUATION / MEAN_REVERSION / UNKNOWN)
        - Mag7 breadth alignment
        """
        if engine._current_state is None:
            return "No market state available yet."
        return engine._current_state.to_prompt_text()

    @tool
    def recall_memory() -> str:
        """Recall past trade outcomes and learned rules for current market conditions.

        Returns 3 types of memory:
        - OUTCOME: WIN/LOSS — actual past trades with same labels. Learn from wins AND losses.
        - MISSED_OPPORTUNITY — times you said WAIT but should have entered. Don't repeat this.
        - [LEARNED RULE] — extracted patterns from daily consolidation. Trust these.

        CRITICAL: If you see MISSED_OPPORTUNITY entries matching current conditions,
        you MUST enter this time. The whole point of memory is to stop repeating mistakes.
        """
        memories = engine._current_memories
        if not memories:
            return "No relevant memories found for current market conditions."

        lines = [f"Found {len(memories)} relevant memories:\n"]
        for i, mem in enumerate(memories, 1):
            lines.append(f"{i}. {mem}")
        return "\n".join(lines)

    @tool
    def check_position() -> str:
        """Check your current active position (or confirm you're flat).

        Returns position details if active:
        - Direction (CALL/PUT), entry price, stop, target, conviction
        - Entry time and session

        Or confirms you are flat and scanning for new entries.
        """
        pos = engine.position
        if not pos:
            return "FLAT — no active position. Scanning for entries."

        return (
            f"ACTIVE {pos['action']} POSITION:\n"
            f"  Entry: ${pos.get('entry', '?')} | Stop: ${pos.get('stop', '?')} | "
            f"Target: ${pos.get('target', '?')}\n"
            f"  Conviction: {pos.get('conviction', '?')} | "
            f"Entered: {pos.get('entry_time', '?')}"
        )

    return [get_market_state, recall_memory, check_position]
