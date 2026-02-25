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
        val = engine._current_state.to_prompt_text()
        print(val)
        return val

    @tool
    def get_current_spy_price() -> str:
        """Get the absolute latest SPY price from Redis (updated every 1 second).
        Use this when you need the most current price for entry/exit decisions.

        Returns: Current SPY price with bid/ask/mid and timestamp.
        """
        import json
        try:
            # Primary: fresh quote from market-quote poller
            raw = engine.redis.get("market:spy:quote")
            if raw:
                q = json.loads(raw)
                return (
                    f"SPY: ${q['mid']:.2f} (bid={q['bid']:.2f} ask={q['ask']:.2f} "
                    f"spread={q['spread_bps']:.1f}bps) as of {q['timestamp']}"
                )
            # Fallback: technicals key
            raw = engine.redis.get("market:spy:data")
            if raw:
                data = json.loads(raw)
                price = data.get("price", {}).get("current", 0)
                timestamp = data.get("timestamp", "")
                change_pct = data.get("price", {}).get("change_pct", 0)
                return f"SPY: ${price:.2f} ({change_pct:+.2f}%) as of {timestamp}"
            return "SPY price not available"
        except Exception as e:
            return f"Error fetching price: {e}"


    @tool
    def recall_memory(search_query: str) -> str:
        """Search past trade outcomes and lessons with a custom query.

        NOTE: Current-condition memories are already in your prompt under <memory>.
        Use this tool ONLY when you want to search for something specific beyond
        what's already provided.

        Args:
            search_query: Natural language query describing what to search for.
                         Example: "PUT trades with early exits when flow weakened after entry"
                         Example: "missed opportunities in afternoon session with lean buying"

        Returns 4 types of memory:
        - OUTCOME: WIN/LOSS — actual past trades with flow metrics and labels.
        - MISSED_OPPORTUNITY — times you said WAIT but price moved in flow direction.
        - POST_EXIT_ANALYSIS — what happened 5 min after exit (EARLY_EXIT/GOOD_EXIT/NEUTRAL) with lessons.
        - POST_WAIT_ANALYSIS — what happened 5 min after WAIT (MISSED/GOOD_WAIT/NEUTRAL) with lessons.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 Custom memory search: {search_query}")

        memory_id = engine.memory.get_or_create_memory()
        if not memory_id:
            return "No memory store available."

        records = engine.memory.memory_client.retrieve_memories(
            memory_id=memory_id,
            namespace="/facts/trader/SPY/",
            query=search_query,
            top_k=15,
        )

        memories = []
        for record in records:
            score = record.get("score", 1.0)
            if score < 0.3:
                continue
            text = engine.memory._extract_text(record)
            if text:
                memories.append(text)

        logger.info(f"✅ Found {len(memories)} relevant memories (score >= 0.3)")

        if not memories:
            return "No relevant memories found for that query."

        lines = [f"Found {len(memories)} memories:\n"]
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

    return [get_market_state, get_current_spy_price, recall_memory, check_position]
