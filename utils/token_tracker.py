"""
Token Tracker - Track token usage per agent invocation

Provides:
- TokenTracker class for tracking tokens across a cycle
- Records per-agent input/output tokens
- Stores to Redis (real-time) and JSONL (historical)

Usage:
    tracker = TokenTracker(mode="fast")
    tracker.record("zero_dte_agent", input_tokens=2450, output_tokens=380)
    tracker.record("order_flow", input_tokens=1200, output_tokens=450)
    tracker.finish()  # Saves to Redis and JSONL
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PT_TZ = ZoneInfo("America/Los_Angeles")

# JSONL log directory
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class TokenTracker:
    """Track token usage for a single analysis cycle."""

    def __init__(self, mode: str = "fast", model: str = "haiku-4.5"):
        """
        Initialize tracker for a new cycle.

        Args:
            mode: Analysis mode (fast, full, auto)
            model: Model being used (haiku-4.5, sonnet-4, etc.)
        """
        self.mode = mode
        self.model = model
        self.start_time = datetime.now(PT_TZ)
        self.agents: dict[str, dict] = {}
        self.total_input = 0
        self.total_output = 0

    def record(self, agent_name: str, input_tokens: int = 0, output_tokens: int = 0):
        """
        Record token usage for an agent.

        Args:
            agent_name: Name of the agent (e.g., "order_flow", "tech", "coordinator")
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
        """
        if agent_name not in self.agents:
            self.agents[agent_name] = {"input": 0, "output": 0}

        self.agents[agent_name]["input"] += input_tokens
        self.agents[agent_name]["output"] += output_tokens
        self.total_input += input_tokens
        self.total_output += output_tokens

        logger.debug(f"Token record: {agent_name} +{input_tokens}in/{output_tokens}out")

    def finish(self) -> dict:
        """
        Finish the cycle and save to Redis + JSONL.

        Returns:
            Dict with cycle token data
        """
        timestamp = self.start_time.strftime("%H:%M:%S")
        today = self.start_time.strftime("%Y-%m-%d")

        cycle_data = {
            "timestamp": timestamp,
            "date": today,
            "mode": self.mode,
            "model": self.model,
            "agents": self.agents,
            "total_input": self.total_input,
            "total_output": self.total_output
        }

        # Save to Redis
        self._save_to_redis(cycle_data, today)

        # Save to JSONL
        self._save_to_jsonl(cycle_data, today)

        logger.info(f"Cycle tokens: {self.total_input} in / {self.total_output} out")

        return cycle_data

    def _save_to_redis(self, cycle_data: dict, today: str):
        """Save cycle data to Redis for real-time UI."""
        try:
            from redis_stream import get_stream

            stream = get_stream()

            # Update daily summary
            summary_key = f"zero_dte:tokens:summary:{today}"
            summary_str = stream.redis.get(summary_key)

            if summary_str:
                summary = json.loads(summary_str)
            else:
                summary = {"cycles": 0, "input_tokens": 0, "output_tokens": 0}

            summary["cycles"] += 1
            summary["input_tokens"] += self.total_input
            summary["output_tokens"] += self.total_output

            stream.redis.set(summary_key, json.dumps(summary))
            # Set TTL to 24 hours
            stream.redis.expire(summary_key, 60 * 60 * 24)

            # Add to cycle history (lpush = newest first)
            history_key = f"zero_dte:tokens:history:{today}"
            stream.redis.lpush(history_key, json.dumps(cycle_data))
            # Trim to last 100 cycles
            stream.redis.ltrim(history_key, 0, 99)
            stream.redis.expire(history_key, 60 * 60 * 24)

        except Exception as e:
            logger.error(f"Failed to save tokens to Redis: {e}")

    def _save_to_jsonl(self, cycle_data: dict, today: str):
        """Save cycle data to JSONL file for historical analysis."""
        try:
            log_file = LOG_DIR / f"tokens_{today}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(cycle_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to save tokens to JSONL: {e}")


def get_daily_summary(date: Optional[str] = None) -> dict:
    """
    Get token usage summary for a day.

    Args:
        date: Date string (YYYY-MM-DD). Defaults to today.

    Returns:
        Dict with cycles, input_tokens, output_tokens
    """
    if date is None:
        date = datetime.now(PT_TZ).strftime("%Y-%m-%d")

    try:
        from redis_stream import get_stream

        stream = get_stream()
        summary_key = f"zero_dte:tokens:summary:{date}"
        summary_str = stream.redis.get(summary_key)

        if summary_str:
            return json.loads(summary_str)
        return {"cycles": 0, "input_tokens": 0, "output_tokens": 0}
    except Exception as e:
        logger.error(f"Failed to get daily summary: {e}")
        return {"cycles": 0, "input_tokens": 0, "output_tokens": 0}