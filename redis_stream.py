"""
Redis Stream - Fast pub/sub + history for Zero-DTE Agent UI

Features:
- Pub/sub for real-time streaming across processes
- History storage for instant UI loading
- Session key resets on server restart (clean slate)
- TTL support for automatic cleanup

Usage:
    from redis_stream import RedisStream

    stream = RedisStream()
    stream.publish({"type": "AGENT_QUESTION", "content": "..."})

    # Get history
    history = stream.get_history(limit=100)

    # Subscribe to real-time events
    for event in stream.subscribe():
        print(event)
"""

import json
import uuid
import redis
from datetime import datetime
from typing import Generator, Optional
from rich.console import Console

console = Console()

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Channel and key names
CHANNEL_NAME = "zero_dte:events"
HISTORY_KEY = "zero_dte:history"
SESSION_KEY = "zero_dte:session"

# History settings
MAX_HISTORY = 500  # Keep last 500 events
HISTORY_TTL = 3600 * 8  # 8 hours TTL (trading day)


class RedisStream:
    """
    Redis-based event streaming for Zero-DTE Agent.

    Provides:
    - publish(): Send events to subscribers + store in history
    - subscribe(): Real-time event stream (generator)
    - get_history(): Load past events instantly
    - reset_session(): Clear history on server restart
    """

    def __init__(self, reset_on_init: bool = False):
        """
        Initialize Redis connection.

        Args:
            reset_on_init: If True, clears history on initialization (for server)
        """
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        self.pubsub = None
        self.session_id = None

        # Test connection
        try:
            self.redis.ping()
            console.print("[green]Redis connected[/green]")
        except redis.ConnectionError:
            console.print("[red]Redis connection failed - make sure Redis is running[/red]")
            console.print("[yellow]Run: brew services start redis (macOS)[/yellow]")
            raise

        if reset_on_init:
            self.reset_session()

    def reset_session(self) -> str:
        """
        Reset session - clears history and generates new session ID.
        Call this on server restart for clean slate.

        Returns:
            New session ID
        """
        self.session_id = str(uuid.uuid4())[:8]

        # Clear old history
        self.redis.delete(HISTORY_KEY)

        # Store new session ID
        self.redis.set(SESSION_KEY, self.session_id)

        console.print(f"[cyan]New session: {self.session_id}[/cyan]")

        # Publish session reset event
        reset_event = {
            "type": "SESSION_RESET",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "session_id": self.session_id,
            "content": "New session started"
        }
        self.redis.publish(CHANNEL_NAME, json.dumps(reset_event))

        return self.session_id

    def get_session_id(self) -> Optional[str]:
        """Get current session ID"""
        return self.redis.get(SESSION_KEY)

    def publish(self, event: dict) -> None:
        """
        Publish an event to subscribers and store in history.

        Args:
            event: Event dict with type, content, timestamp, etc.
        """
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.now().strftime("%H:%M:%S")

        # Add milliseconds for ordering
        event["ts_ms"] = datetime.now().timestamp()

        event_json = json.dumps(event)

        # Publish to real-time subscribers
        self.redis.publish(CHANNEL_NAME, event_json)

        # Store in history (LPUSH = prepend, newest first)
        self.redis.lpush(HISTORY_KEY, event_json)

        # Trim history to max size
        self.redis.ltrim(HISTORY_KEY, 0, MAX_HISTORY - 1)

        # Set TTL on history key
        self.redis.expire(HISTORY_KEY, HISTORY_TTL)

    def get_history(self, limit: int = 100) -> list[dict]:
        """
        Get event history (newest first by default).

        Args:
            limit: Max number of events to return

        Returns:
            List of events, oldest first (for UI display order)
        """
        # Get from Redis (newest first due to LPUSH)
        events_json = self.redis.lrange(HISTORY_KEY, 0, limit - 1)

        # Parse and reverse (so oldest first for UI)
        events = [json.loads(e) for e in events_json]
        events.reverse()

        return events

    def subscribe(self) -> Generator[dict, None, None]:
        """
        Subscribe to real-time events.

        Yields:
            Event dicts as they arrive
        """
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe(CHANNEL_NAME)

        console.print(f"[green]Subscribed to {CHANNEL_NAME}[/green]")

        for message in self.pubsub.listen():
            if message["type"] == "message":
                try:
                    event = json.loads(message["data"])
                    yield event
                except json.JSONDecodeError:
                    continue

    def subscribe_nonblocking(self) -> Optional[dict]:
        """
        Non-blocking subscribe check.

        Returns:
            Event dict if available, None otherwise
        """
        if self.pubsub is None:
            self.pubsub = self.redis.pubsub()
            self.pubsub.subscribe(CHANNEL_NAME)

        message = self.pubsub.get_message(ignore_subscribe_messages=True, timeout=0.01)

        if message and message["type"] == "message":
            try:
                return json.loads(message["data"])
            except json.JSONDecodeError:
                return None
        return None

    def close(self):
        """Close Redis connections"""
        if self.pubsub:
            self.pubsub.close()
        self.redis.close()


# Singleton instance for easy import
_stream_instance: Optional[RedisStream] = None


def get_stream(reset_on_init: bool = False) -> RedisStream:
    """
    Get the singleton RedisStream instance.

    Args:
        reset_on_init: Pass True from server.py to reset session
    """
    global _stream_instance
    if _stream_instance is None:
        _stream_instance = RedisStream(reset_on_init=reset_on_init)
    return _stream_instance


def publish_event(event_type: str, content: str, signal: dict = None) -> None:
    """
    Convenience function to publish an event.

    Args:
        event_type: AGENT_QUESTION, SWARM_RESPONSE, SIGNAL_UPDATE, etc.
        content: Event content/message
        signal: Optional signal data (direction, conviction, etc.)
    """
    stream = get_stream()

    event = {
        "type": event_type,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "content": content
    }

    if signal:
        event["signal"] = signal

    stream.publish(event)