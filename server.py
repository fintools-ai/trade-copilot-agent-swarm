"""
SSE Server for Zero-DTE Agent UI Streaming (Redis-powered)

Serves the UI and streams agent/swarm messages via Server-Sent Events.
Uses Redis pub/sub for real-time cross-process communication.

Usage:
    # Start Redis first
    brew services start redis  # macOS

    # Terminal 1: Start the server
    python server.py

    # Terminal 2: Start the agent
    python zero_dte_agent.py

    # Open browser to http://localhost:5000
"""

import json
from pathlib import Path
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from http.server import HTTPServer
from rich.console import Console


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a separate thread"""
    daemon_threads = True

from redis_stream import get_stream, RedisStream

console = Console()

# Redis stream instance (reset session on server start)
redis_stream: RedisStream = None

# Default mode when not set in Redis (must match zero_dte_agent.py)
DEFAULT_MODE = "fast"


class StreamingHandler(SimpleHTTPRequestHandler):
    """HTTP handler with SSE and history support"""

    def __init__(self, *args, **kwargs):
        # Serve files from the ui/ directory
        self.directory = str(Path(__file__).parent / "ui")
        super().__init__(*args, directory=self.directory, **kwargs)

    def do_GET(self):
        if self.path == "/stream":
            self.handle_sse()
        elif self.path == "/history":
            self.handle_history()
        elif self.path == "/get-mode":
            self.handle_get_mode()
        elif self.path == "/":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/set-mode":
            self.handle_set_mode()
        else:
            self.send_error(404, "Not Found")

    def handle_get_mode(self):
        """Return current mode override setting"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        mode = redis_stream.redis.get("zero_dte:mode_override") or DEFAULT_MODE
        self.wfile.write(json.dumps({"mode": mode}).encode())

    def handle_set_mode(self):
        """Set mode override (auto, fast, or full)"""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode()

        try:
            data = json.loads(body)
            mode = data.get('mode', 'auto')

            # Validate mode
            if mode not in ('auto', 'fast', 'full'):
                mode = 'auto'

            # Store in Redis
            redis_stream.redis.set("zero_dte:mode_override", mode)
            console.print(f"[cyan]Mode override set to: {mode}[/cyan]")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "mode": mode}).encode())

        except (json.JSONDecodeError, ValueError) as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def handle_history(self):
        """Return event history as JSON for instant UI loading"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        history = redis_stream.get_history(limit=100)
        self.wfile.write(json.dumps(history).encode())

    def handle_sse(self):
        """Handle Server-Sent Events connection with Redis pub/sub"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        console.print("[green]SSE client connected[/green]")

        # Send connection confirmation
        connect_event = {
            "type": "CONNECTED",
            "content": "Connected to Zero-DTE Agent stream",
            "session_id": redis_stream.get_session_id()
        }
        self.wfile.write(f"data: {json.dumps(connect_event)}\n\n".encode())
        self.wfile.flush()

        # Subscribe to Redis and stream events
        try:
            for event in redis_stream.subscribe():
                data = json.dumps(event)
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            console.print("[yellow]SSE client disconnected[/yellow]")

    def log_message(self, format, *args):
        # Suppress default logging for cleaner output
        pass


def run_server(port: int = 5000):
    """Run the HTTP/SSE server with Redis"""
    global redis_stream

    # Initialize Redis (history persists with TTL)
    try:
        redis_stream = get_stream(reset_on_init=False)
    except Exception as e:
        console.print(f"[red]Failed to connect to Redis: {e}[/red]")
        console.print("[yellow]Make sure Redis is running: brew services start redis[/yellow]")
        return

    server = ThreadingHTTPServer(("", port), StreamingHandler)

    console.print(f"""
[bold cyan]╔══════════════════════════════════════════════════════════╗
║           Zero-DTE Agent - SSE Server (Redis)            ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  UI:      http://localhost:{port}                          ║
║  SSE:     http://localhost:{port}/stream                   ║
║  History: http://localhost:{port}/history                  ║
║                                                          ║
║  [yellow]History persists (8hr TTL)[/yellow]                            ║
║                                                          ║
║  [green]Start the agent in another terminal:[/green]                   ║
║  [yellow]python zero_dte_agent.py[/yellow]                              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
[/bold cyan]""")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[bold red]Shutting down server...[/bold red]")
        redis_stream.close()
        server.shutdown()


if __name__ == "__main__":
    run_server(port=5000)