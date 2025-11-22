"""
SSE Server for Zero-DTE Agent UI Streaming

Serves the UI and streams agent/swarm messages via Server-Sent Events.

Usage:
    # Terminal 1: Start the server
    python server.py

    # Terminal 2: Start the agent
    python zero_dte_agent.py

    # Open browser to http://localhost:5000
"""

import json
import threading
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from rich.console import Console

console = Console()

# Import the signal queue from the agent
# This will be populated when zero_dte_agent runs
signal_queue = None

# Store connected SSE clients
sse_clients = []


class StreamingHandler(SimpleHTTPRequestHandler):
    """HTTP handler with SSE support"""

    def __init__(self, *args, **kwargs):
        # Serve files from the ui/ directory
        self.directory = str(Path(__file__).parent / "ui")
        super().__init__(*args, directory=self.directory, **kwargs)

    def do_GET(self):
        if self.path == "/stream":
            self.handle_sse()
        elif self.path == "/":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def handle_sse(self):
        """Handle Server-Sent Events connection"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        console.print("[green]SSE client connected[/green]")

        # Send initial connection message
        self.wfile.write(b"data: {\"type\": \"CONNECTED\", \"content\": \"Connected to Zero-DTE Agent stream\"}\n\n")
        self.wfile.flush()

        # Keep connection open and stream messages
        try:
            while True:
                if signal_queue and not signal_queue.empty():
                    try:
                        event = signal_queue.get_nowait()
                        data = json.dumps(event)
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                    except Exception as e:
                        console.print(f"[red]Error sending SSE: {e}[/red]")
                        break
                else:
                    # Send keepalive every 15 seconds
                    time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            console.print("[yellow]SSE client disconnected[/yellow]")

    def log_message(self, format, *args):
        # Suppress default logging for cleaner output
        pass


def run_server(port: int = 8080):
    """Run the HTTP/SSE server"""
    global signal_queue

    # Import the queue from the agent module
    try:
        from zero_dte_agent import get_signal_queue
        signal_queue = get_signal_queue()
        console.print("[green]Connected to agent signal queue[/green]")
    except ImportError:
        console.print("[yellow]Agent not imported yet - queue will connect when agent starts[/yellow]")

    server = HTTPServer(("", port), StreamingHandler)

    console.print(f"""
[bold cyan]╔══════════════════════════════════════════════════════════╗
║           Zero-DTE Agent - SSE Server                    ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  UI:     http://localhost:{port}                          ║
║  SSE:    http://localhost:{port}/stream                   ║
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
        server.shutdown()


if __name__ == "__main__":
    run_server(port=5000)