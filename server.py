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
import os
import threading
import asyncio
from datetime import datetime
from pathlib import Path
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from http.server import HTTPServer
from rich.console import Console


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a separate thread"""
    daemon_threads = True

    def handle_error(self, request, client_address):
        """Suppress noisy disconnect errors from SSE clients"""
        import sys
        exc_type = sys.exc_info()[0]
        # Ignore normal disconnect errors
        if exc_type in (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return
        # Log other errors normally
        super().handle_error(request, client_address)

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
        elif self.path == "/get-position":
            self.handle_get_position()
        elif self.path == "/token-stats":
            self.handle_token_stats()
        elif self.path == "/oi-results":
            self.handle_oi_results()
        elif self.path == "/oi-ondemand-results":
            self.handle_oi_ondemand_results()
        elif self.path == "/oi-status":
            self.handle_oi_status()
        elif self.path == "/oi-stream":
            self.handle_oi_stream()
        elif self.path == "/":
            self.path = "/index.html"
            super().do_GET()
        elif self.path == "/v2":
            self.path = "/terminal_v2.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/set-mode":
            self.handle_set_mode()
        elif self.path == "/set-position":
            self.handle_set_position()
        elif self.path == "/clear-position":
            self.handle_clear_position()
        elif self.path == "/exit-trade":
            self.handle_exit_trade()
        elif self.path == "/run-oi-analysis":
            self.handle_run_oi_analysis()
        elif self.path == "/oi-analyze-ticker":
            self.handle_oi_analyze_ticker()
        elif self.path == "/cancel-oi-analysis":
            self.handle_cancel_oi_analysis()
        elif self.path == "/reset-oi-status":
            self.handle_reset_oi_status()
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

    def handle_get_position(self):
        """Return current position/custom question"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        position = redis_stream.redis.get("zero_dte:position") or ""
        self.wfile.write(json.dumps({"position": position}).encode())

    def handle_set_position(self):
        """Set position/custom question for validation mode"""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode()

        try:
            data = json.loads(body)
            position = data.get('position', '').strip()

            if position:
                # Store in Redis
                redis_stream.redis.set("zero_dte:position", position)
                display = position[:50] + "..." if len(position) > 50 else position
                console.print(f"[green]Position set: {display}[/green]")
            else:
                redis_stream.redis.delete("zero_dte:position")
                console.print("[yellow]Position cleared[/yellow]")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "position": position}).encode())

        except (json.JSONDecodeError, ValueError) as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def handle_clear_position(self):
        """Clear position - return to scanning mode"""
        redis_stream.redis.delete("zero_dte:position")
        console.print("[yellow]Position cleared - returning to scanning mode[/yellow]")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

    def handle_exit_trade(self):
        """
        Exit current trade and return to scan mode.
        Writes an EXIT event to history so get_last_recommendation() returns {}.
        """
        from zoneinfo import ZoneInfo
        pt_tz = ZoneInfo("America/Los_Angeles")
        timestamp = datetime.now(pt_tz).strftime("%Y-%m-%d %H:%M:%S PT")

        # Create EXIT event that matches SWARM_RESPONSE format
        exit_event = {
            "type": "SWARM_RESPONSE",
            "content": "Manual exit - position closed, scanning for new trade",
            "timestamp": timestamp,
            "signal": {
                "action": "EXIT",
                "signal": "EXIT",
                "conviction": "HIGH",
                "source": "manual"
            }
        }

        # Push to history (lpush = newest first)
        redis_stream.redis.lpush("zero_dte:history", json.dumps(exit_event))

        # Also publish to SSE so UI updates immediately
        redis_stream.publish(exit_event)

        console.print("[bold yellow]MANUAL EXIT - Position closed, scanning for new trade[/bold yellow]")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "message": "Position closed"}).encode())

    def handle_history(self):
        """Return event history as JSON for instant UI loading"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        history = redis_stream.get_history(limit=100)
        self.wfile.write(json.dumps(history).encode())

    def handle_token_stats(self):
        """Return token usage statistics for the day"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        from zoneinfo import ZoneInfo
        pt_tz = ZoneInfo("America/Los_Angeles")
        today = datetime.now(pt_tz).strftime("%Y-%m-%d")

        # Get daily summary from Redis
        summary_key = f"zero_dte:tokens:summary:{today}"
        summary_str = redis_stream.redis.get(summary_key)

        if summary_str:
            summary = json.loads(summary_str)
        else:
            summary = {"cycles": 0, "input_tokens": 0, "output_tokens": 0}

        # Get cycle history (last 50 cycles)
        history_key = f"zero_dte:tokens:history:{today}"
        history_raw = redis_stream.redis.lrange(history_key, 0, 49)
        history = []
        for item in history_raw:
            try:
                history.append(json.loads(item))
            except json.JSONDecodeError:
                pass

        response = {
            "cycles": summary.get("cycles", 0),
            "input_tokens": summary.get("input_tokens", 0),
            "output_tokens": summary.get("output_tokens", 0),
            "history": history
        }

        self.wfile.write(json.dumps(response).encode())

    def handle_oi_results(self):
        """Return latest OI analysis results"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        from oi.redis_manager import get_full_results
        results = get_full_results()
        self.wfile.write(json.dumps(results or {}).encode())

    def handle_oi_ondemand_results(self):
        """Return all on-demand results for today"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        from oi.redis_manager import get_ondemand_results
        results = get_ondemand_results()
        self.wfile.write(json.dumps(results).encode())

    def handle_oi_status(self):
        """Return current OI analysis status"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        from oi.redis_manager import get_status
        status = get_status()
        self.wfile.write(json.dumps(status).encode())

    def handle_oi_stream(self):
        """SSE stream for OI analysis progress events"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        import redis as redis_lib
        r = redis_lib.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        ps = r.pubsub()
        ps.subscribe("oi:events")

        try:
            while True:
                message = ps.get_message(timeout=5)
                if message and message["type"] == "message":
                    self.wfile.write(f"data: {message['data']}\n\n".encode())
                    self.wfile.flush()
                else:
                    # Heartbeat — detects broken connections
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            ps.close()
            r.close()

    def handle_run_oi_analysis(self):
        """Start OI analysis in background thread"""
        from oi.redis_manager import get_status

        # Check if already running
        status = get_status()
        if status.get("status") == "running":
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Analysis already running"}).encode())
            return

        def run_analysis():
            from oi.analyzer import OIAnalyzer
            analyzer = OIAnalyzer()
            asyncio.run(analyzer.run_analysis())

        threading.Thread(target=run_analysis, daemon=True).start()
        console.print("[bold cyan]OI Analysis started in background[/bold cyan]")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "started"}).encode())

    def handle_oi_analyze_ticker(self):
        """Analyze a single ticker on-demand"""
        from oi.redis_manager import get_status

        content_length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_length).decode())
        ticker = body.get("ticker", "").upper().strip()
        dtes = body.get("dtes", [30, 60, 90])

        if not ticker:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "ticker is required"}).encode())
            return

        # Check if already running
        status = get_status()
        if status.get("status") == "running":
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Analysis already running"}).encode())
            return

        def run():
            from oi.analyzer import OIAnalyzer
            analyzer = OIAnalyzer()
            asyncio.run(analyzer.analyze_single_ticker(ticker, dtes))

        threading.Thread(target=run, daemon=True).start()
        console.print(f"[bold cyan]OI Analysis started for {ticker} (DTEs: {dtes})[/bold cyan]")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "started", "ticker": ticker, "dtes": dtes}).encode())

    def handle_cancel_oi_analysis(self):
        """Cancel a running OI analysis"""
        from oi.redis_manager import request_cancel

        request_cancel()
        console.print("[bold yellow]OI Analysis cancel requested[/bold yellow]")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "cancelling"}).encode())

    def handle_reset_oi_status(self):
        """Force-reset OI analysis status to idle (unstick crashed analysis)"""
        from oi.redis_manager import reset_status

        reset_status()
        console.print("[bold yellow]OI Analysis status reset to idle[/bold yellow]")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "idle"}).encode())

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

        # Subscribe to Redis and stream events with heartbeat
        try:
            ps = redis_stream.redis.pubsub()
            ps.subscribe("zero_dte:events")
            while True:
                message = ps.get_message(timeout=5)
                if message and message["type"] == "message":
                    self.wfile.write(f"data: {message['data']}\n\n".encode())
                    self.wfile.flush()
                else:
                    # Heartbeat — detects broken connections fast
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            try:
                ps.close()
            except Exception:
                pass

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

    # Reset OI status in case previous run left it stuck on "running"
    from oi.redis_manager import reset_status
    reset_status()

    server = ThreadingHTTPServer(("", port), StreamingHandler)

    console.print(f"""
[bold cyan]╔══════════════════════════════════════════════════════════╗
║           Zero-DTE Agent - SSE Server (Redis)            ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  UI v1:   http://localhost:{port}                          ║
║  UI v2:   http://localhost:{port}/v2  [green](Bloomberg-style)[/green]     ║
║  SSE:     http://localhost:{port}/stream                   ║
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