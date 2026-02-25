"""
Test script: Twelve Data WebSocket for SPY real-time price ticks.

Connects to wss://ws.twelvedata.com, subscribes to SPY, prints ticks.
Uses the async `websockets` library (already installed).

Usage:
    python test_ws_twelvedata.py
    python test_ws_twelvedata.py --duration 60   # run for 60 seconds
"""

import asyncio
import json
import time
import signal
import argparse
from datetime import datetime

# Load API key same way as config/settings.py
def _load_api_key() -> str:
    try:
        with open("/etc/twelve_data_api_key.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        import os
        key = os.getenv("TWELVE_DATA_API_KEY", "")
        if not key:
            raise RuntimeError(
                "No API key: /etc/twelve_data_api_key.txt not found and "
                "TWELVE_DATA_API_KEY env var not set"
            )
        return key


async def main(duration: int, symbol: str = "SPY"):
    import websockets

    api_key = _load_api_key()
    url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={api_key}"

    print(f"Connecting to Twelve Data WebSocket...")
    print(f"  URL: wss://ws.twelvedata.com/v1/quotes/price?apikey=***")
    print(f"  Symbol: {symbol}")
    print(f"  Duration: {duration}s")
    print()

    tick_count = 0
    start = time.time()
    last_heartbeat = time.time()
    last_price = None

    shutdown = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            print(f"Connected!\n")

            # Subscribe to SPY
            subscribe_msg = json.dumps({
                "action": "subscribe",
                "params": {"symbols": symbol},
            })
            await ws.send(subscribe_msg)
            print(f"Sent: {subscribe_msg}")
            print()

            while not shutdown.is_set():
                # Check duration
                elapsed = time.time() - start
                if elapsed >= duration:
                    print(f"\nDuration reached ({duration}s). Stopping.")
                    break

                # Send heartbeat every 10s
                if time.time() - last_heartbeat >= 10:
                    await ws.send(json.dumps({"action": "heartbeat"}))
                    last_heartbeat = time.time()

                # Wait for message with timeout
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"  [non-json] {raw[:200]}")
                    continue

                event_type = msg.get("event")
                now = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                if event_type == "subscribe-status":
                    status = msg.get("status")
                    print(f"[{now}] subscribe-status: {status}")
                    print(f"  Full message: {json.dumps(msg, indent=2)}")
                    if status == "error":
                        print(f"\n  Check your plan supports WebSocket (Pro+ required)")
                    print()

                elif event_type == "price":
                    tick_count += 1
                    price = msg.get("price")
                    ts = msg.get("timestamp")
                    day_vol = msg.get("day_volume")
                    # Debug: print full first tick to see all fields
                    if tick_count == 1:
                        print(f"  Raw first tick: {json.dumps(msg)}")
                        print()

                    # Calculate delta from last tick
                    delta = ""
                    if last_price is not None and price is not None:
                        diff = float(price) - float(last_price)
                        arrow = "^" if diff > 0 else "v" if diff < 0 else "="
                        delta = f" {arrow} {diff:+.2f}"
                    if price is not None:
                        last_price = float(price)

                    # Quote age: how old is this trade?
                    age_str = ""
                    if ts:
                        try:
                            msg_time = float(ts)
                            age_s = time.time() - msg_time
                            if age_s < 2:
                                age_str = f" | fresh ({age_s:.1f}s)"
                            else:
                                age_str = f" | age={age_s:.0f}s"
                        except (ValueError, TypeError):
                            pass

                    vol_str = f" | vol={int(day_vol):,}" if day_vol else ""
                    print(
                        f"[{now}] #{tick_count:4d} | {symbol} ${price}{delta}"
                        f"{vol_str}{age_str}"
                    )

                elif event_type == "heartbeat":
                    status = msg.get("status", "")
                    print(f"[{now}] heartbeat: {status}")

                else:
                    # Unknown event — print raw
                    print(f"[{now}] {event_type}: {json.dumps(msg)[:200]}")

    except Exception as e:
        print(f"\nError: {e}")

    # Summary
    elapsed = time.time() - start
    rate = tick_count / elapsed if elapsed > 0 else 0
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Ticks received: {tick_count}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Rate: {rate:.1f} ticks/sec")
    print(f"  Last price: ${last_price}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Twelve Data WebSocket for SPY")
    parser.add_argument("--duration", type=int, default=30, help="Run for N seconds (default: 30)")
    parser.add_argument("--symbol", type=str, default="SPY", help="Symbol to subscribe (default: SPY)")
    args = parser.parse_args()
    asyncio.run(main(args.duration, args.symbol))
