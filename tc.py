#!/Users/sayantan/Documents/Workspace/mcp-order-flow-server/.env/bin/python3
"""
tc — Unified process manager for the trading copilot.

Usage:
    python tc.py doctor              Check status of all services
    python tc.py start [component]   Start server, engine, or both
    python tc.py stop  [component]   Stop engine, server, or both
    python tc.py logs  <component>   Tail logs for server or engine
"""

from __future__ import annotations

import os
import sys
import signal
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PID_DIR = ROOT / "data" / "pids"
LOG_DIR = ROOT / "logs"

COMPONENTS = {
    "server": {"script": "server.py", "label": "Server"},
    "engine": {"script": "zero_dte_v2.py", "label": "Engine"},
}

# ANSI colors
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _pid_path(name: str) -> Path:
    return PID_DIR / f"{name}.pid"


def _read_pid(name: str) -> int | None:
    """Read PID file and verify process is alive. Cleans up stale files."""
    path = _pid_path(name)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)  # check alive
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        path.unlink(missing_ok=True)
        return None


def _write_pid(name: str, pid: int) -> None:
    PID_DIR.mkdir(parents=True, exist_ok=True)
    _pid_path(name).write_text(str(pid))

def _create_log(name: str) -> Path:
    """Create timestamped log file and update _latest symlink."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = LOG_DIR / f"{name}_{ts}.log"
    log_file.touch()

    symlink = LOG_DIR / f"{name}_latest.log"
    symlink.unlink(missing_ok=True)
    symlink.symlink_to(log_file.name)
    return log_file


def _latest_log(name: str) -> Path | None:
    symlink = LOG_DIR / f"{name}_latest.log"
    if symlink.exists():
        return symlink
    return None


def _check_redis() -> bool:
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=2)
        return r.ping()
    except Exception:
        return False


def _check_server_http() -> bool:
    try:
        req = urllib.request.urlopen("http://localhost:5000/get-mode", timeout=3)
        return req.status == 200
    except Exception:
        return False


def cmd_doctor():
    """Check status of all services."""
    print(f"\n{BOLD}  Service     Status{RESET}")
    print(f"  {'─' * 25}")

    for name, info in COMPONENTS.items():
        pid = _read_pid(name)
        if pid:
            status = f"{GREEN}online{RESET}  (pid {pid})"
        else:
            status = f"{RED}offline{RESET}"
        print(f"  {info['label']:<11} {status}")

    redis_ok = _check_redis()
    status = f"{GREEN}online{RESET}" if redis_ok else f"{RED}offline{RESET}"
    print(f"  {'Redis':<11} {status}")
    print()


def cmd_start(target: str | None):
    """Start server, engine, or both."""
    if target and target not in COMPONENTS:
        print(f"{RED}Unknown component: {target}{RESET}")
        print(f"Valid: {', '.join(COMPONENTS)}")
        sys.exit(1)

    names = [target] if target else ["server", "engine"]

    for name in names:
        existing = _read_pid(name)
        if existing:
            print(f"{YELLOW}{COMPONENTS[name]['label']} already running (pid {existing}){RESET}")
            continue

        info = COMPONENTS[name]
        log_file = _create_log(name)
        script = ROOT / info["script"]

        with open(log_file, "a") as log_f:
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                stdout=log_f,
                stderr=log_f,
                cwd=str(ROOT),
                start_new_session=True,
            )

        _write_pid(name, proc.pid)

        # Verify
        if name == "server":
            ok = False
            for _ in range(10):
                time.sleep(0.5)
                if _check_server_http():
                    ok = True
                    break
            if not ok:
                # Check process still alive at least
                if _read_pid(name):
                    print(f"{YELLOW}{info['label']} started (pid {proc.pid}) but health check pending{RESET}")
                    print(f"  logs: {log_file.relative_to(ROOT)}")
                    continue
                else:
                    print(f"{RED}{info['label']} failed to start — check {log_file.relative_to(ROOT)}{RESET}")
                    continue
        else:
            time.sleep(1)
            if not _read_pid(name):
                print(f"{RED}{info['label']} failed to start — check {log_file.relative_to(ROOT)}{RESET}")
                continue

        print(f"{GREEN}{info['label']} started{RESET} (pid {proc.pid}) — {log_file.relative_to(ROOT)}")


def cmd_stop(target: str | None):
    """Stop engine, server, or both."""
    if target and target not in COMPONENTS:
        print(f"{RED}Unknown component: {target}{RESET}")
        print(f"Valid: {', '.join(COMPONENTS)}")
        sys.exit(1)

    # Stop engine first (it has child processes), then server
    names = [target] if target else ["engine", "server"]

    for name in names:
        pid = _read_pid(name)
        if not pid:
            print(f"{COMPONENTS[name]['label']} is not running")
            continue

        info = COMPONENTS[name]
        print(f"Stopping {info['label']} (pid {pid})...", end=" ", flush=True)

        # Send SIGTERM to the process group for engine (kills market_poller child)
        try:
            if name == "engine":
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            print(f"{GREEN}done{RESET} (already exited)")
            _pid_path(name).unlink(missing_ok=True)
            continue

        # Wait up to 5s for graceful exit
        for _ in range(50):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            # Still alive — force kill
            try:
                if name == "engine":
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
                print(f"{YELLOW}killed{RESET}")
            except ProcessLookupError:
                pass

        print(f"{GREEN}done{RESET}")
        _pid_path(name).unlink(missing_ok=True)


def cmd_logs(target: str | None):
    """Tail logs for a component."""
    if not target:
        print(f"{RED}Usage: tc logs <server|engine>{RESET}")
        sys.exit(1)
    if target not in COMPONENTS:
        print(f"{RED}Unknown component: {target}{RESET}")
        sys.exit(1)

    log = _latest_log(target)
    if not log:
        print(f"No logs yet — run: python tc.py start {target}")
        sys.exit(1)

    print(f"Tailing {log.relative_to(ROOT)}  (Ctrl-C to stop)\n")
    try:
        subprocess.run(["tail", "-f", str(log)])
    except KeyboardInterrupt:
        pass


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        sys.exit(0)

    cmd = args[0]
    arg = args[1] if len(args) > 1 else None

    if cmd == "doctor":
        cmd_doctor()
    elif cmd == "start":
        cmd_start(arg)
    elif cmd == "stop":
        cmd_stop(arg)
    elif cmd == "logs":
        cmd_logs(arg)
    else:
        print(f"{RED}Unknown command: {cmd}{RESET}")
        print(__doc__.strip())
        sys.exit(1)


if __name__ == "__main__":
    main()
