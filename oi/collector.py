"""
OI Data Collector - Interfaces with MCP services for open interest and market data
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from config.settings import MCP_OI_EXECUTABLE, MCP_MARKET_DATA_EXECUTABLE, OI_TICKERS, OI_ANALYSIS_DAYS

logger = logging.getLogger(__name__)


class MCPOIClient:
    """MCP stdio JSON-RPC client for OI and market data services"""

    def __init__(self, cmd: str = MCP_OI_EXECUTABLE, args: Optional[List[str]] = None):
        self.cmd = cmd
        self.args = args or []
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.req_id = 0
        self.initialized = False

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            self.cmd, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=100 * 1024 * 1024,
        )
        await self._initialize()

    async def stop(self) -> None:
        if self.proc:
            self.proc.terminate()
            await self.proc.wait()

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._assert_ready()
        resp = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        content = resp.get("result", {}).get("content", [])
        if isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
            try:
                return json.loads(content[0]["text"])
            except json.JSONDecodeError:
                pass
        return resp

    async def _initialize(self) -> None:
        await self._rpc("initialize", {
            "protocolVersion": "0.1.0",
            "capabilities": {},
            "clientInfo": {"name": "oi-tracker", "version": "1.0.0"},
        })
        await self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        self.initialized = True

    async def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.req_id += 1
        await self._send({"jsonrpc": "2.0", "id": self.req_id, "method": method, "params": params})
        line = await self._readline()
        return json.loads(line)

    async def _send(self, obj: Dict[str, Any]) -> None:
        self._assert_ready(proc_ok=False)
        data = (json.dumps(obj) + "\n").encode("utf-8")
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()

    async def _readline(self) -> str:
        self._assert_ready()
        line = await self.proc.stdout.readline()
        if not line:
            err = (await self.proc.stderr.read()).decode(errors="replace") if self.proc.stderr else ""
            raise RuntimeError(f"MCP server closed pipe.\n{err}")
        return line.decode("utf-8").strip()

    def _assert_ready(self, proc_ok: bool = True) -> None:
        if not self.proc:
            raise RuntimeError("MCP server not started")
        if proc_ok and self.proc.returncode is not None:
            raise RuntimeError(f"MCP server exited with code {self.proc.returncode}")


class OIDataCollector:
    """Collects OI + market data for all tickers across multiple DTEs"""

    def __init__(self):
        self.tickers = OI_TICKERS
        self.analysis_days = OI_ANALYSIS_DAYS

    async def collect_ticker_data(self, ticker, target_dte):
        """Collect OI + market data for a single ticker/DTE"""
        t0 = time.time()
        try:
            oi_task = self._call_oi_mcp(ticker, target_dte)
            market_task = self._call_market_data_mcp(ticker)

            oi_result, market_result = await asyncio.gather(
                oi_task, market_task, return_exceptions=True
            )

            data = {"ticker": ticker, "dte_period": target_dte}

            if isinstance(oi_result, Exception):
                data["oi_data"] = None
                data["oi_error"] = str(oi_result)
                logger.warning(f"{ticker} {target_dte}DTE OI: FAILED - {oi_result}")
            else:
                data["oi_data"] = oi_result
                _log_oi_summary(ticker, target_dte, oi_result)

            if isinstance(market_result, Exception):
                data["market_data"] = None
                data["market_error"] = str(market_result)
                logger.warning(f"{ticker} market data: FAILED - {market_result}")
            else:
                data["market_data"] = market_result
                _log_market_summary(ticker, market_result)

            dur = time.time() - t0
            logger.debug(f"{ticker} {target_dte}DTE: done ({dur:.1f}s)")
            return {"status": "success", "data": data, "timestamp": datetime.now().isoformat()}

        except Exception as e:
            dur = time.time() - t0
            logger.error(f"{ticker} {target_dte}DTE: ERROR ({dur:.1f}s) - {e}")
            return {"status": "error", "ticker": ticker, "error": str(e), "timestamp": datetime.now().isoformat()}

    async def collect_all_tickers(self, progress_callback=None):
        """
        Collect all tickers across all DTEs.
        Returns data grouped by ticker: {ticker: {dte: {oi_data, market_data}}}
        """
        ticker_data = {}
        total = len(self.tickers) * len(self.analysis_days)
        completed = 0

        for dte_period in self.analysis_days:
            for ticker in self.tickers:
                if progress_callback:
                    progress_callback(f"Collecting {ticker} ({dte_period} DTE)", completed, total)

                result = await self.collect_ticker_data(ticker, dte_period)
                completed += 1

                if ticker not in ticker_data:
                    ticker_data[ticker] = {}

                if result["status"] == "success":
                    d = result["data"]
                    ticker_data[ticker][str(dte_period)] = {
                        "oi_data": d.get("oi_data"),
                        "market_data": d.get("market_data"),
                        "dte_period": dte_period,
                        "timestamp": result["timestamp"]
                    }
                else:
                    ticker_data[ticker][str(dte_period)] = {
                        "oi_data": None,
                        "market_data": None,
                        "error": result.get("error"),
                        "dte_period": dte_period,
                        "timestamp": result["timestamp"]
                    }

        successful = sum(
            1 for t in ticker_data.values()
            for d in t.values()
            if d.get("oi_data") is not None
        )

        return {
            "data": ticker_data,
            "summary": {
                "total_processed": total,
                "successful": successful,
                "failed": total - successful,
            }
        }

    async def _call_oi_mcp(self, ticker, target_dte):
        """Call MCP OI server"""
        client = MCPOIClient()
        try:
            await client.start()
            return await client.call_tool("analyze_open_interest", {
                "ticker": ticker,
                "days": target_dte,
                "target_dte": target_dte,
                "include_news": True
            })
        finally:
            await client.stop()

    async def _call_market_data_mcp(self, ticker):
        """Call MCP market data server"""
        init_msg = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "oi-tracker", "version": "1.0.0"}
            }
        })
        initialized_msg = json.dumps({
            "jsonrpc": "2.0", "method": "notifications/initialized"
        })
        tool_msg = json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {
                "name": "financial_technical_analysis_tool",
                "arguments": {"symbol": ticker}
            }
        })

        process = await asyncio.create_subprocess_exec(
            MCP_MARKET_DATA_EXECUTABLE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )

        input_data = f"{init_msg}\n{initialized_msg}\n{tool_msg}\n"
        stdout_lines = []

        async def read_output():
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                line_text = line.decode().strip()
                stdout_lines.append(line_text)

        read_task = asyncio.create_task(read_output())
        process.stdin.write(input_data.encode())
        process.stdin.close()
        await process.wait()
        await read_task

        responses = [l for l in stdout_lines if l.startswith('{')]
        if len(responses) < 2:
            raise Exception(f"Unexpected MCP response format - only {len(responses)} JSON responses")

        tool_response = json.loads(responses[-1])
        if "error" in tool_response:
            raise Exception(f"MCP Market Data error: {tool_response['error']}")

        result_content = tool_response["result"]["content"][0]["text"]
        return json.loads(result_content)


def _log_oi_summary(ticker, dte, oi_data):
    """Log key OI metrics from raw data"""
    try:
        data_by_date = oi_data.get("data_by_date", {})
        if not data_by_date:
            logger.info(f"{ticker} {dte}DTE OI: empty response")
            return

        latest_date = max(data_by_date.keys())
        d = data_by_date[latest_date]
        summary = d.get("summary_metrics", {})

        total_oi = summary.get("total_open_interest", d.get("total_oi", 0))
        call_oi = summary.get("call_open_interest", d.get("call_oi", 0))
        put_oi = summary.get("put_open_interest", d.get("put_oi", 0))
        pcr = summary.get("put_call_ratio", d.get("put_call_ratio", 0))
        max_pain = summary.get("max_pain", d.get("max_pain", 0))

        # Top strikes by OI
        strikes = d.get("strikes", {})
        top_calls = []
        for strike, oi in sorted(strikes.get("calls", {}).items(), key=lambda x: x[1], reverse=True)[:3]:
            top_calls.append(f"${strike}:{oi:,}")
        top_puts = []
        for strike, oi in sorted(strikes.get("puts", {}).items(), key=lambda x: x[1], reverse=True)[:3]:
            top_puts.append(f"${strike}:{oi:,}")

        logger.info(
            f"{ticker} {dte}DTE OI [{latest_date}]: "
            f"total={total_oi:,} call={call_oi:,} put={put_oi:,} P/C={pcr:.2f} maxpain=${max_pain} | "
            f"top calls: {', '.join(top_calls) or 'none'} | "
            f"top puts: {', '.join(top_puts) or 'none'}"
        )
    except Exception as e:
        logger.debug(f"{ticker} {dte}DTE OI: couldn't parse summary - {e}")


def _log_market_summary(ticker, market_data):
    """Log key technical/market data metrics"""
    try:
        if not market_data or not isinstance(market_data, dict):
            logger.info(f"{ticker} market: empty response")
            return

        price = market_data.get("price") or market_data.get("close") or market_data.get("current_price", "?")
        rsi = market_data.get("rsi", market_data.get("RSI", "?"))
        vwap = market_data.get("vwap", market_data.get("VWAP", "?"))
        macd = market_data.get("macd", market_data.get("MACD", {}))
        trend = market_data.get("trend", market_data.get("overall_trend", "?"))

        macd_signal = ""
        if isinstance(macd, dict):
            macd_val = macd.get("macd", macd.get("value", "?"))
            macd_sig = macd.get("signal", "?")
            macd_signal = f"MACD={macd_val}/{macd_sig}"
        elif macd != "?":
            macd_signal = f"MACD={macd}"

        logger.info(
            f"{ticker} market: price=${price} RSI={rsi} VWAP={vwap} {macd_signal} trend={trend}"
        )
    except Exception as e:
        logger.debug(f"{ticker} market: couldn't parse summary - {e}")
