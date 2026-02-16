"""
Post-exit analysis - Track price and flow for 5 minutes after exit.
Uses LLM to analyze patterns and saves insights to Bedrock Memory.

HOW IT WORKS:
=============

1. EXIT SIGNAL → start_tracking() called with exit_signal + entry_data

2. CAPTURES EXIT SNAPSHOT:
   - Action (CALL/PUT)
   - Exit price ($685.50)
   - Entry price ($684.00)
   - Exit flow (1.25:1, lifts=125, drops=100)

3. SPAWNS ASYNC BACKGROUND TASK (doesn't block trading engine)

4. 5-MINUTE TRACKING LOOP (20 samples × 15 seconds):
   Every 15 seconds, captures:
   - Time offset: +15s, +30s, +45s... +300s
   - SPY price from Redis (market:spy:data)
   - Flow metrics from order flow service (localhost:8300):
     * bid_lifts, bid_drops, ratio
     * bid_volume, ask_volume
   
   Example samples:
   +15s: $685.75 | Flow 1.30:1 (lifts=130, drops=100)
   +30s: $686.00 | Flow 1.28:1 (lifts=128, drops=100)
   ...
   +300s: $687.25 | Flow 0.95:1 (lifts=95, drops=100)

5. LLM ANALYSIS (Haiku):
   After 5 minutes, sends all samples to LLM to determine:
   - EARLY_EXIT: Price kept moving favorably, flow stayed directional → Should have held longer
   - GOOD_EXIT: Price reversed against position → Exit was correct
   - NEUTRAL: Price stayed flat → Exit timing was fine
   
   LLM also provides:
   - Insight: 1-2 sentence analysis
   - Flow pattern: How flow behaved
   - Lesson: What to do differently next time

6. SAVES TO BEDROCK MEMORY:
   POST_EXIT_ANALYSIS: EARLY_EXIT
   CALL exited @ $685.50 (entered @ $684.00)
   Insight: Price moved $1.75 favorable after exit with sustained flow
   Flow pattern: Ratio stayed above 1.2 for 4 minutes before weakening
   Lesson: Hold CALL positions longer when flow ratio >1.2 and price momentum strong

7. PRINTS TO CONSOLE for immediate feedback

This helps the system learn optimal exit timing by analyzing what happened AFTER each exit.

POST-WAIT ANALYSIS:
===================
Also tracks what happens after WAIT signals to detect missed opportunities.
If price moves significantly in the direction flow suggested, stores as MISSED_OPPORTUNITY.
"""

import json
import time
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# Dedicated logger for post-analysis with separate file handler
logger = logging.getLogger(__name__)
post_analysis_logger = logging.getLogger("post_analysis")
post_analysis_logger.setLevel(logging.INFO)

# Create logs directory if needed
Path("logs/post_analysis").mkdir(parents=True, exist_ok=True)

# File handler for post-analysis logs
fh = logging.FileHandler("logs/post_analysis/analysis.log")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
post_analysis_logger.addHandler(fh)

# Console handler for post-analysis logs
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(asctime)s [POST-ANALYSIS] %(message)s'))
post_analysis_logger.addHandler(ch)

POST_EXIT_PROMPT = """<role>You are an expert 0DTE options trader analyzing post-exit performance.</role>

<context>
You just exited a position. Now you're reviewing what happened in the 5 minutes AFTER the exit to determine if the exit timing was optimal.
</context>

<exit_snapshot>
Position: {action}
Entry Price: ${entry_price}
Exit Price: ${exit_price}
P&L: ${pnl:.2f} ({pnl_pct:.1f}%)
Exit Flow: {exit_flow_ratio}:1 (bid_lifts={exit_lifts}, bid_drops={exit_drops})
Bid Volume: {exit_bid_vol:,} | Ask Volume: {exit_ask_vol:,}
</exit_snapshot>

<5min_post_exit_data>
{samples_text}
</5min_post_exit_data>

<analysis_framework>
As a professional trader, analyze:

1. PRICE ACTION POST-EXIT
   - Did price continue moving favorably? (suggests early exit)
   - Did price reverse against the position? (exit was correct)
   - Did price consolidate/chop? (exit timing was neutral)

2. FLOW BEHAVIOR POST-EXIT
   - Did flow stay directional in the same direction?
   - Did flow weaken or reverse?
   - Did volume ratios confirm the move or show exhaustion?

3. EXIT TIMING ASSESSMENT
   - EARLY_EXIT: Price moved $0.50+ favorable AND flow stayed directional
   - GOOD_EXIT: Price reversed $0.30+ against position OR flow reversed
   - NEUTRAL: Price stayed within $0.30 range OR mixed signals

4. ACTIONABLE LESSON
   - What specific flow/price threshold should trigger holding longer?
   - What specific flow/price pattern confirmed this exit was correct?
   - What would you do differently next time in similar conditions?
</analysis_framework>

<output_format>
Return ONLY valid JSON:
{{
  "verdict": "EARLY_EXIT|GOOD_EXIT|NEUTRAL",
  "insight": "1-2 sentence analysis of what happened after exit",
  "flow_pattern": "Describe flow behavior: stayed directional/weakened/reversed",
  "price_movement": "Describe price action: continued favorable/reversed/consolidated",
  "lesson": "Specific actionable rule for next time (e.g., 'Hold CALL longer when flow >1.2 and price momentum strong')"
}}
</output_format>"""

POST_WAIT_PROMPT = """<role>You are an expert 0DTE options trader analyzing missed opportunities.</role>

<context>
You said WAIT (no entry) when flow was directional. Now you're reviewing what happened in the 5 minutes AFTER to determine if you missed a trade opportunity.
</context>

<wait_snapshot>
Wait Price: ${wait_price}
Wait Flow: {wait_flow_ratio}:1 (bid_lifts={wait_lifts}, bid_drops={wait_drops})
Flow Signal: {flow_signal}
Bid Volume: {wait_bid_vol:,} | Ask Volume: {wait_ask_vol:,}
Volume Ratio: {wait_vol_ratio:.2f}:1 (bid/ask)
</wait_snapshot>

<5min_post_wait_data>
{samples_text}
</5min_post_wait_data>

<analysis_framework>
As a professional trader, analyze:

1. PRICE MOVEMENT POST-WAIT
   - Did price move $0.50+ in the direction flow suggested? (missed opportunity)
   - Did price stay flat or reverse? (WAIT was correct)
   - What was the max favorable move you could have captured?

2. FLOW CONFIRMATION POST-WAIT
   - Did flow strengthen in the same direction?
   - Did flow stay consistent or weaken?
   - Did volume ratios confirm directional bias?

3. OPPORTUNITY ASSESSMENT
   - MISSED_OPPORTUNITY: Price moved $0.50+ favorable AND flow stayed/strengthened
   - GOOD_WAIT: Price stayed flat (<$0.30 move) OR flow reversed
   - NEUTRAL: Mixed signals, unclear outcome

4. ACTIONABLE LESSON
   - What flow/volume threshold should have triggered entry?
   - What additional confirmation was needed before entering?
   - What specific conditions warrant being more aggressive vs cautious?
</analysis_framework>

<output_format>
Return ONLY valid JSON:
{{
  "verdict": "MISSED_OPPORTUNITY|GOOD_WAIT|NEUTRAL",
  "insight": "1-2 sentence analysis of what happened after WAIT",
  "flow_pattern": "Describe flow behavior: strengthened/stayed consistent/weakened/reversed",
  "price_movement": "Describe price action: moved favorable/stayed flat/reversed",
  "max_favorable_move": "Maximum $ move in flow direction (e.g., '$0.75 up')",
  "lesson": "Specific entry criteria for next time (e.g., 'Enter CALL when flow >1.2 + volume ratio >1.3')"
}}
</output_format>"""


class PostExitAnalyzer:
    """Track price and flow for 5 minutes after exit/wait, analyze with LLM."""
    
    def __init__(self, redis_client, bedrock_client, memory_store, log_dir: str = "logs/post_exit"):
        self.redis = redis_client
        self.bedrock = bedrock_client
        self.memory = memory_store
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def start_tracking(self, exit_signal: dict, entry_data: dict, market_labels: dict = None):
        """Start 5-minute tracking after exit."""
        exit_id = f"exit_{int(time.time())}"

        tracking = {
            "exit_id": exit_id,
            "action": entry_data.get("action"),
            "exit_price": exit_signal.get("price", 0),
            "exit_time": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
            "entry_price": entry_data.get("entry_price", 0),
            "exit_flow": self._get_current_flow(),
            "market_labels": market_labels or {},
            "samples": [],
            "start_time": time.time(),
            "type": "exit",
        }

        post_analysis_logger.info("[POST] Started post-exit tracking: %s | %s @ $%.2f",
                                  exit_id, tracking['action'], tracking['exit_price'])
        asyncio.create_task(self._track_and_analyze(exit_id, tracking))
    
    def start_wait_tracking(self, wait_signal: dict, market_state: dict, market_labels: dict = None):
        """Start 5-minute tracking after WAIT signal."""
        wait_id = f"wait_{int(time.time())}"

        tracking = {
            "wait_id": wait_id,
            "wait_price": wait_signal.get("price", 0),
            "wait_time": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
            "wait_flow": self._get_current_flow(),
            "flow_signal": market_state.get("flow", "UNKNOWN"),
            "market_labels": market_labels or {},
            "samples": [],
            "start_time": time.time(),
            "type": "wait",
        }

        post_analysis_logger.info("[POST] Started post-wait tracking: %s | Flow %s @ $%.2f",
                                  wait_id, tracking['flow_signal'], tracking['wait_price'])
        asyncio.create_task(self._track_and_analyze(wait_id, tracking))
    
    async def _track_and_analyze(self, tracking_id: str, tracking: dict):
        """Track for 5 minutes, then analyze with LLM and save to memory."""
        try:
            post_analysis_logger.info("[POST] Tracking %s for 5 minutes (20 samples)...", tracking['type'])

            # Sample every 15 seconds for 5 minutes
            for i in range(20):
                await asyncio.sleep(15)

                sample = {
                    "time_offset": int(time.time() - tracking["start_time"]),
                    "price": self._get_current_price(),
                    "flow": self._get_current_flow(),
                }
                tracking["samples"].append(sample)

                # Log every 4th sample (every minute)
                if (i + 1) % 4 == 0:
                    post_analysis_logger.info(
                        "[POST]   +%ds: $%.2f | Flow %.2f:1",
                        sample['time_offset'], sample['price'],
                        sample['flow'].get('ratio', 0),
                    )

            post_analysis_logger.info("[POST] Tracking complete. Analyzing with LLM...")

            # Analyze with LLM
            if tracking["type"] == "exit":
                await self._analyze_exit_with_llm(tracking)
            else:
                await self._analyze_wait_with_llm(tracking)

        except Exception as e:
            post_analysis_logger.error("[POST] %s tracking failed: %s", tracking['type'], e)
    
    def _get_current_price(self) -> float:
        """Get current SPY price from Redis."""
        try:
            raw = self.redis.get("market:spy:data")
            if raw:
                data = json.loads(raw)
                return data.get("price", {}).get("current", 0)
        except:
            pass
        return 0
    
    def _get_current_flow(self) -> dict:
        """Get current flow metrics from order flow service."""
        import requests
        try:
            resp = requests.get("http://localhost:8300/flow/all", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                spy = data.get("SPY", {}).get("windows", {}).get("60s", {})
                return {
                    "bid_lifts": spy.get("bid_lifts", 0),
                    "bid_drops": spy.get("bid_drops", 0),
                    "ratio": round(spy.get("bid_lifts", 0) / max(spy.get("bid_drops", 1), 1), 2),
                    "bid_vol": spy.get("bid_volume", 0),
                    "ask_vol": spy.get("ask_volume", 0),
                }
        except:
            pass
        return {}
    
    async def _analyze_exit_with_llm(self, tracking: dict):
        """Use LLM to analyze the 5-minute data after exit."""
        samples = tracking["samples"]
        if not samples:
            return
        
        samples_text = self._format_samples(samples)
        exit_flow = tracking["exit_flow"]
        
        # Calculate P&L
        entry_price = tracking["entry_price"]
        exit_price = tracking["exit_price"]
        action = tracking["action"]
        
        if action == "CALL":
            pnl = exit_price - entry_price
        else:  # PUT
            pnl = entry_price - exit_price
        pnl_pct = (pnl / entry_price * 100) if entry_price > 0 else 0
        
        prompt = POST_EXIT_PROMPT.format(
            action=action,
            exit_price=exit_price,
            entry_price=entry_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_flow_ratio=exit_flow.get("ratio", 0),
            exit_lifts=exit_flow.get("bid_lifts", 0),
            exit_drops=exit_flow.get("bid_drops", 0),
            exit_bid_vol=exit_flow.get("bid_vol", 0),
            exit_ask_vol=exit_flow.get("ask_vol", 0),
            samples_text=samples_text,
        )
        
        analysis = await self._invoke_llm(prompt)
        if not analysis:
            return

        verdict = analysis.get('verdict', 'UNKNOWN')
        pnl_str = f"${abs(pnl):.2f} {'profit' if pnl >= 0 else 'loss'}"

        # Build structured header from market_labels
        labels = tracking.get("market_labels", {})
        now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
        header = json.dumps({
            "v": 1,
            "type": "POST_EXIT_ANALYSIS",
            "verdict": verdict,
            "action": tracking["action"],
            "regime": labels.get("regime", "UNKNOWN"),
            "session": labels.get("session", "morning"),
            "flow_dir": labels.get("flow_dir", "MIXED"),
            "pnl": round(pnl, 2),
            "date": now_pt.strftime("%Y-%m-%d"),
        })

        # NLP body (drives semantic search)
        body = (
            f"POST_EXIT_ANALYSIS: {verdict} | {tracking['action']} | {pnl_str}\n"
            f"Exited {tracking['action']} position at ${tracking['exit_price']:.2f} "
            f"(entered at ${tracking['entry_price']:.2f}) with {pnl_str}. "
            f"{analysis.get('insight', '')} "
            f"Flow behavior after exit: {analysis.get('flow_pattern', '')}. "
            f"Lesson: {analysis.get('lesson', '')}"
        )

        memory_text = header + "\n" + body

        self._save_to_memory(memory_text)
        self._log_analysis("POST-EXIT", analysis)
    
    async def _analyze_wait_with_llm(self, tracking: dict):
        """Use LLM to analyze the 5-minute data after WAIT."""
        samples = tracking["samples"]
        if not samples:
            return
        
        samples_text = self._format_samples(samples)
        wait_flow = tracking["wait_flow"]
        
        # Calculate volume ratio
        bid_vol = wait_flow.get("bid_vol", 0)
        ask_vol = wait_flow.get("ask_vol", 0)
        vol_ratio = bid_vol / max(ask_vol, 1) if ask_vol > 0 else 0
        
        prompt = POST_WAIT_PROMPT.format(
            wait_price=tracking["wait_price"],
            wait_flow_ratio=wait_flow.get("ratio", 0),
            wait_lifts=wait_flow.get("bid_lifts", 0),
            wait_drops=wait_flow.get("bid_drops", 0),
            wait_bid_vol=bid_vol,
            wait_ask_vol=ask_vol,
            wait_vol_ratio=vol_ratio,
            flow_signal=tracking["flow_signal"],
            samples_text=samples_text,
        )
        
        analysis = await self._invoke_llm(prompt)
        if not analysis:
            return

        verdict = analysis.get('verdict', 'UNKNOWN')
        flow_desc = tracking['flow_signal'].lower().replace('_', ' ')
        max_move = analysis.get('max_favorable_move', '')

        # Parse max_move as float for header
        max_move_val = 0.0
        if max_move:
            try:
                max_move_val = abs(float(str(max_move).replace('$', '').split()[0]))
            except (ValueError, IndexError, AttributeError):
                pass

        # Build structured header from market_labels
        labels = tracking.get("market_labels", {})
        now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
        header = json.dumps({
            "v": 1,
            "type": "POST_WAIT_ANALYSIS",
            "verdict": verdict,
            "regime": labels.get("regime", "UNKNOWN"),
            "session": labels.get("session", "morning"),
            "flow_dir": labels.get("flow_dir", "MIXED"),
            "flow_ratio": round(labels.get("flow_ratio", 1.0), 2),
            "max_move": round(max_move_val, 2),
            "date": now_pt.strftime("%Y-%m-%d"),
        })

        # NLP body (drives semantic search)
        body = (
            f"POST_WAIT_ANALYSIS: {verdict} | {flow_desc} flow\n"
            f"Said WAIT at ${tracking['wait_price']:.2f} with {flow_desc} flow "
            f"({wait_flow.get('ratio', 0):.2f}:1 ratio). "
            f"{analysis.get('insight', '')} "
            f"Flow behavior after wait: {analysis.get('flow_pattern', '')}. "
            f"Maximum favorable move was {max_move}. "
            f"Lesson: {analysis.get('lesson', '')}"
        )

        memory_text = header + "\n" + body

        self._save_to_memory(memory_text)
        self._log_analysis("POST-WAIT", analysis)
    
    def _format_samples(self, samples: list) -> str:
        """Format samples for LLM prompt."""
        lines = []
        for s in samples:
            flow = s.get("flow", {})
            lines.append(
                f"+{s['time_offset']}s: ${s['price']:.2f} | "
                f"Flow {flow.get('ratio', 0):.2f}:1 (lifts={flow.get('bid_lifts', 0)}, drops={flow.get('bid_drops', 0)})"
            )
        return "\n".join(lines)
    
    async def _invoke_llm(self, prompt: str) -> dict:
        """Call Haiku for analysis."""
        loop = asyncio.get_event_loop()
        
        def _invoke():
            response = self.bedrock.converse(
                modelId="global.anthropic.claude-opus-4-5-20251101-v1:0",
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 300, "temperature": 0},
            )
            text = response["output"]["message"]["content"][0]["text"].strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            return {}
        
        try:
            return await loop.run_in_executor(None, _invoke)
        except Exception as e:
            post_analysis_logger.error("[POST] LLM analysis failed: %s", e)
            return {}
    
    def _save_to_memory(self, memory_text: str):
        """Save analysis directly to Bedrock Memory via batch_create_memory_records."""
        try:
            now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
            record_id = f"post-analysis-{now_pt.strftime('%Y-%m-%d-%H%M%S')}"
            dur = self.memory._write_record("/facts/trader/SPY/", memory_text, record_id)

            dur_str = f"{dur:.1f}s" if dur else "rejected"
            post_analysis_logger.info("[POST] Stored to Bedrock Memory (%s): %s",
                                      dur_str, memory_text.split('\n')[0][:120])

        except Exception as e:
            post_analysis_logger.error("[POST] Failed to save to memory: %s", e)
    
    def _log_analysis(self, title: str, analysis: dict):
        """Log analysis results."""
        post_analysis_logger.info("[POST] %s: %s — %s", title, analysis.get('verdict'),
                                  analysis.get('insight', ''))
        post_analysis_logger.info("[POST]   Flow: %s", analysis.get('flow_pattern', ''))
        post_analysis_logger.info("[POST]   Lesson: %s", analysis.get('lesson', ''))

