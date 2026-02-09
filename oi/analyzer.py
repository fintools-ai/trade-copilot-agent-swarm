"""
OI Analyzer - Main orchestrator for OI pattern analysis
Coordinates: Collection → Market Context → Delta → LLM Analysis → Clustering
"""

import asyncio
import json
import time
import logging
from datetime import datetime

from oi.collector import OIDataCollector
from oi.redis_manager import (
    store_oi_data, get_previous_oi_data,
    store_delta_data, store_analysis_result, store_full_results,
    set_status, publish_progress, clear_cancel, is_cancelled
)
from oi.delta_calculator import DeltaCalculator
from oi.market_context import MarketContextProvider
from oi.llm_analyzer import LLMAnalyzer
from oi.clustering import ClusteringEngine
from oi.memory import recall, store_episode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class AnalysisCancelledError(Exception):
    pass


class OIAnalyzer:
    def __init__(self):
        self.collector = OIDataCollector()
        self.delta_calculator = DeltaCalculator()
        self.market_context_provider = MarketContextProvider()
        self.llm_analyzer = LLMAnalyzer()
        self.clustering_engine = ClusteringEngine()

    def _check_cancelled(self):
        """Raise if cancellation was requested"""
        if is_cancelled():
            raise AnalysisCancelledError("Analysis cancelled by user")

    async def run_analysis(self):
        """Execute complete OI analysis pipeline"""
        start_time = time.time()
        clear_cancel()
        set_status("running", "Starting analysis...", 0)
        publish_progress("Starting OI analysis...", 0)
        logger.info("=== OI Analysis starting ===")

        try:
            # Phase 1: Data Collection
            phase_start = time.time()
            set_status("running", "Phase 1: Collecting data...", 5)
            publish_progress("Phase 1: Collecting OI + market data...", 5)
            logger.info("Phase 1: Collecting data...")

            collection_results = await self.collector.collect_all_tickers(
                progress_callback=self._collection_progress
            )
            ticker_data = collection_results["data"]
            summary = collection_results["summary"]
            phase_dur = time.time() - phase_start
            msg = f"Phase 1 complete: {summary['successful']}/{summary['total_processed']} successful ({phase_dur:.1f}s)"
            logger.info(msg)
            publish_progress(msg, 30, level="success")

            self._check_cancelled()

            # Phase 2: Market Context
            phase_start = time.time()
            set_status("running", "Phase 2: Market context...", 30)
            publish_progress("Phase 2: Analyzing VIX market context...", 30)
            logger.info("Phase 2: Market context...")

            market_context = await self.market_context_provider.get_market_context()
            phase_dur = time.time() - phase_start
            if market_context:
                msg = f"Phase 2 complete: {market_context['regime']} regime, {market_context['fear_level']} fear ({phase_dur:.1f}s)"
                logger.info(msg)
                publish_progress(msg, 40, level="success")
            else:
                msg = f"Phase 2: No VIX context available ({phase_dur:.1f}s)"
                logger.warning(msg)
                publish_progress(msg, 40, level="warning")

            self._check_cancelled()

            # Phase 3: Delta Calculation & Storage
            phase_start = time.time()
            set_status("running", "Phase 3: Calculating deltas...", 40)
            publish_progress("Phase 3: Calculating OI deltas...", 40)
            logger.info("Phase 3: Calculating deltas...")

            today = datetime.now().strftime('%Y-%m-%d')
            processed_tickers = {}

            for ticker, dte_data in ticker_data.items():
                processed_tickers[ticker] = {}

                for dte_str, data in dte_data.items():
                    oi_data = data.get("oi_data")
                    market_data_val = data.get("market_data")
                    dte_key = f"{dte_str}DTE"

                    if oi_data:
                        store_oi_data(ticker, dte_key, today, oi_data)
                        previous = get_previous_oi_data(ticker, dte_key, days_back=1)
                        delta = self.delta_calculator.calculate_deltas(oi_data, previous, f"{ticker}:{dte_key}")
                        store_delta_data(ticker, dte_key, today, delta)
                    else:
                        delta = {
                            "ticker": ticker, "dte_period": int(dte_str),
                            "is_baseline": True, "message": "No OI data available"
                        }

                    processed_tickers[ticker][dte_str] = {
                        "oi_data": oi_data,
                        "delta": delta,
                        "market_data": market_data_val
                    }

            phase_dur = time.time() - phase_start
            msg = f"Phase 3 complete: {len(processed_tickers)} tickers processed ({phase_dur:.1f}s)"
            logger.info(msg)
            publish_progress(msg, 50, level="success")

            self._check_cancelled()

            # Phase 4: LLM Analysis — per-DTE parallel + synthesis
            phase_start = time.time()
            set_status("running", "Phase 4: LLM analysis...", 50)
            publish_progress("Phase 4: Running per-DTE LLM analysis...", 50)
            logger.info("Phase 4: Per-DTE LLM analysis + synthesis...")

            analyses = []
            ticker_list = list(processed_tickers.keys())
            total_tickers = len(ticker_list)

            for i, ticker in enumerate(ticker_list):
                self._check_cancelled()

                progress = 50 + int((i / total_tickers) * 40)
                set_status("running", f"Analyzing {ticker} ({i+1}/{total_tickers})...", progress)
                publish_progress(f"{ticker}: analyzing per-DTE ({i+1}/{total_tickers})...", progress)

                all_dte_data = processed_tickers[ticker]

                has_data = any(d.get("oi_data") for d in all_dte_data.values())
                if not has_data:
                    analyses.append({
                        "ticker": ticker, "status": "error",
                        "error": "No OI data available",
                        "analysis_timestamp": datetime.now().isoformat()
                    })
                    logger.warning(f"{ticker}: No OI data, skipping")
                    publish_progress(f"{ticker}: No OI data, skipping", progress, level="warning")
                    continue

                ticker_start = time.time()

                # Recall historical context from Bedrock Memory (once per ticker)
                historical_context = recall(ticker)

                # Per-DTE parallel analysis + synthesis
                analysis = await self.llm_analyzer.analyze_ticker_full(
                    ticker, all_dte_data, market_context,
                    historical_context if historical_context else None
                )
                analyses.append(analysis)
                store_analysis_result(ticker, today, analysis)

                # Store episode in Bedrock Memory for future recall
                store_episode(ticker, analysis, market_context)

                ticker_dur = time.time() - ticker_start
                n_dtes = len(analysis.get("dte_analyses", []))
                if analysis.get("status") == "error":
                    msg = f"{ticker}: ERROR - {analysis.get('error', 'unknown')} ({ticker_dur:.1f}s)"
                    logger.warning(msg)
                    publish_progress(msg, progress, level="warning")
                else:
                    msg = f"{ticker}: {analysis.get('direction', '?')} {analysis.get('confidence', '?')}% {analysis.get('confluence', '?')} [{n_dtes} DTEs] ({ticker_dur:.1f}s)"
                    logger.info(msg)
                    publish_progress(msg, progress, level="success")

            phase_dur = time.time() - phase_start
            msg = f"Phase 4 complete: {len(analyses)} tickers ({phase_dur:.1f}s)"
            logger.info(msg)
            publish_progress(msg, 92, level="success")

            self._check_cancelled()

            # Phase 5: Clustering
            phase_start = time.time()
            set_status("running", "Phase 5: Clustering...", 92)
            publish_progress("Phase 5: Clustering signals...", 92)
            logger.info("Phase 5: Clustering...")

            clusters = self.clustering_engine.cluster_analyses(analyses)

            phase_dur = time.time() - phase_start
            total_dur = time.time() - start_time
            b = clusters["summary"]["bullish"]
            e = clusters["summary"]["bearish"]
            u = clusters["summary"]["unclear"]

            results = {
                "clusters": clusters,
                "market_context": market_context,
                "analyses": analyses,
                "summary": {
                    "total_tickers": total_tickers,
                    "bullish": b,
                    "bearish": e,
                    "unclear": u,
                    "market_bias": clusters["summary"]["market_bias"],
                    "duration": f"{total_dur:.0f}s",
                    "timestamp": datetime.now().isoformat()
                }
            }

            store_full_results(results)
            final_msg = f"Complete: {total_tickers} tickers in {total_dur:.0f}s ({b} bullish, {e} bearish, {u} unclear)"
            set_status("complete", final_msg, 100)
            publish_progress(final_msg, 100, level="success")
            logger.info(f"=== {final_msg} ===")

            return results

        except AnalysisCancelledError:
            total_dur = time.time() - start_time
            n_done = len(analyses) if 'analyses' in locals() else 0
            msg = f"Analysis cancelled after {total_dur:.0f}s ({n_done} tickers completed)"
            logger.warning(msg)
            set_status("cancelled", msg, 0)
            publish_progress(msg, 0, level="warning")
            return {"status": "cancelled", "message": msg}

        except Exception as e:
            error_msg = str(e)
            set_status("error", error_msg, 0)
            publish_progress(f"Error: {error_msg}", 0, level="error")
            logger.error(f"OI Analysis failed: {error_msg}", exc_info=True)
            return {"status": "error", "error": error_msg}

    def _collection_progress(self, message, completed, total):
        """Callback for collection progress"""
        if total > 0:
            pct = 5 + int((completed / total) * 25)  # 5-30% range
            set_status("running", message, pct)
            publish_progress(message, pct)
