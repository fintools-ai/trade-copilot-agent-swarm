"""
OI Analyzer - Main orchestrator for OI pattern analysis
Coordinates: Collection → Market Context → Delta → LLM Analysis → Clustering
"""

import asyncio
import json
from datetime import datetime

from oi.collector import OIDataCollector
from oi.redis_manager import (
    store_oi_data, get_previous_oi_data,
    store_delta_data, store_analysis_result, store_full_results,
    set_status, publish_progress
)
from oi.delta_calculator import DeltaCalculator
from oi.market_context import MarketContextProvider
from oi.llm_analyzer import LLMAnalyzer
from oi.clustering import ClusteringEngine
from oi.memory import recall, store_episode


class OIAnalyzer:
    def __init__(self):
        self.collector = OIDataCollector()
        self.delta_calculator = DeltaCalculator()
        self.market_context_provider = MarketContextProvider()
        self.llm_analyzer = LLMAnalyzer()
        self.clustering_engine = ClusteringEngine()

    async def run_analysis(self):
        """Execute complete OI analysis pipeline"""
        start_time = datetime.now()
        set_status("running", "Starting analysis...", 0)
        publish_progress("Starting OI analysis...", 0)

        try:
            # Phase 1: Data Collection
            set_status("running", "Phase 1: Collecting data...", 5)
            publish_progress("Phase 1: Collecting OI + market data...", 5)

            collection_results = await self.collector.collect_all_tickers(
                progress_callback=self._collection_progress
            )
            ticker_data = collection_results["data"]
            summary = collection_results["summary"]
            print(f"Collected: {summary['successful']}/{summary['total_processed']} successful")

            # Phase 2: Market Context
            set_status("running", "Phase 2: Market context...", 30)
            publish_progress("Phase 2: Analyzing VIX market context...", 30)

            market_context = await self.market_context_provider.get_market_context()
            if market_context:
                print(f"Market context: {market_context['regime']} regime, {market_context['fear_level']} fear")

            # Phase 3: Delta Calculation & Storage
            set_status("running", "Phase 3: Calculating deltas...", 40)
            publish_progress("Phase 3: Calculating OI deltas...", 40)

            today = datetime.now().strftime('%Y-%m-%d')
            processed_tickers = {}  # {ticker: {dte: {oi_data, delta, market_data}}}

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

            # Phase 4: LLM Analysis — one call per ticker
            set_status("running", "Phase 4: LLM analysis...", 50)
            publish_progress("Phase 4: Running LLM analysis...", 50)

            analyses = []
            ticker_list = list(processed_tickers.keys())
            total_tickers = len(ticker_list)

            for i, ticker in enumerate(ticker_list):
                progress = 50 + int((i / total_tickers) * 40)
                set_status("running", f"Analyzing {ticker} ({i+1}/{total_tickers})...", progress)
                publish_progress(f"Analyzing {ticker} ({i+1}/{total_tickers})...", progress)

                all_dte_data = processed_tickers[ticker]

                # Check if we have any OI data for this ticker
                has_data = any(d.get("oi_data") for d in all_dte_data.values())
                if not has_data:
                    analyses.append({
                        "ticker": ticker, "status": "error",
                        "error": "No OI data available",
                        "analysis_timestamp": datetime.now().isoformat()
                    })
                    continue

                # Recall historical context from Bedrock Memory
                historical_context = recall(ticker)

                analysis = self.llm_analyzer.analyze_ticker(
                    ticker, all_dte_data, market_context,
                    historical_context if historical_context else None
                )
                analyses.append(analysis)
                store_analysis_result(ticker, today, analysis)

                # Store episode in Bedrock Memory for future recall
                store_episode(ticker, analysis, market_context)

            print(f"LLM analysis complete: {len(analyses)} tickers")

            # Phase 5: Clustering
            set_status("running", "Phase 5: Clustering...", 92)
            publish_progress("Phase 5: Clustering signals...", 92)

            clusters = self.clustering_engine.cluster_analyses(analyses)

            # Build final results
            duration = str(datetime.now() - start_time)
            results = {
                "clusters": clusters,
                "market_context": market_context,
                "analyses": analyses,
                "summary": {
                    "total_tickers": total_tickers,
                    "bullish": clusters["summary"]["bullish"],
                    "bearish": clusters["summary"]["bearish"],
                    "unclear": clusters["summary"]["unclear"],
                    "market_bias": clusters["summary"]["market_bias"],
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                }
            }

            # Store results for UI
            store_full_results(results)
            set_status("complete", f"Analysis complete ({duration})", 100)
            publish_progress(f"Analysis complete! {total_tickers} tickers analyzed in {duration}", 100)

            print(f"OI Analysis complete in {duration}")
            return results

        except Exception as e:
            error_msg = str(e)
            set_status("error", error_msg, 0)
            publish_progress(f"Error: {error_msg}", 0)
            print(f"OI Analysis failed: {error_msg}")
            return {"status": "error", "error": error_msg}

    def _collection_progress(self, message, completed, total):
        """Callback for collection progress"""
        if total > 0:
            pct = 5 + int((completed / total) * 25)  # 5-30% range
            set_status("running", message, pct)
            publish_progress(message, pct)
