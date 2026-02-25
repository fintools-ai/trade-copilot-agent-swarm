"""
Market Context Provider - Uses VIX OI data to determine market regime
"""

import json
import time
import logging
from datetime import datetime
from config.settings import OI_ANALYSIS_DAYS

logger = logging.getLogger(__name__)


class MarketContextProvider:
    def __init__(self):
        self.vix_ticker = "VIX"

    async def get_market_context(self):
        """Get market context using VIX open interest data"""
        try:
            t0 = time.time()
            vix_data = await self._get_vix_oi_data()
            if not vix_data:
                logger.warning("VIX: no OI data returned")
                return None

            context = self._analyze_vix_context(vix_data)
            if not context:
                logger.warning("VIX: analysis returned no context")
                return None

            context["timestamp"] = datetime.now().isoformat()
            dur = time.time() - t0
            logger.info(f"VIX context: {context['regime']} regime, {context['fear_level']} fear, P/C {context.get('vix_put_call_ratio', 0):.2f} ({dur:.1f}s)")
            return context

        except Exception as e:
            logger.error(f"Market context failed: {e}")
            return None

    async def _get_vix_oi_data(self):
        """Get VIX OI data via MCP service"""
        from oi.collector import MCPOIClient

        client = MCPOIClient()
        try:
            await client.start()
            return await client.call_tool("analyze_open_interest", {
                "ticker": self.vix_ticker,
                "days": OI_ANALYSIS_DAYS[0],
                "include_news": True
            })
        finally:
            await client.stop()

    def _analyze_vix_context(self, vix_data):
        """Analyze VIX OI data to determine market context"""
        data_by_date = vix_data.get("data_by_date", {})
        if not data_by_date:
            return {
                "source": "VIX_OI_Analysis",
                "regime": "unknown",
                "vix_put_call_ratio": 0,
                "vix_total_oi": 0,
                "vix_call_oi": 0,
                "vix_put_oi": 0,
                "fear_level": "unknown",
                "volatility_expectation": "unknown",
                "market_summary": "No VIX data available"
            }

        latest_date = max(data_by_date.keys())
        vix_oi = data_by_date[latest_date]
        summary = vix_oi.get("summary_metrics", {})

        vix_pcr = summary.get("put_call_ratio", 0)
        vix_call_oi = summary.get("call_open_interest", 0)
        vix_put_oi = summary.get("put_open_interest", 0)

        regime = self._determine_regime(vix_pcr)
        fear_level = self._calculate_fear_level(vix_call_oi, vix_put_oi)

        return {
            "source": "VIX_OI_Analysis",
            "regime": regime,
            "vix_put_call_ratio": vix_pcr,
            "vix_total_oi": vix_call_oi + vix_put_oi,
            "vix_call_oi": vix_call_oi,
            "vix_put_oi": vix_put_oi,
            "fear_level": fear_level,
            "volatility_expectation": "high" if vix_pcr < 0.8 else "low",
            "market_summary": self._generate_summary(regime, fear_level)
        }

    def _determine_regime(self, vix_pcr):
        if vix_pcr < 0.6:
            return "bearish"
        elif vix_pcr > 1.5:
            return "bullish"
        return "sideways"

    def _calculate_fear_level(self, call_oi, put_oi):
        total = call_oi + put_oi
        if total == 0:
            return "unknown"
        call_pct = call_oi / total
        if call_pct > 0.6:
            return "high"
        elif call_pct < 0.3:
            return "low"
        return "moderate"

    def _generate_summary(self, regime, fear_level):
        regime_desc = {
            "bullish": "Low volatility expectation, supportive for directional trades",
            "bearish": "High volatility expectation, defensive positioning recommended",
            "sideways": "Mixed volatility signals, range-bound environment likely"
        }
        fear_desc = {
            "high": "Elevated fear levels in options market",
            "moderate": "Moderate fear levels, normal market conditions",
            "low": "Low fear levels, complacent market conditions"
        }
        return f"{regime_desc.get(regime, 'Unknown')}. {fear_desc.get(fear_level, 'Unknown')}."
