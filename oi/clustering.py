"""
Clustering Engine - Groups ticker analyses into bullish/bearish/unclear clusters
Adapted for simplified LLM output schema
"""

from datetime import datetime


def safe_int(value):
    """Safely convert any value to integer"""
    try:
        return int(str(value).replace('%', '').replace('"', '').replace("'", '').split('.')[0].split()[0] or 0)
    except (ValueError, TypeError):
        return 0


class ClusteringEngine:
    def cluster_analyses(self, all_analyses):
        """Group all ticker analyses into bullish/bearish/unclear clusters"""
        clusters = {
            "bullish_group": {"tickers": [], "avg_confidence": 0, "total_count": 0, "pattern_types": {}},
            "bearish_group": {"tickers": [], "avg_confidence": 0, "total_count": 0, "pattern_types": {}},
            "unclear_group": {"tickers": [], "avg_confidence": 0, "total_count": 0, "pattern_types": {}},
        }

        for analysis in all_analyses:
            classification = self._classify(analysis)
            info = self._extract_info(analysis, classification)

            if classification == "bullish":
                clusters["bullish_group"]["tickers"].append(info)
            elif classification == "bearish":
                clusters["bearish_group"]["tickers"].append(info)
            else:
                clusters["unclear_group"]["tickers"].append(info)

        # Calculate stats
        for group_key in ["bullish_group", "bearish_group", "unclear_group"]:
            tickers = clusters[group_key]["tickers"]
            clusters[group_key]["total_count"] = len(tickers)
            if tickers:
                clusters[group_key]["avg_confidence"] = sum(
                    safe_int(t.get("confidence", 0)) for t in tickers
                ) / len(tickers)

        # Summary
        b = clusters["bullish_group"]["total_count"]
        e = clusters["bearish_group"]["total_count"]
        u = clusters["unclear_group"]["total_count"]

        clusters["total_analyzed"] = len(all_analyses)
        clusters["timestamp"] = datetime.now().isoformat()
        clusters["summary"] = {
            "bullish": b, "bearish": e, "unclear": u,
            "market_bias": self._market_bias(b, e),
        }

        # High conviction picks
        all_directional = clusters["bullish_group"]["tickers"] + clusters["bearish_group"]["tickers"]
        clusters["high_conviction"] = sorted(
            all_directional,
            key=lambda x: safe_int(x.get("confidence", 0)),
            reverse=True
        )[:5]

        return clusters

    def _classify(self, analysis):
        """Classify as bullish/bearish/unclear from simplified schema"""
        if analysis.get("status") == "error":
            return "unclear"

        direction = analysis.get("direction", "").upper()
        if direction == "CALL":
            return "bullish"
        elif direction == "PUT":
            return "bearish"
        return "unclear"

    def _extract_info(self, analysis, classification):
        """Extract key info for UI display"""
        trade = analysis.get("trade", {})
        term = analysis.get("term_structure", {})
        short_term = term.get("short_term", {})
        long_term = term.get("long_term", {})

        risk_mgmt = analysis.get("risk_management", {})

        return {
            "ticker": analysis.get("ticker", "UNKNOWN"),
            "direction": analysis.get("direction", "N/A"),
            "confidence": safe_int(analysis.get("confidence", 0)),
            "success_probability": safe_int(analysis.get("success_probability", 0)),
            "thesis": analysis.get("thesis", ""),
            "confluence": analysis.get("confluence", "unclear"),
            "pattern_type": analysis.get("pattern_type", ""),
            "smart_money_summary": analysis.get("smart_money_summary", ""),
            "term_structure": {
                "short_term": short_term,
                "long_term": long_term
            },
            "key_strikes": analysis.get("key_strikes", []),
            "trade": {
                "instrument": trade.get("instrument", "N/A"),
                "entry": trade.get("entry"),
                "stop": trade.get("stop"),
                "target": trade.get("target"),
                "expiry_dte": trade.get("expiry_dte"),
                "risk_reward": trade.get("risk_reward", "N/A"),
                "current_price": trade.get("current_price"),
                "entry_triggers": trade.get("entry_triggers", []),
                "exit_strategy": trade.get("exit_strategy", ""),
            },
            "risk_management": {
                "primary_risks": risk_mgmt.get("primary_risks", analysis.get("risks", [])),
                "hedge_strategy": risk_mgmt.get("hedge_strategy", ""),
                "volatility_considerations": risk_mgmt.get("volatility_considerations", ""),
                "position_adjustments": risk_mgmt.get("position_adjustments", ""),
            },
            "data_quality": analysis.get("data_quality", {}),
            "dte_analyses": analysis.get("dte_analyses", []),
            "classification": classification,
        }

    def _market_bias(self, bullish, bearish):
        total = bullish + bearish
        if total == 0:
            return "unclear"
        pct = (bullish / total) * 100
        if pct > 65:
            return f"bullish_{pct:.0f}%"
        elif pct < 35:
            return f"bearish_{100-pct:.0f}%"
        return f"mixed_{pct:.0f}%_bullish"
