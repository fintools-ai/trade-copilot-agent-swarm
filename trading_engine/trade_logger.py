"""
Simple trade logger for training data collection.
Logs all entries, exits, and outcomes to JSONL (one JSON per line) for Bedrock Memory ingestion.
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


class TradeLogger:
    """Log trades to JSONL for Bedrock Memory ingestion."""
    
    def __init__(self, log_dir: str = "logs/trades"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Daily log file (JSONL format)
        today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"trades_{today}.jsonl"
    
    def log_entry(self, signal: dict, market_state, timestamp: str = None):
        """Log trade entry in Bedrock Memory format."""
        if not timestamp:
            timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
        
        entry_record = {
            "timestamp": timestamp,
            "event_type": "ENTRY",
            "action": signal.get('action', ''),
            "signal": signal.get('signal', ''),
            "conviction": signal.get('conviction', ''),
            "entry_price": signal.get('entry', 0),
            "stop": signal.get('stop', 0),
            "target": signal.get('target', 0),
            "flow": {
                "direction": market_state.flow.direction,
                "ratio": round(market_state.flow.ratio, 2),
                "net": market_state.flow.net,
                "momentum": market_state.flow.momentum,
                "bid_lifts": market_state.flow.bid_lifts_60,
                "bid_drops": market_state.flow.bid_drops_60,
                "bid_vol": market_state.flow.bid_vol_60,
                "ask_vol": market_state.flow.ask_vol_60,
                "vol_ratio": round(market_state.flow.bid_vol_60 / max(market_state.flow.ask_vol_60, 1), 2),
            },
            "regime": {
                "type": market_state.orb_regime.regime,
                "orb_range": market_state.orb_regime.range_dollars,
                "orb_pct": round(market_state.orb_regime.range_pct, 3),
            },
            "technicals": {
                "rsi": market_state.tech.rsi_value,
                "rsi_state": market_state.tech.rsi_state,
                "vwap_position": market_state.tech.vwap_position,
                "ema_cross": market_state.tech.ema_cross,
                "macd_state": market_state.tech.macd_state,
            },
            "breadth": {
                "bias": market_state.breadth.bias,
                "aligned_count": market_state.breadth.aligned_count,
                "divergent": market_state.breadth.divergent_tickers,
            },
            "session": market_state.session,
            "labels_hash": market_state.labels_hash(),
            "memory_text": self._format_for_memory(signal, market_state, "ENTRY"),
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry_record) + '\n')
    
    def log_exit(self, signal: dict, entry_data: dict, timestamp: str = None):
        """Log trade exit with outcome in Bedrock Memory format."""
        if not timestamp:
            timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
        
        entry_price = entry_data.get('entry_price', 0)
        
        # Handle price - could be float or dict with 'current' key
        exit_price_raw = signal.get('price', 0)
        if isinstance(exit_price_raw, dict):
            exit_price = exit_price_raw.get('current', 0)
        else:
            exit_price = exit_price_raw
            
        action = entry_data.get('action', '')
        
        # Calculate P&L
        if action == 'CALL':
            pnl = exit_price - entry_price
            result = 'WIN' if pnl > 0 else 'LOSS'
        elif action == 'PUT':
            pnl = entry_price - exit_price
            result = 'WIN' if pnl > 0 else 'LOSS'
        else:
            pnl = 0
            result = 'UNKNOWN'
        
        pnl_pct = (pnl / entry_price * 100) if entry_price > 0 else 0
        flow_metrics = entry_data.get('flow_metrics', {})
        
        exit_record = {
            "timestamp": timestamp,
            "event_type": "EXIT",
            "action": action,
            "result": result,
            "conviction": entry_data.get('conviction', ''),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop": entry_data.get('stop', 0),
            "target": entry_data.get('target', 0),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "entry_flow": flow_metrics,
            "session": entry_data.get('session', ''),
            "labels_hash": entry_data.get('labels_hash', ''),
            "memory_text": self._format_outcome_for_memory(action, result, pnl, pnl_pct, entry_price, exit_price, flow_metrics, entry_data),
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(exit_record) + '\n')
    
    def _format_for_memory(self, signal: dict, market_state, event_type: str) -> str:
        """Format entry for Bedrock Memory ingestion."""
        flow = market_state.flow
        return (
            f"{event_type}: {signal.get('action')} @ ${signal.get('entry')} | {signal.get('conviction')} conviction\n"
            f"Flow: {flow.direction} {flow.ratio:.2f}:1 net={flow.net} {flow.momentum} "
            f"(lifts={flow.bid_lifts_60}, drops={flow.bid_drops_60}, vol_ratio={flow.bid_vol_60/max(flow.ask_vol_60,1):.2f}:1)\n"
            f"Regime: {market_state.orb_regime.regime} | RSI: {market_state.tech.rsi_state} ({market_state.tech.rsi_value}) | "
            f"VWAP: {market_state.tech.vwap_position} | Breadth: {market_state.breadth.bias} ({market_state.breadth.aligned_count}/7)\n"
            f"Session: {market_state.session}"
        )
    
    def _format_outcome_for_memory(self, action: str, result: str, pnl: float, pnl_pct: float, 
                                   entry_price: float, exit_price: float, flow_metrics: dict, entry_data: dict) -> str:
        """Format outcome for Bedrock Memory ingestion."""
        pnl_sign = "+" if pnl > 0 else ""
        return (
            f"OUTCOME: {result} {pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_pct:.2f}%) | {action} | {entry_data.get('conviction')} conviction\n"
            f"Flow at entry: lifts={flow_metrics.get('bid_lifts', '?')}, drops={flow_metrics.get('bid_drops', '?')}, "
            f"ratio={flow_metrics.get('ratio', '?')}:1, vol_ratio={flow_metrics.get('vol_ratio', '?')}:1\n"
            f"Entry: ${entry_price} → Exit: ${exit_price} | Session: {entry_data.get('session', '')}"
        )
    
    def get_today_summary(self) -> dict:
        """Get summary of today's trades."""
        if not self.log_file.exists():
            return {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0}
        
        trades = 0
        wins = 0
        losses = 0
        total_pnl = 0
        
        with open(self.log_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                if record.get('event_type') == 'EXIT':
                    trades += 1
                    if record.get('result') == 'WIN':
                        wins += 1
                    elif record.get('result') == 'LOSS':
                        losses += 1
                    total_pnl += record.get('pnl', 0)
        
        return {
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': round(wins / trades * 100, 1) if trades > 0 else 0,
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(total_pnl / trades, 2) if trades > 0 else 0,
        }
