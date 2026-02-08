"""
OI Redis Manager - Handles all Redis operations for OI data storage and retrieval
Uses oi: namespace prefix, reuses existing Redis connection from redis_stream
"""

import json
from datetime import datetime, timedelta
from redis_stream import get_stream


EXPIRY_SECONDS = 604800  # 7 days


def _redis():
    """Get shared Redis client"""
    return get_stream().redis


def store_oi_data(ticker, dte_key, date, oi_data):
    """Store OI data: oi:{ticker}:{dte_key}:{date}"""
    key = f"oi:{ticker}:{dte_key}:{date}"
    _redis().setex(key, EXPIRY_SECONDS, json.dumps(oi_data))


def get_oi_data(ticker, dte_key, date):
    """Retrieve OI data for specific ticker/dte/date"""
    key = f"oi:{ticker}:{dte_key}:{date}"
    data = _redis().get(key)
    return json.loads(data) if data else None


def get_previous_oi_data(ticker, dte_key, days_back=1):
    """Get OI data from N days ago"""
    target_date = datetime.now() - timedelta(days=days_back)
    return get_oi_data(ticker, dte_key, target_date.strftime('%Y-%m-%d'))


def get_oi_history(ticker, dte_key, days=5):
    """Get last N days of OI data for a ticker/DTE — for LLM multi-day analysis"""
    history = []
    for days_back in range(days, 0, -1):  # oldest first
        target_date = datetime.now() - timedelta(days=days_back)
        date_str = target_date.strftime('%Y-%m-%d')
        data = get_oi_data(ticker, dte_key, date_str)
        if data:
            history.append({"date": date_str, "data": data})
    return history


def store_delta_data(ticker, dte_key, date, delta_data):
    """Store delta data: oi:delta:{ticker}:{dte_key}:{date}"""
    key = f"oi:delta:{ticker}:{dte_key}:{date}"
    _redis().setex(key, EXPIRY_SECONDS, json.dumps(delta_data))


def store_analysis_result(ticker, date, analysis_data):
    """Store LLM analysis results: oi:analysis:{ticker}:{date}"""
    key = f"oi:analysis:{ticker}:{date}"
    _redis().setex(key, EXPIRY_SECONDS, json.dumps(analysis_data))


def store_full_results(results):
    """Store complete analysis results for UI consumption"""
    _redis().setex("oi:results", EXPIRY_SECONDS, json.dumps(results))


def get_full_results():
    """Get complete analysis results"""
    data = _redis().get("oi:results")
    return json.loads(data) if data else None


def set_status(status, message="", progress=0):
    """Set analysis status: oi:status"""
    status_data = {
        "status": status,
        "message": message,
        "progress": progress,
        "timestamp": datetime.now().isoformat()
    }
    _redis().setex("oi:status", 3600, json.dumps(status_data))


def get_status():
    """Get current analysis status"""
    data = _redis().get("oi:status")
    return json.loads(data) if data else {"status": "idle", "message": "", "progress": 0}


def publish_progress(message, progress=0, level="info"):
    """Publish progress event to SSE channel"""
    event = {
        "type": "OI_PROGRESS",
        "message": message,
        "progress": progress,
        "level": level,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    _redis().publish("oi:events", json.dumps(event))


def request_cancel():
    """Set cancel flag — checked by analyzer between phases/tickers"""
    _redis().set("oi:cancel", "1", ex=300)


def clear_cancel():
    """Clear cancel flag (called at start of new run)"""
    _redis().delete("oi:cancel")


def is_cancelled():
    """Check if cancellation was requested"""
    return _redis().get("oi:cancel") == "1"
