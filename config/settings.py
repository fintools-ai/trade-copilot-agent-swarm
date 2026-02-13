"""
Configuration settings for Trade Copilot Agent Swarm
"""

import os

ORDER_FLOW_SERVER_URL = os.getenv('ORDER_FLOW_SERVER_URL', 'http://localhost:8000/api')
MARKET_STRUCTURE_SERVER_URL = os.getenv('MARKET_STRUCTURE_SERVER_URL', 'http://localhost:8001/api')
OPTIONS_FLOW_SERVER_URL = os.getenv('OPTIONS_FLOW_SERVER_URL', 'http://localhost:8400')
GREEKS_SERVER_URL = os.getenv('GREEKS_SERVER_URL', 'http://localhost:8004')
GREEKS_SERVER_V2_URL = os.getenv('GREEKS_SERVER_V2_URL', 'http://localhost:8005')

# Request timeouts (in seconds)
DEFAULT_TIMEOUT = int(os.getenv('DEFAULT_TIMEOUT', '10'))
GREEKS_TIMEOUT = int(os.getenv('GREEKS_TIMEOUT', '15'))


MCP_OI_EXECUTABLE = "../mcp-openinterest-server"
MCP_MARKET_DATA_EXECUTABLE = "../mcp-market-data-server"

# OI Analysis Configuration
OI_TICKERS = [
    "SPY", "AMD", "ACHR", "QQQ", "HOOD", "TTD", "COIN",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
    "ORCL", "HIMS", "OSCR", "UNH"
]
OI_ANALYSIS_DAYS = [30, 50, 60, 90]
AWS_REGION = "us-east-1"
BEDROCK_MODEL_ID = "us.anthropic.claude-opus-4-6-20250610-v1:0"
OI_BEDROCK_CONCURRENCY = int(os.getenv('OI_BEDROCK_CONCURRENCY', '4'))

# Market Data Poller intervals (seconds)
POLLER_SPY_INTERVAL = int(os.getenv('POLLER_SPY_INTERVAL', '1'))
POLLER_MAG7_INTERVAL = int(os.getenv('POLLER_MAG7_INTERVAL', '5'))
POLLER_MAX_STALENESS = int(os.getenv('POLLER_MAX_STALENESS', '10'))

# Trading Engine v2
ENGINE_MODEL_ID = os.getenv('ENGINE_MODEL_ID', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')
ENGINE_SCAN_INTERVAL = int(os.getenv('ENGINE_SCAN_INTERVAL', '10'))
ENGINE_MONITOR_INTERVAL = int(os.getenv('ENGINE_MONITOR_INTERVAL', '7'))
ENGINE_MEMORY_NAME = os.getenv('ENGINE_MEMORY_NAME', 'zero_dte_v2')
ENGINE_MAX_TOKENS = int(os.getenv('ENGINE_MAX_TOKENS', '512'))
ENGINE_CONVERSATION_WINDOW = int(os.getenv('ENGINE_CONVERSATION_WINDOW', '20'))
CLASSIFIER_MODEL_ID = os.getenv('CLASSIFIER_MODEL_ID', 'us.anthropic.claude-haiku-4-5-20251001-v1:0')
CLASSIFIER_ALWAYS_LLM = os.getenv('CLASSIFIER_ALWAYS_LLM', 'true').lower() == 'true'

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# === Securely Load Twelve Data API Key ===
try:
    with open("/etc/twelve_data_api_key.txt", "r") as f:
        TWELVE_DATA_API_KEY: str = f.read().strip()
except FileNotFoundError:
    TWELVE_DATA_API_KEY: str = ""
    print("Warning: API key file not found at /etc/twelve_data_api_key.txt")
except Exception as e:
    TWELVE_DATA_API_KEY: str = ""
    print(f"Warning: Failed to read API key: {e}")
