"""
Configuration settings for Trade Copilot Agent Swarm
"""

import os

ORDER_FLOW_SERVER_URL = os.getenv('ORDER_FLOW_SERVER_URL', 'http://localhost:8000/api')
MARKET_STRUCTURE_SERVER_URL = os.getenv('MARKET_STRUCTURE_SERVER_URL', 'http://localhost:8001/api')
OPTIONS_FLOW_SERVER_URL = os.getenv('OPTIONS_FLOW_SERVER_URL', 'http://localhost:8002/api')
GREEKS_SERVER_URL = os.getenv('GREEKS_SERVER_URL', 'http://localhost:8004')
GREEKS_SERVER_V2_URL = os.getenv('GREEKS_SERVER_V2_URL', 'http://localhost:8005')

# Request timeouts (in seconds)
DEFAULT_TIMEOUT = int(os.getenv('DEFAULT_TIMEOUT', '10'))
GREEKS_TIMEOUT = int(os.getenv('GREEKS_TIMEOUT', '15'))


MCP_OI_EXECUTABLE = "../mcp-openinterest-server"
MCP_MARKET_DATA_EXECUTABLE = "../mcp-market-data-server"
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
