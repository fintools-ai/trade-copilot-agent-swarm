"""
LLM Analyzer - One call per ticker with all DTEs combined
Uses AWS Bedrock Claude to analyze OI patterns across the full term structure
"""

import json
import boto3
from datetime import datetime
from config.settings import AWS_REGION, BEDROCK_MODEL_ID


class LLMAnalyzer:
    def __init__(self):
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
        self.model_id = BEDROCK_MODEL_ID

    def analyze_ticker(self, ticker, all_dte_data, market_context=None, historical_context=None):
        """
        Analyze ONE ticker with ALL DTEs in a single LLM call.

        Args:
            ticker: Stock symbol
            all_dte_data: {dte: {oi_data, delta, market_data}} for all DTEs
            market_context: VIX regime data
            historical_context: list of fact strings from Bedrock Memory
        """
        try:
            prompt = self._build_prompt(ticker, all_dte_data, market_context, historical_context)
            response = self._call_bedrock(prompt)
            analysis = self._parse_response(response)
            analysis["ticker"] = ticker
            analysis["analysis_timestamp"] = datetime.now().isoformat()
            return analysis
        except Exception as e:
            return {
                "ticker": ticker,
                "status": "error",
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }

    def _build_prompt(self, ticker, all_dte_data, market_context, historical_context=None):
        """Build prompt with all DTEs for one ticker"""

        # Build per-DTE data sections
        dte_sections = []
        for dte in sorted(all_dte_data.keys(), key=lambda x: int(x)):
            data = all_dte_data[dte]
            section = f"""### {dte} DTE

#### Open Interest Data:
{json.dumps(data.get('oi_data'), indent=2) if data.get('oi_data') else 'No data'}

#### Delta Changes (vs yesterday):
{json.dumps(data.get('delta'), indent=2) if data.get('delta') else 'No delta'}

#### Technical/Market Data:
{json.dumps(data.get('market_data'), indent=2) if data.get('market_data') else 'No data'}"""

            dte_sections.append(section)

        dte_block = "\n\n".join(dte_sections)

        prompt = f"""You are a professional options trader analyzing institutional open interest patterns for {ticker}.
You have data across MULTIPLE expiration timeframes (DTEs). Analyze the FULL TERM STRUCTURE to identify what institutions are positioning for.

# CRITICAL INSTRUCTIONS
- Analyze ALL DTEs together to see the full picture
- Short-term (30 DTE) = momentum/gamma plays
- Medium-term (50-60 DTE) = swing setups
- Long-term (90 DTE) = institutional hedging/strategic positioning
- When short and long term agree = HIGH conviction (aligned)
- When they disagree = investigate why (divergent)
- If historical context is provided, compare past patterns vs present data to spot SUSTAINED accumulation vs one-day spikes
- Feed your analysis from the RAW DATA — do not make up numbers

# ANTI-BIAS RULES
1. Not every OI pattern is actionable — most are noise
2. Put walls can break, call resistance can fail
3. Large positions may be unwinding, not accumulating
4. Hedging activity != directional prediction
5. If signals conflict across timeframes, lower your confidence

# RAW DATA FOR {ticker}

{dte_block}

## Market Context (VIX):
{json.dumps(market_context, indent=2) if market_context else 'Not available'}

{self._build_memory_section(ticker, historical_context)}

# OUTPUT FORMAT
Return ONLY valid JSON with this exact structure. No text before or after.

CRITICAL FORMATTING RULES:
- ALL price fields must be STOCK PRICES (not option premiums), plain numbers without $ signs
- confidence must be integer 0-100
- direction must be exactly "CALL" or "PUT"
- confluence must be exactly "aligned" or "divergent"
- key_strikes array should have 3-6 most important strikes

{{
  "direction": "CALL or PUT",
  "confidence": 75,
  "thesis": "2-3 sentence institutional thesis explaining what smart money is doing and why",
  "term_structure": {{
    "short_term": {{"bias": "bullish or bearish or neutral", "key_strike": 580, "key_oi": 45000, "note": "brief note"}},
    "long_term": {{"bias": "bullish or bearish or neutral", "key_strike": 600, "key_oi": 120000, "note": "brief note"}}
  }},
  "key_strikes": [
    {{"strike": 580, "type": "call_wall or put_wall or max_pain", "oi": 45000, "change_5d": "+12000 or N/A"}}
  ],
  "trade": {{
    "instrument": "Buy Call or Buy Put or Put Credit Spread",
    "entry": 582.50,
    "stop": 580.00,
    "target": 585.00,
    "expiry_dte": 30,
    "risk_reward": "2.5:1",
    "current_price": 582.30
  }},
  "risks": ["risk1", "risk2"],
  "confluence": "aligned or divergent"
}}"""

        return prompt

    def _build_memory_section(self, ticker, historical_context):
        """Build historical context section from Bedrock Memory facts"""
        if not historical_context:
            return ""

        facts = "\n".join(f"- {fact}" for fact in historical_context)
        return f"""# HISTORICAL CONTEXT (from past analyses)
These are facts extracted from your previous analyses of {ticker}.
Use them to identify TRENDS — is today confirming or contradicting the pattern?
Do NOT blindly repeat past conclusions. Compare past vs present data.

{facts}
"""

    def _call_bedrock(self, prompt):
        """Call AWS Bedrock"""
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body)
        )

        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']

    def _parse_response(self, response_text):
        """Parse and validate LLM JSON response"""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start == -1 or json_end <= json_start:
                raise ValueError("No valid JSON found in response")

            analysis = json.loads(response_text[json_start:json_end])

            # Validate required fields
            required = ["direction", "confidence", "thesis", "trade"]
            for field in required:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")

            # Normalize confidence to int
            analysis["confidence"] = int(str(analysis["confidence"]).replace('%', '').split('.')[0])

            return analysis

        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse LLM response: {str(e)}",
                "raw_response": response_text[:500]
            }
