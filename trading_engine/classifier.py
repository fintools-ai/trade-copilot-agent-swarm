"""
Deterministic market data classifier — converts raw numbers to categorical labels.

Tier 1 (Python, ~1ms): handles clear-cut cases.
Tier 2 (Haiku, ~600ms): handles borderline cases where thresholds are ambiguous.

Data sources:
- Order flow: GET http://localhost:8300/flow/all → per-ticker TickerSnapshot
  Shape: {symbol: {ticker, timestamp, current: {bid, ask, ...}, windows: {5s: {...}, 15s: {...}, 60s: {...}}}}
- SPY technicals: Redis GET market:spy:data → price, RSI, VWAP, EMA, MACD, ORB
- Mag7 quotes: Redis GET market:mag7:data → {symbols: {SYM: {price, change, change_pct}}}
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo


# ── Flow Classification ──────────────────────────────────────────────

@dataclass
class FlowLabel:
    direction: str  # STRONG_BUYING, BUYING, LEAN_BUYING, MIXED, LEAN_SELLING, SELLING, STRONG_SELLING
    ratio: float  # bid_lifts / bid_drops
    net: int  # bid_lifts - bid_drops
    momentum: str  # ACCELERATING, STEADY, FADING (5s vs 60s comparison)
    borderline: bool = False  # True if Python thresholds are ambiguous → needs LLM
    bid_lifts_60: int = 0  # raw values for LLM
    bid_drops_60: int = 0
    ask_lifts_60: int = 0
    ask_drops_60: int = 0
    bid_vol_60: int = 0
    ask_vol_60: int = 0


def _get_window(ticker_data: dict, window: str) -> dict:
    """Extract window data from flow API response.
    Handles both {windows: {60s: {...}}} and flat {60s: {...}} shapes."""
    windows = ticker_data.get("windows", ticker_data)
    return windows.get(window, {})


# Tier 1 thresholds — tighter MIXED zone to catch more directional signals
_STRONG_BUY_RATIO = 2.5   # strong buying is 2.5:1+
_BUY_RATIO = 1.5          # clear buying
_LEAN_BUY_RATIO = 1.10    # lean buying (tightened from 1.15)
_LEAN_SELL_RATIO = 0.90   # lean selling (tightened from 0.85)
_SELL_RATIO = 0.65        # clear selling
_STRONG_SELL_RATIO = 0.4  # strong selling is 0.4:1-
# True MIXED: 0.85-1.15 ratio — genuinely balanced, no edge
# Borderline: only the narrow gaps between tiers → LLM decides


def classify_flow(flow_data: dict, symbol: str = "SPY") -> FlowLabel:
    """
    Classify order flow direction from WindowMetrics.

    7 levels: STRONG_BUYING > BUYING > LEAN_BUYING > MIXED > LEAN_SELLING > SELLING > STRONG_SELLING
    Tier 1 (Python): handles clear cases. Borderline gaps flagged for Haiku.
    """
    ticker_data = flow_data.get(symbol, {})
    if not ticker_data:
        return FlowLabel("NO_DATA", 1.0, 0, "STEADY")

    # Extract 60s window (primary)
    w60 = _get_window(ticker_data, "60s")
    bid_lifts = w60.get("bid_lifts", 0)
    bid_drops = w60.get("bid_drops", 0)
    ask_lifts = w60.get("ask_lifts", 0)
    ask_drops = w60.get("ask_drops", 0)
    bid_vol = w60.get("bid_volume", 0)
    ask_vol = w60.get("ask_volume", 0)

    net = bid_lifts - bid_drops
    ratio = bid_lifts / max(bid_drops, 1)

    # Tier 1: classify by ratio bands
    borderline = False
    if ratio >= _STRONG_BUY_RATIO and net > 30:
        direction = "STRONG_BUYING"
    elif ratio >= _BUY_RATIO:
        direction = "BUYING"
    elif ratio >= _LEAN_BUY_RATIO:
        direction = "LEAN_BUYING"
    elif ratio <= _STRONG_SELL_RATIO and net < -30:
        direction = "STRONG_SELLING"
    elif ratio <= _SELL_RATIO:
        direction = "SELLING"
    elif ratio <= _LEAN_SELL_RATIO:
        direction = "LEAN_SELLING"
    else:
        # True MIXED: ratio 0.90-1.10 (tightened zone)
        direction = "MIXED"

    # Flag borderline cases at tier boundaries for Haiku override
    if direction in ("LEAN_BUYING", "LEAN_SELLING") or \
       (direction == "MIXED" and (ratio < 0.9 or ratio > 1.1)):
        borderline = True

    # Momentum from 5s vs 60s trend
    w5 = _get_window(ticker_data, "5s")
    r5 = w5.get("bid_lifts", 0) / max(w5.get("bid_drops", 1), 1)

    if direction in ("STRONG_BUYING", "BUYING", "LEAN_BUYING"):
        if r5 > ratio * 1.2:
            momentum = "ACCELERATING"
        elif r5 < ratio * 0.6:
            momentum = "FADING"
        else:
            momentum = "STEADY"
    elif direction in ("STRONG_SELLING", "SELLING", "LEAN_SELLING"):
        if r5 < ratio * 0.8:
            momentum = "ACCELERATING"
        elif r5 > ratio * 1.4:
            momentum = "FADING"
        else:
            momentum = "STEADY"
    else:
        momentum = "STEADY"

    return FlowLabel(
        direction=direction,
        ratio=round(ratio, 2),
        net=net,
        momentum=momentum,
        borderline=borderline,
        bid_lifts_60=bid_lifts,
        bid_drops_60=bid_drops,
        ask_lifts_60=ask_lifts,
        ask_drops_60=ask_drops,
        bid_vol_60=bid_vol,
        ask_vol_60=ask_vol,
    )


# ── Technical Classification ─────────────────────────────────────────

@dataclass
class TechLabel:
    rsi_state: str  # OVERBOUGHT, STRONG, NEUTRAL, WEAK, OVERSOLD
    rsi_value: float
    vwap_position: str  # ABOVE_2SD, ABOVE_1SD, AT_VWAP, BELOW_1SD, BELOW_2SD
    price_vs_vwap: float
    ema_cross: str  # BULLISH_CROSS, BEARISH_CROSS
    macd_state: str  # BULLISH, BEARISH, BULL_CROSS, BEAR_CROSS
    macd_histogram: float = 0.0
    orb_status: str = "INSIDE"  # INSIDE, BREAKOUT_HIGH, BREAKDOWN_LOW


def classify_technicals(spy_data: dict) -> TechLabel:
    """Classify SPY technical state from Redis market:spy:data."""
    rsi = spy_data.get("rsi", 50)
    if rsi > 70:
        rsi_state = "OVERBOUGHT"
    elif rsi > 60:
        rsi_state = "STRONG"
    elif rsi >= 40:
        rsi_state = "NEUTRAL"
    elif rsi >= 30:
        rsi_state = "WEAK"
    else:
        rsi_state = "OVERSOLD"

    price = spy_data.get("price", {}).get("current", 0)
    vwap = spy_data.get("vwap", 0)
    price_vs_vwap = round(price - vwap, 2) if price and vwap else 0

    vwap_plus_1 = spy_data.get("vwap_plus_1", 0)
    vwap_plus_2 = spy_data.get("vwap_plus_2", 0)
    vwap_minus_1 = spy_data.get("vwap_minus_1", 0)
    vwap_minus_2 = spy_data.get("vwap_minus_2", 0)

    if vwap_plus_2 and price >= vwap_plus_2:
        vwap_position = "ABOVE_2SD"
    elif vwap_plus_1 and price >= vwap_plus_1:
        vwap_position = "ABOVE_1SD"
    elif vwap_minus_2 and price <= vwap_minus_2:
        vwap_position = "BELOW_2SD"
    elif vwap_minus_1 and price <= vwap_minus_1:
        vwap_position = "BELOW_1SD"
    else:
        vwap_position = "AT_VWAP"

    ema9 = spy_data.get("ema_9", 0)
    ema21 = spy_data.get("ema_21", 0)
    ema_cross = "BULLISH_CROSS" if ema9 > ema21 else "BEARISH_CROSS"

    macd_data = spy_data.get("macd", {})
    macd_hist = macd_data.get("histogram", 0)
    macd_val = macd_data.get("macd", 0)
    macd_sig = macd_data.get("signal", 0)

    if macd_val > macd_sig and macd_hist > 0:
        macd_state = "BULL_CROSS" if abs(macd_val - macd_sig) < 0.05 else "BULLISH"
    elif macd_val < macd_sig and macd_hist < 0:
        macd_state = "BEAR_CROSS" if abs(macd_val - macd_sig) < 0.05 else "BEARISH"
    else:
        macd_state = "BULLISH" if macd_hist > 0 else "BEARISH"

    orb = spy_data.get("orb", {})
    orb_high = orb.get("high", 0)
    orb_low = orb.get("low", 0)

    if orb_high and price > orb_high:
        orb_status = "BREAKOUT_HIGH"
    elif orb_low and price < orb_low:
        orb_status = "BREAKDOWN_LOW"
    else:
        orb_status = "INSIDE"

    return TechLabel(
        rsi_state=rsi_state,
        rsi_value=round(rsi, 1),
        vwap_position=vwap_position,
        price_vs_vwap=price_vs_vwap,
        ema_cross=ema_cross,
        macd_state=macd_state,
        macd_histogram=round(macd_hist, 3),
        orb_status=orb_status,
    )


# ── ORB Regime Classification ────────────────────────────────────────

@dataclass
class OrbRegime:
    regime: str  # TREND_CONTINUATION, MEAN_REVERSION, UNKNOWN
    range_dollars: float  # ORB range in dollars
    range_pct: float  # ORB range as % of price
    direction: str  # UP, DOWN, FLAT — first-hour bias
    confidence: int  # from research: 62-85%


def classify_orb_regime(spy_data: dict) -> OrbRegime:
    """
    Classify ORB range size into trading regime.

    Research (Option Alpha, SPY 2022):
    - Small range (<0.06%):   85% trend continues
    - Medium range (0.06-0.18%): 62-67% mean reverts
    - Large range (>0.18%):   76% trend continues

    Thresholds are %-based to be robust across SPY price levels.
    """
    # Read ORB data (stored as flat keys in Redis)
    orb_high = spy_data.get("orb_high", 0)
    orb_low = spy_data.get("orb_low", 0)
    orb_range = spy_data.get("orb_range", 0)
    price = spy_data.get("price", {}).get("current", 0)

    if not orb_high or not orb_low or not price:
        return OrbRegime("UNKNOWN", 0, 0, "FLAT", 0)

    if not orb_range:
        orb_range = orb_high - orb_low

    range_pct = (orb_range / price) * 100

    # First-hour direction: where did price go relative to ORB midpoint?
    orb_mid = (orb_high + orb_low) / 2
    open_price = spy_data.get("price", {}).get("open", orb_mid)
    if price > orb_mid + orb_range * 0.1:
        direction = "UP"
    elif price < orb_mid - orb_range * 0.1:
        direction = "DOWN"
    else:
        direction = "FLAT"

    # Regime classification
    if range_pct < 0.06:
        regime = "TREND_CONTINUATION"
        confidence = 85
    elif range_pct > 0.18:
        regime = "TREND_CONTINUATION"
        confidence = 76
    else:
        regime = "MEAN_REVERSION"
        confidence = 65

    return OrbRegime(
        regime=regime,
        range_dollars=round(orb_range, 2),
        range_pct=round(range_pct, 3),
        direction=direction,
        confidence=confidence,
    )


# ── Breadth Classification ───────────────────────────────────────────

@dataclass
class TickerBreadth:
    symbol: str
    price_direction: str  # UP, DOWN, FLAT
    change_pct: float
    flow: FlowLabel
    aligned: bool
    divergent: bool


@dataclass
class BreadthLabel:
    bias: str  # STRONG_BULL, LEAN_BULL, MIXED, LEAN_BEAR, STRONG_BEAR
    aligned_count: int
    divergent_tickers: list[str]
    per_ticker: list[TickerBreadth] = field(default_factory=list)


MAG7_SYMBOLS = ["SPY", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def classify_mag7(mag7_data: dict, flow_data: dict) -> BreadthLabel:
    """Cross-validate Mag7 price direction with flow direction per ticker."""
    symbols_data = mag7_data.get("symbols", {})
    per_ticker = []
    bullish = 0
    bearish = 0
    divergent_tickers = []

    for sym in MAG7_SYMBOLS:
        quote = symbols_data.get(sym, {})
        pct = quote.get("change_pct", 0)

        if pct > 0.15:
            price_dir = "UP"
        elif pct < -0.15:
            price_dir = "DOWN"
        else:
            price_dir = "FLAT"

        flow_label = classify_flow(flow_data, sym)
        flow_dir = "BUYING" if flow_label.direction in ("STRONG_BUYING", "BUYING", "LEAN_BUYING") else \
                   "SELLING" if flow_label.direction in ("STRONG_SELLING", "SELLING", "LEAN_SELLING") else "MIXED"

        aligned = (price_dir == "UP" and flow_dir == "BUYING") or \
                  (price_dir == "DOWN" and flow_dir == "SELLING") or \
                  price_dir == "FLAT"

        divergent = (price_dir == "UP" and flow_dir == "SELLING") or \
                    (price_dir == "DOWN" and flow_dir == "BUYING")

        if divergent:
            divergent_tickers.append(sym)

        if price_dir == "UP" and flow_dir != "SELLING":
            bullish += 1
        elif price_dir == "DOWN" and flow_dir != "BUYING":
            bearish += 1

        per_ticker.append(TickerBreadth(
            symbol=sym,
            price_direction=price_dir,
            change_pct=round(pct, 2),
            flow=flow_label,
            aligned=aligned,
            divergent=divergent,
        ))

    if bullish >= 6:
        bias = "STRONG_BULL"
    elif bullish >= 4:
        bias = "LEAN_BULL"
    elif bearish >= 6:
        bias = "STRONG_BEAR"
    elif bearish >= 4:
        bias = "LEAN_BEAR"
    else:
        bias = "MIXED"

    aligned_count = sum(1 for t in per_ticker if t.aligned)

    return BreadthLabel(
        bias=bias,
        aligned_count=aligned_count,
        divergent_tickers=divergent_tickers,
        per_ticker=per_ticker,
    )


# ── Combined Market State ────────────────────────────────────────────

@dataclass
class MarketState:
    flow: FlowLabel
    tech: TechLabel
    breadth: BreadthLabel
    orb_regime: OrbRegime
    timestamp: str
    price: float
    session: str  # morning, midday, afternoon

    def labels_hash(self) -> str:
        """Hash of categorical labels only — used to detect changes for recall caching."""
        key = (
            f"{self.flow.direction}|{self.flow.momentum}|"
            f"{self.tech.rsi_state}|{self.tech.vwap_position}|{self.tech.ema_cross}|"
            f"{self.tech.macd_state}|{self.tech.orb_status}|"
            f"{self.breadth.bias}|{self.orb_regime.regime}|{self.session}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def to_prompt_text(self) -> str:
        """Labels + raw numbers so Claude can sanity-check classifications."""
        f = self.flow
        t = self.tech
        orb = self.orb_regime

        lines = [
            f"SPY ${self.price:.2f} | {self.session}",
            f"Regime: {orb.regime} (ORB range ${orb.range_dollars} = {orb.range_pct:.3f}% | "
            f"first-hour {orb.direction} | {orb.confidence}% historical)",
            f"Flow: {f.direction} {f.ratio}:1 net={f.net} {f.momentum}"
            f"  (bid_lifts={f.bid_lifts_60} drops={f.bid_drops_60} ask_lifts={f.ask_lifts_60} drops={f.ask_drops_60}"
            f" bid_vol={f.bid_vol_60} ask_vol={f.ask_vol_60})",
            f"RSI: {t.rsi_state} ({t.rsi_value}) | VWAP: {t.vwap_position} ({t.price_vs_vwap:+.2f})",
            f"EMA: {t.ema_cross} | MACD: {t.macd_state} (hist={t.macd_histogram}) | ORB: {t.orb_status}",
            f"Breadth: {self.breadth.bias} ({self.breadth.aligned_count}/7 aligned)",
        ]

        for tb in self.breadth.per_ticker:
            fl = tb.flow
            align = "OK" if tb.aligned else ("DIVERGENT" if tb.divergent else "~")
            lines.append(
                f"  {tb.symbol:5s} {tb.price_direction:4s} {tb.change_pct:+.2f}% | "
                f"Flow {fl.direction} {fl.ratio}:1 "
                f"(lifts={fl.bid_lifts_60} drops={fl.bid_drops_60}) [{align}]"
            )

        return "\n".join(lines)

    def to_search_text(self) -> str:
        """Compact label-only text for memory search queries."""
        div = f" Divergent: {', '.join(self.breadth.divergent_tickers)}" if self.breadth.divergent_tickers else ""
        return (
            f"Flow {self.flow.direction} {self.flow.momentum} "
            f"RSI {self.tech.rsi_state} VWAP {self.tech.vwap_position} "
            f"ORB {self.tech.orb_status} Regime {self.orb_regime.regime} "
            f"Breadth {self.breadth.bias} {self.session}{div}"
        )


def _get_session() -> str:
    """Current trading session based on PT time."""
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    if now.hour < 10:
        return "morning"
    elif now.hour < 12:
        return "midday"
    else:
        return "afternoon"


def classify_all(flow_data: dict, spy_data: dict, mag7_data: dict) -> MarketState:
    """Run all classifiers, return a MarketState dataclass."""
    flow = classify_flow(flow_data, "SPY")
    tech = classify_technicals(spy_data)
    breadth = classify_mag7(mag7_data, flow_data)
    orb_regime = classify_orb_regime(spy_data)
    price = spy_data.get("price", {}).get("current", 0)

    return MarketState(
        flow=flow,
        tech=tech,
        breadth=breadth,
        orb_regime=orb_regime,
        timestamp=datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S"),
        price=price,
        session=_get_session(),
    )
