# main.py — Hyperliquid 1H scanner -> Telegram shortlist -> TradingView (manual trading)
# Upgrades: Momentum phase + momentum candle scoring
# Full code, no shortcuts.

import os
import time
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import ta

from flask import Flask, request
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# =========================
# ENV / CONFIG
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
PORT = int(os.getenv("PORT", "8080"))

API_BASE = "https://api.hyperliquid.xyz"
HEADERS = {"Content-Type": "application/json"}

INTERVAL = os.getenv("CANDLE_INTERVAL", "1h")
LOOKBACK = int(os.getenv("SIGNAL_LOOKBACK", "70"))
TOP_N = int(os.getenv("TOP_N", "120"))
MAX_RESULTS_IN_TELEGRAM = int(os.getenv("TOP_K_TELEGRAM", "10"))

MIN_DAY_NTL_VLM = float(os.getenv("MIN_DAY_NTL_VLM", "1000000"))
MAX_IMPACT_SPREAD_PCT = float(os.getenv("MAX_IMPACT_SPREAD_PCT", "0.80"))
FILTER_CHOP_ENTRIES = os.getenv("FILTER_CHOP_ENTRIES", "1").strip() == "1"

ADX_TREND_THR = float(os.getenv("ADX_TREND_THR", "25"))
ADX_CHOP_THR = float(os.getenv("ADX_CHOP_THR", "20"))

ADX_PRE_MIN = float(os.getenv("ADX_PRE_MIN", "22"))
ADX_PRE_MAX = float(os.getenv("ADX_PRE_MAX", "25"))

MIN_SECONDS_BETWEEN_SCANS = int(os.getenv("MIN_SECONDS_BETWEEN_SCANS", "30"))
META_TTL_SEC = int(os.getenv("META_TTL_SEC", "60"))

TV_PREFIX = os.getenv("TV_PREFIX", "https://www.tradingview.com/chart/?symbol=CRYPTO:")

# =========================
# RATE LIMITER
# =========================
@dataclass
class RateLimiter:
    max_weight_per_min: int = 1100
    window_sec: int = 60

    def __post_init__(self):
        self._lock = threading.Lock()
        self._events = deque()

    def _prune(self, now: float):
        cutoff = now - self.window_sec
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def acquire(self, weight: int):
        while True:
            with self._lock:
                now = time.time()
                self._prune(now)
                used = sum(w for _, w in self._events)
                if used + weight <= self.max_weight_per_min:
                    self._events.append((now, weight))
                    return
                oldest_ts = self._events[0][0] if self._events else now
                sleep_for = max(0.05, (oldest_ts + self.window_sec) - now)
            time.sleep(min(sleep_for, 2.0))

RL = RateLimiter()

def weight_info_default() -> int:
    return 20

def weight_candle_snapshot(approx_items: int) -> int:
    extra = approx_items // 60
    return 20 + extra

# =========================
# TELEGRAM
# =========================
def send_telegram_message(msg: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload, timeout=12)
    except Exception as e:
        print(f"Telegram send error: {e}")

# =========================
# HELPERS
# =========================
def tv_link_for_coin(coin: str) -> str:
    return f"{TV_PREFIX}{coin}USD"

def regime_from_adx(adx: float) -> str:
    if adx > ADX_TREND_THR:
        return "TREND"
    if adx < ADX_CHOP_THR:
        return "CHOP"
    return "NEUTRAL"

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

# =========================
# HL INFO + CACHING
# =========================
_meta_cache: Dict[str, Any] = {"ts": 0.0, "data": None}

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=0.6, min=0.6, max=6),
    retry=retry_if_exception_type((requests.RequestException,)),
)
def hl_info(payload: dict, weight: int) -> Any:
    RL.acquire(weight)
    r = requests.post(f"{API_BASE}/info", headers=HEADERS, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def get_meta_and_ctxs_cached() -> Any:
    now = time.time()
    if _meta_cache["data"] is not None and (now - _meta_cache["ts"]) < META_TTL_SEC:
        return _meta_cache["data"]
    data = hl_info({"type": "metaAndAssetCtxs"}, weight=weight_info_default())
    _meta_cache["ts"] = now
    _meta_cache["data"] = data
    return data

def select_topn_filtered(top_n: int) -> List[Tuple[str, float, float, float]]:
    # Haal alle beschikbare perp-DEXes op (eerste element null => default DEX)
    try:
        dexs_res = hl_info({"type": "perpDexs"}, weight=weight_info_default())
    except Exception:
        dexs_res = []
    dex_list = []
    if isinstance(dexs_res, list) and len(dexs_res) > 0:
        dex_list.append("")  # default DEX (lege string)
        for entry in dexs_res[1:]:
            if isinstance(entry, dict) and "name" in entry:
                dex_list.append(entry["name"])
    if not dex_list:
        dex_list = [""]

    rows: List[Tuple[str, float, float, float]] = []
    seen_coins = set()
    # Loop over elke DEX (inclusief default)
    for dex in dex_list:
        payload = {"type": "metaAndAssetCtxs"}
        if dex:
            payload["dex"] = dex
        try:
            res = hl_info(payload, weight=weight_info_default())
        except Exception:
            continue
        if not (isinstance(res, list) and len(res) >= 2):
            continue
        meta = res[0] if isinstance(res[0], dict) else {}
        ctxs = res[1] if isinstance(res[1], list) else []
        universe = meta.get("universe", []) if isinstance(meta, dict) else []
        for i, asset in enumerate(universe):
            if not isinstance(asset, dict):
                continue
            coin = asset.get("name")
            if not coin or coin in seen_coins:
                continue
            seen_coins.add(coin)
            c = ctxs[i] if i < len(ctxs) else {}
            day_ntl = safe_float(c.get("dayNtlVlm", 0.0), 0.0)
            if day_ntl < MIN_DAY_NTL_VLM:
                continue
            mid = safe_float(c.get("midPx", 0.0), 0.0)
            impact = c.get("impactPxs") or [None, None]
            buy_imp = safe_float(impact[0], mid) if len(impact)>0 else mid
            sell_imp = safe_float(impact[1], mid) if len(impact)>1 else mid
            spread_pct = (abs(sell_imp - buy_imp) / mid * 100.0) if mid > 0 else 999.0
            if spread_pct > MAX_IMPACT_SPREAD_PCT:
                continue
            score = math.log1p(day_ntl) / (1.0 + 0.6 * spread_pct)
            rows.append((coin, score, day_ntl, spread_pct))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]

# =========================
# CANDLES
# =========================
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=0.6, min=0.6, max=6),
    retry=retry_if_exception_type((requests.RequestException,)),
)
def get_ohlcv(coin: str, lookback: int) -> Optional[pd.DataFrame]:
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - int(lookback * 3600 * 1000)

    payload = {
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": INTERVAL, "startTime": start_ts, "endTime": end_ts},
    }

    w = weight_candle_snapshot(approx_items=lookback)
    data = hl_info(payload, weight=w)

    if not data or not isinstance(data, list):
        return None

    rows = []
    for x in data:
        if not isinstance(x, dict):
            continue
        t = safe_float(x.get("t"))
        o = safe_float(x.get("o"))
        h = safe_float(x.get("h"))
        l = safe_float(x.get("l"))
        c = safe_float(x.get("c"))
        v = safe_float(x.get("v"), 0.0)
        if t == 0.0 or c == 0.0:
            continue
        rows.append([t, o, h, l, c, v])

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol"])
    df = df.sort_values("ts").reset_index(drop=True)
    if len(df) > lookback:
        df = df.iloc[-lookback:].reset_index(drop=True)
    return df

# =========================
# SIGNALS + MOMENTUM PHASE
# =========================
def check_signals(df: pd.DataFrame) -> Optional[Tuple[str, float, float, float, float, str]]:
    if df is None or len(df) < 35:
        return None

    df["EMA_9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["EMA_21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["RSI_14"] = ta.momentum.rsi(df["close"], window=14)
    df["ADX_14"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["ATR_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["ATRp_14"] = (df["ATR_14"] / df["close"]) * 100.0

    # NEW: volume baseline
    df["VOL_MA20"] = df["vol"].rolling(20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if any(pd.isna(last[k]) for k in ["EMA_9", "EMA_21", "RSI_14", "ADX_14", "ATRp_14"]):
        return None

    ema_bull = last["EMA_9"] > last["EMA_21"]
    ema_bear = last["EMA_9"] < last["EMA_21"]

    rsi = float(last["RSI_14"])
    adx = float(last["ADX_14"])
    atrp = float(last["ATRp_14"])
    close = float(last["close"])

    # ATR VOLATILITY FILTER (unchanged)
    if atrp < 0.7 or atrp > 6.0:
        return None

    adx_prev = float(prev["ADX_14"])
    rsi_prev = float(prev["RSI_14"])

    adx_slope = adx - adx_prev
    rsi_slope = rsi - rsi_prev

    # NEW: allow small RSI pullbacks
    if rsi_slope < -0.2:
        return None

    adx_strong = adx > ADX_TREND_THR
    adx_rising = ADX_PRE_MIN <= adx <= ADX_PRE_MAX and adx_slope > 0

    rsi_long = rsi > 55
    rsi_short = rsi < 45
    rsi_near_long = 50 < rsi <= 55
    rsi_near_short = 45 <= rsi < 50

    signal = None

    if adx_strong:
        if ema_bull and rsi_long:
            signal = "LONG"
        elif ema_bear and rsi_short:
            signal = "SHORT"

    elif adx_rising:
        if ema_bull and rsi_near_long:
            signal = "PRE-LONG"
        elif ema_bear and rsi_near_short:
            signal = "PRE-SHORT"

    if not signal:
        return None

    # MOMENTUM CANDLE
    ema_sep_raw = abs(last["EMA_9"] - last["EMA_21"])
    ema_sep = ema_sep_raw / close

    body = abs(last["close"] - last["open"])
    avg_body = (df["close"] - df["open"]).abs().tail(10).mean()

    range_size = last["high"] - last["low"]
    avg_range = (df["high"] - df["low"]).tail(10).mean()

    momentum_candle = body > avg_body * 1.2 and range_size > avg_range * 1.1

    candle_range = last["high"] - last["low"]

    upper_wick = last["high"] - max(last["open"], last["close"])
    lower_wick = min(last["open"], last["close"]) - last["low"]

    wick_ratio = (upper_wick + lower_wick) / candle_range if candle_range > 0 else 0

    if wick_ratio > 0.9 and signal in ("LONG","SHORT"):
        return None
    
    # NEW: ATR normalized separation
    atr = float(last["ATR_14"])
    ema_sep_norm = ema_sep_raw / atr if atr > 0 else ema_sep

    # PHASE
    phase = "MID"

    if adx > 25 and adx_slope > 0 and ema_sep_norm < 0.6:
        phase = "EARLY"

    elif adx > 35 and (rsi > 70 or rsi < 30 or ema_sep_norm > 1.2):
        phase = "LATE"

    # RANKING SCORE
    if "LONG" in signal:
        rsi_dist = max(0.0, rsi - 50.0)
    else:
        rsi_dist = max(0.0, 50.0 - rsi)

    pre_penalty = -2.0 if "PRE" in signal else 0.0

    # ADX weight slightly reduced
    score = float((adx * 0.8) + (rsi_dist * 0.6) + (ema_sep * 500.0) + pre_penalty)

    if momentum_candle:
        score += 2.5

    if phase == "EARLY":
        score += 3

    if phase == "LATE":
        score -= 4

    # NEW: volume score bonus
    vol = float(last["vol"])
    vol_ma = float(last["VOL_MA20"]) if not pd.isna(last["VOL_MA20"]) else 0

    if vol_ma > 0:
        vol_ratio = vol / vol_ma

        if vol_ratio > 1.8:
            score += 2.5
        elif vol_ratio > 1.4:
            score += 1.2

    return (signal, adx, rsi, atrp, score, phase)

# =========================
# SIZE HINT
# =========================
def size_hint(score: float) -> str:
    if score >= 40: return "Setup: A (normale size)"
    if score >= 34: return "Setup: B (kleiner)"
    return "Setup: C (heel klein / alleen als chart perfect is)"

# =========================
# SCAN & NOTIFY
# =========================
_last_scan_ts = 0.0

def scan_and_notify():
    global _last_scan_ts
    now = time.time()
    if now - _last_scan_ts < MIN_SECONDS_BETWEEN_SCANS:
        send_telegram_message(f"⏳ Even wachten: scan cooldown ({MIN_SECONDS_BETWEEN_SCANS}s).")
        return
    _last_scan_ts = now

    t0 = time.time()
    top_coins = select_topn_filtered(TOP_N)
    if not top_coins:
        send_telegram_message("❌ Geen coins gevonden (filters te streng of API issue).")
        return

    found, scanned, no_data, no_signal = [], 0, 0, 0

    for coin, activity_score, day_ntl, spread_pct in top_coins:
        scanned += 1
        try:
            df = get_ohlcv(coin, LOOKBACK)
        except Exception:
            no_data += 1
            continue
        if df is None or df.empty:
            no_data += 1
            continue

        res = check_signals(df)
        if not res:
            no_signal += 1
            continue

        signal, adx, rsi, atrp, score, phase = res
        regime = regime_from_adx(adx)

        if FILTER_CHOP_ENTRIES and regime == "CHOP" and signal in ("LONG", "SHORT"):
            no_signal += 1
            continue

        found.append((coin, signal, adx, rsi, atrp, score, regime, phase, day_ntl, spread_pct))

    dt = time.time() - t0
    send_telegram_message(
        f"⚙️ Debug: gescand={scanned} | signals={len(found)} | no_signal={no_signal} | no_data={no_data} | {dt:.1f}s\n"
        f"Filters: dayNtlVlm>={MIN_DAY_NTL_VLM:.0f} | impactSpread<={MAX_IMPACT_SPREAD_PCT:.2f}% | TOP_N={TOP_N}"
    )

    if not found:
        send_telegram_message("🔍 Geen kansen volgens jouw strategie (met huidige filters).")
        return

    found.sort(key=lambda x: x[5], reverse=True)
    topk = found[:MAX_RESULTS_IN_TELEGRAM]

    msg = "📊 Beste kansen (1H) — Hyperliquid\nKlik link → check chart (manual)\n\n"
    for coin, signal, adx, rsi, atrp, score, regime, phase, day_ntl, spread_pct in topk:
        tv = tv_link_for_coin(coin)
        hint = size_hint(score)
        emoji = "🟢" if signal == "LONG" else "🔴" if signal == "SHORT" else "🟡 ⚠️"
        pre_text = "Dit is een VOOR-signaal (kijken, niet blind traden)\n" if "PRE" in signal else ""
        msg += (
            f"{emoji} {signal} — {coin}\n"
            f"Regime: {regime}\n"
            f"Momentum phase: {phase}\n"
            f"ADX: {adx:.1f} | RSI: {rsi:.1f} | ATR%: {atrp:.2f} | Score: {score:.1f}\n"
            f"{pre_text}"
            f"Liquidity: dayNtlVlm={day_ntl/1e6:.1f}M | impact≈{spread_pct:.2f}%\n"
            f"{hint}\n"
            f"Open chart:\n{tv}\n\n"
        )
    send_telegram_message(msg)

# =========================
# FLASK WEBHOOK
# =========================
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "ok"

@app.route(f"/{WEBHOOK_SECRET}", methods=["POST"])
def telegram_webhook():
    data = request.get_json(silent=True) or {}
    text = (data.get("message", {}).get("text") or "").strip().lower()

    if text == "zoek":
        threading.Thread(target=scan_and_notify, daemon=True).start()
        return "ok"

    if text == "status":
        send_telegram_message(
            f"✅ Bot online.\n"
            f"TOP_N={TOP_N}, LOOKBACK={LOOKBACK}, FILTER_CHOP_ENTRIES={FILTER_CHOP_ENTRIES}\n"
            f"Filters: MIN_DAY_NTL_VLM={MIN_DAY_NTL_VLM:.0f}, MAX_IMPACT_SPREAD_PCT={MAX_IMPACT_SPREAD_PCT:.2f}"
        )
        return "ok"

    return "ok"

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
