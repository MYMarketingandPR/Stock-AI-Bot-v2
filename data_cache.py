"""
data_cache.py — 研究用嘅下載快取（Esther 2026-10-05 提議：下載過嘅資料儲低，唔使每次重新下載）

只喺環境變數 DATA_CACHE 有設定時先用（research_runner.yml 會設定，並用 GitHub Actions cache 喺每次執行之間保存）；
模擬交易 bot 冇設定 → 照舊直接下載最新數據。

- 價錢（每隻股票）：儲低成段 Close + Volume（已計股息）同當時要求嘅日期範圍；之後要求嘅範圍喺入面就直接切出嚟用。
  ⚠️ 已計股息嘅舊價錢，每次派息後Yahoo會微調；研究用嚟講影響好細，可以接受。
- 盈利紀錄 / 分析師紀錄：60日內下載過就用返。
- 「冇數據」（已除牌等）都會記低，唔會每次再試；但「被限制/斷線」唔會記低，下次會再試。
"""
import os
import pickle
import re
from datetime import date

import pandas as pd

DIR = os.environ.get("DATA_CACHE")
TRANSIENT = ("rate", "too many", "timed out", "timeout", "connection", "429", "remote", "temporarily")


def _path(kind, key):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return os.path.join(DIR, kind, safe + ".pkl")


def _load(kind, key):
    p = _path(kind, key)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _save(kind, key, obj):
    p = _path(kind, key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def history(t, start, end):
    """同 yf.Ticker(t).history(start, end, auto_adjust=True) 一樣，但有快取。"""
    import yfinance as yf
    if not DIR:
        return yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
    c = _load("px", t)
    if c and c["start"] <= start and c["end"] >= end:
        h = c["data"]
        if h is None or h.empty:
            return h
        idx = pd.DatetimeIndex(h.index).tz_localize(None) if getattr(h.index, "tz", None) else pd.DatetimeIndex(h.index)
        keep = (idx >= pd.Timestamp(start)) & (idx < pd.Timestamp(end))
        return h[keep]
    try:
        h = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
    except Exception as e:
        if not any(w in str(e).lower() for w in TRANSIENT):
            _save("px", t, {"start": start, "end": end, "data": None})     # 已除牌等：記低，下次唔再試
        raise
    if h is not None:
        cols = [x for x in ("Close", "Volume") if x in h.columns]
        _save("px", t, {"start": start, "end": end, "data": h[cols] if cols else h})
    return h


def _info(kind, t, getter):
    if not DIR:
        return getter()
    c = _load(kind, t)
    if c and (date.today() - c["fetched"]).days <= 60:
        if c["data"] is None:
            raise LookupError("冇數據（快取）")
        return c["data"]
    try:
        d = getter()
    except Exception as e:
        if not any(w in str(e).lower() for w in TRANSIENT):
            _save(kind, t, {"fetched": date.today(), "data": None})
        raise
    _save(kind, t, {"fetched": date.today(), "data": d if (d is not None and len(d)) else None})
    return d


def earnings(t):
    import yfinance as yf
    return _info("earn", t, lambda: yf.Ticker(t).get_earnings_dates(limit=100))


def upgrades(t):
    import yfinance as yf
    return _info("ana", t, lambda: yf.Ticker(t).upgrades_downgrades)
