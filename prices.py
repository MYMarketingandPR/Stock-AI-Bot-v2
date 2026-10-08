"""
prices.py — 確保用到「當日收市價」（2026-10-08 Esther 發現：Telegram 用咗前一日收市價）

問題：bot 喺美股收市後幾個鐘跑，但 Yahoo 有時仲未有部分股票當日嘅日線。
舊程式只檢查「成個價錢表最後一日係咪今日」——只要有一隻股票有今日價就當過關，
其他未更新嘅股票就靜靜雞用咗前一日收市價。
而家：逐隻檢查；未有就等幾分鐘再下載；仲未有就用 Yahoo 即時報價（要係當日收市後嘅報價）；
再唔得就話俾程式知，由程式決定遲啲再試（workflow 每晚有後備時間再跑）。
"""
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
CUTOFF_HOUR, CUTOFF_MIN = 16, 15     # 紐約時間 16:15 之後先當今日已收市


def now_ny():
    return datetime.now(NY)


def market_day(now=None):
    """最近一個已收市嘅平日（紐約時間 16:15 前 = 上一個平日）。GitHub 排程遲咗過咗半夜都計返啱嗰日。"""
    now = now or now_ny()
    d = now.date()
    if (now.hour, now.minute) < (CUTOFF_HOUR, CUTOFF_MIN):
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def last_chance(now=None):
    """今晚最後一次機會（紐約 22:00 之後，或者已經過咗半夜）：仲未齊價都要照做，但會標明。"""
    now = now or now_ny()
    return now.hour >= 22 or (now.hour, now.minute) < (CUTOFF_HOUR, CUTOFF_MIN)


def missing(px, tickers, day):
    day = pd.Timestamp(day)
    out = []
    for t in tickers:
        if t not in px.columns:
            out.append(t)
            continue
        v = px[t].get(day, np.nan)
        if not np.isfinite(v):
            out.append(t)
    return out


def _quote(t, day):
    """Yahoo 報價：要係 day 收市之後先用（regularMarketTime 係 day 16:00 或之後）。"""
    import yfinance as yf
    try:
        info = yf.Ticker(t).info
        p, ts = info.get("regularMarketPrice"), info.get("regularMarketTime")
        if p and ts:
            when = datetime.fromtimestamp(int(ts), NY)
            if when.date() == day and when.hour >= 16:
                return float(p)
    except Exception:
        pass
    return None


def ensure_fresh(px, tickers, day, rounds=3, wait=300, log=print):
    """要 tickers 每隻都有 day 嘅收市價。回傳 (px, 仲係冇嘅股票)。px 會加返 day 嗰行。"""
    import yfinance as yf
    day_ts = pd.Timestamp(day)
    px = px.copy()
    if day_ts not in px.index:
        px.loc[day_ts] = np.nan
        px = px.sort_index()
    miss = missing(px, tickers, day)
    for k in range(rounds):
        if not miss:
            break
        log(f"{len(miss)} 隻未有 {day} 收市價（例如 {miss[:8]}），等 {wait // 60} 分鐘再下載（第 {k + 1} 次）")
        time.sleep(wait)
        for t in list(miss):
            try:
                h = yf.Ticker(t).history(start=(day_ts - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                                         end=(day_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), auto_adjust=True)
                s = h["Close"].astype(float)
                s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
                if day_ts in s.index and np.isfinite(s[day_ts]):
                    px.loc[day_ts, t] = float(s[day_ts])
            except Exception:
                pass
            time.sleep(0.2)
        miss = missing(px, tickers, day)
        if len(tickers) > 1 and len(miss) == len(tickers):
            break                              # 全部都冇：多數係休市，唔使再等
    for t in list(miss):
        q = _quote(t, day)
        if q:
            px.loc[day_ts, t] = q
            log(f"{t}：用 Yahoo 收市後報價 ${q:,.2f}")
    miss = missing(px, tickers, day)
    if miss:
        log(f"仲有 {len(miss)} 隻未有 {day} 收市價：{miss[:20]}")
    return px, miss
