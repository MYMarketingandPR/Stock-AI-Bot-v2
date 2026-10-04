"""
strategy_v2.py — Minervini 趨勢模板 + RS評級 + 突破入場（預先登記嘅規則）

⚠️ 預先登記（pre-registration）：以下所有數字都係喺第一次跑回測之前定好，
   而且全部嚟自公開嘅書／業界慣例，唔係由我哋睇住數據揀出嚟。
   跑完之後唔會因為結果去改呢度嘅任何數字——要改就要當新策略、用新數據重新驗證。
   (git history 可以證明呢個file喺回測結果出嚟之前已經commit咗。)

來源：
- 趨勢模板8條：Mark Minervini《超級績效》(Trade Like a Stock Market Wizard, 2013)
- RS評級：IBD RS Rating 常見近似公式（最近3個月佔40%，之前三個季度各佔20%），
  每日喺股票池入面排百分位(1-99)，門檻70（書入面寫明最少70，最好80至90幾）
- 突破成交量：O'Neil 準則，突破日成交量高過50日平均最少40%
- 唔追高：O'Neil 準則，入場價唔高過突破位(pivot)5%
- 止損上限：O'Neil 7-8% 止損原則；正常用2倍ATR，兩者取較近（較細蝕幅）嗰個
- 浮動止損：Chandelier exit（LeBeau）：持倉期間最高位 − 3倍ATR，只升唔跌
- 組合風險：每單最多蝕組合0.75%、單一倉位上限25%、最多10個倉位、每日最多3個新倉
- 交易成本：每邊0.10%（買賣各一次）

已知限制：
- 只係 Minervini 方法嘅「技術部分」：冇基本面（盈利/營收增長）、冇VCP形態偵測、冇主觀判斷。
- 股票名單係今日嘅名單（倖存者偏差），愈早年份偏差愈大。
- 歷史價格係拆股/派息調整後價格。
"""
import numpy as np
import pandas as pd

from strategy_core import MEGA_CAP_TICKERS, HIGH_BETA_TICKERS

# ---------- 股票池：剔除ETF（反向/槓桿ETF、債券、商品、指數ETF唔係個股） ----------
EXCLUDED_ETFS = {
    "DIA", "GDX", "GLD", "IWM", "SLV", "SMH", "SPY",                    # Mega Cap名單入面嘅ETF
    "ARKK", "EEM", "KRE", "QQQ", "SOXL", "SOXS", "SOXX", "SQQQ",       # High Beta名單入面嘅ETF
    "TLT", "TQQQ", "UNG",
}

# ---------- 趨勢模板（Minervini 8條） ----------
SMA_50, SMA_150, SMA_200 = 50, 150, 200
SMA200_RISING_DAYS = 21          # 第3條：200日線至少上升咗1個月（約21個交易日）
MIN_ABOVE_52W_LOW = 1.30         # 第6條：股價比52週低位高出最少30%
MIN_VS_52W_HIGH = 0.75           # 第7條：股價離52週高位25%以內
RS_MIN = 70                      # 第8條：RS評級最少70
WEEKS_52 = 252

# ---------- 入場 ----------
MARKET_SMA = 200                 # 大市濾網：SPY收市 > SPY 200日線
PIVOT_DAYS = 20                  # 突破位 = 之前20個交易日最高價（唔計當日）
VOLUME_AVG_DAYS = 50
BREAKOUT_VOLUME_MULT = 1.4       # 突破日成交量 >= 50日平均 x 1.4
MAX_CHASE = 1.05                 # 第二日開市價 > 突破位 x 1.05 就唔追

# ---------- 風險同出場 ----------
ATR_DAYS = 14
INITIAL_STOP_ATR = 2.0
MAX_INITIAL_STOP_PCT = 0.08
TRAIL_ATR = 3.0
RISK_PER_TRADE = 0.0075
MAX_POSITION_PCT = 0.25
MAX_POSITIONS = 10
MAX_NEW_PER_DAY = 3
COST_PER_SIDE = 0.001


def v2_universe():
    """回傳 [(ticker, group), ...]，已剔除ETF。"""
    out = [(t, "Mega Cap") for t in MEGA_CAP_TICKERS if t not in EXCLUDED_ETFS]
    out += [(t, "High Beta") for t in HIGH_BETA_TICKERS if t not in EXCLUDED_ETFS]
    return out


def add_indicators(df):
    """每隻股自己計（喺佢自己有數據嘅日子上面計，之後先對齊總日曆）。"""
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    df["SMA50"] = c.rolling(SMA_50).mean()
    df["SMA150"] = c.rolling(SMA_150).mean()
    df["SMA200"] = c.rolling(SMA_200).mean()
    df["SMA200_prev"] = df["SMA200"].shift(SMA200_RISING_DAYS)
    df["Low52w"] = l.rolling(WEEKS_52).min()
    df["High52w"] = h.rolling(WEEKS_52).max()
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / ATR_DAYS, adjust=False, min_periods=ATR_DAYS).mean()
    df["VolAvg50"] = df["Volume"].rolling(VOLUME_AVG_DAYS).mean()
    df["Pivot"] = h.rolling(PIVOT_DAYS).max().shift(1)
    # IBD風格RS原始分：40%最近一季 + 20% x 之前三季（用累積回報比率，排名結果同用ROC一樣）
    df["RSRaw"] = (0.4 * c / c.shift(63) + 0.2 * c / c.shift(126)
                   + 0.2 * c / c.shift(189) + 0.2 * c / c.shift(252))
    return df


def rs_rating(rs_raw_panel):
    """每日喺有數據嘅股票之間排百分位，轉做1-99。"""
    pct = rs_raw_panel.rank(axis=1, pct=True)
    return (pct * 98 + 1).where(rs_raw_panel.notna())


def trend_template(P, rs):
    """P: dict of wide DataFrames (日期 x 股票)。回傳布林DataFrame。"""
    c = P["Close"]
    t1 = (c > P["SMA150"]) & (c > P["SMA200"])
    t2 = P["SMA150"] > P["SMA200"]
    t3 = P["SMA200"] > P["SMA200_prev"]
    t4 = (P["SMA50"] > P["SMA150"]) & (P["SMA50"] > P["SMA200"])
    t5 = c > P["SMA50"]
    t6 = c >= MIN_ABOVE_52W_LOW * P["Low52w"]
    t7 = c >= MIN_VS_52W_HIGH * P["High52w"]
    t8 = rs >= RS_MIN
    return t1 & t2 & t3 & t4 & t5 & t6 & t7 & t8


def breakout(P):
    return (P["Close"] > P["Pivot"]) & (P["Volume"] >= BREAKOUT_VOLUME_MULT * P["VolAvg50"])


def initial_stop(entry, atr):
    """2倍ATR，但最多蝕8%（取較近入場價嗰個）。"""
    return max(entry - INITIAL_STOP_ATR * atr, entry * (1 - MAX_INITIAL_STOP_PCT))


def position_value(equity, entry, stop):
    """按風險計注碼：蝕到止損時最多蝕組合0.75%，但單一倉位唔超過25%。"""
    stop_pct = (entry - stop) / entry
    if stop_pct <= 0:
        return 0.0
    return equity * min(RISK_PER_TRADE / stop_pct, MAX_POSITION_PCT)
