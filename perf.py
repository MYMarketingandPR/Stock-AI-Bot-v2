"""
perf.py — 成績表用嘅計數工具（模擬交易每月成績表 + 第十三輪研究共用）

全部用「每日回報」計：
- 總回報、年化回報、最大跌幅、波幅（年化）、Sharpe（扣現金利率）
- Calmar = 年化回報 ÷ 最大跌幅
- 回本時間 = 由最大跌幅嗰個最低點，返到之前最高位要幾多日（未返到 = 未回本）
- 升市/跌市捕捉（對比基準，例如QQQ）：
    升市捕捉 = 基準升嗰啲時段，策略平均升幅 ÷ 基準平均升幅 ×100%
    跌市捕捉 = 基準跌嗰啲時段，策略平均跌幅 ÷ 基準平均跌幅 ×100%
  升市捕捉高、跌市捕捉低先係好；如果兩樣都高，即係「多賺」只係因為承受多咗跌。
"""
import numpy as np
import pandas as pd


def nav_of(r):
    return (1 + r.fillna(0.0)).cumprod()


def max_dd(r):
    v = np.concatenate([[1.0], nav_of(r).values])
    return (v / np.maximum.accumulate(v) - 1).min()


def recovery_days(r):
    """最大跌幅最低點 → 返到之前高位嘅日曆日數。回傳 (日數, 有冇回本)；冇跌過回傳 (0, True)。"""
    nav = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)]), nav_of(r)])
    peak = nav.cummax()
    dd = nav / peak - 1
    if dd.min() >= 0:
        return 0, True
    trough = dd.idxmin()
    after = nav[nav.index > trough]
    back = after[after >= peak.loc[trough]]
    if back.empty:
        return (nav.index[-1] - trough).days, False
    return (back.index[0] - trough).days, True


def capture(r, b, freq=None):
    """升市/跌市捕捉（%）。freq=None 用每日，'ME' 用每月。"""
    d = pd.concat([r.rename("s"), b.rename("b")], axis=1).dropna()
    if freq:
        d = (1 + d).resample(freq).prod() - 1
    up, dn = d["b"] > 0, d["b"] < 0
    upc = d.loc[up, "s"].mean() / d.loc[up, "b"].mean() * 100 if up.sum() >= 3 else np.nan
    dnc = d.loc[dn, "s"].mean() / d.loc[dn, "b"].mean() * 100 if dn.sum() >= 3 else np.nan
    return upc, dnc


def perf(r, rf=None, bench=None, turn=None, cap_freq=None, asof_year=None):
    """r = 每日回報（Series）。回傳 dict（百分比已乘100）。"""
    r = r.dropna()
    n = len(r)
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1 / 365.25) if n > 1 else 1 / 365.25
    nav = nav_of(r)
    out = {"days": n, "total": (nav.iloc[-1] - 1) * 100}
    out["cagr"] = (nav.iloc[-1] ** (1 / yrs) - 1) * 100 if n >= 2 else np.nan
    out["vol"] = r.std() * np.sqrt(252) * 100 if n >= 5 else np.nan
    ex = r - (rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0)
    out["sharpe"] = ex.mean() / r.std() * np.sqrt(252) if n >= 5 and r.std() > 0 else np.nan
    out["mdd"] = max_dd(r) * 100
    out["calmar"] = out["cagr"] / abs(out["mdd"]) if out["mdd"] < 0 else np.nan
    out["rec_days"], out["recovered"] = recovery_days(r)
    if asof_year is not None:
        y = r[r.index.year == asof_year]
        out["ytd"] = ((1 + y).prod() - 1) * 100 if len(y) else np.nan
    for k, w in (("r3m", 63), ("r6m", 126), ("r12m", 252)):
        out[k] = ((1 + r.iloc[-w:]).prod() - 1) * 100 if n >= w else np.nan
    if turn is not None:
        t = turn.reindex(r.index).fillna(0.0)
        out["turn_total"] = t.sum() * 100
        out["turn_year"] = t.sum() / yrs * 100
    if bench is not None:
        out["up_cap"], out["down_cap"] = capture(r, bench, cap_freq)
    return out
