"""
research_v8.py — 第八輪：換倉頻率（每週 / 每半個月 / 每月 / 每兩個月）

⚠️ 預先登記（跑之前定好，跑完唔改）：
設定同research_v6/v7一樣（當時標普500成分股、12-1動量、每次買賣0.1%、收市決定→下日收市交易）。
動量計分永遠係「252個交易日前到21個交易日前」嘅回報，只係改幾耐重新排名一次：
  每週：每個星期最後一個交易日
  每半個月：每月15號或之前最後一個交易日 + 月尾最後一個交易日
  每月：月尾最後一個交易日（而家用緊）
  每兩個月：雙數月月尾
策略：
  S1 原始動量前10隻 70% + SPY訊號 30%（模擬交易線A）
  S2 原始動量前3隻 100%（模擬交易線B）
  S3 原始動量前10隻 100%
揀選期2010-2017；驗證期2018-2026/09。
問題：每半個月（或每週）係咪喺兩段時間都好過每月？同時列出每年換手（交易量）。
輸出：v8_results.csv、v8_progress.txt
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v5 import COST, PERIODS, stats
from research_v6 import effective_weights, daily_returns, LOOK, SKIP

M.PROGRESS = "v8_progress.txt"
log = M.log


def decision_days(days, kind):
    s = pd.Series(days, index=days)
    if kind == "每週":
        return list(s.groupby([days.isocalendar().year, days.isocalendar().week]).max())
    month_end = list(s.groupby([days.year, days.month]).max())
    if kind == "每月":
        return month_end
    if kind == "每兩個月":
        return [d for d in month_end if d.month % 2 == 0]
    mid = list(s[days.day <= 15].groupby([days[days.day <= 15].year, days[days.day <= 15].month]).max())
    return sorted(set(month_end) | set(mid))


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    month_ends = pd.date_range("2008-12-31", "2026-09-30", freq="ME")
    everyone = sorted(set().union(*[members_fn(d) for d in month_ends]))
    closes = M.download_close(everyone + ["SPY"], "2007-06-01", "2026-10-01")
    spy = closes.pop("SPY")
    px = pd.DataFrame(closes).reindex(spy.index)
    px, spy = px[px.index <= pd.Timestamp("2026-09-30")], spy[spy.index <= pd.Timestamp("2026-09-30")]
    rets = px.pct_change(fill_method=None)
    spy_r = spy.pct_change().fillna(0.0)
    cash_r = M.get_cash(px.index)
    days = px.index

    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    st = (df["state"] != "out").astype(float).reindex(days).fillna(0.0).shift(2).fillna(0.0)
    bot_r = st * spy_r + (1 - st) * cash_r - st.diff().abs().fillna(0.0) * COST

    results, turnover = {}, {}
    for kind in ("每週", "每半個月", "每月", "每兩個月"):
        tg = {3: {}, 10: {}}
        for d in decision_days(days, kind):
            i = days.get_loc(d)
            if i < LOOK + 5 or d < pd.Timestamp("2009-06-01"):
                continue
            mem = members_fn(d)
            alive = [t for t in mem if t in px.columns and np.isfinite(px[t].iloc[i])
                     and np.isfinite(px[t].iloc[i - LOOK])]
            if len(alive) < 50:
                continue
            R = (px[alive].iloc[i - SKIP] / px[alive].iloc[i - LOOK] - 1).dropna().sort_values(ascending=False)
            for n in (3, 10):
                tg[n][d] = pd.Series(1 / n, index=R.index[:n])
        for n in (3, 10):
            W, _ = effective_weights(rets, tg[n])
            mom = daily_returns(W, rets, cash_r)
            tv = W.diff().abs().sum(axis=1)
            turnover[(kind, n)] = tv[tv.index >= pd.Timestamp("2010-01-01")].sum() / 16.75
            if n == 10:
                results[(kind, "S1 前10隻70% + SPY訊號30%")] = 0.7 * mom + 0.3 * bot_r
                results[(kind, "S3 前10隻100%")] = mom
            else:
                results[(kind, "S2 前3隻100%")] = mom
        log(f"{kind} 完成")

    pv, pp = "驗證期2018-2026", "揀選期2010-2017"
    L = ["=== 第八輪：換倉頻率（規則預先登記，見research_v8.py開頭）===",
         f"股票池：當時標普500成分股（{source}）；每次買賣0.1%", ""]
    for strat in ("S1 前10隻70% + SPY訊號30%", "S2 前3隻100%", "S3 前10隻100%"):
        L.append(f"=== {strat} ===")
        base = stats(results[("每月", strat)], *[p for p in PERIODS if p[0] == pv][0][1:])
        for kind in ("每週", "每半個月", "每月", "每兩個月"):
            s = {pn: stats(results[(kind, strat)], a, b) for pn, a, b in PERIODS}
            n = 3 if "前3" in strat else 10
            L.append(f"　{kind}：" + " | ".join(f"{pn} {x['cagr']:+.2f}%（跌 {x['mdd']:.1f}%）" for pn, x in s.items())
                     + f" | 每年換手約 {turnover[(kind, n)] * 100:.0f}%")
        sm = {pn: stats(results[("每半個月", strat)], a, b) for pn, a, b in PERIODS}
        mm = {pn: stats(results[("每月", strat)], a, b) for pn, a, b in PERIODS}
        both = sm[pp]["cagr"] > mm[pp]["cagr"] and sm[pv]["cagr"] > mm[pv]["cagr"]
        L.append(f"　→ 每半個月 {'✅ 兩段時間都好過每月' if both else '❌ 唔係兩段時間都好過每月'}"
                 f"（揀選期 {sm[pp]['cagr'] - mm[pp]['cagr']:+.2f}%，驗證期 {sm[pv]['cagr'] - mm[pv]['cagr']:+.2f}%）")
        L.append("")
    pd.DataFrame({"report": L}).to_csv("v8_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
