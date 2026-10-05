"""
research_v11.py — 第十一輪：相對SPY、動量加速、動量廣度、行業→個股、動量 + 回調買入

⚠️ 預先登記（跑之前定好，跑完唔改）：

共同設定：當時標普500成分股、已計股息價格、每半個月換倉（同線A）、收市決定→下日收市交易、
每次買賣0.1%、唔用槓桿。除非另外寫明，都係「70%動量部分 + 30% SPY訊號方案（main.py）」。
  R121 = 12-1個月回報（252至21個交易日前）；R3 = 3個月回報（63日）
候選：
  T1 線A基準：R121前10隻，平均分配
  T2a 相對SPY過濾：R121前10隻入面，只揸R121高過SPY同期R121嘅，其餘份額放現金
  T2b 扣大市：R121 − β×SPY R121（β = 過去252日對SPY），前10隻
  T3 動量加速：R121前30名入面，揀 (R3 − R121÷4) 最高嘅10隻
  T4a 動量廣度：R121前10隻；動量部分只投入「前10入面R3>0嘅比例」，其餘放現金
  T4b 廣度（對SPY）：同上，但用「R3高過SPY R3嘅比例」
  T5 行業→個股：行業按成分股R121平均排名，頭3個行業，每個行業R121最高3隻（9隻平均）
  回調買入引擎（MR）：每個交易日，當時成分股入面：收市高過200日線、R121排頭30%、5日內跌7%或以上
     → 下日收市買，揸10個交易日；最多10隻，每隻10%；信號太多揀跌得最多嘅
  T6a 50% T1動量 + 20% MR + 30% SPY訊號
  T6b 35% T1動量 + 35% MR + 30% SPY訊號
評估：
  - 全期2010-2026、揀選期2010-2017、驗證期2018-2026、乾淨樣本2020-2026、2015-2026
  - Sharpe（每日回報平均÷標準差×√252，同第十輪一樣）
  - Walk-forward 2015-2026：每年用之前5年評分（40%回報+30%跌幅+20%Sharpe+10%換手，跌幅好過-35%入面）揀
  - MR引擎單獨表現，同MR同T1嘅每日回報相關系數
  - 目標：每年 ≥ +20%、Sharpe ≥ 1.0、最大跌幅好過 -30%
輸出：v11_results.csv、v11_progress.txt
==== 結果（2026-10-05 第一次跑，記錄低，唔再改）====
- 冇一個達到目標（+20%、Sharpe 1.0、跌幅好過-30%）。
- T1線A：全期+18.95%（跌-35.1%，Sharpe 0.83）；2020-2026 +25.04%。
- T2a相對SPY過濾：同T1完全一樣（前10名一定跑贏SPY）→ 冇作用。T2b扣大市：跌幅-39.4%，更差。
- T3加速：全期+12.34%，明顯變差。
- T4a廣度：+13.82%（跌-26.3%）；T4b廣度對SPY：+15.25%（跌-26.8%，Sharpe 0.81）→ 跌幅細咗但回報少咗。
- T5行業→個股：+15.92%（跌-29.3%）；驗證期+20.12%但揀選期只有+11.51%。
- 回調買入引擎：單獨+10.66%（跌-34.2%，Sharpe 0.59），同動量相關系數0.77（唔低）→ T6a/T6b都差過T1。
- Walk-forward 2015-2026：+14.79%（Sharpe 0.74）< T1固定 +17.44% → 按過去表現轉方法冇幫助。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v5 import PERIODS, stats, sectors_for
from research_v6 import effective_weights, LOOK, SKIP
from research_v8 import decision_days
from research_v10 import sleeve, metrics, score_pick

M.PROGRESS = "v11_progress.txt"
log = M.log
COST = 0.001


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
    sec = sectors_for(sorted(px.columns))
    days = px.index

    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    st = (df["state"] != "out").astype(float).reindex(days).fillna(0.0).shift(2).fillna(0.0)
    bot_r = st * spy_r + (1 - st) * cash_r - st.diff().abs().fillna(0.0) * COST

    # ---------- 動量候選（每半個月）----------
    names = ["T1 線A基準", "T2a 相對SPY過濾", "T2b 扣大市", "T3 動量加速", "T4a 動量廣度", "T4b 廣度（對SPY）", "T5 行業→個股"]
    tg = {n: {} for n in names}
    for d in decision_days(days, "每半個月"):
        i = days.get_loc(d)
        if i < LOOK + 5 or d < pd.Timestamp("2009-06-01"):
            continue
        alive = [t for t in members_fn(d) if t in px.columns and np.isfinite(px[t].iloc[i])
                 and np.isfinite(px[t].iloc[i - LOOK])]
        if len(alive) < 50:
            continue
        P = px[alive]
        R121 = (P.iloc[i - SKIP] / P.iloc[i - LOOK] - 1).dropna()
        R3 = (P.iloc[i] / P.iloc[i - 63] - 1)
        S121 = spy.iloc[i - SKIP] / spy.iloc[i - LOOK] - 1
        S3 = spy.iloc[i] / spy.iloc[i - 63] - 1
        order = list(R121.sort_values(ascending=False).index)
        top = order[:10]
        ew = lambda lst, total=1.0: pd.Series(total / len(lst), index=lst) if lst else pd.Series(dtype=float)
        tg[names[0]][d] = ew(top)
        tg[names[1]][d] = pd.Series({t: 0.1 for t in top if R121[t] > S121}, dtype=float)
        win = rets[alive].iloc[i - LOOK + 1:i + 1]
        sw = spy_r.iloc[i - LOOK + 1:i + 1]
        beta = win.apply(lambda c: c.cov(sw) / sw.var())
        adj = (R121 - beta.reindex(R121.index) * S121).dropna().sort_values(ascending=False)
        tg[names[2]][d] = ew(list(adj.index[:10]))
        acc = (R3.reindex(order[:30]) - R121.reindex(order[:30]) / 4).dropna().sort_values(ascending=False)
        tg[names[3]][d] = ew(list(acc.index[:10]))
        tg[names[4]][d] = ew(top, float((R3.reindex(top) > 0).mean()))
        tg[names[5]][d] = ew(top, float((R3.reindex(top) > S3).mean()))
        secs = pd.Series({t: sec.get(t, "未知") for t in R121.index})
        srank = R121.groupby(secs).mean().drop("未知", errors="ignore").sort_values(ascending=False)
        pick = []
        for g in srank.index[:3]:
            pick += list(R121[secs == g].sort_values(ascending=False).index[:3])
        tg[names[6]][d] = ew(pick)
    log("動量候選揀股完成")

    sl, tv = {}, {}
    for n in names:
        W, _ = effective_weights(rets, tg[n])
        sl[n], tv[n] = sleeve(W, rets, cash_r)

    # ---------- 回調買入引擎 ----------
    sma200 = px.rolling(200).mean()
    R5 = px / px.shift(5) - 1
    R121d = px.shift(SKIP) / px.shift(LOOK) - 1
    Wmr = pd.DataFrame(0.0, index=days, columns=px.columns)
    active = []          # (ticker, start_i, end_i)
    mem_cache = {}
    for i in range(LOOK + 5, len(days) - 2):
        d = days[i]
        if d < pd.Timestamp("2009-06-01"):
            continue
        key = (d.year, d.month)
        if key not in mem_cache:
            mem_cache[key] = [t for t in members_fn(d) if t in px.columns]
        mem = mem_cache[key]
        rk = R121d.iloc[i][mem].rank(pct=True)
        cond = (px.iloc[i][mem] > sma200.iloc[i][mem]) & (rk >= 0.7) & (R5.iloc[i][mem] <= -0.07)
        sig = R5.iloc[i][mem][cond.fillna(False)].sort_values()
        start = i + 2
        active = [a for a in active if a[2] > start]
        held = {a[0] for a in active if a[1] <= start < a[2]}
        free = 10 - len([a for a in active if a[1] <= start < a[2]])
        for t in sig.index:
            if free <= 0:
                break
            if t in held:
                continue
            active.append((t, start, min(start + 10, len(days))))
            Wmr.iloc[start:min(start + 10, len(days)), Wmr.columns.get_loc(t)] = 0.1
            held.add(t)
            free -= 1
    mr, mr_tv = sleeve(Wmr, rets, cash_r)
    log("回調買入引擎完成")

    cands = {n: 0.7 * sl[n] + 0.3 * bot_r for n in names}
    turns = {n: 0.7 * tv[n] for n in names}
    cands["T6a 50%動量 + 20%回調 + 30%SPY"] = 0.5 * sl[names[0]] + 0.2 * mr + 0.3 * bot_r
    turns["T6a 50%動量 + 20%回調 + 30%SPY"] = 0.5 * tv[names[0]] + 0.2 * mr_tv
    cands["T6b 35%動量 + 35%回調 + 30%SPY"] = 0.35 * sl[names[0]] + 0.35 * mr + 0.3 * bot_r
    turns["T6b 35%動量 + 35%回調 + 30%SPY"] = 0.35 * tv[names[0]] + 0.35 * mr_tv

    PER = PERIODS + [("2015-2026", "2015-01-01", "2026-09-30")]
    zero = pd.Series(0.0, index=days)
    L = ["=== 第十一輪（規則預先登記，見research_v11.py開頭）===",
         f"股票池：當時標普500成分股（{source}）；已計股息；每半個月換倉；每次買賣0.1%；唔用槓桿",
         "目標：每年 ≥ +20%、Sharpe ≥ 1.0、最大跌幅好過 -30%", ""]

    def fmt(name, r, t):
        parts = []
        for pn, a, b in PER:
            m = metrics(r, t, a, b)
            parts.append(f"{pn} {m['cagr']:+.2f}%（跌 {m['mdd']:.1f}%，Sharpe {m['sharpe']:.2f}）")
        mm = metrics(r, t, "2010-01-01", "2026-09-30")
        goal = "🌟達標" if (mm["cagr"] >= 20 and mm["sharpe"] >= 1.0 and mm["mdd"] > -30) else ""
        return f"　{name}：" + " | ".join(parts) + f" | 每年換手 {mm['turn'] * 100:.0f}% {goal}"

    L.append("=== 比較對象 ===")
    L.append(fmt("SPY長揸（總回報）", spy_r, zero))
    L.append(fmt("SPY訊號方案", bot_r, zero))
    L.append(fmt("回調買入引擎（單獨100%）", mr, mr_tv))
    c = pd.concat([mr, sl[names[0]]], axis=1).dropna()
    c = c[c.index >= pd.Timestamp("2010-01-01")]
    L.append(f"　回調引擎同T1動量嘅每日回報相關系數：{c.corr().iloc[0, 1]:.2f}；回調引擎平均持倉 {Wmr.sum(axis=1)[Wmr.index >= pd.Timestamp('2010-01-01')].mean() * 100:.0f}%")
    L.append("")
    L.append("=== 候選（全部組合）===")
    for n, r in cands.items():
        L.append(fmt(n, r, turns[n]))
    L.append("")
    L.append("=== Walk-forward 2015-2026（每年用之前5年評分揀）===")
    oos = []
    for y in range(2015, 2027):
        a, b = f"{y - 5}-01-01", f"{y - 1}-12-31"
        best, _ = score_pick({n: metrics(r, turns[n], a, b) for n, r in cands.items()})
        r = cands[best]
        seg = r[(r.index >= pd.Timestamp(f"{y}-01-01")) & (r.index <= pd.Timestamp(f"{y}-12-31"))]
        oos.append(seg)
        L.append(f"　{y}：揀「{best}」→ 當年 {((1 + seg).prod() - 1) * 100:+.1f}%（SPY {((1 + spy_r[spy_r.index.year == y]).prod() - 1) * 100:+.1f}%）")
    wf = pd.concat(oos)
    m_wf = metrics(wf, zero, "2015-01-01", "2026-09-30")
    m_t1 = metrics(cands[names[0]], turns[names[0]], "2015-01-01", "2026-09-30")
    m_sp = metrics(spy_r, zero, "2015-01-01", "2026-09-30")
    L.append(f"　串埋2015-2026：每年 {m_wf['cagr']:+.2f}%（跌 {m_wf['mdd']:.1f}%，Sharpe {m_wf['sharpe']:.2f}）｜ "
             f"T1線A {m_t1['cagr']:+.2f}%（跌 {m_t1['mdd']:.1f}%，Sharpe {m_t1['sharpe']:.2f}）｜ "
             f"SPY {m_sp['cagr']:+.2f}%（跌 {m_sp['mdd']:.1f}%，Sharpe {m_sp['sharpe']:.2f}）")
    pd.DataFrame({"report": L}).to_csv("v11_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
