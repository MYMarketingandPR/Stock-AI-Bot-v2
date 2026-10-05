"""
trades_ma200.py — 重播第十五輪「揸5隻・跌穿MA200賣」（當時真實標普500），列出每一次買賣。
規則同 research_v15.py 第一部分一模一樣：
  買入：當月標普500成分股 + 過去252日升 ≥ +100% + 收市 > MA50 + 收市 > MA200；有空位揀12個月升幅最大；
  賣出：收市跌穿MA200（或者冇價 = 除牌，用最後價）；收市決定、下一日收市執行；每次買賣0.1%；最多5隻。
輸出：ma200_trades.csv（每次買賣）、ma200_summary.csv（摘要）、ma200_progress.txt
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v13 import get_cash

M.PROGRESS = "ma200_progress.txt"
COST = 0.001
END = "2026-09-30"
N = 5


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    closes = M.download_close(everyone + ["SPY", "QQQ"], "1998-01-01", "2026-10-01")
    spy = closes.pop("SPY")
    qqq = closes.pop("QQQ")
    spy = spy[spy.index <= pd.Timestamp(END)]
    px = pd.DataFrame(closes).reindex(spy.index)
    days, cols = px.index, list(px.columns)
    P = px.values
    rets = px.pct_change(fill_method=None).values
    ma50, ma200 = px.rolling(50).mean().values, px.rolling(200).mean().values
    r12 = (px / px.shift(252) - 1).values
    cr = get_cash(days).values
    ix = {t: k for k, t in enumerate(cols)}
    mem = np.zeros(P.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True
    entry_ok = mem & (r12 >= 1.0) & (P > ma50) & (P > ma200) & np.isfinite(P)

    hold, info, cash = {}, {}, 1.0
    ps, pb, navs, trades = [], [], [], []
    for i in range(len(days)):
        for c in list(hold):
            r = rets[i, c]
            if np.isfinite(r):
                hold[c] *= 1 + r
        cash *= 1 + cr[i]
        for c, why in ps:
            if c in hold:
                v = hold.pop(c)
                cash += v * (1 - COST)
                b = info.pop(c)
                sp = P[i, c] if np.isfinite(P[i, c]) else b["last"]
                trades.append({"股票": cols[c], "買入日": days[b["i"]].date(), "買入價": round(b["px"], 2),
                               "賣出日": days[i].date(), "賣出價": round(float(sp), 2),
                               "回報%": round((sp / b["px"] - 1) * 100, 1), "持有日數": (days[i] - days[b["i"]]).days,
                               "買入時12個月升幅%": round(b["r12"] * 100, 0), "賣出原因": why})
        total = cash + sum(hold.values())
        for c in pb:
            if c in hold or len(hold) >= N or not np.isfinite(P[i, c]):
                continue
            amt = min(total / N, cash)
            if amt <= total * 0.01:
                break
            cash -= amt
            hold[c] = amt * (1 - COST)
            info[c] = {"i": i, "px": float(P[i, c]), "r12": float(r12[i - 1, c]), "last": float(P[i, c])}
        ps, pb = [], []
        navs.append(cash + sum(hold.values()))
        for c in hold:
            p = P[i, c]
            if np.isfinite(p):
                info[c]["last"] = float(p)
            if not np.isfinite(p):
                if i + 1 < len(days) and not np.isfinite(P[i + 1, c]):
                    ps.append((c, "冇價（除牌/收購）"))
                continue
            if np.isfinite(ma200[i, c]) and p < ma200[i, c]:
                ps.append((c, "跌穿MA200"))
        free = N - (len(hold) - len(ps))
        if free > 0:
            cand = [c for c in np.where(entry_ok[i])[0] if c not in hold]
            cand.sort(key=lambda c: -r12[i, c])
            pb = cand[:free]
    for c in hold:      # 仲揸緊
        b = info[c]
        trades.append({"股票": cols[c], "買入日": days[b["i"]].date(), "買入價": round(b["px"], 2), "賣出日": "仲揸緊",
                       "賣出價": round(b["last"], 2), "回報%": round((b["last"] / b["px"] - 1) * 100, 1),
                       "持有日數": (days[-1] - days[b["i"]]).days, "買入時12個月升幅%": round(b["r12"] * 100, 0),
                       "賣出原因": "—"})
    T = pd.DataFrame(trades).sort_values("買入日")
    T.to_csv("ma200_trades.csv", index=False, encoding="utf-8-sig")
    r = pd.Series(navs, index=days).pct_change().fillna(0.0)
    q = qqq.reindex(days).pct_change(fill_method=None).fillna(0.0)
    L = [f"重播：揸5隻・跌穿MA200賣（{source}）"]
    for a, b in (("2015-01-01", END), ("2000-01-01", "2009-12-31"), ("2000-01-01", END)):
        p, pq = perf(r.loc[a:b]), perf(q.loc[a:b])
        L.append(f"{a[:4]}-{b[:4]}：年化 {p['cagr']:+.2f}%、最大跌幅 {p['mdd']:.1f}%（QQQ {pq['cagr']:+.2f}%、{pq['mdd']:.1f}%）")
    done = T[T["賣出日"] != "仲揸緊"]
    for lab, sub in (("全部", done), ("2015年起買入", done[pd.to_datetime(done["買入日"]) >= "2015-01-01"])):
        L.append(f"{lab}：{len(sub)} 次買賣，賺錢 {(sub['回報%'] > 0).mean() * 100:.0f}%，每次平均 {sub['回報%'].mean():+.1f}%、"
                 f"中位數 {sub['回報%'].median():+.1f}%，平均持有 {sub['持有日數'].mean():.0f} 日，"
                 f"賺嘅平均 {sub[sub['回報%'] > 0]['回報%'].mean():+.1f}%、蝕嘅平均 {sub[sub['回報%'] <= 0]['回報%'].mean():+.1f}%")
    L.append("賺最多10次：" + "；".join(f"{x['股票']} {x['買入日']}→{x['賣出日']} {x['回報%']:+.0f}%" for _, x in done.nlargest(10, "回報%").iterrows()))
    L.append("蝕最多10次：" + "；".join(f"{x['股票']} {x['買入日']}→{x['賣出日']} {x['回報%']:+.0f}%" for _, x in done.nsmallest(10, "回報%").iterrows()))
    cnt = T["股票"].value_counts()
    L.append("買得最多次：" + "，".join(f"{k} {v}次" for k, v in cnt.head(12).items()))
    L.append(f"總共買過 {T['股票'].nunique()} 隻唔同股票")
    L.append("2026-09-30 仲揸緊：" + "；".join(f"{x['股票']}（{x['買入日']} 買，{x['回報%']:+.0f}%）" for _, x in T[T["賣出日"] == "仲揸緊"].iterrows()))
    pd.DataFrame({"摘要": L}).to_csv("ma200_summary.csv", index=False, encoding="utf-8-sig")
    print("\n".join(L))


if __name__ == "__main__":
    main()
