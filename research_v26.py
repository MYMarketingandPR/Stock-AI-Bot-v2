"""
research_v26.py — 第二十六輪：Esther 揀股條件「今年升 ≥30% + 離最高位唔夠30% + 喺MA200上面」（2026-10-08）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同：當時真實標普500；已計股息；收市決定 → 下一個交易日收市買賣；每次買賣0.1%；冇投資嘅錢揸現金（^IRX）；
  每隻 = 買入日總資產 ÷ 揸幾多隻；賣咗之後再符合條件可以再買。
揀股條件（新）：
  ① 年初至今升幅 ≥ +30%（今日收市 ÷ 上年最後一個交易日收市 − 1）
  ② 離52週最高收市價唔夠30%（今日收市 ≥ 過去252個交易日最高收市價 × 0.7）
  ③ 收市 > 200日平均
  有空位揀「年初至今升幅」最大嘅。（1月初好少股票會符合，係正常。）
版本：
  Y1 揸5隻，由最高收市價跌20%就賣（同線D一樣嘅賣法）
  Y2 Y1 + SPY每日訊號持有先開新倉（同線D2一樣）
  Y3 揸5隻，條件唔再符合就賣：收市 < MA200，或者離52週最高位跌超過30%
  Y4 揸10隻，由最高收市價跌20%就賣
對照：T0 線D（12個月升 ≥100% + MA50 + MA200，揸5隻，跌20%賣）、QQQ、SPY。
判斷（同之前一樣）：
  ① 「有改善」（同 T0 比，驗證期 2015-01 至 2026-09）
  ② 四條問題（驗證期贏唔贏QQQ等）
  ③ 跨年代合格：2000-2009、2010-2014、2015-2019、2020-2026 每段年化 ≥ SPY 而且最大跌幅好過 -35%
  另外列：每日平均有幾多隻符合條件、每次買賣統計、最差5年年化。4個版本。
輸出：v26_results.csv、v26_progress.txt
==== 結果（2026-10-08 第一次跑，記錄低，唔再改）====
- 核對：T0 +21.38% / -33.8%。每日平均36隻符合（1月平均1隻，12月83隻；8%日子一隻都冇）。
- Y3 條件唔再符合就賣：驗證期 +28.74% / -32.6% / Sharpe 0.90 → 「有改善」、四條問題保留；
  但 2000-2009 +4.0% / -64%、金融海嘯 -60%、最差5年 -11.2% → 跨年代唔合格（✘✔✔✔）。
- Y4 揸10隻：+17.67% / -27.9% → 同線D比冇改善，但四條問題③保留；最差5年 -2.4%（所有版本最好）；2000-2009 +10.2% / -49%。
- Y2 加SPY訊號：驗證期 +16.28% / -40.6%（2015-2019 只有 +2.3%）；但 2000-2009 +14.2% / -41%（嗰十年最好）。
- Y1 揸5隻跌20%賣：+15.03% / -37.0% → 淘汰。
- 跨年代合格：0個。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash

M.PROGRESS = "v26_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"


def sim(P, rets_v, cr, ok, rank, sell_fn, n=5, allow=None):
    T = P.shape[0]
    hold, info, cash = {}, {}, 1.0
    ps, pb, navs, trades = [], [], [], []
    for i in range(T):
        for c in list(hold):
            r = rets_v[i, c]
            if np.isfinite(r):
                hold[c] *= 1 + r
        cash *= 1 + cr[i]
        for c in ps:
            if c in hold:
                v = hold.pop(c)
                cash += v * (1 - COST)
                b = info.pop(c)
                sp = P[i, c] if np.isfinite(P[i, c]) else b["last"]
                trades.append(sp / b["px"] - 1)
        total = cash + sum(hold.values())
        for c in pb:
            if c in hold or len(hold) >= n or not np.isfinite(P[i, c]):
                continue
            amt = min(total / n, cash)
            if amt <= total * 0.01:
                break
            cash -= amt
            hold[c] = amt * (1 - COST)
            info[c] = {"px": float(P[i, c]), "peak": float(P[i, c]), "last": float(P[i, c])}
        ps, pb = [], []
        navs.append(cash + sum(hold.values()))
        for c in hold:
            p = P[i, c]
            if not np.isfinite(p):
                if i + 1 < T and not np.isfinite(P[i + 1, c]):
                    ps.append(c)
                continue
            info[c]["last"] = float(p)
            info[c]["peak"] = max(info[c]["peak"], float(p))
            if sell_fn(i, c, float(p), info[c]["peak"]):
                ps.append(c)
        if allow is not None and not allow[i]:
            continue
        free = n - (len(hold) - len(ps))
        if free > 0:
            cand = [c for c in np.where(ok[i])[0] if c not in hold and c not in ps]
            cand.sort(key=lambda c: -rank[i, c])
            pb = cand[:free]
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1], trades


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    closes = M.download_close(everyone + ["SPY", "QQQ"], "1998-01-01", "2026-10-01")
    spy, qqq = closes.pop("SPY"), closes.pop("QQQ")
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    px = pd.DataFrame(closes).reindex(days)
    cols = list(px.columns)
    P = px.values
    rets_v = px.pct_change(fill_method=None).values
    ma50, ma200 = px.rolling(50).mean().values, px.rolling(200).mean().values
    r12n = np.nan_to_num((px / px.shift(252) - 1).values, nan=-9)
    ye = px.groupby(days.year).last()                       # 每年最後收市價
    prev_ye = ye.shift(1).reindex(days.year).values         # 上年最後收市價
    ytd = P / prev_ye - 1
    ytdn = np.nan_to_num(ytd, nan=-9)
    hi52 = px.rolling(252, min_periods=200).max().values
    cash_r = get_cash(days)
    cr = cash_r.values
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = qqq.reindex(days).pct_change(fill_method=None).fillna(0.0)
    ix = {t: k for k, t in enumerate(cols)}
    mem = np.zeros(P.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True
    fin = np.isfinite(P)
    base_ok = mem & (r12n >= 1.0) & (P > ma50) & (P > ma200) & fin
    y_ok = mem & (ytdn >= 0.30) & (P >= np.nan_to_num(hi52, nan=np.inf) * 0.7) & (P > ma200) & fin
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    sig_ok = (df["state"] != "out").reindex(days).fillna(False).values
    log("數據準備好")

    def trail20(i, c, p, pk):
        return p <= pk * 0.8

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    V = {"T0 線D（對照）": sim(P, rets_v, cr, base_ok, r12n, trail20),
         "Y1 新條件・揸5隻・跌20%賣": sim(P, rets_v, cr, y_ok, ytdn, trail20),
         "Y2 新條件・揸5隻・跌20%賣・SPY訊號持有先開新倉": sim(P, rets_v, cr, y_ok, ytdn, trail20, allow=sig_ok),
         "Y3 新條件・揸5隻・條件唔再符合就賣": sim(P, rets_v, cr, y_ok, ytdn, broken),
         "Y4 新條件・揸10隻・跌20%賣": sim(P, rets_v, cr, y_ok, ytdn, trail20, n=10)}
    log("全部版本完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)
    tab = {}
    L = ["=== 第二十六輪：今年升≥30% + 離最高位<30% + 喺MA200上面（規則預先登記，見research_v26.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%", ""]
    cnt = pd.Series(y_ok.sum(axis=1), index=days).loc["2000-01-03":END]
    by_m = cnt.groupby(cnt.index.month).mean()
    L.append(f"每日符合新條件嘅股票：平均 {cnt.mean():.0f} 隻 ｜ 最少 {cnt.min()} 隻 ｜ 最多 {cnt.max()} 隻 ｜ 冇任何一隻嘅日子 {(cnt == 0).mean() * 100:.0f}%")
    L.append("　按月份平均：" + "、".join(f"{m}月 {v:.0f}" for m, v in by_m.items()))
    L.append("")
    for wn, a, b in windows:
        L.append(f"=== {wn}（{a} 至 {b}）===")
        for nm, r in allser.items():
            rr = r.loc[a:b].dropna()
            p = perf(rr, rf=cash_r, bench=qqq_r.loc[a:b])
            tab[(wn, nm)] = p
            L.append(f"　{nm}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}")
        L.append("")
    L.append("=== 每次買賣（1999-2026）===")
    for nm, v in V.items():
        tr = np.array(v[1])
        L.append(f"　{nm}：{len(tr)} 次 ｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
                 f"蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}%")
    L.append("")

    def worst5(r):
        nav = (1 + r.loc["2000-01-03":END].fillna(0.0)).cumprod()
        out = []
        for d0 in decision_days(nav.index, "每月"):
            d1 = d0 + pd.DateOffset(years=5)
            if d1 > nav.index[-1]:
                break
            j0, j1 = nav.index.get_loc(d0), nav.index.searchsorted(d1) - 1
            out.append((nav.iat[j1] / nav.iat[j0]) ** (1 / 5) - 1)
        return min(out) * 100

    eras = ["2000-2009", "2010-2014", "2015-2019", "2020-2026"]
    q, s = tab[("驗證期", "QQQ長揸")], tab[("驗證期", "SPY長揸")]
    b0 = tab[("驗證期", "T0 線D（對照）")]
    L.append(f"=== 預先登記判斷 ｜ T0 {b0['cagr']:+.2f}% / {b0['mdd']:.1f}%（應該 +21.38% / -33.8%）｜ QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% / Sharpe {q['sharpe']:.2f} ｜ SPY {s['cagr']:+.2f}% ===")
    for nm in V:
        x = tab[("驗證期", nm)]
        e = tab[("2000-2009", nm)]
        marks = "".join("✔" if (tab[(er, nm)]["cagr"] >= tab[(er, "SPY長揸")]["cagr"] and tab[(er, nm)]["mdd"] > -35) else "✘" for er in eras)
        imp = "—"
        if not nm.startswith("T0"):
            better = (x["mdd"] >= b0["mdd"] + 3 and x["cagr"] >= b0["cagr"] - 1) or (x["cagr"] >= b0["cagr"] + 2 and x["mdd"] >= b0["mdd"] - 2)
            imp = "有改善" if better else "冇改善"
        q1 = x["cagr"] > q["cagr"]
        q2 = q1 and x["mdd"] > -35 and x["sharpe"] >= q["sharpe"]
        q3 = (not q1) and x["mdd"] >= q["mdd"] + 5 and x["cagr"] >= s["cagr"]
        q4 = e["mdd"] > -35
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} ｜ 同線D比 {imp} ｜ "
                 f"四條問題 ①{'✔' if q1 else '✘'} ②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'} ⇒ "
                 f"{'保留' if (q2 or q3) else '淘汰'} ｜ 跨年代 {marks} {'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 "
                 + " / ".join(f"{tab[(er, nm)]['cagr']:+.1f}%" for er in eras) + " ｜ 各段最大跌幅 "
                 + " / ".join(f"{tab[(er, nm)]['mdd']:.0f}%" for er in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
    pd.DataFrame({"report": L}).to_csv("v26_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
