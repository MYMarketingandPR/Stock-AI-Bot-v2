"""
research_v44.py — 第四十四輪：線E 揀股加「今年升幅上限」（Esther 2026-10-28：「我見你線E揀出嚟個股票都已經升≥100%」
  「今年30%-50%，30%-70%，30%-90%，30%-110% + 離52週最高位唔夠30% + 喺200日線上面先買（揀今年升最多）；
   跌穿200日線或者離52週高位超過30%就賣；揸最多5隻」）

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同設定同第四十一輪一樣：當時真實標普500、已計股息、收市決定→下一個交易日收市、每次買賣0.1%、揸5隻、閒錢收^IRX。
對照：線E（今年 ≥ +30%、≥ 52週最高收市 × 0.7、> 200日平均，揀今年升最多；跌穿200日平均或低過52週最高收市 × 0.7 就賣）。
版本（只有呢4個，跑完唔加唔減唔改數字）：只改「買」，今年升幅要喺範圍入面（包埋兩邊），其他同線E一樣：
  C50  今年 +30% 至 +50%
  C70  今年 +30% 至 +70%
  C90  今年 +30% 至 +90%
  C110 今年 +30% 至 +110%
  喺範圍入面揀今年升最多嘅。賣出規則同線E一樣；已經買咗之後再升過上限唔會因此賣（上限只係用嚟揀買邊隻）。
判斷（同第四十一輪一樣，同線E比；驗證期 2015-01 至 2026-09）：有改善、四條問題、跨年代合格、最差5年年化。
另外列：每次買賣統計；線E 買入嗰陣今年升幅嘅分佈。
就算合格都唔會直接改模擬交易，要你同意先加新線。

輸出：v44_results.csv、v44_progress.txt
==== 結果（2026-10-28 第一次跑，記錄低，唔再改）====
- 線E候選股今年升幅：30-50% 佔 67%、50-70% 20%、70-90% 6%、90-110% 3%、>110% 4%。
- 驗證期：C50 +11.36% / -29.3%；C70 +15.59% / -33.5%；C90 +26.89% / -31.3%；C110 +23.76% / -36.0%；線E +28.74% / -32.6% → 全部冇改善。
- C90 最接近：年化少 1.9 點、最大跌幅好 1.3 點（標準要好 3 點）；四條問題保留（同線E一樣），跨年代 ✘✔✔✔ 唔合格（同線E一樣）。
  但 2000-2009 +12.3%（線E +4.5%）、全期 +20.5%（線E +16.8%）、最差5年年化 -6.3%（線E -9.8%）、科網爆煲 -2.8%（線E -16.9%）。
- 上限唔係越高越好：C110 差過 C90（2020-26 +23.1% vs +31.2%）→ 結果對上限數字好敏感，C90 好可能有運氣成分。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim

M.PROGRESS = "v44_progress.txt"
log = M.log
END = "2026-09-30"


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    closes = M.download_close(everyone + ["SPY", "QQQ"], "1998-01-01", "2026-10-01")
    spy, qqq = closes.pop("SPY"), closes.pop("QQQ")
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    px = pd.DataFrame(closes).reindex(days)
    P = px.values
    rets_v = px.pct_change(fill_method=None).values
    ma200 = px.rolling(200).mean().values
    ye = px.groupby(days.year).last()
    ytdn = np.nan_to_num(P / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    hi52 = px.rolling(252, min_periods=200).max().values
    hi52f = np.nan_to_num(hi52, nan=np.inf)
    cash_r = get_cash(days)
    cr = cash_r.values
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = qqq.reindex(days).pct_change(fill_method=None).fillna(0.0)
    cols = list(px.columns)
    ix = {t: k for k, t in enumerate(cols)}
    mem = np.zeros(P.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True
    fin = np.isfinite(P)
    with np.errstate(invalid="ignore", divide="ignore"):
        d1 = P / ma200 - 1
        d2 = (P - ma200) / P
    e_ok = mem & (ytdn >= 0.30) & (P >= hi52f * 0.7) & np.nan_to_num(P > ma200, nan=False) & fin
    sel = e_ok[days >= pd.Timestamp("2000-01-01")]
    log("數據準備好")

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    V = {"線E（對照）": sim(P, rets_v, cr, e_ok, ytdn, broken)}
    for cap in (0.5, 0.7, 0.9, 1.1):
        V[f"C{int(round(cap * 100))} 今年 +30% 至 +{int(round(cap * 100))}%"] = sim(P, rets_v, cr, e_ok & (ytdn <= cap), ytdn, broken)
    log("全部版本完成")

    sel_y = ytdn[days >= pd.Timestamp("2000-01-01")][sel]
    L = ["=== 第四十四輪：線E + 今年升幅上限（規則預先登記，見research_v44.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻", "",
         "線E候選股（2000年起，股票×日）今年升幅分佈：" + " ｜ ".join(
             f"{lo * 100:.0f}%-{hi * 100:.0f}% 佔 {((sel_y >= lo) & (sel_y < hi)).mean() * 100:.0f}%"
             for lo, hi in [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 99)]), ""]
    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31"),
               ("2025年", "2025-01-01", "2025-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)
    tab = {}
    for wn, a_, b_ in windows:
        L.append(f"=== {wn}（{a_} 至 {b_}）===")
        for nm, r in allser.items():
            rr = r.loc[a_:b_].dropna()
            p = perf(rr, rf=cash_r, bench=qqq_r.loc[a_:b_])
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
            d1_ = d0 + pd.DateOffset(years=5)
            if d1_ > nav.index[-1]:
                break
            j0, j1 = nav.index.get_loc(d0), nav.index.searchsorted(d1_) - 1
            out.append((nav.iat[j1] / nav.iat[j0]) ** (1 / 5) - 1)
        return min(out) * 100

    eras = ["2000-2009", "2010-2014", "2015-2019", "2020-2026"]
    q, s = tab[("驗證期", "QQQ長揸")], tab[("驗證期", "SPY長揸")]
    b0 = tab[("驗證期", "線E（對照）")]
    L.append(f"=== 預先登記判斷 ｜ 線E {b0['cagr']:+.2f}% / {b0['mdd']:.1f}% ｜ QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% / Sharpe {q['sharpe']:.2f} ｜ SPY {s['cagr']:+.2f}% ===")
    for nm in V:
        x, e = tab[("驗證期", nm)], tab[("2000-2009", nm)]
        marks = "".join("✔" if (tab[(er, nm)]["cagr"] >= tab[(er, "SPY長揸")]["cagr"] and tab[(er, nm)]["mdd"] > -35) else "✘" for er in eras)
        imp = "—"
        if "對照" not in nm:
            better = (x["mdd"] >= b0["mdd"] + 3 and x["cagr"] >= b0["cagr"] - 1) or (x["cagr"] >= b0["cagr"] + 2 and x["mdd"] >= b0["mdd"] - 2)
            imp = "有改善" if better else "冇改善"
        q1 = x["cagr"] > q["cagr"]
        q2 = q1 and x["mdd"] > -35 and x["sharpe"] >= q["sharpe"]
        q3 = (not q1) and x["mdd"] >= q["mdd"] + 5 and x["cagr"] >= s["cagr"]
        q4 = e["mdd"] > -35
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} ｜ 同線E比 {imp} ｜ "
                 f"四條問題 ①{'✔' if q1 else '✘'} ②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'} ⇒ "
                 f"{'保留' if (q2 or q3) else '淘汰'} ｜ 跨年代 {marks} {'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 "
                 + " / ".join(f"{tab[(er, nm)]['cagr']:+.1f}%" for er in eras) + " ｜ 各段最大跌幅 "
                 + " / ".join(f"{tab[(er, nm)]['mdd']:.0f}%" for er in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
    pd.DataFrame({"report": L}).to_csv("v44_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
