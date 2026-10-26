"""
research_v41b.py — 第四十一B輪：線E + 「200日線只係股價 38% 或以下」就唔買／已買就賣（Esther 2026-10-26 原意，
  靈感嚟自 MRNA 2021-08-10：高位 $497.49、200日線 $171.6）

純研究，唔改任何模擬交易，唔改之前任何結果（包括第四十一輪）。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同設定同第四十一輪一樣。比例 r = 200日平均 ÷ 收市。r ≤ 0.38（即係股價 ≥ 200日線 × 2.63）= 「升到好盡」。
版本（只有呢一個）：
  G1 r ≤ 0.38 唔買 + 已揸 r ≤ 0.38 就賣（止賺）；回落到 r > 0.38 而且符合線E條件先可以再買（唔會即日賣完即刻買返）。
另外列：線E候選股入面 r ≤ 0.38 佔幾多；歷史上觸發止賺嘅次數同股票。
判斷同第四十一輪一樣。就算合格都唔會直接改模擬交易。

輸出：v41b_results.csv、v41b_progress.txt
==== 結果（2026-10-26 第一次跑，記錄低，唔再改）====
- 線E候選股入面 r ≤ 0.38 只佔 0.11%（極少見）。2000年起曾經出現過嘅成分股 15 隻：ORCL、QCOM、NTAP、ADI、AMD（2000年科網）、
  EP（2001）、GNW（2009）、MRNA（2021-08-09）、PLTR（2025-02）、SNDK（2025-12）、LITE、INTC、MU、DELL、WDC（2026-04至06）。
- 驗證期：G1 +25.82% / -28.7% / Sharpe 0.86；線E +28.74% / -32.6% / Sharpe 0.90 → 最大跌幅好咗 3.9 點，但年化少咗 2.9 點
  （標準要年化唔少過 1 點）→ 冇改善。四條問題保留（同線E一樣）；跨年代 ✘✔✔✔ 唔合格（同線E一樣）。
- 2020-26：年化 +27.0%（線E +32.2%），最大跌幅 -28%（線E -33%）；2025年 +26.5%（線E +20.0%）。
- 每次買賣同線E差唔多（282 vs 279 次；最好一單一樣 +669%）。
- 意思：呢個「升到好盡」嘅訊號好少出現，出現嗰陣止賺可以減少跌幅，但亦會賣早咗部分仲繼續升嘅股；2025-26 年一次過有好多隻 AI／記憶體股觸發，
  上一次咁多隻同時觸發係 2000 年科網泡沫。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim

M.PROGRESS = "v41b_progress.txt"
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
    far1 = np.nan_to_num(ma200 / P <= 0.38, nan=False)      # 200日線 ≤ 股價 38%
    far2 = far1
    e_ok = mem & (ytdn >= 0.30) & (P >= hi52f * 0.7) & np.nan_to_num(P > ma200, nan=False) & fin
    sel = e_ok[days >= pd.Timestamp("2000-01-01")]
    log("數據準備好")

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    def plus(far):
        return lambda i, c, p, pk: broken(i, c, p, pk) or bool(far[i, c])

    V = {"線E（對照）": sim(P, rets_v, cr, e_ok, ytdn, broken),
         "G1 200日線≤股價38%：唔買＋止賺": sim(P, rets_v, cr, e_ok & ~far1, ytdn, plus(far1))}
    hits = np.argwhere(far1 & mem)
    first = {}
    for i_, c_ in hits:
        t_ = cols[c_]
        if days[i_] >= pd.Timestamp("2000-01-01"):
            first.setdefault(t_, days[i_].date())
    log("全部版本完成")

    L = ["=== 第四十一B輪：線E + 200日線只係股價38%就唔買／止賺（規則預先登記，見research_v41b.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻", "",
         f"線E候選股（2000年起，股票×日）入面：200日線 ≤ 股價38% 佔 {far1[days >= pd.Timestamp('2000-01-01')][sel].mean() * 100:.2f}%",
         f"2000年起曾經出現過嘅成分股：{len(first)} 隻 — " + "、".join(f"{k}（{v}）" for k, v in sorted(first.items(), key=lambda x: x[1])), ""]
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
    pd.DataFrame({"report": L}).to_csv("v41b_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
