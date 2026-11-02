"""
research_v46.py — 第四十六輪：RSI「5日RSI跌穿10買、RSI 90以上賣」（Esther 2026-11-02，靈感嚟自 Money錢 短片：
  「RSI只要5日有跌破10的時候（買）」「RSI > 90，90幾以上，包括10日（賣）」）
  Esther 指定：1. 用 SPY 同全部標普500個股；2.「10日RSI高過90」同「RSI高過90連續10日」兩個都試；3. 由最高位跌20%止蝕。

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
RSI 計法：Wilder（同第三十六輪、spy_indicators 一樣）。RSI(n) 用收市價（已計股息）。
買：5日RSI < 10（收市知道，下一個交易日收市買）。
賣（邊個先到用邊個，收市知道，下一個交易日收市賣）：
  版本A：10日RSI > 90
  版本B：10日RSI 連續10個交易日都 > 90（為咗同版本A只差「連續10日」，兩個都用10日RSI）
  兩個版本都加：收市 ≤ 買入後最高收市 × 0.8（由最高位跌20%）就賣。
每次買賣0.1%；唔用槓桿；閒錢收^IRX。

第一部分：SPY（1994-01 至 2026-09）
  SPY-A、SPY-B：有訊號就全部買SPY，賣咗就揸現金。
  對照：SPY長揸。
  判斷：同SPY長揸比「有改善」= 最大跌幅好3點或以上而年化唔少過1點以上；或者年化多2點或以上而最大跌幅唔差過2點。
  分段列：1994-2012、2013-2026、2015-2026（同個股驗證期一樣）、2000-2009、2008、2020、2022。

第二部分：標普500個股（當時真實成分股，1999-2026，同第四十一輪一樣嘅設定）
  ST-A、ST-B：揸最多5隻，每隻 = 總資產 ÷ 5；同一日太多隻合資格就揀5日RSI最低嘅。
  對照：線E。判斷同第四十一輪一樣（驗證期 2015-01 至 2026-09；有改善、四條問題、跨年代合格、最差5年年化）。
另外列：每次買賣統計、賣出原因（RSI 定 止蝕）次數。
就算合格都唔會直接改模擬交易，要你同意先加新線。

輸出：v46_results.csv、v46_progress.txt
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim
from research_v36 import rsi

M.PROGRESS = "v46_progress.txt"
log = M.log
END = "2026-09-30"


def judge_line(tab, nm, b0, q, s, eras, worst):
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
    return (f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} ｜ 同對照比 {imp} ｜ "
            f"四條問題 ①{'✔' if q1 else '✘'} ②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'} ⇒ "
            f"{'保留' if (q2 or q3) else '淘汰'} ｜ 跨年代 {marks} {'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 "
            + " / ".join(f"{tab[(er, nm)]['cagr']:+.1f}%" for er in eras) + " ｜ 各段最大跌幅 "
            + " / ".join(f"{tab[(er, nm)]['mdd']:.0f}%" for er in eras) + f" ｜ 最差5年年化 {worst:+.1f}%")


def worst5(r, start="2000-01-03"):
    nav = (1 + r.loc[start:END].fillna(0.0)).cumprod()
    out = []
    for d0 in decision_days(nav.index, "每月"):
        d1_ = d0 + pd.DateOffset(years=5)
        if d1_ > nav.index[-1]:
            break
        j0, j1 = nav.index.get_loc(d0), nav.index.searchsorted(d1_) - 1
        out.append((nav.iat[j1] / nav.iat[j0]) ** (1 / 5) - 1)
    return min(out) * 100


def make_sell(r10, streak, reasons):
    def fn(i, c, p, pk):
        hit_rsi = bool(streak[i, c]) if streak is not None else bool(np.isfinite(r10[i, c]) and r10[i, c] > 90)
        hit_stop = p <= pk * 0.8
        if hit_rsi:
            reasons["RSI"] += 1
        elif hit_stop:
            reasons["止蝕"] += 1
        return hit_rsi or hit_stop
    return fn


def trade_line(nm, v, reasons=None):
    tr = np.array(v[1])
    if len(tr) == 0:
        return f"　{nm}：冇買賣"
    return (f"　{nm}：{len(tr)} 次 ｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
            f"賺嘅平均 {tr[tr > 0].mean() * 100 if (tr > 0).any() else 0:+.1f}% ｜ 蝕嘅平均 {tr[tr <= 0].mean() * 100 if (tr <= 0).any() else 0:+.1f}% ｜ "
            f"最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}%"
            + (f" ｜ 賣出原因：RSI {reasons['RSI']} 次、止蝕 {reasons['止蝕']} 次" if reasons else ""))


def main():
    open(M.PROGRESS, "w").close()
    L = ["=== 第四十六輪：5日RSI<10買、RSI>90賣、高位跌20%止蝕（規則預先登記，見research_v46.py開頭）===", ""]

    # ---------------- 第一部分：SPY ----------------
    sc = M.download_close(["SPY"], "1993-01-01", "2026-10-01")["SPY"]
    sc = sc[sc.index <= pd.Timestamp(END)]
    sd = sc.index
    Ps = sc.values.reshape(-1, 1)
    rs = sc.pct_change().fillna(0.0).values.reshape(-1, 1)
    cr_s = get_cash(sd)
    r5s, r10s = rsi(sc, 5).values.reshape(-1, 1), rsi(sc, 10).values.reshape(-1, 1)
    st10s = (pd.Series(r10s[:, 0]).rolling(10).min() > 90).values.reshape(-1, 1)
    buy_s = np.nan_to_num(r5s < 10, nan=False) & (np.arange(len(sd)) >= 30)[:, None]
    rank_s = np.zeros_like(Ps)
    reasons = {"SPY-A": {"RSI": 0, "止蝕": 0}, "SPY-B": {"RSI": 0, "止蝕": 0}}
    VS = {"SPY-A 10日RSI>90賣": sim(Ps, rs, cr_s.values, buy_s, rank_s, make_sell(r10s, None, reasons["SPY-A"]), n=1),
          "SPY-B 10日RSI連續10日>90賣": sim(Ps, rs, cr_s.values, buy_s, rank_s, make_sell(r10s, st10s, reasons["SPY-B"]), n=1)}
    L += ["=== 第一部分：SPY（1994-2026）===",
          f"5日RSI<10 嘅日子：{int(buy_s.sum())} 日（佔 {buy_s.mean() * 100:.1f}%）｜ 10日RSI>90：{int(np.nan_to_num(r10s > 90).sum())} 日 ｜ "
          f"10日RSI 連續10日>90：{int(st10s.sum())} 日", ""]
    ser = {"SPY長揸": pd.Series(rs[:, 0], index=sd)}
    for nm, v in VS.items():
        ser[nm] = pd.Series(v[0], index=sd)
    tabs = {}
    for wn, a_, b_ in [("全期", "1994-01-03", END), ("1994-2012", "1994-01-03", "2012-12-31"), ("2013-2026", "2013-01-01", END),
                       ("2015-2026", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2008年", "2008-01-01", "2008-12-31"),
                       ("2020年", "2020-01-01", "2020-12-31"), ("2022年", "2022-01-01", "2022-12-31")]:
        L.append(f"【{wn}】")
        for nm, r in ser.items():
            p = perf(r.loc[a_:b_], rf=cr_s)
            tabs[(wn, nm)] = p
            L.append(f"　{nm}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}")
    L.append("")
    for nm, v in VS.items():
        k = "SPY-A" if nm.startswith("SPY-A") else "SPY-B"
        L.append(trade_line(nm, v, reasons[k]))
    L.append("")
    L.append("判斷（同SPY長揸比，全期 1994-2026；另外列兩段）：")
    b0 = tabs[("全期", "SPY長揸")]
    for nm in VS:
        x = tabs[("全期", nm)]
        better = (x["mdd"] >= b0["mdd"] + 3 and x["cagr"] >= b0["cagr"] - 1) or (x["cagr"] >= b0["cagr"] + 2 and x["mdd"] >= b0["mdd"] - 2)
        both = all(tabs[(w, nm)]["cagr"] >= tabs[(w, "SPY長揸")]["cagr"] for w in ["1994-2012", "2013-2026"])
        L.append(f"　{nm}：全期 {x['cagr']:+.2f}% / {x['mdd']:.1f}%（SPY長揸 {b0['cagr']:+.2f}% / {b0['mdd']:.1f}%）⇒ {'有改善' if better else '冇改善'} ｜ "
                 f"前後兩段年化都唔輸SPY：{'係' if both else '唔係'}")
    L.append("")
    log("SPY 完成")

    # ---------------- 第二部分：個股 ----------------
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
    ma200 = px.rolling(200).mean().values
    ye = px.groupby(days.year).last()
    ytdn = np.nan_to_num(P / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    hi52 = px.rolling(252, min_periods=200).max().values
    hi52f = np.nan_to_num(hi52, nan=np.inf)
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
    e_ok = mem & (ytdn >= 0.30) & (P >= hi52f * 0.7) & np.nan_to_num(P > ma200, nan=False) & fin
    r5 = rsi(px, 5).values
    r10 = rsi(px, 10).values
    st10 = (pd.DataFrame(r10).rolling(10).min() > 90).values
    buy = mem & fin & np.nan_to_num(r5 < 10, nan=False)
    log("個股數據準備好")

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    rsn = {"ST-A": {"RSI": 0, "止蝕": 0}, "ST-B": {"RSI": 0, "止蝕": 0}}
    V = {"線E（對照）": sim(P, rets_v, cr, e_ok, ytdn, broken),
         "ST-A 10日RSI>90賣": sim(P, rets_v, cr, buy, -np.nan_to_num(r5, nan=100), make_sell(r10, None, rsn["ST-A"])),
         "ST-B 10日RSI連續10日>90賣": sim(P, rets_v, cr, buy, -np.nan_to_num(r5, nan=100), make_sell(r10, st10, rsn["ST-B"]))}
    log("個股模擬完成")
    sel = mem[days >= pd.Timestamp("2000-01-01")] & fin[days >= pd.Timestamp("2000-01-01")]
    L += ["=== 第二部分：標普500個股 ===", f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻",
          f"成分股日子（2000年起）：5日RSI<10 佔 {buy[days >= pd.Timestamp('2000-01-01')][sel].mean() * 100:.2f}%；"
          f"10日RSI>90 佔 {np.nan_to_num(r10[days >= pd.Timestamp('2000-01-01')] > 90)[sel].mean() * 100:.2f}%；"
          f"連續10日>90 佔 {st10[days >= pd.Timestamp('2000-01-01')][sel].mean() * 100:.3f}%", ""]
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
            p = perf(r.loc[a_:b_].dropna(), rf=cash_r, bench=qqq_r.loc[a_:b_])
            tab[(wn, nm)] = p
            L.append(f"　{nm}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}")
        L.append("")
    L.append("=== 每次買賣（1999-2026）===")
    for nm, v in V.items():
        k = "ST-A" if nm.startswith("ST-A") else "ST-B" if nm.startswith("ST-B") else None
        L.append(trade_line(nm, v, rsn[k] if k else None))
    L.append("")
    eras = ["2000-2009", "2010-2014", "2015-2019", "2020-2026"]
    q, s = tab[("驗證期", "QQQ長揸")], tab[("驗證期", "SPY長揸")]
    b0 = tab[("驗證期", "線E（對照）")]
    L.append(f"=== 預先登記判斷 ｜ 線E {b0['cagr']:+.2f}% / {b0['mdd']:.1f}% ｜ QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% / Sharpe {q['sharpe']:.2f} ｜ SPY {s['cagr']:+.2f}% ===")
    for nm in V:
        L.append(judge_line(tab, nm, b0, q, s, eras, worst5(allser[nm])))
    pd.DataFrame({"report": L}).to_csv("v46_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
