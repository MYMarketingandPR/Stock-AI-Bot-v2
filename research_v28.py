"""
research_v28.py — 第二十八輪：新條件 + RS ≥ 80，由最高位跌20%就賣（Esther 2026-10-08：「試吓新條件 + RS ≥80, 跌 20% 賣」）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同同第二十七輪一樣（當時真實標普500、已計股息、收市決定→下一個交易日收市、0.1%、揸5隻平均分、冇投資揸現金）。
新條件：過去12個月升 +30% 至 +200% + 收市 ≥ 52週最高收市價 × 0.7 + 收市 > MA200 + RS Rating ≥ 80（揀RS最高）。
Z7 賣出：由買入後最高收市價跌20%（同線D一樣）。
對照：T0 線D、線E、Z4（同樣買入，但「條件唔再符合就賣」，第二十七輪 +25.93% / -43.2%）。
判斷：同第二十七輪一樣（有改善同線D比、四條問題、跨年代合格）。只得1個新版本。
輸出：v28_results.csv、v28_progress.txt
==== 結果（2026-10-08 第一次跑，記錄低，唔再改）====
- 核對：T0 +21.38% / -33.8%、線E +28.74% / -32.6%、Z4 +25.93% / -43.2%，同之前一樣。
- Z7 新條件+RS≥80・跌20%賣：驗證期 +15.92% / -42.7% / Sharpe 0.65 → 同線D比冇改善、四條問題淘汰、跨年代 ✘✔✘✔ 唔合格。
  各段：2000-2009 +2.0% / -75%；2010-2014 +27.2% / -15%（呢段最好）；2015-2019 +12.6% / -36%；2020-2026 +18.5% / -31%。
  金融海嘯 -47% / -74%；最差5年 -16.8%。跌20%賣喺呢個條件下比「條件唔再符合就賣」差（+15.9% vs +25.9%）。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v23 import rs_rating
from research_v26 import sim

M.PROGRESS = "v28_progress.txt"
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
    cols = list(px.columns)
    P = px.values
    rets_v = px.pct_change(fill_method=None).values
    ma50, ma200 = px.rolling(50).mean().values, px.rolling(200).mean().values
    r12n = np.nan_to_num((px / px.shift(252) - 1).values, nan=-9)
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
    rs = rs_rating(px, mem)
    rsn = np.nan_to_num(rs, nan=0)
    base_ok = mem & (r12n >= 1.0) & (P > ma50) & (P > ma200) & fin
    e_ok = mem & (ytdn >= 0.30) & (P >= hi52f * 0.7) & (P > ma200) & fin
    z_ok = mem & (r12n >= 0.30) & (r12n <= 2.0) & (P >= hi52f * 0.7) & (P > ma200) & fin
    zrs_ok = z_ok & (rsn >= 80)
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    sig_ok = (df["state"] != "out").reindex(days).fillna(False).values
    log("數據準備好（包括RS Rating）")

    def trail20(i, c, p, pk):
        return p <= pk * 0.8

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    def broken_rs(i, c, p, pk):
        r = rs[i, c]
        return broken(i, c, p, pk) or (np.isfinite(r) and r < 70)

    V = {"T0 線D（對照）": sim(P, rets_v, cr, base_ok, r12n, trail20),
         "線E（對照）": sim(P, rets_v, cr, e_ok, ytdn, broken),
         "Z4 新條件+RS≥80・條件唔再符合就賣（對照）": sim(P, rets_v, cr, zrs_ok, rsn, broken),
         "Z7 新條件+RS≥80・跌20%賣": sim(P, rets_v, cr, zrs_ok, rsn, trail20)}
    log("全部版本完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)
    tab = {}
    L = ["=== 第二十八輪：新條件 + RS≥80，跌20%賣（規則預先登記，見research_v28.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻", ""]
    c1 = pd.Series(z_ok.sum(axis=1), index=days).loc["2000-01-03":END]
    c2 = pd.Series(zrs_ok.sum(axis=1), index=days).loc["2000-01-03":END]
    L.append(f"每日符合基本條件：平均 {c1.mean():.0f} 隻（冇任何一隻嘅日子 {(c1 == 0).mean() * 100:.0f}%）｜ 加 RS≥80：平均 {c2.mean():.0f} 隻（冇嘅日子 {(c2 == 0).mean() * 100:.0f}%）")
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
        x, e = tab[("驗證期", nm)], tab[("2000-2009", nm)]
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
    pd.DataFrame({"report": L}).to_csv("v28_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
