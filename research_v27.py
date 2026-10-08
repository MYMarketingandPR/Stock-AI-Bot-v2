"""
research_v27.py — 第二十七輪：Esther 揀股條件「過去12個月升30%至200% + 離最高位唔夠30% + 喺MA200上面」，
加／唔加 SPY訊號同 RS Rating（買入同賣出）（2026-10-08）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同：當時真實標普500；已計股息；收市決定 → 下一個交易日收市買賣；每次買賣0.1%；冇投資嘅錢揸現金（^IRX）；
  揸最多5隻、每隻 = 買入日總資產 ÷ 5；賣咗之後再符合條件可以再買。
基本揀股條件：① 過去12個月（252個交易日）升 +30% 至 +200%（包兩頭）；② 收市 ≥ 52週最高收市價 × 0.7；③ 收市 > MA200。
RS Rating：同第二十三輪一樣（仿IBD：0.4×3個月 + 0.2×6個月 + 0.2×9個月 + 0.2×12個月，同當日標普500比，1-99）。
排名：冇用RS嘅版本揀12個月升幅最大；有用RS嘅版本揀RS最高。
賣出：F =「條件唔再符合」：收市 < MA200 或者 收市 < 52週最高收市價 × 0.7（同線E一樣；12個月升幅唔用嚟賣）；
      T = 由買入後最高收市價跌20%（同線D一樣）。
版本：
  Z1 基本條件 + F
  Z2 基本條件 + T
  Z3 基本條件 + SPY每日訊號持有先開新倉 + F
  Z4 基本條件 + RS ≥ 80 先買（揀RS最高）+ F
  Z5 基本條件 + RS ≥ 80 先買 + 賣出：F 或者 RS < 70
  Z6 基本條件 + SPY訊號 + RS ≥ 80 先買 + 賣出：F 或者 RS < 70
對照：T0 線D、線E（第二十六輪Y3：今年升≥30% + 離高位<30% + MA200；F賣出）、QQQ、SPY。
判斷（同之前一樣）：① 「有改善」（同T0比，驗證期 2015-01 至 2026-09）② 四條問題 ③ 跨年代合格
  （2000-2009、2010-2014、2015-2019、2020-2026 每段年化 ≥ SPY 而且最大跌幅好過 -35%）。
  另外列：每次買賣統計、最差5年年化、每日平均有幾多隻符合。6個版本，有改善都要模擬交易再驗證。
輸出：v27_results.csv、v27_progress.txt
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v23 import rs_rating
from research_v26 import sim

M.PROGRESS = "v27_progress.txt"
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
         "線E（對照，第二十六輪Y3）": sim(P, rets_v, cr, e_ok, ytdn, broken),
         "Z1 基本條件・條件唔再符合就賣": sim(P, rets_v, cr, z_ok, r12n, broken),
         "Z2 基本條件・跌20%賣": sim(P, rets_v, cr, z_ok, r12n, trail20),
         "Z3 基本條件+SPY訊號・條件唔再符合就賣": sim(P, rets_v, cr, z_ok, r12n, broken, allow=sig_ok),
         "Z4 基本條件+RS≥80・條件唔再符合就賣": sim(P, rets_v, cr, zrs_ok, rsn, broken),
         "Z5 基本條件+RS≥80・條件唔再符合或RS<70就賣": sim(P, rets_v, cr, zrs_ok, rsn, broken_rs),
         "Z6 基本條件+SPY訊號+RS≥80・條件唔再符合或RS<70就賣": sim(P, rets_v, cr, zrs_ok, rsn, broken_rs, allow=sig_ok)}
    log("全部版本完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)
    tab = {}
    L = ["=== 第二十七輪：12個月升30%-200% + 離高位<30% + MA200，加減SPY同RS（規則預先登記，見research_v27.py開頭）===",
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
    pd.DataFrame({"report": L}).to_csv("v27_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
