"""
research_v34.py — 第三十四輪：用「大市寬度」代替 SPY 做訊號（Esther 2026-10-15：「呢兩樣我覺得非常值得試」）

純研究，唔改任何模擬交易（線A/B/C/D/D2/E/E2、輪動、SPY訊號全部照舊），唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
大市寬度 = 當日標普500成分股（fja05680 當時名單，有200日平均嘅）入面，收市高過自己200日平均嘅比例（%）。
寬度狀態（有緩衝，避免日日轉）：寬度 < 40% → 「弱」；之後要 > 50% 先返「強」。開始時 = 強。
共同設定同第二十八、三十三輪一樣（已計股息、收市決定→下一個交易日收市、0.1%、揸5隻、閒錢收^IRX）。

第一部分：揸 SPY 嘅時機（同 main.py SPY每日訊號比）
  S0 SPY長揸；S1 SPY每日訊號（main.py，持有=揸SPY，現金=收^IRX）；S2 寬度狀態「強」揸SPY，「弱」收^IRX。
  （收市決定、下一個交易日收市先轉；同其他測試一樣）
  另外列：S1 每次轉現金之前，寬度早幾多日轉「弱」（中位數）；兩個訊號每年平均轉幾多次。
  判斷：S2 同 S1 比「有改善」（標準同下面一樣）。

第二部分：線D、線E 嘅開新倉閘門（每條線4個版本；只有呢啲，跑完唔加唔減唔改數字）
  G0 SPY閘（SPY每日訊號 = 現金就唔開新倉；即係線D2，線E版本之前未測過）— 參考
  B1 寬度 < 50% 就唔開新倉（冇緩衝）
  B2 寬度狀態「弱」就唔開新倉（40%/50% 緩衝）
  B3 寬度狀態「弱」就全部賣出兼唔開新倉；返「強」先再買

判斷（同第三十三輪一樣，同自己對照線比；驗證期 2015-01 至 2026-09）：
  有改善：最大跌幅好過對照 ≥3點而年化唔低過對照超過1點；或者年化高過對照 ≥2點而最大跌幅唔差過對照超過2點。
  四條問題；跨年代合格（2000-09、2010-14、2015-19、2020-26 每段年化 ≥ SPY 而且最大跌幅好過 -35%）；最差5年年化。
  就算合格都唔會直接改模擬交易，要你同意先加新線。

已知限制：Yahoo 冇數據嘅已除牌股唔計入寬度（多數係跌得差嘅股）→ 寬度會比真實高少少。

輸出：v34_results.csv、v34_progress.txt
==== 結果（2026-10-15 第一次跑，run 38742146705，記錄低，唔再改）====
- 寬度平均 64%，最低 1%（2008-11-20）；「弱」日子 18%（SPY訊號「現金」日子都係 18%）；每年轉 2.4 次（SPY訊號 2.0 次）。
- SPY訊號 27 次轉現金，之前180日內寬度已轉弱 19 次，中位數只係早 2 個交易日；8 次寬度冇事先轉弱（例如 2022-02、2025-03、2026-03）。
- 揸SPY時機：S1 SPY每日訊號 +12.34% / -26.9%；S2 寬度 +8.87% / -22.3%（2000-09 +1.4% / -43%）→ 冇改善。
- 線D：B1 +21.39% / -34.3%；B2 +20.70% / -34.3%；B3 +17.99% / -26.7% → 全部冇改善。參考：G0 SPY閘（=線D2）+26.83% / -29.5% 有改善（同第二十二輪一致）。
- 線E：B1 +21.27% / -34.9%；B2 +22.73% / -34.9%；B3 +18.12% / -44.8% → 全部冇改善。
  參考（冇預先登記做判斷）：G0 線E+SPY閘 +28.14% / -32.6%，驗證期同線E差唔多，但 2000-09 +12.1% / -46%（線E +4.5% / -62%），最差5年 -0.6%（線E -9.8%）。
- 結論：寬度冇比 SPY訊號更早或者更好；SPY訊號仍然係最好嘅大市過濾。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim

M.PROGRESS = "v34_progress.txt"
log = M.log
END = "2026-09-30"


def hyst(b, lo=40.0, hi=50.0):
    st = np.ones(len(b), dtype=bool)
    s = True
    for i, x in enumerate(b):
        if np.isfinite(x):
            if s and x < lo:
                s = False
            elif (not s) and x > hi:
                s = True
        st[i] = s
    return st


def timed(r, state, cr):
    """state[i] = 收市 i 決定；收市 i+1 先轉 → 第 i+2 日嘅回報先跟新狀態。"""
    pos = np.r_[True, True, state[:-2]]
    return np.where(pos, r, cr)


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
    valid = mem & fin & np.isfinite(ma200)
    above = valid & (P > np.nan_to_num(ma200, nan=np.inf))
    nvalid = valid.sum(axis=1)
    breadth = np.where(nvalid >= 100, above.sum(axis=1) / np.maximum(nvalid, 1) * 100, np.nan)
    strong = hyst(breadth)
    b50 = np.nan_to_num(breadth, nan=100) >= 50
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    sig_ok = (df["state"] != "out").reindex(days).fillna(False).values
    log(f"數據準備好；寬度平均 {np.nanmean(breadth):.0f}%")

    d_ok = mem & (r12n >= 1.0) & (P > ma50) & (P > ma200) & fin
    e_ok = mem & (ytdn >= 0.30) & (P >= hi52f * 0.7) & (P > ma200) & fin

    def trail20(i, c, p, pk):
        return p <= pk * 0.8

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    def plus_off(f):
        return lambda i, c, p, pk: f(i, c, p, pk) or (not strong[i])

    V = {}
    for base, ok, rank, sell in [("線D", d_ok, r12n, trail20), ("線E", e_ok, ytdn, broken)]:
        V[f"{base}（對照）"] = sim(P, rets_v, cr, ok, rank, sell)
        V[f"{base}・G0 SPY閘（參考）"] = sim(P, rets_v, cr, ok, rank, sell, allow=sig_ok)
        V[f"{base}・B1 寬度<50%唔開新倉"] = sim(P, rets_v, cr, ok, rank, sell, allow=b50)
        V[f"{base}・B2 寬度弱唔開新倉"] = sim(P, rets_v, cr, ok, rank, sell, allow=strong)
        V[f"{base}・B3 寬度弱全部賣"] = sim(P, rets_v, cr, ok, rank, plus_off(sell), allow=strong)
        log(f"{base} 完成")

    sr = spy_r.values
    T = {"S1 SPY每日訊號": pd.Series(timed(sr, sig_ok, cr), index=days),
         "S2 寬度訊號揸SPY": pd.Series(timed(sr, strong, cr), index=days)}

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r, **T}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)
    tab = {}
    L = ["=== 第三十四輪：大市寬度代替SPY做訊號（規則預先登記，見research_v34.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻", ""]
    bs = pd.Series(breadth, index=days).loc["2000-01-03":END]
    ss = pd.Series(strong, index=days).loc["2000-01-03":END]
    sg = pd.Series(sig_ok, index=days).loc["2000-01-03":END]
    yrs = len(ss) / 252
    L.append(f"寬度：平均 {bs.mean():.0f}%｜最低 {bs.min():.0f}%（{bs.idxmin().date()}）｜「弱」日子 {(~ss).mean() * 100:.0f}% ｜ "
             f"SPY訊號「現金」日子 {(~sg).mean() * 100:.0f}%")
    L.append(f"每年轉幾多次：寬度狀態 {(ss.astype(int).diff().abs().sum()) / yrs:.1f} 次 ｜ SPY訊號 {(sg.astype(int).diff().abs().sum()) / yrs:.1f} 次")
    outs = list(sg.index[(sg.shift(1, fill_value=True)) & (~sg)])
    offs = list(ss.index[(ss.shift(1, fill_value=True)) & (~ss)])
    leads = []
    for d in outs:
        prev = [o for o in offs if o <= d and (d - o).days <= 180]
        if prev:
            leads.append((days.get_loc(d) - days.get_loc(prev[-1])))
    L.append(f"SPY訊號轉現金 {len(outs)} 次；之前180日內寬度已經轉弱 {len(leads)} 次；早咗交易日中位數 {np.median(leads) if leads else float('nan'):.0f} 日")
    L.append("　每次 SPY 轉現金日期 → 寬度轉弱日期：" + "；".join(
        f"{d.date()}←" + (str([o for o in offs if o <= d and (d - o).days <= 180][-1].date())
                          if [o for o in offs if o <= d and (d - o).days <= 180] else "冇") for d in outs))
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

    def judge(nm, b0, base):
        x, e = tab[("驗證期", nm)], tab[("2000-2009", nm)]
        marks = "".join("✔" if (tab[(er, nm)]["cagr"] >= tab[(er, "SPY長揸")]["cagr"] and tab[(er, nm)]["mdd"] > -35) else "✘" for er in eras)
        imp = "—"
        if b0 is not None:
            better = (x["mdd"] >= b0["mdd"] + 3 and x["cagr"] >= b0["cagr"] - 1) or (x["cagr"] >= b0["cagr"] + 2 and x["mdd"] >= b0["mdd"] - 2)
            imp = "有改善" if better else "冇改善"
        q1 = x["cagr"] > q["cagr"]
        q2 = q1 and x["mdd"] > -35 and x["sharpe"] >= q["sharpe"]
        q3 = (not q1) and x["mdd"] >= q["mdd"] + 5 and x["cagr"] >= s["cagr"]
        q4 = e["mdd"] > -35
        return (f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} ｜ 同{base}比 {imp} ｜ "
                f"四條問題 ①{'✔' if q1 else '✘'} ②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'} ⇒ "
                f"{'保留' if (q2 or q3) else '淘汰'} ｜ 跨年代 {marks} {'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 "
                + " / ".join(f"{tab[(er, nm)]['cagr']:+.1f}%" for er in eras) + " ｜ 各段最大跌幅 "
                + " / ".join(f"{tab[(er, nm)]['mdd']:.0f}%" for er in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")

    b1 = tab[("驗證期", "S1 SPY每日訊號")]
    L.append(f"=== 預先登記判斷：揸SPY時機 ｜ S1 SPY每日訊號 {b1['cagr']:+.2f}% / {b1['mdd']:.1f}% ===")
    L.append(judge("S1 SPY每日訊號", None, "S1"))
    L.append(judge("S2 寬度訊號揸SPY", b1, "S1"))
    L.append("")
    for base in ["線D", "線E"]:
        b0 = tab[("驗證期", f"{base}（對照）")]
        L.append(f"=== 預先登記判斷：{base} ｜ 對照 {b0['cagr']:+.2f}% / {b0['mdd']:.1f}% ｜ QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% / Sharpe {q['sharpe']:.2f} ｜ SPY {s['cagr']:+.2f}% ===")
        for nm in [k for k in V if k.startswith(base)]:
            L.append(judge(nm, None if "對照" in nm else b0, base))
        L.append("")
    pd.DataFrame({"report": L}).to_csv("v34_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
