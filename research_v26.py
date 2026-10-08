"""
research_v26.py — 第二十六輪：揀股改用「年初至今升 ≥30%」+「離52週高位唔超過30%」（Esther 2026-10-08）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同（同線D一樣，除非下面講明）：當時真實標普500；已計股息；收市決定 → 下一個交易日收市買賣；每次買賣0.1%；
  最多5隻、每隻 = 買入日總資產 ÷ 5；冇投資嘅錢揸現金（^IRX）；由最高收市價跌20%就賣（同線D一樣）。
定義：
  年初至今 = 今日收市 ÷ 上一年最後一個交易日收市 − 1。
  離高位 = 今日收市 ÷ 過去252個交易日最高收市價（52週高位）；「唔超過30%」= ≥ 0.70。
  （Esther 講「距離最高位少於30%」，呢度用52週高位；歷史最高位唔用，因為好多股票十幾年前高位到而家都未返到。）
新版本（有空位揀「年初至今」升最多嘅）：
  Y1 年初至今 ≥ +30% + 離52週高位 ≤ 30%。
  Y2 Y1 + 收市 > MA50 + 收市 > MA200（加埋線D嘅趨勢條件）。
  Y3 Y2 + SPY每日訊號持有先開新倉（線D2做法）。
對照：T0 線D、T2 線D2。
⚠️ 年初嗰幾個月「年初至今」好少股票會 ≥30%，所以一月至三月好多時會揸現金，呢個係規則本身嘅特性。
判斷（同之前一樣）：
  ① 「有改善」（Y1、Y2 同 T0 比；Y3 同 T2 比，驗證期 2015-01 至 2026-09）。
  ② 跨年代合格：2000-2009、2010-2014、2015-2019、2020-2026 每段年化 ≥ SPY 而且最大跌幅好過 -35%。
  另外列：每月平均揸幾多隻股票（睇吓年初係咪成日空倉）、每次買賣統計、最差5年年化。
輸出：v26_results.csv、v26_progress.txt
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v13 import get_cash
from research_v24 import sim
from research_v8 import decision_days

M.PROGRESS = "v26_progress.txt"
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
    yend = px.ffill().groupby(px.index.year).last()                  # 每年最後一日收市（股票當年有數）
    base = yend.shift(1).reindex(px.index.year).values               # 上一年最後收市
    ytd = P / base - 1
    ytdn = np.nan_to_num(ytd, nan=-9)
    hi52 = px.rolling(252, min_periods=200).max().values
    near = np.nan_to_num(P / hi52, nan=0) >= 0.70
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
    trend = (P > ma50) & (P > ma200)
    base_ok = mem & (r12n >= 1.0) & trend & fin
    y_ok = mem & (ytdn >= 0.30) & near & fin
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    sig_ok = (df["state"] != "out").reindex(days).fillna(False).values
    log("數據準備好")

    V = {"T0 線D": sim(P, rets_v, cr, base_ok, r12n),
         "T2 線D2": sim(P, rets_v, cr, base_ok, r12n, allow=sig_ok),
         "Y1 年初至今≥30% + 離高位≤30%": sim(P, rets_v, cr, y_ok, ytdn),
         "Y2 Y1 + 喺MA50同MA200上面": sim(P, rets_v, cr, y_ok & trend, ytdn),
         "Y3 Y2 + SPY訊號持有先開新倉": sim(P, rets_v, cr, y_ok & trend, ytdn, allow=sig_ok)}
    log("全部版本完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)
    tab = {}
    L = ["=== 第二十六輪：年初至今升≥30% + 離52週高位≤30%（規則預先登記，見research_v26.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻；由高位跌20%賣", ""]
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
    # 每個月平均有幾多隻股票符合條件（睇年初係咪好少）
    L.append("=== 每個月平均有幾多隻股票符合買入條件（2000-2026）===")
    for nm, ok in (("線D", base_ok), ("Y1", y_ok), ("Y2", y_ok & trend)):
        cnt = pd.Series(ok.sum(axis=1), index=days).loc["2000-01-01":END]
        bym = cnt.groupby(cnt.index.month).mean()
        L.append(f"　{nm}：" + "、".join(f"{m}月 {v:.0f}" for m, v in bym.items()))
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
    L.append("=== 預先登記判斷 ===")
    for nm, ref in (("Y1 年初至今≥30% + 離高位≤30%", "T0 線D"), ("Y2 Y1 + 喺MA50同MA200上面", "T0 線D"),
                    ("Y3 Y2 + SPY訊號持有先開新倉", "T2 線D2"), ("T0 線D", None), ("T2 線D2", None)):
        x = tab[("驗證期", nm)]
        marks = "".join("✔" if (tab[(e, nm)]["cagr"] >= tab[(e, "SPY長揸")]["cagr"] and tab[(e, nm)]["mdd"] > -35) else "✘" for e in eras)
        imp = ""
        if ref:
            y = tab[("驗證期", ref)]
            better = (x["mdd"] >= y["mdd"] + 3 and x["cagr"] >= y["cagr"] - 1) or (x["cagr"] >= y["cagr"] + 2 and x["mdd"] >= y["mdd"] - 2)
            imp = f"同 {ref}（{y['cagr']:+.2f}% / {y['mdd']:.1f}%）比 → {'有改善' if better else '冇改善'} ｜ "
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} ｜ {imp}跨年代 {marks} "
                 f"{'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 " + " / ".join(f"{tab[(e, nm)]['cagr']:+.1f}%" for e in eras)
                 + " ｜ 各段最大跌幅 " + " / ".join(f"{tab[(e, nm)]['mdd']:.0f}%" for e in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
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
