"""
research_v29b.py — 第二十九輪B：細價股（全美國上市股票，唔止標普500）用線E規則（Esther 2026-10-14）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
⚠️ 偏差：冇免費嘅歷史中小型股成分名單，只可以用「今日仲上市」嘅全美國普通股（nasdaqtrader.com，同第十五輪B一樣）。
   已經倒閉／除牌嘅公司唔喺入面 → 結果一定偏好（倖存者偏差），只可以當「最好情況」。
   第十五輪C估計：標普500 用今日名單（U2）比當時真實名單（U1）每年中位數高 +11.9%；細價股偏差應該更大。
股票池：全美國上市普通股 + 50日平均成交額 ≥ 500萬美元（同第十五輪B嘅U3一樣）；「非標普500」= 再剔除當時標普500成分股。
共同（線E規則）：今年升 ≥30% + 收市 ≥ 52週最高收市價 × 0.7 + 收市 > MA200 就買（揀今年升最多）；
  收市 < MA200 或者 < 52週最高收市價 × 0.7 就賣；收市決定→下一個交易日收市；每次買賣0.1%；每隻 = 總資產 ÷ 揸幾多隻。
版本：
  S0 線E（當時真實標普500，對照）
  S1 全美國・揸5隻   S2 全美國・揸3隻   S3 非標普500（中小型）・揸5隻   S4 非標普500・揸5隻・SPY每日訊號持有先開新倉
判斷：
  ① 如果有偏差都輸 S0（驗證期年化低過，或者最大跌幅差過）→ 細價股淘汰。
  ② 如果贏：驗證期年化扣除 11.9 個百分點（偏差估計）之後仲贏 S0、而且最大跌幅好過 -35% → 「值得用模擬交易驗證」；否則唔夠證據。
  另外列：四條問題、跨年代（2000-2009、2010-2014、2015-2019、2020-2026 每段年化 ≥ SPY 而且最大跌幅好過 -35%）、
  Esther 目標（驗證期 ≥ 50% 而且最大跌幅好過 -35%）、每次買賣統計。
輸出：v29b_results.csv、v29b_progress.txt
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v15b import download, us_listed
from research_v26 import sim

M.PROGRESS = "v29b_progress.txt"
log = M.log
END = "2026-09-30"


def main():
    import yfinance as yf
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    hist_sp = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    allus = us_listed()
    log(f"全美國上市普通股 {len(allus)} 隻；歷史標普500 {len(hist_sp)} 隻")
    tickers = sorted(set(allus) | set(hist_sp))

    def one(t):
        h = yf.Ticker(t).history(start="1998-01-01", end="2026-10-01", auto_adjust=True)["Close"]
        h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
        return h
    spy, qqq = one("SPY"), one("QQQ")
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    C, V = download(tickers, "1998-01-01", "2026-10-01")
    cols = sorted(C)
    ix = {t: k for k, t in enumerate(cols)}
    px = pd.DataFrame(C).reindex(days)[cols].astype("float32")
    vol = pd.DataFrame(V).reindex(days)[cols].astype("float32")
    del C, V
    dv50 = (px * vol).rolling(50, min_periods=40).mean().values
    del vol
    P = px.values
    rets_v = px.pct_change(fill_method=None).values
    ma200 = px.rolling(200).mean().values
    hi52 = px.rolling(252, min_periods=200).max().values
    ye = px.groupby(days.year).last()
    ytdn = np.nan_to_num(P / ye.shift(1).reindex(days.year).values - 1, nan=-9).astype("float32")
    del px
    cash_r = get_cash(days)
    cr = cash_r.values
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = qqq.reindex(days).pct_change(fill_method=None).fillna(0.0)
    m1 = np.zeros(P.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        m1[i, cache[key]] = True
    m3 = np.zeros(P.shape, dtype=bool)
    m3[:, [ix[t] for t in allus if t in ix]] = True
    m3 &= np.nan_to_num(dv50, nan=0.0) >= 5e6
    fin = np.isfinite(P)
    cond = (ytdn >= 0.30) & (P >= np.nan_to_num(hi52, nan=np.inf) * 0.7) & (P > ma200) & fin
    ok_sp, ok_all, ok_small = m1 & cond, m3 & cond, m3 & ~m1 & cond
    log(f"每日平均符合：標普 {ok_sp.sum(axis=1).mean():.0f} 隻 ｜ 全美國 {ok_all.sum(axis=1).mean():.0f} 隻 ｜ 非標普 {ok_small.sum(axis=1).mean():.0f} 隻")
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    sig_ok = (df["state"] != "out").reindex(days).fillna(False).values.astype(bool)
    log("數據準備好")

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    Vs, TR = {}, {}
    for nm, ok, kw in (("S0 線E・標普500（對照）", ok_sp, {}), ("S1 全美國・揸5隻", ok_all, {}), ("S2 全美國・揸3隻", ok_all, {"n": 3}),
                       ("S3 非標普500・揸5隻", ok_small, {}), ("S4 非標普500・揸5隻・SPY訊號", ok_small, {"allow": sig_ok})):
        r, tr = sim(P, rets_v, cr, ok, ytdn, broken, **kw)
        Vs[nm], TR[nm] = pd.Series(r, index=days), tr
        log(f"{nm} 完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    allser.update(Vs)
    tab = {}
    L = ["=== 第二十九輪B：細價股（全美國上市股票）用線E規則（規則預先登記，見research_v29b.py開頭）===",
         "⚠️ 全美國名單只有今日仲上市嘅公司 → 有倖存者偏差，結果偏好（最好情況）", ""]
    for wn, a, b in windows:
        L.append(f"=== {wn}（{a} 至 {b}）===")
        for nm, r in allser.items():
            rr = r.loc[a:b].dropna()
            p = perf(rr, rf=cash_r, bench=qqq_r.loc[a:b])
            tab[(wn, nm)] = p
            L.append(f"　{nm}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}")
        L.append("")
    L.append("=== 每次買賣（1999-2026）===")
    for nm, tr in TR.items():
        tr = np.array(tr)
        if len(tr):
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
    b0 = tab[("驗證期", "S0 線E・標普500（對照）")]
    L.append(f"=== 預先登記判斷 ｜ S0 線E {b0['cagr']:+.2f}% / {b0['mdd']:.1f}%（應該 +28.74% / -32.6%）｜ QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% ｜ SPY {s['cagr']:+.2f}% ===")
    goal = []
    for nm in Vs:
        x, e = tab[("驗證期", nm)], tab[("2000-2009", nm)]
        marks = "".join("✔" if (tab[(er, nm)]["cagr"] >= tab[(er, "SPY長揸")]["cagr"] and tab[(er, nm)]["mdd"] > -35) else "✘" for er in eras)
        verdict = "—"
        if not nm.startswith("S0"):
            if x["cagr"] < b0["cagr"] or x["mdd"] < b0["mdd"]:
                verdict = "有偏差都輸線E → 淘汰"
            elif x["cagr"] - 11.9 > b0["cagr"] and x["mdd"] > -35:
                verdict = "扣除偏差仍然贏 → 值得模擬交易驗證"
            else:
                verdict = "贏但扣除偏差之後唔夠證據"
        q1 = x["cagr"] > q["cagr"]
        q2 = q1 and x["mdd"] > -35 and x["sharpe"] >= q["sharpe"]
        q3 = (not q1) and x["mdd"] >= q["mdd"] + 5 and x["cagr"] >= s["cagr"]
        q4 = e["mdd"] > -35
        if x["cagr"] >= 50 and x["mdd"] > -35:
            goal.append(nm)
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} ｜ {verdict} ｜ 四條問題 ①{'✔' if q1 else '✘'} "
                 f"②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'} ｜ "
                 f"跨年代 {marks} {'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 "
                 + " / ".join(f"{tab[(er, nm)]['cagr']:+.1f}%" for er in eras) + " ｜ 各段最大跌幅 "
                 + " / ".join(f"{tab[(er, nm)]['mdd']:.0f}%" for er in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
    L.append(f"　Esther目標（驗證期 ≥50% 而且最大跌幅好過 -35%，未扣偏差）：{'、'.join(goal) if goal else '冇任何版本做到'}")
    pd.DataFrame({"report": L}).to_csv("v29b_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
