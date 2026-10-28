"""
research_v43.py — 第四十三輪：線E 用「第二次接觸周線 EMA21」做止損，同之前嘅賣出方法比較
  （Esther 2026-10-28：「試吓用第二次接觸周線EMA 21做止損，比較一下同之前既止損或止賺方法哪一個較好？」）

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同設定同第四十一輪一樣：當時真實標普500、已計股息、收市決定→下一個交易日收市、每次買賣0.1%、揸5隻、閒錢收^IRX。
入場全部同線E一樣（今年 ≥ +30%、≥ 52週最高收市 × 0.7、> 200日平均，揀今年升最多）。
周線：每個星期最後一個交易日收市 = 周收市；周最低 = 嗰個星期每日最低價嘅最低（只計買入之後嘅日子）。
周線 EMA21 = 周收市嘅 21 週指數平均（同圖表軟件一樣，包括今個星期）。只喺每個星期最後一個交易日檢查一次。
「接觸」兩種：
  接觸A（摸到）：周最低 ≤ 周EMA21
  接觸B（收市跌穿）：周收市 < 周EMA21
點樣數「第二次」：買入之後，第一個「接觸」嘅星期 = 第一次；之後要有最少一個星期冇接觸，再接觸 = 第二次。
  （連續幾個星期都接觸住，只算一次。）第二次接觸嗰個星期最後一個交易日收市決定，下一個交易日收市賣。
版本（只有呢6個，跑完唔加唔減唔改數字）：
  W1 第二次「摸到」周EMA21 就賣（取代線E原本賣出條件）
  W2 第二次「收市跌穿」周EMA21 就賣（取代）
  W3 線E原本賣出條件 + W1（邊個先到用邊個）
  W4 第一次「摸到」周EMA21 就賣（取代）—— 對照：睇吓「等第二次」有冇用
  參考R1 由買入後最高收市跌 20% 就賣（取代；線D嗰種）
  參考R2 收市跌穿 50日平均就賣（取代）
防止「賣完即刻買返」（第四十一輪 F4 嘅問題）：W1-W4、R1、R2 賣出之後 20 個交易日（約1個月）內唔再買同一隻。線E（對照）照原本規則。
判斷（同第四十一輪一樣，同線E比；驗證期 2015-01 至 2026-09）：有改善、四條問題、跨年代合格、最差5年年化。
另外列：每次買賣統計（次數、賺錢比例、平均）。
最低價用 Yahoo 已調整 開高低收（同第三十七輪一樣嘅快取）；按當日 收市 比例換算返同收市價一致。冇最低價數據嘅日子當冇摸到。
就算合格都唔會直接改模擬交易，要你同意先加新線。

輸出：v43_results.csv、v43_progress.txt
==== 結果（2026-10-28 第一次跑，run 40725356827，記錄低，唔再改）====
- 線E 對照 +28.74% / -32.6%（同之前一樣，核對通過）。最低價數據 715/715 隻。
- 驗證期：W1 +27.22% / -43.4%；W2 +31.81% / -39.9%（年化多 3.1 點，但最大跌幅差 7.3 點，標準只容許差 2 點）；
  W3 +27.74% / -39.1%；W4 +16.97% / -42.4%；R1 +18.32% / -29.9%；R2 +11.97% / -36.4% → 全部冇改善。
- 跨年代全部唔合格。W2 好嘅地方集中喺 2020-26（+48.5%，線E +32.2%）同 2025 年（+73.3%，線E +20.0%）；
  但 2015-19 只有 +12.2%（線E +24.3%），2000-09 +1.7%（線E +4.5%）。
- 每單：W1 最差一單 -97%、W2 -89%（線E 最差 -40%）→ 只用周EMA21做止損，股價急跌時 EMA 跟住跌，未必及時賣。
- 「第一次摸到」(W4) 差過「第二次」(W1/W2) 好多 → 等第二次確實好啲，但都唔夠線E原本規則穩陣。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim
from research_v37 import ohlc

M.PROGRESS = "v43_progress.txt"
log = M.log
END = "2026-09-30"
COOL = 20


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
    cols = list(px.columns)
    log("收市價準備好，下載最高最低價")
    data = ohlc(cols)
    low_rel = pd.DataFrame({t: (d["Low"] / d["Close"]) for t, d in data.items()}).reindex(index=days, columns=cols)
    LOW = (low_rel * px).values                                   # 換算返同收市價一致
    log(f"最低價：{len(data)}/{len(cols)} 隻有數據")

    rets_v = px.pct_change(fill_method=None).values
    ma200 = px.rolling(200).mean().values
    ma50 = px.rolling(50).mean().values
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

    # 周線：每個星期最後一個交易日
    wk = pd.Series(days.to_period("W-FRI"), index=days)
    week_end = (wk != wk.shift(-1)).values
    week_end[-1] = True
    wk_id = pd.factorize(wk)[0]
    wclose = px[week_end].ffill()
    wema = wclose.ewm(span=21, adjust=False, min_periods=21).mean()
    EMA_W = np.full(P.shape, np.nan)
    EMA_W[week_end] = wema.values
    WCLOSE = np.full(P.shape, np.nan)
    WCLOSE[week_end] = px[week_end].values
    log("數據準備好")

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    def make(kind, need, add_e, ok):
        """kind: 'touch'（摸到）或 'close'（收市跌穿）；need: 第幾次接觸就賣；add_e: 加埋線E原本賣出條件。"""
        st = {}

        def fn(i, c, p, pk):
            s = st.get(c)
            if s is None or s["last"] < i - 5:                  # 新買入
                s = st[c] = {"entry": i, "low": np.inf, "n": 0, "prev": False, "last": i}
            s["last"] = i
            if i > s["entry"] and np.isfinite(LOW[i, c]):          # 買入嗰日之前嘅最低唔計
                s["low"] = min(s["low"], LOW[i, c])
            sell = add_e and broken(i, c, p, pk)
            if week_end[i] and np.isfinite(EMA_W[i, c]):
                e = EMA_W[i, c]
                t = (s["low"] <= e) if kind == "touch" else (WCLOSE[i, c] < e)
                if t and not s["prev"]:
                    s["n"] += 1
                s["prev"] = bool(t)
                s["low"] = np.inf                                   # 下個星期重新計最低
                if s["n"] >= need:
                    sell = True
            if sell:
                st.pop(c, None)
                ok[i + 1:i + 1 + COOL, c] = False                   # 冷靜期
            return sell
        return fn

    def make_simple(rule, ok):
        def fn(i, c, p, pk):
            if rule == "R1":
                sell = p <= pk * 0.8
            else:
                sell = np.isfinite(ma50[i, c]) and p < ma50[i, c]
            if sell:
                ok[i + 1:i + 1 + COOL, c] = False
            return sell
        return fn

    V = {}
    hold_days = {}

    def run(name, fn_maker):
        ok = e_ok.copy()
        V[name] = sim(P, rets_v, cr, ok, ytdn, fn_maker(ok))
        log(f"{name} 完成")

    run("線E（對照）", lambda ok: broken)
    run("W1 第二次摸到周EMA21", lambda ok: make("touch", 2, False, ok))
    run("W2 第二次收市跌穿周EMA21", lambda ok: make("close", 2, False, ok))
    run("W3 線E賣出＋第二次摸到", lambda ok: make("touch", 2, True, ok))
    run("W4 第一次摸到周EMA21", lambda ok: make("touch", 1, False, ok))
    run("參考R1 高位跌20%", lambda ok: make_simple("R1", ok))
    run("參考R2 跌穿50日線", lambda ok: make_simple("R2", ok))

    L = ["=== 第四十三輪：線E + 第二次接觸周線EMA21 做止損（規則預先登記，見research_v43.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻；最低價有數據 {len(data)}/{len(cols)} 隻", ""]
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
    yrs = (days[-1] - days[0]).days / 365.25
    L.append("=== 每次買賣（1999-2026）===")
    for nm, v in V.items():
        tr = np.array(v[1])
        L.append(f"　{nm}：{len(tr)} 次（每年約 {len(tr) / yrs:.0f} 次）｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
                 f"賺嘅平均 {tr[tr > 0].mean() * 100:+.1f}% ｜ 蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}%")
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
    pd.DataFrame({"report": L}).to_csv("v43_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
