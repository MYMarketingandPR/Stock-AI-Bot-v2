"""
research_v15d.py — 第十五輪D：Esther 嘅組合規則（2026-10-05 11:20 提出）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
股票池、倍升股定義、第一次買入條件、每位大小、成本、判斷標準①-④、偏差估計：同 research_v15b.py 一樣。
賣出（邊個先到用邊個，全部用收市價判斷，下一個收市執行）：
  ① 止蝕：收市價 ≤ 買入價 × 0.93（跌穿買入價7%）
  ② 止賺/離場：買入之後，最近10個交易日內第2次收市低過MA50
再買入（只限之前曾經買過嘅股票）：收市價由MA50下面升返上MA50上面（今日 > MA50、上一日 ≤ MA50），
  而且今日成交量 > 過去50日平均成交量（MAVOL50）；仍然要喺股票池入面（U1要係當時成分股；U3要成交額夠）。
  再買入唔需要「仲係倍升股」。
有空位時：先買「再買入訊號」嘅股票（多過空位就揀12個月升幅大嘅），之後先買新嘅倍升股（12個月升幅最大優先）。
版本：揸 1 / 3 / 5 隻。
輸出：v15d_results.csv、v15d_progress.txt
==== 結果（未跑）====
"""
import io
import time

import numpy as np
import pandas as pd
import requests

import momentum_v1 as M
from perf import perf
from research_v10 import metrics, score_pick
from research_v13 import get_cash

M.PROGRESS = "v15d_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"
BAD = ("WARRANT", "UNIT", " RIGHT", "PREFERRED", "NOTES", "DEBENTURE", "% ", "DEPOSITARY SHARES REPRESENTING")


def us_listed():
    out = set()
    for url, sym in (("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
                     ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol")):
        txt = requests.get(url, headers=M.HDR, timeout=60).text
        df = pd.read_csv(io.StringIO(txt), sep="|")
        df = df[df[sym].notna() & ~df[sym].astype(str).str.startswith("File Creation")]
        df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]
        name = df["Security Name"].astype(str).str.upper()
        df = df[~name.apply(lambda s: any(b in s for b in BAD))]
        syms = df[sym].astype(str)
        syms = syms[~syms.str.contains(r"[\$\^]")]
        out |= set(M.norm(s) for s in syms)
    return sorted(out)


def download(tickers, start, end):
    import yfinance as yf
    C, V, fails = {}, {}, 0
    for n, t in enumerate(tickers):
        for attempt in range(3):
            try:
                h = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
                if h is not None and len(h) > 260:
                    idx = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
                    c = pd.Series(h["Close"].astype("float32").values, index=idx)
                    v = pd.Series(h["Volume"].astype("float32").values, index=idx)
                    C[t] = c[~c.index.duplicated(keep="last")]
                    V[t] = v[~v.index.duplicated(keep="last")]
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        if t not in C:
            fails += 1
        time.sleep(0.25)
        if (n + 1) % 250 == 0:
            log(f"下載進度 {n + 1}/{len(tickers)}，有數據 {len(C)}")
    log(f"下載完成：有數據 {len(C)}/{len(tickers)}")
    return C, V


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    month_ends = pd.date_range("1999-01-31", END, freq="ME")
    hist_sp = sorted(set().union(*[members_fn(d) for d in month_ends]))
    now_sp = sorted(members_fn(pd.Timestamp(END)))
    try:
        allus = us_listed()
    except Exception as e:
        log(f"攞唔到全美國名單：{e}")
        allus = []
    log(f"全美國上市普通股 {len(allus)} 隻；歷史標普500 {len(hist_sp)}；今日標普500 {len(now_sp)}")
    tickers = sorted(set(allus) | set(hist_sp) | set(now_sp))
    import yfinance as yf
    spy = yf.Ticker("SPY").history(start="1998-01-01", end="2026-10-01", auto_adjust=True)["Close"]
    spy.index = pd.DatetimeIndex(spy.index).tz_localize(None).normalize()
    qqq = yf.Ticker("QQQ").history(start="1998-01-01", end="2026-10-01", auto_adjust=True)["Close"]
    qqq.index = pd.DatetimeIndex(qqq.index).tz_localize(None).normalize()
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    C, V = download(tickers, "1998-01-01", "2026-10-01")
    cols = sorted(C)
    col_ix = {t: k for k, t in enumerate(cols)}
    P = pd.DataFrame(C).reindex(days)[cols].astype("float32")
    Vol = pd.DataFrame(V).reindex(days)[cols].astype("float32")
    del C, V
    dv50 = (P * Vol).rolling(50, min_periods=40).mean().values.astype("float32")
    mavol50 = Vol.rolling(50, min_periods=40).mean().values.astype("float32")
    Vv = Vol.values
    del Vol
    ma50 = P.rolling(50).mean().values.astype("float32")
    ma200 = P.rolling(200).mean().values.astype("float32")
    r12 = (P / P.shift(252) - 1).values.astype("float32")
    rets = P.pct_change(fill_method=None).values.astype("float32")
    Pv = P.values
    below50 = (np.nan_to_num(Pv, nan=np.inf) < np.nan_to_num(ma50, nan=-np.inf))
    cs50 = np.cumsum(below50, axis=0, dtype=np.int32)
    Pf = P.ffill().values.astype("float32")
    log("指標計好")
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = qqq.reindex(days).pct_change(fill_method=None).fillna(0.0)
    cash_r = get_cash(days)
    cr = cash_r.values

    # 股票池
    U = {}
    m1 = np.zeros(Pv.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [col_ix[t] for t in members_fn(d) if t in col_ix]
        m1[i, cache[key]] = True
    U["U1 當時真實標普500"] = m1
    m2 = np.zeros(Pv.shape, dtype=bool)
    m2[:, [col_ix[t] for t in now_sp if t in col_ix]] = True
    U["U2 今日標普500（有偏差）"] = m2
    if allus:
        m3 = np.zeros(Pv.shape, dtype=bool)
        m3[:, [col_ix[t] for t in allus if t in col_ix]] = True
        m3 &= np.nan_to_num(dv50, nan=0.0) >= 5e6
        m3 &= np.isfinite(r12)
        U["U3 全美國（有偏差，上限）"] = m3
        log(f"U3 平均每日合資格股票：2000年 {m3[days.year == 2000].sum(axis=1).mean():.0f}，"
            f"2010年 {m3[days.year == 2010].sum(axis=1).mean():.0f}，2025年 {m3[days.year == 2025].sum(axis=1).mean():.0f}")
    base_ok = (r12 >= 1.0) & (Pv > ma50) & (Pv > ma200) & np.isfinite(Pv)

    def simulate(entry_ok, N, exit_ma):
        MA = ma50 if exit_ma in (50, "50x2") else ma200
        trail = exit_ma if isinstance(exit_ma, float) else None
        entered, peak = {}, {}
        hold, cash = {}, 1.0
        pend_sell, pend_buy = [], []
        navs, turns, nheld = [], [], []
        for i in range(len(days)):
            for c in list(hold):
                r = rets[i, c]
                if np.isfinite(r):
                    hold[c] *= 1 + r
            cash *= 1 + cr[i]
            traded = 0.0
            for c in pend_sell:
                if c in hold:
                    v = hold.pop(c)
                    traded += v
                    cash += v * (1 - COST)
            total = cash + sum(hold.values())
            for c in pend_buy:
                if c in hold or len(hold) >= N or not np.isfinite(Pv[i, c]):
                    continue
                amt = min(total / N, cash)
                if amt <= total * 0.01:
                    break
                cash -= amt
                hold[c] = amt * (1 - COST)
                entered[c] = i
                peak[c] = float(Pv[i, c])
                traded += amt
            pend_sell, pend_buy = [], []
            total = cash + sum(hold.values())
            navs.append(total)
            turns.append(traded / total if total > 0 else 0.0)
            nheld.append(len(hold))
            for c in hold:
                p = Pv[i, c]
                if not np.isfinite(p):
                    if i + 1 < len(days) and not np.isfinite(Pv[i + 1, c]):
                        pend_sell.append(c)
                    continue
                if trail is not None:
                    peak[c] = max(peak[c], float(p))
                    if p <= peak[c] * (1 - trail):
                        pend_sell.append(c)
                elif exit_ma == "50x2":
                    st = max(i - 9, entered[c])
                    n_below = cs50[i, c] - (cs50[st - 1, c] if st > 0 else 0)
                    if n_below >= 2:
                        pend_sell.append(c)
                elif np.isfinite(MA[i, c]) and p < MA[i, c]:
                    pend_sell.append(c)
            free = N - (len(hold) - len(pend_sell))
            if free > 0:
                cand = [c for c in np.where(entry_ok[i])[0] if c not in hold]
                if cand:
                    cand.sort(key=lambda c: -r12[i, c])
                    pend_buy = cand[:free]
        nav = pd.Series(navs, index=days)
        return nav.pct_change().fillna(0.0), pd.Series(turns, index=days), pd.Series(nheld, index=days)

    def simulate_combo(entry_ok, mask, N):
        hold, cash = {}, 1.0
        buyp, entered, ever = {}, {}, set()
        pend_sell, pend_buy = [], []
        navs, turns, nheld = [], [], []
        stops = {"止蝕7%": 0, "MA50離場": 0, "除牌": 0}
        for i in range(len(days)):
            for c in list(hold):
                r = rets[i, c]
                if np.isfinite(r):
                    hold[c] *= 1 + r
            cash *= 1 + cr[i]
            traded = 0.0
            for c in pend_sell:
                if c in hold:
                    v = hold.pop(c)
                    traded += v
                    cash += v * (1 - COST)
            total = cash + sum(hold.values())
            for c in pend_buy:
                if c in hold or len(hold) >= N or not np.isfinite(Pv[i, c]):
                    continue
                amt = min(total / N, cash)
                if amt <= total * 0.01:
                    break
                cash -= amt
                hold[c] = amt * (1 - COST)
                buyp[c] = float(Pv[i, c])
                entered[c] = i
                ever.add(c)
                traded += amt
            pend_sell, pend_buy = [], []
            total = cash + sum(hold.values())
            navs.append(total)
            turns.append(traded / total if total > 0 else 0.0)
            nheld.append(len(hold))
            for c in hold:
                p = Pv[i, c]
                if not np.isfinite(p):
                    if i + 1 < len(days) and not np.isfinite(Pv[i + 1, c]):
                        pend_sell.append(c)
                        stops["除牌"] += 1
                    continue
                if p <= buyp[c] * 0.93:
                    pend_sell.append(c)
                    stops["止蝕7%"] += 1
                    continue
                st = max(i - 9, entered[c])
                if cs50[i, c] - (cs50[st - 1, c] if st > 0 else 0) >= 2:
                    pend_sell.append(c)
                    stops["MA50離場"] += 1
            free = N - (len(hold) - len(pend_sell))
            if free > 0 and i > 0:
                re_c = [c for c in ever if c not in hold and mask[i, c] and np.isfinite(Pv[i, c])
                        and np.isfinite(ma50[i, c]) and Pv[i, c] > ma50[i, c]
                        and np.isfinite(Pv[i - 1, c]) and np.isfinite(ma50[i - 1, c]) and Pv[i - 1, c] <= ma50[i - 1, c]
                        and np.isfinite(Vv[i, c]) and np.isfinite(mavol50[i, c]) and Vv[i, c] > mavol50[i, c]]
                re_c.sort(key=lambda c: -np.nan_to_num(r12[i, c], nan=-9))
                new_c = [c for c in np.where(entry_ok[i])[0] if c not in hold and c not in re_c]
                new_c.sort(key=lambda c: -r12[i, c])
                pend_buy = (re_c + new_c)[:free]
        nav = pd.Series(navs, index=days)
        return nav.pct_change().fillna(0.0), pd.Series(turns, index=days), pd.Series(nheld, index=days), stops

    L = ["=== 第十五輪D：Esther組合規則（7%止蝕 + 10日內第2次低過MA50離場 + 升穿MA50帶量再買）===",
         f"全美國名單 {len(allus)} 隻；有價錢數據 {len(cols)} 隻；已計股息；每次買賣0.1%；唔用槓桿", ""]
    zero = pd.Series(0.0, index=days)

    def show(name, r, t, a, b):
        p = perf(r.loc[a:b], rf=cash_r, bench=qqq_r.loc[a:b], turn=t, cap_freq="ME")
        return p, (f"　{name}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f} ｜ "
                   f"Calmar {p['calmar']:.2f} ｜ 每年換手 {p['turn_year']:.0f}% ｜ "
                   f"對QQQ 升市捕捉 {p['up_cap']:.0f}% / 跌市捕捉 {p['down_cap']:.0f}%")

    windows = [("揀規則期", "2000-01-01", "2014-12-31"), ("驗證期", "2015-01-01", END),
               ("2000-2009", "2000-01-01", "2009-12-31"), ("2010-2019", "2010-01-01", "2019-12-31"),
               ("2020-2026", "2020-01-01", END), ("全期", "2000-01-01", END),
               ("金融海嘯", "2007-07-01", "2009-06-30"), ("科網爆煲", "2000-03-01", "2002-12-31")]
    qtab = {wn: show("QQQ長揸", qqq_r, zero, a, b) for wn, a, b in windows}
    stab = {wn: show("SPY長揸", spy_r, zero, a, b) for wn, a, b in windows}
    best_cagr = {}
    verdicts = {}
    for un, mask in U.items():
        entry_ok = base_ok & mask
        res, stop_info = {}, {}
        for N in (1, 3, 5):
            r_, t_, n_, st_ = simulate_combo(entry_ok, mask, N)
            res[f"揸{N}隻・組合規則"] = (r_, t_, n_)
            stop_info[N] = st_
        log(f"{un} 完成")
        L.append(f"################ {un} ################")
        L.append("　賣出原因次數：" + "；".join(f"揸{N}隻 " + "，".join(f"{k} {v}" for k, v in st.items()) for N, st in stop_info.items()))
        tab = {}
        for wn, a, b in windows:
            L.append(f"=== {wn}（{a} 至 {b}）===")
            L.append(qtab[wn][1])
            L.append(stab[wn][1])
            for nm, (r, t, nh) in res.items():
                tab[(wn, nm)], txt = show(nm, r, t, a, b)
                L.append(txt + f" ｜ 平均持有 {nh.loc[a:b].mean():.1f} 隻")
            L.append("")
        cands = {k: metrics(r, t, "2000-01-01", "2014-12-31") for k, (r, t, _) in res.items()}
        best, sc = score_pick(cands)
        v, q = tab[("驗證期", best)], qtab["驗證期"][0]
        c1, c2, c3 = v["cagr"] >= q["cagr"] + 3, v["sharpe"] >= q["sharpe"], v["mdd"] > -35
        eras = [tab[(e, best)]["cagr"] > qtab[e][0]["cagr"] for e in ("2000-2009", "2010-2019", "2020-2026")]
        c4 = sum(eras) >= 2
        L.append(f"=== {un} 預先登記判斷 ===")
        L.append("　揀規則期評分：" + "，".join(f"{k} {s:.2f}" for k, s in sc.items()) + f" → 揀 {best}")
        L.append(f"　① 年化 {v['cagr']:+.2f}% vs QQQ {q['cagr']:+.2f}%+3 → {'✔' if c1 else '✘'}")
        L.append(f"　② Sharpe {v['sharpe']:.2f} vs QQQ {q['sharpe']:.2f} → {'✔' if c2 else '✘'}")
        L.append(f"　③ 最大跌幅 {v['mdd']:.1f}%（要好過-35%）→ {'✔' if c3 else '✘'}")
        L.append(f"　④ 三個年代贏QQQ：{sum(eras)}/3 → {'✔' if c4 else '✘'}")
        ok = c1 and c2 and c3 and c4
        L.append(f"　⇒ {best} {'過晒①-④' if ok else '唔合格'}")
        L.append("")
        verdicts[un] = (ok, v["cagr"], q["cagr"])
        good = [k for k in res if tab[("驗證期", k)]["cagr"] >= q["cagr"] + 3 and tab[("驗證期", k)]["mdd"] > -35]
        L.append(f"　3個版本入面，驗證期年化 ≥ QQQ+3 而且最大跌幅好過-35%：{len(good)} 個 {good}")
        L.append("")
        # 每個版本嘅全期年化，用嚟估偏差
        best_cagr[un] = {k: tab[("全期", k)]["cagr"] for k in res}
    k1, k2 = "U1 當時真實標普500", "U2 今日標普500（有偏差）"
    if k1 in best_cagr and k2 in best_cagr:
        diffs = {k: best_cagr[k2][k] - best_cagr[k1][k] for k in best_cagr[k1]}
        bias = float(np.median(list(diffs.values())))
        L.append("=== 倖存者偏差估計（今日標普500名單 − 當時真實名單，全期年化差距）===")
        L.append("　" + "，".join(f"{k} {d:+.1f}%" for k, d in diffs.items()) + f" → 中位數 {bias:+.1f}% 一年")
        k3 = "U3 全美國（有偏差，上限）"
        if k3 in verdicts:
            ok, vc, qc = verdicts[k3]
            adj = vc - bias
            L.append(f"　U3 驗證期年化扣除偏差：{vc:+.2f}% − {bias:.1f}% = {adj:+.2f}%（要 ≥ QQQ {qc:+.2f}%+3）→ "
                     f"{'值得模擬交易' if ok and adj >= qc + 3 else '唔合格'}")
        L.append("")

    pd.DataFrame({"report": L}).to_csv("v15d_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
