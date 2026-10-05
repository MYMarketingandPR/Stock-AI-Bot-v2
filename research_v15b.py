"""
research_v15b.py — 第十五輪B：倍升股均線輪換，股票池擴大到「全美國上市股票」（包括中細價股）

點解：Esther 指出標普500股票入選時通常已經升咗一大段，好多幾倍股最勁嗰段係中細價股時期。
限制：冇免費「當時」全市場名單，已除牌公司Yahoo亦冇價 → 只可以用「今日仍然上市」嘅股票 → 結果偏好（倖存者偏差）。
  處理方法：將全市場結果當「最好情況（上限）」：上限都贏唔到QQQ = 肯定唔得；贏到 = 未證實，要模擬交易驗證。
  另外用同一方法比較「今日標普500名單（有偏差）」vs「當時真實標普500名單（冇偏差）」，量度偏差大約有幾大。

⚠️ 預先登記（2026-10-05 跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：

股票池（三個）：
  U1 當時真實標普500成分股（冇倖存者偏差）
  U2 今日標普500成分股，套落成段歷史（有偏差，用嚟量度偏差）
  U3 今日全美國上市普通股（nasdaqtrader.com 名單：剔除ETF、測試代號、認股證、單位、優先股、權利、票據），
     每日要符合：過去50日平均每日成交額 ≥ 500萬美元、有最少252日價錢（有偏差，當上限）
規則（同 research_v15.py 第一部分一模一樣）：
  倍升股 = 過去252個交易日升 ≥ +100%；買入：倍升股 + 收市價 > MA50 + > MA200 + 喺股票池入面；
  賣出（三個版本）：X50 收市跌穿MA50 / X200 收市跌穿MA200 /
    X50二（Esther 2026-10-05 10:35 加，跑之前）：買入之後，最近10個交易日內第2次收市低過MA50先賣
    （例如跌穿一日之後即刻企返上去，唔賣；10日內再收市低過MA50 = 賣；一直企喺MA50下面 = 第2日賣）；
  有空位揀12個月升幅最大嘅；賣咗可以再買；每位 = 總資產 ÷ 位數；
  收市決定→下一個收市交易；每次買賣0.1%；冇股揸現金（^IRX）。版本：1/3/5隻 × X50/X200/X50二 = 9個。
判斷（每個股票池各自）：揀規則期 2000-2014 用評分揀1個 → 驗證期 2015-2026：
  ① 年化 ≥ QQQ+3 ② Sharpe ≥ QQQ ③ 最大跌幅好過 -35% ④ 三個年代最少兩個贏QQQ
  U3 先要過晒①-④，再扣除「偏差估計」（U2減U1嘅年化差距）之後①仍然成立，先叫「值得模擬交易」。
另外（純描述）：U3 每月尾按過去12個月升幅分組（跌咗/0-50%/50-100%/2-3倍/3-5倍/5倍以上），
  睇之後 6個月/1年/2年 中位數、平均、贏SPY機率、跌≥30%機率。
輸出：v15b_results.csv、v15b_progress.txt
==== 結果（2026-10-05 第一次跑，記錄低，唔再改）====
- 全美國名單5812隻，有價錢5379隻；U3每日合資格：2000年408隻、2010年941隻、2025年2619隻。
- U1（當時標普500）：照規則揀咗「揸5隻・跌穿MA200賣」→ 唔合格（同第十五輪一樣）。
  「10日內第2次低過MA50賣」：揸1隻 驗證期 +26.7%（跌-60.9%）、2000-09 每年 -11.6%；揸5隻 +15.2%（跌-37.7%）→ 冇好過MA200。
- U3（全美國，有偏差都照計）：慘敗。揀咗「揸3隻・跌穿MA200賣」驗證期 -1.6%/年，最大跌幅 -94.9%；
  其他版本大部分最大跌幅 -90%至-100%。原因：喺全市場揀「12個月升最多」= 揀最瘋狂嘅泡沫股（升5倍以上嘅之後2年中位數 -57%）。
- 倖存者偏差：用今日標普500名單（U2）比當時真實名單（U1），每年多咗 +19%至+31%（中位數 +23.5%）→ 用今日名單做回測幾乎冇參考價值。
- U3 描述：過去12個月升2-3倍 → 之後1年中位數 -2%、26%跌≥30%；3-5倍 → -16%；5倍以上 → -33%（52%跌≥30%）。
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

M.PROGRESS = "v15b_progress.txt"
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
        entered = {}
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
                if exit_ma == "50x2":
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

    L = ["=== 第十五輪B：倍升股均線輪換，三個股票池（規則預先登記，見research_v15b.py開頭）===",
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
        res = {}
        for N in (1, 3, 5):
            for x in (50, 200, "50x2"):
                lab = f"揸{N}隻・10日內第2次低過MA50賣" if x == "50x2" else f"揸{N}隻・跌穿MA{x}賣"
                res[lab] = simulate(entry_ok, N, x)
        log(f"{un} 完成")
        L.append(f"################ {un} ################")
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

    # ---------- U3 已升幾倍之後 ----------
    if "U3 全美國（有偏差，上限）" in U:
        m3 = U["U3 全美國（有偏差，上限）"]
        L.append("=== U3：已經升咗幾倍之後先買，之後點？（每月尾抽樣，有倖存者偏差，偏好）===")
        spy_v = spy.reindex(days).values
        me = [days.searchsorted(d, side="right") - 1 for d in pd.date_range("2000-01-31", "2025-09-30", freq="ME")]
        bins = [-np.inf, 0, 0.5, 1.0, 2.0, 4.0, np.inf]
        labs = ["跌咗", "升0-50%", "升50-100%", "2-3倍", "3-5倍", "5倍以上"]
        acc = {lb: {h: [] for h in (126, 252, 504)} for lb in labs}
        for i in me:
            cs = np.where(m3[i] & np.isfinite(Pv[i]))[0]
            b = np.digitize(r12[i, cs], bins[1:-1])
            for h in (126, 252, 504):
                e = i + h
                if e >= len(days):
                    continue
                f = Pf[e, cs] / Pv[i, cs] - 1
                s = spy_v[e] / spy_v[i] - 1
                for k, lb in enumerate(labs):
                    sel = b == k
                    if sel.any():
                        acc[lb][h].append(np.column_stack([f[sel], np.full(sel.sum(), s)]))
        for lb in labs:
            parts = []
            n = 0
            for h, hl in ((126, "6個月"), (252, "1年"), (504, "2年")):
                if not acc[lb][h]:
                    continue
                x = np.vstack(acc[lb][h])
                x = x[np.isfinite(x[:, 0])]
                n = max(n, len(x))
                parts.append(f"{hl} 中位數 {np.median(x[:, 0]) * 100:+.0f}% 平均 {x[:, 0].mean() * 100:+.0f}% "
                             f"贏SPY {(x[:, 0] > x[:, 1]).mean() * 100:.0f}% 跌≥30% {(x[:, 0] <= -0.3).mean() * 100:.0f}%")
            L.append(f"　{lb}（{n}）：" + " ｜ ".join(parts))
    pd.DataFrame({"report": L}).to_csv("v15b_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
