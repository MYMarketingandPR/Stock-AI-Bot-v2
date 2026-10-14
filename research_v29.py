"""
research_v29.py — 第二十九輪A：集中揸1至3隻 + 期權（Esther 2026-10-14：「集中揸1至3隻、期權，或者買細價股。我覺得可以試」）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同：當時真實標普500；已計股息；收市決定 → 下一個交易日收市買賣；每次買賣股票0.1%；冇投資嘅錢揸現金（^IRX）。

━━ A 集中（線E規則）━━
  買：今年升 ≥30% + 收市 ≥ 52週最高收市價 × 0.7 + 收市 > MA200（揀今年升最多）；
  賣：收市 < MA200 或者 < 52週最高收市價 × 0.7。每隻 = 買入日總資產 ÷ 揸幾多隻。
  K5 揸5隻（= 線E，對照）  K3 揸3隻  K2 揸2隻  K1 揸1隻  K3S 揸3隻 + SPY每日訊號持有先開新倉

━━ C 期權（⚠️ 冇真實歷史期權價，用公式估算，只作參考）━━
  價錢：Black-Scholes 公式；波幅 = VIX（SPY）／VXN（QQQ，2001年起有）÷100；利率 = ^IRX；唔計股息（令認購期權估貴咗少少，偏保守）；
    正股用實際收市價（唔計股息）。每次買賣期權收 2% 期權金做買賣差價。
  做法：每月最後一個交易日、或者 SPY每日訊號轉變嗰日，收市決定，下一個交易日收市執行：
    賣走手上期權（按公式估值），如果SPY訊號 = 持有 → 用總資產 X% 買 3個月（63個交易日）到期、行使價 = 當時價（平價）嘅認購期權；
    其餘資金揸現金。訊號 = 現金 → 全部現金。
  O1 SPY認購 X=10%   O2 SPY認購 X=20%   O3 QQQ認購 X=10%   O4 QQQ認購 X=20%
  O5 QQQ認購 X=20% 唔理SPY訊號（每月都買，對照）
  （大約槓桿：X=10% ≈ 1.2至1.5倍正股；X=20% ≈ 2.5至3倍。最多每月蝕晒 X% 期權金。）

判斷（同之前一樣）：① 四條問題（驗證期 2015-01 至 2026-09 對比QQQ）② 跨年代合格（2000-2009、2010-2014、2015-2019、2020-2026
  每段年化 ≥ SPY 而且最大跌幅好過 -35%）③ 列出有冇版本驗證期年化 ≥ 50% 而且最大跌幅好過 -35%（Esther 目標）。
  另外：最差5年年化、每次買賣統計（A部分）。10個版本，試得多總有幾個好彩。
輸出：v29_results.csv、v29_progress.txt
"""
import math

import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim

M.PROGRESS = "v29_progress.txt"
log = M.log
END = "2026-09-30"
OPT_COST = 0.02


def bs_call(S, K, T, r, sig):
    if T <= 0:
        return max(S - K, 0.0)
    sig = max(sig, 0.05)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return S * N(d1) - K * math.exp(-r * T) * N(d2)


def opt_sim(S, sig, r, cr, hold_sig, budget, decide):
    """S 正股收市、sig 年化波幅、r 年利率、cr 每日現金回報、hold_sig 每日收市後SPY訊號（True=持有）、decide 每日係咪決定日。"""
    T = len(S)
    cash, pos = 1.0, None
    navs = []
    pend = None
    for i in range(T):
        cash *= 1 + cr[i]
        val = 0.0
        if pos is not None and np.isfinite(S[i]) and np.isfinite(sig[i]):
            val = pos["n"] * bs_call(S[i], pos["K"], (pos["exp"] - i) / 252, r[i], sig[i])
            if pos["exp"] - i <= 0:
                cash += val * (1 - OPT_COST)
                pos, val = None, 0.0
        if pend is not None and np.isfinite(S[i]) and np.isfinite(sig[i]):
            if pos is not None:
                cash += val * (1 - OPT_COST)
                pos, val = None, 0.0
            if pend:
                prem = budget * cash
                price = bs_call(S[i], S[i], 63 / 252, r[i], sig[i])
                if price > 0:
                    n = prem * (1 - OPT_COST) / price
                    cash -= prem
                    pos = {"n": n, "K": S[i], "exp": i + 63}
                    val = n * price
            pend = None
        navs.append(cash + val)
        if decide[i]:
            pend = bool(hold_sig[i])
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1]


def main():
    import yfinance as yf
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
    ma200 = px.rolling(200).mean().values
    ye = px.groupby(days.year).last()
    ytdn = np.nan_to_num(P / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    hi52 = px.rolling(252, min_periods=200).max().values
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
    e_ok = mem & (ytdn >= 0.30) & (P >= np.nan_to_num(hi52, nan=np.inf) * 0.7) & (P > ma200) & np.isfinite(P)
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    sig_state = (df["state"] != "out").reindex(days).fillna(False).values.astype(bool)
    log("股票數據準備好")

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    V, TR = {}, {}
    for nm, kw in (("K5 線E揸5隻（對照）", dict(n=5)), ("K3 線E揸3隻", dict(n=3)), ("K2 線E揸2隻", dict(n=2)),
                   ("K1 線E揸1隻", dict(n=1)), ("K3S 線E揸3隻+SPY訊號", dict(n=3, allow=sig_state))):
        r, tr = sim(P, rets_v, cr, e_ok, ytdn, broken, **kw)
        V[nm], TR[nm] = pd.Series(r, index=days), tr
        log(f"{nm} 完成")

    # ---------- 期權 ----------
    def raw(t):
        h = yf.Ticker(t).history(start="1998-01-01", end="2026-10-01", auto_adjust=False)["Close"]
        h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
        return h.reindex(days).ffill()
    S_spy, S_qqq = raw("SPY").values, raw("QQQ").values
    vix, vxn = raw("^VIX").values / 100, raw("^VXN").values / 100
    irx = raw("^IRX").fillna(0.0).values / 100
    sig_ch = np.r_[False, sig_state[1:] != sig_state[:-1]]
    me = np.zeros(len(days), dtype=bool)
    me[[days.get_loc(d) for d in decision_days(days, "每月")]] = True
    decide = me | sig_ch
    always = np.ones(len(days), dtype=bool)
    for nm, S, sg, b, hs in (("O1 SPY認購10%（SPY訊號）", S_spy, vix, 0.10, sig_state), ("O2 SPY認購20%（SPY訊號）", S_spy, vix, 0.20, sig_state),
                             ("O3 QQQ認購10%（SPY訊號）", S_qqq, vxn, 0.10, sig_state), ("O4 QQQ認購20%（SPY訊號）", S_qqq, vxn, 0.20, sig_state),
                             ("O5 QQQ認購20%（唔理訊號）", S_qqq, vxn, 0.20, always)):
        r = opt_sim(S, sg, irx, cr, hs, b, decide if hs is sig_state else me)
        s_ = pd.Series(r, index=days)
        first = np.where(np.isfinite(sg))[0]
        if len(first):
            s_.iloc[:first[0] + 1] = np.nan
        V[nm] = s_
        log(f"{nm} 完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    allser.update(V)
    tab = {}
    L = ["=== 第二十九輪A：集中揸1至3隻 + 期權（規則預先登記，見research_v29.py開頭）===",
         f"股票池：當時真實標普500（{source}）；期權價用公式估算（冇真實歷史期權價），只作參考", ""]
    for wn, a, b in windows:
        L.append(f"=== {wn}（{a} 至 {b}）===")
        for nm, r in allser.items():
            rr = r.loc[a:b].dropna()
            if len(rr) < 60:
                continue
            p = perf(rr, rf=cash_r, bench=qqq_r.loc[a:b])
            tab[(wn, nm)] = p
            note = f"（由{rr.index[0].date()}）" if rr.index[0] > pd.Timestamp(a) + pd.Timedelta(days=10) else ""
            L.append(f"　{nm}{note}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}")
        L.append("")
    L.append("=== 每次買賣（A部分，1999-2026）===")
    for nm, tr in TR.items():
        tr = np.array(tr)
        L.append(f"　{nm}：{len(tr)} 次 ｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
                 f"蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}%")
    L.append("")

    def worst5(r):
        nav = (1 + r.loc["2000-01-03":END].dropna()).cumprod()
        out = []
        for d0 in decision_days(nav.index, "每月"):
            d1 = d0 + pd.DateOffset(years=5)
            if d1 > nav.index[-1]:
                break
            j0, j1 = nav.index.get_loc(d0), nav.index.searchsorted(d1) - 1
            out.append((nav.iat[j1] / nav.iat[j0]) ** (1 / 5) - 1)
        return min(out) * 100 if out else np.nan

    eras = ["2000-2009", "2010-2014", "2015-2019", "2020-2026"]
    q, s = tab[("驗證期", "QQQ長揸")], tab[("驗證期", "SPY長揸")]
    L.append(f"=== 預先登記判斷 ｜ QQQ 驗證期 {q['cagr']:+.2f}% / {q['mdd']:.1f}% / Sharpe {q['sharpe']:.2f} ｜ SPY {s['cagr']:+.2f}% ===")
    goal = []
    for nm in V:
        x = tab[("驗證期", nm)]
        e = tab.get(("2000-2009", nm))
        marks = "".join("✔" if ((er, nm) in tab and tab[(er, nm)]["cagr"] >= tab[(er, "SPY長揸")]["cagr"] and tab[(er, nm)]["mdd"] > -35) else "✘" for er in eras)
        q1 = x["cagr"] > q["cagr"]
        q2 = q1 and x["mdd"] > -35 and x["sharpe"] >= q["sharpe"]
        q3 = (not q1) and x["mdd"] >= q["mdd"] + 5 and x["cagr"] >= s["cagr"]
        q4 = e is not None and e["mdd"] > -35
        if x["cagr"] >= 50 and x["mdd"] > -35:
            goal.append(nm)
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} ｜ 四條問題 ①{'✔' if q1 else '✘'} "
                 f"②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'} ⇒ {'保留' if (q2 or q3) else '淘汰'} ｜ "
                 f"跨年代 {marks} {'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 "
                 + " / ".join(f"{tab[(er, nm)]['cagr']:+.1f}%" if (er, nm) in tab else "—" for er in eras) + " ｜ 各段最大跌幅 "
                 + " / ".join(f"{tab[(er, nm)]['mdd']:.0f}%" if (er, nm) in tab else "—" for er in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
    L.append(f"　Esther目標（驗證期 ≥50% 而且最大跌幅好過 -35%）：{'、'.join(goal) if goal else '冇任何版本做到'}")
    pd.DataFrame({"report": L}).to_csv("v29_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
