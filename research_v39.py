"""
research_v39.py — 第三十九輪：強勢股大跌 30-50% 之後「企穩先撈底」（Esther 2026-10-26）
  「每日喺第37輪嘅選股揀10隻出嚟，留意住佢，當佢回撤30%至50%，睇吓有冇止跌，止跌就買入，買入價跌8%就止損」
  確認：止跌用 A 款；止賺 = 「由買入後最高收市跌20%」或者「升返到52週高位95%」邊個先到用邊個。

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同：當時真實標普500（fja05680）；已計股息；收市決定 → 下一個交易日收市買賣；每次買賣0.1%；揸最多5隻、
  每隻 = 買入日總資產 ÷ 5（唔借錢）；閒錢收^IRX。多過空位就揀過去12個月升幅最大嘅。
觀察名單：每日「回調訊號」（同第三十七輪／每日掃描：線E條件 + 20日線向上 + 收市 ≥ 20日線 + 最低 ≤ 10日線×1.01；
  止蝕距離 > 8% 剔走）按今年升幅揀頭10隻加入名單；加入後觀察 126 個交易日（再次入選就延長）。
回撤：由加入名單之後嘅最高收市（高位）計；高位之後嘅最低收市（谷底）。谷底比高位低 > 50% → 剔出名單（當佢壞咗）。
止跌（A款）：谷底比高位低 30% 至 50% + 谷底係最少 10 個交易日之前（最近10日冇再創新低）+ 收市 > 20日平均
  + 收市 < 52週最高收市 × 0.95（未返到高位）→ 買入訊號（每次觀察期每隻最多買一次）。
止蝕：買入價 × 0.92（盤中執行：開市已經低過就用開市價，否則最低價掂到就用止蝕價）。
止賺（邊個先到用邊個，收市決定、下一個交易日收市賣）：收市 ≤ 買入後最高收市 × 0.8；或者 收市 ≥ 52週最高收市 × 0.95。
版本（只有呢兩個，跑完唔加唔減唔改數字）：
  V1 照上面規則
  V2 = V1 + SPY每日訊號「持有」先買（大市過濾，同線D2一樣）
對照：線E、線E3（線E + SPY閘）、QQQ、SPY。

判斷（驗證期 2015-01 至 2026-09）：同第三十三輪一樣：同線E比「有改善」、四條問題、跨年代合格、最差5年年化。
另外列：每年加入名單幾多次、出現回撤30-50%幾多次、買入訊號幾多個；每單勝率／贏平均／輸平均／平均揸幾日；
  止蝕、止賺各佔幾多；訊號之後60個交易日中位數同贏SPY機率。
就算合格都唔會直接改模擬交易，要你同意先加新線。

已知限制：已除牌股冇數據唔入選（真正爆煲嘅撈底股唔喺數據入面 → 結果可能偏樂觀）；同一日先掂止蝕定先升分唔到（當先掂止蝕）。

輸出：v39_results.csv、v39_progress.txt
==== 結果（2026-10-26 第一次跑，run 40323271699，記錄低，唔再改）====
- 加入觀察名單 40,445 次（每年約1,512）；回撤到30-50% 613 次；企穩買入訊號 500 個（加SPY閘 366 個）。
- 訊號之後60日：中位數 +5.46%（所有成分股 +2.95%）、贏SPY 53%，但跌≥20% 有 13%（所有成分股 6%）→ 平均好啲但大跌機會加倍。
- 每單：V1 333單（每年約12單）勝率30%、贏 +26.0%／輸 -8.2%、平均 +2.04%、最差 -31%（裂口）；離場：止蝕213、返到高位95% 89、高位跌20% 31。
  V2 253單，勝率30%、平均 +1.99%。
- 驗證期：V1 +7.65% / -34.6%；V2 +5.88% / -32.5%；線E +28.74% / -32.6% → 冇改善、淘汰、跨年代唔合格。
- 金融海嘯：V2 只跌 3.8%（大市過濾令佢差唔多冇買）；2025年 V2 +22.9% / -19.8%。
- 意思：撈企穩嘅前強勢股，之後兩三個月平均的確好過隨便揀，但三分之二嘅單被 8% 止蝕掃走；每年只有十零單，大部分時間錢閒置，總回報遠低過線E。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim as sim_close
from research_v37 import ohlc

M.PROGRESS = "v39_progress.txt"
log = M.log
END = "2026-09-30"
COST = 0.001
N = 5
WATCH = 126


def sim_dip(O, Lo, C, cr, sig, rank, hi52):
    T, K = C.shape
    cash = 1.0
    hold = {}
    pend_buy, pend_sell = [], []
    navs, trades, days_held, kinds = [], [], [], []
    i = 0

    def close_pos(c, px, kind):
        nonlocal cash
        h = hold.pop(c)
        cash += h["sh"] * px * (1 - COST)
        trades.append(px / h["px"] - 1)
        days_held.append(i - h["i0"])
        kinds.append(kind)

    for i in range(T):
        cash *= 1 + cr[i]
        for c in list(hold):
            h = hold[c]
            if i <= h["i0"]:
                continue
            o, lo = O[i, c], Lo[i, c]
            if np.isfinite(o) and o <= h["stop"]:
                close_pos(c, o, "止蝕")
            elif np.isfinite(lo) and lo <= h["stop"]:
                close_pos(c, h["stop"], "止蝕")
        for c, kind in pend_sell:
            if c in hold:
                close_pos(c, C[i, c] if np.isfinite(C[i, c]) else hold[c]["last"], kind)
        for c in list(hold):
            if not np.isfinite(C[i, c]) and (i + 1 >= T or not np.isfinite(C[i + 1, c])):
                close_pos(c, hold[c]["last"], "除牌")
        equity = cash + sum(h["sh"] * (C[i, c] if np.isfinite(C[i, c]) else h["last"]) for c, h in hold.items())
        for c in pend_buy:
            p = C[i, c]
            if c in hold or len(hold) >= N or not np.isfinite(p):
                continue
            amt = min(equity / N, cash)
            if amt <= equity * 0.01:
                break
            cash -= amt
            hold[c] = {"sh": amt * (1 - COST) / p, "px": float(p), "stop": float(p) * 0.92, "i0": i, "last": float(p), "peak": float(p)}
        pend_buy, pend_sell = [], []
        for c, h in hold.items():
            if np.isfinite(C[i, c]):
                h["last"] = float(C[i, c])
                h["peak"] = max(h["peak"], float(C[i, c]))
        navs.append(cash + sum(h["sh"] * h["last"] for h in hold.values()))
        for c, h in hold.items():
            p = C[i, c]
            if not np.isfinite(p):
                continue
            if np.isfinite(hi52[i, c]) and p >= hi52[i, c] * 0.95:
                pend_sell.append((c, "返到高位95%"))
            elif p <= h["peak"] * 0.8:
                pend_sell.append((c, "高位跌20%"))
        free = N - (len(hold) - len(pend_sell))
        if free > 0:
            cand = [c for c in np.where(sig[i])[0] if c not in hold]
            cand.sort(key=lambda c: -rank[i, c])
            pend_buy = cand[:free]
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1], np.array(trades), np.array(days_held), kinds


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    closes = M.download_close(["SPY", "QQQ"], "1998-01-01", "2026-10-01")
    spy, qqq = closes["SPY"], closes["QQQ"]
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    data = ohlc(everyone)
    cols = sorted(data)
    O = pd.DataFrame({t: data[t]["Open"] for t in cols}).reindex(days).values
    Lo = pd.DataFrame({t: data[t]["Low"] for t in cols}).reindex(days)
    px = pd.DataFrame({t: data[t]["Close"] for t in cols}).reindex(days)
    P = px.values
    T, K = P.shape
    Lv = Lo.values
    rets_v = px.pct_change(fill_method=None).values
    ma10, ma20, ma200 = px.rolling(10).mean().values, px.rolling(20).mean().values, px.rolling(200).mean().values
    ma20_5 = px.rolling(20).mean().shift(5).values
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
    e_ok = mem & (ytdn >= 0.30) & (P >= hi52f * 0.7) & np.nan_to_num(P > ma200, nan=False) & fin
    with np.errstate(invalid="ignore", divide="ignore"):
        setup = (e_ok & np.nan_to_num(ma20 > ma20_5, nan=False) & np.nan_to_num(P >= ma20, nan=False)
                 & np.nan_to_num(Lv <= ma10 * 1.01, nan=False) & np.nan_to_num(1 - Lv / P <= 0.08, nan=False))
    # 觀察名單 → 買入訊號
    sig = np.zeros(P.shape, dtype=bool)
    watch = {}           # c -> dict(end, peak, trough, ti, bought)
    adds = dd_hits = 0
    for i in range(T):
        idx = np.where(setup[i])[0]
        if len(idx):
            top = idx[np.argsort(-ytdn[i, idx], kind="stable")[:10]]
            for c in top:
                adds += 1
                if c in watch:
                    watch[c]["end"] = i + WATCH
                else:
                    watch[c] = {"end": i + WATCH, "peak": P[i, c], "trough": P[i, c], "ti": i, "bought": False, "hit": False}
        for c in list(watch):
            w = watch[c]
            p = P[i, c]
            if i > w["end"]:
                del watch[c]
                continue
            if not np.isfinite(p):
                continue
            if p > w["peak"]:
                w["peak"], w["trough"], w["ti"] = p, p, i
            elif p < w["trough"]:
                w["trough"], w["ti"] = p, i
            dd = 1 - w["trough"] / w["peak"]
            if dd > 0.50:
                del watch[c]
                continue
            if dd >= 0.30 and not w["hit"]:
                w["hit"] = True
                dd_hits += 1
            if (dd >= 0.30 and i - w["ti"] >= 10 and np.isfinite(ma20[i, c]) and p > ma20[i, c]
                    and np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.95 and not w["bought"] and mem[i, c]):
                sig[i, c] = True
                w["bought"] = True
    import main as bot
    sdf = bot.replay(bot.add_signals(bot.load_data()))
    spy_ok = (sdf["state"] != "out").reindex(days).fillna(False).values
    sig2 = sig & spy_ok[:, None]
    yrs = 26.75
    log(f"加入名單 {adds} 次（每年約 {adds / yrs:.0f}）；回撤30-50% {dd_hits} 次；買入訊號 {sig.sum()} 個；加SPY閘 {sig2.sum()} 個")

    def trail20(i, c, p, pk):
        return p <= pk * 0.8

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b_ = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b_

    S, ST = {}, {}
    S["線E（對照）"] = pd.Series(sim_close(P, rets_v, cr, e_ok, ytdn, broken)[0], index=days)
    S["線E3 線E+SPY閘（對照）"] = pd.Series(sim_close(P, rets_v, cr, e_ok, ytdn, broken, allow=spy_ok)[0], index=days)
    for nm, sg in [("V1 大跌30-50%企穩先買", sig), ("V2 V1＋SPY大市過濾", sig2)]:
        r, tr, hd, kd = sim_dip(O, Lv, P, cr, sg, r12n, hi52)
        S[nm] = pd.Series(r, index=days)
        ST[nm] = (tr, hd, kd)
        log(f"{nm} 完成")

    L = ["=== 第三十九輪：強勢股大跌30-50%企穩先撈底（規則預先登記，見research_v39.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻；有OHLC數據 {len(cols)} 隻", "",
         f"加入觀察名單 {adds} 次（每年約 {adds / yrs:.0f}）；其中回撤到30-50% {dd_hits} 次；企穩買入訊號 {sig.sum()} 個（加SPY閘 {sig2.sum()} 個）"]
    f = np.full(P.shape, np.nan)
    f[:-61] = P[61:] / P[1:-60] - 1
    sf = np.full(T, np.nan)
    sv = spy.values
    sf[:-61] = sv[61:] / sv[1:-60] - 1
    for lab, m in [("所有成分股", mem & fin), ("買入訊號", sig), ("買入訊號＋SPY閘", sig2)]:
        ii, cc = np.where(m & np.isfinite(f))
        x = f[ii, cc]
        if len(x):
            L.append(f"　之後60日｜{lab}：{len(x)} 次 ｜ 中位數 {np.median(x) * 100:+.2f}% ｜ 平均 {x.mean() * 100:+.2f}% ｜ 贏SPY {(x > sf[ii]).mean() * 100:.0f}% ｜ 跌≥20% {(x <= -0.2).mean() * 100:.0f}%")
    L.append("")
    L.append("=== 每單統計（1999-2026）===")
    for nm, (tr, hd, kd) in ST.items():
        if len(tr) == 0:
            L.append(f"　{nm}：冇交易")
            continue
        w, l_ = tr[tr > 0], tr[tr <= 0]
        kc = pd.Series(kd).value_counts()
        L.append(f"　{nm}：{len(tr)} 單（每年約 {len(tr) / 27:.0f} 單）｜ 勝率 {(tr > 0).mean() * 100:.0f}% ｜ 贏平均 {w.mean() * 100:+.1f}% ｜ "
                 f"輸平均 {l_.mean() * 100:+.1f}% ｜ 平均每單 {tr.mean() * 100:+.2f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}% ｜ "
                 f"平均揸 {hd.mean():.1f} 日 ｜ 離場：" + "、".join(f"{k} {v}" for k, v in kc.items()))
    L.append("")
    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31"),
               ("2025年", "2025-01-01", "2025-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r, **S}
    tab = {}
    for wn, a_, b_ in windows:
        L.append(f"=== {wn}（{a_} 至 {b_}）===")
        for nm, r in allser.items():
            rr = r.loc[a_:b_].dropna()
            p = perf(rr, rf=cash_r, bench=qqq_r.loc[a_:b_])
            tab[(wn, nm)] = p
            L.append(f"　{nm}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}")
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
    b0 = tab[("驗證期", "線E（對照）")]
    L.append(f"=== 預先登記判斷 ｜ 線E {b0['cagr']:+.2f}% / {b0['mdd']:.1f}% ｜ QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% / Sharpe {q['sharpe']:.2f} ｜ SPY {s['cagr']:+.2f}% ===")
    for nm in S:
        x, ee = tab[("驗證期", nm)], tab[("2000-2009", nm)]
        marks = "".join("✔" if (tab[(er, nm)]["cagr"] >= tab[(er, "SPY長揸")]["cagr"] and tab[(er, nm)]["mdd"] > -35) else "✘" for er in eras)
        imp = "—"
        if "對照" not in nm:
            better = (x["mdd"] >= b0["mdd"] + 3 and x["cagr"] >= b0["cagr"] - 1) or (x["cagr"] >= b0["cagr"] + 2 and x["mdd"] >= b0["mdd"] - 2)
            imp = "有改善" if better else "冇改善"
        q1 = x["cagr"] > q["cagr"]
        q2 = q1 and x["mdd"] > -35 and x["sharpe"] >= q["sharpe"]
        q3 = (not q1) and x["mdd"] >= q["mdd"] + 5 and x["cagr"] >= s["cagr"]
        q4 = ee["mdd"] > -35
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} ｜ 同線E比 {imp} ｜ "
                 f"四條問題 ①{'✔' if q1 else '✘'} ②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'} ⇒ "
                 f"{'保留' if (q2 or q3) else '淘汰'} ｜ 跨年代 {marks} {'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 "
                 + " / ".join(f"{tab[(er, nm)]['cagr']:+.1f}%" for er in eras) + " ｜ 各段最大跌幅 "
                 + " / ".join(f"{tab[(er, nm)]['mdd']:.0f}%" for er in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
    pd.DataFrame({"report": L}).to_csv("v39_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
