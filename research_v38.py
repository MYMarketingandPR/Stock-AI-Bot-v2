"""
research_v38.py — 第三十八輪：「放量突破箱體 → 縮量整固唔跌返落箱 → 買」＋板塊效應（Esther 2026-10-24：「我想做第38輪測試」，
  參考宋建毅「買入分歧，賣出一致」、「冇板塊效應嘅唔買」）

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同：當時真實標普500（fja05680）；已計股息（收市價）；收市決定 → 下一個交易日收市買賣；每次買賣0.1%；揸最多5隻、
  每隻 = 買入日總資產 ÷ 5（唔借錢）；閒錢收^IRX。多過空位就揀過去12個月升幅最大嘅。
箱體：突破日之前 40 個交易日嘅最高收市（箱頂）同最低收市（箱底），箱頂 ÷ 箱底 − 1 ≤ 25%（即係橫行）。
放量突破（第 b 日）：收市 > 箱頂 + 成交量 ≥ 之前50日平均成交量 × 2 + 收市 > 200日平均 + 50日平均向上（今日 > 10日前）。
縮量整固（分歧，第 i 日，3 ≤ i − b ≤ 10）：突破後每日收市都 ≥ 箱頂（冇跌返落箱）+ 突破後（唔計突破日）平均成交量 ≤ 突破日之前50日平均
  → 第一日符合就係買入訊號（每次突破最多一次），下一個交易日收市買入。
止蝕（所有版本）：收市跌穿箱頂（關鍵支持）→ 下一個交易日收市賣。
版本（只有呢4個，跑完唔加唔減唔改數字）：
  V1 賺錢離場：由買入後最高收市跌20%（同線D）
  V2 賺錢離場：收市跌穿50日平均
  V3 「賣出一致」：收市高過50日平均 30% 或以上（升到好熱）就賣；否則同 V2（跌穿50日平均賣）
  V4 = V1 + 板塊效應：同一行業（今日 GICS 分類；冇分類 = 唔符合）喺訊號日或之前10個交易日內，最少有另外2隻都出現放量突破先買
對照：線E、線D、QQQ、SPY。

判斷（驗證期 2015-01 至 2026-09）：同第三十三輪一樣：同線E比「有改善」、四條問題、跨年代合格（2000-09、2010-14、2015-19、
  2020-26 每段年化 ≥ SPY 而且最大跌幅好過 -35%）、最差5年年化。另外列：每年幾多個訊號、每單勝率／贏平均／輸平均／平均揸幾日；
  訊號之後 20 同 60 個交易日表現（中位數、贏SPY機率）對比所有成分股。
就算合格都唔會直接改模擬交易，要你同意先加新線。

已知限制：行業用今日分類；已除牌股冇數據唔入選；A股嘅「題材」「散戶情緒」無法量化，冇測。

輸出：v38_results.csv、v38_progress.txt
"""
import numpy as np
import pandas as pd

import data_cache as DC
import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim as sim_close
from research_v33 import ETF, sectors

M.PROGRESS = "v38_progress.txt"
log = M.log
END = "2026-09-30"
COST = 0.001
N = 5


def volumes(tickers, start="1998-01-01", end="2026-10-01"):
    out = {}
    for t in tickers:
        try:
            h = DC.history(t, start, end)
        except Exception:
            continue
        if h is not None and len(h) and "Volume" in h:
            v = h["Volume"].astype(float)
            v.index = pd.DatetimeIndex(v.index).tz_localize(None).normalize() if getattr(v.index, "tz", None) else pd.DatetimeIndex(v.index).normalize()
            out[t] = v[~v.index.duplicated(keep="last")]
    log(f"成交量：{len(out)}/{len(tickers)} 隻")
    return out


def sim_box(C, cr, sig, rank, support, exit_kind, ma50):
    """sig[i,c]：第 i 日收市出現買入訊號；support[i,c]：嗰次嘅箱頂。exit_kind: 'trail20' / 'ma50' / 'hot30'。"""
    T, K = C.shape
    cash = 1.0
    hold = {}
    pend_buy, pend_sell = [], []
    navs, trades, days_held = [], [], []
    i = 0

    def close_pos(c, px):
        nonlocal cash
        h = hold.pop(c)
        cash += h["sh"] * px * (1 - COST)
        trades.append(px / h["px"] - 1)
        days_held.append(i - h["i0"])

    for i in range(T):
        cash *= 1 + cr[i]
        for c in pend_sell:
            if c in hold:
                close_pos(c, C[i, c] if np.isfinite(C[i, c]) else hold[c]["last"])
        for c in list(hold):
            if not np.isfinite(C[i, c]) and (i + 1 >= T or not np.isfinite(C[i + 1, c])):
                close_pos(c, hold[c]["last"])
        equity = cash + sum(h["sh"] * (C[i, c] if np.isfinite(C[i, c]) else h["last"]) for c, h in hold.items())
        for c, sup in pend_buy:
            p = C[i, c]
            if c in hold or len(hold) >= N or not np.isfinite(p) or p < sup:
                continue
            amt = min(equity / N, cash)
            if amt <= equity * 0.01:
                break
            cash -= amt
            hold[c] = {"sh": amt * (1 - COST) / p, "px": float(p), "sup": float(sup), "i0": i, "last": float(p), "peak": float(p)}
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
            m = ma50[i, c]
            if p < h["sup"]:
                pend_sell.append(c)
            elif exit_kind == "trail20" and p <= h["peak"] * 0.8:
                pend_sell.append(c)
            elif exit_kind in ("ma50", "hot30") and np.isfinite(m) and p < m:
                pend_sell.append(c)
            elif exit_kind == "hot30" and np.isfinite(m) and p >= m * 1.30:
                pend_sell.append(c)
        free = N - (len(hold) - len(pend_sell))
        if free > 0:
            cand = [c for c in np.where(sig[i])[0] if c not in hold]
            cand.sort(key=lambda c: -rank[i, c])
            pend_buy = [(c, float(support[i, c])) for c in cand[:free]]
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1], trades, days_held


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
    vol = pd.DataFrame(volumes(cols)).reindex(index=days, columns=cols)
    P, V = px.values, vol.values
    T, K = P.shape
    rets_v = px.pct_change(fill_method=None).values
    ma50, ma200 = px.rolling(50).mean().values, px.rolling(200).mean().values
    ma50_10 = px.rolling(50).mean().shift(10).values
    r12n = np.nan_to_num((px / px.shift(252) - 1).values, nan=-9)
    ye = px.groupby(days.year).last()
    ytdn = np.nan_to_num(P / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    hi52 = px.rolling(252, min_periods=200).max().values
    hi52f = np.nan_to_num(hi52, nan=np.inf)
    box_top = px.rolling(40).max().shift(1).values
    box_bot = px.rolling(40).min().shift(1).values
    v50 = vol.rolling(50, min_periods=40).mean().shift(1).values
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
    with np.errstate(invalid="ignore", divide="ignore"):
        brk = (mem & fin & (P > box_top) & (box_top / box_bot - 1 <= 0.25) & (V >= 2 * v50) & (P > ma200) & (ma50 > ma50_10))
    brk = np.nan_to_num(brk, nan=False).astype(bool)
    log(f"放量突破：{brk.sum()} 次（2000年起每年約 {brk[days >= pd.Timestamp('2000-01-01')].sum() / 26.75:.0f} 次）")
    sig = np.zeros(P.shape, dtype=bool)
    sup = np.full(P.shape, np.nan)
    for b, c in zip(*np.where(brk)):
        top, base = box_top[b, c], v50[b, c]
        for k in range(1, 11):
            i = b + k
            if i >= T or not np.isfinite(P[i, c]) or P[i, c] < top:
                break
            if k >= 3 and np.nanmean(V[b + 1:i + 1, c]) <= base and mem[i, c]:
                sig[i, c] = True
                sup[i, c] = top
                break
    log(f"縮量整固買入訊號：{sig.sum()} 個")
    # 板塊效應：同行業另外 ≥2 隻喺過去11日（包括今日）有放量突破
    smap = sectors()
    names = list(ETF)
    sec = np.array([names.index(smap[t]) if t in smap else -1 for t in cols])
    recent = pd.DataFrame(brk).rolling(11, min_periods=1).max().values.astype(bool)
    cnt = np.zeros((T, len(names)), dtype=int)
    for s_ in range(len(names)):
        cnt[:, s_] = recent[:, sec == s_].sum(axis=1)
    peers = np.zeros(P.shape, dtype=int)
    kn = sec >= 0
    peers[:, kn] = cnt[:, sec[kn]] - recent[:, kn].astype(int)
    sig_sec = sig & (peers >= 2)
    log(f"加板塊效應後：{sig_sec.sum()} 個；有行業分類 {(sec >= 0).mean() * 100:.0f}%")

    def trail20(i, c, p, pk):
        return p <= pk * 0.8

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b_ = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b_

    d_ok = mem & (r12n >= 1.0) & (P > ma50) & (P > ma200) & fin
    e_ok = mem & (ytdn >= 0.30) & (P >= hi52f * 0.7) & (P > ma200) & fin
    S, ST = {}, {}
    S["線E（對照）"] = pd.Series(sim_close(P, rets_v, cr, e_ok, ytdn, broken)[0], index=days)
    S["線D（對照）"] = pd.Series(sim_close(P, rets_v, cr, d_ok, r12n, trail20)[0], index=days)
    for nm, sg, ek in [("V1 突破整固・高位跌20%賣", sig, "trail20"), ("V2 突破整固・跌穿50日線賣", sig, "ma50"),
                       ("V3 突破整固・升到好熱（高過50日線30%）就賣", sig, "hot30"), ("V4 V1＋板塊效應", sig_sec, "trail20")]:
        r, tr, hd = sim_box(P, cr, sg, r12n, sup, ek, ma50)
        S[nm] = pd.Series(r, index=days)
        ST[nm] = (np.array(tr), np.array(hd))
        log(f"{nm} 完成")

    # 訊號之後表現（描述）
    def fwd(h):
        f = np.full(P.shape, np.nan)
        f[:-h - 1] = P[h + 1:] / P[1:-h] - 1          # 由下一日收市買，揸 h 日
        return f
    spy_v = spy.values
    L = ["=== 第三十八輪：放量突破箱體＋縮量整固＋板塊效應（規則預先登記，見research_v38.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻", "",
         f"放量突破 {brk.sum()} 次；縮量整固買入訊號 {sig.sum()} 個（每年約 {sig[days >= pd.Timestamp('2000-01-01')].sum() / 26.75:.0f} 個）；"
         f"加板塊效應 {sig_sec.sum()} 個", ""]
    for h in (20, 60):
        f = fwd(h)
        sf = np.full(T, np.nan)
        sf[:-h - 1] = spy_v[h + 1:] / spy_v[1:-h] - 1
        base = f[mem & fin & np.isfinite(f)]
        for lab, m in [("所有成分股", mem & fin), ("縮量整固訊號", sig), ("加板塊效應", sig_sec)]:
            x = f[m & np.isfinite(f)]
            ii = np.where(m & np.isfinite(f))[0]
            beat = (x > sf[ii]).mean() * 100 if len(x) else np.nan
            L.append(f"　之後 {h} 日｜{lab}：{len(x)} 次 ｜ 中位數 {np.median(x) * 100 if len(x) else np.nan:+.2f}% ｜ "
                     f"平均 {np.mean(x) * 100 if len(x) else np.nan:+.2f}% ｜ 贏SPY {beat:.0f}%")
        L.append(f"　（參考：所有成分股樣本 {len(base)}）")
    L.append("")
    L.append("=== 每單統計（1999-2026）===")
    for nm, (tr, hd) in ST.items():
        if len(tr) == 0:
            L.append(f"　{nm}：冇交易")
            continue
        w, l_ = tr[tr > 0], tr[tr <= 0]
        L.append(f"　{nm}：{len(tr)} 單（每年約 {len(tr) / 27:.0f} 單）｜ 勝率 {(tr > 0).mean() * 100:.0f}% ｜ 贏平均 {w.mean() * 100:+.1f}% ｜ "
                 f"輸平均 {l_.mean() * 100:+.1f}% ｜ 平均每單 {tr.mean() * 100:+.2f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}% ｜ 平均揸 {hd.mean():.1f} 日")
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
    pd.DataFrame({"report": L}).to_csv("v38_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
