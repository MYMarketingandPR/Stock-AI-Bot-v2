"""
research_v36.py — 第三十六輪：動量、均值回歸、兩樣各一半（Esther 2026-10-15：「我想試吓呢三個方法」）

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同：當時真實標普500（fja05680）；已計股息；收市決定 → 下一個交易日收市買賣；每次買賣0.1%；
  冇投資嘅錢收短期國債息（^IRX）。

1) 動量（經典 1993 年做法）
  M1 每月尾揀過去12個月（扣最近1個月，即 252 至 21 個交易日前）升最多嘅10隻，平均分，下個月尾再揀（跌出前10就賣）。
  另外列我哋已有嘅線E（今年升≥30%…，第二十六輪）做對照。

2) 均值回歸（只有呢兩個版本）
  R1 「超賣反彈」（Connors RSI(2) 做法）：收市 > 200日平均 而且 RSI(2) < 10 → 買（揀 RSI(2) 最低嘅，最多5隻，每隻 = 總資產÷5）；
     收市 > 5日平均 就賣，或者揸滿 10 個交易日都賣。
  R2 「一週大跌反彈」：收市 > 200日平均 而且 5日跌 ≥ 8% → 買（揀5日跌最多嘅，最多5隻）；揸 5 個交易日就賣。

3) 兩樣各一半（每日固定 50% / 50%，即係每日都調返一半一半）
  C1 = 50% M1 + 50% R1
  C2 = 50% 線E + 50% R1
  另外列 M1 同 R1 每日回報嘅相關系數（越低 = 越互補）。

判斷（驗證期 2015-01 至 2026-09）：
  每個版本：四條問題（同QQQ/SPY比）；跨年代合格（2000-09、2010-14、2015-19、2020-26 每段年化 ≥ SPY 而且最大跌幅好過 -35%）；
  最差5年年化；均值回歸另外列勝率、平均每次賺幾多、平均揸幾日。
  「一半一半」有冇改善：C1 同 M1 比、C2 同線E比，用同之前一樣嘅「有改善」標準
  （最大跌幅好過 ≥3點而年化唔低過超過1點；或者年化高過 ≥2點而最大跌幅唔差過超過2點）。
  就算合格都唔會直接改模擬交易，要你同意先加新線。

已知限制：Yahoo 冇數據嘅已除牌股唔會入選（對均值回歸可能有利：真正爆煲嘅超賣股唔喺數據入面）；
  均值回歸買賣好密，0.1% 成本假設已經偏樂觀（真實滑價可能更高）。

輸出：v36_results.csv、v36_progress.txt
==== 結果（2026-10-15 第一次跑，run 38767164422，記錄低，唔再改）====
- 相關系數：M1 同 R1 = 0.50；線E 同 R1 = 0.44（一齊升一齊跌嘅程度唔低，唔算好互補）。
- R1 超賣反彈：每年約 292 次，勝率 60%，平均每次 +0.30%（未扣成本），贏平均 +2.3%、輸平均 -2.7%，最差一次 -52%，平均揸 3.9 日。
  R2 一週大跌反彈：每年約 125 次，勝率 54%，平均每次 +0.60%，最差 -50%。
- 驗證期（年化 / 最大跌幅）：M1 經典動量 +16.33% / -38.6%；R1 -1.29% / -43.2%；R2 +6.54% / -44.2%；
  線E +28.74% / -32.6%；C1 一半M1＋一半R1 +8.20% / -37.4%（同M1比冇改善）；C2 一半線E＋一半R1 +13.93% / -31.5%（同線E比冇改善）。
- 全部唔合格（四條問題全部淘汰，除咗線E；跨年代冇一個合格）。
- 金融海嘯：R1 -46%、R2 -48%，冇幫到手；2022熊市：R1 只跌 12.5%（好過其他），但其他年代賺唔到錢。
- 意思：均值回歸勝率的確有六成，但每次賺得少、扣成本後接近零，偶然一次大蝕（-50%）；同動量一半一半只係拉低回報。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim as sim_trend

M.PROGRESS = "v36_progress.txt"
log = M.log
END = "2026-09-30"
COST = 0.001


def sim_mr(P, rets_v, cr, ok, rank, exit_fn, n=5, max_hold=10):
    """均值回歸：有空位就買（ok & 排名），exit_fn(i, c) 為真或者揸滿 max_hold 日就賣。"""
    T = P.shape[0]
    hold, info, cash = {}, {}, 1.0
    ps, pb, navs, trades, held_days = [], [], [], [], []
    for i in range(T):
        for c in list(hold):
            r = rets_v[i, c]
            if np.isfinite(r):
                hold[c] *= 1 + r
        cash *= 1 + cr[i]
        for c in ps:
            if c in hold:
                v = hold.pop(c)
                cash += v * (1 - COST)
                b = info.pop(c)
                sp = P[i, c] if np.isfinite(P[i, c]) else b["last"]
                trades.append(sp / b["px"] - 1)
                held_days.append(i - b["i0"])
        total = cash + sum(hold.values())
        for c in pb:
            if c in hold or len(hold) >= n or not np.isfinite(P[i, c]):
                continue
            amt = min(total / n, cash)
            if amt <= total * 0.01:
                break
            cash -= amt
            hold[c] = amt * (1 - COST)
            info[c] = {"px": float(P[i, c]), "last": float(P[i, c]), "i0": i}
        ps, pb = [], []
        navs.append(cash + sum(hold.values()))
        for c in hold:
            p = P[i, c]
            if np.isfinite(p):
                info[c]["last"] = float(p)
            if (not np.isfinite(p) and i + 1 < T and not np.isfinite(P[i + 1, c])) or exit_fn(i, c) or i - info[c]["i0"] >= max_hold:
                ps.append(c)
        free = n - (len(hold) - len(ps))
        if free > 0:
            cand = [c for c in np.where(ok[i])[0] if c not in hold and c not in ps]
            cand.sort(key=lambda c: -rank[i, c])
            pb = cand[:free]
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1], trades, held_days


def sim_monthly(P, rets_v, cr, score, mem, dec_idx, n=10):
    """經典動量：每個決定日揀分數最高 n 隻（要係成分股、有價），下一個交易日收市換倉，平均分。"""
    T = P.shape[0]
    hold, cash = {}, 1.0
    navs = []
    target = None
    dec = set(dec_idx)
    for i in range(T):
        for c in list(hold):
            r = rets_v[i, c]
            if np.isfinite(r):
                hold[c] *= 1 + r
            elif i + 1 < T and not np.isfinite(P[i, c]) and not np.isfinite(P[i + 1, c]):
                cash += hold.pop(c) * (1 - COST)          # 除牌：用最後價賣
        cash *= 1 + cr[i]
        if target is not None:
            total = cash + sum(hold.values())
            tgt = [c for c in target if np.isfinite(P[i, c])]
            for c in list(hold):
                if c not in tgt:
                    cash += hold.pop(c) * (1 - COST)
            total = cash + sum(hold.values())
            each = total / max(len(tgt), 1) if tgt else 0
            for c in tgt:
                cur = hold.get(c, 0.0)
                diff = each - cur
                if diff > 0:
                    amt = min(diff, cash)
                    cash -= amt
                    hold[c] = cur + amt * (1 - COST)
                elif diff < 0:
                    hold[c] = cur + diff
                    cash += -diff * (1 - COST)
            target = None
        navs.append(cash + sum(hold.values()))
        if i in dec:
            s = np.where(mem[i] & np.isfinite(score[i]) & np.isfinite(P[i]), score[i], -np.inf)
            idx = np.argsort(-s)[:n]
            target = [c for c in idx if np.isfinite(s[c])]
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1]


def rsi(px, n=2):
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn > 0, 100.0)


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
    ma5, ma200 = px.rolling(5).mean().values, px.rolling(200).mean().values
    ye = px.groupby(days.year).last()
    ytdn = np.nan_to_num(P / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    hi52 = px.rolling(252, min_periods=200).max().values
    hi52f = np.nan_to_num(hi52, nan=np.inf)
    mom = (px.shift(21) / px.shift(252) - 1).values
    r5 = (px / px.shift(5) - 1).values
    rsi2 = rsi(px).values
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
    log("數據準備好")

    up200 = np.nan_to_num(P > ma200, nan=False)
    r1_ok = mem & fin & up200 & np.nan_to_num(rsi2 < 10, nan=False)
    r2_ok = mem & fin & up200 & np.nan_to_num(r5 <= -0.08, nan=False)
    e_ok = mem & (ytdn >= 0.30) & (P >= hi52f * 0.7) & up200 & fin

    def above5(i, c):
        return np.isfinite(P[i, c]) and np.isfinite(ma5[i, c]) and P[i, c] > ma5[i, c]

    def never(i, c):
        return False

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    dec_idx = [days.get_loc(d) for d in decision_days(days, "每月")]
    m1 = sim_monthly(P, rets_v, cr, mom, mem, dec_idx)
    log("M1 完成")
    r1, r1_tr, r1_hd = sim_mr(P, rets_v, cr, r1_ok, -np.nan_to_num(rsi2, nan=100), above5, max_hold=10)
    log("R1 完成")
    r2, r2_tr, r2_hd = sim_mr(P, rets_v, cr, r2_ok, -np.nan_to_num(r5, nan=0), never, max_hold=5)
    log("R2 完成")
    e, e_tr = sim_trend(P, rets_v, cr, e_ok, ytdn, broken)
    log("線E 完成")
    S = {"M1 經典動量（前10隻每月換）": pd.Series(m1, index=days),
         "R1 超賣反彈（RSI2）": pd.Series(r1, index=days),
         "R2 一週大跌反彈": pd.Series(r2, index=days),
         "線E（對照）": pd.Series(e, index=days)}
    S["C1 一半M1＋一半R1"] = 0.5 * S["M1 經典動量（前10隻每月換）"] + 0.5 * S["R1 超賣反彈（RSI2）"]
    S["C2 一半線E＋一半R1"] = 0.5 * S["線E（對照）"] + 0.5 * S["R1 超賣反彈（RSI2）"]

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r, **S}
    tab = {}
    L = ["=== 第三十六輪：動量、均值回歸、兩樣各一半（規則預先登記，見research_v36.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%", ""]
    a, b = S["M1 經典動量（前10隻每月換）"].loc["2000-01-03":END], S["R1 超賣反彈（RSI2）"].loc["2000-01-03":END]
    c_, d_ = S["線E（對照）"].loc["2000-01-03":END], b
    L.append(f"每日回報相關系數：M1 同 R1 = {a.corr(b):.2f} ｜ 線E 同 R1 = {c_.corr(d_):.2f} ｜ M1 同 SPY = {a.corr(spy_r.loc['2000-01-03':END]):.2f} ｜ R1 同 SPY = {b.corr(spy_r.loc['2000-01-03':END]):.2f}")
    for nm, tr, hd in [("R1 超賣反彈", r1_tr, r1_hd), ("R2 一週大跌反彈", r2_tr, r2_hd)]:
        tr = np.array(tr)
        L.append(f"{nm}：{len(tr)} 次（每年約 {len(tr) / 27:.0f} 次）｜ 勝率 {(tr > 0).mean() * 100:.0f}% ｜ 平均每次 {tr.mean() * 100:+.2f}% ｜ "
                 f"贏嘅平均 {tr[tr > 0].mean() * 100:+.1f}% ｜ 輸嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 平均揸 {np.mean(hd):.1f} 日")
    L.append("")
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
    base_of = {"C1 一半M1＋一半R1": "M1 經典動量（前10隻每月換）", "C2 一半線E＋一半R1": "線E（對照）"}
    L.append(f"=== 預先登記判斷 ｜ QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% / Sharpe {q['sharpe']:.2f} ｜ SPY {s['cagr']:+.2f}% / {s['mdd']:.1f}% ===")
    for nm in S:
        x, ee = tab[("驗證期", nm)], tab[("2000-2009", nm)]
        marks = "".join("✔" if (tab[(er, nm)]["cagr"] >= tab[(er, "SPY長揸")]["cagr"] and tab[(er, nm)]["mdd"] > -35) else "✘" for er in eras)
        imp = ""
        if nm in base_of:
            b0 = tab[("驗證期", base_of[nm])]
            better = (x["mdd"] >= b0["mdd"] + 3 and x["cagr"] >= b0["cagr"] - 1) or (x["cagr"] >= b0["cagr"] + 2 and x["mdd"] >= b0["mdd"] - 2)
            imp = f" ｜ 同{base_of[nm].split('（')[0]}比 {'有改善' if better else '冇改善'}"
        q1 = x["cagr"] > q["cagr"]
        q2 = q1 and x["mdd"] > -35 and x["sharpe"] >= q["sharpe"]
        q3 = (not q1) and x["mdd"] >= q["mdd"] + 5 and x["cagr"] >= s["cagr"]
        q4 = ee["mdd"] > -35
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f}{imp} ｜ "
                 f"四條問題 ①{'✔' if q1 else '✘'} ②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'} ⇒ "
                 f"{'保留' if (q2 or q3) else '淘汰'} ｜ 跨年代 {marks} {'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 "
                 + " / ".join(f"{tab[(er, nm)]['cagr']:+.1f}%" for er in eras) + " ｜ 各段最大跌幅 "
                 + " / ".join(f"{tab[(er, nm)]['mdd']:.0f}%" for er in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
    pd.DataFrame({"report": L}).to_csv("v36_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
