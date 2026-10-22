"""
research_v37b.py — 第三十七B輪：窄止蝕＋回調買入，贏嘅單用唔同方法賣（Esther 2026-10-22：
  「試下贏嘅單要跌穿 10, 20, 50日線先賣，等佢跑耐啲。試埋高位跌20%，30%就賣」）

純研究，唔改任何模擬交易，唔改之前任何結果（包括第三十七輪）。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同設定、股票池、回調訊號、入場、止蝕全部同第三十七輪 T1 一樣：
  止蝕 = 回調日最低價（回調日收市距離佢超過 8% 就唔做），買入之後第二日開始即市執行（裂口就用開市價）；
  揸最多5隻、每隻 = 總資產 ÷ 5；每次買賣0.1%；閒錢收^IRX。止蝕價一直唔郁。
只改「贏嘅單點賣」（收市決定，下一個交易日收市賣；同止蝕一齊用，邊個先到用邊個）：
  X10 收市跌穿10日平均（＝第三十七輪T1，對照）
  X20 收市跌穿20日平均（＝第三十七輪T3，對照）
  X50 收市跌穿50日平均
  R20 收市 ≤ 買入後最高收市 × 0.8
  R30 收市 ≤ 買入後最高收市 × 0.7
只有呢5個版本，跑完唔加唔減唔改數字。
判斷同第三十七輪一樣（同線E比有冇改善、四條問題、跨年代合格、最差5年；另外列每單統計）。
就算合格都唔會直接改模擬交易，要你同意先加新線。
已知限制同第三十七輪一樣（日線模擬唔到即市入場；同一日先掂止蝕定先升分唔到，一律當先掂止蝕）。

輸出：v37b_results.csv、v37b_progress.txt
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim as sim_close
from research_v37 import ohlc

M.PROGRESS = "v37b_progress.txt"
log = M.log
END = "2026-09-30"
COST = 0.001
N = 5


def sim_exit(O, Lo, C, cr, setup, rank, setup_low, exit_ma=None, trail=None):
    """止蝕 = 回調日最低價（風險 > 8% 唔做）；贏嘅單：exit_ma（收市低過就賣）或者 trail（收市 ≤ 買入後最高收市 × (1-trail)）。"""
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
        for c in list(hold):
            h = hold[c]
            if i <= h["i0"]:
                continue
            o, lo = O[i, c], Lo[i, c]
            if np.isfinite(o) and o <= h["stop"]:
                close_pos(c, o)
            elif np.isfinite(lo) and lo <= h["stop"]:
                close_pos(c, h["stop"])
        for c in pend_sell:
            if c in hold:
                p = C[i, c] if np.isfinite(C[i, c]) else hold[c]["last"]
                close_pos(c, p)
        for c in list(hold):
            if not np.isfinite(C[i, c]) and (i + 1 >= T or not np.isfinite(C[i + 1, c])):
                close_pos(c, hold[c]["last"])
        equity = cash + sum(h["sh"] * (C[i, c] if np.isfinite(C[i, c]) else h["last"]) for c, h in hold.items())
        for c, stop in pend_buy:
            p = C[i, c]
            if c in hold or len(hold) >= N or not np.isfinite(p) or p <= stop:
                continue
            amt = min(equity / N, cash)
            if amt <= equity * 0.01:
                break
            cash -= amt
            hold[c] = {"sh": amt * (1 - COST) / p, "px": float(p), "stop": float(stop), "i0": i, "last": float(p), "peak": float(p)}
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
            if exit_ma is not None and np.isfinite(exit_ma[i, c]) and p < exit_ma[i, c]:
                pend_sell.append(c)
            elif trail is not None and p <= h["peak"] * (1 - trail):
                pend_sell.append(c)
        free = N - (len(hold) - len(pend_sell))
        if free > 0:
            cand = [c for c in np.where(setup[i])[0] if c not in hold
                    and np.isfinite(setup_low[i, c]) and C[i, c] > 0 and 1 - setup_low[i, c] / C[i, c] <= 0.08]
            cand.sort(key=lambda c: -rank[i, c])
            pend_buy = [(c, float(setup_low[i, c])) for c in cand[:free]]
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1], trades, days_held


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
    O = pd.DataFrame({t: data[t]["Open"] for t in cols}).reindex(days)
    H = pd.DataFrame({t: data[t]["High"] for t in cols}).reindex(days)
    Lo = pd.DataFrame({t: data[t]["Low"] for t in cols}).reindex(days)
    px = pd.DataFrame({t: data[t]["Close"] for t in cols}).reindex(days)
    P = px.values
    rets_v = px.pct_change(fill_method=None).values
    ma10, ma20, ma50, ma200 = (px.rolling(10).mean().values, px.rolling(20).mean().values,
                               px.rolling(50).mean().values, px.rolling(200).mean().values)
    ma20_5 = px.rolling(20).mean().shift(5).values
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
    Lv = Lo.values
    setup = (e_ok & np.nan_to_num(ma20 > ma20_5, nan=False) & np.nan_to_num(P >= ma20, nan=False)
             & np.nan_to_num(Lv <= ma10 * 1.01, nan=False))
    log(f"數據準備好：{len(cols)} 隻；回調訊號平均每日 {setup.sum(axis=1)[days >= pd.Timestamp('2000-01-01')].mean():.1f} 個")

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    Ov, Hv = O.values, H.values
    V, ST = {}, {}
    e, e_tr = sim_close(P, rets_v, cr, e_ok, ytdn, broken)
    V["線E（對照）"] = pd.Series(e, index=days)
    for nm, ex, tr_ in [("X10 跌穿10日線賣（對照）", ma10, None), ("X20 跌穿20日線賣（對照）", ma20, None),
                        ("X50 跌穿50日線賣", ma50, None), ("R20 由高位跌20%賣", None, 0.20), ("R30 由高位跌30%賣", None, 0.30)]:
        r, tr, hd = sim_exit(Ov, Lv, P, cr, setup, ytdn, Lv, exit_ma=ex, trail=tr_)
        V[nm] = pd.Series(r, index=days)
        ST[nm] = (np.array(tr), np.array(hd))
        log(f"{nm} 完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31"),
               ("2025年", "2025-01-01", "2025-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r, **V}
    tab = {}
    L = ["=== 第三十七B輪：窄止蝕＋回調買入・唔同賣法（規則預先登記，見research_v37b.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻；有OHLC數據 {len(cols)} 隻", "",
         "=== 每單統計（1999-2026；影片：勝率約22%、贏平均約+14%、輸平均約-2%）==="]
    yrs = 27.0
    for nm, (tr, hd) in ST.items():
        w, l_ = tr[tr > 0], tr[tr <= 0]
        L.append(f"　{nm}：{len(tr)} 單（每年約 {len(tr) / yrs:.0f} 單）｜ 勝率 {(tr > 0).mean() * 100:.0f}% ｜ 贏平均 {w.mean() * 100:+.1f}% ｜ "
                 f"輸平均 {l_.mean() * 100:+.1f}% ｜ 平均每單 {tr.mean() * 100:+.2f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}% ｜ "
                 f"平均揸 {hd.mean():.1f} 日（贏 {hd[tr > 0].mean():.1f} 日／輸 {hd[tr <= 0].mean():.1f} 日）")
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
    b0 = tab[("驗證期", "線E（對照）")]
    L.append(f"=== 預先登記判斷 ｜ 線E {b0['cagr']:+.2f}% / {b0['mdd']:.1f}% ｜ QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% / Sharpe {q['sharpe']:.2f} ｜ SPY {s['cagr']:+.2f}% ===")
    for nm in V:
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
    pd.DataFrame({"report": L}).to_csv("v37b_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
