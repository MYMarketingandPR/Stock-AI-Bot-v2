"""
research_v37.py — 第三十七輪：「窄止蝕 + 回調買入」（參考 Martin Luk 做法；Esther 2026-10-22：「好，試下做第 37 輪測試」）

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
數據：每日 開市／最高／最低／收市（Yahoo 已計股息調整）。當時真實標普500（fja05680）。每次買賣0.1%；閒錢收^IRX；
  揸最多5隻，每隻 = 買入嗰日總資產 ÷ 5（唔借錢）。
股票池：線E條件（今年 ≥ +30%、收市 ≥ 52週最高收市 × 0.7、收市 > 200日平均）。
回調訊號（第 i 日收市後判斷）：股票池入面 + 20日平均向上（今日 > 5日前）+ 收市 ≥ 20日平均 + 當日最低價 ≤ 10日平均 × 1.01
  （即係回落掂到10日線附近，但收返上20日線上面）。多過空位就揀今年升最多嘅。
入場：第 i+1 日收市買入；如果第 i+1 日收市已經低過止蝕位就唔買。
止蝕（買入之後第二日開始，即市執行）：開市已經低過止蝕位 → 用開市價賣（裂口）；否則當日最低價掂到止蝕位 → 用止蝕價賣。
版本（只有呢三個，跑完唔加唔減唔改數字）：
  T1 止蝕 = 回調日最低價；如果回調日收市距離佢超過 8% 就唔做（風險太大）。贏嘅：收市跌穿10日平均 → 下一日收市賣。
  T2 止蝕 = 買入價 × 0.97（固定 3%）。贏嘅：收市跌穿10日平均 → 下一日收市賣。
  T3 同 T1，但贏嘅：收市跌穿20日平均先賣（俾贏嘅單跑耐啲）。
對照：線E（第二十六輪，收市價版本）、QQQ、SPY。

判斷（驗證期 2015-01 至 2026-09）：同第三十三輪一樣：同線E比「有改善」、四條問題、跨年代合格（2000-09、2010-14、2015-19、
  2020-26 每段年化 ≥ SPY 而且最大跌幅好過 -35%）、最差5年年化。
另外列：每年幾多單、勝率、贏嘅平均、輸嘅平均、平均揸幾日、平均每單（同影片 22% 勝率／贏 +14%／輸 -2% 比較）。
就算合格都唔會直接改模擬交易，要你同意先加新線。

已知限制：日線模擬唔到即市入場位（佢用即市圖揀入場）；同一日先掂止蝕定先升，日線分唔到（一律當先掂止蝕，偏保守）；
  已除牌股冇數據唔入選；滑價可能高過 0.1%。

輸出：v37_results.csv、v37_progress.txt
==== 結果（2026-10-22 第一次跑，run 39697314943，記錄低，唔再改）====
- 有OHLC數據 716 隻；回調訊號平均每日 13.6 個。
- 每單：T1 每年約189單，勝率34%，贏 +4.4%／輸 -2.2%，平均每單 +0.09%，平均揸4.5日；
  T2 每年約243單，勝率40%，贏 +3.6%／輸 -2.4%，平均 +0.04%，最差 -41%（裂口跌穿止蝕）；
  T3 每年約112單，勝率27%，贏 +8.0%／輸 -2.5%，平均 +0.32%，平均揸8.2日（贏19.9日）。
- 驗證期：T1 -2.11% / -42.2%；T2 -3.79% / -65.3%；T3 +3.83% / -29.7%；線E +28.74% / -32.6% → 全部冇改善、全部淘汰、跨年代全部唔合格。
- 2022熊市：T1 -16.5% / -17%；T3 -3.5% / -14%（跌得比線E少）；2025年：T3 +25.2% / -13.1%（好過線E +20.0%）。
- 意思：輸嘅單的確控制到大約 -2%，但贏嘅單只有 +4% 至 +8%（影片係 +14%），扣成本後每單差唔多冇得賺；
  日線收市先賣，贏嘅單跑唔夠遠；窄止蝕要配合即市揀入場位先可能有用，日線規則複製唔到。
"""
import time

import numpy as np
import pandas as pd

import data_cache as DC
import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim as sim_close

M.PROGRESS = "v37_progress.txt"
log = M.log
END = "2026-09-30"
COST = 0.001
N = 5


def ohlc(tickers, start="1998-01-01", end="2026-10-01"):
    """逐隻下載已調整嘅 開/高/低/收（有快取就用快取）。"""
    import yfinance as yf
    out, fails = {}, []
    for n, t in enumerate(tickers):
        c = DC._load("ohlc", t) if DC.DIR else None
        if c is not None:
            if c["data"] is not None:
                out[t] = c["data"]
            continue
        h = None
        for attempt in range(3):
            try:
                h = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
                break
            except Exception as e:
                if not any(w in str(e).lower() for w in DC.TRANSIENT):
                    break
                time.sleep(5 * (attempt + 1))
        if h is not None and len(h) > 20:
            h = h[["Open", "High", "Low", "Close"]].astype(float)
            h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
            h = h[~h.index.duplicated(keep="last")]
            out[t] = h
            if DC.DIR:
                DC._save("ohlc", t, {"data": h})
        else:
            fails.append(t)
            if DC.DIR and h is not None:
                DC._save("ohlc", t, {"data": None})
        time.sleep(0.3)
        if (n + 1) % 100 == 0:
            log(f"OHLC 下載進度 {n + 1}/{len(tickers)}，有數據 {len(out)}")
    log(f"OHLC 完成：{len(out)}/{len(tickers)}")
    return out


def sim_tight(O, H, Lo, C, cr, setup, rank, setup_low, kind, exit_ma):
    """kind: 'low' = 止蝕喺回調日最低價（風險 > 8% 唔做）；'pct3' = 買入價 × 0.97。exit_ma: 收市低過呢條線就下一日收市賣。"""
    T, K = C.shape
    cash = 1.0
    hold = {}            # c -> dict(sh, px, stop, i0, last)
    pend_buy, pend_sell = [], []
    navs, trades, days_held = [], [], []

    def close_pos(c, px):
        nonlocal cash
        h = hold.pop(c)
        cash += h["sh"] * px * (1 - COST)
        trades.append(px / h["px"] - 1)
        days_held.append(i - h["i0"])

    for i in range(T):
        cash *= 1 + cr[i]
        # 1) 即市止蝕（買入之後第二日開始）
        for c in list(hold):
            h = hold[c]
            if i <= h["i0"]:
                continue
            o, lo = O[i, c], Lo[i, c]
            if np.isfinite(o) and o <= h["stop"]:
                close_pos(c, o)
            elif np.isfinite(lo) and lo <= h["stop"]:
                close_pos(c, h["stop"])
        # 2) 收市賣（昨晚決定）／除牌
        for c in pend_sell:
            if c in hold:
                p = C[i, c] if np.isfinite(C[i, c]) else hold[c]["last"]
                close_pos(c, p)
        for c in list(hold):
            if not np.isfinite(C[i, c]) and (i + 1 >= T or not np.isfinite(C[i + 1, c])):
                close_pos(c, hold[c]["last"])
        # 3) 收市買（昨晚決定）
        equity = cash + sum(h["sh"] * (C[i, c] if np.isfinite(C[i, c]) else h["last"]) for c, h in hold.items())
        for c, stop0 in pend_buy:
            p = C[i, c]
            if c in hold or len(hold) >= N or not np.isfinite(p):
                continue
            stop = p * 0.97 if kind == "pct3" else stop0
            if p <= stop:
                continue
            amt = min(equity / N, cash)
            if amt <= equity * 0.01:
                break
            cash -= amt
            hold[c] = {"sh": amt * (1 - COST) / p, "px": float(p), "stop": float(stop), "i0": i, "last": float(p)}
        pend_buy, pend_sell = [], []
        for c, h in hold.items():
            if np.isfinite(C[i, c]):
                h["last"] = float(C[i, c])
        navs.append(cash + sum(h["sh"] * h["last"] for h in hold.values()))
        # 4) 今晚決定
        for c, h in hold.items():
            m = exit_ma[i, c]
            if np.isfinite(C[i, c]) and np.isfinite(m) and C[i, c] < m:
                pend_sell.append(c)
        free = N - (len(hold) - len(pend_sell))
        if free > 0:
            cand = [c for c in np.where(setup[i])[0] if c not in hold]
            if kind == "low":
                cand = [c for c in cand if np.isfinite(setup_low[i, c]) and C[i, c] > 0 and 1 - setup_low[i, c] / C[i, c] <= 0.08]
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
    ma10, ma20, ma200 = px.rolling(10).mean().values, px.rolling(20).mean().values, px.rolling(200).mean().values
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
    for nm, kind, ex in [("T1 止蝕喺回調日最低・跌穿10日線賣", "low", ma10), ("T2 固定3%止蝕・跌穿10日線賣", "pct3", ma10),
                         ("T3 止蝕喺回調日最低・跌穿20日線賣", "low", ma20)]:
        r, tr, hd = sim_tight(Ov, Hv, Lv, P, cr, setup, ytdn, Lv, kind, ex)
        V[nm] = pd.Series(r, index=days)
        ST[nm] = (np.array(tr), np.array(hd))
        log(f"{nm} 完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31"),
               ("2025年", "2025-01-01", "2025-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r, **V}
    tab = {}
    L = ["=== 第三十七輪：窄止蝕＋回調買入（規則預先登記，見research_v37.py開頭）===",
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
    pd.DataFrame({"report": L}).to_csv("v37_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
