"""
research_v45.py — 第四十五輪：換手率（成交量）有冇用？（Esther 2026-10-28：「我見好多時見頂或者見底準備反彈既換手率都比平常高」；
  要求兩部分都試：A 事件統計、B 放落線E試）

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
換手率代替品：Yahoo 冇以前嘅流通股數，所以用「相對成交量」RV = 今日成交量 ÷ 之前50個交易日平均成交量（唔包今日）。
  同一隻股流通股數短期變化好細，RV 升跌同換手率差唔多。成交量係0或者冇數據 = 當冇訊號。
股票池：當時真實標普500；已計股息；1999-2026（數據到 2026-09-30）。

A 部分：事件統計
  事件：RV ≥ 3（成交量係平時3倍或以上）；同一隻股之前20個交易日入面已經有過 RV ≥ 3 嘅日子就唔再計（避免同一段大成交重複計）。
  位置分類（用收市價）：
    高位 = 收市 ≥ 52週最高收市 × 0.95
    大跌後 = 收市 ≤ 52週最高收市 × 0.70
    其他 = 中間
  當日K線分類（用已調整 開高低收）：收市喺當日（最高−最低）範圍嘅上半 =「收得高」；下半 =「收得低」。
    高位 + 收得低 = 「放量滯漲」（可能見頂）；大跌後 + 收得高 = 「放量收返上去」（可能見底）。
  由事件「翌日收市」計（收市後先知），睇之後 1個月（21日）、3個月（63日）、6個月（126日）：
    中位數、平均、升嘅比例、贏同期SPY嘅比例。
  對照：同一個位置分類、但當日 RV < 1.5（成交正常）嘅全部股票日，同樣由翌日收市計。
  意思：如果「高位+放量滯漲」之後明顯差過「高位+成交正常」，就代表高換手對見頂有參考價值；見底同樣道理。

B 部分：放落線E（共同設定同第四十一輪一樣：收市決定→下一個交易日收市、0.1%、揸5隻、閒錢收^IRX）
  版本（只有呢4個，跑完唔加唔減唔改數字）：
    V1 高換手唔買：過去5個交易日（包今日）最高 RV ≥ 3 就唔買（其他同線E）
    V2 放量滯漲就賣：已揸嘅股，當日 高位 + RV ≥ 3 + 收得低 → 賣（加埋線E原本賣出條件，邊個先到用邊個）
    V3 = V1 + V2
    V4 長期高換手唔買：過去20日平均成交量 ≥ 過去250日平均成交量 × 2 就唔買（參考 Lee & Swaminathan 2000：升得勁又高換手嘅股，升勢較早完）
  V2、V3 賣出之後 20 個交易日內唔再買同一隻（避免賣完即刻買返）。
  判斷同第四十一輪一樣（同線E比；驗證期 2015-01 至 2026-09）：有改善、四條問題、跨年代合格、最差5年年化。
就算合格都唔會直接改模擬交易，要你同意先加新線。

輸出：v45_results.csv、v45_progress.txt
==== 結果（2026-10-29 第一次跑，記錄低，唔再改）====
A 部分（17,363 次事件）：高成交同成交正常之後差唔多。
- 高位 + 放量滯漲：6個月中位數 +5.4%（對照 +5.3%），升 65%（64%），贏SPY 50%（49%）→ 冇見頂訊號。
- 高位 + 收得高：6個月 +5.7%（對照 +5.3%）。其他位置：+5.8%（+5.5%）。
- 大跌後 + 放量收返上去：1個月 +2.3%（+2.1%）、3個月 +5.9%（+5.2%）、6個月 +7.0%（+10.1%）→ 短期差唔多，6個月反而差啲；
  2020-26 6個月 +5.7% vs 對照 +11.3%。冇見底反彈訊號。
B 部分（驗證期）：V1 +28.72% / -32.6%（同線E幾乎一樣，線E候選股只有 5% 有高換手）；V2 +22.67% / -32.7%；
  V3 +18.05% / -29.9%；V4 +25.58% / -33.3%；線E +28.74% / -32.6% → 全部冇改善。
- V2 每單平均 +9.0%（線E +15.3%），最好一單 +272%（線E +669%）→ 放量滯漲就賣，斬咗大贏家。
意思：單靠成交量（換手率）高唔高，分唔到之後升定跌；見頂見底嗰陣成交大係真，但平時成交大嘅日子更多。
"""
import numpy as np
import pandas as pd

import data_cache as DC
import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim
from research_v37 import ohlc

M.PROGRESS = "v45_progress.txt"
log = M.log
END = "2026-09-30"
COOL = 20
H = {"1個月": 21, "3個月": 63, "6個月": 126}


def volumes(tickers, start="1998-01-01", end="2026-10-01"):
    out = {}
    for t in tickers:
        try:
            h = DC.history(t, start, end)
        except Exception:
            continue
        if h is not None and len(h) and "Volume" in h:
            v = h["Volume"].astype(float)
            idx = pd.DatetimeIndex(v.index)
            v.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
            out[t] = v[~v.index.duplicated(keep="last")]
    log(f"成交量：{len(out)}/{len(tickers)} 隻")
    return out


def pct(x):
    return "—" if not np.isfinite(x) else f"{x * 100:+.1f}%"


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
    vol = pd.DataFrame(volumes(cols)).reindex(index=days, columns=cols)
    vol = vol.where(vol > 0)
    avg50 = vol.shift(1).rolling(50, min_periods=40).mean()
    RV = (vol / avg50).values
    RVf = np.nan_to_num(RV, nan=0.0)
    v20 = vol.rolling(20, min_periods=15).mean()
    v250 = vol.rolling(250, min_periods=200).mean()
    longhigh = np.nan_to_num((v20 / v250).values >= 2.0, nan=False)
    data = ohlc(cols)
    hi_ = pd.DataFrame({t: d["High"] for t, d in data.items()}).reindex(index=days, columns=cols).values
    lo_ = pd.DataFrame({t: d["Low"] for t, d in data.items()}).reindex(index=days, columns=cols).values
    cl_ = pd.DataFrame({t: d["Close"] for t, d in data.items()}).reindex(index=days, columns=cols).values
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (cl_ - lo_) / (hi_ - lo_)
    close_hi = np.nan_to_num(pos >= 0.5, nan=False)
    close_lo = np.nan_to_num(pos < 0.5, nan=False)
    log(f"開高低收：{len(data)}/{len(cols)} 隻")

    rets_v = px.pct_change(fill_method=None).values
    ma200 = px.rolling(200).mean().values
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
    with np.errstate(invalid="ignore"):
        near_hi = np.nan_to_num(P >= hi52 * 0.95, nan=False)
        deep = np.nan_to_num(P <= hi52 * 0.70, nan=False)
    e_ok = mem & (ytdn >= 0.30) & (P >= hi52f * 0.7) & np.nan_to_num(P > ma200, nan=False) & fin
    log("數據準備好")

    # ---------------- A 部分 ----------------
    T = len(days)
    pxf = px.ffill().values
    spyv = spy.values
    fwd, sfwd = {}, {}
    for nm, n in H.items():
        f = np.full(P.shape, np.nan)
        s_ = np.full(T, np.nan)
        # 由翌日收市（i+1）計到 i+1+n
        f[:T - 1 - n] = pxf[1 + n:] / pxf[1:T - n] - 1
        s_[:T - 1 - n] = spyv[1 + n:] / spyv[1:T - n] - 1
        fwd[nm], sfwd[nm] = f, s_
    start_i = int(np.searchsorted(days, pd.Timestamp("1999-06-01")))
    ev = np.zeros(P.shape, dtype=bool)
    raw = (RVf >= 3) & mem & fin
    raw[:start_i] = False
    last = np.full(P.shape[1], -10 ** 9)
    for i in range(T):
        hit = np.flatnonzero(raw[i])
        if len(hit):
            ok_ = hit[i - last[hit] > 20]
            ev[i, ok_] = True
            last[hit] = i
    normal = mem & fin & np.nan_to_num((RV < 1.5), nan=False)
    normal[:start_i] = False
    mid = ~near_hi & ~deep

    def stats(mask, nm):
        f = fwd[nm][mask]
        s_ = np.broadcast_to(sfwd[nm][:, None], P.shape)[mask]
        k = np.isfinite(f) & np.isfinite(s_)
        f, s_ = f[k], s_[k]
        if len(f) == 0:
            return "冇數據"
        return (f"{len(f):,} 次 ｜ 中位數 {pct(np.median(f))} ｜ 平均 {pct(f.mean())} ｜ 升 {(f > 0).mean() * 100:.0f}% ｜ "
                f"贏SPY {(f > s_).mean() * 100:.0f}%")

    L = ["=== 第四十五輪：換手率（相對成交量）有冇用？（規則預先登記，見research_v45.py開頭）===",
         f"股票池：當時真實標普500（{source}）；成交量有數據 {int(vol.notna().any().sum())}/{len(cols)} 隻；開高低收 {len(data)}/{len(cols)} 隻", "",
         "=== A 部分：成交量 ≥ 平時3倍 之後點？（由翌日收市計）===", f"事件總數：{int(ev.sum()):,}", ""]
    groups = [("高位 + 收得低（放量滯漲，可能見頂）", near_hi & close_lo), ("高位 + 收得高", near_hi & close_hi),
              ("大跌後 + 收得高（放量收返上去，可能見底）", deep & close_hi), ("大跌後 + 收得低", deep & close_lo),
              ("其他位置", mid)]
    for gname, gm in groups:
        L.append(f"【{gname}】")
        base_m = normal & (near_hi if "高位" in gname else deep if "大跌後" in gname else mid)
        for nm in H:
            L.append(f"　{nm}：高成交 {stats(ev & gm, nm)}")
            L.append(f"　　　　對照（同位置、成交正常）{stats(base_m, nm)}")
        L.append("")
    L.append("年代分開（6個月中位數；高成交 vs 對照）：")
    for a_, b_, lab in [("1999-06-01", "2009-12-31", "1999-2009"), ("2010-01-01", "2019-12-31", "2010-2019"), ("2020-01-01", END, "2020-2026")]:
        tm = np.zeros(T, dtype=bool)
        tm[(days >= pd.Timestamp(a_)) & (days <= pd.Timestamp(b_))] = True
        tm2 = np.broadcast_to(tm[:, None], P.shape)
        parts = []
        for gname, gm in groups[:4]:
            pos_m = near_hi if "高位" in gname else deep
            a = fwd["6個月"][ev & gm & tm2]
            b = fwd["6個月"][normal & pos_m & tm2]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            parts.append(f"{gname[:9]} {pct(np.median(a)) if len(a) else '—'}（{len(a)}次）vs {pct(np.median(b)) if len(b) else '—'}")
        L.append(f"　{lab}：" + " ｜ ".join(parts))
    L.append("")
    log("A 部分完成")

    # ---------------- B 部分 ----------------
    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    rv5 = pd.DataFrame(RVf).rolling(5, min_periods=1).max().values
    climax = near_hi & (RVf >= 3) & close_lo

    def make_v2(ok):
        def fn(i, c, p, pk):
            s = broken(i, c, p, pk) or bool(climax[i, c])
            if s and climax[i, c]:
                ok[i + 1:i + 1 + COOL, c] = False
            return s
        return fn

    V = {"線E（對照）": sim(P, rets_v, cr, e_ok, ytdn, broken)}
    V["V1 高換手唔買"] = sim(P, rets_v, cr, e_ok & (rv5 < 3), ytdn, broken)
    ok2 = e_ok.copy()
    V["V2 放量滯漲就賣"] = sim(P, rets_v, cr, ok2, ytdn, make_v2(ok2))
    ok3 = e_ok & (rv5 < 3)
    V["V3 V1+V2"] = sim(P, rets_v, cr, ok3, ytdn, make_v2(ok3))
    V["V4 長期高換手唔買"] = sim(P, rets_v, cr, e_ok & ~longhigh, ytdn, broken)
    log("B 部分完成")
    sel = e_ok[days >= pd.Timestamp("2000-01-01")]
    L += ["=== B 部分：放落線E ===",
          f"線E候選股（2000年起，股票×日）：過去5日有 RV≥3 佔 {(rv5[days >= pd.Timestamp('2000-01-01')][sel] >= 3).mean() * 100:.0f}%；"
          f"長期高換手 佔 {longhigh[days >= pd.Timestamp('2000-01-01')][sel].mean() * 100:.0f}%", ""]

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
    L.append("=== 每次買賣（1999-2026）===")
    for nm, v in V.items():
        tr = np.array(v[1])
        L.append(f"　{nm}：{len(tr)} 次 ｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
                 f"蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}%")
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
    pd.DataFrame({"report": L}).to_csv("v45_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
