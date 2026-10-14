"""
research_v30.py — 第三十輪：業績跳空（Esther 2026-10-14：「業績跳空…我地都可以試」）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同：當時真實標普500；已計股息；收市決定 → 下一個交易日收市買賣；每次買賣0.1%；揸最多5隻、每隻 = 買入日總資產 ÷ 5；
  冇投資嘅錢揸現金（^IRX）。有空位揀「跳空嗰日升幅」最大嘅；只可以喺跳空之後 5 個交易日內買入（過咗就唔追）。
跳空日（快取冇開市價，用收市價代替）：
  價量跳空 = 一日收市升 ≥ +7% 而且 成交量 ≥ 之前50日平均成交量 × 3，而且收市 > MA200。
  業績跳空 = 價量跳空 + 呢日係公布業績之後第一或第二個交易日（Yahoo 業績日期；公布喺收市後就計下一日）。
  ⚠️ Yahoo 對已除牌公司好多冇業績日期 → 「業績跳空」有倖存者偏差；所以正式判斷用「價量跳空」（冇呢個偏差），業績版本作參考。
版本：
  G1 價量跳空，揸63個交易日（約3個月）就賣
  G2 價量跳空，由買入後最高收市價跌20%就賣
  G3 業績跳空，揸63日   G4 業績跳空，跌20%賣
  G5 G2 + SPY每日訊號持有先開新倉
對照：T0 線D、線E、QQQ、SPY。
判斷：① 「有改善」（同T0比，驗證期 2015-01 至 2026-09）② 四條問題 ③ 跨年代合格（2000-2009、2010-2014、2015-2019、2020-2026
  每段年化 ≥ SPY 而且最大跌幅好過 -35%）④ Esther 目標（驗證期 ≥ 50% 而且最大跌幅好過 -35%）。
另外（描述）：價量跳空之後 21/63/126 日表現（中位數、平均、贏SPY機率），分「有業績」同「冇業績」；每年平均幾多次跳空。
輸出：v30_results.csv、v30_progress.txt
==== 結果（2026-10-14 第一次跑，記錄低，唔再改）====
- 價量跳空 2,277 次（每年約85次）；有業績日期 679/719 隻；業績跳空 1,356 次。
- 全部淘汰：G1 價量跳空揸63日 +2.11% / -54.9%；G2 跌20%賣 +4.46% / -30.6%；G3 業績跳空揸63日 +6.08% / -40.4%；
  G4 業績跳空跌20%賣 +8.70% / -34.3%；G5 G2+SPY訊號 +6.12% / -26.6%（2000-2009 +8.5% / -31%）。
- 描述：跳空之後6個月中位數 +4.5%（有業績）/ +5.0%（冇），贏SPY 47% / 51% → 標普500大公司業績跳空冇明顯優勢。
"""
import time

import numpy as np
import pandas as pd

import data_cache as DC
import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v15 import fetch_earn, known_index
from research_v26 import sim

M.PROGRESS = "v30_progress.txt"
log = M.log
END = "2026-09-30"


def load_cv(tickers, start, end):
    C, V = {}, {}
    for n, t in enumerate(tickers):
        for attempt in range(3):
            try:
                h = DC.history(t, start, end)
                if h is not None and len(h) > 60:
                    idx = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
                    C[t] = pd.Series(h["Close"].astype(float).values, index=idx)
                    V[t] = pd.Series(h["Volume"].astype(float).values, index=idx)
                    C[t] = C[t][~C[t].index.duplicated(keep="last")]
                    V[t] = V[t][~V[t].index.duplicated(keep="last")]
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        time.sleep(0.2)
        if (n + 1) % 200 == 0:
            log(f"下載進度 {n + 1}/{len(tickers)}，有數據 {len(C)}")
    return C, V


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    C, Vd = load_cv(everyone + ["SPY", "QQQ"], "1998-01-01", "2026-10-01")
    spy, qqq = C.pop("SPY"), C.pop("QQQ")
    Vd.pop("SPY", None)
    Vd.pop("QQQ", None)
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    px = pd.DataFrame(C).reindex(days)
    vol = pd.DataFrame(Vd).reindex(days)[px.columns]
    cols = list(px.columns)
    P = px.values
    rets = px.pct_change(fill_method=None)
    rets_v = rets.values
    ma50, ma200 = px.rolling(50).mean().values, px.rolling(200).mean().values
    r12n = np.nan_to_num((px / px.shift(252) - 1).values, nan=-9)
    ye = px.groupby(days.year).last()
    ytdn = np.nan_to_num(P / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    hi52 = px.rolling(252, min_periods=200).max().values
    v50 = vol.shift(1).rolling(50, min_periods=40).mean().values
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
    gap = mem & (np.nan_to_num(rets_v, nan=0) >= 0.07) & (np.nan_to_num(vol.values, nan=0) >= 3 * np.nan_to_num(v50, nan=np.inf)) & (P > ma200) & fin
    log(f"價量跳空：{int(gap.sum())} 次（每年平均 {gap.sum() / 26.7:.0f} 次）")
    # 業績日期
    earn = fetch_earn(cols)
    edays = np.zeros(P.shape, dtype=bool)
    for t, df_e in earn.items():
        c = ix[t]
        for ts in df_e.index:
            k = known_index(days, ts)
            if k is None:
                continue
            for kk in (k, k + 1):
                if 0 <= kk < len(days):
                    edays[kk, c] = True
    egap = gap & edays
    log(f"有業績日期嘅股票：{len(earn)}/{len(cols)}；業績跳空：{int(egap.sum())} 次")
    # 跳空之後5日內可以買；排名用跳空嗰日升幅
    def window(ev):
        ok = np.zeros(P.shape, dtype=bool)
        rank = np.full(P.shape, -9.0)
        rv = np.nan_to_num(rets_v, nan=0)
        for lag in range(5):
            sh = np.zeros(P.shape, dtype=bool)
            sh[lag:] = ev[:len(days) - lag]
            rr = np.full(P.shape, -9.0)
            rr[lag:] = np.where(ev[:len(days) - lag], rv[:len(days) - lag], -9.0)
            ok |= sh
            rank = np.maximum(rank, rr)
        return ok & mem & fin, rank
    g_ok, g_rank = window(gap)
    e_ok2, e_rank = window(egap)
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    sig_ok = (df["state"] != "out").reindex(days).fillna(False).values.astype(bool)
    base_ok = mem & (r12n >= 1.0) & (P > ma50) & (P > ma200) & fin
    e_ok = mem & (ytdn >= 0.30) & (P >= np.nan_to_num(hi52, nan=np.inf) * 0.7) & (P > ma200) & fin
    log("數據準備好")


    def trail20(i, c, p, pk):
        return p <= pk * 0.8

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    def make_fixed(k):
        state = {}

        def fn(i, c, p, pk):
            key = c
            if key not in state or state[key][1] < i - 1:      # 新一次持有（之前冇連續睇到）
                state[key] = [i, i]
            state[key][1] = i
            if i - state[key][0] >= k:
                del state[key]
                return True
            return False
        return fn

    V, TR = {}, {}
    for nm, args in (("T0 線D（對照）", (base_ok, r12n, trail20, None)), ("線E（對照）", (e_ok, ytdn, broken, None)),
                     ("G1 價量跳空・揸63日", (g_ok, g_rank, make_fixed(63), None)), ("G2 價量跳空・跌20%賣", (g_ok, g_rank, trail20, None)),
                     ("G3 業績跳空・揸63日", (e_ok2, e_rank, make_fixed(63), None)), ("G4 業績跳空・跌20%賣", (e_ok2, e_rank, trail20, None)),
                     ("G5 價量跳空・跌20%賣・SPY訊號", (g_ok, g_rank, trail20, sig_ok))):
        ok, rk, fn, al = args
        r, tr = sim(P, rets_v, cr, ok, rk, fn, allow=al)
        V[nm], TR[nm] = pd.Series(r, index=days), tr
        log(f"{nm} 完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    allser.update(V)
    tab = {}
    L = ["=== 第三十輪：業績跳空（規則預先登記，見research_v30.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻",
         f"價量跳空 {int(gap.sum())} 次；有業績日期嘅股票 {len(earn)}/{len(cols)} 隻；業績跳空 {int(egap.sum())} 次（有倖存者偏差）", ""]
    for wn, a, b in windows:
        L.append(f"=== {wn}（{a} 至 {b}）===")
        for nm, r in allser.items():
            rr = r.loc[a:b].dropna()
            p = perf(rr, rf=cash_r, bench=qqq_r.loc[a:b])
            tab[(wn, nm)] = p
            L.append(f"　{nm}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}")
        L.append("")
    L.append("=== 每次買賣（1999-2026）===")
    for nm, tr in TR.items():
        tr = np.array(tr)
        if len(tr):
            L.append(f"　{nm}：{len(tr)} 次 ｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
                     f"蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}%")
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
    b0 = tab[("驗證期", "T0 線D（對照）")]
    L.append(f"=== 預先登記判斷 ｜ T0 {b0['cagr']:+.2f}% / {b0['mdd']:.1f}%（應該 +21.38% / -33.8%）｜ QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% ｜ SPY {s['cagr']:+.2f}% ===")
    goal = []
    for nm in V:
        x, e = tab[("驗證期", nm)], tab[("2000-2009", nm)]
        marks = "".join("✔" if (tab[(er, nm)]["cagr"] >= tab[(er, "SPY長揸")]["cagr"] and tab[(er, nm)]["mdd"] > -35) else "✘" for er in eras)
        imp = "—"
        if nm.startswith("G"):
            better = (x["mdd"] >= b0["mdd"] + 3 and x["cagr"] >= b0["cagr"] - 1) or (x["cagr"] >= b0["cagr"] + 2 and x["mdd"] >= b0["mdd"] - 2)
            imp = "有改善" if better else "冇改善"
        q1 = x["cagr"] > q["cagr"]
        q2 = q1 and x["mdd"] > -35 and x["sharpe"] >= q["sharpe"]
        q3 = (not q1) and x["mdd"] >= q["mdd"] + 5 and x["cagr"] >= s["cagr"]
        q4 = e["mdd"] > -35
        if x["cagr"] >= 50 and x["mdd"] > -35:
            goal.append(nm)
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} ｜ 同線D比 {imp} ｜ 四條問題 ①{'✔' if q1 else '✘'} "
                 f"②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'} ⇒ {'保留' if (q2 or q3) else '淘汰'} ｜ "
                 f"跨年代 {marks} {'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 "
                 + " / ".join(f"{tab[(er, nm)]['cagr']:+.1f}%" for er in eras) + " ｜ 各段最大跌幅 "
                 + " / ".join(f"{tab[(er, nm)]['mdd']:.0f}%" for er in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
    L.append(f"　Esther目標（驗證期 ≥50% 而且最大跌幅好過 -35%）：{'、'.join(goal) if goal else '冇任何版本做到'}")
    L.append("")
    # 描述
    L.append("=== 描述：價量跳空之後表現（2000-2026；同一隻股票可以有好多次）===")
    S = spy.values
    rows = []
    for i, c in zip(*np.where(gap)):
        if days[i] < pd.Timestamp("2000-01-01"):
            continue
        for h in (21, 63, 126):
            if i + 1 + h < len(days) and np.isfinite(P[i + 1 + h, c]) and np.isfinite(P[i + 1, c]):
                rows.append((bool(edays[i, c]), h, P[i + 1 + h, c] / P[i + 1, c] - 1, S[i + 1 + h] / S[i + 1] - 1))
    D = pd.DataFrame(rows, columns=["earn", "h", "fwd", "spy"])
    for h, lab in ((21, "1個月"), (63, "3個月"), (126, "6個月")):
        for ev, el in ((True, "有業績"), (False, "冇業績（或者冇業績日期）")):
            sub = D[(D["h"] == h) & (D["earn"] == ev)]
            if len(sub) >= 20:
                L.append(f"　之後{lab}・{el}：樣本 {len(sub)} ｜ 中位數 {sub['fwd'].median() * 100:+.1f}% ｜ 平均 {sub['fwd'].mean() * 100:+.1f}% ｜ "
                         f"贏SPY {(sub['fwd'] > sub['spy']).mean() * 100:.0f}%")
    L.append("　（由跳空翌日收市計起，即係實際買得到嘅價錢。）")
    pd.DataFrame({"report": L}).to_csv("v30_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
