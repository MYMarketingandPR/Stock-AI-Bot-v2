"""
research_v35.py — 第三十五輪：業績公布前避開（Esther 2026-10-15：「呢兩樣我覺得非常值得試」）

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
業績日：Yahoo 業績日期（同第十五、三十輪一樣）。「反應日」= 市場第一個可以反應嘅交易日
  （公布喺收市前 = 當日；收市後或者冇時間 = 下一個交易日）。冇業績日期嘅股票 = 規則唔適用（照原本做）。
共同設定同第三十三、三十四輪一樣（當時真實標普500、已計股息、收市決定→下一個交易日收市、0.1%、揸5隻、閒錢收^IRX）。

版本（每條線2個；只有呢啲，跑完唔加唔減唔改數字）：
  R1 業績前唔買：下一個反應日喺之後 10 個交易日內 → 唔買（已揸嘅照舊）
  R2 業績前先賣：R1 + 已揸嘅股票，如果下一個反應日係 2 至 3 個交易日之後 → 今日決定賣（下一個交易日收市賣，
     即係反應日之前最少一日已經賣咗）；反應日過咗之後，如果仲符合條件可以再買（照平時揀股次序）。
對照：線D、線E。

判斷（同第三十三輪一樣，同自己對照線比）：
  驗證期 2015-01 至 2026-09。有改善：最大跌幅好過對照 ≥3點而年化唔低過對照超過1點；或者年化高過對照 ≥2點而
  最大跌幅唔差過對照超過2點。四條問題；跨年代合格；最差5年年化。
  就算合格都唔會直接改模擬交易，要你同意先加新線。
另外（描述）：
  ① 每年有業績日期嘅股票比例（睇數據由幾時開始齊）
  ② 符合線D或線E條件嘅股票：反應日一日回報分佈（中位數、跌 ≥10% 機率），同非反應日比
  ③ 所有「一日跌 ≥10%」入面，幾多成係發生喺反應日

已知限制：
  - Yahoo 對已除牌公司好多冇業績日期 → 呢啲股票規則唔適用；早年數據可能唔齊（見描述①）。
  - 用咗實際業績日期；現實中公司通常2至4星期前先公布日期，10日內大多已知，但唔係100%。

輸出：v35_results.csv、v35_progress.txt
==== 結果（2026-10-15 第一次跑，run 38742220608，記錄低，唔再改）====
- 有業績日期 670/716 隻；成分股覆蓋率：2000-01 年只有 2-4%，2002 年起 80%，2015 年起 ≥94%。
- 動量股反應日：一日中位數 -0.16%，一日跌≥10% 機率 4.1%（其他日子 0.15%，即係約 27 倍）。
  所有成分股一日跌≥10% 嘅日子，19% 喺反應日；動量股 28% 喺反應日。
- 線D：R1 業績前唔買 +24.74% / -37.0%；R2 業績前先賣 +15.54% / -39.0%（買賣 240 → 613 次）→ 全部冇改善。
- 線E：R1 +24.76% / -33.7%；R2 +20.78% / -36.3%（買賣 279 → 807 次）→ 全部冇改善。
- 意思：業績日的確係大跌高危日，但避開佢同時避開咗業績大升（線E最好一單 +669% → +248%），加上買賣次數多咗兩倍，得不償失。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v15 import fetch_earn, known_index
from research_v26 import sim

M.PROGRESS = "v35_progress.txt"
log = M.log
END = "2026-09-30"
BIG = 10 ** 6


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
    n_d, n_c = P.shape
    rets_v = px.pct_change(fill_method=None).values
    ma50, ma200 = px.rolling(50).mean().values, px.rolling(200).mean().values
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

    earn = fetch_earn(cols)
    react = np.zeros(P.shape, dtype=bool)
    has = np.zeros(n_c, dtype=bool)
    first_year = {}
    for t, dfe in earn.items():
        c = ix[t]
        ks = set()
        for ts in dfe.index:
            k = known_index(days, ts)
            if k is not None:
                ks.add(k)
        if ks:
            has[c] = True
            react[sorted(ks), c] = True
            first_year[t] = days[min(ks)].year
    # 下一個反應日距離（交易日）：dist[i,c] = 最細 k>i 嘅 k-i
    dist = np.full(P.shape, BIG, dtype=np.int64)
    nxt = np.full(n_c, BIG, dtype=np.int64)
    for i in range(n_d - 1, -1, -1):
        dist[i] = np.where(nxt < BIG, nxt - i, BIG)
        nxt = np.where(react[i], i, nxt)
    log(f"有業績日期：{has.sum()}/{n_c} 隻")

    d_ok = mem & (r12n >= 1.0) & (P > ma50) & (P > ma200) & fin
    e_ok = mem & (ytdn >= 0.30) & (P >= hi52f * 0.7) & (P > ma200) & fin
    soon = (dist >= 1) & (dist <= 10)

    def trail20(i, c, p, pk):
        return p <= pk * 0.8

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    def pre_sell(f):
        return lambda i, c, p, pk: f(i, c, p, pk) or (2 <= dist[i, c] <= 3)

    V = {}
    for base, ok, rank, sell in [("線D", d_ok, r12n, trail20), ("線E", e_ok, ytdn, broken)]:
        V[f"{base}（對照）"] = sim(P, rets_v, cr, ok, rank, sell)
        V[f"{base}・R1 業績前唔買"] = sim(P, rets_v, cr, ok & ~soon, rank, sell)
        V[f"{base}・R2 業績前先賣"] = sim(P, rets_v, cr, ok & ~soon, rank, pre_sell(sell))
        log(f"{base} 完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)
    tab = {}
    L = ["=== 第三十五輪：業績公布前避開（規則預先登記，見research_v35.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻；有業績日期 {has.sum()}/{n_c} 隻", ""]
    # 描述①：每年有業績日期嘅成分股比例（嗰年有最少一個反應日）
    yr = days.year
    cov = []
    for y in range(2000, 2027):
        sel = (yr == y)
        m_any = mem[sel].any(axis=0)
        r_any = react[sel].any(axis=0)
        cov.append(f"{y} {(r_any & m_any).sum() / max(m_any.sum(), 1) * 100:.0f}%")
    L.append("① 每年有業績日期嘅成分股比例：" + "，".join(cov))
    # 描述②③
    mom = (d_ok | e_ok)
    rv = rets_v
    nxt_r = np.full(P.shape, np.nan)
    nxt_r[:-1] = rv[1:]                     # 第 i+1 日回報（決定日之後）
    mask_mom = mom & np.isfinite(nxt_r)
    react_next = np.zeros(P.shape, dtype=bool)
    react_next[:-1] = react[1:]
    a = nxt_r[mask_mom & react_next]
    b = nxt_r[mask_mom & ~react_next]
    L.append(f"② 動量股（符合線D或線E條件）：反應日一日回報中位數 {np.median(a) * 100:+.2f}%，跌≥10% 機率 {(a <= -0.10).mean() * 100:.1f}%"
             f"（{len(a)} 次）｜ 其他日子中位數 {np.median(b) * 100:+.2f}%，跌≥10% 機率 {(b <= -0.10).mean() * 100:.2f}%（{len(b)} 次）")
    allr = mem & np.isfinite(rv) & has[None, :]
    drop = allr & (rv <= -0.10)
    L.append(f"③ 有業績日期嘅成分股，一日跌≥10% 一共 {drop.sum()} 次，其中喺反應日 {(drop & react).sum()} 次"
             f"（{(drop & react).sum() / max(drop.sum(), 1) * 100:.0f}%）")
    dm = drop & mom
    L.append(f"　動量股一日跌≥10% 一共 {dm.sum()} 次，其中喺反應日 {(dm & react).sum()} 次（{(dm & react).sum() / max(dm.sum(), 1) * 100:.0f}%）")
    L.append("")
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
            d1 = d0 + pd.DateOffset(years=5)
            if d1 > nav.index[-1]:
                break
            j0, j1 = nav.index.get_loc(d0), nav.index.searchsorted(d1) - 1
            out.append((nav.iat[j1] / nav.iat[j0]) ** (1 / 5) - 1)
        return min(out) * 100

    eras = ["2000-2009", "2010-2014", "2015-2019", "2020-2026"]
    q, s = tab[("驗證期", "QQQ長揸")], tab[("驗證期", "SPY長揸")]
    for base in ["線D", "線E"]:
        b0 = tab[("驗證期", f"{base}（對照）")]
        L.append(f"=== 預先登記判斷：{base} ｜ 對照 {b0['cagr']:+.2f}% / {b0['mdd']:.1f}% ｜ QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% / Sharpe {q['sharpe']:.2f} ｜ SPY {s['cagr']:+.2f}% ===")
        for nm in [k for k in V if k.startswith(base)]:
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
            L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} ｜ 同{base}比 {imp} ｜ "
                     f"四條問題 ①{'✔' if q1 else '✘'} ②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'} ⇒ "
                     f"{'保留' if (q2 or q3) else '淘汰'} ｜ 跨年代 {marks} {'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 "
                     + " / ".join(f"{tab[(er, nm)]['cagr']:+.1f}%" for er in eras) + " ｜ 各段最大跌幅 "
                     + " / ".join(f"{tab[(er, nm)]['mdd']:.0f}%" for er in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
        L.append("")
    pd.DataFrame({"report": L}).to_csv("v35_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
