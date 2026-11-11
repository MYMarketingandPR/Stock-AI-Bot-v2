"""
research_v47.py — 第四十七輪：線E + 行業上限 + 大市濾網
  （Esther 2026-11-11：「要唔要我寫一輪新測試，例如『行業上限 + 大市濾網』？…好」→「跑」）
  背景：線E 而家（10月底至11月9日）實盤全部虧損，五隻持股入面三隻係 AI 硬件相關，買入又集中喺追高位。

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同設定：當時真實標普500成分股（主要用 M.sp500_history）、已計股息（Yahoo 調整後收市）、
  2000-01-03 至 2026-09-30、收市決定 → 下一個交易日收市成交、每次買賣 0.1%、最多 5 隻、等額、閒錢收 ^IRX。
入場同賣出 全部同線E 一樣（不改）：
  入場：今年 ≥ +30%、收市 ≥ 52週最高收市 × 0.7、收市 > 200日平均；揀今年升幅最多。
  賣出：收市 < 200日平均，或者 收市 < 52週最高收市 × 0.7。
  冷靜期：無（同線E 一樣）。

版本（只有以下 4 個，跑完唔加唔減、唔改數字）：
  基準 線E（對照）
  F1 行業上限：同一個行業最多 2 隻（同第二十四輪 A2 一樣）。排第三嘅同行業股票跳過，揀下一隻。
  F2 大市濾網：SPY 收市 > SPY 200日平均先可以揀新股票（收市決定）。SPY 未夠 200 日數據嘅日子當作通過。
       SPY 跌穿 200日線：唔買新股票；現有持股仍然按線E 賣出規則賣。
  F3 F1 + F2 合併。

行業分類：用 research_v24.sectors()（Wikipedia 標普500 GICS 行業 + Nasdaq 補充），即係「今日」嘅分類，唔係歷史分類。
  ⚠️ 已知限制：我而家搵唔到歷史行業數據，所以用今日分類去回溯。股票嘅行業可能改過，結果可能有偏差。
  冇分類嘅股票（已除牌或者搵唔到）唔受行業上限限制，會喺結果寫明有幾多隻。

判斷（同第四十三輪一樣；同線E比；驗證期 2015-01 至 2026-09）：
  「有改善」：(a) 最大跌幅 ≥ 線E + 3 點 且 年化 ≥ 線E − 1 點；或 (b) 年化 ≥ 線E + 2 點 且 最大跌幅 ≥ 線E − 2 點。
  四條問題（同第四十三輪一樣）：
    ① 驗證期年化 > QQQ
    ② 若 ① 成立：驗證期最大跌幅 > −35% 且 Sharpe ≥ QQQ 嘅 Sharpe
    ③ 若 ① 唔成立：最大跌幅 ≥ QQQ + 5 點 且 年化 ≥ SPY
    ④ 2000-2009 最大跌幅 > −35%
  保留條件：② 或 ③ 成立。
  跨年代合格：2000-2009、2010-2014、2015-2019、2020-2026 四段，每段年化 ≥ SPY 且 最大跌幅 > −35%。
  最差5年年化（每月開始計，只計完整5年）：同線E 比較。
  另外列：每次買賣統計、各段（科網爆煲、金融海嘯、2022熊市、2025年）數字。
  硬性：最大跌幅唔可以差過 −35%；唔用槓桿。
  三個版本都會報告，唔會因為某個版本最好就揀佢做結論。如果三個都唔合格，寫「冇改善」。

輸出：v47_results.csv、v47_progress.txt
==== 結果（待跑）====
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v24 import sectors

M.PROGRESS = "v47_progress.txt"
log = M.log
END = "2026-09-30"
COST = 0.001
N = 5
SECTOR_CAP = 2


def sim47(P, rets_v, cr, ok, rank, sell_fn, sec, use_sec, allow=None):
    """同 research_v26.sim 一樣嘅流程，加咗行業上限同大市濾網。"""
    T = P.shape[0]
    hold, info, cash = {}, {}, 1.0
    ps, pb, navs, trades = [], [], [], []
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
        total = cash + sum(hold.values())
        for c in pb:
            if c in hold or len(hold) >= N or not np.isfinite(P[i, c]):
                continue
            amt = min(total / N, cash)
            if amt <= total * 0.01:
                break
            cash -= amt
            hold[c] = amt * (1 - COST)
            info[c] = {"px": float(P[i, c]), "last": float(P[i, c])}
        navs.append(cash + sum(hold.values()))
        ps, pb = [], []
        for c in hold:
            p = P[i, c]
            if not np.isfinite(p):
                if i + 1 < T and not np.isfinite(P[i + 1, c]):
                    ps.append(c)
                continue
            info[c]["last"] = float(p)
            if sell_fn(i, c, float(p)):
                ps.append(c)
        # 揀下一日買入嘅股票（今日收市決定）
        if allow is not None and not allow[i]:
            continue
        kept = [c for c in hold if c not in ps]
        free = N - len(kept)
        if free <= 0:
            continue
        cnt = {}
        for c in kept:
            if sec[c] is not None:
                cnt[sec[c]] = cnt.get(sec[c], 0) + 1
        cand = [c for c in np.where(ok[i])[0] if c not in hold and c not in ps]
        cand.sort(key=lambda c: -rank[i, c])
        for c in cand:
            if len(pb) >= free:
                break
            s = sec[c]
            if use_sec and s is not None and cnt.get(s, 0) >= SECTOR_CAP:
                continue
            pb.append(c)
            if s is not None:
                cnt[s] = cnt.get(s, 0) + 1
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1], trades


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    closes = M.download_close(everyone + ["SPY", "QQQ"], "1998-01-01", "2026-10-01")
    spy, qqq = closes.pop("SPY"), closes.pop("QQQ")
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    px = pd.DataFrame(closes).reindex(days)
    P = px.values
    cols = list(px.columns)
    log(f"收市價準備好：{len(cols)} 隻，{len(days)} 日")

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
    e_ok = mem & (ytdn >= 0.30) & (P >= hi52f * 0.7) & np.nan_to_num(P > ma200, nan=False) & fin

    sm = sectors()
    sec = [sm.get(t) for t in cols]
    log(f"行業分類：{sum(s is not None for s in sec)}/{len(cols)} 隻有分類（今日分類）")

    spy_ma = spy.rolling(200).mean().reindex(days).values
    spy_v = spy.reindex(days).values
    allow_f2 = np.where(np.isnan(spy_ma), True, spy_v > spy_ma)

    def broken(i, c, p):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    V = {}
    specs = [
        ("線E（對照）", False, None),
        ("F1 行業上限", True, None),
        ("F2 大市濾網", False, allow_f2),
        ("F3 行業上限+大市濾網", True, allow_f2),
    ]
    for name, son, al in specs:
        ok = e_ok.copy()
        V[name] = sim47(P, rets_v, cr, ok, ytdn, lambda i, c, p: broken(i, c, p), sec, son, allow=al)
        log(f"{name} 完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31"),
               ("2025年", "2025-01-01", "2025-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)

    L = ["=== 第四十七輪：線E + 行業上限 + 大市濾網（規則預先登記，見research_v47.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻；行業分類為今日 GICS（有分類 {sum(s is not None for s in sec)}/{len(cols)} 隻）", ""]
    tab = {}
    for wn, a_, b_ in windows:
        L.append(f"=== {wn}（{a_} 至 {b_}）===")
        for nm, r in allser.items():
            rr = r.loc[a_:b_].dropna()
            p = perf(rr, rf=cash_r, bench=qqq_r.loc[a_:b_])
            tab[(wn, nm)] = p
            L.append(f"　{nm}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}")
        L.append("")
    yrs = (days[-1] - days[0]).days / 365.25
    L.append("=== 每次買賣（1999-2026）===")
    for nm, v in V.items():
        tr = np.array(v[1])
        L.append(f"　{nm}：{len(tr)} 次（每年約 {len(tr) / yrs:.0f} 次）｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
                 f"賺嘅平均 {tr[tr > 0].mean() * 100:+.1f}% ｜ 蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}%")
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
    pd.DataFrame({"report": L}).to_csv("v47_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
