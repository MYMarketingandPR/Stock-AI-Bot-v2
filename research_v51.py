"""
research_v51.py — 第五十一輪：止損位探索（2000–2014）+ 驗證（2015–2026）
  （Esther 2026-11-12：「用 2000 至 2014 年數據，喺一個預先列出嘅候選清單入面揀止損位…之後再用 2015-2026 做驗證」）

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：

入場（所有候選共用）：線E 入場（今年 ≥ +30%、收市 ≥ 52週最高收市 × 0.7、收市 > 200日平均；按今年升幅排名）。
持倉：最多 5 隻，等額；每次買賣 0.1%；收市決定 → 下一個交易日收市成交；閒錢收 ^IRX。
股票池：當時真實標普500（M.sp500_history），已計股息。

候選出場（只測出場，入場全部一樣）：
  C0 線E出場（對照）：收市 < 200日平均，或 收市 < 52週最高收市 × 0.7。
  C1 10日最低收市：買入執行日計算嘅 10 日最低收市，之後收市跌穿就賣。
  C2 20日最低收市：同 C1，但用 20 日（第四十九輪嗰個止損位）。
  C3 50日最低收市：同 C1，但用 50 日。
  C4 收市跌穿50日線：每日收市 < 50日平均 就賣（動態）。
  C5 由買入價跌10%：買入執行價 × 0.9 為止損位。
  觸發後一律下一個交易日收市賣出。

探索規則（2000-01-03 至 2014-12-31）：
  在 C0–C5 之中，揀「年化回報最高」**而且**「最大跌幅 > −35%」嘅一個。
  如果冇任何候選符合，就寫「探索冇結果」。

驗證規則（2015-01-01 至 2026-09-30）：
  將探索揀中嗰個，同 C0（線E）比較。
  判斷標準：同第四十七至四十九輪一樣（有改善／四條問題／跨年代／硬性 -35%／最差5年年化）。
  同時列出 C1–C5 喺驗證期嘅結果，只供參考，唔用嚟改揀中嘅候選。

每次止損嘅統計（回答「會唔會蝕太多」同「止損後會唔會反彈」）：
  每個候選列出：止損次數、止損平均回報、賺錢比例、止損後 60 日最高升幅平均（反彈）、止損後 60 日最低跌幅平均。

已知限制：
  - 2015–2026 數據已經喺第四十九及第五十輪見過，所以驗證期有少少污染。
  - 候選清單有限（6 個），仍然有「撞大運」嘅風險。
  - 倖存者偏差（已除牌嘅股票可能冇數據）。

輸出：v51_results.csv、v51_events.csv、v51_progress.txt
==== 結果（2026-11-12 跑完，run 43146759537，記錄低，唔再改）====
探索期 2000–2014：C0 線E出場 +8.17%／−61.7%（年化最高，但跌幅超過 −35%）；C1 −5.26%／−84.2%；C2 +0.78%／−79.5%；C3 −2.40%／−82.8%；C4 −1.37%／−71.8%；C5 −0.81%／−78.1%。
⇒ 冇任何候選符合「最大跌幅 > −35%」，探索冇結果，按預先登記唔揀中、唔做驗證判斷。
驗證期 2015–2026（只供參考）：C0 +28.74%／−32.6%（Sharpe 0.90）；C1 +14.24%／−49.3%；C2 +15.20%／−26.6%；C3 +26.83%／−45.1%；C4 +9.80%／−40.4%；C5 +17.24%／−54.4%。
止損統計（全期 2000–2026）：
  C0 線E出場：279 次，平均 +15.3%，賺錢 41%，持有平均 119 日，賣後60日最高平均 +20.6%，最低平均 −14.1%
  C1 10日最低收市：72 次，平均 −11.4%，賺錢 8%，持有 240 日，賣後60日最高 +25.1%，最低 −20.0%
  C2 20日最低收市：26 次，平均 −19.8%，賺錢 4%，持有 586 日，賣後60日最高 +25.5%，最低 −19.6%
  C3 50日最低收市：18 次，平均 −32.0%，賺錢 6%，持有 646 日，賣後60日最高 +34.2%，最低 −10.8%
  C4 收市跌穿50日線：2296 次，平均 +0.8%，賺錢 46%，持有 14 日，賣後60日最高 +19.4%，最低 −12.8%
  C5 由買入價跌10%：51 次，平均 −12.7%，賺錢 0%，持有 319 日，賣後60日最高 +26.6%，最低 −14.2%
重點：
  - 跌穿 10/20/50 日低位嘅止損，幾乎全部係蝕錢先賣，而且賣後 60 日內平均仲有 +25% 至 +34% 反彈；即係「止損太快」係問題之一。
  - 但線E出場（C0）喺兩段期間都係年化最高嘅出場方法，代價係 2000–2014 最大跌幅 −61.7%；單靠簡單止損，冇辦法同時壓低跌幅同保留升幅。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import COST

M.PROGRESS = "v51_progress.txt"
log = M.log
END = "2026-09-30"
N = 5
CANDS = ["C0 線E出場", "C1 10日最低收市", "C2 20日最低收市", "C3 50日最低收市", "C4 收市跌穿50日線", "C5 由買入價跌10%"]
DISC = ("2000-01-03", "2014-12-31")
VAL = ("2015-01-01", END)


def sim_stop(P, rets_v, cr, ok, rank, kind, S10, S20, S50, MA50, MA200, HI52, ev, name, days):
    T, C = P.shape
    hold, info, cash = {}, {}, 1.0
    ps, pb, navs = [], [], []
    for i in range(T):
        for c in list(hold):
            r = rets_v[i, c]
            if np.isfinite(r):
                hold[c] *= 1 + r
        cash *= 1 + cr[i]
        for c, why in ps:
            if c in hold:
                v = hold.pop(c)
                cash += v * (1 - COST)
                b = info.pop(c)
                sp = P[i, c] if np.isfinite(P[i, c]) else b["last"]
                ev.append(dict(候選=name, 買入日=b["date"], 賣出日=str(days[i].date()), 股票=b["tk"], 買入價=b["px"], 賣出價=sp,
                               回報=sp / b["px"] - 1, 持有日=i - b["day"], 原因=why, 止損位=b["stop"]))
        total = cash + sum(hold.values())
        for c, sc in pb:
            if c in hold or len(hold) >= N or not np.isfinite(P[i, c]):
                continue
            amt = min(total / N, cash)
            if amt <= total * 0.01:
                break
            cash -= amt
            hold[c] = amt * (1 - COST)
            if kind == "C1":
                stop = S10[i, c]
            elif kind == "C2":
                stop = S20[i, c]
            elif kind == "C3":
                stop = S50[i, c]
            elif kind == "C5":
                stop = 0.9 * P[i, c]
            else:
                stop = np.nan
            info[c] = {"px": float(P[i, c]), "last": float(P[i, c]), "day": i, "date": str(days[i].date()),
                       "stop": float(stop) if np.isfinite(stop) else np.nan, "tk": int(c)}
        navs.append(cash + sum(hold.values()))
        ps, pb = [], []
        for c in hold:
            p = P[i, c]
            if not np.isfinite(p):
                if i + 1 < T and not np.isfinite(P[i + 1, c]):
                    ps.append((c, "無價格"))
                continue
            info[c]["last"] = float(p)
            why = None
            if kind == "C0":
                a = np.isfinite(MA200[i, c]) and p < MA200[i, c]
                b_ = np.isfinite(HI52[i, c]) and p < HI52[i, c] * 0.7
                if a or b_:
                    why = "線E條件（跌穿200日線或離高位30%）"
            elif kind == "C4":
                if np.isfinite(MA50[i, c]) and p < MA50[i, c]:
                    why = "收市跌穿50日線"
            else:
                st = info[c]["stop"]
                if np.isfinite(st) and p < st:
                    why = f"收市跌穿止損位（{kind}）"
            if why:
                ps.append((c, why))
        free = N - (len(hold) - len(ps))
        if free > 0:
            sold = {c for c, _ in ps}
            cand = [c for c in np.where(ok[i])[0] if c not in hold and c not in sold]
            cand.sort(key=lambda c: -rank[i, c])
            pb = [(c, rank[i, c]) for c in cand[:free]]
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1]


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
    MA200 = px.rolling(200).mean().values
    MA50 = px.rolling(50).mean().values
    S10 = px.rolling(10, min_periods=10).min().values
    S20 = px.rolling(20, min_periods=20).min().values
    S50 = px.rolling(50, min_periods=50).min().values
    ye = px.groupby(days.year).last()
    ytdn = np.nan_to_num(P / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    HI52 = px.rolling(252, min_periods=200).max().values
    hi52f = np.nan_to_num(HI52, nan=np.inf)
    cr = get_cash(days).values
    cash_r = get_cash(days)
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
    ok = mem & (ytdn >= 0.30) & fin & (P >= hi52f * 0.7) & np.nan_to_num(P > MA200, nan=False)

    ev = []
    R = {}
    for k in ["C0", "C1", "C2", "C3", "C4", "C5"]:
        name = dict(zip(["C0", "C1", "C2", "C3", "C4", "C5"], CANDS))[k]
        R[name] = sim_stop(P, rets_v, cr, ok, ytdn, k, S10, S20, S50, MA50, MA200, HI52, ev, name, days)
        log(f"{name} 完成")

    def stats(r, a, b):
        rr = pd.Series(r, index=days).loc[a:b].dropna()
        return perf(rr, rf=cash_r, bench=qqq_r.loc[a:b])

    disc = {nm: stats(r, *DISC) for nm, r in R.items()}
    valid = {nm: stats(r, *VAL) for nm, r in R.items()}
    eligible = {nm: d for nm, d in disc.items() if d["mdd"] > -35}
    winner = max(eligible, key=lambda nm: eligible[nm]["cagr"]) if eligible else None

    df = pd.DataFrame(ev)
    if len(df):
        dix = {str(d.date()): k for k, d in enumerate(days)}
        up, dn = [], []
        for _, e in df.iterrows():
            i_ = dix[e["賣出日"]]
            c_ = int(e["股票"])
            seg = P[i_:min(i_ + 61, len(days)), c_]
            seg = seg[np.isfinite(seg)]
            up.append(seg.max() / P[i_, c_] - 1 if len(seg) > 1 else np.nan)
            dn.append(seg.min() / P[i_, c_] - 1 if len(seg) > 1 else np.nan)
        df["賣後60日最高"] = up
        df["賣後60日最低"] = dn
        df["股票"] = df["股票"].map(lambda c: cols[int(c)])
        df.to_csv("v51_events.csv", index=False, encoding="utf-8-sig")

    L = ["=== 第五十一輪：止損位探索（2000–2014）→ 驗證（2015–2026）（規則預先登記，見research_v51.py開頭）===", ""]
    L.append(f"=== 探索期 {DISC[0]} 至 {DISC[1]}（揀年化最高，且最大跌幅 > -35%）===")
    for nm in R:
        d = disc[nm]
        L.append(f"　{nm}：年化 {d['cagr']:+.2f}% ｜ 最大跌幅 {d['mdd']:.1f}% ｜ Sharpe {d['sharpe']:.2f}"
                 + ("　← 探索揀中" if nm == winner else ""))
    L.append(f"　探索結果：{winner if winner else '冇符合 -35% 跌幅限制嘅候選，探索冇結果'}")
    L.append("")
    L.append(f"=== 驗證期 {VAL[0]} 至 {VAL[1]} ===")
    for nm in R:
        v = valid[nm]
        L.append(f"　{nm}：年化 {v['cagr']:+.2f}% ｜ 最大跌幅 {v['mdd']:.1f}% ｜ Sharpe {v['sharpe']:.2f}")
    L.append("")
    L.append("=== 止損統計（全期 2000–2026，只計賣出交易）===")
    if len(df):
        for nm, g in df.groupby("候選"):
            r_ = g["回報"]
            L.append(f"　{nm}：{len(g)} 次賣出 ｜ 平均 {r_.mean() * 100:+.1f}% ｜ 賺錢 {(r_ > 0).mean() * 100:.0f}% ｜ 持有平均 {g['持有日'].mean():.0f} 日 ｜ "
                     f"賣後60日最高平均 {g['賣後60日最高'].mean() * 100:+.1f}% ｜ 賣後60日最低平均 {g['賣後60日最低'].mean() * 100:+.1f}%")
    L.append("")
    L.append("=== 判斷（揀中嘅候選 vs 線E）===")
    if winner:
        x, b0 = valid[winner], valid["C0 線E出場"]
        better = (x["mdd"] >= b0["mdd"] + 3 and x["cagr"] >= b0["cagr"] - 1) or (x["cagr"] >= b0["cagr"] + 2 and x["mdd"] >= b0["mdd"] - 2)
        L.append(f"　{winner}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% ⇒ {'有改善' if better else '冇改善'}（線E {b0['cagr']:+.2f}% / {b0['mdd']:.1f}%）")
    pd.DataFrame({"report": L}).to_csv("v51_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
