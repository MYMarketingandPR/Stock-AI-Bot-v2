"""
research_v18.py — 第十八輪：倍升股輪換 + 「乖離止賺」（Esther 2026-10-05 提出）

乖離 = 收市價 ÷ MA50 − 1（例如 +40% = 股價高過50日平均線40%）。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
股票池：當時真實標普500（U1）；已計股息；收市決定→下一日收市；每次買賣0.1%；最多5隻、每隻 = 買入日總資產 ÷ 5；
  冇股揸現金（^IRX）。（全美國股票池之前幾輪都係垃圾股太多，今次唔測。）
買入（同第十五輪）：12個月升 ≥ +100% + 收市 > MA50 + 收市 > MA200；有空位揀12個月升幅最大；
  加乖離止賺嘅版本，買入時乖離要 < X（否則一買即刻要賣）。賣咗之後再符合條件可以再買。
賣出：基本離場（兩種）＋ 乖離止賺（邊個先到用邊個）：
  基本離場 M = 收市跌穿MA200；基本離場 T = 由買入後最高位跌20%（線D）
  乖離止賺 X = 冇 / 20% / 30% / 40% / 50% / 60% / 80% / 100%：收市乖離 ≥ X 就賣（止賺）
  → 2 × 8 = 16 個版本。
判斷：
  ① 揀規則期 2000-2014 用評分（40%回報+30%最大跌幅+20%Sharpe+10%換手）喺16個入面揀1個，驗證期 2015-2026：
     三條問題：有冇贏QQQ？贏嘅話最大跌幅好過-35%而且Sharpe ≥ QQQ？唔贏嘅話最大跌幅好過QQQ 5個百分點而且年化 ≥ SPY？
  ② 「乖離止賺有冇用」：每種基本離場，揀規則期用評分喺 X=20%-100% 揀1個，驗證期同「冇乖離止賺」比：
     最大跌幅改善 ≥ 3個百分點，而且年化唔少過1個百分點 → 有用。
另外（描述）：倍升股乖離第一次升穿 X 之後 1個月 / 3個月 嘅表現（對比QQQ）；每個版本每次買賣嘅平均蝕幅同最差一次。
輸出：v18_results.csv、v18_progress.txt
==== 結果（未跑）====
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v10 import metrics, score_pick
from research_v13 import get_cash

M.PROGRESS = "v18_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"
N = 5
XS = [None, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    closes = M.download_close(everyone + ["SPY", "QQQ"], "1998-01-01", "2026-10-01")
    spy, qqq = closes.pop("SPY"), closes.pop("QQQ")
    spy = spy[spy.index <= pd.Timestamp(END)]
    px = pd.DataFrame(closes).reindex(spy.index)
    days, cols = px.index, list(px.columns)
    P = px.values
    rets = px.pct_change(fill_method=None).values
    ma50, ma200 = px.rolling(50).mean().values, px.rolling(200).mean().values
    bias = P / ma50 - 1
    r12 = (px / px.shift(252) - 1).values
    cash_r = get_cash(days)
    cr = cash_r.values
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = qqq.reindex(days).pct_change(fill_method=None).fillna(0.0)
    qv = qqq.reindex(days).ffill().values
    ix = {t: k for k, t in enumerate(cols)}
    mem = np.zeros(P.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True
    base_ok = mem & (np.nan_to_num(r12, nan=-9) >= 1.0) & (P > ma50) & (P > ma200) & np.isfinite(P)
    log("數據準備好")

    def simulate(exit_kind, X):
        ok = base_ok if X is None else base_ok & (np.nan_to_num(bias, nan=9) < X)
        hold, info, cash = {}, {}, 1.0
        ps, pb, navs, turns, trades = [], [], [], [], []
        why_cnt = {"跌穿MA200": 0, "由高位跌20%": 0, "乖離止賺": 0, "除牌": 0}
        for i in range(len(days)):
            for c in list(hold):
                r = rets[i, c]
                if np.isfinite(r):
                    hold[c] *= 1 + r
            cash *= 1 + cr[i]
            traded = 0.0
            for c in ps:
                if c in hold:
                    v = hold.pop(c)
                    traded += v
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
                info[c] = {"px": float(P[i, c]), "peak": float(P[i, c]), "last": float(P[i, c])}
                traded += amt
            ps, pb = [], []
            total = cash + sum(hold.values())
            navs.append(total)
            turns.append(traded / total if total > 0 else 0.0)
            for c in hold:
                p = P[i, c]
                if not np.isfinite(p):
                    if i + 1 < len(days) and not np.isfinite(P[i + 1, c]):
                        ps.append(c)
                        why_cnt["除牌"] += 1
                    continue
                info[c]["last"] = float(p)
                info[c]["peak"] = max(info[c]["peak"], float(p))
                if X is not None and np.isfinite(bias[i, c]) and bias[i, c] >= X:
                    ps.append(c)
                    why_cnt["乖離止賺"] += 1
                elif exit_kind == "M" and np.isfinite(ma200[i, c]) and p < ma200[i, c]:
                    ps.append(c)
                    why_cnt["跌穿MA200"] += 1
                elif exit_kind == "T" and p <= info[c]["peak"] * 0.8:
                    ps.append(c)
                    why_cnt["由高位跌20%"] += 1
            free = N - (len(hold) - len(ps))
            if free > 0:
                cand = [c for c in np.where(ok[i])[0] if c not in hold and c not in ps]
                cand.sort(key=lambda c: -r12[i, c])
                pb = cand[:free]
        nav = pd.Series(navs, index=days)
        return nav.pct_change().fillna(0.0), pd.Series(turns, index=days), np.array(trades), why_cnt

    res = {}
    for ek, en in (("M", "跌穿MA200"), ("T", "由高位跌20%")):
        for X in XS:
            name = f"{en}" + ("・冇乖離止賺" if X is None else f"・乖離≥{int(X * 100)}%止賺")
            res[name] = (ek, X) + simulate(ek, X)
            log(f"{name} 完成")

    L = ["=== 第十八輪：倍升股輪換 + 乖離止賺（規則預先登記，見research_v18.py開頭）===",
         f"股票池：當時真實標普500（{source}）；最多5隻；已計股息；每次買賣0.1%；唔用槓桿", ""]

    def show(name, r, a, b, t=None):
        p = perf(r.loc[a:b], rf=cash_r, bench=qqq_r.loc[a:b], turn=t, cap_freq="ME")
        ex = f" ｜ 每年換手 {p['turn_year']:.0f}%" if t is not None else ""
        return p, f"　{name}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}{ex}"

    windows = [("驗證期", "2015-01-01", END), ("揀規則期", "2000-01-03", "2014-12-31"), ("2000-2009", "2000-01-03", "2009-12-31"),
               ("2010-2019", "2010-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    tab = {}
    for wn, a, b in windows:
        L.append(f"=== {wn}（{a} 至 {b}）===")
        for nm, r in (("QQQ長揸", qqq_r), ("SPY長揸", spy_r)):
            tab[(wn, nm)], txt = show(nm, r, a, b)
            L.append(txt)
        for nm, v in res.items():
            tab[(wn, nm)], txt = show(nm, v[2], a, b, v[3])
            L.append(txt)
        L.append("")

    L.append("=== 每次買賣（1998-2026）：賺錢比例、平均、蝕嘅平均、最差一次、賣出原因 ===")
    for nm, v in res.items():
        tr, why = v[4], v[5]
        if len(tr):
            L.append(f"　{nm}：{len(tr)} 次，賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
                     f"蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 賺最多 {tr.max() * 100:+.0f}% ｜ "
                     + "，".join(f"{k} {c}" for k, c in why.items() if c))
    L.append("")

    # 描述：乖離第一次升穿X之後
    L.append("=== 描述：倍升股乖離第一次升穿 X 之後（下一日收市買入計，對比QQQ）===")
    for X in XS[1:]:
        rows = []
        cross = (np.nan_to_num(bias, nan=-9) >= X) & (np.nan_to_num(np.vstack([np.full((1, P.shape[1]), 9.0), bias[:-1]]), nan=9) < X)
        cross &= mem & (np.nan_to_num(r12, nan=-9) >= 1.0)
        for i, c in zip(*np.where(cross)):
            if i < 252 or i + 64 >= len(days) or not np.isfinite(P[i + 1, c]):
                continue
            seg = pd.Series(P[i + 1:i + 65, c]).ffill().values
            rows.append((seg[21] / seg[0] - 1, qv[i + 22] / qv[i + 1] - 1, seg[63] / seg[0] - 1, qv[i + 64] / qv[i + 1] - 1))
        x = np.array(rows)
        if len(x) >= 20:
            L.append(f"　乖離 ≥{int(X * 100)}%（{len(x)} 次）：1個月 中位數 {np.median(x[:, 0]) * 100:+.1f}% 贏QQQ {(x[:, 0] > x[:, 1]).mean() * 100:.0f}% ｜ "
                     f"3個月 中位數 {np.median(x[:, 2]) * 100:+.1f}% 平均 {x[:, 2].mean() * 100:+.1f}% 贏QQQ {(x[:, 2] > x[:, 3]).mean() * 100:.0f}% "
                     f"跌≥20% {(x[:, 2] <= -0.2).mean() * 100:.0f}%")
    L.append("")

    # 判斷
    L.append("=== 預先登記判斷 ===")
    cands = {nm: metrics(v[2], v[3], "2000-01-03", "2014-12-31") for nm, v in res.items()}
    best, sc = score_pick(cands)
    v, q, s = tab[("驗證期", best)], tab[("驗證期", "QQQ長揸")], tab[("驗證期", "SPY長揸")]
    q1 = v["cagr"] > q["cagr"]
    q2 = q1 and v["mdd"] > -35 and v["sharpe"] >= q["sharpe"]
    q3 = (not q1) and v["mdd"] >= q["mdd"] + 5 and v["cagr"] >= s["cagr"]
    L.append("　揀規則期評分最高：" + "，".join(f"{k} {x:.2f}" for k, x in sc.sort_values(ascending=False).head(5).items()))
    L.append(f"　① 揀中 {best}：驗證期 年化 {v['cagr']:+.2f}%（QQQ {q['cagr']:+.2f}%）、最大跌幅 {v['mdd']:.1f}%（QQQ {q['mdd']:.1f}%）、"
             f"Sharpe {v['sharpe']:.2f}（{q['sharpe']:.2f}）→ {'保留' if (q2 or q3) else '淘汰'}")
    for ek, en in (("M", "跌穿MA200"), ("T", "由高位跌20%")):
        sub = {nm: cands[nm] for nm, vv in res.items() if vv[0] == ek and vv[1] is not None}
        b2, _ = score_pick(sub)
        base = f"{en}・冇乖離止賺"
        vb, vx = tab[("驗證期", base)], tab[("驗證期", b2)]
        useful = vx["mdd"] >= vb["mdd"] + 3 and vx["cagr"] >= vb["cagr"] - 1
        L.append(f"　② {en}：揀規則期揀咗 {b2} → 驗證期 年化 {vx['cagr']:+.2f}% vs 冇乖離 {vb['cagr']:+.2f}%；"
                 f"最大跌幅 {vx['mdd']:.1f}% vs {vb['mdd']:.1f}% → 乖離止賺{'有用' if useful else '冇用'}")
    pd.DataFrame({"report": L}).to_csv("v18_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
