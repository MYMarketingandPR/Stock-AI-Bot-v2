"""
research_v20.py — 第二十輪：「離MA200太遠唔追」+「高位爆量下跌離場」（Esther 2026-10-05 20:06 提出）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同：當時真實標普500；已計股息；收市決定→下一日收市；每次買賣0.1%；最多5隻、每隻 = 買入日總資產 ÷ 5；冇股揸現金。
倍升股買入條件（同第十五輪）：12個月升 ≥ +100% + 收市 > MA50 + 收市 > MA200；有空位揀12個月升幅最大。
基本離場：M = 收市跌穿MA200；T = 由買入後最高位跌20%（線D，即係建議第1點，之前已經測過）。
新規則：
  X1「過度延伸唔追」：買入時 收市 ÷ MA200 < 2.0（股價唔可以高過200日線兩倍）。
  X2「高位爆量下跌離場」：持有期間，如果 ① 上一日收市喺52週最高收市價3%以內（即係喺高位），
     ② 今日成交量 ≥ 50日平均成交量 × 2，③ 今日收市比上一日跌 ≥ 3% → 唔使等均線，直接賣（獲利了結）。
     ⚠️ 因為快取冇開市價，「收陰線」用「收市比上一日跌 ≥3%」代替。
  版本：M/T × {X1, X2, X1+X2} = 6個，同基準 M0、T0 比。
判斷（同自己基準比，驗證期 2015-2026）：
  「有改善」= 最大跌幅改善 ≥ 3個百分點而且年化唔少過1個百分點；或者年化多 ≥ 2個百分點而且最大跌幅差唔超過2個百分點。
  另外列出 2000-2009、全期、每次買賣統計、MRNA 嘅買賣紀錄。
另外（描述）：倍升股出現「高位爆量下跌」之後 1個月 / 3個月 表現（對比QQQ）。
輸出：v20_results.csv、v20_progress.txt
==== 結果（未跑）====
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v13 import get_cash
from research_v15b import download

M.PROGRESS = "v20_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"
N = 5


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    C, V = download(everyone + ["SPY", "QQQ"], "1998-01-01", "2026-10-01")
    spy, qqq = C.pop("SPY").astype(float), C.pop("QQQ").astype(float)
    V.pop("SPY", None)
    V.pop("QQQ", None)
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    cols = sorted(C)
    px = pd.DataFrame(C).reindex(days)[cols].astype(float)
    vol = pd.DataFrame(V).reindex(days)[cols].astype(float)
    P = px.values
    rets_v = px.pct_change(fill_method=None).values
    ma50, ma200 = px.rolling(50).mean().values, px.rolling(200).mean().values
    r12 = (px / px.shift(252) - 1).values
    hi52 = px.rolling(252, min_periods=200).max().values
    vv = vol.values
    va50 = vol.rolling(50, min_periods=40).mean().values
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
    r12n = np.nan_to_num(r12, nan=-9)
    base_ok = mem & (r12n >= 1.0) & (P > ma50) & (P > ma200) & np.isfinite(P)
    ext_ok = np.nan_to_num(P / ma200, nan=9) < 2.0
    # 高位爆量下跌：上一日喺52週高位3%以內、今日成交量 ≥ 2倍50日平均、今日跌 ≥3%
    prevP = np.vstack([np.full((1, P.shape[1]), np.nan), P[:-1]])
    prevH = np.vstack([np.full((1, P.shape[1]), np.nan), hi52[:-1]])
    dist = np.nan_to_num(prevP / prevH, nan=0) >= 0.97
    volx = np.nan_to_num(vv / va50, nan=0) >= 2.0
    drop = np.nan_to_num(P / prevP - 1, nan=0) <= -0.03
    climax = dist & volx & drop
    log(f"數據準備好；高位爆量下跌日數 {int(climax.sum())}")

    def simulate(exit_kind, x1=False, x2=False):
        ok = base_ok & ext_ok if x1 else base_ok
        hold, info, cash = {}, {}, 1.0
        ps, pb, navs, turns, trades = [], [], [], [], []
        n_cl = 0
        for i in range(len(days)):
            for c in list(hold):
                r = rets_v[i, c]
                if np.isfinite(r):
                    hold[c] *= 1 + r
            cash *= 1 + cr[i]
            traded = 0.0
            for c, why in ps:
                if c in hold:
                    v = hold.pop(c)
                    traded += v
                    cash += v * (1 - COST)
                    b = info.pop(c)
                    sp = P[i, c] if np.isfinite(P[i, c]) else b["last"]
                    trades.append({"股票": cols[c], "買入日": days[b["i"]].date(), "買入價": round(b["px"], 2), "賣出日": days[i].date(),
                                   "賣出價": round(float(sp), 2), "回報%": round((sp / b["px"] - 1) * 100, 1),
                                   "買入時12個月%": round(b["r12"] * 100), "原因": why})
            total = cash + sum(hold.values())
            for c in pb:
                if c in hold or len(hold) >= N or not np.isfinite(P[i, c]):
                    continue
                amt = min(total / N, cash)
                if amt <= total * 0.01:
                    break
                cash -= amt
                hold[c] = amt * (1 - COST)
                info[c] = {"i": i, "px": float(P[i, c]), "peak": float(P[i, c]), "last": float(P[i, c]), "r12": float(r12n[i - 1, c])}
                traded += amt
            ps, pb = [], []
            total = cash + sum(hold.values())
            navs.append(total)
            turns.append(traded / total if total > 0 else 0.0)
            for c in hold:
                p = P[i, c]
                if not np.isfinite(p):
                    if i + 1 < len(days) and not np.isfinite(P[i + 1, c]):
                        ps.append((c, "除牌"))
                    continue
                info[c]["last"] = float(p)
                info[c]["peak"] = max(info[c]["peak"], float(p))
                if x2 and climax[i, c]:
                    ps.append((c, "高位爆量下跌"))
                    n_cl += 1
                elif exit_kind == "M" and np.isfinite(ma200[i, c]) and p < ma200[i, c]:
                    ps.append((c, "跌穿MA200"))
                elif exit_kind == "T" and p <= info[c]["peak"] * 0.8:
                    ps.append((c, "由高位跌20%"))
            selling = {c for c, _ in ps}
            free = N - (len(hold) - len(selling))
            if free > 0:
                cand = [c for c in np.where(ok[i])[0] if c not in hold and c not in selling]
                cand.sort(key=lambda c: -r12n[i, c])
                pb = cand[:free]
        nav = pd.Series(navs, index=days)
        return nav.pct_change().fillna(0.0), pd.Series(turns, index=days), pd.DataFrame(trades), n_cl

    base = {"M": "跌穿MA200", "T": "由高位跌20%（線D）"}
    res = {}
    for ek, en in base.items():
        res[f"{en}・基準"] = (ek,) + simulate(ek)
        res[f"{en}・X1離MA200兩倍唔追"] = (ek,) + simulate(ek, x1=True)
        res[f"{en}・X2高位爆量下跌離場"] = (ek,) + simulate(ek, x2=True)
        res[f"{en}・X1+X2"] = (ek,) + simulate(ek, x1=True, x2=True)
        log(f"{en} 完成")

    L = ["=== 第二十輪：離MA200太遠唔追 + 高位爆量下跌離場（規則預先登記，見research_v20.py開頭）===",
         f"股票池：當時真實標普500（{source}）；最多5隻；已計股息；每次買賣0.1%；唔用槓桿；"
         f"「收陰線」用「收市比上一日跌≥3%」代替（冇開市價）", ""]

    def show(name, r, a, b, t=None):
        p = perf(r.loc[a:b], rf=cash_r, bench=qqq_r.loc[a:b], turn=t, cap_freq="ME")
        ex = f" ｜ 每年換手 {p['turn_year']:.0f}%" if t is not None else ""
        return p, f"　{name}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}{ex}"

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2019", "2010-01-01", "2019-12-31"),
               ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END), ("科網爆煲", "2000-03-01", "2002-12-31"),
               ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    tab = {}
    for wn, a, b in windows:
        L.append(f"=== {wn}（{a} 至 {b}）===")
        for nm, r in (("QQQ長揸", qqq_r), ("SPY長揸", spy_r)):
            tab[(wn, nm)], txt = show(nm, r, a, b)
            L.append(txt)
        for nm, v in res.items():
            tab[(wn, nm)], txt = show(nm, v[1], a, b, v[2])
            L.append(txt)
        L.append("")

    L.append("=== 每次買賣（1998-2026）===")
    for nm, v in res.items():
        T = v[3]
        if len(T):
            tr = T["回報%"].values
            L.append(f"　{nm}：{len(tr)} 次，賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean():+.1f}% ｜ 蝕嘅平均 {tr[tr <= 0].mean():+.1f}% ｜ "
                     f"最差 {tr.min():+.0f}% ｜ 賺最多 {tr.max():+.0f}% ｜ " + "，".join(f"{k} {c}" for k, c in T["原因"].value_counts().items()))
    L.append("")
    L.append("=== MRNA 買賣紀錄 ===")
    for nm, v in res.items():
        T = v[3]
        mr = T[T["股票"] == "MRNA"] if len(T) else T
        if len(mr):
            L.append(f"　{nm}：" + "；".join(f"{x['買入日']} 買 ${x['買入價']}（12個月 +{x['買入時12個月%']}%）→ {x['賣出日']} 賣 ${x['賣出價']}"
                                           f"（{x['回報%']:+.0f}%，{x['原因']}）" for _, x in mr.iterrows()))
        else:
            L.append(f"　{nm}：冇買過 MRNA")
    L.append("")

    L.append("=== 描述：倍升股「高位爆量下跌」之後（下一日收市計，對比QQQ）===")
    rows = []
    ev = climax & mem & (r12n >= 1.0)
    for i, c in zip(*np.where(ev)):
        if i + 64 >= len(days) or not np.isfinite(P[i + 1, c]):
            continue
        seg = pd.Series(P[i + 1:i + 65, c]).ffill().values
        rows.append((seg[21] / seg[0] - 1, qv[i + 22] / qv[i + 1] - 1, seg[63] / seg[0] - 1, qv[i + 64] / qv[i + 1] - 1))
    x = np.array(rows)
    if len(x) >= 20:
        L.append(f"　{len(x)} 次：1個月 中位數 {np.median(x[:, 0]) * 100:+.1f}% 贏QQQ {(x[:, 0] > x[:, 1]).mean() * 100:.0f}% ｜ "
                 f"3個月 中位數 {np.median(x[:, 2]) * 100:+.1f}% 平均 {x[:, 2].mean() * 100:+.1f}% 贏QQQ {(x[:, 2] > x[:, 3]).mean() * 100:.0f}% "
                 f"跌≥20% {(x[:, 2] <= -0.2).mean() * 100:.0f}%")
    rows = []
    allup = mem & (r12n >= 1.0)
    for i in range(260, len(days) - 65, 21):
        for c in np.where(allup[i])[0]:
            if np.isfinite(P[i + 1, c]):
                seg = pd.Series(P[i + 1:i + 65, c]).ffill().values
                rows.append((seg[21] / seg[0] - 1, seg[63] / seg[0] - 1))
    y = np.array(rows)
    if len(y):
        L.append(f"　對照：所有倍升股（每月抽樣 {len(y)} 次）：1個月 中位數 {np.median(y[:, 0]) * 100:+.1f}% ｜ 3個月 中位數 {np.median(y[:, 1]) * 100:+.1f}%")
    L.append("")

    L.append("=== 預先登記判斷（同基準比，驗證期 2015-2026）===")
    for nm, v in res.items():
        if nm.endswith("基準"):
            continue
        b0 = f"{base[v[0]]}・基準"
        x_, y_ = tab[("驗證期", nm)], tab[("驗證期", b0)]
        better = (x_["mdd"] >= y_["mdd"] + 3 and x_["cagr"] >= y_["cagr"] - 1) or (x_["cagr"] >= y_["cagr"] + 2 and x_["mdd"] >= y_["mdd"] - 2)
        e1, e0 = tab[("2000-2009", nm)], tab[("2000-2009", b0)]
        f1, f0 = tab[("全期", nm)], tab[("全期", b0)]
        L.append(f"　{nm}：驗證期 {x_['cagr']:+.2f}% / {x_['mdd']:.1f}%（基準 {y_['cagr']:+.2f}% / {y_['mdd']:.1f}%）→ {'有改善' if better else '冇改善'} ｜ "
                 f"2000-2009 {e1['cagr']:+.2f}% / {e1['mdd']:.1f}%（基準 {e0['cagr']:+.2f}% / {e0['mdd']:.1f}%）｜ "
                 f"全期 {f1['cagr']:+.2f}% / {f1['mdd']:.1f}%（基準 {f0['cagr']:+.2f}% / {f0['mdd']:.1f}%）")
    pd.DataFrame({"report": L}).to_csv("v20_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
