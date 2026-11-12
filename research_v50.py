"""
research_v50.py — 第五十輪（第一步）：詳細交易日誌 + 按「股價同 SPY 對 200日線位置」分組
  （Esther 2026-11-12：「跑第50輪。第一步：詳細交易日誌，我想加埋股價處於200 MA之上/之下、SPY處於200 MA之上/之下」）

純研究，唔改任何模擬交易，唔改之前任何結果。
⚠️ 第一步只係加記錄同分組統計，**規則完全同線E、第四十九輪一樣**（唔改任何入場、出場、止賺、加倉、止損、資金比例）。
   如果 R49A/R49B 嘅驗證期結果同第四十九輪一致，就證明日誌冇改動策略。

預先登記：
  股票池、成本、資金、執行方式：同第四十九輪一樣。
  版本：線E（對照）、R49A 有200日線、R49B 冇200日線。
  每一次動作都記錄（見下面欄位）。以下係記錄欄位，唔影響結果：
    策略、日期、股票、動作（買入第一批／買入第二批／止賺一半／回調加倉／止損全部賣出／線E賣出）、
    原因、價錢（執行日收市）、金額、回報（賣出）、今年升幅（信號日）、距52週高位%（執行日）、
    股價對200日線（上／下，執行日）、SPY對200日線（上／下，執行日）、
    買入後20日、60日回報（買入）；賣出後60日最高、最低（賣出）。
  分組統計：按「股價對200日線」×「SPY對200日線」，列出每組嘅次數、平均回報、賺錢比例。
  注意：「後20日／60日」係事後資料，只用嚟分析，唔用嚟決定任何買賣。

判斷：本輪唔做「合格」判斷。目的係搵出邊類交易有效、邊類交易拖累。
輸出：v50_events.csv（完整日誌）、v50_results.csv（報告）、v50_progress.txt
==== 結果（2026-11-12 第一次跑，run 43138731007，記錄低，唔再改）====
- 核對通過：線E +28.74% / -32.6%；R49A +11.75% / -40.4%；R49B +10.53% / -39.6%（同第四十九輪一致，證明日誌冇改動策略）。
- 買入（買入後60日平均）：R49A／R49B 股價上200日線 SPY上：-0.3%／-1.1%（賺錢 52%／48%）；股價上 SPY下：+3.8%／+6.1%。
  股價下200日線嘅買入只有 1 至 7 次，樣本太少，唔可以下結論。線E 股價上 SPY上：+8.1%（賺錢 60%）。
- 賣出：
  - R49 止損（股價上200日線）：平均 -8% 至 -12%；（股價下200日線）：平均 -27% 至 -31%。
  - R49 止賺一半：全部賺錢，平均 +29% 至 +33%。
  - 線E 賣出（股價上200日線 SPY上）：平均 +30.1%；其餘三組平均 +5.6% 至 +13.0%。
- 主要發現：蝕得最大嘅交易，多數係止損時股價已經跌穿200日線，即係止損位太遲；止損後60日內平均反彈 +31% 至 +40%，同時亦有 Whipsaw。
- 模擬交易唔改。

"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import COST

M.PROGRESS = "v50_progress.txt"
log = M.log
END = "2026-09-30"
N = 5
RISK = 0.03
TP_MULT = 1.3
READD_DAYS = 40


def sim_E_log(P, rets_v, cr, ok, rank, sell_reason, ev, name, cols, days):
    """線E：同 research_v26.sim 完全一樣嘅買賣規則，加咗日誌。"""
    T = P.shape[0]
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
                ev.append(dict(策略=name, i=i, c=c, 動作="線E賣出", 原因=why, 價錢=sp, 金額=v, 回報=sp / b["px"] - 1,
                               持有日=i - b["day"], 買入日=b["date"]))
        total = cash + sum(hold.values())
        for c, ytd_sig in pb:
            if c in hold or len(hold) >= N or not np.isfinite(P[i, c]):
                continue
            amt = min(total / N, cash)
            if amt <= total * 0.01:
                break
            cash -= amt
            hold[c] = amt * (1 - COST)
            info[c] = {"px": float(P[i, c]), "last": float(P[i, c]), "day": i, "date": days[i]}
            ev.append(dict(策略=name, i=i, c=c, 動作="線E買入", 原因=f"今年升{ytd_sig * 100:+.0f}%、近高位、>200日線（按升幅排名）",
                           價錢=float(P[i, c]), 金額=amt, 今年升幅=ytd_sig))
        navs.append(cash + sum(hold.values()))
        ps, pb = [], []
        for c in hold:
            p = P[i, c]
            if not np.isfinite(p):
                if i + 1 < T and not np.isfinite(P[i + 1, c]):
                    ps.append((c, "無價格"))
                continue
            info[c]["last"] = float(p)
            why = sell_reason(i, c, float(p))
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


def sim49_log(P, rets_v, cr, elig, ytd, E9, E21, LOW20, ev, name, cols, days):
    """第四十九輪：同 research_v49.sim49 完全一樣嘅規則，加咗日誌。"""
    T, C = P.shape
    cash = 1.0
    pos = {}
    pend_exit, pend_tp, pend_readd, pend_buy = [], [], [], []
    navs = []

    def log_ev(i, c, act, why, px_, amt, **kw):
        d = dict(策略=name, i=i, c=c, 動作=act, 原因=why, 價錢=px_, 金額=amt)
        d.update(kw)
        ev.append(d)

    for i in range(T):
        for c in pos:
            r = rets_v[i, c]
            if np.isfinite(r):
                pos[c]["val"] *= 1 + r
        cash *= 1 + cr[i]
        for c, why in pend_exit:
            if c in pos:
                p = pos.pop(c)
                proceeds = p["val"] * (1 - COST)
                cash += proceeds
                log_ev(i, c, "全部賣出（止損）", why, P[i, c], proceeds, 回報=proceeds / p["cost"] - 1, 持有日=i - p["day"], 買入日=p["date"])
        pend_exit = []
        for c in pend_tp:
            if c not in pos or pos[c]["tp_done"]:
                continue
            p = pos[c]
            sell_val = 0.5 * p["val"]
            proceeds = sell_val * (1 - COST)
            cost_part = 0.5 * p["cost"]
            cash += proceeds
            p["val"] -= sell_val
            p["cost"] -= cost_part
            p["tp_done"] = True
            p["tp_day"] = i
            p["readd_amt"] = proceeds
            log_ev(i, c, "止賺一半", "資產升到投入成本1.3倍", P[i, c], proceeds, 回報=proceeds / cost_part - 1,
                   持有日=i - p["day"], 買入日=p["date"])
        pend_tp = []
        total = cash + sum(p["val"] for p in pos.values())
        for c, tr, stop, why_sig in pend_buy:
            if not np.isfinite(P[i, c]):
                continue
            px_ = P[i, c]
            if tr == 1:
                if c in pos or len(pos) >= N:
                    continue
                d = (px_ - stop) / px_
                if not d > 0:
                    continue
                target = min(RISK * total / d, total / N)
                amt = min(0.5 * target, cash)
                if amt <= 0:
                    continue
                cash -= amt
                pos[c] = {"val": amt * (1 - COST), "cost": amt, "tr": 1, "stop": stop, "target": target,
                          "day": i, "date": days[i], "tp_done": False, "tp_day": None, "readd_done": False,
                          "readd_amt": 0.0, "dipped": False}
                log_ev(i, c, "買入第一批", why_sig, px_, amt, 止損位=stop, 距止損=d, 今年升幅=ytd[i - 1, c],
                       距52週高=1 - px_ / P_hi(c, i), 股價對200日線=above200(c, i), SPY對200日線=spy_above(i))
            else:
                if c not in pos or pos[c]["tr"] != 1:
                    continue
                amt = min(0.5 * pos[c]["target"], cash)
                if amt <= 0:
                    continue
                cash -= amt
                pos[c]["val"] += amt * (1 - COST)
                pos[c]["cost"] += amt
                pos[c]["tr"] = 2
                log_ev(i, c, "買入第二批", why_sig, px_, amt, 距52週高=1 - px_ / P_hi(c, i),
                       股價對200日線=above200(c, i), SPY對200日線=spy_above(i))
        pend_buy = []
        for c in pend_readd:
            if c in pos and pos[c]["tp_done"] and not pos[c]["readd_done"]:
                amt = min(pos[c]["readd_amt"], cash)
                if amt > 0:
                    cash -= amt
                    pos[c]["val"] += amt * (1 - COST)
                    pos[c]["cost"] += amt
                    pos[c]["readd_done"] = True
                    log_ev(i, c, "回調加倉", "止賺後跌穿EMA9再重新站上", P[i, c], amt, 距52週高=1 - P[i, c] / P_hi(c, i),
                           股價對200日線=above200(c, i), SPY對200日線=spy_above(i))
        pend_readd = []
        navs.append(cash + sum(p["val"] for p in pos.values()))
        for c in list(pos):
            p_ = P[i, c]
            if not np.isfinite(p_):
                continue
            pp = pos[c]
            if p_ < pp["stop"]:
                pend_exit.append((c, "收市跌穿止損位"))
                continue
            if not pp["tp_done"] and pp["val"] >= TP_MULT * pp["cost"]:
                pend_tp.append(c)
            if pp["tp_done"] and not pp["readd_done"] and i - pp["tp_day"] <= READD_DAYS:
                if p_ <= E9[i, c]:
                    pp["dipped"] = True
                elif pp["dipped"]:
                    pend_readd.append(c)
        for c in pos:
            if any(c == x[0] for x in pend_exit) or pos[c]["tr"] != 1:
                continue
            if np.isfinite(P[i - 1, c]) and P[i - 1, c] <= E21[i - 1, c] and P[i, c] > E21[i, c]:
                pend_buy.append((c, 2, None, "收市重新站上EMA21"))
        free = N - len(pos)
        if free > 0 and i > 0:
            reclaim = elig[i] & (P[i - 1] <= E9[i - 1]) & (P[i] > E9[i]) & np.isfinite(P[i]) & np.isfinite(LOW20[i])
            idx = [c for c in np.where(reclaim)[0] if c not in pos]
            idx.sort(key=lambda c: -ytd[i, c])
            for c in idx[:free]:
                pend_buy.append((c, 1, LOW20[i, c], "回調後收市重新站上EMA9"))
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1]


# 全域變數（喺 main 設定）
HI52 = None
MA200 = None
SPY_ABOVE = None
COLS = None


def P_hi(c, i):
    v = HI52[i, c]
    return v if np.isfinite(v) and v > 0 else np.nan


def above200(c, i):
    return "上" if (np.isfinite(MA200[i, c]) and PX[i, c] > MA200[i, c]) else "下"


def spy_above(i):
    return "上" if SPY_ABOVE[i] else "下"


PX = None


def main():
    global HI52, MA200, SPY_ABOVE, COLS, PX
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    closes = M.download_close(everyone + ["SPY", "QQQ"], "1998-01-01", "2026-10-01")
    spy, qqq = closes.pop("SPY"), closes.pop("QQQ")
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    px = pd.DataFrame(closes).reindex(days)
    P = px.values
    PX = P
    cols = list(px.columns)
    COLS = cols
    log(f"收市價準備好：{len(cols)} 隻，{len(days)} 日")

    rets_v = px.pct_change(fill_method=None).values
    ma200 = px.rolling(200).mean().values
    MA200 = ma200
    E9 = px.ewm(span=9, adjust=False).mean().values
    E21 = px.ewm(span=21, adjust=False).mean().values
    LOW20 = px.rolling(20, min_periods=20).min().values
    ye = px.groupby(days.year).last()
    ytdn = np.nan_to_num(P / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    hi52 = px.rolling(252, min_periods=200).max().values
    HI52 = hi52
    hi52f = np.nan_to_num(hi52, nan=np.inf)
    cr = get_cash(days).values
    cash_r = get_cash(days)
    spy_ma = spy.rolling(200).mean().reindex(days).values
    SPY_ABOVE = np.where(np.isnan(spy_ma), True, spy.reindex(days).values > spy_ma)
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
    base_ok = mem & (ytdn >= 0.30) & fin
    e_ok = base_ok & (P >= hi52f * 0.7) & np.nan_to_num(P > ma200, nan=False)
    elig_a = base_ok & np.nan_to_num(P > ma200, nan=False)
    elig_b = base_ok

    def e_reason(i, c, p):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        if a and b:
            return "跌穿200日線，同時離52週高位超過30%"
        if a:
            return "跌穿200日線"
        if b:
            return "離52週高位超過30%"
        return None

    ev = []
    rE = sim_E_log(P, rets_v, cr, e_ok.copy(), ytdn, e_reason, ev, "線E（對照）", cols, days)
    log("線E 完成")
    ev_a, ev_b = [], []
    LOW20_ = LOW20
    rA = sim49_log(P, rets_v, cr, elig_a, ytdn, E9, E21, LOW20_, ev_a, "R49A 有200日線", cols, days)
    log("R49A 完成")
    rB = sim49_log(P, rets_v, cr, elig_b, ytdn, E9, E21, LOW20_, ev_b, "R49B 冇200日線", cols, days)
    log("R49B 完成")
    ev = ev + ev_a + ev_b

    # 補上後續回報（事後資料，只用作分析）
    rows = []
    for e in ev:
        i, c = e["i"], e["c"]
        e["日期"] = str(days[i].date())
        e["股票"] = cols[c]
        e["SPY對200日線"] = spy_above(i)
        if "股價對200日線" not in e:
            e["股價對200日線"] = above200(c, i)
        if "距52週高" in e and "距52週高%" not in e:
            pass
        if e["動作"].endswith("買入第一批") or e["動作"] in ("線E買入", "買入第二批", "回調加倉"):
            for h in (20, 60):
                j = i + h
                e[f"買入後{h}日"] = P[j, c] / P[i, c] - 1 if j < len(days) and np.isfinite(P[j, c]) and np.isfinite(P[i, c]) else np.nan
        if e["動作"] in ("線E賣出", "全部賣出（止損）", "止賺一半"):
            seg = P[i:min(i + 61, len(days)), c]
            seg = seg[np.isfinite(seg)]
            if len(seg) > 1:
                e["賣後60日最高"] = seg.max() / P[i, c] - 1
                e["賣後60日最低"] = seg.min() / P[i, c] - 1
        rows.append(e)
    df = pd.DataFrame(rows)
    df["距52週高%"] = df.get("距52週高", np.nan)
    df = df.drop(columns=[c for c in ["i", "c", "距52週高"] if c in df.columns])
    df.to_csv("v50_events.csv", index=False, encoding="utf-8-sig")
    log(f"日誌 {len(df)} 行")

    # 報告
    windows = [("驗證期", "2015-01-01", END), ("全期", "2000-01-03", END)]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r, "線E（對照）": pd.Series(rE, index=days),
              "R49A 有200日線": pd.Series(rA, index=days), "R49B 冇200日線": pd.Series(rB, index=days)}
    L = ["=== 第五十輪（第一步）：詳細交易日誌 + 股價/SPY 對 200日線 分組（規則同線E、第四十九輪一樣，只加記錄）===",
         f"股票池：當時真實標普500（{source}）；日誌見 v50_events.csv", ""]
    for wn, a_, b_ in windows:
        L.append(f"=== {wn}（{a_} 至 {b_}）===")
        for nm, r in allser.items():
            rr = r.loc[a_:b_].dropna()
            p = perf(rr, rf=cash_r, bench=qqq_r.loc[a_:b_])
            L.append(f"　{nm}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}")
        L.append("")
    L.append("=== 核對：線E 驗證期應為 +28.74% / -32.6%；R49A/R49B 應同第四十九輪結果一致 ===")
    L.append("")
    L.append("=== 分組統計：買入（按 股價對200日線 × SPY對200日線）— 買入後60日回報 ===")
    buys = df[df["動作"].str.contains("買入|回調加倉")].copy()
    if len(buys):
        for strat, gs in buys.groupby("策略"):
            for (ps_, pp_), g in gs.groupby(["股價對200日線", "SPY對200日線"]):
                x = g["買入後60日"].dropna()
                y = g["買入後20日"].dropna()
                L.append(f"　{strat}｜股價{ps_}200日線 SPY{pp_}200日線：{len(g)} 次 ｜ 後20日平均 "
                         f"{(y.mean() * 100 if len(y) else np.nan):+.1f}% ｜ 後60日平均 {(x.mean() * 100 if len(x) else np.nan):+.1f}% ｜ "
                         f"後60日賺錢 {((x > 0).mean() * 100 if len(x) else np.nan):.0f}%")
    L.append("")
    L.append("=== 分組統計：賣出（按 動作 × 股價對200日線 × SPY對200日線）— 回報 ＋ 賣出後60日最高 ===")
    sells = df[df["動作"].str.contains("賣出|止賺")].copy()
    if len(sells):
        for (strat, act), gs in sells.groupby(["策略", "動作"]):
            for (ps_, pp_), g in gs.groupby(["股價對200日線", "SPY對200日線"]):
                r_ = g["回報"].dropna()
                mx = g["賣後60日最高"].dropna()
                L.append(f"　{strat}｜{act}｜股價{ps_}200日線 SPY{pp_}200日線：{len(g)} 次 ｜ 平均回報 "
                         f"{(r_.mean() * 100 if len(r_) else np.nan):+.1f}% ｜ 賺錢 {((r_ > 0).mean() * 100 if len(r_) else np.nan):.0f}% ｜ "
                         f"賣後60日最高平均 {(mx.mean() * 100 if len(mx) else np.nan):+.1f}%")
    L.append("")
    L.append("=== 日誌行數 ===")
    L.append(f"　{len(df)} 行（買入 {len(buys)}，賣出 {len(sells)}）")
    pd.DataFrame({"report": L}).to_csv("v50_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
