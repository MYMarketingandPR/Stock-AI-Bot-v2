"""
research_v49.py — 第四十九輪：第四十八輪修訂入場 + 3% 風險定額 + 止賺一半 + 回調後加倉 + 止損位出場
  （Esther 2026-11-12：「1. 第四十八輪嘅修訂入場 2. EMA9 跌穿再重新站上 3. 係(40個交易日) 4. 以BE為例…止損定在157.334…」；
    之後更正：止損用 156.000（11 月回調最低）；止損用收市價；買入嗰陣設定；X = 3%；「跑49輪測試」）

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：

共同設定（同第四十七、四十八輪一樣）：
  股票池：當時真實標普500（M.sp500_history）；已計股息（Yahoo 調整後收市）；2000-01-03 至 2026-09-30。
  收市決定 → 下一個交易日收市成交；每次買賣 0.1%；最多 5 隻；閒錢收 ^IRX。
  行業上限、大市濾網：唔用。

對照：線E（同第四十七輪一樣，research_v26.sim，線E 嘅入場同出場）。

入場（R49A 同 R49B 共用）：
  合資格（每日）：今年 ≥ +30%。
  R49A 另加：收市 > 200日平均；R49B 唔要求。
  第一批（目標金額嘅一半）：合資格 且 昨日收市 ≤ EMA9 且 今日收市 > EMA9。同日多隻符合，按今年升幅由大到細揀，最多填滿空位。
  第二批（目標金額嘅另一半）：已持有第一批，而且 昨日收市 ≤ EMA21 且 今日收市 > EMA21。冇出現就只持有一半。

止損位（買入信號當日定）：
  今日（信號日）及之前 19 個交易日，合共 20 日嘅最低收市價。
  執行價 ≤ 止損位，唔買入。

買入金額（第一批定目標金額）：
  目標金額 = min( 3% × 總資產 ÷ ((執行價 − 止損位) ÷ 執行價) ，總資產 ÷ 5 )。
  即係：止損位跌穿時，呢隻股票最多蝕總資產 3%。
  第一批 = 目標金額 × 0.5；第二批 = 目標金額 × 0.5（按第一批嘅目標金額，唔按當日總資產）。

止賺（每隻股票只做一次）：
  資產 ≥ 投入成本 × 1.3（投入成本已包括買入費用）→ 下一個交易日賣出一半。

回調後加倉（每隻股票只做一次）：
  止賺之後，收市 ≤ EMA9，之後再出現收市 > EMA9，而且距止賺日 ≤ 40 個交易日 → 下一個交易日用止賺所得金額買返。
  40 個交易日內冇觸發就取消。

出場（R49A 同 R49B 共用）：
  只有止損：收市 < 止損位 → 下一個交易日全部賣出。
  ⚠️ 取消咗線E嘅「跌穿200日線」同「離高位跌30%」出場，同第四十八輪唔同。

判斷（同第四十七、四十八輪一樣；同線E比；驗證期 2015-01 至 2026-09）：
  「有改善」：(a) 最大跌幅 ≥ 線E + 3 點 且 年化 ≥ 線E − 1 點；或 (b) 年化 ≥ 線E + 2 點 且 最大跌幅 ≥ 線E − 2 點。
  四條問題（同第四十三、四十七、四十八輪一樣）：
    ① 驗證期年化 > QQQ
    ② 若 ① 成立：驗證期最大跌幅 > −35% 且 Sharpe ≥ QQQ 嘅 Sharpe
    ③ 若 ① 唔成立：最大跌幅 ≥ QQQ + 5 點 且 年化 ≥ SPY
    ④ 2000-2009 最大跌幅 > −35%
  保留條件：② 或 ③ 成立。
  跨年代合格：2000-2009、2010-2014、2015-2019、2020-2026 四段，每段年化 ≥ SPY 且 最大跌幅 > −35%。
  硬性：驗證期最大跌幅唔可以差過 −35%；唔用槓桿。
  最差5年年化（每月開始計，只計完整5年）：同線E 比較。
  每次買賣統計：每次賣出（止賺一半、全部賣出）都算一次；另列平均持有期。
  三個版本都會報告，唔會因為某個版本最好就揀佢做結論。

已知限制：
  - 止損位用收市價，唔用盤中低位（第四十九輪嘅已確認設定）。
  - 倖存者偏差（已除牌嘅股票可能冇數據）。
  - 呢個策略同時改咗入場、止損、止賺、加倉、出場。如果有效，唔可以單獨講邊個部分有效；如果無效，唔可以單獨講邊個部分失敗。

輸出：v49_results.csv、v49_progress.txt
==== 結果（待跑）====
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim, COST

M.PROGRESS = "v49_progress.txt"
log = M.log
END = "2026-09-30"
N = 5
RISK = 0.03
TP_MULT = 1.3
READD_DAYS = 40


def sim49(P, rets_v, cr, elig, ytd, E9, E21, LOW20):
    T, C = P.shape
    cash = 1.0
    pos = {}
    pend_exit, pend_tp, pend_readd, pend_buy = [], [], [], []
    navs, trades, holds = [], [], []
    for i in range(T):
        for c in pos:
            r = rets_v[i, c]
            if np.isfinite(r):
                pos[c]["val"] *= 1 + r
        cash *= 1 + cr[i]
        # 執行全部賣出（止損）
        for c in pend_exit:
            if c in pos:
                p = pos.pop(c)
                proceeds = p["val"] * (1 - COST)
                cash += proceeds
                trades.append(proceeds / p["cost"] - 1)
                holds.append(i - p["day"])
        pend_exit = []
        # 執行止賺一半
        for c in pend_tp:
            if c not in pos or pos[c]["tp_done"]:
                continue
            p = pos[c]
            sell_val = 0.5 * p["val"]
            proceeds = sell_val * (1 - COST)
            cost_part = 0.5 * p["cost"]
            cash += proceeds
            trades.append(proceeds / cost_part - 1)
            p["val"] -= sell_val
            p["cost"] -= cost_part
            p["tp_done"] = True
            p["tp_day"] = i
            p["readd_amt"] = proceeds
        pend_tp = []
        total = cash + sum(p["val"] for p in pos.values())
        # 執行買入（第一、二批）
        for c, tr, stop in pend_buy:
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
                          "day": i, "tp_done": False, "tp_day": None, "readd_done": False,
                          "readd_amt": 0.0, "dipped": False}
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
        pend_buy = []
        # 執行回調後加倉
        for c in pend_readd:
            if c in pos and pos[c]["tp_done"] and not pos[c]["readd_done"]:
                amt = min(pos[c]["readd_amt"], cash)
                if amt > 0:
                    cash -= amt
                    pos[c]["val"] += amt * (1 - COST)
                    pos[c]["cost"] += amt
                    pos[c]["readd_done"] = True
        pend_readd = []
        navs.append(cash + sum(p["val"] for p in pos.values()))
        # 今日收市信號
        for c in list(pos):
            p_ = P[i, c]
            if not np.isfinite(p_):
                continue
            pp = pos[c]
            if p_ < pp["stop"]:
                pend_exit.append(c)
                continue
            if not pp["tp_done"] and pp["val"] >= TP_MULT * pp["cost"]:
                pend_tp.append(c)
            if pp["tp_done"] and not pp["readd_done"] and i - pp["tp_day"] <= READD_DAYS:
                if p_ <= E9[i, c]:
                    pp["dipped"] = True
                elif pp["dipped"]:
                    pend_readd.append(c)
        # 第二批信號
        for c in pos:
            if c in pend_exit or pos[c]["tr"] != 1:
                continue
            if np.isfinite(P[i - 1, c]) and P[i - 1, c] <= E21[i - 1, c] and P[i, c] > E21[i, c]:
                pend_buy.append((c, 2, None))
        # 第一批信號
        free = N - len(pos)
        if free > 0 and i > 0:
            reclaim = elig[i] & (P[i - 1] <= E9[i - 1]) & (P[i] > E9[i]) & np.isfinite(P[i]) & np.isfinite(LOW20[i])
            idx = [c for c in np.where(reclaim)[0] if c not in pos]
            idx.sort(key=lambda c: -ytd[i, c])
            for c in idx[:free]:
                pend_buy.append((c, 1, LOW20[i, c]))
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1], trades, holds


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
    E9 = px.ewm(span=9, adjust=False).mean().values
    E21 = px.ewm(span=21, adjust=False).mean().values
    LOW20 = px.rolling(20, min_periods=20).min().values
    ye = px.groupby(days.year).last()
    ytdn = np.nan_to_num(P / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    hi52 = px.rolling(252, min_periods=200).max().values
    hi52f = np.nan_to_num(hi52, nan=np.inf)
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
    base_ok = mem & (ytdn >= 0.30) & fin
    e_ok = base_ok & (P >= hi52f * 0.7) & np.nan_to_num(P > ma200, nan=False)
    elig_a = base_ok & np.nan_to_num(P > ma200, nan=False)
    elig_b = base_ok

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    V = {}
    V["線E（對照）"] = sim(P, rets_v, cr, e_ok.copy(), ytdn, lambda i, c, p, pk: broken(i, c, p, pk))
    V["R49A 有200日線"] = sim49(P, rets_v, cr, elig_a, ytdn, E9, E21, LOW20)
    V["R49B 冇200日線"] = sim49(P, rets_v, cr, elig_b, ytdn, E9, E21, LOW20)
    for k in V:
        log(f"{k} 完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31"),
               ("2025年", "2025-01-01", "2025-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)

    L = ["=== 第四十九輪：修訂入場 + 3%風險定額 + 止賺一半 + 回調加倉 + 止損位出場（規則預先登記，見research_v49.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻", ""]
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
        hd = np.array(v[2]) if len(v) > 2 else None
        hold_s = f"平均持有 {hd.mean():.0f} 個交易日" if hd is not None and len(hd) else ""
        L.append(f"　{nm}：{len(tr)} 次賣出（每年約 {len(tr) / yrs:.0f} 次）｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
                 f"賺嘅平均 {tr[tr > 0].mean() * 100:+.1f}% ｜ 蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}% ｜ {hold_s}")
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
                 f"{'保留' if (q2 or q3) else '淘汰'} ｜ 硬性（驗證期跌幅 > -35%）{'✔' if x['mdd'] > -35 else '✘'} ｜ 跨年代 {marks} {'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 "
                 + " / ".join(f"{tab[(er, nm)]['cagr']:+.1f}%" for er in eras) + " ｜ 各段最大跌幅 "
                 + " / ".join(f"{tab[(er, nm)]['mdd']:.0f}%" for er in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
    pd.DataFrame({"report": L}).to_csv("v49_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
