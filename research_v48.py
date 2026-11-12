"""
research_v48.py — 第四十八輪：分批入市 + 按離高位遠近縮倉（修訂版入場方向）
  （Esther 2026-11-11 截圖 BE/MU/SNDK/SPY 後提出：「為買而買，唔考慮股價位置」；
    2026-11-12：「好，那你修訂方向，我地試下做回測」）

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：

共同設定（同第四十七輪一樣）：
  股票池：當時真實標普500（M.sp500_history）；已計股息（Yahoo 調整後收市）；2000-01-03 至 2026-09-30。
  收市決定 → 下一個交易日收市成交；每次買賣 0.1%；最多 5 個倉位；閒錢收 ^IRX。
  行業上限同大市濾網：唔用（呢輪只測入場、分批、縮倉、出場）。

對照：線E（同第四十七輪一樣，research_v26.sim）。

修訂版入場（R48A 同 R48B 共用）：
  合資格（每日）：今年 ≥ +30%。
  （R48A 另加：收市 > 200日平均；R48B 唔要求。）
  第一批（半倉）：合資格 且 昨日收市 ≤ EMA9 且 今日收市 > EMA9（重新站上 9 日指數平均）。
     同日有多隻符合，按今年升幅由大到細揀，最多填滿空位。
  第二批（再半倉）：已持有第一批，而且 昨日收市 ≤ EMA21 且 今日收市 > EMA21（重新站上 21 日指數平均）。
     第二批唔一定會出現；冇就只持有半倉。
  每批金額 = 0.5 × 縮倉係數 × （總資產 ÷ 5）。
  縮倉係數（按執行當日）：離 52 週最高收市 ≤ 25% → 1.0；25% 至 50% → 0.5；> 50% 或者冇數據 → 0.25。
  同一隻股票最多兩批；賣出之後可以再次觸發第一批。

修訂版出場（R48A 同 R48B 共用，**取代線E嘅出場**）：
  由買入後最高收市（含買入日）跌穿 70% 就賣（即係由買入後高位跌 30%）。
  原因：唔可以再用「跌穿200日線」或「離52週高位 30%」，因為第一批可能喺離高位 40% 以上嘅位置買入，會即時觸發賣出。
  ⚠️ 呢個係出場改動，所以 R48 係一套新策略，唔純係入場改動。如果合格，將來要分開試每個部分先可以知道邊部分有效。

版本（只有以下，跑完唔加唔減、唔改數字）：
  線E（對照）
  R48A 修訂版，有 200日線 入場條件
  R48B 修訂版，冇 200日線 入場條件（測試早期反彈）

判斷（同第四十七輪一樣；同線E比；驗證期 2015-01 至 2026-09）：
  「有改善」：(a) 最大跌幅 ≥ 線E + 3 點 且 年化 ≥ 線E − 1 點；或 (b) 年化 ≥ 線E + 2 點 且 最大跌幅 ≥ 線E − 2 點。
  四條問題（同第四十三及四十七輪一樣）：
    ① 驗證期年化 > QQQ
    ② 若 ① 成立：驗證期最大跌幅 > −35% 且 Sharpe ≥ QQQ 嘅 Sharpe
    ③ 若 ① 唔成立：最大跌幅 ≥ QQQ + 5 點 且 年化 ≥ SPY
    ④ 2000-2009 最大跌幅 > −35%
  保留條件：② 或 ③ 成立。
  跨年代合格：2000-2009、2010-2014、2015-2019、2020-2026 四段，每段年化 ≥ SPY 且 最大跌幅 > −35%。
  硬性：驗證期最大跌幅唔可以差過 −35%；唔用槓桿。
  最差5年年化（每月開始計，只計完整5年）：同線E 比較。
  另外列：每次買賣統計（一次完整賣出 = 一次買賣）、平均持有期（交易日）、各段（科網爆煲、金融海嘯、2022熊市、2025年）數字。
  三個版本都會報告，唔會因為某個版本最好就揀佢做結論。

已知限制：
  - 行業分類唔用，所以唔受第四十七輪嘅行業限制問題影響。
  - 倖存者偏差（已除牌嘅股票可能冇數據）。
  - 「重新站上 EMA」係一個簡單定義，可能會因為震盪市而多次觸發。

輸出：v48_results.csv、v48_progress.txt
==== 結果（2026-11-12 第一次跑，run 43132549799，記錄低，唔再改）====
- 線E 對照 +28.74% / -32.6%（核對通過）。
- 驗證期：R48A 有200日線 +7.81% / -34.4%；R48B 冇200日線 +8.81% / -30.9% → 全部冇改善，四條問題全部淘汰。
  R48B 最大跌幅細過線E（-30.9% vs -32.6%），但年化少咗 20 點，唔符合「有改善」標準。
- 跨年代：R48A ✘✔✔✘；R48B ✘✘✘✘ → 全部唔合格。
- 每次買賣：線E 279 次（賺錢 41%）；R48A 87 次（賺錢 36%，平均持有 351 個交易日）；R48B 96 次（賺錢 33%，平均持有 322 個交易日）。
  賺錢嗰批平均 +125%，但蝕錢嗰批平均 -19% 至 -21%，而且次數少、資金大部分時間唔係在用。
- 結論：修訂版（早期分批入市 + 縮倉 + 買入後高位跌30%出場）冇改善線E。
  ⚠️ 入場同出場兩樣都改咗，所以唔可以單獨講邊個部分失敗。30% 追蹤出場太寬，令交易次數大減、持有時間好長。

"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import sim, COST

M.PROGRESS = "v48_progress.txt"
log = M.log
END = "2026-09-30"
N = 5
EXIT_TRAIL = 0.7


def scale(dd):
    """dd = 1 − 收市 ÷ 52週最高收市。"""
    if not np.isfinite(dd) or dd > 0.5:
        return 0.25
    if dd > 0.25:
        return 0.5
    return 1.0


def sim48(P, rets_v, cr, elig, ytd, E9, E21, DD):
    """分批入市 + 縮倉 + 買入後高位跌30%出場。信號 = 收市，執行 = 下一日收市。"""
    T, C = P.shape
    cash = 1.0
    pos = {}
    pend_exit, pend_buy = [], []
    navs, trades, holds = [], [], []
    for i in range(T):
        for c in pos:
            r = rets_v[i, c]
            if np.isfinite(r):
                pos[c]["val"] *= 1 + r
        cash *= 1 + cr[i]
        # 執行昨日嘅賣出
        for c in pend_exit:
            if c in pos:
                p = pos.pop(c)
                cash += p["val"] * (1 - COST)
                trades.append(p["val"] * (1 - COST) / p["cost"] - 1)
                holds.append(i - p["day"])
        pend_exit = []
        total = cash + sum(p["val"] for p in pos.values())
        # 執行昨日嘅買入
        for c, tr in pend_buy:
            if not np.isfinite(P[i, c]):
                continue
            amt_base = 0.5 * scale(DD[i, c]) * total / N
            if tr == 1:
                if c in pos or len(pos) >= N:
                    continue
                amt = min(amt_base, cash)
                if amt <= 0:
                    continue
                cash -= amt
                pos[c] = {"val": amt * (1 - COST), "cost": amt, "tr": 1, "peak": P[i, c], "day": i}
            else:
                if c not in pos or pos[c]["tr"] != 1:
                    continue
                amt = min(amt_base, cash)
                if amt <= 0:
                    continue
                cash -= amt
                pos[c]["val"] += amt * (1 - COST)
                pos[c]["cost"] += amt
                pos[c]["tr"] = 2
        pend_buy = []
        navs.append(cash + sum(p["val"] for p in pos.values()))
        # 今日收市信號
        for c in list(pos):
            p = P[i, c]
            if not np.isfinite(p):
                continue
            pos[c]["peak"] = max(pos[c]["peak"], p)
            if p < EXIT_TRAIL * pos[c]["peak"]:
                pend_exit.append(c)
        # 第二批信號
        for c in list(pos):
            if c in pend_exit or pos[c]["tr"] != 1:
                continue
            if np.isfinite(P[i - 1, c]) and P[i - 1, c] <= E21[i - 1, c] and P[i, c] > E21[i, c]:
                pend_buy.append((c, 2))
        # 第一批信號
        free = N - len(pos)
        if free > 0 and i > 0:
            reclaim = elig[i] & (P[i - 1] <= E9[i - 1]) & (P[i] > E9[i]) & np.isfinite(P[i])
            idx = [c for c in np.where(reclaim)[0] if c not in pos]
            idx.sort(key=lambda c: -ytd[i, c])
            for c in idx[:free]:
                pend_buy.append((c, 1))
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
    ye = px.groupby(days.year).last()
    ytdn = np.nan_to_num(P / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    hi52 = px.rolling(252, min_periods=200).max().values
    hi52f = np.nan_to_num(hi52, nan=np.inf)
    DD = 1 - P / hi52
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
    V["R48A 有200日線"] = sim48(P, rets_v, cr, elig_a, ytdn, E9, E21, DD)
    V["R48B 冇200日線"] = sim48(P, rets_v, cr, elig_b, ytdn, E9, E21, DD)
    for k in V:
        log(f"{k} 完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31"),
               ("2025年", "2025-01-01", "2025-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)

    L = ["=== 第四十八輪：分批入市 + 按離高位遠近縮倉（規則預先登記，見research_v48.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5個倉位", ""]
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
        L.append(f"　{nm}：{len(tr)} 次（每年約 {len(tr) / yrs:.0f} 次）｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
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
    pd.DataFrame({"report": L}).to_csv("v48_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
