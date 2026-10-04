"""
research_v5.py — 下一輪：A 波動控制、B 風控動量、C 剔除大市/板塊嘅動量、D 先揀板塊再揀股

⚠️ 預先登記（跑之前定好，跑完唔改）。用戶規矩：本金唔用槓桿，所以「揀方案」只喺唔用槓桿嘅版本入面揀；
   槓桿版本照計，只作參考。

共同設定
- 數據同股票池同 momentum_v1.py 一樣：每月用當時嘅標普500成分股（fja05680/sp500），yfinance一隻隻下載（已計股息）。
- 收市判斷，下一個交易日收市交易。每次買賣成本0.1%。現金收益13週國債息。
- 板塊：用Wikipedia今日嘅GICS板塊（⚠️ 已剔出指數嘅公司冇板塊資料；用今日分類套返過去，有少少偏差）。
- 冇免費嘅歷史基本面數據，所以唔做「公司質素」篩選。

A. 你而家嘅SPY訊號方案（main.py規則）× 按波動調整倉位
   - 持倉比例 = 方案狀態（持有/現金）× min(上限, 目標波動 ÷ SPY過去N日實際波動)
   - 目標波動：12%、15%、18%、20%；N = 20、60日；上限：1.0（唔用槓桿）、1.25、1.5（槓桿，只作參考）
   - 新比例同現時相差超過10%（或者方案轉狀態）先交易。借錢部分每年成本 = 國債息 + 1.5%（IB大約）。
   - 測試期1994-2026/09，前半1994-2012，後半2013-2026/09（同之前SPY研究一樣）。
   - 揀法：上限1.0嘅8個入面，前半最大跌幅好過-35%而回報最高，睇後半。
B. 風控動量（每月換倉，12-1動量前10隻，冇200日線濾網）
   - 每隻股票資金 ∝ 1 ÷ 過去60日波動，單一股票最多15%
   - 整體目標波動：冇（只做按波動分配）、12%、15%、18%（用過去60日估計，上限1.0，唔用槓桿，多出嘅放現金）
C. 剔除影響之後嘅動量（每月，前10隻 / 前20隻，平均分配）
   - 原始12-1動量（對照）
   - 剔除大市：過去252至21日，股票每日回報對SPY做回歸，用「剩餘回報平均 ÷ 剩餘回報波動」排名
   - 剔除板塊：同上，但對所屬板塊ETF（冇板塊資料或者ETF未上市就用SPY）
   - 剔除大市 + 板塊：同時對SPY同板塊ETF做回歸
D. 先揀板塊再揀股（每月）
   - 板塊ETF（XLB XLE XLF XLI XLK XLP XLU XLV XLY；XLRE、XLC有夠12個月數據先加入）按12-1動量排名
   - 揀最強3個板塊，每個板塊揀12-1動量最強2隻（或3隻）股票，平均分配
B/C/D 測試期2005-2026/09，前半2005-2015，後半2016-2026/09；另列2020-2026（數據覆蓋率約九成）。
B/C/D 揀法：全部14個組合入面，前半最大跌幅好過-35%而回報最高，原封不動睇後半，同SPY比。
輸出：v5_results.csv、v5_progress.txt
"""
import io
import time

import numpy as np
import pandas as pd
import requests

import momentum_v1 as M

M.PROGRESS = "v5_progress.txt"
log = M.log
COST = 0.001
LEV_SPREAD = 0.015
SECTOR_ETF = {"Information Technology": "XLK", "Health Care": "XLV", "Financials": "XLF",
              "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
              "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU",
              "Real Estate": "XLRE", "Communication Services": "XLC"}
P_SPY = [("全期", "1994-01-01", "2026-09-30"), ("前半", "1994-01-01", "2012-12-31"), ("後半", "2013-01-01", "2026-09-30")]
P_STK = [("全期", "2005-01-01", "2026-09-30"), ("前半", "2005-01-01", "2015-12-31"),
         ("後半", "2016-01-01", "2026-09-30"), ("2020後", "2020-01-01", "2026-09-30")]


def stats(eq, a, b):
    return M.stats(eq, a, b)


def fmt_row(name, eq, periods):
    parts = []
    for pn, a, b in periods:
        c, d, _ = stats(eq, a, b)
        parts.append(f"{pn} {c:+.2f}%（跌 {d:.1f}%）")
    return f"　{name}：" + " | ".join(parts)


# ---------------- A ----------------
def test_a():
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    spy = df["adj"]
    cash_r = M.get_cash(spy.index)
    r = spy.pct_change().fillna(0.0)
    state = (df["state"] != "out").astype(float)
    out, curves = [], {}

    def run(weights):
        w = weights.shift(2).fillna(0.0)
        borrow = (w - 1).clip(lower=0)
        daily = w * r + (1 - w).clip(lower=0) * cash_r - borrow * (cash_r + LEV_SPREAD / 252) \
            - w.diff().abs().fillna(0.0) * COST
        return (1 + daily).cumprod()
    curves["SPY長揸"] = spy / spy.iloc[0]
    curves["SPY訊號方案（而家）"] = run(state)
    for n in (20, 60):
        vol = r.rolling(n).std() * np.sqrt(252)
        for tgt in (0.12, 0.15, 0.18, 0.20):
            for cap in (1.0, 1.25, 1.5):
                want = (state * np.minimum(cap, tgt / vol)).fillna(0.0)
                cur, ws = 0.0, []
                for x, s0, s1 in zip(want.values, state.shift(1).fillna(0).values, state.values):
                    if s0 != s1 or abs(x - cur) > 0.10:
                        cur = x
                    ws.append(cur)
                curves[f"波動{n}日 目標{tgt:.0%} 上限{cap}倍"] = run(pd.Series(ws, index=spy.index))
    L = ["=== A. SPY訊號方案 × 按波動調整倉位（1994-2026）==="]
    rows = []
    for k, eq in curves.items():
        L.append(fmt_row(k, eq, P_SPY) + f" | $80,000 → ${80000 * stats(eq, *P_SPY[0][1:])[2]:,.0f}")
        if "上限1.0倍" in k:
            rows.append((k, {pn: stats(eq, a, b) for pn, a, b in P_SPY}))
    ok = [x for x in rows if x[1]["前半"][1] > -35]
    if ok:
        p = max(ok, key=lambda x: x[1]["前半"][0])
        base = {pn: stats(curves["SPY訊號方案（而家）"], a, b) for pn, a, b in P_SPY}
        L.append(f"　揀法（唔用槓桿）→ 前半揀中：{p[0]}（前半 {p[1]['前半'][0]:+.2f}%）→ 後半 {p[1]['後半'][0]:+.2f}%"
                 f"（跌 {p[1]['後半'][1]:.1f}%）；而家方案後半 {base['後半'][0]:+.2f}%（跌 {base['後半'][1]:.1f}%）")
    return L


# ---------------- 股票部分 ----------------
def sectors_now():
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=M.HDR, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    return {M.norm(s): g for s, g in zip(t["Symbol"], t["GICS Sector"])}


def simulate_w(px, targets, cash_r):
    """targets：{決定日: {股票: 比重}}（比重加埋 ≤ 1，其餘現金），下一個交易日收市交易；月中隨價格浮動。"""
    days = px.index
    rets = px.pct_change()
    trade_on = {}
    for d, w in targets.items():
        i = days.searchsorted(d, side="right")
        if i < len(days):
            trade_on[days[i]] = w
    hold, cash, vals = {}, 1.0, []
    for k, day in enumerate(days):
        if k > 0:
            rr = rets.iloc[k]
            for t in list(hold):
                x = rr.get(t, np.nan)
                if np.isfinite(x):
                    hold[t] *= 1 + x
            cash *= 1 + cash_r.iloc[k]
        if day in trade_on:
            w = trade_on[day]
            total = cash + sum(hold.values())
            new = {t: total * x for t, x in w.items()}
            turnover = sum(abs(new.get(t, 0) - hold.get(t, 0)) for t in set(new) | set(hold))
            total -= turnover * COST
            hold = {t: total * x for t, x in w.items()}
            cash = total - sum(hold.values())
        vals.append(cash + sum(hold.values()))
    return pd.Series(vals, index=days)


def cap_weights(w, cap):
    w = w / w.sum()
    for _ in range(20):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        w[under] += excess * w[under] / w[under].sum()
    return w


def resid_score(Y, X):
    """Y：日數×股票，X：日數×因子（含常數）。回傳每隻股票 剩餘回報平均 ÷ 剩餘回報波動。"""
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    sd = res.std(axis=0)
    return np.where(sd > 0, res.mean(axis=0) / sd, np.nan)


def test_bcd():
    members_fn, source = M.sp500_history()
    month_ends = pd.date_range("2004-12-31", "2026-09-30", freq="ME")
    members = {d: members_fn(d) for d in month_ends}
    everyone = sorted(set().union(*members.values()))
    etfs = sorted(set(SECTOR_ETF.values()))
    closes = M.download_close(everyone + ["SPY"] + etfs, "2003-06-01", "2026-10-01")
    spy = closes.pop("SPY")
    etf_px = pd.DataFrame({e: closes.pop(e) for e in etfs if e in closes}).reindex(spy.index)
    px = pd.DataFrame(closes).reindex(spy.index)
    px, spy, etf_px = (x[x.index <= pd.Timestamp("2026-09-30")] for x in (px, spy, etf_px))
    cash_r = M.get_cash(px.index)
    sec = sectors_now()
    log(f"板塊資料：{sum(1 for t in px.columns if t in sec)}/{len(px.columns)}隻有")
    days = px.index
    R = px.pct_change()
    Rm = spy.pct_change()
    Re = etf_px.pct_change()
    mom = px.shift(M.SKIP) / px.shift(M.LOOK) - 1
    emom = etf_px.shift(M.SKIP) / etf_px.shift(M.LOOK) - 1
    vol60 = R.rolling(60).std() * np.sqrt(252)

    names_b = ["B 按波動分配（冇目標）", "B 按波動分配 + 目標12%", "B 按波動分配 + 目標15%", "B 按波動分配 + 目標18%"]
    names_c = [f"C {k} 前{n}隻" for k in ("原始動量", "剔除大市", "剔除板塊", "剔除大市+板塊") for n in (10, 20)]
    names_d = ["D 3個板塊×2隻", "D 3個板塊×3隻"]
    targets = {k: {} for k in names_b + names_c + names_d}

    for d in month_ends:
        i = days.searchsorted(d, side="right") - 1
        if i < M.LOOK:
            continue
        day = days[i]
        alive = [t for t in members[d] if t in px.columns and np.isfinite(px[t].iloc[i])]
        raw = mom.loc[day, alive].dropna()
        # ---- B ----
        top = list(raw.sort_values(ascending=False).index[:10])
        v = vol60.loc[day, top].replace(0, np.nan).dropna()
        if len(v):
            w = cap_weights(1 / v, 0.15)
            hist = R.iloc[i - 59:i + 1][list(w.index)].fillna(0.0)
            pvol = float((hist @ w.values).std() * np.sqrt(252))
            targets[names_b[0]][day] = w.to_dict()
            for nm, tg in zip(names_b[1:], (0.12, 0.15, 0.18)):
                sc = min(1.0, tg / pvol) if pvol > 0 else 1.0
                targets[nm][day] = (w * sc).to_dict()
        # ---- C ----
        win = slice(i - M.LOOK + 1, i - M.SKIP + 1)
        Y = R.iloc[win][alive]
        good = [t for t in alive if Y[t].notna().all()]
        Y = Y[good].values
        m = Rm.iloc[win].values
        scores = {"原始動量": raw}
        if len(good) and np.isfinite(m).all():
            ones = np.ones_like(m)
            scores["剔除大市"] = pd.Series(resid_score(Y, np.column_stack([ones, m])), index=good)
            s_sec, s_both = {}, {}
            for etf in set(SECTOR_ETF.values()) | {None}:
                grp = [k for k, t in enumerate(good) if SECTOR_ETF.get(sec.get(t)) == etf] if etf else \
                      [k for k, t in enumerate(good) if SECTOR_ETF.get(sec.get(t)) is None]
                if not grp:
                    continue
                e = Re[etf].iloc[win].values if etf in Re.columns else None
                if e is None or not np.isfinite(e).all():
                    rs = resid_score(Y[:, grp], np.column_stack([ones, m]))
                    rb = rs
                else:
                    rs = resid_score(Y[:, grp], np.column_stack([ones, e]))
                    rb = resid_score(Y[:, grp], np.column_stack([ones, m, e]))
                for kk, k in enumerate(grp):
                    s_sec[good[k]] = rs[kk]
                    s_both[good[k]] = rb[kk]
            scores["剔除板塊"] = pd.Series(s_sec)
            scores["剔除大市+板塊"] = pd.Series(s_both)
        for k, s in scores.items():
            s = s.dropna().sort_values(ascending=False)
            for n in (10, 20):
                pick = list(s.index[:n])
                if pick:
                    targets[f"C {k} 前{n}隻"][day] = {t: 1 / len(pick) for t in pick}
        # ---- D ----
        em = emom.loc[day].dropna()
        top_s = list(em.sort_values(ascending=False).index[:3])
        for nm, per in zip(names_d, (2, 3)):
            pick = []
            for etf in top_s:
                cand = [t for t in raw.index if SECTOR_ETF.get(sec.get(t)) == etf]
                pick += list(raw[cand].sort_values(ascending=False).index[:per])
            if pick:
                targets[nm][day] = {t: 1 / len(pick) for t in pick}
    log("排名完成，開始模擬")

    curves = {"SPY長揸": spy / spy.iloc[0]}
    try:
        curves["SPY訊號方案（而家）"] = M.spy_bot_curve(spy, cash_r)
    except Exception as e:
        log(f"SPY訊號方案計唔到：{e}")
    for k in names_b + names_c + names_d:
        curves[k] = simulate_w(px, targets[k], cash_r)
        log(f"完成 {k}")

    L = ["", "=== B/C/D 股票策略（2005-2026；前半2005-2015，後半2016-2026，另列2020後）===",
         f"成分股來源：{source}；有價格 {px.shape[1]} 隻"]
    rows = []
    for k, eq in curves.items():
        L.append(fmt_row(k, eq, P_STK) + f" | $10,000 → ${10000 * stats(eq, *P_STK[0][1:])[2]:,.0f}")
        if k[:2] in ("B ", "C ", "D "):
            rows.append((k, {pn: stats(eq, a, b) for pn, a, b in P_STK}))
    spy_s = {pn: stats(curves["SPY長揸"], a, b) for pn, a, b in P_STK}
    ok = [x for x in rows if x[1]["前半"][1] > -35]
    L.append("")
    L.append("=== 誠實測試（B/C/D共14個，前半跌幅好過-35%入面回報最高，原封不動睇後半）===")
    if ok:
        p = max(ok, key=lambda x: x[1]["前半"][0])
        rank = sum(1 for x in rows if x[1]["後半"][0] > p[1]["後半"][0]) + 1
        L.append(f"　前半揀中：{p[0]}（前半 {p[1]['前半'][0]:+.2f}%，跌 {p[1]['前半'][1]:.1f}%）")
        L.append(f"　→ 後半 {p[1]['後半'][0]:+.2f}%（跌 {p[1]['後半'][1]:.1f}%），排第{rank}/{len(rows)}；"
                 f"SPY後半 {spy_s['後半'][0]:+.2f}%（跌 {spy_s['後半'][1]:.1f}%）")
        L.append(f"　結論：{'✅ 後半段贏SPY' if p[1]['後半'][0] > spy_s['後半'][0] else '❌ 後半段輸SPY'}")
    both = [x for x in rows if x[1]["前半"][0] > spy_s["前半"][0] and x[1]["後半"][0] > spy_s["後半"][0]
            and x[1]["全期"][1] > -35]
    L.append(f"　前後兩段都贏SPY、而且全期跌幅好過-35%：{len(both)}個 " + "、".join(x[0] for x in both))
    L.append("")
    L.append("=== 逐年回報（SPY vs 各策略）===")
    ys = {k: M.yearly(v) for k, v in curves.items()}
    keys = list(curves)
    L.append("　年份 | " + " | ".join(keys))
    for y in ys["SPY長揸"].index:
        L.append(f"　{y} | " + " | ".join(f"{ys[k].get(y, np.nan) * 100:+.1f}%" for k in keys))
    return L


def main():
    open(M.PROGRESS, "w").close()
    L = ["=== research_v5（規則預先登記，見research_v5.py開頭）==="]
    try:
        L += test_a()
    except Exception:
        import traceback
        L += ["A 出錯：", *traceback.format_exc().splitlines()]
    pd.DataFrame({"report": L}).to_csv("v5_results.csv", index=False, encoding="utf-8-sig")
    log("A 完成")
    L += test_bcd()
    pd.DataFrame({"report": L}).to_csv("v5_results.csv", index=False, encoding="utf-8-sig")
    log("全部完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
