"""
research_v15.py — 第十五輪：倍升股研究（Esther 嘅均線止損輪換 + 已升幾倍之後 + 倍升股DNA）

⚠️ 預先登記（2026-10-05 跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：

共同設定：當時標普500成分股（細價股歷史名單攞唔到，所以漏咗好多由細價股開始嘅10倍股）；已計股息；
  收市決定 → 下一個收市交易；每次買賣0.1%；唔用槓桿；現金收 ^IRX 利率。數據由1998年起，結果由2000年起計。

━━━ 第一部分：倍升股 + 均線止損輪換（Esther 2026-10-05 提出）━━━
  倍升股 = 過去252個交易日升 ≥ +100%（升咗一倍）。
  買入條件：係倍升股 + 收市價 > MA50 + 收市價 > MA200 + 當月係標普500成分股。
  賣出條件（兩個版本）：X50 = 收市跌穿 MA50；X200 = 收市跌穿 MA200。冇其他止賺/止蝕。中途冇價（除牌）= 用最後價賣出。
  有空位就買：喺所有符合買入條件、未持有嘅股票入面，揀過去12個月升幅最大嘅。賣咗嘅股票之後再符合條件可以再買。
  每個位 = 買入嗰日總資產 ÷ 位數；冇股可買就揸現金。
  版本：位數 1 / 3 / 5 × 賣出 X50 / X200 = 6個。
  揀規則期 2000-2014：用之前一樣嘅評分（40%回報+30%最大跌幅+20%Sharpe+10%換手，只比最大跌幅好過-35%）揀1個；
  驗證期 2015-2026 判斷。
  合格標準（全部要符合）：
    ① 驗證期年化回報 ≥ QQQ + 3個百分點
    ② 驗證期 Sharpe ≥ QQQ
    ③ 驗證期最大跌幅好過 -35%（Esther底線；同時列出「QQQ+5個百分點」作參考）
    ④ 三個年代（2000-2009 / 2010-2019 / 2020-2026）最少兩個年化回報贏QQQ

━━━ 第二部分：已經升咗幾倍之後先買，之後點？（純描述）━━━
  每個月尾（2000-01 至 2025-09），所有成分股按「過去12個月升幅」分組：
    跌咗 / 0-50% / 50-100% / 2-3倍(+100%至+200%) / 3-5倍(+200%至+400%) / 5倍以上(≥+400%)
  睇之後 6/12/24/36 個月：中位數回報、平均回報、跑贏SPY機率、跌≥30%機率。（每月重疊抽樣，只作描述）
  再將「升咗一倍以上」嘅股票分：仲喺MA50同MA200上面 vs 唔係；最近一季EPS按年增長 >25% vs 唔係。

━━━ 第三部分：倍升股DNA ━━━
  定義：某年1月第一個交易日之後5年內，最高收市價 ≥ 當日價 × 5 = 「5倍股」。
  當日特徵（全部只用當時已知資料）：
    T1 過去12個月升 ≥ +50%    T2 離52週高位 15%以內      T3 收市價 > MA200
    T4 60日波幅喺成分股最高30%  T5 最近一季EPS按年增長 >25%  T6 最近一次盈利好過預期
    T7 離52週高位跌咗 ≥40%     T8 過去12個月跌 ≥30%
  發現期（2000-2010年1月）：每個特徵計「有呢個特徵 → 變5倍股機率」係「冇呢個特徵」嘅幾多倍；
    倍數 ≥1.5 而且最少30個樣本有呢個特徵 → 入選。
  DNA分數 = 入選特徵符合咗幾多個。
  驗證期（2011-2025年1月）：每年1月揀DNA分數最高10隻（同分揀12個月升幅大嘅），揸一年，同QQQ比；
    合格標準同第一部分①②③（用2011-2026）+ 揀中5倍股嘅比例要明顯高過平均（≥2倍）。

輸出：v15_results.csv、v15_progress.txt
==== 結果（2026-10-05 第一次跑，記錄低，唔再改）====
- 第一部分：揀規則期揀咗「揸5隻・跌穿MA200賣」→ 驗證期 +21.0%（QQQ +19.1%）、Sharpe 0.72（0.82）、最大跌幅 -38.8% → 唔合格（只過④）。
  MA200 版本全部好過 MA50 版本（MA50 太易被震走，換手 >1200%/年）。
  「揸1隻・跌穿MA200賣」驗證期 +42.0%、2020-26 +44.8%，但 2000-09 每年 -12.5%、最大跌幅 -82.7%（全期 +14.6%）
  → 近十年極好、之前極差；係睇咗驗證期先發現，唔可以當合格。
- 第二部分：過去12個月升5倍以上（50個樣本）→ 之後1年中位數 -39%、56% 跌≥30%；3-5倍 → 2年中位數 -6%；
  2-3倍同普通股差唔多。仲喺均線上面 / EPS增長>25% 都冇明顯幫助。
- 第三部分：DNA入選特徵 = 波幅大 + 離高位跌≥40% + 過去12個月跌≥30%（即係「大跌後反彈」型）。
  驗證期揀中5倍股機率係平均3.5倍，但組合 +10.0%（QQQ +19.0%），最大跌幅 -50.3% → 唔合格。
"""
import bisect
import time

import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v10 import metrics, score_pick
from research_v13 import get_cash

M.PROGRESS = "v15_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"


def fetch_earn(tickers):
    import yfinance as yf
    out = {}
    for n, t in enumerate(tickers):
        for attempt in range(3):
            try:
                df = yf.Ticker(t).get_earnings_dates(limit=100)
                if df is not None and len(df):
                    out[t] = df
                break
            except Exception as e:
                if not any(w in str(e).lower() for w in ("rate", "too many", "timed out", "timeout", "connection", "429")):
                    break
                time.sleep(4 * (attempt + 1))
        time.sleep(0.4)
        if (n + 1) % 100 == 0:
            log(f"盈利紀錄 下載進度 {n + 1}/{len(tickers)}，有數據 {len(out)}")
    log(f"盈利紀錄 完成：{len(out)}/{len(tickers)}")
    return out


def known_index(days, ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    d = ts.normalize()
    after_close = ts.hour >= 16 or (ts.hour == 0 and ts.minute == 0)
    i = days.searchsorted(d, side="left")
    if i < len(days) and days[i] == d and after_close:
        i += 1
    return i if i < len(days) else None


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    month_ends = pd.date_range("1999-01-31", END, freq="ME")
    everyone = sorted(set().union(*[members_fn(d) for d in month_ends]))
    log(f"成分股來源：{source}；1999-2026 曾經入過標普500：{len(everyone)} 隻")
    closes = M.download_close(everyone + ["SPY", "QQQ"], "1998-01-01", "2026-10-01")
    spy = closes.pop("SPY")
    qqq = closes.pop("QQQ")
    spy = spy[spy.index <= pd.Timestamp(END)]
    px = pd.DataFrame(closes).reindex(spy.index)
    days = px.index
    cols = list(px.columns)
    P = px.values
    rets = px.pct_change(fill_method=None).values
    ma50 = px.rolling(50).mean().values
    ma200 = px.rolling(200).mean().values
    r12 = (px / px.shift(252) - 1).values
    hi52 = px.rolling(252, min_periods=200).max().values
    vol60 = (px.pct_change(fill_method=None).rolling(60).std() * np.sqrt(252)).values
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = qqq.reindex(days).pct_change(fill_method=None).fillna(0.0)
    cash_r = get_cash(days)
    cr = cash_r.values
    col_ix = {t: k for k, t in enumerate(cols)}

    mem = np.zeros(P.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [col_ix[t] for t in members_fn(d) if t in col_ix]
        mem[i, cache[key]] = True
    log("價錢同成分股準備好")

    # ================= 第一部分 =================
    entry_ok = mem & (r12 >= 1.0) & (P > ma50) & (P > ma200)
    entry_ok &= np.isfinite(P)

    def simulate(N, exit_ma):
        MA = ma50 if exit_ma == 50 else ma200
        hold = {}            # col -> value
        cash = 1.0
        pend_sell, pend_buy = [], []
        navs, turns = [], []
        last_px = {}
        for i in range(len(days)):
            # (1) 今日回報
            for c in list(hold):
                r = rets[i, c]
                if np.isfinite(r):
                    hold[c] *= 1 + r
            cash *= 1 + cr[i]
            traded = 0.0
            # (2) 執行昨日決定（今日收市價）
            for c in pend_sell:
                if c in hold:
                    v = hold.pop(c)
                    traded += v
                    cash += v * (1 - COST)
            total = cash + sum(hold.values())
            for c in pend_buy:
                if c in hold or len(hold) >= N:
                    continue
                if not np.isfinite(P[i, c]):
                    continue
                amt = min(total / N, cash)
                if amt <= total * 0.01:
                    break
                cash -= amt
                hold[c] = amt * (1 - COST)
                traded += amt
            pend_sell, pend_buy = [], []
            total = cash + sum(hold.values())
            navs.append(total)
            turns.append(traded / total if total > 0 else 0.0)
            # (3) 今日收市決定
            for c in hold:
                p = P[i, c]
                if not np.isfinite(p):
                    # 冇價：之後幾日都冇 → 當除牌，用最後價賣
                    if i + 1 < len(days) and not np.isfinite(P[i + 1, c]):
                        pend_sell.append(c)
                    continue
                if np.isfinite(MA[i, c]) and p < MA[i, c]:
                    pend_sell.append(c)
            free = N - (len(hold) - len(pend_sell))
            if free > 0:
                cand = np.where(entry_ok[i])[0]
                cand = [c for c in cand if c not in hold]
                if cand:
                    cand.sort(key=lambda c: -r12[i, c])
                    pend_buy = cand[:free]
        nav = pd.Series(navs, index=days)
        return nav.pct_change().fillna(0.0), pd.Series(turns, index=days)

    variants = {}
    for N in (1, 3, 5):
        for x in (50, 200):
            name = f"揸{N}隻・跌穿MA{x}賣"
            variants[name] = simulate(N, x)
            log(f"{name} 完成")

    L = ["=== 第十五輪：倍升股研究（規則預先登記，見research_v15.py開頭）===",
         f"股票池：當時標普500成分股（{source}）；已計股息；每次買賣0.1%；唔用槓桿；細價股唔包括", ""]

    def show(name, r, t, a, b):
        p = perf(r.loc[a:b], rf=cash_r, bench=qqq_r.loc[a:b], turn=t, cap_freq="ME")
        rec = f"{p['rec_days'] / 30.4:.0f}個月" if p["recovered"] else "未回本"
        return p, (f"　{name}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f} ｜ "
                   f"Calmar {p['calmar']:.2f} ｜ 回本 {rec} ｜ 每年換手 {p['turn_year']:.0f}% ｜ "
                   f"對QQQ 升市捕捉 {p['up_cap']:.0f}% / 跌市捕捉 {p['down_cap']:.0f}%")

    zero = pd.Series(0.0, index=days)
    windows = [("揀規則期", "2000-01-01", "2014-12-31"), ("驗證期", "2015-01-01", END),
               ("2000-2009", "2000-01-01", "2009-12-31"), ("2010-2019", "2010-01-01", "2019-12-31"),
               ("2020-2026", "2020-01-01", END), ("全期", "2000-01-01", END),
               ("金融海嘯", "2007-07-01", "2009-06-30"), ("科網爆煲", "2000-03-01", "2002-12-31")]
    tab = {}
    for wn, a, b in windows:
        L.append(f"=== 第一部分 {wn}（{a} 至 {b}）===")
        for nm, r in (("QQQ長揸", qqq_r), ("SPY長揸", spy_r)):
            tab[(wn, nm)], txt = show(nm, r, zero, a, b)
            L.append(txt)
        for nm, (r, t) in variants.items():
            tab[(wn, nm)], txt = show(nm, r, t, a, b)
            L.append(txt)
        L.append("")
    # 平均持有幾多隻 / 幾多日揸現金
    L.append("=== 第一部分 預先登記判斷 ===")
    cands = {k: metrics(r, t, "2000-01-01", "2014-12-31") for k, (r, t) in variants.items()}
    best, sc = score_pick(cands)
    L.append("　揀規則期評分：" + "，".join(f"{k} {v:.2f}" for k, v in sc.items()) + f" → 揀 {best}")
    v, q = tab[("驗證期", best)], tab[("驗證期", "QQQ長揸")]
    c1, c2, c3 = v["cagr"] >= q["cagr"] + 3, v["sharpe"] >= q["sharpe"], v["mdd"] > -35
    eras = [tab[(e, best)]["cagr"] > tab[(e, "QQQ長揸")]["cagr"] for e in ("2000-2009", "2010-2019", "2020-2026")]
    c4 = sum(eras) >= 2
    L.append(f"　① 年化 {v['cagr']:+.2f}% vs QQQ {q['cagr']:+.2f}%+3 → {'✔' if c1 else '✘'}")
    L.append(f"　② Sharpe {v['sharpe']:.2f} vs QQQ {q['sharpe']:.2f} → {'✔' if c2 else '✘'}")
    L.append(f"　③ 最大跌幅 {v['mdd']:.1f}%（要好過-35%；參考：QQQ {q['mdd']:.1f}%）→ {'✔' if c3 else '✘'}")
    L.append(f"　④ 三個年代贏QQQ：{sum(eras)}/3 → {'✔' if c4 else '✘'}")
    L.append(f"　⇒ {best} {'合格' if (c1 and c2 and c3 and c4) else '唔合格'}")
    L.append("")
    log("第一部分完成")

    # ================= 盈利數據（第二、三部分用）=================
    earn = fetch_earn(cols)
    eg_known = {}          # col -> (known idx list, eps growth list, surprise>0 list)
    for t, df in earn.items():
        df = df[~df.index.duplicated(keep="first")].sort_index()
        rep = df["Reported EPS"]
        est = df["EPS Estimate"]
        dates = [pd.Timestamp(x).tz_convert("America/New_York").tz_localize(None) if pd.Timestamp(x).tzinfo else pd.Timestamp(x)
                 for x in df.index]
        ks, gs, ss = [], [], []
        for j in range(len(df)):
            if not np.isfinite(rep.iloc[j]):
                continue
            k = known_index(days, df.index[j])
            if k is None:
                continue
            g = np.nan
            for jj in range(j - 1, -1, -1):        # 大約一年前嗰季（330-400日前）
                gap = (dates[j] - dates[jj]).days
                if gap > 400:
                    break
                if 330 <= gap <= 400 and np.isfinite(rep.iloc[jj]):
                    p0 = rep.iloc[jj]
                    if abs(p0) > 0.01:
                        g = (rep.iloc[j] - p0) / abs(p0)
                    break
            s = (rep.iloc[j] > est.iloc[j]) if np.isfinite(est.iloc[j]) else np.nan
            ks.append(k)
            gs.append(g)
            ss.append(s)
        if ks:
            o = np.argsort(ks, kind="stable")
            eg_known[col_ix[t]] = ([ks[x] for x in o], [gs[x] for x in o], [ss[x] for x in o])

    def eps_at(c, i):
        if c not in eg_known:
            return np.nan, np.nan
        ks, gs, ss = eg_known[c]
        p = bisect.bisect_right(ks, i) - 1
        if p < 0 or i - ks[p] > 126:
            return np.nan, np.nan
        return gs[p], ss[p]

    # ================= 第二部分 =================
    L.append("=== 第二部分：已經升咗幾倍之後先買，之後點？（每月尾抽樣，2000-01 至 2025-09）===")
    me = [days[days.searchsorted(d, side="right") - 1] for d in pd.date_range("2000-01-31", "2025-09-30", freq="ME")]
    spy_v = spy.reindex(days).values
    rows = []
    for d in me:
        i = days.get_loc(d)
        cs = np.where(mem[i] & np.isfinite(P[i]) & np.isfinite(r12[i]))[0]
        for c in cs:
            row = {"i": i, "c": c, "r12": r12[i, c], "trend": bool(P[i, c] > ma50[i, c] and P[i, c] > ma200[i, c])}
            for h in (126, 252, 504, 756):
                e = i + h
                if e < len(days):
                    seg = P[i:e + 1, c]
                    ok = seg[np.isfinite(seg)]
                    if len(ok):
                        row[h] = ok[-1] / P[i, c] - 1
                        row[f"s{h}"] = spy_v[e] / spy_v[i] - 1
            if r12[i, c] >= 1.0:
                row["eg"] = eps_at(c, i)[0]
            rows.append(row)
    D2 = pd.DataFrame(rows)
    bins = [-np.inf, 0, 0.5, 1.0, 2.0, 4.0, np.inf]
    labs = ["跌咗", "升0-50%", "升50-100%", "2-3倍", "3-5倍", "5倍以上"]
    D2["b"] = pd.cut(D2["r12"], bins, labels=labs, right=False)

    def describe(g, tag):
        parts = []
        for h, lab in ((126, "6個月"), (252, "1年"), (504, "2年"), (756, "3年")):
            if h not in g:
                continue
            x = g[[h, f"s{h}"]].dropna()
            if len(x) < 20:
                continue
            parts.append(f"{lab} 中位數 {x[h].median() * 100:+.0f}% 平均 {x[h].mean() * 100:+.0f}% "
                         f"贏SPY {(x[h] > x[f's{h}']).mean() * 100:.0f}% 跌≥30% {(x[h] <= -0.3).mean() * 100:.0f}%")
        L.append(f"　{tag}（{len(g)}）：" + " ｜ ".join(parts))

    for b in labs:
        describe(D2[D2["b"] == b], b)
    up = D2[D2["r12"] >= 1.0]
    L.append("　—— 升咗一倍以上嘅股票再細分 ——")
    describe(up[up["trend"]], "仲喺MA50同MA200上面")
    describe(up[~up["trend"]], "已經跌穿MA50或MA200")
    describe(up[up["eg"] > 0.25], "最近一季EPS按年增長>25%")
    describe(up[(up["eg"] <= 0.25)], "EPS增長≤25%")
    describe(up[up["eg"].isna()], "EPS增長唔知")
    describe(up[up["trend"] & (up["eg"] > 0.25)], "喺均線上面 + EPS增長>25%")
    L.append("")
    log("第二部分完成")

    # ================= 第三部分 =================
    L.append("=== 第三部分：倍升股DNA ===")
    rows = []
    for y in range(2000, 2026):
        i = days.searchsorted(pd.Timestamp(f"{y}-01-01"))
        if i - 252 < 0:
            continue
        e = min(i + 1260, len(days) - 1)
        cs = np.where(mem[i] & np.isfinite(P[i]) & np.isfinite(r12[i]))[0]
        if not len(cs):
            continue
        v = vol60[i, cs]
        vcut = np.nanpercentile(v, 70)
        for c in cs:
            fut = P[i + 1:e + 1, c]
            fut = fut[np.isfinite(fut)]
            g, s = eps_at(c, i)
            off = P[i, c] / hi52[i, c] - 1 if np.isfinite(hi52[i, c]) else np.nan
            rows.append({"year": y, "i": i, "c": c, "r12": r12[i, c],
                         "x5": (len(fut) > 0 and fut.max() / P[i, c] >= 5) if y <= 2021 else np.nan,
                         "T1": r12[i, c] >= 0.5, "T2": off >= -0.15, "T3": P[i, c] > ma200[i, c],
                         "T4": vol60[i, c] >= vcut, "T5": g > 0.25, "T6": s is True or s == 1,
                         "T7": off <= -0.40, "T8": r12[i, c] <= -0.30})
    D3 = pd.DataFrame(rows)
    traits = {"T1": "過去12個月升≥50%", "T2": "離52週高位15%以內", "T3": "收市價>MA200", "T4": "波幅最高30%",
              "T5": "EPS按年增長>25%", "T6": "最近盈利好過預期", "T7": "離52週高位跌≥40%", "T8": "過去12個月跌≥30%"}
    disc = D3[(D3["year"] <= 2010)].copy()
    disc["x5"] = disc["x5"].astype(bool)
    base = disc["x5"].mean()
    L.append(f"　發現期 2000-2010：{len(disc)} 個樣本，5年內變5倍股 {int(disc['x5'].sum())} 個（{base * 100:.1f}%）")
    chosen = []
    for k, lab in traits.items():
        has = disc[disc[k] == True]
        no = disc[disc[k] != True]
        p1 = has["x5"].mean() if len(has) else np.nan
        p0 = no["x5"].mean() if len(no) else np.nan
        lift = p1 / p0 if p0 and p0 > 0 else np.nan
        pick = len(has) >= 30 and np.isfinite(lift) and lift >= 1.5
        if pick:
            chosen.append(k)
        L.append(f"　　{k} {lab}：有 {len(has)} 個，變5倍 {p1 * 100:.1f}% vs 冇 {p0 * 100:.1f}% → {lift:.1f}倍 {'✔入選' if pick else ''}")
    L.append(f"　入選特徵：{', '.join(traits[k] for k in chosen) if chosen else '冇'}")
    val = D3[D3["year"] >= 2011].copy()
    val["score"] = val[chosen].fillna(False).astype(int).sum(axis=1) if chosen else 0
    parts, picked = [], []
    for y in range(2011, 2026):
        g = val[val["year"] == y]
        if g.empty:
            continue
        top = g.sort_values(["score", "r12"], ascending=[False, False]).head(10)
        picked.append(top)
        i = int(top["i"].iloc[0])
        e = min(i + 252, len(days) - 1)
        seg = px.iloc[i:e + 1, list(top["c"])].ffill()
        nav = (seg / seg.iloc[0]).mean(axis=1)
        r = nav.pct_change().iloc[1:]
        r.iloc[0] -= COST
        parts.append(r)
    if parts:
        rr = pd.concat(parts).sort_index()
        rr = rr[~rr.index.duplicated()]
        pk = pd.concat(picked)
        vv = pk[pk["year"] <= 2021]
        allv = val[val["year"] <= 2021]
        hit = vv["x5"].astype(float).mean()
        basev = allv["x5"].astype(float).mean()
        L.append(f"　驗證期 2011-2021 揀中5倍股比例：{hit * 100:.1f}%（平均 {basev * 100:.1f}%，"
                 f"{hit / basev if basev > 0 else np.nan:.1f}倍）")
        p, txt = show("DNA前10隻（每年1月換）", rr, zero, "2011-01-01", END)
        q, qtxt = show("QQQ長揸", qqq_r, zero, "2011-01-01", END)
        L.append(qtxt)
        L.append(txt)
        c1, c2, c3 = p["cagr"] >= q["cagr"] + 3, p["sharpe"] >= q["sharpe"], p["mdd"] > -35
        c5 = basev > 0 and hit >= 2 * basev
        L.append(f"　① 年化 ≥ QQQ+3 {'✔' if c1 else '✘'} ② Sharpe ≥ QQQ {'✔' if c2 else '✘'} ③ 最大跌幅好過-35% {'✔' if c3 else '✘'} "
                 f"④ 揀中5倍股比例 ≥ 平均2倍 {'✔' if c5 else '✘'}")
        L.append(f"　⇒ DNA策略 {'合格' if (c1 and c2 and c3 and c5) else '唔合格'}")
    pd.DataFrame({"report": L}).to_csv("v15_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
