"""
research_v12.py — 第十二輪：驗證 alpha（唔再搵新條件）

⚠️ 預先登記（跑之前定好，跑完唔改）：
線A = 12-1動量前10隻平均分配，每半個月換倉（15號或之前最後交易日 + 月尾），70% + 30% SPY訊號方案（main.py）。
廣度線 = 第十一輪T4b：同線A一樣揀股，但動量部分只投入「前10入面3個月回報高過SPY嘅比例」。
其他設定同第十一輪（當時標普500成分股、已計股息、每次買賣0.1%、唔用槓桿）。

① D組合：線A同廣度線按 75/25、50/50、25/75 混合（每日固定比例），同埋兩條線每日回報嘅相關系數。
② 回報來源拆解（線A，同埋線A入面嘅動量部分）：
   a. 學術因子：Fama-French 每日 市場、規模、價值、動量 四因子（Kenneth French網站）做回歸，
      計每年 alpha、t值、各因子β、R²、剩餘部分嘅Sharpe（alpha ÷ 剩餘波幅，即 information ratio）
   b. 可買賣嘅ETF：用 SPY 同 MTUM（iShares動量ETF，2013年起）做回歸
③ 隨機換倉日：線A規則唔變，但每半個月入面隨機揀一日換倉，300次 → 睇固定日子嘅結果喺隨機分佈入面排第幾
④ QQQ基準：QQQ長揸（已計股息）同線A、線C比較
時段：2010-2026、2015-2026、2020-2026（覆蓋率最高）。
輸出：v12_results.csv、v12_progress.txt
"""
import io
import zipfile

import numpy as np
import pandas as pd
import requests

import momentum_v1 as M
from research_v5 import stats
from research_v6 import effective_weights, LOOK, SKIP
from research_v8 import decision_days
from research_v10 import sleeve, metrics

M.PROGRESS = "v12_progress.txt"
log = M.log
COST = 0.001
WINDOWS = [("2010-2026", "2010-01-01", "2026-09-30"), ("2015-2026", "2015-01-01", "2026-09-30"),
           ("2020-2026", "2020-01-01", "2026-09-30")]


def french(name):
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
    z = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=60, headers=M.HDR).content))
    txt = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    start = next(k for k, l in enumerate(txt) if "," in l and l.split(",")[0].strip() == "" and len(l.split(",")) > 1)
    rows = []
    header = [c.strip() for c in txt[start].split(",")]
    for l in txt[start + 1:]:
        p = [c.strip() for c in l.split(",")]
        if len(p) != len(header) or not p[0].isdigit() or len(p[0]) != 8:
            if rows:
                break
            continue
        rows.append(p)
    df = pd.DataFrame(rows, columns=["Date"] + header[1:])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    return df.set_index("Date").astype(float) / 100


def ols(y, X):
    X = X.copy()
    X.insert(0, "alpha", 1.0)
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    yv, Xv = d["y"].values, d.drop(columns="y").values
    b, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    res = yv - Xv @ b
    n, k = Xv.shape
    s2 = res @ res / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xv.T @ Xv)))
    r2 = 1 - (res @ res) / ((yv - yv.mean()) @ (yv - yv.mean()))
    out = dict(zip(d.drop(columns="y").columns, b))
    return out, dict(zip(d.drop(columns="y").columns, b / se)), r2, res.std() * np.sqrt(252), n


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    month_ends = pd.date_range("2008-12-31", "2026-09-30", freq="ME")
    everyone = sorted(set().union(*[members_fn(d) for d in month_ends]))
    closes = M.download_close(everyone + ["SPY", "QQQ", "MTUM"], "2007-06-01", "2026-10-01")
    spy = closes.pop("SPY")
    qqq = closes.pop("QQQ", None)
    mtum = closes.pop("MTUM", None)
    px = pd.DataFrame(closes).reindex(spy.index)
    px, spy = px[px.index <= pd.Timestamp("2026-09-30")], spy[spy.index <= pd.Timestamp("2026-09-30")]
    days = px.index
    rets = px.pct_change(fill_method=None)
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = qqq.reindex(days).pct_change().fillna(0.0) if qqq is not None else None
    mtum_r = mtum.reindex(days).pct_change() if mtum is not None else None
    cash_r = M.get_cash(days)

    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    st = (df["state"] != "out").astype(float).reindex(days).fillna(0.0).shift(2).fillna(0.0)
    bot_r = st * spy_r + (1 - st) * cash_r - st.diff().abs().fillna(0.0) * COST

    R121 = px.shift(SKIP) / px.shift(LOOK) - 1
    R3 = px / px.shift(63) - 1
    S3 = spy / spy.shift(63) - 1
    S121 = spy.shift(SKIP) / spy.shift(LOOK) - 1
    mem_cache = {}

    def members(d):
        k = (d.year, d.month)
        if k not in mem_cache:
            mem_cache[k] = [t for t in members_fn(d) if t in px.columns]
        return mem_cache[k]

    def pick(i):
        d = days[i]
        m = members(d)
        row = R121.iloc[i][m]
        ok = px.iloc[i][m].notna() & px.iloc[i - LOOK][m].notna()
        return list(row[ok].dropna().sort_values(ascending=False).index[:10])

    def targets_for(dlist, breadth=False):
        tg = {}
        for d in dlist:
            i = days.get_loc(d)
            if i < LOOK + 5 or d < pd.Timestamp("2009-06-01"):
                continue
            top = pick(i)
            if not top:
                continue
            frac = float((R3.iloc[i][top] > S3.iloc[i]).mean()) if breadth else 1.0
            tg[d] = pd.Series(frac / len(top), index=top)
        return tg

    fixed_days = decision_days(days, "每半個月")
    WA, _ = effective_weights(rets, targets_for(fixed_days))
    slA, tvA = sleeve(WA, rets, cash_r)
    lineA = 0.7 * slA + 0.3 * bot_r
    WB, _ = effective_weights(rets, targets_for(fixed_days, breadth=True))
    slB, tvB = sleeve(WB, rets, cash_r)
    lineBr = 0.7 * slB + 0.3 * bot_r
    log("線A同廣度線完成")

    # 線C（多週期）作比較
    R6 = px / px.shift(126) - 1
    tgC = {}
    for d in fixed_days:
        i = days.get_loc(d)
        if i < LOOK + 5 or d < pd.Timestamp("2009-06-01"):
            continue
        m = members(d)
        ok = px.iloc[i][m].notna() & px.iloc[i - LOOK][m].notna()
        mm = [t for t in m if ok[t]]
        comp = pd.concat([R3.iloc[i][mm].rank(pct=True), R6.iloc[i][mm].rank(pct=True), R121.iloc[i][mm].rank(pct=True)],
                         axis=1).mean(axis=1, skipna=False).dropna()
        top = list(comp.sort_values(ascending=False).index[:10])
        tgC[d] = pd.Series(0.1, index=top)
    WC, _ = effective_weights(rets, tgC)
    slC, tvC = sleeve(WC, rets, cash_r)
    lineC = 0.7 * slC + 0.3 * bot_r

    zero = pd.Series(0.0, index=days)
    L = ["=== 第十二輪：驗證 alpha（規則預先登記，見research_v12.py開頭）===",
         f"股票池：當時標普500成分股（{source}）；已計股息；每次買賣0.1%；唔用槓桿", ""]

    def row(name, r, t=zero):
        parts = []
        for wn, a, b in WINDOWS:
            m = metrics(r, t, a, b)
            parts.append(f"{wn} {m['cagr']:+.2f}%（跌 {m['mdd']:.1f}%，Sharpe {m['sharpe']:.2f}）")
        return f"　{name}：" + " | ".join(parts)

    # ④ 基準
    L.append("=== ④ 基準比較（全部已計股息）===")
    L.append(row("SPY長揸", spy_r))
    if qqq_r is not None:
        L.append(row("QQQ長揸", qqq_r))
    L.append(row("SPY訊號方案", bot_r))
    L.append(row("線A", lineA, 0.7 * tvA))
    L.append(row("線C（多週期）", lineC, 0.7 * tvC))
    L.append(row("廣度線（T4b）", lineBr, 0.7 * tvB))
    L.append("")

    # ① D組合
    L.append("=== ① D組合：線A + 廣度線 ===")
    cc = pd.concat([lineA, lineBr], axis=1)
    cc = cc[cc.index >= pd.Timestamp("2010-01-01")]
    L.append(f"　線A同廣度線每日回報相關系數：{cc.corr().iloc[0, 1]:.2f}")
    for w in (0.75, 0.5, 0.25):
        L.append(row(f"D {w:.0%}線A + {1 - w:.0%}廣度", w * lineA + (1 - w) * lineBr, 0.7 * (w * tvA + (1 - w) * tvB)))
    L.append("")

    # ② 回報來源拆解
    L.append("=== ② 回報來源拆解 ===")
    try:
        ff = french("F-F_Research_Data_Factors_daily")
        mom = french("F-F_Momentum_Factor_daily")
        mcol = [c for c in mom.columns if "mom" in c.lower()][0]
        fac = ff.join(mom[[mcol]].rename(columns={mcol: "Mom"}), how="inner")
        L.append(f"　Fama-French數據：{fac.index[0].date()} 至 {fac.index[-1].date()}")
        for nm, r in (("線A（成個組合）", lineA), ("線A動量部分（100%股票）", slA), ("線C（成個組合）", lineC)):
            for wn, a, b in WINDOWS:
                y = (r - fac["RF"].reindex(r.index)).loc[a:b]
                X = fac[["Mkt-RF", "SMB", "HML", "Mom"]].reindex(y.index)
                if len(X.dropna()) < 250:
                    continue
                coef, tval, r2, resvol, n = ols(y, X)
                alpha = coef["alpha"] * 252 * 100
                L.append(f"　{nm} {wn}：alpha 每年 {alpha:+.2f}%（t={tval['alpha']:.2f}）｜ 市場β {coef['Mkt-RF']:.2f}、"
                         f"規模β {coef['SMB']:.2f}、價值β {coef['HML']:.2f}、動量β {coef['Mom']:.2f} ｜ R² {r2:.2f} ｜ "
                         f"剩餘Sharpe {alpha / (resvol * 100):.2f}（{n}日）")
    except Exception as e:
        L.append(f"　Fama-French數據用唔到：{e}")
    if mtum_r is not None:
        L.append("　—— 用可買賣ETF：SPY + MTUM（2013-05起）——")
        for nm, r in (("線A", lineA), ("線A動量部分", slA)):
            for wn, a, b in WINDOWS:
                aa = max(pd.Timestamp(a), pd.Timestamp("2013-06-01"))
                y = (r - cash_r).loc[aa:b]
                X = pd.DataFrame({"SPY": spy_r - cash_r, "MTUM": mtum_r - cash_r}).reindex(y.index)
                if len(X.dropna()) < 250:
                    continue
                coef, tval, r2, resvol, n = ols(y, X)
                alpha = coef["alpha"] * 252 * 100
                L.append(f"　{nm} {wn}（由{aa.date()}）：alpha 每年 {alpha:+.2f}%（t={tval['alpha']:.2f}）｜ SPY β {coef['SPY']:.2f}、"
                         f"MTUM β {coef['MTUM']:.2f} ｜ R² {r2:.2f} ｜ 剩餘Sharpe {alpha / (resvol * 100):.2f}")
        L.append(row("MTUM長揸（參考）", mtum_r.fillna(0.0)))
    L.append("")

    # ③ 隨機換倉日
    L.append("=== ③ 隨機換倉日（線A規則唔變，每半個月入面隨機揀一日），300次 ===")
    rng = np.random.default_rng(12)
    halves = pd.Series(days, index=days)
    key = days.year * 100 + days.month * 2 + (days.day > 15)
    groups = [g.index for _, g in halves.groupby(key)]
    res = {wn: [] for wn, _, _ in WINDOWS}
    for k in range(300):
        dl = [g[rng.integers(len(g))] for g in groups]
        W, _ = effective_weights(rets, targets_for(dl))
        s, _ = sleeve(W, rets, cash_r)
        mix = 0.7 * s + 0.3 * bot_r
        for wn, a, b in WINDOWS:
            res[wn].append(stats(mix, a, b)["cagr"])
        if (k + 1) % 50 == 0:
            log(f"隨機換倉日 {k + 1}/300")
    for wn, a, b in WINDOWS:
        x = np.array(res[wn])
        fixed = stats(lineA, a, b)["cagr"]
        L.append(f"　{wn}：固定日子 {fixed:+.2f}% ｜ 隨機日子 中位數 {np.median(x):+.2f}%（5%-95%：{np.percentile(x, 5):+.2f}% 至 "
                 f"{np.percentile(x, 95):+.2f}%）｜ 固定日子好過 {np.mean(x < fixed) * 100:.0f}% 隨機")
    pd.DataFrame({"report": L}).to_csv("v12_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
