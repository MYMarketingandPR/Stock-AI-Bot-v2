"""
research_v16.py — 第十六輪：倍升股「提早認出」研究（Multibagger DNA，Esther + ChatGPT + 另一AI 建議，2026-10-05）

問題：唔係「已經升咗先追」，而係「未大升之前，用當時已知資料，可唔可以提高揀中『未來2年升3倍』股票嘅機會？」

⚠️ 預先登記（2026-10-05 跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：

數據：Yahoo 價錢+成交量（1998起）、Yahoo 每季EPS（實際+預期，2002起）。
  冇免費歷史：營收、利潤率、現金流、負債、機構持股、分析師預測 → 唔測。
  行業：用今日分類（Nasdaq screener；標普500用Wikipedia GICS），已除牌冇分類 = 唔用行業條件（當過關）。
股票池：U1 當時真實標普500（正式判斷）；U3 今日全美國上市普通股 + 50日平均成交額≥500萬美元（有倖存者偏差，只當上限參考）。
決定日：每月最後交易日（2003-01 至 2026-08），收市決定、下一日收市買賣，每次買賣0.1%；最多10隻平均分配，唔夠10隻其餘揸現金。

特徵（全部只用當日已知資料）：
  r12/r6/r3 = 12/6/3個月回報；RS6 = 6個月回報喺股票池入面嘅排名（0-1）；離52週高位；喺MA50/MA200上面；
  波幅收窄 = 10日波幅 ÷ 50日波幅；成交量增加 = 20日平均成交量 ÷ 120日平均；公司大細 = log(50日平均成交額)；
  EPS按年增長（最近一季 vs 一年前同季）；EPS加速 = 今季按年增長 − 上季按年增長；盈利驚喜 = (實際−預期)÷公布前股價；
  （EPS要126個交易日內公布先算）；行業強度 = 同行業股票6個月回報中位數嘅排名。
目標：之後504個交易日內最高收市價 ≥ 今日 × 3（「3倍股」）；另外列 ×5。

方法（只有呢6個）：
  A 純動量：RS6 最高10隻
  B 純盈利：EPS按年增長 >25% 而且加速 >0，按EPS增長揀最高10隻
  C 盈利+動量+近高位（第一份建議）：喺MA50同MA200上面 + 離52週高位10%以內 + EPS增長 >20%，按RS6揀10隻
  D 盈利加速+動量+行業（ChatGPT）：行業強度前20% + RS6 前10% + EPS加速 >0，按RS6揀10隻
  F 「未爆但開始爆」（ChatGPT）：12個月升 +20%至+80% + 離52週高位5%以內 + EPS加速 >0 + 成交量增加 >1，按r3揀10隻
  E 電腦逐年學習：用上面全部特徵做邏輯回歸（logistic regression），預測「未來2年升3倍」機率；
    每年1月重新學，只用結果已經完全知道嘅舊樣本（決定日 ≤ 今日 − 504日）；2007年起先開始揀；按機率揀最高10隻。
    （冇任何人手揀條件；L2 正則化 λ=1；特徵按訓練樣本標準化，1%/99%截斷）

判斷（U1，同QQQ同一段時間比）：
  組合（100%）：驗證期 2015-01 至 2026-09：① 年化 ≥ QQQ+3 ② Sharpe ≥ QQQ ③ 最大跌幅好過 -35%
    ④ 三個年代（2003-2009 / 2010-2019 / 2020-2026）最少兩個贏QQQ → 全部符合先叫合格。
  命中率：驗證期（2015-01 至 2024-08，要有2年結果）揀中3倍股比例 ≥ 同期股票池平均 × 2。
  85% QQQ + 15% 組合（每日固定比例）：驗證期年化 ≥ QQQ+1 而且最大跌幅唔差過 QQQ 3個百分點以上 → 「值得做衛星倉」。
  6個方法 × 2個股票池，試得多總會有幾個好彩 → 只用 U1 判斷；U3 只作參考。

另外（純描述）：
  ① 3倍股 vs 其他股票：事前特徵中位數比較；按EPS增長、EPS加速、RS6、離高位、成交量增加、公司大細分10組嘅3倍股機率
  ② 已升1-2倍/2-3倍/3-5倍/5倍以上 × EPS加速中/減速中 → 之後1年中位數、平均、贏SPY機率、跌≥30%機率

輸出：v16_results.csv、v16_progress.txt
==== 結果（2026-10-05 第一次跑，記錄低，唔再改）====
- U1 樣本105,181（股票×月）；平均2年內升3倍 1.69%、升5倍 0.29%。U3 樣本403,131；升3倍 4.42%。
- 3倍股事前（U1中位數）：12個月 -20%、離52週高位 -33%、EPS增長 -5%、EPS減速 → 「睇落好差」。
  最強預測：離52週高位最遠嗰10% = 平均5.1倍機率（大跌後反彈）；EPS增長/相對強度最低同最高兩頭都高（即係波動大）；
  成交量增加最高10% = 1.9倍；波幅收窄（VCP）冇預測力（約1.0倍）。
- 命中率（驗證期）：E電腦學習 5.5倍、A純動量 5.4倍、D 3.3倍、C 2.6倍、B 2.5倍、F「未爆但開始爆」0.8倍（差過隨便揀）。
- 但組合（U1驗證期，QQQ +19.1%/-35.1%）：A +17.6%/-38.0%、B +14.3%/-42.6%、C +9.0%、D +9.7%、E +10.5%/-64.8%、F +5.7%
  → 6個全部唔合格。揀中多啲3倍股，同時揀中更多大跌股。
- 85% QQQ + 15%：U1 全部同QQQ差唔多（+17.3%至+19.2%）→ 唔值得；U3 只有D +20.3%/-33.1% 過關，但U3有倖存者偏差，唔算數。
- 已升2-3倍・EPS加速中（U1，106個）：之後1年中位數 +24%、贏SPY 59%；但U3同一組只有 +2% → 樣本細，未可信。
  全市場：EPS「唔知」嘅倍升股最危險（升5倍以上・唔知：中位數 -48%）。
"""
import bisect
import io

import numpy as np
import pandas as pd
import requests

import momentum_v1 as M
from perf import perf
from research_v13 import get_cash
from research_v15 import fetch_earn, known_index
from research_v15b import us_listed, download

M.PROGRESS = "v16_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"
H = 504
TOP = 10


def industries():
    out = {}
    try:
        r = requests.get("https://api.nasdaq.com/api/screener/stocks", params={"tableonly": "true", "download": "true"},
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"},
                         timeout=60)
        for row in r.json()["data"]["rows"]:
            ind = row.get("industry") or row.get("sector")
            if ind:
                out[M.norm(row["symbol"])] = ind
        log(f"Nasdaq 行業分類：{len(out)} 隻")
    except Exception as e:
        log(f"Nasdaq 行業分類攞唔到：{e}")
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=M.HDR, timeout=30).text
        t = pd.read_html(io.StringIO(html))[0]
        n = 0
        for s, ind in zip(t["Symbol"], t["GICS Sub-Industry"]):
            k = M.norm(s)
            if k not in out:
                out[k] = ind
                n += 1
        log(f"Wikipedia GICS 補充：{n} 隻")
    except Exception as e:
        log(f"Wikipedia 行業攞唔到：{e}")
    return out


def logistic_fit(X, y, lam=1.0, iters=30):
    n, k = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(k + 1)
    reg = lam * np.eye(k + 1)
    reg[0, 0] = 0
    for _ in range(iters):
        z = np.clip(Xb @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) + reg @ w
        Wd = p * (1 - p) + 1e-9
        Hm = (Xb * Wd[:, None]).T @ Xb + reg
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    month_ends = pd.date_range("1999-01-31", END, freq="ME")
    hist_sp = sorted(set().union(*[members_fn(d) for d in month_ends]))
    try:
        allus = us_listed()
    except Exception as e:
        log(f"攞唔到全美國名單：{e}")
        allus = []
    log(f"全美國 {len(allus)}；歷史標普500 {len(hist_sp)}")
    import yfinance as yf
    def idx_series(t):
        s = yf.Ticker(t).history(start="1998-01-01", end="2026-10-01", auto_adjust=True)["Close"]
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        return s
    spy, qqq = idx_series("SPY"), idx_series("QQQ")
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    C, V = download(sorted(set(allus) | set(hist_sp)), "1998-01-01", "2026-10-01")
    cols = sorted(C)
    col_ix = {t: k for k, t in enumerate(cols)}
    Pdf = pd.DataFrame(C).reindex(days)[cols].astype("float32")
    Vdf = pd.DataFrame(V).reindex(days)[cols].astype("float32")
    del C, V
    f32 = lambda x: x.values.astype("float32")
    Pv = f32(Pdf)
    Pf = f32(Pdf.ffill())
    ma50, ma200 = f32(Pdf.rolling(50).mean()), f32(Pdf.rolling(200).mean())
    r12, r6, r3 = f32(Pdf / Pdf.shift(252) - 1), f32(Pdf / Pdf.shift(126) - 1), f32(Pdf / Pdf.shift(63) - 1)
    hi52 = f32(Pdf.rolling(252, min_periods=200).max())
    dr = Pdf.pct_change(fill_method=None)
    sd10, sd50 = f32(dr.rolling(10).std()), f32(dr.rolling(50).std())
    del dr
    v20, v120 = f32(Vdf.rolling(20, min_periods=15).mean()), f32(Vdf.rolling(120, min_periods=90).mean())
    dv50 = f32((Pdf * Vdf).rolling(50, min_periods=40).mean())
    del Pdf, Vdf
    # 未來504日最高價（用反轉滾動最大值）
    fut_max = np.full(Pv.shape, np.nan, dtype="float32")
    rev = pd.DataFrame(Pv[::-1])
    fm = rev.rolling(H, min_periods=1).max().values[::-1]          # 包括今日嘅未來504日最高
    fut_max[:-1] = fm[1:]
    del rev, fm
    log("價錢指標計好")
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = qqq.reindex(days).pct_change(fill_method=None).fillna(0.0)
    cash_r = get_cash(days)
    cr = cash_r.values
    spy_v = spy.values

    ind_map = industries()
    ind_of = np.array([ind_map.get(t, "") for t in cols], dtype=object)

    # ---------- EPS ----------
    earn = fetch_earn(cols)
    E = {}
    for t, df in earn.items():
        df = df[~df.index.duplicated(keep="first")].sort_index()
        rep, est = df["Reported EPS"].astype(float), df["EPS Estimate"].astype(float)
        dates = [pd.Timestamp(x).tz_convert("America/New_York").tz_localize(None) if pd.Timestamp(x).tzinfo else pd.Timestamp(x)
                 for x in df.index]
        c = col_ix[t]
        ks, gs, sus = [], [], []
        for j in range(len(df)):
            if not np.isfinite(rep.iloc[j]):
                continue
            k = known_index(days, df.index[j])
            if k is None:
                continue
            g = np.nan
            for jj in range(j - 1, -1, -1):
                gap = (dates[j] - dates[jj]).days
                if gap > 400:
                    break
                if 330 <= gap <= 400 and np.isfinite(rep.iloc[jj]):
                    if abs(rep.iloc[jj]) > 0.01:
                        g = (rep.iloc[j] - rep.iloc[jj]) / abs(rep.iloc[jj])
                    break
            jb = days.searchsorted(dates[j].normalize(), side="left") - 1
            pb = Pf[jb, c] if jb >= 0 else np.nan
            su = (rep.iloc[j] - est.iloc[j]) / pb if np.isfinite(est.iloc[j]) and np.isfinite(pb) and pb > 0 else np.nan
            ks.append(k)
            gs.append(g)
            sus.append(su)
        if ks:
            o = np.argsort(ks, kind="stable")
            ks = [ks[x] for x in o]
            gs = [gs[x] for x in o]
            sus = [sus[x] for x in o]
            acc = [np.nan] + [gs[x] - gs[x - 1] if np.isfinite(gs[x]) and np.isfinite(gs[x - 1]) else np.nan for x in range(1, len(gs))]
            E[c] = (ks, gs, acc, sus)
    log(f"EPS 整理好：{len(E)} 隻")

    def eps_at(c, i):
        if c not in E:
            return np.nan, np.nan, np.nan
        ks, gs, acc, sus = E[c]
        p = bisect.bisect_right(ks, i) - 1
        if p < 0 or i - ks[p] > 126:
            return np.nan, np.nan, np.nan
        return gs[p], acc[p], sus[p]

    # ---------- 股票池 ----------
    U = {}
    m1 = np.zeros(Pv.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [col_ix[t] for t in members_fn(d) if t in col_ix]
        m1[i, cache[key]] = True
    U["U1 當時真實標普500"] = m1
    if allus:
        m3 = np.zeros(Pv.shape, dtype=bool)
        m3[:, [col_ix[t] for t in allus if t in col_ix]] = True
        m3 &= np.nan_to_num(dv50, nan=0.0) >= 5e6
        U["U3 全美國（有偏差，只作參考）"] = m3

    me = [days.searchsorted(d, side="right") - 1 for d in pd.date_range("2003-01-31", "2026-08-31", freq="ME")]
    FEATS = ["r12", "r6", "r3", "rs6", "off_hi", "above200", "contract", "volx", "size", "eps_g", "eps_acc",
             "eps_known", "surprise", "ind"]

    def build_samples(mask):
        rows = []
        for i in me:
            cs = np.where(mask[i] & np.isfinite(Pv[i]) & np.isfinite(r12[i]) & np.isfinite(r6[i]))[0]
            if len(cs) < 20:
                continue
            rs6 = pd.Series(r6[i, cs]).rank(pct=True).values
            inds = ind_of[cs]
            ind_med = pd.Series(r6[i, cs]).groupby(inds).median()
            ind_cnt = pd.Series(1, index=range(len(cs))).groupby(inds).sum()
            ok_ind = {k: v for k, v in ind_med.items() if k and ind_cnt.get(k, 0) >= 3}
            rk = pd.Series(ok_ind).rank(pct=True).to_dict() if ok_ind else {}
            for n, c in enumerate(cs):
                g, a, su = eps_at(c, i)
                fm = fut_max[i, c]
                y3 = (fm / Pv[i, c] >= 3) if (i + H < len(days) and np.isfinite(fm)) else np.nan
                y5 = (fm / Pv[i, c] >= 5) if (i + H < len(days) and np.isfinite(fm)) else np.nan
                e12 = i + 252
                f12 = Pf[e12, c] / Pv[i, c] - 1 if e12 < len(days) else np.nan
                s12 = spy_v[e12] / spy_v[i] - 1 if e12 < len(days) else np.nan
                rows.append((i, c, r12[i, c], r6[i, c], r3[i, c], rs6[n],
                             Pv[i, c] / hi52[i, c] - 1 if np.isfinite(hi52[i, c]) else np.nan,
                             float(Pv[i, c] > ma200[i, c]), float(Pv[i, c] > ma50[i, c]),
                             sd10[i, c] / sd50[i, c] if sd50[i, c] > 0 else np.nan,
                             v20[i, c] / v120[i, c] if v120[i, c] > 0 else np.nan,
                             np.log10(max(dv50[i, c], 1.0)) if np.isfinite(dv50[i, c]) else np.nan,
                             g, a, float(np.isfinite(g)), su, rk.get(inds[n], np.nan), y3, y5, f12, s12))
        df = pd.DataFrame(rows, columns=["i", "c", "r12", "r6", "r3", "rs6", "off_hi", "above200", "above50", "contract",
                                         "volx", "size", "eps_g", "eps_acc", "eps_known", "surprise", "ind",
                                         "y3", "y5", "f12", "s12"])
        df["date"] = days[df["i"].values]
        return df

    def portfolio(picks):
        r = cr.copy()
        r[:] = 0.0
        ks = sorted(picks)
        if not ks:
            return pd.Series(0.0, index=days)
        r[:ks[0] + 2] = cr[:ks[0] + 2]
        prev = set()
        for a, i in enumerate(ks):
            s = i + 1
            e = ks[a + 1] + 1 if a + 1 < len(ks) else len(days) - 1
            if s >= len(days) - 1:
                break
            cs = [c for c in picks[i] if np.isfinite(Pf[s, c]) and Pf[s, c] > 0][:TOP]
            seg_len = e - s
            cash_nav = np.cumprod(np.r_[1.0, 1 + cr[s + 1:e + 1]])
            nav = cash_nav * (TOP - len(cs)) / TOP
            if cs:
                st = Pf[s:e + 1][:, cs] / Pf[s, cs]
                nav = nav + np.nansum(st, axis=1) / TOP
            dr_ = nav[1:] / nav[:-1] - 1
            new = set(cs) - prev
            gone = prev - set(cs)
            dr_[0] -= (len(new) + len(gone)) / TOP * COST
            r[s + 1:s + 1 + seg_len] = dr_[:seg_len]
            prev = set(cs)
        return pd.Series(r, index=days)

    def rule_picks(df, model):
        out = {}
        for i, g in df.groupby("i"):
            if model == "A":
                sel = g.sort_values("rs6", ascending=False)
            elif model == "B":
                sel = g[(g["eps_g"] > 0.25) & (g["eps_acc"] > 0)].sort_values("eps_g", ascending=False)
            elif model == "C":
                sel = g[(g["above50"] == 1) & (g["above200"] == 1) & (g["off_hi"] >= -0.10) & (g["eps_g"] > 0.20)]
                sel = sel.sort_values("rs6", ascending=False)
            elif model == "D":
                sel = g[((g["ind"] >= 0.8) | g["ind"].isna()) & (g["rs6"] >= 0.9) & (g["eps_acc"] > 0)]
                sel = sel.sort_values("rs6", ascending=False)
            elif model == "F":
                sel = g[(g["r12"] >= 0.2) & (g["r12"] <= 0.8) & (g["off_hi"] >= -0.05) & (g["eps_acc"] > 0) & (g["volx"] > 1)]
                sel = sel.sort_values("r3", ascending=False)
            out[i] = list(sel["c"].head(TOP))
        return out

    def model_e(df):
        X_all = df[FEATS].copy()
        X_all["eps_g"] = X_all["eps_g"].clip(-2, 5)
        X_all["eps_acc"] = X_all["eps_acc"].clip(-5, 5)
        X_all["surprise"] = X_all["surprise"].clip(-0.2, 0.2)
        out, coefs = {}, {}
        years = sorted(set(days[df["i"]].year))
        for y in years:
            if y < 2007:
                continue
            cut = days.searchsorted(pd.Timestamp(f"{y}-01-01")) - H
            tr = df[(df["i"] <= cut) & df["y3"].notna()]
            log(f"E {y}：訓練樣本 {len(tr)}，3倍股 {int(tr['y3'].astype(float).sum())}")
            if len(tr) < 1000 or tr["y3"].astype(float).sum() < 20:
                continue
            Xt = X_all.loc[tr.index]
            lo, hi = Xt.quantile(0.01), Xt.quantile(0.99)
            Xt = Xt.clip(lo, hi, axis=1)
            med = Xt.median()
            Xt = Xt.fillna(med)
            mu, sd = Xt.mean(), Xt.std().replace(0, 1)
            w = logistic_fit(((Xt - mu) / sd).values, tr["y3"].astype(float).values)
            coefs[y] = dict(zip(FEATS, w[1:]))
            te = df[days[df["i"]].year == y]
            Xs = ((X_all.loc[te.index].clip(lo, hi, axis=1).fillna(med) - mu) / sd).values
            pr = 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(Xs)), Xs] @ w, -30, 30)))
            te = te.assign(p=pr)
            for i, g in te.groupby("i"):
                out[i] = list(g.sort_values("p", ascending=False)["c"].head(TOP))
        return out, coefs

    L = ["=== 第十六輪：倍升股提早認出（規則預先登記，見research_v16.py開頭）===",
         f"全美國名單 {len(allus)}；有價錢 {len(cols)}；有EPS {len(E)}；有行業分類 {int((ind_of != '').sum())}", ""]
    zero = pd.Series(0.0, index=days)

    def show(name, r, a, b):
        p = perf(r.loc[a:b], rf=cash_r, bench=qqq_r.loc[a:b], cap_freq="ME")
        return p, (f"　{name}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f} ｜ "
                   f"Calmar {p['calmar']:.2f} ｜ 對QQQ 升市捕捉 {p['up_cap']:.0f}% / 跌市捕捉 {p['down_cap']:.0f}%")

    windows = [("驗證期", "2015-01-01", END), ("2003-2009", "2003-02-01", "2009-12-31"),
               ("2010-2019", "2010-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END),
               ("全期", "2003-02-01", END), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    qt = {wn: show("QQQ長揸", qqq_r, a, b) for wn, a, b in windows}
    names = {"A": "A 純動量", "B": "B 純盈利", "C": "C 盈利+動量+近高位", "D": "D 盈利加速+動量+行業",
             "F": "F 未爆但開始爆", "E": "E 電腦逐年學習"}

    for un, mask in U.items():
        df = build_samples(mask)
        log(f"{un} 樣本 {len(df)}")
        L.append(f"################ {un} ################")
        L.append(f"　樣本（股票×月）{len(df)}；EPS有數 {df['eps_known'].mean() * 100:.0f}%；有行業 {df['ind'].notna().mean() * 100:.0f}%")
        known = df[df["y3"].notna()]
        base3 = known["y3"].astype(float).mean()
        base5 = known["y5"].astype(float).mean()
        L.append(f"　平均：2年內升3倍 {base3 * 100:.2f}%、升5倍 {base5 * 100:.2f}%")
        # ---- 描述① DNA ----
        L.append("=== 描述①：3倍股事前特徵（中位數）===")
        for lab, g in (("之後2年升3倍", known[known["y3"] == True]), ("其他", known[known["y3"] == False])):
            L.append(f"　{lab}（{len(g)}）：12個月 {g['r12'].median() * 100:+.0f}% ｜ 6個月 {g['r6'].median() * 100:+.0f}% ｜ "
                     f"離高位 {g['off_hi'].median() * 100:.0f}% ｜ EPS增長 {g['eps_g'].median() * 100:+.0f}% ｜ "
                     f"EPS加速 {g['eps_acc'].median() * 100:+.0f}% ｜ 成交量增加 {g['volx'].median():.2f} ｜ "
                     f"波幅收窄 {g['contract'].median():.2f} ｜ 大細(log成交額) {g['size'].median():.2f}")
        for col, lab in (("eps_g", "EPS增長"), ("eps_acc", "EPS加速"), ("rs6", "6個月相對強度"), ("off_hi", "離52週高位"),
                         ("volx", "成交量增加"), ("size", "公司大細"), ("contract", "波幅收窄")):
            q = known.dropna(subset=[col]).copy()
            if len(q) < 1000:
                continue
            q["dec"] = pd.qcut(q[col].rank(method="first"), 10, labels=False) + 1
            g = q.groupby("dec")["y3"].apply(lambda s: s.astype(float).mean())
            L.append(f"　按{lab}分10組（1低→10高）3倍股機率係平均幾多倍：" + " ".join(f"{v / base3:.1f}" for v in g.values))
        # ---- 描述② 已升幾倍 × EPS加速 ----
        L.append("=== 描述②：已升幾倍 × EPS加速中/減速中 → 之後1年 ===")
        up = df[(df["r12"] >= 1.0) & df["f12"].notna()].copy()
        up["b"] = pd.cut(up["r12"], [1, 2, 3, 5, np.inf], labels=["升1-2倍", "升2-3倍", "升3-5倍", "升5倍以上"], right=False)
        up["a"] = np.where(up["eps_acc"] > 0, "加速中", np.where(up["eps_acc"] <= 0, "減速中", "唔知"))
        for (b, a), g in up.groupby(["b", "a"], observed=True):
            if len(g) < 30:
                continue
            L.append(f"　{b}・{a}（{len(g)}）：中位數 {g['f12'].median() * 100:+.0f}% ｜ 平均 {g['f12'].mean() * 100:+.0f}% ｜ "
                     f"贏SPY {(g['f12'] > g['s12']).mean() * 100:.0f}% ｜ 跌≥30% {(g['f12'] <= -0.3).mean() * 100:.0f}%")
        # ---- 6個方法 ----
        picks = {m: rule_picks(df, m) for m in "ABCDF"}
        pe, coefs = model_e(df)
        picks["E"] = pe
        log(f"{un} 揀股完成")
        if coefs:
            last = coefs[max(coefs)]
            L.append(f"=== E 模型最近一年學到嘅權重（正數 = 越大越易變3倍股）===")
            L.append("　" + "，".join(f"{k} {v:+.2f}" for k, v in sorted(last.items(), key=lambda x: -abs(x[1]))))
        L.append("=== 6個方法：命中率（驗證期 2015-01 至 2024-08；要有2年結果）===")
        vk = known[(known["date"] >= "2015-01-01")]
        vbase = vk["y3"].astype(float).mean()
        key = known.set_index(["i", "c"])["y3"]
        hits = {}
        for m, pk in picks.items():
            ys = [key.get((i, c), np.nan) for i, cs in pk.items() if days[i] >= pd.Timestamp("2015-01-01") for c in cs]
            ys = np.array([float(y) for y in ys if y == y and y is not None])
            hr = ys.mean() if len(ys) else np.nan
            hits[m] = hr
            L.append(f"　{names[m]}：揀咗 {len(ys)} 隻次，3倍股 {hr * 100:.1f}%（平均 {vbase * 100:.1f}%，{hr / vbase if vbase else np.nan:.1f}倍）")
        rets = {m: portfolio(pk) for m, pk in picks.items()}
        for wn, a, b in windows:
            L.append(f"=== {un} {wn}（{a} 至 {b}）===")
            L.append(qt[wn][1])
            for m, r in rets.items():
                L.append(show(names[m], r, a, b)[1])
            for m, r in rets.items():
                L.append(show(f"85%QQQ+15%{names[m][:1]}", 0.85 * qqq_r + 0.15 * r, a, b)[1])
        L.append(f"=== {un} 預先登記判斷 ===")
        for m, r in rets.items():
            v, q = show("", r, "2015-01-01", END)[0], qt["驗證期"][0]
            eras = [show("", r, a, b)[0]["cagr"] > qt[wn][0]["cagr"]
                    for wn, a, b in windows if wn in ("2003-2009", "2010-2019", "2020-2026")]
            c4 = sum(eras) >= 2
            if m == "E":
                c4 = eras[1] and eras[2]           # E 2007年先開始：用 2010-2019、2020-2026 兩個年代，兩個都要贏
            c1, c2, c3 = v["cagr"] >= q["cagr"] + 3, v["sharpe"] >= q["sharpe"], v["mdd"] > -35
            c5 = hits[m] == hits[m] and vbase > 0 and hits[m] >= 2 * vbase
            cs_ = show("", 0.85 * qqq_r + 0.15 * r, "2015-01-01", END)[0]
            c6 = cs_["cagr"] >= q["cagr"] + 1 and cs_["mdd"] >= q["mdd"] - 3
            L.append(f"　{names[m]}：①{'✔' if c1 else '✘'} ②{'✔' if c2 else '✘'} ③{'✔' if c3 else '✘'} ④{'✔' if c4 else '✘'} "
                     f"→ 組合{'合格' if c1 and c2 and c3 and c4 else '唔合格'} ｜ 命中率 ≥ 平均2倍 {'✔' if c5 else '✘'} ｜ "
                     f"85/15衛星倉 年化 {cs_['cagr']:+.2f}%、跌 {cs_['mdd']:.1f}% → {'值得' if c6 else '唔值得'}")
        L.append("")
    pd.DataFrame({"report": L}).to_csv("v16_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
