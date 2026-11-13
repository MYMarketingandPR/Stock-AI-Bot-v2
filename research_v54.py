"""
research_v54.py — 第五十四輪：J Law / XQ 「02_VCP突破_警示」原型腳本，美股版（54A 開相對強度；54B 關相對強度）
  （Esther 2026-11-13：「我地係咪可以試下驗證J law既選股方法？做下回測？」；
    同意：出場「跌穿50日線 或 跌8%」；兩個版本都跑（開／關相對強度）；倉位同通過標準跟第53輪）

純研究，唔改任何之前結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：

一、要測試嘅係乜嘢
  測試嘅係 XQ 公開腳本「02_VCP突破_警示」嘅入場條件（原型），唔係 J Law 本人嘅完整方法。
  腳本原文說明 J Law 嘅額外條件無法取得，所以唔包括。
  腳本原為台股（XQ、TSE.TW）；本輪改用美股（標普500 歷史成分股）。

二、測試範圍
  期間：2016-01-01 至 2026-09-30（指標用 2013 年起嘅數據暖身）。
  股票池：當時真實標普500成分股（M.sp500_history，逐月；同線E、第53輪一樣）。
  本金：80,000 美元；唔用槓桿；閒錢收 ^IRX；每次買賣 0.1%。

三、入場條件（信號日 D 收市後判斷，只用 D 或之前嘅數據；全部符合先入市）
  腳本參數（預設值，唔改）：
    每段天數 SegLen = 25；末段收縮上限 10%；首段收縮下限 10%；量縮比 0.7；突破量能 1.5 倍；
    收盤相對52週低點最低倍數 1.3；收盤距52週高點最大 25%；MA200 上揚比較 20 日。
  (1) 趨勢樣板：
      收市 > MA150 且 > MA200；MA150 > MA200；MA200 > 20 日前嘅 MA200；
      MA50 > MA150 且 > MA200；收市 > MA50；
      收市 ≥ 250 日最低價 × 1.3；收市 ≥ 250 日最高價 × (1 − 25%)。
  (2) 波動收縮（回檔一次比一次淺）：
      切分突破前約 75 日為三段，每段 25 日，段深 = (最高 − 最低) ÷ 最高 × 100；
      d1（最早段）≥ 10%；d1 > d2 > d3；d3（末段）≤ 10%。
      樞紐點 = 末段最高價（即前 25 日最高，不含 D 日）。
  (3) 量縮：D 日前 10 日平均成交量 < 前 50 日平均成交量 × 0.7（唔含 D 日）。
  (4) 突破：收市由 D−1 日 ≤ 樞紐點，變成 D 日 > 樞紐點；
      D 日成交量 > 前 50 日平均成交量 × 1.5。
  (5) 流動性：過去 20 日平均成交額 ≥ 20,000,000 美元（取代台股「500 張」）。
  (6) 相對強度（只係 54A 用）：個股 250 日升幅 ≥ SPY 250 日升幅。54B 唔用呢條。

四、入場執行
  D 日符合條件 → D+1 日收市買入（同線E、第53輪一樣，唔用 D 日收市）。
  已持有嘅股票唔會再買。
  最多同時 5 隻。同日多隻符合，按 250 日升幅由高到低揀。

五、出場（收市判斷，下一個交易日收市執行）
  收市 < 入場價 × 0.92（即跌 8%）→ 賣出；
  或者 收市 < 50 日平均線 → 賣出。
  （腳本本身冇出場規則，呢個係事先定好嘅。）

六、倉位（同第53輪一樣）
  每單風險 = 總資產 1%；止損距離 = 8%；倉位比例 = min(1% ÷ 8%, 20%) = 12.5%。
  最多 5 隻 × 12.5% = 62.5%，其餘現金收 ^IRX。

七、基準同通過標準（同第53輪一樣）
  基準：線E（同期、同股票池，research_v51 嘅 C0 規則）；另列 SPY、QQQ 長揸。
  ① 驗證期年化 > QQQ 年化？
  ② 若 ① 係：最大跌幅 > −35% 且 Sharpe ≥ QQQ Sharpe → 符合；
  ③ 若 ① 唔係：最大跌幅 ≥ QQQ 最大跌幅 + 5 個百分點 且 年化 ≥ SPY 年化 → 符合；
  硬限制：2016–2026 最大跌幅 > −35%。
  跨年代：2016–2019、2020–2026 兩段，每段年化 ≥ SPY 同段年化，而且最大跌幅 > −35%。
  通過 = 硬限制 ✓ 且 跨年代 ✓ 且（② 或 ③）✓。
  另外報告（唔影響通過與否）：對線E 有改善嗎；最差 5 年年化（每月開始，用月底淨值計）。

八、記錄
  每次買入：日期、股票、原因、入場價、按 80,000 美元計嘅金額同股數、250 日相對強度。
  每次賣出：日期、股票、原因、買入日、買入價、賣出價、回報、持有日數。
  輸出：v54_events.csv（逐筆）、v54_results.csv（報告）。

已知限制：
  - 測試嘅係公開原型腳本，唔係 J Law 本人嘅方法；J Law 嘅額外條件冇包括。
  - 文章提到嘅「100 萬變 1,600 萬」同比賽成績，冇法核實。
  - 出場規則係我哋定嘅，腳本冇寫。
  - 兩個版本同時測試，有「撞大運」風險；所以兩個版本都要過同一條標準。
  - 行業分類等無用到；但股票池用歷史成分股，冇存活偏差。
  - 數據來自 Yahoo（auto_adjust），可能有缺漏。

==== 結果（2026-11-13 跑完，run 43414831097，記錄低，唔再改）====
基準（同期）：線E +30.14%／−32.6%／0.91；SPY +14.91%／−33.7%／0.75；QQQ +20.08%／−35.1%／0.83。
54A（開相對強度）：年化 +2.05%，最大跌幅 −3.1%，Sharpe −0.23，結束約 $99,498。買入 13 次，賣出 13 次（12 次跌穿50日線平均 +0.4%，1 次止損 −10.0%）。
  ① 否 ⇒ ②③ 不符合；硬限制符合；跨年代 2016–19 年化 +1.71%、2020–26 +2.26%，均跑輸 SPY。⇒ 唔通過。
54B（關相對強度）：年化 +2.08%，最大跌幅 −3.8%，Sharpe −0.13，結束約 $99,753。買入 16 次，賣出 16 次（15 次跌穿50日線平均 +0.5%，1 次止損 −10.0%）。
  ① 否 ⇒ ②③ 不符合；硬限制符合；跨年代 2016–19 年化 +1.71%、2020–26 +2.29%，均跑輸 SPY。⇒ 唔通過。
重點：兩個版本幾乎全部時間持現金，所以跌幅細，但回報都接近零；信號極少（13–16 次，10 年內）。相對強度濾網差異極細。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v13 import get_cash
from research_v26 import COST
from research_v51 import sim_stop as line_sim
from research_v53 import dl_ohlcv, worst5

M.PROGRESS = "v54_progress.txt"
log = M.log
START, END = "2016-01-01", "2026-09-30"
CAP = 80000
N = 5
RISK = 0.01
STOP = 0.08
MAXW = 0.20
LIQ = 20_000_000
SEG = 25
MAXLAST, MINFIRST = 10, 10
VDRY, VBRK = 0.7, 1.5
LOWM, HIGHP = 1.3, 25
MA200UP = 20


def vcp_base(Cdf, Hdf, Ldf, Vdf):
    ma50 = Cdf.rolling(50).mean()
    ma150 = Cdf.rolling(150).mean()
    ma200 = Cdf.rolling(200).mean()
    low52 = Ldf.rolling(250, min_periods=250).min()
    high52 = Hdf.rolling(250, min_periods=250).max()
    trend = ((Cdf > ma150) & (Cdf > ma200) & (ma150 > ma200) & (ma200 > ma200.shift(MA200UP))
             & (ma50 > ma150) & (ma50 > ma200) & (Cdf > ma50)
             & (low52 > 0) & (Cdf >= low52 * LOWM) & (high52 > 0) & (Cdf >= high52 * (1 - HIGHP / 100)))

    def seg(k):
        h = Hdf.shift(k).rolling(SEG, min_periods=SEG).max()
        l = Ldf.shift(k).rolling(SEG, min_periods=SEG).min()
        return h, l

    h3, l3 = seg(1)
    h2, l2 = seg(SEG + 1)
    h1, l1 = seg(2 * SEG + 1)
    d1 = (h1 - l1) / h1 * 100
    d2 = (h2 - l2) / h2 * 100
    d3 = (h3 - l3) / h3 * 100
    contract = (d1 >= MINFIRST) & (d1 > d2) & (d2 > d3) & (d3 <= MAXLAST)
    av10 = Vdf.shift(1).rolling(10, min_periods=10).mean()
    av50 = Vdf.shift(1).rolling(50, min_periods=50).mean()
    voldry = (av50 > 0) & (av10 / av50 < VDRY)
    pivot = h3
    crossup = (Cdf > pivot) & (Cdf.shift(1) <= pivot.shift(1))
    volbrk = Vdf > av50 * VBRK
    liq = (Cdf * Vdf).rolling(20, min_periods=20).mean() >= LIQ
    base = trend & contract & voldry & crossup & volbrk & liq & (pivot > 0)
    return base.fillna(False), ma50


def sim54(label, SIG, RK, Cv, RETv, MA50v, days, cr):
    T, C = Cv.shape
    hold, cash = {}, 1.0
    pend_sell, pend_buy = [], []
    navs, ev = [], []
    f = min(RISK / STOP, MAXW)
    for i in range(T):
        for c in hold:
            r = RETv[i, c]
            if np.isfinite(r):
                hold[c]["val"] *= 1 + r
        cash *= 1 + cr[i]
        # 1) 執行昨日嘅賣出
        for c, why in pend_sell:
            if c not in hold or not np.isfinite(Cv[i, c]):
                continue
            p = hold.pop(c)
            proceeds = p["val"] * (1 - COST)
            cash += proceeds
            ev.append(dict(動作="賣出", 股票=c, 日期=str(days[i].date()), 原因=why, 買入日=p["date"],
                           買入價=round(p["entry"], 4), 賣出價=round(float(Cv[i, c]), 4),
                           回報=proceeds / p["cost"] - 1, 持有日=i - p["eday"],
                           按80000美元計賣出金額=proceeds * CAP, _c=c, _i=i))
        pend_sell = []
        # 2) 執行昨日嘅買入
        total = cash + sum(p["val"] for p in hold.values())
        for c, rk in pend_buy:
            if c in hold or len(hold) >= N or not np.isfinite(Cv[i, c]):
                continue
            amt = min(f * total, cash)
            if amt <= 0:
                continue
            cash -= amt
            hold[c] = dict(val=amt * (1 - COST), cost=amt, entry=float(Cv[i, c]), eday=i,
                           date=str(days[i].date()))
            ev.append(dict(動作="買入", 股票=c, 日期=str(days[i].date()),
                           原因="VCP 放量突破樞紐點", 買入價=round(float(Cv[i, c]), 4),
                           止損價=round(float(Cv[i, c]) * (1 - STOP), 4), 倉位比例=f,
                           按80000美元計買入金額=amt * CAP, 股數=amt * CAP / float(Cv[i, c]),
                           相對強度250日=rk, _c=c, _i=i))
        pend_buy = []
        navs.append(cash + sum(p["val"] for p in hold.values()))
        # 3) 今日判斷出場（明日執行）
        sold = set()
        for c, p in hold.items():
            px = Cv[i, c]
            if not np.isfinite(px):
                continue
            if px < p["entry"] * (1 - STOP):
                pend_sell.append((c, "跌穿入場價8%止損"))
                sold.add(c)
            elif np.isfinite(MA50v[i, c]) and px < MA50v[i, c]:
                pend_sell.append((c, "收市跌穿50日線"))
                sold.add(c)
        # 4) 今日信號 → 明日買入
        free = N - (len(hold) - len(sold))
        if free > 0:
            idx = [c for c in np.where(SIG[i])[0] if c not in hold]
            idx.sort(key=lambda c: -RK[i, c])
            pend_buy = [(c, RK[i, c]) for c in idx[:free]]
    nav = np.array(navs)
    strat = pd.Series(np.r_[0.0, nav[1:] / nav[:-1] - 1], index=days)
    df = pd.DataFrame(ev)
    if len(df):
        post_max, post_min = [], []
        for _, e in df.iterrows():
            if e["動作"] == "賣出":
                i_, c_ = int(e["_i"]), int(e["_c"])
                seg = Cv[i_:min(i_ + 61, T), c_]
                seg = seg[np.isfinite(seg)]
                post_max.append(seg.max() / Cv[i_, c_] - 1 if len(seg) > 1 else np.nan)
                post_min.append(seg.min() / Cv[i_, c_] - 1 if len(seg) > 1 else np.nan)
            else:
                post_max.append(np.nan)
                post_min.append(np.nan)
        df["賣後60日最高"] = post_max
        df["賣後60日最低"] = post_min
        df = df.drop(columns=["_c", "_i"])
    return strat, df


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("2013-12-31", END, freq="ME")]))
    tick = sorted(set(everyone) | {"SPY", "QQQ"})
    ohlc = dl_ohlcv(tick, "2013-06-01", "2026-10-02")
    days = ohlc["Close"].index
    days = days[days <= pd.Timestamp(END)]
    cl = ohlc["Close"].reindex(index=days)
    spy_c, qqq_c = cl["SPY"], cl["QQQ"]
    stocks = [t for t in everyone if t in cl.columns and t not in ("SPY", "QQQ")]
    Cdf = cl.reindex(columns=stocks)
    Hdf = ohlc["High"].reindex(index=days, columns=stocks)
    Ldf = ohlc["Low"].reindex(index=days, columns=stocks)
    Vdf = ohlc["Volume"].reindex(index=days, columns=stocks)
    log(f"收市價準備好：{len(stocks)} 隻，{len(days)} 日（{source}）")

    base, ma50 = vcp_base(Cdf, Hdf, Ldf, Vdf)
    stk250 = Cdf / Cdf.shift(250) - 1
    spy250 = spy_c / spy_c.shift(250) - 1
    rs_ok = stk250.ge(spy250, axis=0).fillna(False)

    cols = stocks
    ix = {t: k for k, t in enumerate(cols)}
    mem = np.zeros((len(days), len(cols)), dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True

    SIG_base = base.values & mem
    SIG_A = SIG_base & rs_ok.values
    RK = stk250.fillna(-9).values
    log(f"信號數：54A {int(SIG_A.sum())}；54B {int(SIG_base.sum())}")

    cr = get_cash(days).values
    cash_r = get_cash(days)
    spy_r = spy_c.pct_change(fill_method=None).fillna(0.0)
    qqq_r = qqq_c.pct_change(fill_method=None).fillna(0.0)
    RETv = Cdf.pct_change(fill_method=None).values
    Cv = Cdf.values
    MA50v = ma50.values

    sA, evA = sim54("54A", SIG_A, RK, Cv, RETv, MA50v, days, cr)
    sB, evB = sim54("54B", SIG_base, RK, Cv, RETv, MA50v, days, cr)

    # 線E（同期、同股票池）
    MA200 = Cdf.rolling(200).mean().values
    ye = Cdf.groupby(days.year).last()
    ytdn = np.nan_to_num(Cv / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    HI52 = Cdf.rolling(252, min_periods=200).max().values
    hi52f = np.nan_to_num(HI52, nan=np.inf)
    ok = mem & (ytdn >= 0.30) & np.isfinite(Cv) & (Cv >= hi52f * 0.7) & np.nan_to_num(Cv > MA200, nan=False)
    zero = np.zeros_like(Cv)
    line_r = line_sim(Cv, RETv, cr, ok, ytdn, "C0", zero, zero, zero, zero, MA200, HI52, [], "C0 線E出場", days)
    sL = pd.Series(line_r, index=days)

    def stats(s, a=START, b=END):
        rr = s.loc[START:END].dropna().loc[a:b]
        return perf(rr, rf=cash_r, bench=qqq_r.loc[a:b])

    st = {"54A": stats(sA), "54B": stats(sB), "線E": stats(sL), "SPY": stats(spy_r), "QQQ": stats(qqq_r)}
    era = [("2016-01-01", "2019-12-31"), ("2020-01-01", END)]

    def judge(name, s):
        d = st[name]
        q, sp = st["QQQ"], st["SPY"]
        c1 = d["cagr"] > q["cagr"]
        q2 = c1 and d["mdd"] > -35 and d["sharpe"] >= q["sharpe"]
        q3 = (not c1) and d["mdd"] >= q["mdd"] + 5 and d["cagr"] >= sp["cagr"]
        hard = d["mdd"] > -35
        eras = []
        for a, b in era:
            e = stats(s, a, b)
            e_sp = stats(spy_r, a, b)
            eras.append((a, b, e["cagr"], e["mdd"], e_sp["cagr"], e["cagr"] >= e_sp["cagr"] and e["mdd"] > -35))
        era_ok = all(x[5] for x in eras)
        passed = hard and era_ok and (q2 or q3)
        b0 = st["線E"]
        imp = (d["mdd"] >= b0["mdd"] + 3 and d["cagr"] >= b0["cagr"] - 1) or (d["cagr"] >= b0["cagr"] + 2 and d["mdd"] >= b0["mdd"] - 2)
        win_s = s.loc[START:END].dropna()
        return dict(c1=c1, q2=q2, q3=q3, hard=hard, eras=eras, passed=passed, imp=imp,
                    w5=worst5(win_s), final=float((1 + win_s).prod() * CAP))

    jA, jB = judge("54A", sA), judge("54B", sB)
    dA = evA.assign(版本="54A") if len(evA) else pd.DataFrame(columns=["版本"])
    dB = evB.assign(版本="54B") if len(evB) else pd.DataFrame(columns=["版本"])
    pd.concat([dA, dB], ignore_index=True).to_csv("v54_events.csv", index=False, encoding="utf-8-sig")

    def line_for(name, j, ev):
        d = st[name]
        buys = ev[ev["動作"] == "買入"] if len(ev) else pd.DataFrame()
        sells = ev[ev["動作"] == "賣出"] if len(ev) else pd.DataFrame()
        s = [f"=== {name} ==="]
        s.append(f"　年化 {d['cagr']:+.2f}% ｜ 最大跌幅 {d['mdd']:.1f}% ｜ Sharpe {d['sharpe']:.2f} ｜ 最差5年年化 {j['w5']:+.2f}%")
        s.append(f"　按 80,000 美元本金，2026-09-30 結束資產約 {j['final']:,.0f} 美元")
        s.append(f"　買入 {len(buys)} 次 ｜ 賣出 {len(sells)} 次")
        if len(sells):
            s.append(f"　賣出平均回報 {sells['回報'].mean() * 100:+.1f}% ｜ 賺錢 {(sells['回報'] > 0).mean() * 100:.0f}%")
            for r_, g in sells.groupby("原因"):
                s.append(f"　　{r_}：{len(g)} 次 ｜ 平均 {g['回報'].mean() * 100:+.1f}% ｜ 賺錢 {(g['回報'] > 0).mean() * 100:.0f}%")
        s.append(f"　四條問題：① 年化>QQQ {'是' if j['c1'] else '否'} ｜ ② {'符合' if j['q2'] else '不符合'} ｜ ③ {'符合' if j['q3'] else '不符合'}")
        s.append(f"　硬限制（MDD > −35%）：{'符合' if j['hard'] else '不符合'}")
        for a, b, ec, em, es, ok_ in j["eras"]:
            s.append(f"　跨年代 {a[:4]}–{b[:4]}：年化 {ec:+.2f}%（SPY {es:+.2f}%）｜ MDD {em:.1f}% ⇒ {'符合' if ok_ else '不符合'}")
        s.append(f"　對線E 有改善：{'是' if j['imp'] else '否'}")
        s.append(f"　⇒ 總判斷：{'通過' if j['passed'] else '唔通過'}")
        return s

    L = ["=== 第五十四輪：J Law / XQ VCP 突破原型（美股版）（規則預先登記，見research_v54.py開頭）===",
         f"期間 {START} 至 {END} ｜ 股票池：{source} ｜ 本金 80,000 美元，唔用槓桿", ""]
    L.append("=== 基準（同期）===")
    for nm in ["線E", "SPY", "QQQ"]:
        d = st[nm]
        L.append(f"　{nm}：年化 {d['cagr']:+.2f}% ｜ 最大跌幅 {d['mdd']:.1f}% ｜ Sharpe {d['sharpe']:.2f}")
    L.append("")
    L += line_for("54A", jA, evA) + [""]
    L += line_for("54B", jB, evB) + [""]
    L.append("=== 注意：測試嘅係公開原型腳本，唔係 J Law 本人方法；出場規則係預先定好嘅 ===")
    pd.DataFrame({"report": L}).to_csv("v54_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
