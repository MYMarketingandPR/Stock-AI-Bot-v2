"""
research_v53.py — 第五十三輪：Stockbee 動量爆發（53A 純規則；53B = 53A + 行業 ETF 200日線過濾）
  （Esther 2026-11-12：「用2016-2026年，詳細記錄所有交易資料，本金80000美金」；
    同意：股票池同線E一樣、流動性 $20m、單次風險1%、單隻上限20%、升10%賣一半、5日無升就全賣；
    加入：行業 ETF 價格趨勢過濾（53B）；「同意53輪dispatch去GitHub跑」）

純研究，唔改任何之前結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：

一、測試範圍
  期間：2016-01-01 至 2026-09-30（指標用 2013 年起嘅數據暖身）。
  股票池：當時真實標普500成分股（M.sp500_history，逐月；同線E一樣）。
  本金：80,000 美元（以 1.0 = 80,000 美元計）；唔用槓桿；閒錢收 ^IRX；每次買賣 0.1%。

二、觸發（信號日 D 收市後判斷，只用 D 或之前嘅數據）
  以下三種任一：
    T4 4%突破：收市 ÷ 前收市 − 1 ≥ 4%
    TD 金額突破：收市 − 開市 ≥ 0.90 元
    TR 範圍擴大：D 日價幅（高 − 低）> 前三日價幅嘅最大值，而且前一日升幅 < 4%
  同時必須：
    成交量 > 前一日成交量；
    流動性：過去 20 日平均成交額（收市 × 成交量）≥ 20,000,000 美元。
  大市環境：SPY 收市 > SPY 50 日平均線（唔符合即唔入市，對應 skill「收緊」）。

三、評分（0–100；Skill 冇寫明公式，以下係預先定好嘅草案，只用可以量化嘅部分）
  觸發強度 20：T4 = 20；TD = 12；只有 TR = 10
  成交量 15：D 日成交量 ÷ 前一日 ≥ 2 → 15；≥ 1.5 → 10；其他 → 5
  基礎形態 25：前 10 日平均價幅 ÷ 前 50 日平均價幅 ≤ 0.7 → 25；≤ 1.0 → 12；其他 → 0
  收市位置 10：（收市 − 低）÷（高 − 低）≥ 0.75 → 10；≥ 0.5 → 5；其他 → 0
  止損距離 15：（收市 − 低）÷ 收市 ≤ 3% → 15；≤ 6% → 8；其他 → 0
  失敗篩選 10：前 3 日累計升幅 ≤ 15% 得 5 分；過去 5 日冇任何一日跌 ≥ 4% 得 5 分
  大市環境 5：通過第二節大市條件即得 5 分
  評分 ≥ 70（B 或以上）先入市。

四、入場
  信號日 D 收市判斷，**D + 1 日收市買入**（同線E一致；唔用 D 日收市，避免偷睇當日結果）。
  買入前檢查：D+1 收市價 > 止損位，否則放棄呢個信號。
  最多同時 5 隻。信號多過空位時，按評分由高到低，同分按成交量倍數由高到低。
  如果已持有同一隻股票，唔會再買。

五、止損（Skill 原文：突破日低位）
  止損位 = D 日最低價（固定，唔移動）。
  收市價 < 止損位 → 下一個交易日收市全數賣出。

六、倉位（每單風險 1%，單隻上限 20%，唔用槓桿）
  風險百分比 = (入場價 − 止損位) ÷ 入場價
  倉位比例 = min(1% ÷ 風險百分比, 20%)
  即係：止損被觸發時，每單最多蝕總資產 1%；單隻最多佔總資產 20%。
  5 隻 × 20% = 100%，所以唔會超過全倉。

七、出場（以先觸發者為準；全部下一個交易日收市執行）
  1. 收市 < 止損位 → 全數賣出。
  2. 持有期間某日升幅 ≥ 10%（收市 ÷ 前收市 − 1 ≥ 10%）→ 賣一半（每隻股票只做一次）。
  3. 買入後第 5 個交易日收市，如果收市 ≤ 入場價 → 全數賣出。

八、53B 額外條件（只係 53B 用）
  股票所屬行業 ETF 收市 > 該 ETF 200 日平均線，先可以入市。
  行業分類：Wikipedia / Nasdaq GICS 名稱（research_v24.sectors）→ SPDR 行業 ETF：
    資訊科技 XLK、金融 XLF、能源 XLE、醫療 XLV、工業 XLI、非必需消費 XLY、必需消費 XLP、
    公用事業 XLU、物料 XLB、房地產 XLRE、通訊服務 XLC。
  XLC 2018 年 6 月先有數據：之前嘅通訊股冇 ETF 數據 → 唔入市。
  冇行業分類嘅股票 → 唔入市。

九、基準同通過標準
  基準：線E（同期、同股票池、同 research_v51 嘅 C0 規則）；另列 SPY、QQQ 長揸。
  沿用第47至52輪嘅四條問題（④ 因數據由 2016 年開始，改為跨年代檢查）：
    ① 驗證期年化 > QQQ 年化？
    ② 若 ① 係：最大跌幅 > −35% 且 Sharpe ≥ QQQ Sharpe → 符合；
    ③ 若 ① 唔係：最大跌幅 ≥ QQQ 最大跌幅 + 5 個百分點 且 年化 ≥ SPY 年化 → 符合；
  硬限制：2016–2026 最大跌幅 > −35%。
  跨年代：2016–2019、2020–2026 兩段，每段年化 ≥ SPY 同段年化，而且最大跌幅 > −35%。
  通過 = 硬限制 ✓ 且 跨年代 ✓ 且（② 或 ③ 符合）。
  另外報告（唔影響通過與否）：
    對線E 有改善？(a) 最大跌幅 ≥ 線E + 3 且 年化 ≥ 線E − 1，或 (b) 年化 ≥ 線E + 2 且 最大跌幅 ≥ 線E − 2。
    最差 5 年年化（每月開始，用月底淨值計）。

十、記錄（Esther 要求詳細交易日誌）
  每次買入：日期、股票、觸發類型、評分、成交量倍數、止損位、風險百分比、倉位比例、
    入場價、按 80,000 美元計嘅買入金額同股數。
  每次賣出／部分賣出：日期、股票、原因、買入日、買入價、賣出價、回報、持有日數、賣出比例。
  輸出：v53_events.csv（逐筆）、v53_results.csv（報告）。

已知限制：
  - Skill 嘅文字規則係數字化之後先可以測試，評分公式同門檻係我嘅草案（上面已寫明）。
  - 行業分類用今日 GICS 分類，有前瞻偏差（同第47輪一樣）。
  - 53A 同 53B 同時測試，有「多次測試撞大運」風險，所以兩個版本都要過同一條基準同標準。
  - 價格同成交量來自 Yahoo（auto_adjust），可能有缺漏。
  - 冇盈利增長數據，無法加入 Stockbee 以外嘅 CANSLIM 條件。
  - 信號同 skill 原文入場價（信號日收市）唔同：本輪改用次日收市，較保守。

==== 結果（待跑）====
"""
import time

import numpy as np
import pandas as pd
import yfinance as yf

import momentum_v1 as M
from perf import perf
from research_v13 import get_cash
from research_v24 import sectors
from research_v26 import COST
from research_v51 import sim_stop as line_sim  # 線E（C0）

M.PROGRESS = "v53_progress.txt"
log = M.log
START, END = "2016-01-01", "2026-09-30"
CAP = 80000
N = 5
RISK = 0.01
MAXW = 0.20
LIQ = 20_000_000
SCORE_MIN = 70
TP_RET = 0.10
TIME_DAYS = 5

ETF_MAP = {
    "Information Technology": "XLK", "Technology": "XLK",
    "Financials": "XLF", "Financial": "XLF", "Finance": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV", "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU", "Public Utilities": "XLU",
    "Materials": "XLB", "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC", "Telecommunications": "XLC",
}
ETFS = sorted(set(ETF_MAP.values()))


def dl_ohlcv(tickers, start, end):
    """分批下載（每批 50 隻），用調整後價格。"""
    fields = ["Open", "High", "Low", "Close", "Volume"]
    out = {f: {} for f in fields}
    for j in range(0, len(tickers), 50):
        chunk = tickers[j:j + 50]
        df = None
        for _ in range(3):
            try:
                df = yf.download(chunk, start=start, end=end, auto_adjust=True, progress=False, threads=False)
                break
            except Exception as e:
                log(f"下載失敗 {chunk[:2]}：{e}")
                time.sleep(5)
        if df is not None and not df.empty:
            for f in fields:
                sub = df[f]
                for t in sub.columns:
                    out[f][t] = sub[t]
        log(f"OHLCV 下載 {min(j + 50, len(tickers))}/{len(tickers)}")
        time.sleep(1)
    return {f: pd.DataFrame(out[f]).sort_index() for f in fields}


def worst5(r):
    eq = (1 + r.dropna()).cumprod()
    m = eq.resample("ME").last().dropna()
    if len(m) <= 60:
        return np.nan
    rat = m.values[60:] / m.values[:-60]
    return (rat.min() ** (1 / 5) - 1) * 100


def sim53(label, SIG, SC, VR, TY, Cv, Lv, RETv, days, cr):
    T, C = Cv.shape
    hold, cash = {}, 1.0
    pend_sell, pend_buy = [], []
    navs, ev = [], []
    for i in range(T):
        for c in hold:
            r = RETv[i, c]
            if np.isfinite(r):
                hold[c]["val"] *= 1 + r
        cash *= 1 + cr[i]
        # 1) 執行昨日嘅賣出（今日收市）
        for c, why, frac in pend_sell:
            if c not in hold or not np.isfinite(Cv[i, c]):
                continue
            p = hold[c]
            sv = p["val"] * frac
            proceeds = sv * (1 - COST)
            cost_part = p["cost"] * frac
            cash += proceeds
            p["val"] -= sv
            p["cost"] -= cost_part
            ev.append(dict(動作="賣出" if frac == 1 else "部分賣出", 股票=p["tk"], 日期=str(days[i].date()), 原因=why,
                           買入日=p["date"], 買入價=round(p["entry"], 4), 賣出價=round(float(Cv[i, c]), 4),
                           賣出比例=frac, 回報=proceeds / cost_part - 1, 持有日=i - p["eday"],
                           止損位=round(p["stop"], 4), 按80000美元計賣出金額=sv * CAP, _c=c, _i=i))
            if frac == 1:
                del hold[c]
        pend_sell = []
        # 2) 執行昨日嘅買入（今日收市）
        total = cash + sum(p["val"] for p in hold.values())
        for c, stop, sc, vrat, rk, ty in pend_buy:
            if c in hold or len(hold) >= N or not np.isfinite(Cv[i, c]) or Cv[i, c] <= stop:
                continue
            f = min(RISK / rk, MAXW)
            amt = min(f * total, cash)
            if amt <= 0:
                continue
            cash -= amt
            hold[c] = dict(val=amt * (1 - COST), cost=amt, entry=float(Cv[i, c]), stop=float(stop), eday=i,
                           date=str(days[i].date()), tk=c, partial=False)
            ev.append(dict(動作="買入", 股票=c, 日期=str(days[i].date()), 原因=f"{ty} 觸發，評分 {sc:.0f}",
                           觸發類型=ty, 評分=sc, 成交量倍數=round(vrat, 3), 止損位=round(stop, 4),
                           買入價=round(float(Cv[i, c]), 4), 風險百分比=rk, 倉位比例=f,
                           按80000美元計買入金額=amt * CAP, 股數=amt * CAP / float(Cv[i, c]), _c=c, _i=i))
        pend_buy = []
        navs.append(cash + sum(p["val"] for p in hold.values()))
        # 3) 今日收市判斷出場（明日執行）
        sold = set()
        for c, p in hold.items():
            px = Cv[i, c]
            if not np.isfinite(px):
                continue
            held = i - p["eday"]
            if px < p["stop"]:
                pend_sell.append((c, "收市跌穿止損位（信號日最低價）", 1.0))
                sold.add(c)
            elif (not p["partial"]) and np.isfinite(RETv[i, c]) and RETv[i, c] >= TP_RET:
                pend_sell.append((c, "單日升 ≥10%，賣一半", 0.5))
                p["partial"] = True
            elif held == TIME_DAYS and px <= p["entry"]:
                pend_sell.append((c, "買入後第5個交易日仍未高過入場價", 1.0))
                sold.add(c)
        # 4) 今日信號 → 明日買入候選
        free = N - (len(hold) - len(sold))
        if free > 0:
            idx = [c for c in np.where(SIG[i])[0] if c not in hold]
            idx.sort(key=lambda c: (-SC[i, c], -VR[i, c]))
            for c in idx[:free]:
                rk = (Cv[i, c] - Lv[i, c]) / Cv[i, c]
                if np.isfinite(rk) and rk > 0:
                    pend_buy.append((c, Lv[i, c], SC[i, c], VR[i, c], rk, TY[i, c]))
    nav = np.array(navs)
    strat = pd.Series(np.r_[0.0, nav[1:] / nav[:-1] - 1], index=days)
    df = pd.DataFrame(ev)
    if len(df):
        post_max, post_min = [], []
        for _, e in df.iterrows():
            if e["動作"] in ("賣出", "部分賣出"):
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
    return strat, df, label


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("2013-12-31", END, freq="ME")]))
    tick = sorted(set(everyone) | {"SPY", "QQQ"} | set(ETFS))
    ohlc = dl_ohlcv(tick, "2013-06-01", "2026-10-02")
    days = ohlc["Close"].index
    days = days[days <= pd.Timestamp(END)]
    cl = ohlc["Close"].reindex(days)
    spy_c, qqq_c = cl["SPY"], cl["QQQ"]
    stocks = [t for t in everyone if t in cl.columns and t not in ("SPY", "QQQ")]
    Cdf = cl.reindex(columns=stocks)
    Odf = ohlc["Open"].reindex(index=days, columns=stocks)
    Hdf = ohlc["High"].reindex(index=days, columns=stocks)
    Ldf = ohlc["Low"].reindex(index=days, columns=stocks)
    Vdf = ohlc["Volume"].reindex(index=days, columns=stocks)
    log(f"收市價準備好：{len(stocks)} 隻，{len(days)} 日（{source}）")

    ret1 = Cdf.pct_change(fill_method=None)
    rng = Hdf - Ldf
    rmax = rng.shift(1).combine(rng.shift(2), np.maximum).combine(rng.shift(3), np.maximum)
    T4 = ret1 >= 0.04
    TD = (Cdf - Odf) >= 0.90
    TR = (rng > rmax) & (ret1.shift(1) < 0.04)
    volgate = Vdf > Vdf.shift(1)
    liq = (Cdf * Vdf).rolling(20, min_periods=20).mean() >= LIQ
    trig = (T4 | TD | TR) & volgate & liq

    vr = Vdf / Vdf.shift(1)
    trig_pts = pd.DataFrame(np.where(T4.values, 20, np.where(TD.values, 12, 10)), index=days, columns=stocks)
    vol_pts = pd.DataFrame(np.where(vr.values >= 2, 15, np.where(vr.values >= 1.5, 10, 5)), index=days, columns=stocks)
    cr_base = rng.shift(1).rolling(10, min_periods=10).mean() / rng.shift(1).rolling(50, min_periods=50).mean()
    base_pts = pd.DataFrame(np.select([cr_base.values <= 0.7, cr_base.values <= 1.0], [25, 12], 0), index=days, columns=stocks)
    loc = ((Cdf - Ldf) / (Hdf - Ldf)).replace([np.inf, -np.inf], np.nan)
    close_pts = pd.DataFrame(np.select([loc.values >= 0.75, loc.values >= 0.5], [10, 5], 0), index=days, columns=stocks)
    risk_df = (Cdf - Ldf) / Cdf
    stop_pts = pd.DataFrame(np.select([risk_df.values <= 0.03, risk_df.values <= 0.06], [15, 8], 0), index=days, columns=stocks)
    runup = Cdf.shift(1) / Cdf.shift(4) - 1
    brk5 = ret1.rolling(5, min_periods=5).min()
    fail_pts = 5 * (runup <= 0.15).astype(int) + 5 * (brk5 > -0.04).astype(int)
    m_spy = spy_c > spy_c.rolling(50, min_periods=50).mean()
    score = (trig_pts + vol_pts + base_pts + close_pts + stop_pts + fail_pts.fillna(0)
             + 5 * m_spy.astype(int).values.reshape(-1, 1)).where(trig, 0)
    score = score.fillna(0)
    SCv = score.values.astype(float)
    VRv = vr.fillna(0).values

    cols = stocks
    ix = {t: k for k, t in enumerate(cols)}
    mem = np.zeros((len(days), len(cols)), dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True

    TYv = np.where(T4.values, "4%突破", np.where(TD.values, "金額突破", "範圍擴大"))
    SIG_base = trig.values & (SCv >= SCORE_MIN) & mem & m_spy.values.reshape(-1, 1)

    # 53B：行業 ETF 200日線過濾
    etf_close = cl[ETFS]
    etf_ok = (etf_close > etf_close.rolling(200, min_periods=200).mean()).fillna(False).astype(bool)
    sec = sectors()
    etf_mat = np.zeros((len(days), len(cols)), dtype=bool)
    for k, t in enumerate(cols):
        e = ETF_MAP.get(sec.get(M.norm(t)))
        if e is not None:
            etf_mat[:, k] = etf_ok[e].values
    SIG_B = SIG_base & etf_mat
    log(f"信號數：53A {int(SIG_base.sum())}；53B {int(SIG_B.sum())}")

    cr = get_cash(days).values
    cash_r = get_cash(days)
    spy_r = spy_c.pct_change(fill_method=None).fillna(0.0)
    qqq_r = qqq_c.pct_change(fill_method=None).fillna(0.0)
    RETv = Cdf.pct_change(fill_method=None).values
    Cv = Cdf.values
    Lv = Ldf.values

    sA, evA, _ = sim53("53A", SIG_base, SCv, VRv, TYv, Cv, Lv, RETv, days, cr)
    sB, evB, _ = sim53("53B", SIG_B, SCv, VRv, TYv, Cv, Lv, RETv, days, cr)

    # 線E（同期、同股票池）
    MA200 = Cdf.rolling(200).mean().values
    ye = Cdf.groupby(days.year).last()
    ytdn = np.nan_to_num(Cv / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    HI52 = Cdf.rolling(252, min_periods=200).max().values
    hi52f = np.nan_to_num(HI52, nan=np.inf)
    ok = mem & (ytdn >= 0.30) & np.isfinite(Cv) & (Cv >= hi52f * 0.7) & np.nan_to_num(Cv > MA200, nan=False)
    zero = np.zeros_like(Cv)
    ev_line = []
    line_r = line_sim(Cv, RETv, cr, ok, ytdn, "C0", zero, zero, zero, zero, MA200, HI52, ev_line, "C0 線E出場", days)
    sL = pd.Series(line_r, index=days)

    def win(s):
        return s.loc[START:END].dropna()

    def stats(s, a=START, b=END):
        rr = win(s).loc[a:b]
        return perf(rr, rf=cash_r, bench=qqq_r.loc[a:b])

    st = {"53A": stats(sA), "53B": stats(sB), "線E": stats(sL), "SPY": stats(spy_r), "QQQ": stats(qqq_r)}
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
        return dict(c1=c1, q2=q2, q3=q3, hard=hard, eras=eras, passed=passed, imp=imp, w5=worst5(win(s)), final=float((1 + win(s)).prod() * CAP))

    jA, jB = judge("53A", sA), judge("53B", sB)
    dA = evA.assign(版本="53A") if len(evA) else pd.DataFrame(columns=["版本"])
    dB = evB.assign(版本="53B") if len(evB) else pd.DataFrame(columns=["版本"])
    pd.concat([dA, dB], ignore_index=True).to_csv("v53_events.csv", index=False, encoding="utf-8-sig")

    def line_for(name, j, ev):
        d = st[name]
        sells = ev[ev["動作"].isin(["賣出", "部分賣出"])] if len(ev) else pd.DataFrame()
        buys = ev[ev["動作"] == "買入"] if len(ev) else pd.DataFrame()
        s = [f"=== {name} ==="]
        s.append(f"　年化 {d['cagr']:+.2f}% ｜ 最大跌幅 {d['mdd']:.1f}% ｜ Sharpe {d['sharpe']:.2f} ｜ 最差5年年化 {j['w5']:+.2f}%")
        s.append(f"　按 80,000 美元本金，2026-09-30 結束資產約 {j['final']:,.0f} 美元")
        s.append(f"　買入 {len(buys)} 次 ｜ 賣出／部分賣出 {len(sells)} 次")
        if len(buys):
            for ty, g in buys.groupby("觸發類型"):
                s.append(f"　　{ty}：買入 {len(g)} 次")
        if len(sells):
            s.append(f"　賣出平均回報 {sells['回報'].mean() * 100:+.1f}% ｜ 賺錢 {(sells['回報'] > 0).mean() * 100:.0f}%")
            for r_, g in sells.groupby("原因"):
                s.append(f"　　{r_}：{len(g)} 次 ｜ 平均 {g['回報'].mean() * 100:+.1f}% ｜ 賺錢 {(g['回報'] > 0).mean() * 100:.0f}%")
        s.append("　四條問題：" + f"① 年化>QQQ {'是' if j['c1'] else '否'} ｜ ② {'符合' if j['q2'] else '不符合'} ｜ ③ {'符合' if j['q3'] else '不符合'}")
        s.append(f"　硬限制（MDD > −35%）：{'符合' if j['hard'] else '不符合'}")
        for a, b, ec, em, es, ok_ in j["eras"]:
            s.append(f"　跨年代 {a[:4]}–{b[:4]}：年化 {ec:+.2f}%（SPY {es:+.2f}%）｜ MDD {em:.1f}% ⇒ {'符合' if ok_ else '不符合'}")
        s.append(f"　對線E 有改善：{'是' if j['imp'] else '否'}")
        s.append(f"　⇒ 總判斷：{'通過' if j['passed'] else '唔通過'}")
        return s

    L = ["=== 第五十三輪：Stockbee 動量爆發（規則預先登記，見research_v53.py開頭）===",
         f"期間 {START} 至 {END} ｜ 股票池：{source} ｜ 本金 80,000 美元，唔用槓桿", ""]
    L.append("=== 基準（同期）===")
    for nm in ["線E", "SPY", "QQQ"]:
        d = st[nm]
        L.append(f"　{nm}：年化 {d['cagr']:+.2f}% ｜ 最大跌幅 {d['mdd']:.1f}% ｜ Sharpe {d['sharpe']:.2f}")
    L.append("")
    L += line_for("53A", jA, dA) + [""]
    L += line_for("53B", jB, dB) + [""]
    L.append("=== 注意：兩個版本同時測試，多次測試有撞大運風險；行業分類為今日 GICS（有前瞻偏差）===")
    pd.DataFrame({"report": L}).to_csv("v53_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
