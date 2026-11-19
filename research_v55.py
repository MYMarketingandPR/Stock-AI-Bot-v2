"""
research_v55.py — 第五十五輪：修正與穩健性檢查（第1、2、3、4、6項）
  （Esther 2026-11-19：「將1-6都跑」；線D／E 紙上模擬持續虧損）

性質：修正記錄 + 穩健性檢查 + 探索。唔係通過／唔通過測試。
  不改任何已記錄嘅結果（第47至54輪結果保留原樣）。
  本輪嘅數字用嚟判斷「邊個問題係真實嘅」，唔用嚟揀新規則。

一、修正（第1項）：第54輪 VCP 模擬由 2016-01-01 開始，開始時冇持倉；
    股票欄顯示代號（唔再係編號）；完結時未平倉嘅股票列出，按最後收市價計市值。
    呢個係修正版（54C），同原版 54A／54B 分開記錄。

二、投入比例（第2項）：每條模擬報告平均投入比例（已投入資金 ÷ 總資產，逐日平均）。

三、信號事件研究（第3項）：VCP 信號後 20、60 個交易日嘅回報（以信號日後一日收市入場），
    同同期 SPY 比較（超額回報）。每次「去掉一個條件」都做一次，睇邊個條件有冇用。
    呢個係探索性質，樣本可能好細。

四、成本敏感度（第4項）：線E（C0 出場，2016–2026）用 0.1%、0.2%、0.3% 單邊成本重跑，
    並用 2016-01-01 乾淨開始重跑（冇 2013–2015 嘅持倉帶入）。

六、探索（第6項）：VCP 每次放寬一個條件，睇信號數目同超額回報點變。只係探索，唔可以當成通過。

第5項（留一段數據做最後測試）：2026-09-30 之後嘅紙上模擬本身就係未用過嘅數據，
    規則凍結，唔再用呢段數據調整任何參數。呢個唔需要本輪跑數。

已知限制：
  - 探索結果有多次測試風險。
  - 成本只係固定比例，唔模擬跳空、滑價。
  - 事件研究冇去除重疊信號（同一隻股票連續幾日出信號會重複計）。

輸出：v55_results.csv（報告）、v54c_events.csv（54C 逐筆）、v55_event_study.csv（事件研究）
==== 結果（待跑）====
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v13 import get_cash
import research_v51
from research_v51 import sim_stop
from research_v53 import dl_ohlcv, worst5
from research_v54 import vcp_base, SEG, MAXLAST, MINFIRST, VDRY, VBRK, LOWM, HIGHP, MA200UP, LIQ, RISK, STOP, MAXW, N

M.PROGRESS = "v55_progress.txt"
log = M.log
START, END = "2016-01-01", "2026-09-30"
CAP = 80000
COSTS = [0.001, 0.002, 0.003]


def load():
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("2013-12-31", END, freq="ME")]))
    tick = sorted(set(everyone) | {"SPY", "QQQ"})
    ohlc = dl_ohlcv(tick, "2013-06-01", "2026-10-02")
    days = ohlc["Close"].index
    days = days[days <= pd.Timestamp(END)]
    cl = ohlc["Close"].reindex(index=days)
    stocks = [t for t in everyone if t in cl.columns and t not in ("SPY", "QQQ")]
    D = dict(source=source, days=days, stocks=stocks, spy=cl["SPY"], qqq=cl["QQQ"],
             C=cl.reindex(columns=stocks),
             H=ohlc["High"].reindex(index=days, columns=stocks),
             L=ohlc["Low"].reindex(index=days, columns=stocks),
             V=ohlc["Volume"].reindex(index=days, columns=stocks))
    ix = {t: k for k, t in enumerate(stocks)}
    mem = np.zeros((len(days), len(stocks)), dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True
    D["mem"] = mem
    D["i0"] = int(np.searchsorted(days, pd.Timestamp(START)))
    return D


def line_e_arrays(D):
    Cdf = D["C"]
    Cv = Cdf.values
    days = D["days"]
    MA200 = Cdf.rolling(200).mean().values
    ye = Cdf.groupby(days.year).last()
    ytdn = np.nan_to_num(Cv / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    HI52 = Cdf.rolling(252, min_periods=200).max().values
    hi52f = np.nan_to_num(HI52, nan=np.inf)
    ok = D["mem"] & (ytdn >= 0.30) & np.isfinite(Cv) & (Cv >= hi52f * 0.7) & np.nan_to_num(Cv > MA200, nan=False)
    rets = Cdf.pct_change(fill_method=None).values
    return Cv, rets, ok, ytdn, MA200, HI52


def line_e_run(D, cost, clean=True):
    """線E（C0）；clean=True 由 2016-01-01 開始，開始時冇持倉。"""
    days = D["days"]
    cash_r = get_cash(days)
    cr = cash_r.values
    Cv, rets, ok, ytdn, MA200, HI52 = line_e_arrays(D)
    i0 = D["i0"] if clean else 0
    z = np.zeros_like(Cv)
    old = research_v51.COST
    research_v51.COST = cost
    try:
        r = sim_stop(Cv[i0:], rets[i0:], cr[i0:], ok[i0:], ytdn[i0:], "C0", z[i0:], z[i0:], z[i0:], z[i0:],
                     MA200[i0:], HI52[i0:], [], "C0", days[i0:])
    finally:
        research_v51.COST = old
    s = pd.Series(r, index=days[i0:])
    qqq_r = D["qqq"].pct_change(fill_method=None).fillna(0.0)
    st = perf(s.loc[START:END].dropna(), rf=cash_r, bench=qqq_r.loc[START:END])
    return st


def vcp_parts(D):
    C, H, L, V = D["C"], D["H"], D["L"], D["V"]
    base, ma50 = vcp_base(C, H, L, V)
    # 拆開各條件
    ma50 = C.rolling(50).mean()
    ma150 = C.rolling(150).mean()
    ma200 = C.rolling(200).mean()
    low52 = L.rolling(250, min_periods=250).min()
    high52 = H.rolling(250, min_periods=250).max()
    trend = ((C > ma150) & (C > ma200) & (ma150 > ma200) & (ma200 > ma200.shift(MA200UP))
             & (C > ma50) & (ma50 > ma150) & (ma50 > ma200)
             & (low52 > 0) & (C >= low52 * LOWM) & (high52 > 0) & (C >= high52 * (1 - HIGHP / 100)))

    def seg(k):
        return (H.shift(k).rolling(SEG, min_periods=SEG).max(), L.shift(k).rolling(SEG, min_periods=SEG).min())

    h3, l3 = seg(1)
    h2, l2 = seg(SEG + 1)
    h1, l1 = seg(2 * SEG + 1)
    d1 = (h1 - l1) / h1 * 100
    d2 = (h2 - l2) / h2 * 100
    d3 = (h3 - l3) / h3 * 100
    av10 = V.shift(1).rolling(10, min_periods=10).mean()
    av50 = V.shift(1).rolling(50, min_periods=50).mean()
    parts = dict(
        趨勢樣板=trend.fillna(False),
        d1_至少10=(d1 >= MINFIRST).fillna(False),
        d1大於d2=(d1 > d2).fillna(False),
        d2大於d3=(d2 > d3).fillna(False),
        末段細過10=(d3 <= MAXLAST).fillna(False),
        量縮=((av50 > 0) & (av10 / av50 < VDRY)).fillna(False),
        收市越過樞紐點=((C > h3) & (C.shift(1) <= h3.shift(1))).fillna(False),
        放量1_5倍=(V > av50 * VBRK).fillna(False),
        流動性=((C * V).rolling(20, min_periods=20).mean() >= LIQ).fillna(False),
    )
    return parts


def event_study(D, parts, drop=None):
    days = D["days"]
    C = D["C"].values
    spy = D["spy"].values
    keys = [k for k in parts if k != drop]
    m = D["mem"].copy()
    for k in keys:
        m &= parts[k].values
    W = ((days >= pd.Timestamp(START)) & (days <= pd.Timestamp(END)))[:, None]
    m &= W
    T = len(days)
    out = {}
    for h in (20, 60):
        # 入場 = 信號後一日收市；出場 = 入場後 h 日收市
        entry = np.full(C.shape, np.nan)
        entry[:T - 1] = C[1:]
        exitp = np.full(C.shape, np.nan)
        exitp[:T - h - 1] = C[h + 1:]
        fwd = exitp / entry - 1
        spy_entry = np.full(T, np.nan)
        spy_entry[:T - 1] = spy[1:]
        spy_exit = np.full(T, np.nan)
        spy_exit[:T - h - 1] = spy[h + 1:]
        spy_f = spy_exit / spy_entry - 1
        ex = fwd - spy_f.reshape(-1, 1)
        vals = ex[m & np.isfinite(ex)]
        out[h] = dict(n=int(vals.size), mean=float(np.mean(vals)) * 100 if vals.size else np.nan,
                      median=float(np.median(vals)) * 100 if vals.size else np.nan,
                      hit=float(np.mean(vals > 0)) * 100 if vals.size else np.nan)
    return out


def vcp_run54c(D, SIG, RK, Cv, RETv, MA50v, label):
    """54C：同 54A／54B 一樣嘅規則，由 2016 開始，股票用代號，完結時列出未平倉。"""
    days = D["days"]
    names = D["stocks"]
    i0 = D["i0"]
    Cv = Cv[i0:]
    RETv = RETv[i0:]
    MA50v = MA50v[i0:]
    SIG = SIG[i0:]
    RK = RK[i0:]
    dd = days[i0:]
    cr = get_cash(days).values[i0:]
    T, C = Cv.shape
    hold, cash = {}, 1.0
    pend_sell, pend_buy = [], []
    navs, expo, ev = [], [], []
    f = min(RISK / STOP, MAXW)
    for i in range(T):
        for c in hold:
            r = RETv[i, c]
            if np.isfinite(r):
                hold[c]["val"] *= 1 + r
        cash *= 1 + cr[i]
        for c, why in pend_sell:
            if c not in hold or not np.isfinite(Cv[i, c]):
                continue
            p = hold.pop(c)
            proceeds = p["val"] * (1 - 0.001)
            cash += proceeds
            ev.append(dict(動作="賣出", 股票=names[c], 日期=str(dd[i].date()), 原因=why, 買入日=p["date"],
                           買入價=round(p["entry"], 4), 賣出價=round(float(Cv[i, c]), 4),
                           回報=proceeds / p["cost"] - 1, 持有日=i - p["eday"],
                           按80000美元計賣出金額=proceeds * CAP))
        pend_sell = []
        total = cash + sum(p["val"] for p in hold.values())
        for c, rk in pend_buy:
            if c in hold or len(hold) >= N or not np.isfinite(Cv[i, c]):
                continue
            amt = min(f * total, cash)
            if amt <= 0:
                continue
            cash -= amt
            hold[c] = dict(val=amt * (1 - 0.001), cost=amt, entry=float(Cv[i, c]), eday=i, date=str(dd[i].date()))
            ev.append(dict(動作="買入", 股票=names[c], 日期=str(dd[i].date()), 原因="VCP 放量突破樞紐點",
                           買入價=round(float(Cv[i, c]), 4), 止損價=round(float(Cv[i, c]) * (1 - STOP), 4),
                           倉位比例=f, 按80000美元計買入金額=amt * CAP, 股數=amt * CAP / float(Cv[i, c]),
                           相對強度250日=rk))
        pend_buy = []
        tot_now = cash + sum(p["val"] for p in hold.values())
        navs.append(tot_now)
        expo.append(sum(p["val"] for p in hold.values()) / tot_now if tot_now > 0 else 0.0)
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
        free = N - (len(hold) - len(sold))
        if free > 0:
            idx = [c for c in np.where(SIG[i])[0] if c not in hold]
            idx.sort(key=lambda c: -RK[i, c])
            pend_buy = [(c, RK[i, c]) for c in idx[:free]]
    # 完結時未平倉：按最後收市價計
    last = T - 1
    for c, p in hold.items():
        px = Cv[last, c]
        ev.append(dict(動作="未平倉（按最後收市價）", 股票=names[c], 日期=str(dd[last].date()), 原因="完結時仍持有",
                       買入日=p["date"], 買入價=round(p["entry"], 4), 賣出價=round(float(px), 4),
                       回報=p["val"] / p["cost"] - 1, 持有日=last - p["eday"]))
    nav = np.array(navs)
    strat = pd.Series(np.r_[0.0, nav[1:] / nav[:-1] - 1], index=dd)
    return strat, pd.DataFrame(ev), float(np.mean(expo)) * 100


def main():
    open(M.PROGRESS, "w").close()
    D = load()
    days, stocks = D["days"], D["stocks"]
    log(f"資料準備好：{len(stocks)} 隻（{D['source']}）")
    out = []
    out.append("=== 第五十五輪：修正與穩健性檢查（規則唔改；探索唔係通過測試）===")
    out.append("")

    # 四：線E 成本敏感度（2016 乾淨開始）
    out.append("=== 第4項：線E（C0）成本敏感度，2016-01-01 乾淨開始 ===")
    for c in COSTS:
        st = line_e_run(D, c, clean=True)
        out.append(f"　單邊成本 {c * 100:.1f}%：年化 {st['cagr']:+.2f}% ｜ 最大跌幅 {st['mdd']:.1f}% ｜ Sharpe {st['sharpe']:.2f}")
    st_old = line_e_run(D, 0.001, clean=False)
    out.append(f"　對照（原版，由 2013 開始、帶入持倉）：年化 {st_old['cagr']:+.2f}% ｜ 最大跌幅 {st_old['mdd']:.1f}%")
    out.append("")

    # 一、二：54C 修正版 + 投入比例
    Cv = D["C"].values
    RETv = D["C"].pct_change(fill_method=None).values
    parts = vcp_parts(D)
    base, ma50 = vcp_base(D["C"], D["H"], D["L"], D["V"])
    stk250 = D["C"] / D["C"].shift(250) - 1
    spy_c = D["spy"]
    spy250 = spy_c / spy_c.shift(250) - 1
    rs_ok = stk250.ge(spy250, axis=0).fillna(False).values
    SIG_base = base.values & D["mem"]
    RK = stk250.fillna(-9).values
    MA50v = ma50.values
    cr_full = get_cash(days).values
    strat_A, evA, expoA = vcp_run54c(D, SIG_base & rs_ok, RK, Cv, RETv, MA50v, "54C-A")
    strat_B, evB, expoB = vcp_run54c(D, SIG_base, RK, Cv, RETv, MA50v, "54C-B")
    cash_r = get_cash(days)
    qqq_r = D["qqq"].pct_change(fill_method=None).fillna(0.0)
    spy_r = D["spy"].pct_change(fill_method=None).fillna(0.0)
    out.append("=== 第1、2項：54C 修正版（2016 乾淨開始，代號顯示，完結時列未平倉）===")
    for nm, s, ex in [("54C-A（開相對強度）", strat_A, expoA), ("54C-B（關相對強度）", strat_B, expoB)]:
        ss = s.loc[START:END].dropna()
        st = perf(ss, rf=cash_r, bench=qqq_r.loc[START:END])
        out.append(f"　{nm}：年化 {st['cagr']:+.2f}% ｜ 最大跌幅 {st['mdd']:.1f}% ｜ Sharpe {st['sharpe']:.2f} ｜ 平均投入 {ex:.1f}%")
    evA.assign(版本="54C-A").to_csv("v54c_events.csv", index=False, encoding="utf-8-sig")
    evB.assign(版本="54C-B").to_csv("v54c_events_B.csv", index=False, encoding="utf-8-sig")
    out.append("")

    # 三、六：事件研究 + 探索
    out.append("=== 第3、6項：VCP 信號事件研究（信號後一日入場；超額回報 = 股票 − SPY；2016–2026）===")
    out.append("　（每行：去掉某個條件後嘅信號數、20日超額回報平均／中位數／勝率、60日超額回報平均）")
    rows = []
    drops = [None] + list(parts.keys())
    for dr in drops:
        es = event_study(D, parts, drop=dr)
        label = "全部條件（原版）" if dr is None else f"去掉：{dr}"
        rows.append(dict(條件=label, 信號數=es[20]["n"], 二十日超額平均=es[20]["mean"], 二十日超額中位=es[20]["median"],
                         二十日勝率=es[20]["hit"], 六十日超額平均=es[60]["mean"]))
        out.append(f"　{label}：{es[20]['n']} 次 ｜ 20日 平均 {es[20]['mean']:+.2f}% 中位 {es[20]['median']:+.2f}% 勝率 {es[20]['hit']:.0f}% ｜ 60日 平均 {es[60]['mean']:+.2f}%")
    pd.DataFrame(rows).to_csv("v55_event_study.csv", index=False, encoding="utf-8-sig")
    out.append("")
    out.append("注意：探索性質，樣本可能重疊；唔可以當成通過測試。")

    pd.DataFrame({"report": out}).to_csv("v55_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(out))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
