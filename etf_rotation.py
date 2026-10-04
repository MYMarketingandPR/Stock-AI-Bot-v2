"""
etf_rotation.py — 板塊ETF輪動 + SPY槓桿(1x/2x/3x)研究

⚠️ 預先登記：測試範圍、計法、評分方法喺第一次跑之前已經定好。
   我哋會試好多組合（用戶要求「唔好被日數限死」），所以全部組合都會列出，
   而且用「前半段揀、後半段驗證」去分辨真本事同撞彩。

一、板塊ETF輪動
- ETF：9隻SPDR板塊ETF（1998年12月上市）：XLB物料、XLE能源、XLF金融、XLI工業、XLK科技、
  XLP必需消費、XLU公用、XLV醫療、XLY非必需消費。（XLRE 2015、XLC 2018先上市，唔加入，保持前後一致）
- 測試期：2000-01 至 2026-09（1999年用嚟計第一次嘅回望期）。
- 每個月最後一個交易日收市後計分，下一個交易日收市先交易（遲一日，比較貼近真實）。
- 組合（全部都試，共160個）：
  回望期（睇過去幾多個月表現）：1、2、3、4、5、6、9、12個月
  持有幾多隻：1、2、3、4、5隻（平均分配）
  絕對動量濾網：有/冇（有 = 嗰隻ETF回望期回報要高過同期國債，否則嗰份錢放現金）
  大市濾網：有/冇（有 = SPY跌穿200日線就全部轉現金）
- 現金收益：13週美國國債息率(^IRX)。交易成本：每邊0.1%。持倉每日跟價格浮動（唔係每日再平衡）。
- 比較對象：SPY買入持有、9隻板塊平均分配（每月再平衡）、SPY+200日線濾網。
- 前後兩段：前半2000-2012（有科網爆破、2008金融海嘯）；後半2013-2026/09。
  「前半揀」：用前半段回報最高嘅組合，原封不動睇佢後半段表現 —— 呢個先係誠實嘅答案。

二、SPY槓桿
- 1993-02 至 2026-09，用SPY每日回報模擬：
  2倍、3倍：每日回報 × 槓桿 − 借錢成本((槓桿−1) × 國債息) − 管理費每年0.9%
  3倍 + 200日線濾網：SPY收市高過200日線先持有3倍，否則現金（同樣遲一日交易，轉倉收0.1% × 3）
- ⚠️ 2009年之前市場上冇3倍SPY ETF，嗰段係模擬；真實產品仲有追蹤誤差。
"""
import numpy as np
import pandas as pd
import yfinance as yf

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
DL_START = "1992-06-01"
END = "2026-10-01"
TEST_START = "2000-01-01"
SPLIT_END = "2012-12-31"
LEV_START = "1993-02-01"
LOOKBACKS = [1, 2, 3, 4, 5, 6, 9, 12]
TOPN = [1, 2, 3, 4, 5]
ABS_FILTER = [False, True]
MKT_FILTER = [False, True]
COST = 0.001
LEV_FEE = 0.009
OUT_REPORT = "etf_results.csv"
OUT_GRID = "etf_grid.csv"
OUT_EQUITY = "etf_equity.csv"


def close_of(t, start):
    df = yf.download(t, start=start, end=END, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].astype(float).dropna()


def load():
    spy = close_of("SPY", DL_START)
    idx = spy.index
    px = pd.DataFrame({t: close_of(t, DL_START) for t in SECTORS}).reindex(idx)
    irx = close_of("^IRX", DL_START).reindex(idx).ffill().fillna(0.0)
    cash_r = irx / 100 / 252
    return idx, spy, px, cash_r


def run_rotation(idx, px, spy, cash_r, lookback, topn, use_abs, use_mkt, start):
    """每日模擬，持倉跟價格浮動，月尾收市計分、下一日收市交易。"""
    R = px.pct_change().fillna(0.0).values
    cr = cash_r.values
    month_end = px.groupby(idx.to_period("M")).tail(1).index
    me_px = px.loc[month_end]
    cash_cum = (1 + cash_r).cumprod()
    me_cash = cash_cum.loc[month_end]
    spy_ok = (spy > spy.rolling(200).mean())
    targets = {}
    me_list = list(month_end)
    for k, s in enumerate(me_list):
        if k < lookback:
            continue
        past = me_list[k - lookback]
        lb = me_px.loc[s] / me_px.loc[past] - 1
        lb = lb.dropna()
        if lb.empty:
            continue
        chosen = lb.sort_values(ascending=False).index[:topn]
        w = pd.Series(0.0, index=SECTORS)
        each = 1.0 / topn
        cash_lb = me_cash.loc[s] / me_cash.loc[past] - 1
        for t in chosen:
            if use_abs and lb[t] <= cash_lb:
                continue
            w[t] = each
        if use_mkt and not bool(spy_ok.loc[s]):
            w[:] = 0.0
        pos = idx.get_loc(s)
        if pos + 1 < len(idx):
            targets[pos + 1] = w.values
    start_i = idx.searchsorted(pd.Timestamp(start))
    h = np.zeros(len(SECTORS))
    cash = 1.0
    eq = np.full(len(idx), np.nan)
    trades = 0
    for d in range(start_i, len(idx)):
        h *= (1 + R[d])
        cash *= (1 + cr[d])
        if d in targets:
            total = h.sum() + cash
            tgt = total * targets[d]
            turnover = np.abs(tgt - h).sum()
            if turnover > 1e-9:
                trades += 1
            total -= turnover * COST
            h = total * targets[d]
            cash = total - h.sum()
        eq[d] = h.sum() + cash
    return pd.Series(eq, index=idx).dropna(), trades


def equal_weight(idx, px, cash_r, start):
    w = np.full(len(SECTORS), 1 / len(SECTORS))
    R = px.pct_change().fillna(0.0).values
    month_first = set(idx.get_indexer(px.groupby(idx.to_period("M")).head(1).index))
    start_i = idx.searchsorted(pd.Timestamp(start))
    h, cash = w.copy(), 0.0
    eq = np.full(len(idx), np.nan)
    for d in range(start_i, len(idx)):
        h *= (1 + R[d])
        if d in month_first:
            total = h.sum()
            total -= np.abs(total * w - h).sum() * COST
            h = total * w
        eq[d] = h.sum() + cash
    return pd.Series(eq, index=idx).dropna()


def leveraged(spy, cash_r, lev, use_filter, start):
    r = spy.pct_change().fillna(0.0)
    daily = lev * r - (lev - 1) * cash_r - (LEV_FEE / 252 if lev > 1 else 0.0)
    if use_filter:
        ok = (spy > spy.rolling(200).mean()).astype(float)
        inm = ok.shift(2).fillna(0.0)                    # 收市判斷，下一日收市交易，再下一日先食到回報
        switch = inm.diff().abs().fillna(0.0)
        daily = inm * daily + (1 - inm) * cash_r - switch * COST * lev
    daily = daily[daily.index >= pd.Timestamp(start)]
    return (1 + daily).cumprod()


def stats(eq, a=None, b=None):
    s = eq
    if a is not None:
        before = eq[eq.index < pd.Timestamp(a)]
        base = float(before.iloc[-1]) if len(before) else float(eq.iloc[0])
        s = eq[(eq.index >= pd.Timestamp(a)) & (eq.index <= pd.Timestamp(b))]
        base_date = before.index[-1] if len(before) else s.index[0]
    else:
        base, base_date = float(eq.iloc[0]), eq.index[0]
    yrs = (s.index[-1] - base_date).days / 365.25
    cagr = ((float(s.iloc[-1]) / base) ** (1 / yrs) - 1) * 100
    v = np.concatenate([[base], s.values])
    mdd = float((v / np.maximum.accumulate(v) - 1).min() * 100)
    vol = float(s.pct_change().std() * np.sqrt(252) * 100)
    yearly = s.groupby(s.index.year).last()
    prev = np.concatenate([[base], yearly.values[:-1]])
    yr_ret = (yearly.values / prev - 1) * 100
    return {"cagr": cagr, "mdd": mdd, "vol": vol, "worst_year": float(yr_ret.min()),
            "final_mult": float(s.iloc[-1]) / base, "yearly": dict(zip(yearly.index, yr_ret))}


def fmt(name, st):
    return (f"　{name}：每年 {st['cagr']:+.2f}%，最大跌幅 {st['mdd']:.1f}%，最差一年 {st['worst_year']:+.1f}%，"
            f"$80,000 → ${80000 * st['final_mult']:,.0f}")


def main():
    idx, spy, px, cash_r = load()
    lines = ["=== 板塊ETF輪動 + SPY槓桿研究（設計預先登記，見etf_rotation.py開頭）===",
             f"數據：SPY由{spy.index[0].date()}，板塊ETF由{px.dropna(how='all').index[0].date()}；"
             f"現金收益用13週國債；交易成本每邊{COST * 100:.1f}%", ""]

    spy_eq = (1 + spy.pct_change().fillna(0.0))[spy.index >= pd.Timestamp(TEST_START)].cumprod()
    ew_eq = equal_weight(idx, px, cash_r, TEST_START)
    spyf_eq = leveraged(spy, cash_r, 1, True, TEST_START)
    periods = [("全期 2000-2026/09", None, None), ("前半 2000-2012", TEST_START, SPLIT_END),
               ("後半 2013-2026/09", "2013-01-01", "2026-09-30")]

    lines.append("=== 比較對象 ===")
    bench = {"SPY買入持有": spy_eq, "9隻板塊平均分配": ew_eq, "SPY+200日線濾網": spyf_eq}
    for pname, a, b in periods:
        lines.append(f"--- {pname} ---")
        for n, e in bench.items():
            lines.append(fmt(n, stats(e, a, b)))
    lines.append("")

    rows, curves = [], {}
    for lb in LOOKBACKS:
        for n in TOPN:
            for ab in ABS_FILTER:
                for mk in MKT_FILTER:
                    eq, trades = run_rotation(idx, px, spy, cash_r, lb, n, ab, mk, TEST_START)
                    key = f"回望{lb}個月/持{n}隻/絕對動量{'有' if ab else '冇'}/大市濾網{'有' if mk else '冇'}"
                    curves[key] = eq
                    full, h1, h2 = stats(eq), stats(eq, TEST_START, SPLIT_END), stats(eq, "2013-01-01", "2026-09-30")
                    rows.append({"組合": key, "回望月": lb, "持有隻數": n, "絕對動量": ab, "大市濾網": mk,
                                 "全期每年%": round(full["cagr"], 2), "全期最大跌幅%": round(full["mdd"], 1),
                                 "全期最差一年%": round(full["worst_year"], 1), "波動%": round(full["vol"], 1),
                                 "前半每年%": round(h1["cagr"], 2), "前半最大跌幅%": round(h1["mdd"], 1),
                                 "後半每年%": round(h2["cagr"], 2), "後半最大跌幅%": round(h2["mdd"], 1),
                                 "換倉次數": trades})
    grid = pd.DataFrame(rows)
    sp = {p: stats(spy_eq, a, b)["cagr"] if a else stats(spy_eq)["cagr"] for p, a, b in periods}
    grid["全期贏SPY"] = grid["全期每年%"] > sp["全期 2000-2026/09"]
    grid["前半贏SPY"] = grid["前半每年%"] > sp["前半 2000-2012"]
    grid["後半贏SPY"] = grid["後半每年%"] > sp["後半 2013-2026/09"]
    grid.sort_values("全期每年%", ascending=False).to_csv(OUT_GRID, index=False, encoding="utf-8-sig")

    lines.append(f"=== 160個組合整體情況（SPY：全期{sp['全期 2000-2026/09']:+.2f}%、前半{sp['前半 2000-2012']:+.2f}%、"
                 f"後半{sp['後半 2013-2026/09']:+.2f}%）===")
    for col, p in (("全期", "全期 2000-2026/09"), ("前半", "前半 2000-2012"), ("後半", "後半 2013-2026/09")):
        lines.append(f"　{col}：{int(grid[col + '贏SPY'].sum())}/160個組合贏SPY；組合每年回報中位數 {grid[col + '每年%'].median():+.2f}%")
    both = grid[grid["前半贏SPY"] & grid["後半贏SPY"]]
    lines.append(f"　前後兩半都贏SPY：{len(both)}/160個")
    lines.append("")

    lines.append("=== 各個設定嘅平均表現（將其他設定平均，睇邊個方向穩定）===")
    for col, name in (("回望月", "回望期"), ("持有隻數", "持有隻數"), ("絕對動量", "絕對動量濾網"), ("大市濾網", "大市濾網")):
        g = grid.groupby(col)[["前半每年%", "後半每年%", "全期每年%", "全期最大跌幅%"]].mean().round(2)
        lines.append(f"--- {name} ---")
        for k, r in g.iterrows():
            lines.append(f"　{k}：前半{r['前半每年%']:+.2f}% | 後半{r['後半每年%']:+.2f}% | 全期{r['全期每年%']:+.2f}% | 平均最大跌幅{r['全期最大跌幅%']:.1f}%")
    lines.append("")

    pick = grid.sort_values("前半每年%", ascending=False).iloc[0]
    rank2 = int((grid["後半每年%"] > pick["後半每年%"]).sum()) + 1
    lines.append("=== 誠實測試：用前半段(2000-2012)回報最高嘅組合，原封不動睇後半段 ===")
    lines.append(f"　前半最佳：{pick['組合']}，前半每年{pick['前半每年%']:+.2f}%")
    lines.append(f"　→ 後半(2013-2026/09)：每年{pick['後半每年%']:+.2f}%（SPY {sp['後半 2013-2026/09']:+.2f}%），"
                 f"後半排第{rank2}/160，後半最大跌幅{pick['後半最大跌幅%']:.1f}%")
    lines.append("")

    top = grid.sort_values("全期每年%", ascending=False).head(10)
    lines.append("=== 全期回報最高10個組合（⚠️事後揀，一定偏靚，只作參考）===")
    for _, r in top.iterrows():
        lines.append(f"　{r['組合']}：全期{r['全期每年%']:+.2f}%（最大跌幅{r['全期最大跌幅%']:.1f}%）| "
                     f"前半{r['前半每年%']:+.2f}% | 後半{r['後半每年%']:+.2f}%")
    lines.append("")

    lines.append("=== SPY槓桿（1993-02起；2009年前係模擬）===")
    lev = {"SPY 1倍": leveraged(spy, cash_r, 1, False, LEV_START),
           "SPY 2倍": leveraged(spy, cash_r, 2, False, LEV_START),
           "SPY 3倍": leveraged(spy, cash_r, 3, False, LEV_START),
           "SPY 3倍 + 200日線濾網": leveraged(spy, cash_r, 3, True, LEV_START),
           "SPY 1倍 + 200日線濾網": leveraged(spy, cash_r, 1, True, LEV_START)}
    for pname, a, b in (("1993-02 至 2026-09", None, None), ("2000-2012", TEST_START, SPLIT_END),
                        ("2013-2026/09", "2013-01-01", "2026-09-30")):
        lines.append(f"--- {pname} ---")
        for n, e in lev.items():
            lines.append(fmt(n, stats(e, a, b)))
    lines.append("")

    yr_names = {"SPY": spy_eq, "前半最佳輪動": curves[pick["組合"]], "9隻平均": ew_eq,
                "SPY 3倍": lev["SPY 3倍"], "3倍+濾網": lev["SPY 3倍 + 200日線濾網"]}
    yrs = {n: stats(e[e.index >= pd.Timestamp(TEST_START)] / e[e.index >= pd.Timestamp(TEST_START)].iloc[0])["yearly"]
           for n, e in yr_names.items()}
    lines.append("=== 逐年回報 ===")
    lines.append("　年份 | " + " | ".join(yr_names))
    for y in sorted(yrs["SPY"]):
        lines.append(f"　{y} | " + " | ".join(f"{yrs[n].get(y, float('nan')):+.1f}%" for n in yr_names))

    pd.DataFrame({"report": lines}).to_csv(OUT_REPORT, index=False, encoding="utf-8-sig")
    out = pd.DataFrame({n: e for n, e in {**bench, **{"前半最佳輪動": curves[pick["組合"]]}, **lev}.items()})
    out.round(5).to_csv(OUT_EQUITY, encoding="utf-8-sig")
    print("\n".join(lines))
    print(f"\n✅ 已存到 {OUT_REPORT}、{OUT_GRID}、{OUT_EQUITY}")


if __name__ == "__main__":
    main()
