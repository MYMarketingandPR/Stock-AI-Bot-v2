"""
research_v54_diag.py — 第54輪診斷（純檢查，唔改任何規則、唔改任何已記錄結果）
  目的：
    1. 將第54輪交易記錄入面嘅股票編號，對返股票代號；
    2. 逐個條件計算通過次數（漏斗），睇信號少係咪因為某個條件特別嚴，或者程式有錯。
  輸出：v54_diag.txt（報告）、v54_diag_signals.csv（所有符合信號）、v54_map.csv（編號對代號）
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v53 import dl_ohlcv
from research_v54 import SEG, MAXLAST, MINFIRST, VDRY, VBRK, LOWM, HIGHP, MA200UP, LIQ

M.PROGRESS = "v54_diag_progress.txt"
log = M.log
START, END = "2016-01-01", "2026-09-30"

# 第54輪 54A 買入記錄（編號, 日期），來自 v54_events.csv
EVENTS = [(209, "2015-01-05"), (342, "2015-03-05"), (706, "2015-06-08"), (146, "2016-04-28"),
          (184, "2017-04-21"), (562, "2019-04-17"), (648, "2020-02-12"), (684, "2020-08-10"),
          (121, "2021-01-07"), (474, "2021-05-05"), (349, "2022-08-05"), (210, "2024-03-01"),
          (273, "2026-01-09")]


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("2013-12-31", END, freq="ME")]))
    tick = sorted(set(everyone) | {"SPY", "QQQ"})
    ohlc = dl_ohlcv(tick, "2013-06-01", "2026-10-02")
    days = ohlc["Close"].index
    days = days[days <= pd.Timestamp(END)]
    cl = ohlc["Close"].reindex(index=days)
    stocks = [t for t in everyone if t in cl.columns and t not in ("SPY", "QQQ")]
    Cdf = cl.reindex(columns=stocks)
    Hdf = ohlc["High"].reindex(index=days, columns=stocks)
    Ldf = ohlc["Low"].reindex(index=days, columns=stocks)
    Vdf = ohlc["Volume"].reindex(index=days, columns=stocks)
    pd.DataFrame({"編號": range(len(stocks)), "代號": stocks}).to_csv("v54_map.csv", index=False, encoding="utf-8-sig")
    log(f"編號對照：{len(stocks)} 隻（{source}）")

    # 條件（同 research_v54.vcp_base 一樣）
    ma50 = Cdf.rolling(50).mean()
    ma150 = Cdf.rolling(150).mean()
    ma200 = Cdf.rolling(200).mean()
    low52 = Ldf.rolling(250, min_periods=250).min()
    high52 = Hdf.rolling(250, min_periods=250).max()
    trend = ((Cdf > ma150) & (Cdf > ma200) & (ma150 > ma200) & (ma200 > ma200.shift(MA200UP))
             & (ma50 > ma150) & (ma50 > ma200) & (Cdf > ma50)
             & (low52 > 0) & (Cdf >= low52 * LOWM) & (high52 > 0) & (Cdf >= high52 * (1 - HIGHP / 100)))

    def seg(k):
        return (Hdf.shift(k).rolling(SEG, min_periods=SEG).max(),
                Ldf.shift(k).rolling(SEG, min_periods=SEG).min())

    h3, l3 = seg(1)
    h2, l2 = seg(SEG + 1)
    h1, l1 = seg(2 * SEG + 1)
    d1 = (h1 - l1) / h1 * 100
    d2 = (h2 - l2) / h2 * 100
    d3 = (h3 - l3) / h3 * 100
    c_first = d1 >= MINFIRST
    c_12 = d1 > d2
    c_23 = d2 > d3
    c_last = d3 <= MAXLAST
    contract = c_first & c_12 & c_23 & c_last
    av10 = Vdf.shift(1).rolling(10, min_periods=10).mean()
    av50 = Vdf.shift(1).rolling(50, min_periods=50).mean()
    voldry = (av50 > 0) & (av10 / av50 < VDRY)
    pivot = h3
    crossup = (Cdf > pivot) & (Cdf.shift(1) <= pivot.shift(1))
    volbrk = Vdf > av50 * VBRK
    liq = (Cdf * Vdf).rolling(20, min_periods=20).mean() >= LIQ

    ix = {t: k for k, t in enumerate(stocks)}
    mem = np.zeros((len(days), len(stocks)), dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True
    W = ((days >= pd.Timestamp(START)) & (days <= pd.Timestamp(END)))[:, None]
    M_ = mem & W

    def fv(x):
        return x.fillna(False).values.astype(bool)

    conds = [("趨勢樣板", fv(trend)), ("收縮 d1≥10%", fv(c_first)), ("收縮 d1>d2", fv(c_12)),
             ("收縮 d2>d3", fv(c_23)), ("末段 d3≤10%", fv(c_last)), ("量縮", fv(voldry)),
             ("收市越過樞紐點", fv(crossup)), ("放量1.5倍", fv(volbrk)), ("流動性$20m", fv(liq))]

    L = ["=== 第54輪診斷：條件漏斗（2016–2026，只計成員股票日）===",
         f"成員股票日總數：{int(M_.sum())}", ""]
    L.append("--- 單獨條件通過次數（每個條件獨立計）---")
    for name, arr in conds:
        L.append(f"　{name}：{int((arr & M_).sum())}")
    L.append("")
    L.append("--- 逐步疊加（由上至下，累計同時符合）---")
    acc = M_.copy()
    for name, arr in conds:
        acc = acc & arr
        L.append(f"　+ {name}：剩 {int(acc.sum())}")
    L.append("")

    # 對照第54輪買入記錄
    L.append("=== 第54輪 54A 買入記錄對照（編號 → 代號，並檢查當日各條件）===")
    for idx, dstr in EVENTS:
        t = stocks[idx] if idx < len(stocks) else "?"
        i = int(np.searchsorted(days, pd.Timestamp(dstr)))
        if i >= len(days) or idx >= len(stocks):
            L.append(f"　{idx} {dstr}：超出範圍")
            continue
        flags = " ".join(f"{name}={'Y' if arr[i, idx] else 'n'}" for name, arr in conds)
        L.append(f"　{idx} → {t}　{dstr}：{flags}")
    L.append("")

    sig = acc
    rows = []
    for i, c in zip(*np.where(sig)):
        rows.append((str(days[i].date()), stocks[c], int(c)))
    pd.DataFrame(rows, columns=["日期", "代號", "編號"]).to_csv("v54_diag_signals.csv", index=False, encoding="utf-8-sig")
    L.append(f"=== 全部符合信號：{len(rows)} 次（見 v54_diag_signals.csv）===")
    for r in rows:
        L.append(f"　{r[0]}　{r[1]}")

    pd.DataFrame({"report": L}).to_csv("v54_diag.csv", index=False, encoding="utf-8-sig")
    open("v54_diag.txt", "w", encoding="utf-8").write("\n".join(L))
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
