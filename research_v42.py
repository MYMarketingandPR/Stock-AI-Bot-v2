"""
research_v42.py — 第四十二輪：股價升到 200日線 2.6 倍或以上，之後跌返落 2.6 倍以下，跟住會點？（事件研究）
  Esther 2026-10-26：「當股價去至 SMA200 嘅 2.6 倍以下，之後嘅走勢會點樣？」
  例子：MRNA 2026-10-21 收市 $245.81 ÷ 200日平均 $80.51 = 3.05 倍；2026-10-22 收市 $202.63 → 2.49 倍（跌返落 2.6 倍以下）。

純研究，唔係買賣策略，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改定義）：
- 股票池：當日真實標普500成分股（同第四十一B輪一樣嘅成分股紀錄）；已計股息（Yahoo 調整後收市）。2000-01-03 至今。
- 倍數 = 收市 ÷ 過去200日收市平均（包今日）。
- 「跌返落嚟」事件 = 尋日倍數 ≥ 2.6，今日倍數 < 2.6。
  同一隻股：要倍數跌到 2.0 以下之後再升返 ≥ 2.6，先算新一次（避免喺 2.6 上下徘徊算咗好多次）。
- 由「事件翌日收市」計起（因為收市後先知道，翌日先可以做嘢），睇之後 1、3、6、12個月（21、63、126、252個交易日）：
  ① 股價升跌 ② 同期 SPY ③ 期間最低收市跌咗幾多 ④ 期間最高收市升咗幾多 ⑤ 12個月內有冇再升返 ≥ 2.6 倍 ⑥ 12個月內有冇高過嗰次最高收市
- 分開列：全部、只計當日符合線E條件（今年 ≥+30%、≥52週最高×0.7、>200日線）、2000-2002科網、2003-2024、2025-2026。
- 列出每一次事件。未夠時間嘅（例如 MRNA 2026-10-22）只列出，唔計入平均。
- 注意：已除牌而 Yahoo 冇數據嘅股票會漏咗（倖存者偏差），結果可能偏好。

輸出：v42_results.csv、v42_progress.txt
"""
import numpy as np
import pandas as pd

import momentum_v1 as M

M.PROGRESS = "v42_progress.txt"
log = M.log
START, END = "2000-01-03", "2026-10-26"
H = [21, 63, 126, 252]


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    closes = M.download_close(everyone + ["SPY"], "1998-01-01", "2026-10-27")
    spy = closes.pop("SPY")
    days = spy.index[spy.index <= pd.Timestamp(END)]
    spy = spy.reindex(days)
    px = pd.DataFrame(closes).reindex(days)
    ratio = px / px.rolling(200).mean()
    ye = px.groupby(days.year).last()
    ytd = px / ye.shift(1).reindex(days.year).values - 1
    hi52 = px.rolling(252, min_periods=200).max()
    e_ok = (ytd >= 0.30) & (px >= hi52 * 0.7) & (ratio > 1)
    log(f"數據：{px.shape[1]} 隻，{len(days)} 日，最後 {days[-1].date()}")
    rows = []
    cache = {}
    for t in px.columns:
        r = ratio[t].values
        armed, peak_px, peak_r, peak_d = True, 0.0, 0.0, None
        for i in range(1, len(days)):
            if days[i] < pd.Timestamp(START) or not np.isfinite(r[i]) or not np.isfinite(r[i - 1]):
                continue
            if r[i] < 2.0:
                armed = True
            if r[i] >= 2.6 and armed:
                if px[t].iat[i] > peak_px:
                    peak_px, peak_r, peak_d = px[t].iat[i], r[i], days[i]
                if r[i] > peak_r:
                    peak_r = r[i]
            if r[i - 1] >= 2.6 and r[i] < 2.6 and armed:
                key = (days[i].year, days[i].month)
                if key not in cache:
                    cache[key] = set(members_fn(days[i]))
                armed = False
                if t not in cache[key]:
                    peak_px, peak_r = 0.0, 0.0
                    continue
                row = {"股票": t, "跌返落日": days[i].date(), "當日倍數": round(r[i], 2), "高峰倍數": round(peak_r, 2),
                       "高峰收市日": peak_d.date() if peak_d is not None else None, "線E條件": bool(e_ok[t].iat[i]),
                       "入場價": None}
                j = i + 1
                if j < len(days) and np.isfinite(px[t].iat[j]):
                    p0, s0 = px[t].iat[j], spy.iat[j]
                    row["入場價"] = round(p0, 2)
                    for h in H:
                        k = j + h
                        if k < len(days):
                            seg = px[t].iloc[j:k + 1]
                            row[f"{h}日"] = seg.iat[-1] / p0 - 1 if np.isfinite(seg.iat[-1]) else np.nan
                            row[f"SPY{h}日"] = spy.iat[k] / s0 - 1
                            row[f"{h}日最低"] = seg.min() / p0 - 1
                            row[f"{h}日最高"] = seg.max() / p0 - 1
                    k = min(j + 252, len(days) - 1)
                    if j + 252 < len(days):
                        row["12個月內再上2.6倍"] = bool((ratio[t].iloc[j:k + 1] >= 2.6).any())
                        row["12個月內高過高峰"] = bool((px[t].iloc[j:k + 1] > peak_px).any())
                rows.append(row)
                peak_px, peak_r, peak_d = 0.0, 0.0, None
    ev = pd.DataFrame(rows).sort_values("跌返落日").reset_index(drop=True)
    log(f"事件 {len(ev)} 次")

    def pct(x):
        return f"{x * 100:+.0f}%" if np.isfinite(x) else "—"

    L = ["=== 第四十二輪：股價 ≥ 200日線2.6倍 之後跌返落 2.6倍以下，跟住會點？（定義預先登記，見research_v42.py開頭）===",
         f"股票池：當日真實標普500（{source}）；已計股息；由事件翌日收市計；數據到 {days[-1].date()}", ""]
    ev["年"] = pd.to_datetime(ev["跌返落日"]).dt.year
    groups = [("全部", ev), ("當日符合線E條件", ev[ev["線E條件"]]), ("2000-2002 科網", ev[ev["年"] <= 2002]),
              ("2003-2024", ev[(ev["年"] >= 2003) & (ev["年"] <= 2024)]), ("2025-2026", ev[ev["年"] >= 2025])]
    for gn, g in groups:
        L.append(f"=== {gn}：{len(g)} 次事件，{g['股票'].nunique()} 隻股 ===")
        for h, lab in zip(H, ["1個月", "3個月", "6個月", "12個月"]):
            if f"{h}日" not in g:
                continue
            x = g[f"{h}日"].dropna()
            if len(x) == 0:
                continue
            sp = g.loc[x.index, f"SPY{h}日"]
            lo = g.loc[x.index, f"{h}日最低"]
            hi = g.loc[x.index, f"{h}日最高"]
            L.append(f"　{lab}（{len(x)}次）：中位數 {pct(x.median())} ｜ 平均 {pct(x.mean())} ｜ 升嘅比例 {(x > 0).mean() * 100:.0f}% ｜ "
                     f"贏SPY比例 {(x > sp).mean() * 100:.0f}% ｜ SPY中位數 {pct(sp.median())} ｜ 期間最低中位數 {pct(lo.median())} ｜ "
                     f"期間最高中位數 {pct(hi.median())} ｜ 最差 {pct(x.min())} ｜ 最好 {pct(x.max())}")
        if "12個月內再上2.6倍" in g:
            a = g["12個月內再上2.6倍"].dropna()
            b = g["12個月內高過高峰"].dropna()
            if len(a):
                L.append(f"　12個月內再升返 ≥2.6倍：{a.astype(bool).mean() * 100:.0f}%（{int(a.astype(bool).sum())}/{len(a)}）｜ "
                         f"12個月內高過之前最高收市：{b.astype(bool).mean() * 100:.0f}%（{int(b.astype(bool).sum())}/{len(b)}）")
        L.append("")
    L.append("=== 每一次事件 ===")
    for _, r in ev.iterrows():
        L.append(f"{r['跌返落日']} {r['股票']} ｜ 高峰 {r['高峰倍數']}倍（最高收市 {r['高峰收市日']}）→ 當日 {r['當日倍數']}倍 ｜ "
                 f"線E條件 {'✔' if r['線E條件'] else '✘'} ｜ 入場 {r['入場價']} ｜ 之後 1個月 {pct(r.get('21日', np.nan))} / "
                 f"3個月 {pct(r.get('63日', np.nan))} / 6個月 {pct(r.get('126日', np.nan))} / 12個月 {pct(r.get('252日', np.nan))} ｜ "
                 f"12個月內最低 {pct(r.get('252日最低', np.nan))} ｜ 再上2.6倍 {r.get('12個月內再上2.6倍', '—')} ｜ 高過高峰 {r.get('12個月內高過高峰', '—')}")
    pd.DataFrame({"report": L}).to_csv("v42_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
