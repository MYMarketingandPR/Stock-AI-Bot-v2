"""
research_v42b.py — 第四十二B輪：股價「第一次升到」200日線 2.6 倍以上之後，走勢會點？（Esther 2026-10-26：
  「當股價去到SMA 200既2.6倍以上，之後既走勢會點樣？」）
  第四十二輪係由「跌返落 2.6 倍以下」嗰日計；呢輪係由「升到 2.6 倍」嗰日計，兩輪互補。

純研究（事件統計），唔改任何模擬交易，唔改之前任何結果（包括第四十二輪）。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改定義）：
股票池：當日真實標普500成分股（同第四十一B輪一樣嘅數據），已計股息嘅收市價，1999-2026。
倍數 = 收市 ÷ 200日平均（200日平均包括當日）。
事件：倍數第一次 ≥ 2.6 嘅嗰日；之前 126 個交易日（約半年）都冇 ≥ 2.6 先算新事件（避免同一段升浪重複計）。
由事件嗰日收市計：
  ① 之後 1個月（21日）、3個月（63日）、6個月（126日）、12個月（252日）嘅回報
  ② 12個月入面最高收市（比事件日高幾多 %、第幾日到）、最低收市（比事件日低幾多 %）
  ③ 倍數之後最高去到幾多；幾多日之後第一次跌返落 2.6 倍以下
  ④ 「跌返落 2.6 倍以下」嗰日（好似 MRNA 2026-10-22）再計之後 1/3/6/12 個月回報
對照：同一段時間 SPY 嘅回報；同埋全部成分股任何一日（倍數 1.0 至 2.6 之間、即係喺200日線上面但未去到 2.6 倍）之後 12 個月回報中位數。
另外用 2.9 倍再計一次 ①（只係參考）。
如果中途除牌，用最後一個收市價計（會標明）。最近嘅事件未夠時間就寫「未夠時間」。
就算結果好明顯都唔會直接改模擬交易，要你同意先。

輸出：v42b_results.csv、v42b_progress.txt
==== 結果（2026-10-26 第一次跑，記錄低，唔再改）====
- 16 次事件、13 隻股（2.6 倍）。觸發嗰日收市買入：1個月中位數 -10%（SPY 0%）、3個月 -4%（SPY +4%）、6個月 +11%（SPY +3%）、
  12個月 -29%（SPY -6%；夠一年 11 次，賺錢 36%，跌超過30% 45%）。
- 12個月入面：最高點中位數 +56%（第162日），最低點中位數 -47% → 上落都好大。
- 通常好快跌返落 2.6 倍以下：中位數 2 個交易日。
- 年代：1999-2002 12個月中位數 -39%（5次）；2003-2019 +3%（3次）；2020-2026 -29%（3次）。2026 年嘅事件仲未夠一年。
- 對照（之後12個月中位數）：成分股喺 1.0-2.6 倍 +10%（賺錢67%）；2.0-2.6 倍 +4%；≥2.6 倍全部日子（151個股票日）-50%（賺錢18%、跌超過30% 64%）。
- 2.9 倍參考：12個月中位數 -13%（6次）。
- 注意：事件好少（11 次夠一年），而且好多集中喺 2000 年同 2026 年同一段時間，唔係獨立事件；已除牌而 Yahoo 冇數據嘅股會漏咗。
- 程式限制（跑完先發現，冇改）：只計「第一次」觸發；如果第一次觸發嗰日喺 1999-06 之前、或者嗰陣仲未入標普500（例如 QCOM、SNDK 可能係咁），
  之後 126 日內再觸發都唔會計 → 第四十二輪（跌返落）有 QCOM、SNDK，呢輪冇。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M

M.PROGRESS = "v42b_progress.txt"
log = M.log
END = "2026-10-23"
H = {"1個月": 21, "3個月": 63, "6個月": 126, "12個月": 252}


def pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.0f}%"


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", "2026-09-30", freq="ME")]))
    closes = M.download_close(everyone + ["SPY"], "1998-01-01", "2026-10-01")      # 同第四十一B輪一樣（用返快取）
    days0 = closes["SPY"].index
    px0 = pd.DataFrame(closes).reindex(days0)
    r0 = px0 / px0.rolling(200).mean()
    hit_t = sorted(set(r0.columns[(r0 >= 2.6).any()]) | {"SPY"})
    recent = M.download_close(hit_t, "2024-06-01", "2026-10-24")                  # 有觸發過嘅股，補返最新數據
    for t, s in recent.items():
        closes[t] = s.combine_first(closes.get(t, pd.Series(dtype=float)))
    spy = closes.pop("SPY")
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    px = pd.DataFrame(closes).reindex(days)
    last_valid = px.apply(lambda s: s.last_valid_index())
    pxf = px.ffill()
    ratio = px / px.rolling(200).mean()
    log("數據準備好")

    mem_cache = {}

    def is_member(t, d):
        k = (d.year, d.month)
        if k not in mem_cache:
            mem_cache[k] = set(members_fn(d))
        return t in mem_cache[k]

    def events(th):
        out = []
        for t in ratio.columns:
            s = ratio[t]
            hit = (s >= th).values
            last = -10 ** 9
            for i in np.flatnonzero(hit):
                if i - last > 126 and days[i] >= pd.Timestamp("1999-06-01") and is_member(t, days[i]):
                    out.append((t, i))
                last = i
        return sorted(out, key=lambda x: x[1])

    def fwd(t, i, n):
        j = i + n
        if j >= len(days):
            return None
        return pxf[t].iat[j] / pxf[t].iat[i] - 1

    def spy_fwd(i, n):
        j = i + n
        return None if j >= len(days) else spy.iat[j] / spy.iat[i] - 1

    L = ["=== 第四十二B輪：股價 ≥ 200日線 2.6 倍之後嘅走勢（規則預先登記，見research_v42b.py開頭）===",
         f"股票池：當日真實標普500（{source}）；已計股息；數據到 {END}", ""]
    ev = events(2.6)
    rows = []
    for t, i in ev:
        p0 = pxf[t].iat[i]
        w = pxf[t].iloc[i + 1:i + 253]
        full = i + 252 < len(days)
        after = ratio[t].iloc[i:]
        below = np.flatnonzero((after < 2.6).values)
        k_below = int(below[0]) if len(below) else None
        rec = dict(股票=t, 日期=days[i].date(), 收市=p0, 倍數=ratio[t].iat[i],
                   最高倍數=float(ratio[t].iloc[i:i + 253].max()),
                   幾多日跌返落=k_below,
                   最高=(w.max() / p0 - 1) if len(w) else np.nan, 第幾日最高=(int(np.argmax(w.values)) + 1) if len(w) else None,
                   最低=(w.min() / p0 - 1) if len(w) else np.nan, 夠一年=full,
                   除牌=bool(pd.notna(last_valid[t]) and last_valid[t] < days[min(i + 252, len(days) - 1)]))
        for nm, n in H.items():
            rec[nm] = fwd(t, i, n)
            rec["SPY" + nm] = spy_fwd(i, n)
        if k_below is not None:
            ib = i + k_below
            rec["跌返落日期"] = days[ib].date()
            for nm, n in H.items():
                rec["跌返落後" + nm] = fwd(t, ib, n)
        rows.append(rec)
    df = pd.DataFrame(rows)
    log(f"事件 {len(df)} 個")

    L.append(f"=== 每一次事件（{len(df)} 次，{df['股票'].nunique()} 隻股）===")
    for _, r in df.iterrows():
        L.append(f"　{r['股票']} {r['日期']} 收市 ${r['收市']:,.2f}（{r['倍數']:.2f} 倍，之後最高 {r['最高倍數']:.1f} 倍）｜ "
                 + " ".join(f"{nm} {pct(r[nm])}" for nm in H) + f" ｜ 12個月內最高 {pct(r['最高'])}（第{int(r['第幾日最高']) if pd.notna(r['第幾日最高']) else '—'}日）"
                 f" 最低 {pct(r['最低'])} ｜ {('第' + str(int(r['幾多日跌返落'])) + '日跌返落2.6倍以下') if r['幾多日跌返落'] is not None and pd.notna(r['幾多日跌返落']) else '仲未跌返落'}"
                 + ("（未夠一年）" if not r["夠一年"] else "") + ("（中途除牌，用最後收市價）" if r["除牌"] else ""))
    L.append("")

    def summ(col, sub):
        x = sub[col].dropna().astype(float)
        if len(x) == 0:
            return "冇數據"
        return f"{len(x)} 次 ｜ 中位數 {pct(x.median())} ｜ 平均 {pct(x.mean())} ｜ 賺錢 {(x > 0).mean() * 100:.0f}% ｜ 跌超過30% {(x < -0.3).mean() * 100:.0f}% ｜ 最差 {pct(x.min())} ｜ 最好 {pct(x.max())}"

    L.append("=== 觸發嗰日收市買入，之後回報（只計夠時間嘅事件）===")
    for nm in H:
        L.append(f"　{nm}：{summ(nm, df)}")
        L.append(f"　　同期 SPY：{summ('SPY' + nm, df[df[nm].notna()])}")
    f1 = df[df["夠一年"]]
    L.append(f"　12個月內最高（比觸發日）：{summ('最高', f1)}")
    L.append(f"　12個月內最低（比觸發日）：{summ('最低', f1)}")
    L.append(f"　最高點喺第幾日：中位數 {f1['第幾日最高'].median():.0f} 日")
    kb = df["幾多日跌返落"].dropna()
    L.append(f"　幾耐跌返落 2.6 倍以下：中位數 {kb.median():.0f} 個交易日（{len(kb)} 次有跌返落）")
    L.append("")
    L.append("=== 第一次「跌返落 2.6 倍以下」嗰日收市買入，之後回報 ===")
    for nm in H:
        L.append(f"　{nm}：{summ('跌返落後' + nm, df) if '跌返落後' + nm in df else '冇數據'}")
    L.append("")

    L.append("=== 年代分開（12個月回報）===")
    for a_, b_, lab in [("1999", "2002", "1999-2002 科網"), ("2003", "2019", "2003-2019"), ("2020", "2026", "2020-2026")]:
        sub = df[(pd.to_datetime(df["日期"]) >= pd.Timestamp(a_ + "-01-01")) & (pd.to_datetime(df["日期"]) <= pd.Timestamp(b_ + "-12-31"))]
        L.append(f"　{lab}：{summ('12個月', sub)}")
    L.append("")

    log("計對照")
    with np.errstate(invalid="ignore"):
        f12 = pxf.shift(-252) / pxf - 1
    mem = np.zeros(px.shape, dtype=bool)
    cols = list(px.columns)
    ix = {t: k for k, t in enumerate(cols)}
    for i, d in enumerate(days):
        k = (d.year, d.month)
        if k not in mem_cache:
            mem_cache[k] = set(members_fn(d))
        mem[i, [ix[t] for t in mem_cache[k] if t in ix]] = True
    R, F = ratio.values, f12.values
    ok = mem & np.isfinite(F) & (days.values[:, None] >= np.datetime64("1999-06-01"))
    for lo, hi, lab in [(1.0, 2.6, "1.0 至 2.6 倍（喺200日線上面）"), (1.5, 2.6, "1.5 至 2.6 倍"), (2.0, 2.6, "2.0 至 2.6 倍"), (2.6, 99, "≥ 2.6 倍（全部日子，唔止第一日）")]:
        m = ok & (R >= lo) & (R < hi)
        x = F[m]
        L.append(f"　對照｜成分股任何一日 {lab}：{m.sum():,} 個股票日 ｜ 之後12個月中位數 {pct(np.median(x))} ｜ 賺錢 {(x > 0).mean() * 100:.0f}% ｜ 跌超過30% {(x < -0.3).mean() * 100:.0f}%")
    L.append("")

    ev29 = events(2.9)
    L.append(f"=== 參考：用 2.9 倍（{len(ev29)} 次事件）===")
    d29 = pd.DataFrame([{nm: fwd(t, i, n) for nm, n in H.items()} for t, i in ev29])
    for nm in H:
        L.append(f"　{nm}：{summ(nm, d29) if len(d29) else '冇數據'}")

    pd.DataFrame({"report": L}).to_csv("v42b_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
