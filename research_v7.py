"""
research_v7.py — 第七輪：向每年50%推 —— 集中持股 + 槓桿

⚠️ 預先登記（跑之前定好，跑完唔改）：

設定同research_v6一樣（當時標普500成分股、12-1動量、每月換倉、每次買賣0.1%、收市決定→下日收市交易）。
組合 = 動量部分 w + SPY訊號方案（main.py規則）部分 (1−w)，每日保持固定比例，再乘槓桿 L。
  動量計分：原始 / 扣大市
  持股數量：前3隻、前5隻、前10隻（平均分配）
  動量比例 w：70%、100%（另加 w=0，即係淨用SPY訊號方案）
  槓桿 L：1倍、1.5倍、2倍（借錢成本 = 國債息 + 1.5%；每日維持同一槓桿）
  → 2×3×2×3 + 3 = 39個

揀選期2010-2017；驗證期2018-2026/09；近期2020-2026。
報告兩種揀法：
  ① 守住底線：揀選期最大跌幅好過-35%入面回報最高 → 原封不動睇驗證期
  ② 唔理跌幅：揀選期回報最高 → 原封不動睇驗證期（睇吓追50%要付出幾大跌幅）
另外列出：驗證期每年回報達到50%以上嘅設定（如果有）同佢哋嘅最大跌幅、最差一年。
⚠️ 用戶本身規則係唔用槓桿；槓桿結果只係俾用戶知道代價。
輸出：v7_results.csv、v7_progress.txt
==== 結果（2026-10-05 第一次跑，記錄低，唔再改）====
- 39個設定，驗證期冇一個達到每年50%。最高：扣大市前10 70% + SPY訊號30% × 2倍 → 驗證期+28.71%（跌-63.5%）。
- 2020-2026最高：原始前3 70%/30% × 2倍 +42.2%（跌-57.8%，最差一年-58.1%）。
- 唔用槓桿最高：原始前3 100% → 驗證期+22.69%（跌-50.8%，最差一年-41.4%）。
- 預先定好揀法①（守住-35%）：唔用槓桿揀中扣大市前3 70/30 → 驗證期+20.34% 但跌-47.4%；
  用槓桿揀中SPY訊號×2 → +19.57%，跌-49.4%。兩個驗證期跌幅都超過-35%。
- 揀法②（唔理跌幅）：用槓桿揀中原始前3 70/30 ×2 → 驗證期+25.43%，跌-71.3%。
- 結論：集中持股同槓桿可以將回報推到大約每年20%至29%，但最大跌幅變成-47%至-89%。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v5 import COST, BORROW_SPREAD, PERIODS, stats
from research_v6 import effective_weights, daily_returns, LOOK, SKIP

M.PROGRESS = "v7_progress.txt"
log = M.log


def worst_year(daily, a, b):
    d = daily[(daily.index >= pd.Timestamp(a)) & (daily.index <= pd.Timestamp(b))]
    return ((1 + d).groupby(d.index.year).prod() - 1).min() * 100


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    month_ends = pd.date_range("2008-12-31", "2026-09-30", freq="ME")
    members = {d: members_fn(d) for d in month_ends}
    everyone = sorted(set().union(*members.values()))
    closes = M.download_close(everyone + ["SPY"], "2007-06-01", "2026-10-01")
    spy = closes.pop("SPY")
    px = pd.DataFrame(closes).reindex(spy.index)
    px, spy = px[px.index <= pd.Timestamp("2026-09-30")], spy[spy.index <= pd.Timestamp("2026-09-30")]
    rets = px.pct_change(fill_method=None)
    spy_r = spy.pct_change().fillna(0.0)
    cash_r = M.get_cash(px.index)
    days = px.index

    tg = {(s, n): {} for s in ("原始", "扣大市") for n in (3, 5, 10)}
    for d in month_ends:
        i = days.searchsorted(d, side="right") - 1
        if i < LOOK + 5:
            continue
        alive = [t for t in members[d] if t in px.columns and np.isfinite(px[t].iloc[i])
                 and np.isfinite(px[t].iloc[i - LOOK])]
        if len(alive) < 50:
            continue
        R = px[alive].iloc[i - SKIP] / px[alive].iloc[i - LOOK] - 1
        Rm = spy.iloc[i - SKIP] / spy.iloc[i - LOOK] - 1
        win = rets[alive].iloc[i - LOOK + 1:i + 1]
        sw = spy_r.iloc[i - LOOK + 1:i + 1]
        beta = win.apply(lambda c: c.cov(sw) / sw.var())
        for s, sc in {"原始": R, "扣大市": R - beta * Rm}.items():
            order = sc.dropna().sort_values(ascending=False).index
            for n in (3, 5, 10):
                tg[(s, n)][days[i]] = pd.Series(1 / n, index=order[:n])
    log("每月排名完成")

    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    st = (df["state"] != "out").astype(float).reindex(days).fillna(0.0).shift(2).fillna(0.0)
    bot_r = st * spy_r + (1 - st) * cash_r - st.diff().abs().fillna(0.0) * COST

    def lever(r, L):
        return L * r - (L - 1) * (cash_r + BORROW_SPREAD / 252)

    results = {}
    for L in (1.0, 1.5, 2.0):
        results[f"淨用SPY訊號方案 × {L}倍"] = (L, lever(bot_r, L))
    for (s, n), t in tg.items():
        W, _ = effective_weights(rets, t)
        mom = daily_returns(W, rets, cash_r)
        for w in (0.7, 1.0):
            mix = w * mom + (1 - w) * bot_r
            for L in (1.0, 1.5, 2.0):
                results[f"{s}動量前{n}隻 {w:.0%} + SPY訊號 {1 - w:.0%} × {L}倍"] = (L, lever(mix, L))
        log(f"{s}前{n} 完成")

    table = {k: (L, {pn: stats(r, a, b) for pn, a, b in PERIODS},
                 worst_year(r, "2010-01-01", "2026-09-30")) for k, (L, r) in results.items()}
    spy_s = {pn: stats(spy_r, a, b) for pn, a, b in PERIODS}
    pv = "驗證期2018-2026"
    L_ = ["=== 第七輪：集中持股 + 槓桿（規則預先登記，見research_v7.py開頭）===",
          f"股票池：當時標普500成分股（{source}）；每次買賣0.1%；槓桿借錢成本 = 國債息 + 1.5%",
          "SPY長揸：" + " | ".join(f"{pn} {x['cagr']:+.2f}%（跌 {x['mdd']:.1f}%）" for pn, x in spy_s.items()), ""]
    L_.append("=== 全部設定（按驗證期回報排）===")
    for k, (L, s, wy) in sorted(table.items(), key=lambda kv: -kv[1][1][pv]["cagr"]):
        L_.append(f"　{'【槓桿】' if L > 1 else ''}{k}：" + " | ".join(f"{pn} {x['cagr']:+.2f}%（跌 {x['mdd']:.1f}%）" for pn, x in s.items())
                  + f" | 最差一年 {wy:+.1f}% | $10,000(2010起)→${10000 * s['全期2010-2026']['mult']:,.0f}")
    L_.append("")
    for title, cond in [("① 守住底線（揀選期跌幅好過-35%）", lambda s: s["揀選期2010-2017"]["mdd"] > -35),
                        ("② 唔理跌幅", lambda s: True)]:
        for lev_ok in (False, True):
            cand = [(k, s) for k, (L, s, _) in table.items() if (L > 1) == lev_ok and cond(s)]
            if not cand:
                continue
            k, s = max(cand, key=lambda z: z[1]["揀選期2010-2017"]["cagr"])
            rank = sum(1 for _, (L, z, _) in table.items() if z[pv]["cagr"] > s[pv]["cagr"]) + 1
            L_.append(f"{title}，{'用槓桿' if lev_ok else '唔用槓桿'}：揀選期揀中「{k}」（{s['揀選期2010-2017']['cagr']:+.2f}%，"
                      f"跌 {s['揀選期2010-2017']['mdd']:.1f}%）→ 驗證期 {s[pv]['cagr']:+.2f}%（跌 {s[pv]['mdd']:.1f}%），排{rank}/{len(table)}")
    L_.append("")
    big = [(k, s, wy) for k, (L, s, wy) in table.items() if s[pv]["cagr"] >= 50]
    L_.append(f"=== 驗證期每年回報 ≥ 50% 嘅設定：{len(big)}個 ===")
    for k, s, wy in big:
        L_.append(f"　{k}：驗證期 {s[pv]['cagr']:+.2f}%，最大跌幅 {s[pv]['mdd']:.1f}%，最差一年 {wy:+.1f}%")
    top = max(table.items(), key=lambda kv: kv[1][1][pv]["cagr"])
    L_.append(f"　（驗證期最高：{top[0]}，{top[1][1][pv]['cagr']:+.2f}%，跌 {top[1][1][pv]['mdd']:.1f}%）")
    pd.DataFrame({"report": L_}).to_csv("v7_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L_))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
