"""
research_v6.py — 第六輪：動量策略加止損 + 「SPY訊號方案 + 動量」組合

⚠️ 預先登記（跑之前定好，跑完唔改）：

基本策略（兩個，都係research_v5入面嘅設定）
  M1 原始動量前10隻（12-1回報，每月換倉，平均分配）
  M2 扣大市動量前10隻（12-1回報 − β × SPY同期回報）
  ⚠️ M2係睇過驗證期結果先加入（v5驗證期最好）；M1係最簡單、冇揀過嘅版本。結果要分開睇。
  共同設定同research_v5一樣：當時標普500成分股、每次買賣0.1%、收市決定→下日收市交易→再下日計回報。

止損方法（每個基本策略都試）
  S1 個股止損：買入後由最高收市價跌 10% / 15% / 20% 就賣，轉現金，等下個月換倉（仍然入選可以再買）
  S2 組合止損：組合（假設一直持有）由「上次入場後最高位」跌 10% / 15% / 20% 就全部賣出；
     組合淨值升返高過自己50日平均先再買
  S3 用SPY訊號方案（main.py規則）做開關：SPY訊號話「現金」，動量組合都轉現金
組合方案（冇止損嘅動量 + main.py SPY訊號方案，每日保持固定比例）
  E 動量佔 30% / 50% / 70%，其餘用SPY訊號方案

揀選期2010-2017；驗證期2018-2026/09；近期2020-2026。
合格：驗證期回報 > SPY長揸，而且驗證期最大跌幅好過 -35%。
每個基本策略入面，用揀選期（跌幅好過-35%）回報最高嘅設定，原封不動睇驗證期。
輸出：v6_results.csv、v6_progress.txt
==== 結果（2026-10-05 第一次跑，記錄低，唔再改）====
- 止損大多令回報同跌幅都變差：個股止損10% → 驗證期+8.4%（跌-47%）；組合止損10% → +6.1%（跌-64%）；
  SPY訊號開關 → +12.2%（跌-46%）。唯一合格：扣大市前10 + 個股止損20%（驗證期+20.47%，跌-34.8%）。
- 組合（動量 + SPY訊號方案，每日固定比例）6個全部合格：
  原始前10：30% +15.96%（跌-30.4%）、50% +17.47%（-32.6%）、70% +18.61%（-34.7%）
  扣大市前10：30% +16.61%（-30.5%）、50% +18.60%（-32.7%）、70% +20.24%（-34.9%）
- 預先定好揀法（原始前10）：揀選期揀中「70%動量+30%SPY訊號」→ 驗證期+18.61%（跌-34.7%）→ ✅合格（第一個）。
- ⚠️ 組合方案係睇過research_v5驗證期結果之後先提出，驗證期唔算完全冇偷睇；要靠之後嘅模擬交易確認。
- 冇一個接近每年50%（最高約20%，2020-2026約24%）。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v5 import COST, BORROW_SPREAD, PERIODS, stats

M.PROGRESS = "v6_progress.txt"
log = M.log
LOOK, SKIP = 252, 21


def effective_weights(rets, targets):
    """回傳 (W, 換倉生效日)：W係每日實際權重（決定日後第2日生效）。"""
    days = rets.index
    rows = {}
    for d, w in targets.items():
        i = days.searchsorted(d, side="right") - 1
        if i + 2 < len(days):
            rows[days[i + 2]] = w
    filled = pd.DataFrame(rows).T.reindex(columns=rets.columns).fillna(0.0)
    W = pd.DataFrame(np.nan, index=days, columns=rets.columns)
    W.loc[filled.index] = filled.values
    W = W.ffill().fillna(0.0)
    return W, list(filled.index)


def daily_returns(W, rets, cash_r, exposure_eff=None):
    e = pd.Series(1.0, index=W.index) if exposure_eff is None else exposure_eff.reindex(W.index).fillna(0.0)
    eff = W.mul(e, axis=0)
    gross = eff.sum(axis=1)
    port = (eff * rets.fillna(0.0)).sum(axis=1) + (1 - gross).clip(lower=0) * cash_r \
        - (gross - 1).clip(lower=0) * (cash_r + BORROW_SPREAD / 252)
    return port - eff.diff().abs().sum(axis=1).fillna(0.0) * COST


def stock_stop_mask(W, starts, px, x):
    """個股止損：每個換倉期內，由入場（生效日前一日收市）計最高收市價，跌x就由之後第2日起唔持有。"""
    mask = pd.DataFrame(1.0, index=W.index, columns=W.columns)
    days = W.index
    bounds = starts + [days[-1] + pd.Timedelta(days=1)]
    for k in range(len(starts)):
        s, e = bounds[k], bounds[k + 1]
        i0 = days.get_loc(s)
        entry_i = i0 - 1
        period = days[(days >= s) & (days < e)]
        names = W.loc[s][W.loc[s] > 0].index
        for t in names:
            p = px[t].iloc[entry_i:days.get_loc(period[-1]) + 1]
            peak = p.cummax()
            hit = p[p < peak * (1 - x)]
            if len(hit):
                j = days.get_loc(hit.index[0]) + 2
                if j < len(days) and days[j] < e:
                    mask.loc[days[j]:period[-1], t] = 0.0
    return mask


def portfolio_stop_exposure(shadow, x):
    """組合止損：shadow係一直持有嘅每日回報。回傳「決定日」嘅倉位（0/1），之後要遲2日生效。"""
    eq = (1 + shadow).cumprod()
    ma = eq.rolling(50).mean()
    state, peak, out = True, eq.iloc[0], []
    for v, m in zip(eq.values, ma.values):
        if state:
            peak = max(peak, v)
            if v < peak * (1 - x):
                state = False
        elif np.isfinite(m) and v > m:
            state, peak = True, v
        out.append(1.0 if state else 0.0)
    return pd.Series(out, index=eq.index)


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

    # 每月排名
    tg = {"M1": {}, "M2": {}}
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
        for key, sc in {"M1": R, "M2": R - beta * Rm}.items():
            top = sc.dropna().sort_values(ascending=False).index[:10]
            tg[key][days[i]] = pd.Series(0.1, index=top)
    log("每月排名完成")

    # SPY訊號方案
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    state = (df["state"] != "out").astype(float).reindex(days).fillna(0.0)
    bot_r = state.shift(2).fillna(0.0) * spy_r + (1 - state.shift(2).fillna(0.0)) * cash_r \
        - state.shift(2).diff().abs().fillna(0.0) * COST

    results = {"SPY長揸": ("基準", spy_r), "SPY訊號方案": ("基準", bot_r)}
    names = {"M1": "原始動量前10", "M2": "扣大市動量前10"}
    for key, label in names.items():
        W, starts = effective_weights(rets, tg[key])
        base = daily_returns(W, rets, cash_r)
        results[f"{label}（冇止損）"] = (key, base)
        for x in (0.10, 0.15, 0.20):
            mask = stock_stop_mask(W, starts, px, x)
            results[f"{label} + 個股止損{x:.0%}"] = (key, daily_returns(W * mask, rets, cash_r))
        for x in (0.10, 0.15, 0.20):
            e = portfolio_stop_exposure(base, x).shift(2).fillna(0.0)
            results[f"{label} + 組合止損{x:.0%}"] = (key, daily_returns(W, rets, cash_r, e))
        results[f"{label} + SPY訊號開關"] = (key, daily_returns(W, rets, cash_r, state.shift(2).fillna(0.0)))
        for w in (0.3, 0.5, 0.7):
            results[f"組合：{label} {w:.0%} + SPY訊號 {1 - w:.0%}"] = (key, w * base + (1 - w) * bot_r)
        log(f"{label} 完成")

    table = {n: (k, {pn: stats(r, a, b) for pn, a, b in PERIODS}) for n, (k, r) in results.items()}
    spy_v = table["SPY長揸"][1]["驗證期2018-2026"]
    L = ["=== 第六輪：動量止損 + 組合（規則預先登記，見research_v6.py開頭）===",
         f"股票池：當時標普500成分股（{source}）；每次買賣0.1%；唔用槓桿",
         "合格：驗證期回報 > SPY，而且驗證期最大跌幅好過-35%", ""]
    for n, (k, s) in table.items():
        if k != "基準":
            continue
        L.append(f"　{n}：" + " | ".join(f"{pn} {x['cagr']:+.2f}%（跌 {x['mdd']:.1f}%）" for pn, x in s.items()))
    L.append("")
    for key, label in names.items():
        L.append(f"=== {label}{'（⚠️睇過驗證期先揀）' if key == 'M2' else ''} ===")
        group = [(n, s) for n, (k, s) in table.items() if k == key]
        for n, s in group:
            v = s["驗證期2018-2026"]
            ok = v["cagr"] > spy_v["cagr"] and v["mdd"] > -35
            L.append(f"　{n}：" + " | ".join(f"{pn} {x['cagr']:+.2f}%（跌 {x['mdd']:.1f}%）" for pn, x in s.items())
                     + f" | $10,000(2010起)→${10000 * s['全期2010-2026']['mult']:,.0f} | {'✅合格' if ok else '❌'}")
        cand = [(n, s) for n, s in group if s["揀選期2010-2017"]["mdd"] > -35]
        if cand:
            n, s = max(cand, key=lambda z: z[1]["揀選期2010-2017"]["cagr"])
            v = s["驗證期2018-2026"]
            rank = sum(1 for _, z in group if z["驗證期2018-2026"]["cagr"] > v["cagr"]) + 1
            ok = v["cagr"] > spy_v["cagr"] and v["mdd"] > -35
            L.append(f"　→ 誠實測試：揀選期揀中「{n}」（{s['揀選期2010-2017']['cagr']:+.2f}%，跌 {s['揀選期2010-2017']['mdd']:.1f}%）"
                     f"→ 驗證期 {v['cagr']:+.2f}%（跌 {v['mdd']:.1f}%），排{rank}/{len(group)}；"
                     f"SPY驗證期 {spy_v['cagr']:+.2f}% → {'✅合格' if ok else '❌唔合格'}")
        L.append("")
    passed = [(n, s) for n, (k, s) in table.items() if k != "基準" and s["驗證期2018-2026"]["cagr"] > spy_v["cagr"]
              and s["驗證期2018-2026"]["mdd"] > -35]
    L.append(f"=== 驗證期合格嘅設定：{len(passed)}個（⚠️事後睇，只作參考）===")
    for n, s in sorted(passed, key=lambda z: -z[1]["驗證期2018-2026"]["cagr"]):
        L.append(f"　{n}：驗證期 {s['驗證期2018-2026']['cagr']:+.2f}%（跌 {s['驗證期2018-2026']['mdd']:.1f}%）；"
                 f"揀選期 {s['揀選期2010-2017']['cagr']:+.2f}%（跌 {s['揀選期2010-2017']['mdd']:.1f}%）")
    pd.DataFrame({"report": L}).to_csv("v6_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
