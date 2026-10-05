"""
research_v9.py — 第九輪：控制動量組合本身嘅風險（行業上限、波幅權重、多週期、相關性、極端贏家、動態比例）

⚠️ 預先登記（跑之前定好，跑完唔改；全部門檻一早寫死，唔會睇結果再調）：

共同設定：當時標普500成分股、每月尾收市決定→下日收市交易、每次買賣0.1%、唔用槓桿。
動量股候選（全部每月換倉）：
  C1 基準：12-1動量前10隻，平均分配（乜都唔加）
  C2 前10隻，權重 ∝ 1/過去60日波幅
  C3 前10隻，權重 ∝ 1/√過去60日波幅
  C4 行業上限3：按排名揀，每個行業最多3隻，揀夠10隻
  C5 行業上限2：同上，每個行業最多2隻
  C6 多週期動量：3個月、6個月、12-1個月三個排名取平均，揀前10
  C7 剔除極端贏家：過去12個月升超過200%嘅唔要，順延揀夠10隻
  C8 相關性過濾：由前20名按排名逐隻揀，同已揀嘅任何一隻過去60日相關性 > 0.70 就唔要，最多10隻（唔夠10隻就平均分配畀揀中嘅）
每個候選都計：① 單獨（100%動量股）② 固定 70%動量 + 30% SPY訊號方案（main.py）
  C9 動態比例：C1動量股比例 = 15% ÷ C1過去60日波幅，限制喺20%至80%，其餘用SPY訊號方案

評估：
  - 全期2010-2026、揀選期2010-2017、驗證期2018-2026、近期2020-2026：每年回報、最大跌幅、最差一年
  - 交易成本壓力：每次買賣 0 / 0.1% / 0.25% / 0.5% / 1%
  - 集中度：單獨版本總盈利入面，貢獻最多嘅5隻股票佔幾多
  - Walk-forward：2015至2026每一年，用之前5年（最大跌幅好過-35%入面）回報最高嘅70/30組合，用落嗰一年；
    將所有「冇偷睇」年份串埋，同SPY同C1 70/30比較
  - 合格（硬性）：驗證期回報 > SPY，而且驗證期最大跌幅好過 -35%；理想：最大跌幅好過 -25%
輸出：v9_results.csv、v9_progress.txt
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v5 import BORROW_SPREAD, PERIODS, stats, sectors_for
from research_v6 import effective_weights, LOOK, SKIP

M.PROGRESS = "v9_progress.txt"
log = M.log
COSTS = [0.0, 0.001, 0.0025, 0.005, 0.01]


def sleeve_returns(W, rets, cash_r, cost):
    gross = W.sum(axis=1)
    port = (W * rets.fillna(0.0)).sum(axis=1) + (1 - gross).clip(lower=0) * cash_r
    return port - W.diff().abs().sum(axis=1).fillna(0.0) * cost


def worst_year(r, a="2010-01-01", b="2026-09-30"):
    d = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
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
    sec = sectors_for(sorted(px.columns))
    days = px.index
    vol60 = rets.rolling(60).std() * np.sqrt(252)

    names = ["C1 基準（前10平均）", "C2 1/波幅權重", "C3 1/√波幅權重", "C4 行業上限3", "C5 行業上限2",
             "C6 多週期動量", "C7 剔除極端贏家", "C8 相關性過濾"]
    tg = {n: {} for n in names}
    for d in month_ends:
        i = days.searchsorted(d, side="right") - 1
        if i < LOOK + 5:
            continue
        day = days[i]
        alive = [t for t in members[d] if t in px.columns and np.isfinite(px[t].iloc[i])
                 and np.isfinite(px[t].iloc[i - LOOK])]
        if len(alive) < 50:
            continue
        P = px[alive]
        R = (P.iloc[i - SKIP] / P.iloc[i - LOOK] - 1).dropna()
        order = list(R.sort_values(ascending=False).index)
        top10 = order[:10]
        ew = lambda lst: pd.Series(1 / len(lst), index=lst)
        tg[names[0]][day] = ew(top10)
        v = vol60.iloc[i].reindex(top10)
        v = v.where(v > 0)
        w2 = (1 / v).dropna()
        tg[names[1]][day] = w2 / w2.sum() if len(w2) else ew(top10)
        w3 = (1 / np.sqrt(v)).dropna()
        tg[names[2]][day] = w3 / w3.sum() if len(w3) else ew(top10)
        for cap, nm in ((3, names[3]), (2, names[4])):
            cnt, pick = {}, []
            for t in order:
                g = sec.get(t, "未知")
                if cnt.get(g, 0) < cap:
                    pick.append(t)
                    cnt[g] = cnt.get(g, 0) + 1
                if len(pick) == 10:
                    break
            tg[nm][day] = ew(pick)
        R3 = P.iloc[i] / P.iloc[i - 63] - 1
        R6 = P.iloc[i] / P.iloc[i - 126] - 1
        comp = pd.concat([R3.rank(pct=True), R6.rank(pct=True), R.rank(pct=True)], axis=1).mean(axis=1, skipna=False).dropna()
        tg[names[5]][day] = ew(list(comp.sort_values(ascending=False).index[:10]))
        R12 = P.iloc[i] / P.iloc[i - LOOK] - 1
        tg[names[6]][day] = ew([t for t in order if R12.get(t, 0) <= 2.0][:10])
        cand = order[:20]
        cm = rets[cand].iloc[i - 59:i + 1].corr()
        keep = []
        for t in cand:
            if all(not (cm.loc[t, k] > 0.70) for k in keep):
                keep.append(t)
            if len(keep) == 10:
                break
        tg[names[7]][day] = ew(keep)
    log("每月揀股完成")

    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    st = (df["state"] != "out").astype(float).reindex(days).fillna(0.0).shift(2).fillna(0.0)
    bot_r = st * spy_r + (1 - st) * cash_r - st.diff().abs().fillna(0.0) * 0.001

    sleeves, contrib = {}, {}
    for n in names:
        W, _ = effective_weights(rets, tg[n])
        sleeves[n] = {c: sleeve_returns(W, rets, cash_r, c) for c in COSTS}
        cw = (W * rets.fillna(0.0))
        cw = cw[cw.index >= pd.Timestamp("2010-01-01")].sum()
        tot = cw.sum()
        contrib[n] = (cw.sort_values(ascending=False).head(5), tot)
    base = sleeves[names[0]][0.001]
    vol_b = base.rolling(60).std() * np.sqrt(252)
    w_dyn = (0.15 / vol_b).clip(0.2, 0.8).fillna(0.7).shift(2).fillna(0.7)
    mixes = {f"{n} | 70/30": 0.7 * sleeves[n][0.001] + 0.3 * bot_r for n in names}
    mixes["C9 動態比例（20%-80%）"] = w_dyn * base + (1 - w_dyn) * bot_r
    log("組合完成")

    pv, pp = "驗證期2018-2026", "揀選期2010-2017"
    spy_s = {pn: stats(spy_r, a, b) for pn, a, b in PERIODS}
    L = ["=== 第九輪：控制動量組合風險（規則預先登記，見research_v9.py開頭）===",
         f"股票池：當時標普500成分股（{source}）；每月換倉；唔用槓桿",
         "SPY長揸：" + " | ".join(f"{pn} {x['cagr']:+.2f}%（跌 {x['mdd']:.1f}%）" for pn, x in spy_s.items()),
         "SPY訊號方案：" + " | ".join(f"{pn} {stats(bot_r, a, b)['cagr']:+.2f}%（跌 {stats(bot_r, a, b)['mdd']:.1f}%）" for pn, a, b in PERIODS),
         ""]

    def line(name, r):
        s = {pn: stats(r, a, b) for pn, a, b in PERIODS}
        ok = s[pv]["cagr"] > spy_s[pv]["cagr"] and s[pv]["mdd"] > -35
        great = ok and s[pv]["mdd"] > -25
        tag = "🌟理想" if great else ("✅合格" if ok else "❌")
        return (f"　{name}：" + " | ".join(f"{pn} {x['cagr']:+.2f}%（跌 {x['mdd']:.1f}%）" for pn, x in s.items())
                + f" | 最差一年 {worst_year(r):+.1f}% | {tag}"), s

    L.append("=== ① 單獨（100%動量股）===")
    for n in names:
        L.append(line(n, sleeves[n][0.001])[0])
    L.append("")
    L.append("=== ② 70%動量 + 30% SPY訊號（同動態比例）===")
    mix_stats = {}
    for n, r in mixes.items():
        txt, s = line(n, r)
        mix_stats[n] = s
        L.append(txt)
    L.append("")
    L.append("=== ③ 交易成本壓力（單獨版本，全期 / 驗證期 每年回報）===")
    for n in names:
        L.append(f"　{n}：" + " | ".join(
            f"{c * 100:.2f}% → {stats(sleeves[n][c], *PERIODS[0][1:])['cagr']:+.1f}% / {stats(sleeves[n][c], *PERIODS[2][1:])['cagr']:+.1f}%"
            for c in COSTS))
    L.append("")
    L.append("=== ④ 集中度：2010-2026總盈利入面，貢獻最多5隻股票 ===")
    for n in names:
        top5, tot = contrib[n]
        L.append(f"　{n}：前5隻佔 {top5.sum() / tot * 100:.0f}%（" + "、".join(f"{t} {v / tot * 100:.0f}%" for t, v in top5.items()) + "）")
    L.append("")
    # Walk-forward
    L.append("=== ⑤ Walk-forward：每年用之前5年表現揀70/30組合（跌幅好過-35%入面回報最高）===")
    oos, picks = [], []
    for y in range(2015, 2027):
        a, b = f"{y - 5}-01-01", f"{y - 1}-12-31"
        sc = {n: stats(r, a, b) for n, r in mixes.items()}
        okc = {n: s for n, s in sc.items() if s["mdd"] > -35} or sc
        best = max(okc, key=lambda n: okc[n]["cagr"])
        r = mixes[best]
        oos.append(r[(r.index >= pd.Timestamp(f"{y}-01-01")) & (r.index <= pd.Timestamp(f"{y}-12-31"))])
        yr = (1 + oos[-1]).prod() - 1
        sy = (1 + spy_r[(spy_r.index.year == y)]).prod() - 1
        picks.append(f"　{y}：揀「{best}」→ 當年 {yr * 100:+.1f}%（SPY {sy * 100:+.1f}%）")
    L += picks
    wf = pd.concat(oos)
    s_wf = stats(wf, "2015-01-01", "2026-09-30")
    s_sp = stats(spy_r, "2015-01-01", "2026-09-30")
    s_c1 = stats(mixes[f"{names[0]} | 70/30"], "2015-01-01", "2026-09-30")
    L.append(f"　串埋2015-2026（全部冇偷睇）：每年 {s_wf['cagr']:+.2f}%（跌 {s_wf['mdd']:.1f}%）｜ "
             f"SPY {s_sp['cagr']:+.2f}%（跌 {s_sp['mdd']:.1f}%）｜ C1 70/30固定 {s_c1['cagr']:+.2f}%（跌 {s_c1['mdd']:.1f}%）")
    pd.DataFrame({"report": L}).to_csv("v9_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
