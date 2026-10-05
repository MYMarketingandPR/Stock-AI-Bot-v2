"""
research_v10.py — 第十輪：唔再揀「更好嘅股」，而係「少啲無謂賣走好股」+ 嚴格檢查係咪運氣

⚠️ 預先登記（跑之前定好，跑完唔改）：

共同設定：當時標普500成分股、收市決定→下日收市交易、每次買賣0.1%、唔用槓桿。
價格全部係「已計股息」（yfinance auto_adjust），SPY基準亦都係總回報（股息再投資）。
所有候選都係「70%動量股 + 30% SPY訊號方案（main.py）」。

候選 = 4組因素全部組合（3×3×3×3 = 81個）：
  A 換倉頻率：每月 / 每半個月（15號或之前最後交易日 + 月尾）/ 每兩個月
  B 持有緩衝：新買入要排頭10；已持有排名跌出 10 / 15 / 20 先賣（排名唔夠10隻就由最高排名補）
  C 權重：平均 / 1÷60日波幅 / 1÷√60日波幅
  D 動量計法：
     原始 = 12-1個月回報
     多週期 = 3個月、6個月、12-1個月三個「排名」取平均（第九輪C6）
     綜合分數 = 40%×3個月回報 + 30%×6個月回報 + 30%×12-1個月回報
評分（只喺最大跌幅好過-35%入面比較）：40%回報 + 30%最大跌幅 + 20% Sharpe + 10% 換手（越低越好），全部用排名百分比。

主要結果：
  ① Walk-forward 2015-2026：每年用之前5年評分最高嘅候選，用落嗰一年，串埋所有年份
  ② 2020-2026高覆蓋樣本：獨立列出線A（每半個月/緩衝10/平均/原始）、線C（每半個月/緩衝10/平均/多週期）、SPY
  ③ 各因素平均效果（每個選項喺81個入面嘅平均）
  ④ 隨機測試：每月隨機揀10隻（同一股票池），70/30，500次 → 睇線A、線C（每月版）係咪明顯好過隨機
  ⑤ Bootstrap：將每日回報以21日一組隨機重抽2000次 → 5%/50%/95% 嘅每年回報同最大跌幅
  ⑥ 跌市歸因：線A每次跌超過10%，同期線C跌幾多；線A虧最多嘅股票邊幾隻線C冇揸
輸出：v10_results.csv、v10_progress.txt
==== 結果（2026-10-05 第一次跑，記錄低，唔再改）====
- ① Walk-forward 2015-2026（按評分每年揀）：+15.96%（跌-33.1%）；SPY +13.67%（-33.7%）；線A固定 +17.44%（-35.1%）。
- ② 線A（半月）全期+18.95%（跌-35.1%），2020-2026 +25.04%；線C（半月）+16.94%（-31.0%），2020-2026 +21.81%。
  Sharpe：線A 0.83、線C 0.81、SPY 0.86、SPY訊號方案 0.92 → 回報高咗，但「每單位風險回報」冇明顯好過SPY。
- ③ 因素：緩衝（>15、>20）減換手但回報同跌幅都差少少；平均分配最好；綜合分數回報最高（驗證+20.3%）但跌幅-38%；
  多週期跌幅最細（-32.5%）但回報最低；半月略好過每月同每兩個月。
- ④ 隨機測試：線A、線C好過99.6%-100%隨機揀股（隨機中位數+11.7%）→ 動量訊號唔係運氣。
- ⑤ Bootstrap：線A每年回報 5%/50%/95% = +9.1%/+18.7%/+28.9%；最大跌幅中位數-38.9%，差嘅5%係-56.0%（SPY -31.2%/-48.3%）。
- ⑥ 跌市歸因：18次線A跌>10%，線C有15次跌得少啲（例如2015-16：A -16.1%、C -2.9%），3次跌得多啲（2023、2026-06兩次）。
"""
import itertools

import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v5 import PERIODS, stats
from research_v6 import effective_weights, LOOK, SKIP
from research_v8 import decision_days

M.PROGRESS = "v10_progress.txt"
log = M.log
COST = 0.001
FREQS = {"每月": "每月", "每半個月": "每半個月", "每兩個月": "每兩個月"}
BUFFERS = [10, 15, 20]
WEIGHTS = ["平均", "1/波幅", "1/√波幅"]
SCORES = ["原始", "多週期", "綜合分數"]


def sleeve(W, rets, cash_r):
    gross = W.sum(axis=1)
    port = (W * rets.fillna(0.0)).sum(axis=1) + (1 - gross).clip(lower=0) * cash_r
    turn = W.diff().abs().sum(axis=1).fillna(0.0)
    return port - turn * COST, turn


def metrics(r, turn, a, b):
    s = stats(r, a, b)
    d = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    t = turn[(turn.index >= pd.Timestamp(a)) & (turn.index <= pd.Timestamp(b))]
    yrs = max((d.index[-1] - d.index[0]).days / 365.25, 0.1)
    s["sharpe"] = d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else 0.0
    s["turn"] = t.sum() / yrs
    return s


def score_pick(cands):
    """cands: {name: metrics}；只比較最大跌幅好過-35%嘅（冇就全部）。回傳分數最高嘅名。"""
    ok = {k: v for k, v in cands.items() if v["mdd"] > -35} or cands
    df = pd.DataFrame(ok).T
    sc = (0.4 * df["cagr"].rank(pct=True) + 0.3 * df["mdd"].rank(pct=True)
          + 0.2 * df["sharpe"].rank(pct=True) + 0.1 * (-df["turn"]).rank(pct=True))
    return sc.idxmax(), sc


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    month_ends = pd.date_range("2008-12-31", "2026-09-30", freq="ME")
    everyone = sorted(set().union(*[members_fn(d) for d in month_ends]))
    closes = M.download_close(everyone + ["SPY"], "2007-06-01", "2026-10-01")
    spy = closes.pop("SPY")
    px = pd.DataFrame(closes).reindex(spy.index)
    px, spy = px[px.index <= pd.Timestamp("2026-09-30")], spy[spy.index <= pd.Timestamp("2026-09-30")]
    rets = px.pct_change(fill_method=None)
    spy_r = spy.pct_change().fillna(0.0)
    cash_r = M.get_cash(px.index)
    days = px.index
    vol60 = rets.rolling(60).std() * np.sqrt(252)

    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    st = (df["state"] != "out").astype(float).reindex(days).fillna(0.0).shift(2).fillna(0.0)
    bot_r = st * spy_r + (1 - st) * cash_r - st.diff().abs().fillna(0.0) * COST

    # 每個決定日嘅分數（三種計法），所有頻率共用
    all_days = sorted(set().union(*[set(decision_days(days, f)) for f in FREQS]))
    score_at, alive_at = {}, {}
    for d in all_days:
        i = days.get_loc(d)
        if i < LOOK + 5 or d < pd.Timestamp("2009-06-01"):
            continue
        alive = [t for t in members_fn(d) if t in px.columns and np.isfinite(px[t].iloc[i])
                 and np.isfinite(px[t].iloc[i - LOOK])]
        if len(alive) < 50:
            continue
        P = px[alive]
        R121 = P.iloc[i - SKIP] / P.iloc[i - LOOK] - 1
        R3 = P.iloc[i] / P.iloc[i - 63] - 1
        R6 = P.iloc[i] / P.iloc[i - 126] - 1
        multi = pd.concat([R3.rank(pct=True), R6.rank(pct=True), R121.rank(pct=True)], axis=1).mean(axis=1, skipna=False)
        comp = 0.4 * R3 + 0.3 * R6 + 0.3 * R121
        score_at[d] = {"原始": R121.dropna(), "多週期": multi.dropna(), "綜合分數": comp.dropna()}
        alive_at[d] = alive
    log(f"計分日：{len(score_at)}個")

    def build_targets(freq, buf, wt, sc):
        tg, held = {}, []
        for d in decision_days(days, freq):
            if d not in score_at:
                continue
            s = score_at[d][sc].sort_values(ascending=False)
            rank = {t: k + 1 for k, t in enumerate(s.index)}
            keep = [t for t in held if rank.get(t, 10**9) <= buf]
            for t in s.index:
                if len(keep) >= 10:
                    break
                if t not in keep:
                    keep.append(t)
            held = keep[:10]
            i = days.get_loc(d)
            if wt == "平均":
                w = pd.Series(1.0, index=held)
            else:
                v = vol60.iloc[i].reindex(held)
                v = v.where(v > 0).fillna(v.median() if v.notna().any() else 0.3)
                w = 1 / v if wt == "1/波幅" else 1 / np.sqrt(v)
            tg[d] = w / w.sum()
        return tg

    cands, turns, sleeves = {}, {}, {}
    for freq, buf, wt, sc in itertools.product(FREQS, BUFFERS, WEIGHTS, SCORES):
        name = f"{freq}｜賣出>{buf}｜{wt}｜{sc}"
        W, _ = effective_weights(rets, build_targets(freq, buf, wt, sc))
        sr, tv = sleeve(W, rets, cash_r)
        cands[name] = 0.7 * sr + 0.3 * bot_r
        turns[name] = 0.7 * tv
        sleeves[name] = W if name in ("每月｜賣出>10｜平均｜原始", "每月｜賣出>10｜平均｜多週期") else None
    log(f"候選完成：{len(cands)}個")

    A = "每半個月｜賣出>10｜平均｜原始"
    C = "每半個月｜賣出>10｜平均｜多週期"
    A_m, C_m = "每月｜賣出>10｜平均｜原始", "每月｜賣出>10｜平均｜多週期"
    spy_turn = pd.Series(0.0, index=days)
    L = ["=== 第十輪（規則預先登記，見research_v10.py開頭）===",
         f"股票池：當時標普500成分股（{source}）；全部價格同SPY都係已計股息（總回報）；70%動量 + 30% SPY訊號；每次買賣0.1%",
         ""]
    PER = PERIODS + [("Walk-forward期2015-2026", "2015-01-01", "2026-09-30")]

    def fmt(name, r, t):
        out = []
        for pn, a, b in PER:
            m = metrics(r, t, a, b)
            out.append(f"{pn} {m['cagr']:+.2f}%（跌 {m['mdd']:.1f}%）")
        mm = metrics(r, t, "2010-01-01", "2026-09-30")
        return f"　{name}：" + " | ".join(out) + f" | Sharpe {mm['sharpe']:.2f} | 每年換手 {mm['turn'] * 100:.0f}%"

    # ① Walk-forward
    L.append("=== ① Walk-forward 2015-2026（每年用之前5年評分最高嘅候選）===")
    oos = []
    for y in range(2015, 2027):
        a, b = f"{y - 5}-01-01", f"{y - 1}-12-31"
        best, _ = score_pick({n: metrics(r, turns[n], a, b) for n, r in cands.items()})
        r = cands[best]
        seg = r[(r.index >= pd.Timestamp(f"{y}-01-01")) & (r.index <= pd.Timestamp(f"{y}-12-31"))]
        oos.append(seg)
        L.append(f"　{y}：揀「{best}」→ 當年 {((1 + seg).prod() - 1) * 100:+.1f}%"
                 f"（SPY {((1 + spy_r[spy_r.index.year == y]).prod() - 1) * 100:+.1f}%）")
    wf = pd.concat(oos)
    s_wf, s_sp = stats(wf, "2015-01-01", "2026-09-30"), stats(spy_r, "2015-01-01", "2026-09-30")
    s_A = stats(cands[A], "2015-01-01", "2026-09-30")
    L.append(f"　串埋2015-2026：每年 {s_wf['cagr']:+.2f}%（跌 {s_wf['mdd']:.1f}%）｜ SPY {s_sp['cagr']:+.2f}%（跌 {s_sp['mdd']:.1f}%）"
             f"｜ 線A固定 {s_A['cagr']:+.2f}%（跌 {s_A['mdd']:.1f}%）")
    L.append("")

    # ② 2020+ 同主要線
    L.append("=== ② 主要候選（2020-2026係數據覆蓋最好嘅一段）===")
    L.append(fmt("SPY長揸（總回報）", spy_r, spy_turn))
    L.append(fmt("SPY訊號方案", bot_r, spy_turn))
    for nm, lab in ((A, "線A"), (C, "線C"), (A_m, "線A每月版"), (C_m, "線C每月版")):
        L.append(fmt(f"{lab}（{nm}）", cands[nm], turns[nm]))
    full = {n: metrics(r, turns[n], "2010-01-01", "2026-09-30") for n, r in cands.items()}
    best_full, sc = score_pick(full)
    L.append("　全期評分最高5個（⚠️事後睇）：")
    for n in sc.sort_values(ascending=False).index[:5]:
        L.append(fmt(n, cands[n], turns[n]))
    L.append("")

    # ③ 各因素平均效果
    L.append("=== ③ 各因素平均效果（81個候選入面，每個選項嘅平均）===")
    rows = []
    for n, r in cands.items():
        f, b, w, s = n.split("｜")
        m_v = metrics(r, turns[n], "2018-01-01", "2026-09-30")
        m_r = metrics(r, turns[n], "2020-01-01", "2026-09-30")
        rows.append({"頻率": f, "緩衝": b, "權重": w, "計法": s, "驗證回報": m_v["cagr"], "驗證跌幅": m_v["mdd"],
                     "近期回報": m_r["cagr"], "換手": full[n]["turn"]})
    fdf = pd.DataFrame(rows)
    for col in ("頻率", "緩衝", "權重", "計法"):
        L.append(f"　【{col}】")
        for k, g in fdf.groupby(col, sort=False):
            L.append(f"　　{k}：驗證期 {g['驗證回報'].mean():+.2f}%（跌 {g['驗證跌幅'].mean():.1f}%）｜ 2020-2026 {g['近期回報'].mean():+.2f}% ｜ 每年換手 {g['換手'].mean() * 100:.0f}%")
    L.append("")

    # ④ 隨機測試
    L.append("=== ④ 隨機測試：每月隨機揀10隻（70/30），500次 ===")
    rng = np.random.default_rng(2026)
    mdays = [d for d in decision_days(days, "每月") if d in alive_at]
    rand_full, rand_wf = [], []
    for k in range(500):
        tg = {d: pd.Series(0.1, index=rng.choice(alive_at[d], 10, replace=False)) for d in mdays}
        W, _ = effective_weights(rets, tg)
        sr, _ = sleeve(W, rets, cash_r)
        mix = 0.7 * sr + 0.3 * bot_r
        rand_full.append(stats(mix, "2010-01-01", "2026-09-30")["cagr"])
        rand_wf.append(stats(mix, "2020-01-01", "2026-09-30")["cagr"])
        if (k + 1) % 100 == 0:
            log(f"隨機測試 {k + 1}/500")
    rf, rw = np.array(rand_full), np.array(rand_wf)
    for nm, lab in ((A_m, "線A每月版"), (C_m, "線C每月版")):
        x = stats(cands[nm], "2010-01-01", "2026-09-30")["cagr"]
        y = stats(cands[nm], "2020-01-01", "2026-09-30")["cagr"]
        L.append(f"　{lab}：2010-2026 {x:+.2f}%（好過 {np.mean(rf < x) * 100:.1f}% 隨機）｜ 2020-2026 {y:+.2f}%（好過 {np.mean(rw < y) * 100:.1f}% 隨機）")
    L.append(f"　隨機：2010-2026 中位數 {np.median(rf):+.2f}%（5%-95%：{np.percentile(rf, 5):+.2f}% 至 {np.percentile(rf, 95):+.2f}%）"
             f"｜ 2020-2026 中位數 {np.median(rw):+.2f}%（{np.percentile(rw, 5):+.2f}% 至 {np.percentile(rw, 95):+.2f}%）")
    L.append("")

    # ⑤ Bootstrap
    L.append("=== ⑤ Bootstrap（2010-2026每日回報，21日一組重抽，2000次）===")
    for nm, r in (("線A", cands[A]), ("線C", cands[C]), ("SPY長揸", spy_r), ("SPY訊號方案", bot_r)):
        x = r[(r.index >= pd.Timestamp("2010-01-01"))].values
        nb = len(x) // 21
        cg, dd = [], []
        for _ in range(2000):
            starts = rng.integers(0, len(x) - 21, nb)
            path = np.concatenate([x[s:s + 21] for s in starts])
            eq = np.cumprod(1 + path)
            cg.append((eq[-1] ** (252 / len(path)) - 1) * 100)
            dd.append((eq / np.maximum.accumulate(eq) - 1).min() * 100)
        cg, dd = np.array(cg), np.array(dd)
        L.append(f"　{nm}：每年回報 5%={np.percentile(cg, 5):+.1f}% / 中位數={np.median(cg):+.1f}% / 95%={np.percentile(cg, 95):+.1f}%"
                 f" ｜ 最大跌幅 差嘅5%={np.percentile(dd, 5):.1f}% / 中位數={np.median(dd):.1f}%")
    L.append("")

    # ⑥ 跌市歸因（每月版，有權重紀錄）
    L.append("=== ⑥ 跌市歸因：線A每月版每次跌超過10%，同期線C每月版跌幾多 ===")
    ra, rc = cands[A_m], cands[C_m]
    ra2 = ra[ra.index >= pd.Timestamp("2010-01-01")]
    eq = (1 + ra2).cumprod()
    peak_d, peak, in_dd, trough_d = eq.index[0], eq.iloc[0], False, None
    episodes = []
    for d, v in eq.items():
        if v >= peak:
            if in_dd:
                episodes.append((peak_d, trough_d))
                in_dd = False
            peak, peak_d, trough_d = v, d, d
        else:
            if trough_d is None or v < eq[trough_d]:
                trough_d = d
            if v / peak - 1 <= -0.10:
                in_dd = True
    if in_dd:
        episodes.append((peak_d, trough_d))
    WA, WC = sleeves[A_m], sleeves[C_m]
    for p, t in episodes:
        da = (1 + ra.loc[p:t].iloc[1:]).prod() - 1
        dc = (1 + rc.loc[p:t].iloc[1:]).prod() - 1
        ca = (WA.loc[p:t] * rets.loc[p:t].fillna(0.0)).sum()
        cc = (WC.loc[p:t] * rets.loc[p:t].fillna(0.0)).sum()
        worst = ca.sort_values().head(5)
        diff = [f"{s}（A {ca[s] * 100:+.1f}% / C {cc.get(s, 0) * 100:+.1f}%）" for s in worst.index]
        L.append(f"　{p.date()} → {t.date()}：線A {da * 100:+.1f}%，線C {dc * 100:+.1f}%；線A虧最多：" + "、".join(diff))
    pd.DataFrame({"report": L}).to_csv("v10_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
