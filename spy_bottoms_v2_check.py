"""
spy_bottoms_v2_check.py — 檢查「E4 三合一提早買」係真本事定撞彩

背景：spy_bottoms_v2.py 入面，E4（基本規則 + 空倉時三合一信號 RSI<30 + VIX≥30 + 成交量≥2倍 提早買，
跌5%止損）係唯一符合預先定好採用條件嘅策略。呢個檔案唔係用嚟揀更靚嘅參數，只係檢查：
1. E4每一次提早買入嘅詳細紀錄（幾多次、每次賺蝕）—— 次數少就可能係運氣。
2. 隔籬參數（RSI 25/30/35 × VIX 25/30/35 × 成交量 1.5/2/2.5 倍 = 27個；止損 3%/5%/8%）
   係咪大部分都好過基本規則。如果只係啱啱呢一個得，就好可能係撞彩。
3. 逐年比較 E4 同基本規則。
⚠️ 檢查結果唔會用嚟改E4嘅參數。
==== 結果（2026-10，GitHub run 37200432735）====
- E4共13次提早買：6次賺（+10.8%至+36.1%），7次止損（-5.1%至-7.2%）。
- 隔籬27組參數：26/27全期回報好過基本規則，17/27符合採用條件，回報中位數+11.22%，跌幅中位數-27.9%。
- 逐年：E4好過基本規則7年、差3年（次數少，統計上未夠肯定）。
"""
import numpy as np
import pandas as pd

from spy_exits import run, stats, PERIODS, START
from spy_bottoms_v2 import load, add_features, base_plus


def base_plus_log(spy, ma, entry, stop, band=0.02):
    """同base_plus(pct止損)一樣，但記錄每次提早買入。"""
    state, entry_px, entry_d, out, log = "out", None, None, [], []
    for i, (d, p, m) in enumerate(zip(spy.index, spy.values, ma.values)):
        if np.isnan(m):
            out.append(0.0)
            continue
        if state == "in":
            if p < m * (1 - band):
                state = "out"
                if log and log[-1].get("normal_exit") is None and log[-1].get("became_normal"):
                    log[-1]["normal_exit"], log[-1]["normal_exit_px"] = d, p
        elif state == "out":
            if p > m * (1 + band):
                state = "in"
            elif bool(entry.iloc[i]):
                state, entry_px, entry_d = "early", p, d
                log.append({"買入日": d, "買入價": p})
        elif state == "early":
            if p > m * (1 + band):
                state = "in"
                log[-1].update({"became_normal": d, "轉正常持有日": d.date(),
                                "提早階段回報%": round((p / entry_px - 1) * 100, 1)})
            elif p < entry_px * (1 - stop):
                state = "out"
                log[-1].update({"止損日": d.date(), "提早階段回報%": round((p / entry_px - 1) * 100, 1)})
        out.append(0.0 if state == "out" else 1.0)
    return pd.Series(out, index=spy.index), log


def main():
    spy, vol, vix, cash_r = load()
    spy = spy[spy.index <= pd.Timestamp("2026-09-30")]
    vol, vix, cash_r = (x.reindex(spy.index) for x in (vol, vix, cash_r))
    cash_r = cash_r.fillna(0.0)
    f, ma200 = add_features(spy, vol, vix)
    L = ["=== 檢查E4（三合一提早買）係咪撞彩 ===", ""]

    def sig(r, v, q):
        return ((f["RSI14"] < r) & (f["VIX"] >= v) & (f["成交量倍數"] >= q)).fillna(False)

    # 1. 交易紀錄
    tgt, log = base_plus_log(spy, ma200, sig(30, 30, 2.0), 0.05)
    L.append(f"--- 1. E4每次提早買入（共{len(log)}次）---")
    for t in log:
        if t["買入日"] < pd.Timestamp(START):
            continue
        res = (f"→ {t['轉正常持有日']} 升穿200日線+2%，轉為正常持有" if "轉正常持有日" in t
               else f"→ {t.get('止損日', '仍然持有')} 止損" if "止損日" in t else "→ 仍然持有")
        L.append(f"　{t['買入日'].date()} 買入 {res}；提早階段回報 {t.get('提早階段回報%', float('nan')):+.1f}%")
    L.append("")

    # 2. 隔籬參數
    base_t = base_plus(spy, ma200, f, None, None)
    base = {pn: stats(run(spy, cash_r, base_t), a, b) for pn, a, b in PERIODS}
    L.append(f"--- 2. 隔籬參數（基本規則：全期 {base['全期']['cagr']:+.2f}%，前半 {base['前半']['cagr']:+.2f}%，"
             f"後半 {base['後半']['cagr']:+.2f}%，跌 {base['全期']['mdd']:.1f}%）---")
    rows = []
    for r in [25, 30, 35]:
        for v in [25, 30, 35]:
            for q in [1.5, 2.0, 2.5]:
                t = base_plus(spy, ma200, f, sig(r, v, q), "pct5")
                d = run(spy, cash_r, t)
                s = {pn: stats(d, a, b) for pn, a, b in PERIODS}
                ok = (s["前半"]["cagr"] > base["前半"]["cagr"] and s["後半"]["cagr"] > base["後半"]["cagr"]
                      and s["全期"]["mdd"] > -30)
                rows.append((r, v, q, s["全期"]["cagr"], s["前半"]["cagr"], s["後半"]["cagr"], s["全期"]["mdd"], ok))
                L.append(f"　RSI<{r} VIX≥{v} 成交量≥{q}倍：全期 {s['全期']['cagr']:+.2f}% | 前半 {s['前半']['cagr']:+.2f}% | "
                         f"後半 {s['後半']['cagr']:+.2f}% | 跌 {s['全期']['mdd']:.1f}% | {'✅' if ok else '❌'}")
    g = pd.DataFrame(rows, columns=["r", "v", "q", "full", "h1", "h2", "mdd", "ok"])
    L.append(f"　小結：27個入面 {g['ok'].sum()} 個符合採用條件；全期回報好過基本規則 {(g['full'] > base['全期']['cagr']).sum()}/27；"
             f"全期回報中位數 {g['full'].median():+.2f}%；最大跌幅中位數 {g['mdd'].median():.1f}%")
    for stop in [0.03, 0.05, 0.08]:
        t, _ = base_plus_log(spy, ma200, sig(30, 30, 2.0), stop)
        d = run(spy, cash_r, t)
        s = {pn: stats(d, a, b) for pn, a, b in PERIODS}
        L.append(f"　止損{stop:.0%}：全期 {s['全期']['cagr']:+.2f}% | 前半 {s['前半']['cagr']:+.2f}% | "
                 f"後半 {s['後半']['cagr']:+.2f}% | 跌 {s['全期']['mdd']:.1f}%")
    L.append("")

    # 3. 逐年
    L.append("--- 3. 逐年回報：基本規則 vs E4 vs SPY長揸（只列有分別嘅年份）---")
    de = run(spy, cash_r, tgt)
    db = run(spy, cash_r, base_t)
    dh = run(spy, cash_r, pd.Series(1.0, index=spy.index))
    yr = lambda d: (1 + d[d.index >= pd.Timestamp(START)]).groupby(d[d.index >= pd.Timestamp(START)].index.year).prod() - 1
    ye, yb, yh = yr(de), yr(db), yr(dh)
    better = worse = 0
    for y in ye.index:
        if abs(ye[y] - yb[y]) > 0.0005:
            better += ye[y] > yb[y]
            worse += ye[y] < yb[y]
            L.append(f"　{y}：基本 {yb[y] * 100:+.1f}% | E4 {ye[y] * 100:+.1f}% | SPY {yh[y] * 100:+.1f}%")
    L.append(f"　E4好過基本規則 {better} 年，差過 {worse} 年")

    pd.DataFrame({"report": L}).to_csv("spy_bottoms_v2_check.csv", index=False, encoding="utf-8-sig")
    print("\n".join(L))


if __name__ == "__main__":
    main()
