"""
spy_bottoms.py — SPY見底反彈有咩特徵？知唔知幾時可以全倉買入？

⚠️ 預先登記（跑之前定好，跑完唔改）：

第一部分：列出1994-2026年每一次SPY由高位跌10%以上（收市價計），
  記錄：高位日期、最低位日期、跌幅、跌咗幾多日、幾耐升返舊高。
  同埋最低嗰日嘅特徵（全部係當日已經知道嘅數據）：
  RSI(14日)、低過200日線幾多%、低過50日線幾多%、20日波幅（年化）、過去5日/21日回報、連跌日數。
  ⚠️ 事後先知邊日係最低位，所以呢部分只係描述，唔可以直接當買入信號。

第二部分（誠實測試）：「信號出現 → 係咪真係底？」
  信號（全部係當時睇得到）：
   S1 RSI(14) < 30
   S2 由一年高位跌20%以上
   S3 低過200日線10%以上
   S4 20日波幅（年化）高過35%
   S5 5日內跌8%以上
  每個信號第一次出現當一次事件（之後60個交易日內再出現唔重複計）。
  每次事件睇：之後再跌幾多先見真正低位（12個月內）、12個月後回報。

第三部分：「確認反彈先買返」策略（唔用槓桿）
  基本規則：200日線緩衝2%（跌穿200日線2%賣，升返高過200日線2%買返）——上次研究結果。
  加一條提早買返：空倉期間，如果SPY由空倉後最低位反彈 Y%，就提早全倉買返，Y = 5%、10%、15%。
   提早買返之後：升到高過200日線2% → 變返正常持有；跌穿之前嗰個最低位 → 即賣（反彈失敗）。
  採用條件（預先定好）：前後兩段回報都要好過基本規則，而且最大跌幅唔可以差過 -30%。

數據：data/spy_equity.csv（SPY已計息收市價，由etf_rotation.py輸出重建）。只有收市價，冇成交量同VIX。

==== 結果（2026-10 第一次跑，記錄低，唔再改）====
- 12次跌10%以上。見底日特徵：RSI中位數29（平時56）、20日波幅27%（平時14%）、5日跌6.6%。
  但每次差好遠：距200日線由 +2.3% 到 -35.1%；波幅由20%到82%。
- 信號出現當日買入，之後仍然再跌（中位數）：RSI<30 -6.5%、一年高位跌20% -16.8%、
  低過200日線10% -9.7%、波幅>35% -7.8%、5日跌8% -5.9%；最壞全部再跌 -38% 至 -46%（2008）。
- 確認反彈提早買返（5%/10%/15%）：全部唔符合採用條件；2008年反彈5%版本假反彈買咗7次，
  最大跌幅 -57%（基本規則 -19%）。只喺後半段（冇長熊市）回報較高。
"""
import numpy as np
import pandas as pd

from spy_exits import load_csv, run, stats, PERIODS, START

DATA = "data/spy_equity.csv"


def rsi(p, n=14):
    d = p.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def features(spy):
    f = pd.DataFrame(index=spy.index)
    f["RSI14"] = rsi(spy)
    f["距200日線%"] = (spy / spy.rolling(200).mean() - 1) * 100
    f["距50日線%"] = (spy / spy.rolling(50).mean() - 1) * 100
    f["20日波幅%"] = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100
    f["5日回報%"] = (spy / spy.shift(5) - 1) * 100
    f["21日回報%"] = (spy / spy.shift(21) - 1) * 100
    f["距一年高位%"] = (spy / spy.rolling(252, min_periods=1).max() - 1) * 100
    down = (spy.diff() < 0).astype(int)
    f["連跌日數"] = down.groupby((down == 0).cumsum()).cumsum()
    return f


def episodes(spy, thresh=0.10):
    """由高位跌thresh以上嘅每一段：高位→最低→升返舊高。"""
    s = spy[spy.index >= pd.Timestamp(START)]
    out, peak, peak_d, low, low_d, inside = [], s.iloc[0], s.index[0], None, None, False
    for d, p in s.items():
        if p >= peak:
            if inside:
                out.append((peak_d, low_d, low / peak - 1, d))
                inside = False
            peak, peak_d, low, low_d = p, d, p, d
        else:
            if low is None or p < low:
                low, low_d = p, d
            if p / peak - 1 <= -thresh:
                inside = True
    if inside:
        out.append((peak_d, low_d, low / peak - 1, None))
    return out


def signal_events(sig, gap=60):
    ev, last = [], -10**9
    idx = list(sig.index)
    for i, (d, v) in enumerate(sig.items()):
        if v and i - last > gap:
            ev.append(d)
        if v:
            last = i
    return ev


def early_reentry(spy, y, band=0.02, n=200):
    ma = spy.rolling(n).mean()
    state, low, out = "out", None, []
    for p, m in zip(spy.values, ma.values):
        if np.isnan(m):
            out.append(0.0)
            continue
        if state == "in":
            if p < m * (1 - band):
                state, low = "out", p
        elif state == "out":
            low = p if low is None else min(low, p)
            if p > m * (1 + band):
                state = "in"
            elif y is not None and p >= low * (1 + y):
                state = "early"
        elif state == "early":
            if p > m * (1 + band):
                state = "in"
            elif p < low:
                state, low = "out", p
        out.append(0.0 if state == "out" else 1.0)
    return pd.Series(out, index=spy.index)


def main():
    spy, cash_r = load_csv(DATA)
    f = features(spy)
    L = ["=== SPY見底反彈研究（規則預先登記，見spy_bottoms.py開頭）===", ""]

    # ---------- 第一部分 ----------
    eps = episodes(spy)
    L.append(f"=== 第一部分：1994-2026年SPY跌10%以上共{len(eps)}次（⚠️事後先知邊日係底）===")
    L.append("　高位 → 最低位 | 跌幅 | 跌咗幾多個交易日 | 幾耐升返舊高 | 最低嗰日：RSI / 距200日線 / 20日波幅 / 5日回報 / 21日回報")
    rows = []
    for pk, lo, dd, rec in eps:
        days_down = len(spy.loc[pk:lo]) - 1
        rec_txt = f"{len(spy.loc[lo:rec]) - 1}日" if rec is not None else "未升返"
        r = f.loc[lo]
        rows.append(r)
        L.append(f"　{pk.date()} → {lo.date()} | {dd * 100:.1f}% | {days_down}日 | {rec_txt} | "
                 f"{r['RSI14']:.0f} / {r['距200日線%']:+.1f}% / {r['20日波幅%']:.0f}% / "
                 f"{r['5日回報%']:+.1f}% / {r['21日回報%']:+.1f}%")
    bt = pd.DataFrame(rows)
    allf = f[f.index >= pd.Timestamp(START)]
    L.append("")
    L.append("　最低嗰日嘅特徵 vs 平時（所有日子）：")
    for c in ["RSI14", "距200日線%", "20日波幅%", "5日回報%", "21日回報%"]:
        L.append(f"　{c}：見底日中位數 {bt[c].median():.1f}，範圍 {bt[c].min():.1f} 至 {bt[c].max():.1f}；"
                 f"平時中位數 {allf[c].median():.1f}")
    L.append("")

    # ---------- 第二部分 ----------
    L.append("=== 第二部分：信號出現嗰日買入，係咪真係底？===")
    sigs = {
        "S1 RSI低過30": f["RSI14"] < 30,
        "S2 由一年高位跌20%以上": f["距一年高位%"] <= -20,
        "S3 低過200日線10%以上": f["距200日線%"] <= -10,
        "S4 20日波幅高過35%": f["20日波幅%"] >= 35,
        "S5 5日內跌8%以上": f["5日回報%"] <= -8,
    }
    for name, sg in sigs.items():
        sg = sg & (spy.index >= pd.Timestamp(START))
        ev = signal_events(sg)
        further, fwd12 = [], []
        L.append(f"--- {name}：共{len(ev)}次 ---")
        for d in ev:
            i = spy.index.get_loc(d)
            win = spy.iloc[i:i + 253]
            fall = (win.min() / spy.iloc[i] - 1) * 100
            r12 = (spy.iloc[i + 252] / spy.iloc[i] - 1) * 100 if i + 252 < len(spy) else np.nan
            further.append(fall)
            fwd12.append(r12)
            r12t = f"{r12:+.1f}%" if not np.isnan(r12) else "未夠12個月"
            L.append(f"　{d.date()}：之後再跌 {fall:.1f}% 先見底；12個月後 {r12t}")
        fu, fw = np.array(further), np.array([x for x in fwd12 if not np.isnan(x)])
        L.append(f"　小結：再跌中位數 {np.median(fu):.1f}%，最多再跌 {fu.min():.1f}%；"
                 f"再跌少過5%嘅次數 {(fu > -5).sum()}/{len(fu)}；"
                 f"12個月後賺錢 {(fw > 0).sum()}/{len(fw)}，平均 {fw.mean():+.1f}%")
    L.append("")

    # ---------- 第三部分 ----------
    L.append("=== 第三部分：確認反彈先提早買返（基本規則：200日線緩衝2%）===")
    hold = run(spy, cash_r, pd.Series(1.0, index=spy.index))
    variants = {"SPY長揸": None, "基本規則（200日線緩衝2%）": early_reentry(spy, None)}
    for y in [0.05, 0.10, 0.15]:
        variants[f"基本 + 由低位反彈{y:.0%}提早買返"] = early_reentry(spy, y)
    res = {}
    for name, tgt in variants.items():
        d = hold if tgt is None else run(spy, cash_r, tgt)
        res[name] = {pn: stats(d, a, b) for pn, a, b in PERIODS}
        parts = " | ".join(f"{pn} {res[name][pn]['cagr']:+.2f}%（跌 {res[name][pn]['mdd']:.1f}%）"
                           for pn, _, _ in PERIODS)
        L.append(f"　{name}：{parts}")
    base = res["基本規則（200日線緩衝2%）"]
    L.append("")
    for name in [k for k in variants if k.startswith("基本 +")]:
        r = res[name]
        ok = (r["前半"]["cagr"] > base["前半"]["cagr"] and r["後半"]["cagr"] > base["後半"]["cagr"]
              and r["全期"]["mdd"] > -30)
        L.append(f"　{name}：{'✅ 符合採用條件' if ok else '❌ 唔符合採用條件'}")

    pd.DataFrame({"report": L}).to_csv("spy_bottoms_results.csv", index=False, encoding="utf-8-sig")
    print("\n".join(L))


if __name__ == "__main__":
    main()
