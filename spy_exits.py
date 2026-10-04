"""
spy_exits.py — SPY止損位研究：提早止損 vs 跌穿200日線反而加倉

⚠️ 預先登記：以下規則喺第一次跑之前已經定好，跑完唔再改。

共同設定
- 數據：SPY（已計股息）由1993-01-29；測試期1994-01至2026-09（1993年用嚟計均線）。
- 每日收市判斷，下一個交易日收市先交易（遲一日，貼近真實）。
- 交易成本：每次買或賣0.1%。現金收益：13週美國國債息。
- 前後兩段：前半1994-2012（有2000科網爆破、2008金融海嘯）；後半2013-2026/09。
- 唔用槓桿：任何時候持倉最多100%。

A. 提早止損（賣晒轉現金，再買返就100% SPY）
  A1 均線 + 緩衝：均線日數 N = 20、50、100、150、200；緩衝 b = 0%、1%、2%、3%
     收市低過 均線×(1−b) 就賣；收市高過 均線×(1+b) 先買返。（20日只試b=0%）共17個。
  A2 由高位回落止損：持有期間，收市比「買入後最高收市價」低 X% 就賣，
     X = 3%、5%、8%、10%、15%、20%；賣出後，收市創 N 日新高先買返，N = 20、50。共12個。
  A3 月線法：每個月最後一日收市睇SPY有冇高過10個月均線（約200日），有就持有。共1個。
  揀法（預先定好）：用前半段，揀「最大跌幅好過 -30%」入面回報最高嗰個，原封不動睇後半段。

B. 跌穿200日線反而買入
  B0 描述：由「高過200日線」同「低過200日線」嘅日子開始買，之後1、3、6、12個月嘅回報
     （平均、中位數、賺錢機會、最差）。另外睇「低過200日線10%以上」。
  B1 策略：平時（高過200日線）只持有 w% SPY，其餘現金；跌穿200日線就買到100%。
     w = 50%、70%。另一版本：要低過200日線10%以上先買到100%。
     公平比較：同一個平均持倉比例嘅「固定比例」（例如一直持有85%），同100% SPY長揸。

==== 結果（2026-10 第一次跑，記錄低，唔再改參數）====
- SPY長揸1994-2026：+10.85%/年，最大跌 -55.2%。
- 提早止損反而差：均線50日 +4.17%（跌 -38%）、20日 +2.30%；高位回落3%止損 +4.18~4.82%。
  原因：細跌都賣，之後高價買返，來回蝕（50日線每年買賣17.5次）。
- 有用嘅係「緩衝」：200日線 緩衝0% +8.72%（跌 -23.7%，每年6.5次）
  → 緩衝2% +10.15%（跌 -19.1%，1.5次）→ 緩衝3% +10.38%（跌 -20.4%，1.0次）；前後兩半方向一致。
- 預先定好嘅揀法：前半揀中「均線150日 緩衝3%」（前半 +10.29%）→ 後半 +10.97%、跌 -20.0%，排11/30。
  0/30個規則喺後半贏SPY長揸回報（+14.92%），但跌幅細好多。
- 跌穿200日線加倉：最大跌幅大增（-47%至-53%），回報唔好過同平均持倉嘅固定比例 → 唔建議。
- 交易次數少（200日線緩衝2%：33年只有24次來回），0.5%以內嘅差距可以係運氣。

輸出：spy_exits_results.csv（報告）、spy_exits_grid.csv（全部組合）
用法：python spy_exits.py            （用yfinance下載）
      python spy_exits.py --csv data/spy_equity.csv  （用etf_rotation.py輸出嘅數據重建）
"""
import sys
import numpy as np
import pandas as pd

COST = 0.001
START, SPLIT, END = "1994-01-01", "2012-12-31", "2026-09-30"
PERIODS = [("全期", START, END), ("前半", START, SPLIT), ("後半", "2013-01-01", END)]


# ---------------- 數據 ----------------
def load_yf():
    import yfinance as yf

    def close_of(t):
        df = yf.download(t, start="1992-06-01", end="2026-10-01", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df["Close"].astype(float).dropna()
    spy = close_of("SPY")
    irx = close_of("^IRX").reindex(spy.index).ffill().fillna(0.0)
    return spy, irx / 100 / 252


def load_csv(path):
    """由etf_rotation.py嘅etf_equity.csv重建：SPY 1倍 = SPY價格（已計息）；
    現金息由「SPY 1倍 + 200日線濾網」喺空倉日子嘅每日回報反推，持倉日子用前後值填補。"""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date", encoding="utf-8-sig")
    first = pd.Series([1.0], index=[pd.Timestamp("1993-01-29")])
    spy = pd.concat([first, df["SPY 1倍"].astype(float)])
    flt = pd.concat([first, df["SPY 1倍 + 200日線濾網"].astype(float)])
    ok = (spy > spy.rolling(200).mean()).astype(float)
    inm = ok.shift(2).fillna(0.0)
    switch = inm.diff().abs().fillna(0.0)
    fr = flt.pct_change()
    cash = fr.where((inm == 0) & (switch == 0)).clip(lower=0, upper=0.001)
    cash = cash.ffill().bfill().fillna(0.0)
    return spy, cash


# ---------------- 模擬 ----------------
def run(spy, cash_r, target):
    """target：每日收市決定嘅目標持倉比例(0-1)；遲兩格先食到回報（收市判斷→下日收市交易→再下日回報）。"""
    r = spy.pct_change().fillna(0.0)
    w = target.shift(2).fillna(0.0)
    dw = w.diff().abs().fillna(0.0)
    daily = w * r + (1 - w) * cash_r - dw * COST
    return daily


def stats(daily, a, b):
    d = daily[(daily.index >= pd.Timestamp(a)) & (daily.index <= pd.Timestamp(b))]
    eq = (1 + d).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    v = np.concatenate([[1.0], eq.values])
    mdd = (v / np.maximum.accumulate(v) - 1).min() * 100
    yearly = eq.groupby(eq.index.year).last()
    prev = np.concatenate([[1.0], yearly.values[:-1]])
    return {"cagr": (eq.iloc[-1] ** (1 / yrs) - 1) * 100, "mdd": mdd,
            "worst_year": ((yearly.values / prev - 1) * 100).min(), "mult": eq.iloc[-1]}


def ma_band(spy, n, b):
    ma = spy.rolling(n).mean()
    pos, out = 0.0, []
    for p, m in zip(spy.values, ma.values):
        if np.isnan(m):
            out.append(0.0)
            continue
        if pos == 1.0 and p < m * (1 - b):
            pos = 0.0
        elif pos == 0.0 and p > m * (1 + b):
            pos = 1.0
        out.append(pos)
    return pd.Series(out, index=spy.index)


def trailing(spy, x, n):
    hi_n = spy.rolling(n).max()
    pos, peak, out = 0.0, 0.0, []
    for i, p in enumerate(spy.values):
        if i < 200:                       # 同其他規則一樣，等夠數據先開始
            out.append(0.0)
            continue
        if pos == 1.0:
            peak = max(peak, p)
            if p < peak * (1 - x):
                pos = 0.0
        elif p >= hi_n.iloc[i]:           # 收市創N日新高 → 買返
            pos, peak = 1.0, p
        out.append(pos)
    return pd.Series(out, index=spy.index)


def monthly10(spy):
    m = spy.resample("ME").last()
    sig = (m > m.rolling(10).mean()).astype(float)
    sig.index = [spy.index[spy.index <= d][-1] for d in sig.index]   # 月尾最後交易日
    return sig.reindex(spy.index).ffill().fillna(0.0)


def switches_per_year(target, a, b):
    t = target[(target.index >= pd.Timestamp(a)) & (target.index <= pd.Timestamp(b))]
    yrs = (t.index[-1] - t.index[0]).days / 365.25
    return t.diff().abs().fillna(0.0).sum() / yrs


def fmt(name, s, extra=""):
    return (f"　{name}：每年 {s['cagr']:+.2f}%，最大跌幅 {s['mdd']:.1f}%，最差一年 {s['worst_year']:+.1f}%，"
            f"$80,000 → ${80000 * s['mult']:,.0f}{extra}")


def main():
    if "--csv" in sys.argv:
        spy, cash_r = load_csv(sys.argv[sys.argv.index("--csv") + 1])
    else:
        spy, cash_r = load_yf()
    spy = spy[spy.index <= pd.Timestamp(END)]
    cash_r = cash_r.reindex(spy.index).fillna(0.0)

    # ---------- A. 止損規則 ----------
    rules = {}
    for n in [20, 50, 100, 150, 200]:
        for b in ([0.0] if n == 20 else [0.0, 0.01, 0.02, 0.03]):
            rules[f"均線{n}日 緩衝{b:.0%}"] = ma_band(spy, n, b)
    for x in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        for n in [20, 50]:
            rules[f"高位回落{x:.0%}止損 / 創{n}日新高買返"] = trailing(spy, x, n)
    rules["月線法（10個月均線）"] = monthly10(spy)
    hold = pd.Series(1.0, index=spy.index)

    rows = []
    for name, tgt in rules.items():
        d = run(spy, cash_r, tgt)
        row = {"規則": name}
        for pn, a, b in PERIODS:
            s = stats(d, a, b)
            row[f"{pn}回報%"] = round(s["cagr"], 2)
            row[f"{pn}最大跌幅%"] = round(s["mdd"], 1)
        row["全期最差一年%"] = round(stats(d, START, END)["worst_year"], 1)
        row["每年買賣次數"] = round(switches_per_year(tgt, START, END), 1)
        rows.append(row)
    grid = pd.DataFrame(rows)
    grid.to_csv("spy_exits_grid.csv", index=False, encoding="utf-8-sig")

    L = ["=== SPY止損位研究（規則預先登記，見spy_exits.py開頭）===",
         "測試期1994-01至2026-09；前半1994-2012，後半2013-2026/09；每次買賣0.1%；唔用槓桿", ""]
    L.append("=== 比較對象：SPY長揸 ===")
    dh = run(spy, cash_r, hold)
    for pn, a, b in PERIODS:
        L.append(fmt(f"{pn} SPY長揸", stats(dh, a, b)))
    L.append("")

    L.append("=== A. 止損規則（全部列出）===")
    L.append("　規則 | 全期回報 / 最大跌幅 | 前半 | 後半 | 每年買賣次數")
    spy_s = {pn: stats(dh, a, b) for pn, a, b in PERIODS}
    for _, r in grid.iterrows():
        L.append(f"　{r['規則']} | {r['全期回報%']:+.2f}% / {r['全期最大跌幅%']:.1f}% | "
                 f"{r['前半回報%']:+.2f}% / {r['前半最大跌幅%']:.1f}% | "
                 f"{r['後半回報%']:+.2f}% / {r['後半最大跌幅%']:.1f}% | {r['每年買賣次數']:.1f}")
    L.append("")
    L.append(f"　贏SPY長揸回報：全期 {(grid['全期回報%'] > spy_s['全期']['cagr']).sum()}/{len(grid)}，"
             f"前半 {(grid['前半回報%'] > spy_s['前半']['cagr']).sum()}/{len(grid)}，"
             f"後半 {(grid['後半回報%'] > spy_s['後半']['cagr']).sum()}/{len(grid)}")
    ok = grid[grid["前半最大跌幅%"] > -30]
    if len(ok):
        pick = ok.sort_values("前半回報%", ascending=False).iloc[0]
        rank = int((grid["後半回報%"] > pick["後半回報%"]).sum()) + 1
        L.append(f"　預先定好嘅揀法 → 前半揀中：{pick['規則']}（前半 {pick['前半回報%']:+.2f}%，跌幅 {pick['前半最大跌幅%']:.1f}%）")
        L.append(f"　　→ 後半：{pick['後半回報%']:+.2f}%，最大跌幅 {pick['後半最大跌幅%']:.1f}%，"
                 f"後半排第{rank}/{len(grid)}（SPY長揸後半 {spy_s['後半']['cagr']:+.2f}%）")
    L.append("")

    # ---------- B0. 跌穿200日線之後嘅回報 ----------
    L.append("=== B0. 由邊種日子開始買，之後回報點？（SPY已計息，1994-2026）===")
    ma200 = spy.rolling(200).mean()
    gap = spy / ma200 - 1
    mask_all = (spy.index >= pd.Timestamp(START))
    groups = {"高過200日線": (gap > 0), "低過200日線": (gap <= 0), "低過200日線10%以上": (gap <= -0.10)}
    for h, lab in [(21, "1個月"), (63, "3個月"), (126, "6個月"), (252, "12個月")]:
        fwd = spy.shift(-h) / spy - 1
        L.append(f"--- 之後{lab} ---")
        for gname, g in groups.items():
            f = fwd[g & mask_all].dropna() * 100
            if len(f) == 0:
                continue
            L.append(f"　{gname}（{len(f)}日）：平均 {f.mean():+.1f}%，中位數 {f.median():+.1f}%，"
                     f"賺錢機會 {(f > 0).mean() * 100:.0f}%，最差 {f.min():+.1f}%，最好 {f.max():+.1f}%")
    L.append("　⚠️ 相連日子嘅結果重疊好多，日數睇落多，實際獨立例子少好多。")
    L.append("")

    # ---------- B1. 跌穿200日線加倉策略 ----------
    L.append("=== B1. 平時持有部分，跌穿200日線買到100% ===")
    below = (spy <= ma200).astype(float)
    deep = (gap <= -0.10).astype(float)
    b_rules = {}
    for w in [0.5, 0.7]:
        b_rules[f"平時{w:.0%}，跌穿200日線買到100%"] = w + (1 - w) * below
        b_rules[f"平時{w:.0%}，低過200日線10%以上先買到100%"] = w + (1 - w) * deep
    for name, tgt in b_rules.items():
        tgt = tgt.where(ma200.notna(), 0.0)
        avg = tgt[tgt.index >= pd.Timestamp(START)].mean()
        d = run(spy, cash_r, tgt)
        dfix = run(spy, cash_r, pd.Series(avg, index=spy.index).where(ma200.notna(), 0.0))
        L.append(f"--- {name}（平均持倉 {avg:.0%}）---")
        for pn, a, b in PERIODS:
            L.append(fmt(f"{pn} 呢個策略", stats(d, a, b)))
            L.append(fmt(f"{pn} 一直持有{avg:.0%}（公平比較）", stats(dfix, a, b)))
    L.append("")

    pd.DataFrame({"report": L}).to_csv("spy_exits_results.csv", index=False, encoding="utf-8-sig")
    print("\n".join(L))


if __name__ == "__main__":
    main()
