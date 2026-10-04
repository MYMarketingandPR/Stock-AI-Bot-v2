"""
spy_indicators.py — 用唔同技術指標買賣SPY：邊個賺最多、蝕最少？

⚠️ 預先登記（跑之前定好，跑完唔改）：

共同設定
- SPY（已計股息，OHLC）+ ^IRX現金息，yfinance下載；測試期1994-01至2026-09。
- 前半1994-2012，後半2013-2026/09。只做多或現金，唔沽空，唔用槓桿。
- 主要計法：收市判斷，下一日收市交易；每次買賣成本0.1%。
- 敏感度（只供參考，唔用嚟揀）：成本0.02%（IB買SPY實際大約咁）；同埋「當日收市即刻交易」。

指標規則（全部都試）
趨勢類（條件成立就持有）：
  MACD(12,26,9)、MACD(8,17,9)、MACD(19,39,9)：MACD線高過訊號線
  MACD(12,26)線高過0
  KDJ(9,3,3)：K高過D
  收市高過簡單均線 10/20/50/100/200日
  EMA快線高過慢線：5/20、9/21、10/50、20/50、50/200
  ADX(14)>20 而且 +DI > −DI
  保力加突破：收市高過上軌買，跌穿中軌賣
  唐奇安通道：創20日新高買、跌穿10日新低賣；創55日新高買、跌穿20日新低賣
逆勢類（超賣買、反彈賣）：
  保力加：收市低過下軌買，升穿中軌賣（20日，2個標準差）
  KDJ：J<0買，J>100賣
  KDJ：K<20而且K上穿D買，K>80賣
  RSI(14)：<30買，>70賣
  RSI(2)：<10買，>70賣
  CCI(20)：<−100買，>+100賣
  威廉指標%R(14)：<−90買，>−20賣
每條規則再試「加200日線濾網」版本：收市低過200日線就唔持有。
逆勢類另外試「合併」版本：基本規則（200日線緩衝2%）持有 或者 逆勢信號持有，任何一個成立就持有。
另加「投票」：19條趨勢規則入面一半以上話持有就持有。
比較對象：SPY長揸、基本規則（200日線緩衝2%）。

揀法（預先定好）：
  用前半段，喺「最大跌幅好過 -30%」嘅規則入面揀回報最高，原封不動睇後半段同排名。
  另外列出：前後兩段回報都好過基本規則、而且全期最大跌幅好過 -30% 嘅規則。
⚠️ 試咗60幾條規則，一定有啲係撞彩靚，所以只信「前半揀、後半驗」同埋「成個家族都一致」嘅結果。

輸出：spy_indicators_results.csv（報告）、spy_indicators_grid.csv（全部規則）
==== 結果（2026-10 第一次跑，GitHub run 37200504646，記錄低，唔再改）====
- 60條規則：4條全期贏SPY長揸（+10.85%），全部係「合併：基本規則 或 逆勢」，但最大跌幅-41%至-55%。
- 單獨用指標全部輸：MACD線>訊號線 +1.8%、KDJ K>D -2.1%（每年買賣46次）、保力加突破 +2.3%；買賣越密越差。
- 逆勢+200日線濾網：跌幅好細（-4%至-21%）但只持倉1%-22%時間，回報+2.8%至+5.5%。
- 誠實測試：前半揀中「基本規則 或 KDJ K<20上穿D買/K>80賣」前半+10.89% → 後半+12.24%（跌-28.9%），排6/60；
  同時係唯一前後兩段都贏基本規則而跌幅好過-30%嘅規則。全期+11.45%。
"""
import numpy as np
import pandas as pd

START, SPLIT, END = "1994-01-01", "2012-12-31", "2026-09-30"
PERIODS = [("全期", START, END), ("前半", START, SPLIT), ("後半", "2013-01-01", END)]


def load():
    import yfinance as yf

    def dl(t):
        df = yf.download(t, start="1992-06-01", end="2026-10-01", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    s = dl("SPY")[["Open", "High", "Low", "Close"]].astype(float).dropna()
    irx = dl("^IRX")["Close"].astype(float).reindex(s.index).ffill().fillna(0.0)
    s = s[s.index <= pd.Timestamp(END)]
    return s, (irx / 100 / 252).reindex(s.index)


# ---------- 指標 ----------
def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def kdj(h, l, c, n=9):
    rsv = (c - l.rolling(n).min()) / (h.rolling(n).max() - l.rolling(n).min()) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md)


def enter_exit(entry, exit_):
    """入場條件成立就買，出場條件成立就賣，其餘時間保持。"""
    pos, out = 0.0, []
    for e, x in zip(entry.fillna(False).values, exit_.fillna(False).values):
        if pos == 0.0 and e:
            pos = 1.0
        elif pos == 1.0 and x:
            pos = 0.0
        out.append(pos)
    return pd.Series(out, index=entry.index)


def ma_band(c, n=200, b=0.02):
    ma = c.rolling(n).mean()
    return enter_exit(c > ma * (1 + b), c < ma * (1 - b))


# ---------- 模擬 ----------
def simulate(c, cash_r, target, cost, lag):
    r = c.pct_change().fillna(0.0)
    w = target.shift(lag).fillna(0.0)
    dw = w.diff().abs().fillna(0.0)
    return w * r + (1 - w) * cash_r.fillna(0.0) - dw * cost


def stats(daily, a, b):
    d = daily[(daily.index >= pd.Timestamp(a)) & (daily.index <= pd.Timestamp(b))]
    eq = (1 + d).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    v = np.concatenate([[1.0], eq.values])
    return (eq.iloc[-1] ** (1 / yrs) - 1) * 100, (v / np.maximum.accumulate(v) - 1).min() * 100, eq.iloc[-1]


def build_rules(s):
    o, h, l, c = s["Open"], s["High"], s["Low"], s["Close"]
    sma200 = c.rolling(200).mean()
    trend, mr = {}, {}
    for f_, sl, sg in [(12, 26, 9), (8, 17, 9), (19, 39, 9)]:
        m = ema(c, f_) - ema(c, sl)
        trend[f"MACD({f_},{sl},{sg}) 線>訊號線"] = (m > ema(m, sg)).astype(float)
    trend["MACD(12,26) 線>0"] = (ema(c, 12) - ema(c, 26) > 0).astype(float)
    k, d, j = kdj(h, l, c)
    trend["KDJ K>D"] = (k > d).astype(float)
    for n in [10, 20, 50, 100, 200]:
        trend[f"收市>SMA{n}"] = (c > c.rolling(n).mean()).astype(float)
    for a_, b_ in [(5, 20), (9, 21), (10, 50), (20, 50), (50, 200)]:
        trend[f"EMA{a_}>EMA{b_}"] = (ema(c, a_) > ema(c, b_)).astype(float)
    ax, pdi, mdi = adx(h, l, c)
    trend["ADX>20 且 +DI>−DI"] = ((ax > 20) & (pdi > mdi)).astype(float)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    up_, lo_ = mid + 2 * sd, mid - 2 * sd
    trend["保力加突破（上軌買/中軌賣）"] = enter_exit(c > up_, c < mid)
    trend["唐奇安 20日高買/10日低賣"] = enter_exit(c >= h.rolling(20).max(), c <= l.rolling(10).min())
    trend["唐奇安 55日高買/20日低賣"] = enter_exit(c >= h.rolling(55).max(), c <= l.rolling(20).min())

    mr["保力加逆勢（下軌買/中軌賣）"] = enter_exit(c < lo_, c > mid)
    mr["KDJ J<0買/J>100賣"] = enter_exit(j < 0, j > 100)
    mr["KDJ K<20上穿D買/K>80賣"] = enter_exit((k < 20) & (k > d) & (k.shift() <= d.shift()), k > 80)
    r14, r2 = rsi(c, 14), rsi(c, 2)
    mr["RSI14 <30買/>70賣"] = enter_exit(r14 < 30, r14 > 70)
    mr["RSI2 <10買/>70賣"] = enter_exit(r2 < 10, r2 > 70)
    cc = cci(h, l, c)
    mr["CCI <−100買/>100賣"] = enter_exit(cc < -100, cc > 100)
    wr = (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min()) * -100
    mr["威廉%R <−90買/>−20賣"] = enter_exit(wr < -90, wr > -20)

    ready = sma200.notna()
    above = (c > sma200).astype(float)
    base = ma_band(c)
    rules = {}
    for name, t in trend.items():
        rules[("趨勢", name)] = t.where(ready, 0.0)
        rules[("趨勢+200日線濾網", name)] = (t * above).where(ready, 0.0)
    for name, t in mr.items():
        rules[("逆勢", name)] = t.where(ready, 0.0)
        rules[("逆勢+200日線濾網", name)] = (t * above).where(ready, 0.0)
        rules[("合併：基本規則 或 逆勢", name)] = np.maximum(base, t).where(ready, 0.0)
    vote = pd.concat(list(trend.values()), axis=1).mean(axis=1)
    rules[("投票", "一半以上趨勢規則話持有")] = (vote > 0.5).astype(float).where(ready, 0.0)
    return rules, base.where(ready, 0.0)


def main():
    s, cash_r = load()
    c = s["Close"]
    rules, base = build_rules(s)
    hold = pd.Series(1.0, index=c.index)

    def evaluate(t):
        row = {}
        for pn, a, b in PERIODS:
            cg, dd, mult = stats(simulate(c, cash_r, t, 0.001, 2), a, b)
            row[f"{pn}回報%"], row[f"{pn}最大跌幅%"] = round(cg, 2), round(dd, 1)
            if pn == "全期":
                row["$80,000變"] = round(80000 * mult)
        row["全期回報%(成本0.02%)"] = round(stats(simulate(c, cash_r, t, 0.0002, 2), START, END)[0], 2)
        row["全期回報%(當日收市交易)"] = round(stats(simulate(c, cash_r, t, 0.001, 1), START, END)[0], 2)
        tt = t[t.index >= pd.Timestamp(START)]
        row["每年買賣次數"] = round(tt.diff().abs().sum() / ((tt.index[-1] - tt.index[0]).days / 365.25), 1)
        row["持倉時間%"] = round(tt.mean() * 100)
        return row

    bench = {"SPY長揸": evaluate(hold), "基本規則（200日線緩衝2%）": evaluate(base)}
    rows = []
    for (fam, name), t in rules.items():
        rows.append({"類別": fam, "規則": name, **evaluate(t)})
    g = pd.DataFrame(rows)
    g.to_csv("spy_indicators_grid.csv", index=False, encoding="utf-8-sig")

    b = bench["基本規則（200日線緩衝2%）"]
    L = ["=== SPY技術指標大測試（規則預先登記，見spy_indicators.py開頭）===",
         "測試期1994-01至2026-09；前半1994-2012，後半2013-2026/09；收市判斷、下一日收市交易；每次買賣0.1%；只做多，唔用槓桿", ""]
    L.append("=== 比較對象 ===")
    for k, v in bench.items():
        L.append(f"　{k}：全期 {v['全期回報%']:+.2f}%（跌 {v['全期最大跌幅%']:.1f}%）| 前半 {v['前半回報%']:+.2f}% | "
                 f"後半 {v['後半回報%']:+.2f}% | ${v['$80,000變']:,} | 每年買賣 {v['每年買賣次數']}")
    L.append("")
    L.append(f"=== 全部{len(g)}條規則（按全期回報排）===")
    L.append("　類別 | 規則 | 全期回報/跌幅 | 前半 | 後半 | 每年買賣 | 持倉時間 | 成本0.02%時全期 | 當日收市交易時全期")
    for _, r in g.sort_values("全期回報%", ascending=False).iterrows():
        L.append(f"　{r['類別']} | {r['規則']} | {r['全期回報%']:+.2f}% / {r['全期最大跌幅%']:.1f}% | "
                 f"{r['前半回報%']:+.2f}% / {r['前半最大跌幅%']:.1f}% | {r['後半回報%']:+.2f}% / {r['後半最大跌幅%']:.1f}% | "
                 f"{r['每年買賣次數']} | {r['持倉時間%']}% | {r['全期回報%(成本0.02%)']:+.2f}% | {r['全期回報%(當日收市交易)']:+.2f}%")
    L.append("")
    h = bench["SPY長揸"]
    L.append("=== 整體 ===")
    L.append(f"　全期回報贏SPY長揸：{(g['全期回報%'] > h['全期回報%']).sum()}/{len(g)}；贏基本規則：{(g['全期回報%'] > b['全期回報%']).sum()}/{len(g)}")
    L.append(f"　全期最大跌幅好過 -30%：{(g['全期最大跌幅%'] > -30).sum()}/{len(g)}")
    both = g[(g["前半回報%"] > b["前半回報%"]) & (g["後半回報%"] > b["後半回報%"]) & (g["全期最大跌幅%"] > -30)]
    L.append(f"　前後兩段都贏基本規則、而且跌幅好過-30%：{len(both)}條")
    for _, r in both.iterrows():
        L.append(f"　　{r['類別']} | {r['規則']}：全期 {r['全期回報%']:+.2f}%（跌 {r['全期最大跌幅%']:.1f}%）| 前半 {r['前半回報%']:+.2f}% | 後半 {r['後半回報%']:+.2f}%")
    L.append("")
    L.append("=== 各類別平均（睇成個家族係咪一致）===")
    for fam, gg in g.groupby("類別", sort=False):
        L.append(f"　{fam}（{len(gg)}條）：全期平均 {gg['全期回報%'].mean():+.2f}% | 前半 {gg['前半回報%'].mean():+.2f}% | "
                 f"後半 {gg['後半回報%'].mean():+.2f}% | 平均最大跌幅 {gg['全期最大跌幅%'].mean():.1f}% | 每年買賣 {gg['每年買賣次數'].mean():.1f}")
    L.append("")
    ok = g[g["前半最大跌幅%"] > -30]
    pick = ok.sort_values("前半回報%", ascending=False).iloc[0]
    rank = int((g["後半回報%"] > pick["後半回報%"]).sum()) + 1
    L.append("=== 誠實測試：前半揀（跌幅好過-30%入面回報最高），原封不動睇後半 ===")
    L.append(f"　前半揀中：{pick['類別']} | {pick['規則']}，前半 {pick['前半回報%']:+.2f}%（跌 {pick['前半最大跌幅%']:.1f}%）")
    L.append(f"　→ 後半：{pick['後半回報%']:+.2f}%（跌 {pick['後半最大跌幅%']:.1f}%），後半排第{rank}/{len(g)}；"
             f"基本規則後半 {b['後半回報%']:+.2f}%，SPY長揸後半 {h['後半回報%']:+.2f}%")

    pd.DataFrame({"report": L}).to_csv("spy_indicators_results.csv", index=False, encoding="utf-8-sig")
    print("\n".join(L))


if __name__ == "__main__":
    main()
