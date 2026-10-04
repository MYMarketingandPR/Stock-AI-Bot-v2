"""
spy_bottoms_v2.py — SPY見底研究第二版：加入成交量、VIX（恐慌指數）、EMA9、EMA21

⚠️ 預先登記（跑之前定好，跑完唔改）：

共同設定（同spy_exits.py一樣）
- SPY（已計股息）、^VIX、^IRX（現金息）由yfinance下載；測試期1994-01至2026-09。
- 前半1994-2012，後半2013-2026/09。每日收市判斷，下一日收市交易。每次買賣0.1%。唔用槓桿。

新指標（全部係當日收市已經知道）
- 成交量倍數 = 今日成交量 ÷ 過去50日平均成交量
- VIX收市；VIX回落 = 過去5日VIX最高 ≥ 30，而今日VIX比嗰個最高低15%以上
- EMA9、EMA21（收市價指數移動平均）；金叉 = EMA9由下向上穿過EMA21

第一部分：每次跌10%以上（同第一版一樣12次），加列：
  最低嗰日VIX、成交量倍數；成段跌市VIX最高嗰日同最低位差幾多日；
  見底之後第一次EMA9金叉：遲幾多日、已經離底升咗幾多%、之後有冇再跌穿個底。

第二部分：信號出現 → 之後再跌幾多先見底、12個月後回報（60個交易日內唔重複計）
  第一版5個：S1 RSI<30、S2 一年高位跌20%、S3 低過200日線10%、S4 20日波幅>35%、S5 5日跌8%
  新加：
  S6 VIX ≥ 30
  S7 VIX ≥ 40
  S8 恐慌大成交：當日跌市 + 成交量倍數 ≥ 2
  S9 VIX回落（見上面定義）
  S10 EMA9金叉，而收市仍然低過200日線
  S11 三合一：同一日 RSI<30 + VIX≥30 + 成交量倍數≥2

第三部分：策略（基本規則 = 200日線緩衝2%：跌穿200日線2%賣，升返高過2%買返）
  E1 基本 + 空倉時EMA9金叉提早買；EMA9跌穿EMA21就賣
  E2 基本 + 空倉時VIX回落(S9)提早買；比買入價跌5%就賣
  E3 基本 + 空倉時恐慌大成交(S8)而且VIX≥30提早買；比買入價跌5%就賣
  E4 基本 + 空倉時三合一(S11)提早買；比買入價跌5%就賣
  （提早買咗之後，升到高過200日線2% → 變返正常持有）
  X1 淨用EMA9/EMA21：EMA9高過EMA21就持有，否則現金
  X2 淨用EMA21：收市高過EMA21就持有，否則現金
  採用條件：前後兩段回報都要好過基本規則，而且全期最大跌幅唔可以差過 -30%。

輸出：spy_bottoms_v2_results.csv（同時印喺log）
"""
import sys
import numpy as np
import pandas as pd

from spy_exits import run, stats, PERIODS, START, END
from spy_bottoms import rsi, episodes, signal_events


def load():
    import yfinance as yf

    def dl(t):
        df = yf.download(t, start="1992-06-01", end="2026-10-01", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    s = dl("SPY")
    spy, vol = s["Close"].astype(float).dropna(), s["Volume"].astype(float)
    vix = dl("^VIX")["Close"].astype(float).reindex(spy.index).ffill()
    irx = dl("^IRX")["Close"].astype(float).reindex(spy.index).ffill().fillna(0.0)
    return spy, vol.reindex(spy.index), vix, irx / 100 / 252


def load_fake():
    """只用嚟試程式有冇bug：SPY用真收市，成交量同VIX係假嘅。"""
    from spy_exits import load_csv
    spy, cash = load_csv("data/spy_equity.csv")
    rng = np.random.default_rng(0)
    rv = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100
    vix = (rv * 1.1 + 3).bfill()
    vol = pd.Series(rng.lognormal(16, 0.3, len(spy)), index=spy.index) * (1 + rv.bfill() / 20)
    return spy, vol, vix, cash


def add_features(spy, vol, vix):
    f = pd.DataFrame(index=spy.index)
    f["RSI14"] = rsi(spy)
    ma200 = spy.rolling(200).mean()
    f["距200日線%"] = (spy / ma200 - 1) * 100
    f["20日波幅%"] = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100
    f["5日回報%"] = (spy / spy.shift(5) - 1) * 100
    f["距一年高位%"] = (spy / spy.rolling(252, min_periods=1).max() - 1) * 100
    f["成交量倍數"] = vol / vol.rolling(50).mean()
    f["VIX"] = vix
    vmax5 = vix.rolling(5).max()
    f["VIX回落"] = (vmax5 >= 30) & (vix <= vmax5 * 0.85)
    e9, e21 = spy.ewm(span=9, adjust=False).mean(), spy.ewm(span=21, adjust=False).mean()
    f["EMA9"], f["EMA21"] = e9, e21
    f["金叉"] = (e9 > e21) & (e9.shift(1) <= e21.shift(1))
    f["跌市"] = spy.diff() < 0
    return f, ma200


def base_plus(spy, ma, f, entry, exit_mode, band=0.02):
    state, entry_px, out = "out", None, []
    e9, e21 = f["EMA9"].values, f["EMA21"].values
    for i, (p, m) in enumerate(zip(spy.values, ma.values)):
        if np.isnan(m):
            out.append(0.0)
            continue
        if state == "in":
            if p < m * (1 - band):
                state = "out"
        elif state == "out":
            if p > m * (1 + band):
                state = "in"
            elif entry is not None and bool(entry.iloc[i]):
                state, entry_px = "early", p
        elif state == "early":
            if p > m * (1 + band):
                state = "in"
            elif exit_mode == "ema" and e9[i] < e21[i]:
                state = "out"
            elif exit_mode == "pct5" and p < entry_px * 0.95:
                state = "out"
        out.append(0.0 if state == "out" else 1.0)
    return pd.Series(out, index=spy.index)


def main():
    spy, vol, vix, cash_r = load_fake() if "--fake" in sys.argv else load()
    spy = spy[spy.index <= pd.Timestamp(END)]
    vol, vix, cash_r = (x.reindex(spy.index) for x in (vol, vix, cash_r))
    cash_r = cash_r.fillna(0.0)
    f, ma200 = add_features(spy, vol, vix)
    L = ["=== SPY見底研究第二版（加成交量、VIX、EMA9/21；規則預先登記，見spy_bottoms_v2.py開頭）===",
         "⚠️ 用 --fake 跑嘅結果係假數據，只係試程式" if "--fake" in sys.argv else "", ""]

    # ---------- 第一部分 ----------
    eps = episodes(spy)
    L.append(f"=== 第一部分：SPY跌10%以上共{len(eps)}次（⚠️事後先知邊日係底）===")
    L.append("　最低位日期 | 跌幅 | 見底日VIX / 成交量倍數 | VIX最高嗰日（同見底差幾多交易日）| "
             "成交量最大嗰日（差幾多日）| 見底後第一次EMA9金叉：遲幾日 / 離底已升 / 之後有冇再穿底")
    idx = spy.index
    lag_vix, lag_vol, ema_lag, ema_up, ema_fail = [], [], [], [], []
    for pk, lo, dd, rec in eps:
        seg = slice(pk, lo)
        vpk = f["VIX"].loc[seg].idxmax()
        qpk = f["成交量倍數"].loc[seg].idxmax()
        dv = idx.get_loc(vpk) - idx.get_loc(lo)
        dq = idx.get_loc(qpk) - idx.get_loc(lo)
        lag_vix.append(dv)
        lag_vol.append(dq)
        after = f["金叉"].loc[lo:]
        after = after[after]
        if len(after):
            gx = after.index[0]
            lag = idx.get_loc(gx) - idx.get_loc(lo)
            up = (spy.loc[gx] / spy.loc[lo] - 1) * 100
            nxt = spy.loc[gx:].iloc[1:253]
            fail = bool((nxt < spy.loc[lo]).any())
            ema_lag.append(lag)
            ema_up.append(up)
            ema_fail.append(fail)
            gtxt = f"{lag}日 / +{up:.1f}% / {'有，再穿底' if fail else '冇'}"
        else:
            gtxt = "未出現"
        L.append(f"　{lo.date()} | {dd * 100:.1f}% | {f.loc[lo, 'VIX']:.0f} / {f.loc[lo, '成交量倍數']:.1f}倍 | "
                 f"VIX最高{f.loc[vpk, 'VIX']:.0f}（{dv:+d}日）| 成交量{f.loc[qpk, '成交量倍數']:.1f}倍（{dq:+d}日）| {gtxt}")
    L.append(f"　小結：VIX最高嗰日比見底早 {-np.median(lag_vix):.0f} 個交易日（中位數），"
             f"VIX最高同見底同一日 {sum(1 for x in lag_vix if x == 0)}/{len(lag_vix)} 次")
    L.append(f"　　　　成交量最大嗰日比見底早 {-np.median(lag_vol):.0f} 個交易日（中位數），"
             f"同一日 {sum(1 for x in lag_vol if x == 0)}/{len(lag_vol)} 次")
    if ema_lag:
        L.append(f"　　　　EMA9金叉：見底後中位數 {np.median(ema_lag):.0f} 日出現，嗰時已離底升 {np.median(ema_up):.1f}%；"
                 f"之後再穿底 {sum(ema_fail)}/{len(ema_fail)} 次")
    allf = f[f.index >= pd.Timestamp(START)]
    bt = f.loc[[lo for _, lo, _, _ in eps]]
    L.append(f"　見底日VIX中位數 {bt['VIX'].median():.0f}（範圍 {bt['VIX'].min():.0f}-{bt['VIX'].max():.0f}），平時 {allf['VIX'].median():.0f}")
    L.append(f"　見底日成交量倍數中位數 {bt['成交量倍數'].median():.1f}（範圍 {bt['成交量倍數'].min():.1f}-{bt['成交量倍數'].max():.1f}），平時 {allf['成交量倍數'].median():.1f}")
    L.append("")

    # ---------- 第二部分 ----------
    L.append("=== 第二部分：信號出現嗰日買入，之後點？===")
    sigs = {
        "S1 RSI低過30": f["RSI14"] < 30,
        "S2 由一年高位跌20%以上": f["距一年高位%"] <= -20,
        "S3 低過200日線10%以上": f["距200日線%"] <= -10,
        "S4 20日波幅高過35%": f["20日波幅%"] >= 35,
        "S5 5日內跌8%以上": f["5日回報%"] <= -8,
        "S6 VIX≥30": f["VIX"] >= 30,
        "S7 VIX≥40": f["VIX"] >= 40,
        "S8 恐慌大成交（跌市+成交量≥2倍）": f["跌市"] & (f["成交量倍數"] >= 2),
        "S9 VIX回落": f["VIX回落"],
        "S10 EMA9金叉（仍低過200日線）": f["金叉"] & (spy < ma200),
        "S11 三合一（RSI<30+VIX≥30+成交量≥2倍）": (f["RSI14"] < 30) & (f["VIX"] >= 30) & (f["成交量倍數"] >= 2),
    }
    summary = []
    for name, sg in sigs.items():
        sg = sg.fillna(False) & (spy.index >= pd.Timestamp(START))
        ev = signal_events(sg)
        if not ev:
            L.append(f"--- {name}：0次 ---")
            continue
        further, fwd12 = [], []
        L.append(f"--- {name}：共{len(ev)}次 ---")
        for d in ev:
            i = idx.get_loc(d)
            fall = (spy.iloc[i:i + 253].min() / spy.iloc[i] - 1) * 100
            r12 = (spy.iloc[i + 252] / spy.iloc[i] - 1) * 100 if i + 252 < len(spy) else np.nan
            further.append(fall)
            fwd12.append(r12)
            L.append(f"　{d.date()}：之後再跌 {fall:.1f}%；12個月後 " + (f"{r12:+.1f}%" if not np.isnan(r12) else "未夠12個月"))
        fu, fw = np.array(further), np.array([x for x in fwd12 if not np.isnan(x)])
        txt = (f"再跌中位數 {np.median(fu):.1f}%，最多再跌 {fu.min():.1f}%；再跌少過5% {(fu > -5).sum()}/{len(fu)}；"
               f"12個月後賺錢 {(fw > 0).sum()}/{len(fw)}，平均 {fw.mean():+.1f}%" if len(fw) else "")
        L.append(f"　小結：{txt}")
        summary.append(f"　{name}（{len(ev)}次）：{txt}")
    L.append("")
    L.append("--- 第二部分總表 ---")
    L += summary
    L.append("")

    # ---------- 第三部分 ----------
    L.append("=== 第三部分：策略比較 ===")
    sig9 = f["VIX回落"].fillna(False)
    sig8 = (f["跌市"] & (f["成交量倍數"] >= 2) & (f["VIX"] >= 30)).fillna(False)
    sig11 = sigs["S11 三合一（RSI<30+VIX≥30+成交量≥2倍）"].fillna(False)
    e9_above = (f["EMA9"] > f["EMA21"]).astype(float)
    strategies = {
        "SPY長揸": pd.Series(1.0, index=idx),
        "基本規則（200日線緩衝2%）": base_plus(spy, ma200, f, None, None),
        "E1 基本 + EMA9金叉提早買": base_plus(spy, ma200, f, f["金叉"], "ema"),
        "E2 基本 + VIX回落提早買": base_plus(spy, ma200, f, sig9, "pct5"),
        "E3 基本 + 恐慌大成交(VIX≥30)提早買": base_plus(spy, ma200, f, sig8, "pct5"),
        "E4 基本 + 三合一提早買": base_plus(spy, ma200, f, sig11, "pct5"),
        "X1 淨用EMA9/EMA21": e9_above.where(ma200.notna(), 0.0),
        "X2 淨用EMA21": (spy > f["EMA21"]).astype(float).where(ma200.notna(), 0.0),
    }
    res = {}
    for name, tgt in strategies.items():
        d = run(spy, cash_r, tgt)
        res[name] = {pn: stats(d, a, b) for pn, a, b in PERIODS}
        t = tgt[tgt.index >= pd.Timestamp(START)]
        n_sw = t.diff().abs().fillna(0).sum() / ((t.index[-1] - t.index[0]).days / 365.25)
        parts = " | ".join(f"{pn} {res[name][pn]['cagr']:+.2f}%（跌 {res[name][pn]['mdd']:.1f}%）" for pn, _, _ in PERIODS)
        L.append(f"　{name}：{parts} | 每年買賣 {n_sw:.1f} 次")
    base = res["基本規則（200日線緩衝2%）"]
    L.append("")
    for name in [k for k in strategies if k[:2] in ("E1", "E2", "E3", "E4", "X1", "X2")]:
        r = res[name]
        ok = (r["前半"]["cagr"] > base["前半"]["cagr"] and r["後半"]["cagr"] > base["後半"]["cagr"]
              and r["全期"]["mdd"] > -30)
        L.append(f"　{name}：{'✅ 符合採用條件' if ok else '❌ 唔符合採用條件'}")

    pd.DataFrame({"report": L}).to_csv("spy_bottoms_v2_results.csv", index=False, encoding="utf-8-sig")
    print("\n".join(L))


if __name__ == "__main__":
    main()
