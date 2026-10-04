"""
short_term_gap.py — 短炒回測①：機構財報/新聞跳空（照用戶Finviz篩選「Institutional Earnings/News Gap」）

⚠️ 預先登記（跑之前定好，跑完唔改）：
信號（信號日收市後先知道，全部照Finviz條件）：
  - 跳空：開市價 ≥ 昨日收市 × 1.04（Gap Up 4%）
  - 成交量 ≥ 過去3個月平均 × 3（Relative Volume Over 3）
  - 過去3個月平均成交量 > 200萬股（Average Volume Over 2M）
  - 收市高過20日、50日、200日簡單均線
  - 收市價 > $15
  - 市值 > $20億：用「今日標普500 + 標普400成分股」代替（⚠️ 有倖存者偏差：今日仲喺指數入面嘅公司，過去表現通常偏好）
  - 冇歷史新聞，所以唔分財報定新聞跳空。
買入時間：信號日收市買 / 第二日開市買。
賣出方法：揸1、3、5、10、20日；跌穿信號日最低價止損（最多20日）；再加升10%止賺。
組合：$10,000，最多同時3隻，每隻三分一；成本每次買或賣0.15%。
測試期：2005-01至2026-09；前半2005-2015，後半2016-2026/09。
揀法：前半段跌幅好過-35%入面回報最高嘅組合，原封不動睇後半段，同SPY比。
輸出：short_gap_results.csv、short_gap_trades.csv.gz
"""
import pandas as pd

from short_term_common import sp_universe, download, base_features, build_trades, report

START, END = "2004-01-01", "2026-10-01"
PERIODS = [("全期", "2005-01-01", "2026-09-30"), ("前半", "2005-01-01", "2015-12-31"),
           ("後半", "2016-01-01", "2026-09-30")]


def main():
    tickers = sp_universe()
    print(f"股票池：{len(tickers)}隻")
    prices = download(tickers + ["SPY"], START, END)
    spy = prices.pop("SPY")["Close"]
    signals = []
    for t, d in prices.items():
        f = base_features(d)
        cond = ((f["gap"] >= 0.04) & (f["relvol"] >= 3) & (f["avgvol"] > 2_000_000) &
                (f["close"] > f["sma20"]) & (f["close"] > f["sma50"]) & (f["close"] > f["sma200"]) &
                (f["close"] > 15))
        cond &= f.index >= pd.Timestamp(PERIODS[0][1])
        for i in [f.index.get_loc(x) for x in f.index[cond.fillna(False)]]:
            signals.append((t, i, float(f["relvol"].iloc[i])))
    print(f"信號數：{len(signals)}")
    trades = build_trades(signals, prices, spy)
    trades.to_csv("short_gap_trades.csv.gz", index=False, compression="gzip")
    notes = [f"股票池：今日標普500+400共{len(prices)}隻（⚠️倖存者偏差）；信號共{len(signals)}個",
             "成本每次買或賣0.15%；組合$10,000最多同時3隻"]
    L, _ = report(trades, prices, spy, PERIODS, "=== 短炒回測①：機構財報/新聞跳空（規則預先登記，見short_term_gap.py）===", notes)
    yearly = trades[(trades["entry"] == "信號日收市買") & (trades["exit"] == "揸5日")]
    L.append("")
    L.append("=== 每年信號數同平均回報（信號日收市買、揸5日）===")
    for y, g in yearly.groupby(yearly["signal_date"].dt.year):
        L.append(f"　{y}：{len(g)}個，平均 {g['ret'].mean() * 100:+.2f}%，同期SPY {g['spy_ret'].mean() * 100:+.2f}%")
    pd.DataFrame({"report": L}).to_csv("short_gap_results.csv", index=False, encoding="utf-8-sig")
    print("\n".join(L))


if __name__ == "__main__":
    main()
