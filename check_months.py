"""一次性檢查：標普500（^GSPC，唔計股息）每個月嘅歷史表現；同埋 5-10月 vs 11-4月。"""
import pandas as pd
import yfinance as yf

out = []
s = yf.Ticker("^GSPC").history(start="1950-01-01", end="2026-10-01", auto_adjust=True)["Close"]
s.index = pd.DatetimeIndex(s.index).tz_localize(None)
m = s.resample("ME").last().pct_change().dropna()
for lab, a in [("1950-2026", "1950"), ("2000-2026", "2000"), ("2015-2026", "2015")]:
    x = m[m.index >= a]
    out.append(f"=== {lab}（{len(x)} 個月）===")
    for k in range(1, 13):
        y = x[x.index.month == k]
        out.append(f"{k:>2}月：平均 {y.mean() * 100:+.2f}% ｜ 中位數 {y.median() * 100:+.2f}% ｜ 升嘅年份 {(y > 0).mean() * 100:.0f}%（{len(y)}年）")
    w = x[x.index.month.isin([5, 6, 7, 8, 9, 10])]
    b = x[~x.index.month.isin([5, 6, 7, 8, 9, 10])]
    out.append(f"5-10月 每月平均 {w.mean() * 100:+.2f}% ｜ 11-4月 每月平均 {b.mean() * 100:+.2f}%")
open("check_months.txt", "w").write("\n".join(out))
print("\n".join(out))
