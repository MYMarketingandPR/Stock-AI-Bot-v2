"""一次性檢查：MRNA 最近每日 收市 ÷ 200日平均。"""
import yfinance as yf
out = []
for adj in (True, False):
    d = yf.download("MRNA", period="2y", progress=False, auto_adjust=adj)["Close"].squeeze().dropna()
    ma = d.rolling(200).mean()
    out.append(f"auto_adjust={adj} 行數 {len(d)} 重複日子 {d.index.duplicated().sum()} 最後一日 {d.index[-1]}")
    for t in d.index[-20:]:
        out.append(f"{t.date()} 收市 {d[t]:.2f} 200日平均 {ma[t]:.2f} → {d[t] / ma[t]:.2f} 倍")
    s = d[-260:]
    gaps = [(a.date(), b.date()) for a, b in zip(s.index[:-1], s.index[1:]) if (b - a).days > 4]
    out.append(f"最近260日入面超過4日嘅空隙：{gaps}")
open("check_mrna.txt", "w").write("\n".join(out))
print("\n".join(out))
