"""check_prices.py — 核對 2026-10-08 收市價，同重做線E揀股（全部數據齊）。"""
import pandas as pd
import yfinance as yf
import momentum_monthly as MM
import momentum_v1 as M

out = []
T = ["SPY", "SNDK", "MRNA", "LITE", "MU", "DELL", "MRVL", "BE"]
d = yf.download(T, start="2026-10-05", end="2026-10-10", auto_adjust=True, progress=False)["Close"]
out.append(d.round(2).to_string())
tickers, names = MM.current_sp500()
cl = M.download_close(tickers, "2024-09-01", "2026-10-09")
px = pd.DataFrame(cl).sort_index()
px = px[px.index <= "2026-10-08"]
last = px.iloc[-1]
ye = px[px.index.year < 2026].ffill().iloc[-1]
ytd = last / ye - 1
hi = px.iloc[-252:].max()
ma200 = px.iloc[-200:].mean()
sel = ytd[(ytd >= 0.3) & (last >= hi * 0.7) & (last > ma200)].sort_values(ascending=False)
out.append(f"最後一日：{px.index[-1].date()}；10-08 冇價嘅股票：{list(px.columns[px.iloc[-1].isna()])[:30]}")
out.append("線E 頭10名（全部數據）：" + "、".join(f"{t} {v * 100:+.0f}%" for t, v in sel.head(10).items()))
open("check_prices.txt", "w").write("\n".join(out))
print("\n".join(out))
