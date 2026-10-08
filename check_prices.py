"""check_prices.py — 核對Yahoo收市價（2026-10-08 Esther 發現 Telegram 用咗前一日收市價）。"""
import io
import pandas as pd
import requests
import yfinance as yf

T = ["MRNA", "SNDK", "LITE", "MU", "DELL", "SPY", "QQQ", "EFA", "TLT", "GLD"]
out = []
for t in T:
    h = yf.Ticker(t).history(start="2026-10-01", end="2026-10-09", auto_adjust=True)
    out.append(f"{t} history: " + " | ".join(f"{d.date()} {c:.2f}" for d, c in h["Close"].items()))
    try:
        fi = yf.Ticker(t).fast_info
        out.append(f"   fast_info last_price {fi['last_price']:.2f} previous_close {fi['previous_close']:.2f}")
    except Exception as e:
        out.append(f"   fast_info 出錯 {e}")
    try:
        r = requests.get(f"https://stooq.com/q/d/l/?s={t.lower()}.us&i=d", timeout=20)
        s = pd.read_csv(io.StringIO(r.text)).tail(4)
        out.append("   stooq: " + " | ".join(f"{a} {b:.2f}" for a, b in zip(s["Date"], s["Close"])))
    except Exception as e:
        out.append(f"   stooq 出錯 {e}")
d = yf.download(T, start="2026-10-01", end="2026-10-09", auto_adjust=True, progress=False)["Close"]
out.append("download batch:\n" + d.round(2).to_string())
open("check_prices.txt", "w").write("\n".join(out))
print("\n".join(out))
