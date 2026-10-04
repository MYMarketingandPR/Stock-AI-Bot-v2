"""probe_shares.py — 試下喺GitHub主機用yfinance攞歷史股數（一次性探測）"""
import time
import pandas as pd
import yfinance as yf

L = []
for t in ["GME", "UPST", "CVNA", "AMC", "BYND", "AAPL"]:
    try:
        s = yf.Ticker(t).get_shares_full(start="2021-01-01")
        L.append(f"{t} get_shares_full: {0 if s is None else len(s)}點；頭：{None if s is None or s.empty else (str(s.index[0]), float(s.iloc[0]))}；尾：{None if s is None or s.empty else (str(s.index[-1]), float(s.iloc[-1]))}")
    except Exception as e:
        L.append(f"{t} get_shares_full 出錯：{type(e).__name__} {e}")
    try:
        fi = yf.Ticker(t).fast_info
        L.append(f"{t} fast_info shares: {fi.get('shares')}")
    except Exception as e:
        L.append(f"{t} fast_info 出錯：{type(e).__name__} {e}")
    try:
        L.append(f"{t} info floatShares: {yf.Ticker(t).info.get('floatShares')}")
    except Exception as e:
        L.append(f"{t} info 出錯：{type(e).__name__} {e}")
    time.sleep(1)
pd.DataFrame({"report": L}).to_csv("probe_shares.csv", index=False)
print("\n".join(L))
