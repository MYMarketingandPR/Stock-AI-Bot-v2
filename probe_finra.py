"""probe_finra.py — 睇FINRA沽空檔案格式（一次性探測，輸出probe_finra.csv）"""
import io
import requests
import pandas as pd

L = []
for d in ["20210615", "20210630", "20240115", "20260915"]:
    url = f"https://cdn.finra.org/equity/otcmarket/biweekly/shrt{d}.csv"
    try:
        r = requests.get(url, timeout=30)
        L.append(f"{d}: HTTP {r.status_code}, {len(r.content)} bytes")
        if r.ok:
            txt = r.text
            L += ["  " + line for line in txt.splitlines()[:4]]
            for sym in ["GME", "AAPL", "UPST", "CVNA"]:
                hit = [x for x in txt.splitlines() if f"|{sym}|" in x or x.startswith(f"{sym}|") or f'"{sym}"' in x]
                L += ["  " + h for h in hit[:1]]
    except Exception as e:
        L.append(f"{d}: error {e}")
pd.DataFrame({"report": L}).to_csv("probe_finra.csv", index=False)
print("\n".join(L))
