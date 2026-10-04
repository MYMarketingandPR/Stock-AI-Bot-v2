"""probe_sp500.py — 睇fja05680/sp500有咩檔案（一次性探測）"""
import pandas as pd
import requests
r = requests.get("https://api.github.com/repos/fja05680/sp500/contents/", headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
L = [f"HTTP {r.status_code}"]
try:
    L += [f"{f['name']} | {f.get('size')}" for f in r.json()]
except Exception as e:
    L.append(f"{e}: {r.text[:300]}")
pd.DataFrame({"report": L}).to_csv("probe_sp500.csv", index=False)
print("\n".join(L))
