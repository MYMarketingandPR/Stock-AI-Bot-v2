"""一次性檢查：Yahoo 新聞 / SEC 文件 / 證監會網站 喺 GitHub 上攞唔攞到。"""
import json
import requests
import yfinance as yf

out = []
for t in ["MPC", "DVA", "AAPL"]:
    tk = yf.Ticker(t)
    try:
        n = tk.news or []
        out.append(f"{t} news: {len(n)} items; first keys: {list(n[0].keys()) if n else None}")
        if n:
            c = n[0].get("content", n[0])
            out.append(f"  content keys: {list(c.keys())[:20]}")
            out.append(f"  title={c.get('title')} pubDate={c.get('pubDate')} providerPublishTime={n[0].get('providerPublishTime')}")
    except Exception as e:
        out.append(f"{t} news error: {e}")
    try:
        f = tk.sec_filings
        out.append(f"{t} sec_filings: {type(f).__name__} len={len(f) if f is not None else None}")
        if f:
            out.append("  first: " + json.dumps(f[0], default=str)[:400])
    except Exception as e:
        out.append(f"{t} sec_filings error: {e}")
for url in ["https://www.sec.gov/files/company_tickers.json", "https://data.sec.gov/submissions/CIK0000927066.json"]:
    try:
        r = requests.get(url, headers={"User-Agent": "Stock-AI-Bot-v2 research 233071485+MYMarketingandPR@users.noreply.github.com"}, timeout=30)
        out.append(f"{url}: {r.status_code} {r.headers.get('content-type')} {r.text[:150]!r}")
    except Exception as e:
        out.append(f"{url}: error {e}")
open("check_news.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
