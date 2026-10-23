"""一次性檢查：其他新聞來源喺 GitHub 上攞唔攞到。"""
import requests
import xml.etree.ElementTree as ET

out = []
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
for name, url in [("yahoo_rss", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=DVA&region=US&lang=en-US"),
                  ("google_rss", "https://news.google.com/rss/search?q=DaVita+DVA+stock&hl=en-US&gl=US&ceid=US:en"),
                  ("nasdaq", "https://api.nasdaq.com/api/news/topic/articlebysymbol?q=DVA%7Cstocks&offset=0&limit=5")]:
    try:
        r = requests.get(url, headers=UA, timeout=30)
        out.append(f"{name}: {r.status_code} {r.headers.get('content-type')} len={len(r.text)}")
        if "xml" in (r.headers.get("content-type") or "") or r.text.lstrip().startswith("<?xml"):
            root = ET.fromstring(r.content)
            items = root.findall(".//item")[:3]
            for it in items:
                out.append(f"   - {it.findtext('pubDate')} | {it.findtext('title')[:100]}")
        else:
            out.append("   " + r.text[:300].replace("\n", " "))
    except Exception as e:
        out.append(f"{name}: error {e}")
open("check_news.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
