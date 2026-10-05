"""
probe_info.py — 一次性：查「公司資料」類數據免費攞唔攞到、去到幾多年前（第十四輪準備）
A. Yahoo 盈利公布紀錄（預期EPS、實際EPS、驚喜%）
B. Yahoo 分析師評級升降紀錄
C. Yahoo 分析師EPS預測變化（只有今日嘅？）
D. SEC 財務數據（公司質素：ROE、現金流等）
E. iShares 羅素1000/3000 ETF 歷史持股（當時成分股）
輸出：probe_info.txt
"""
import io
import traceback

import pandas as pd
import requests

OUT = "probe_info.txt"
L = []


def say(s):
    print(s, flush=True)
    L.append(str(s))


def section(name, fn):
    say(f"\n===== {name} =====")
    try:
        fn()
    except Exception:
        say("出錯：" + traceback.format_exc()[-800:])


def earnings():
    import yfinance as yf
    for t in ["AAPL", "MSFT", "JPM", "XOM", "NVDA", "GE", "INTC", "XLNX", "TWTR", "ATVI"]:
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=100)
            if df is None or df.empty:
                say(f"{t}: 冇數據")
                continue
            est = [c for c in df.columns if "Estimate" in c]
            say(f"{t}: {len(df)} 次，最早 {df.index.min()}，最遲 {df.index.max()}，欄位 {list(df.columns)}，"
                f"有預期EPS {int(df[est[0]].notna().sum()) if est else 0} 次")
        except Exception as e:
            say(f"{t}: 出錯 {e}")
    df = yf.Ticker("AAPL").get_earnings_dates(limit=100)
    say(df.tail(5).to_string())


def ratings():
    import yfinance as yf
    for t in ["AAPL", "JPM", "XOM", "INTC"]:
        try:
            df = yf.Ticker(t).upgrades_downgrades
            if df is None or df.empty:
                say(f"{t}: 冇數據")
                continue
            say(f"{t}: {len(df)} 條，最早 {df.index.min()}，最遲 {df.index.max()}，欄位 {list(df.columns)}")
        except Exception as e:
            say(f"{t}: 出錯 {e}")
    df = yf.Ticker("AAPL").upgrades_downgrades
    if df is not None and not df.empty:
        say(df.sort_index().head(5).to_string())


def eps_trend():
    import yfinance as yf
    tk = yf.Ticker("AAPL")
    for name in ["eps_trend", "eps_revisions", "earnings_estimate"]:
        try:
            df = getattr(tk, name)
            say(f"{name}:\n{df.to_string() if df is not None else '冇'}")
        except Exception as e:
            say(f"{name}: 出錯 {e}")


def sec():
    hdr = {"User-Agent": "StockAIBot research github.com/MYMarketingandPR/Stock-AI-Bot", "Accept-Encoding": "gzip, deflate"}
    for url in ["https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
                "https://data.sec.gov/api/xbrl/frames/us-gaap/NetIncomeLoss/USD/CY2015.json",
                "https://www.sec.gov/files/dera/data/financial-statement-data-sets/2015q1.zip"]:
        try:
            r = requests.get(url, headers=hdr, timeout=60, stream=True)
            n = len(r.raw.read(200000, decode_content=True)) if r.ok else 0
            say(f"{r.status_code} {url}（讀到 {n} bytes）")
        except Exception as e:
            say(f"出錯 {url}: {e}")


def ishares():
    base = {"IWB": "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax",
            "IWV": "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax"}
    for fund, url in base.items():
        for d in ["20260930", "20200131", "20150130", "20100129", "20060131", "20010131"]:
            try:
                r = requests.get(url, params={"fileType": "csv", "fileName": f"{fund}_holdings", "dataType": "fund",
                                              "asOfDate": d}, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
                txt = r.text
                lines = txt.splitlines()
                k = next((i for i, l in enumerate(lines) if l.startswith("Ticker")), None)
                if r.ok and k is not None:
                    df = pd.read_csv(io.StringIO("\n".join(lines[k:])), on_bad_lines="skip")
                    eq = df[df.get("Asset Class", pd.Series(dtype=str)) == "Equity"] if "Asset Class" in df else df
                    say(f"{fund} {d}: {r.status_code}，股票 {len(eq)} 隻，頭幾行日期：{lines[:3]}")
                else:
                    say(f"{fund} {d}: {r.status_code}，冇持股表（長度 {len(txt)}）頭：{txt[:150]!r}")
            except Exception as e:
                say(f"{fund} {d}: 出錯 {e}")


section("A. 盈利公布紀錄", earnings)
section("B. 分析師評級升降", ratings)
section("C. 分析師EPS預測變化", eps_trend)
section("D. SEC 財務數據", sec)
section("E. iShares 羅素ETF歷史持股", ishares)
open(OUT, "w").write("\n".join(L))
