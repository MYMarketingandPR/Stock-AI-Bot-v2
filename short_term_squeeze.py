"""
short_term_squeeze.py — 短炒回測②：Short Squeeze（照用戶Finviz篩選「Short squeeze 2.0」，做得到嘅部分）

⚠️ 預先登記（跑之前定好，跑完唔改）：
數據：
  - 沽空股數：FINRA每半個月公佈嘅數據（2021-06起先有交易所上市股票）。
    結算日之後9個交易日先當係「已經知道」（FINRA大約7個交易日後公佈，留多兩日）。
  - 流通股（Float）：冇免費歷史數據，而且Finviz封鎖GitHub主機；改用SEC每季公佈嘅「已發行股數」（當時已知，冇偷睇未來），\n    已發行股數 ≥ 流通股，所以「沽空比例」會計細咗，條件其實比Finviz嚴。股數日期後10日先當已知。
  - 機構持股 >50%、機構買賣 >0%：冇免費歷史數據，測試唔到，唔用。
  - 可以買期權（Optionable）：冇歷史數據，唔用。
信號（每日收市後）：
  - 沽空 / 已發行股數 > 20%（代替Short Float Over 20%）
  - 已發行股數 < 1億股（代替Float Under 100M）
  - 收市 > $15、收市高過200日線
  - RSI(14) < 60（Not Overbought）
  - 成交量 ≥ 過去3個月平均 × 1.5（Relative Volume Over 1.5）
  - 當日成交量 > 50萬股（Current Volume Over 500K）
  - 同一隻股票出咗信號之後20個交易日內唔再計（避免重複）
買入時間、賣出方法、組合、成本：同short_term_gap.py一樣（$10,000，最多3隻，每次0.15%）。
股票池：FINRA曾經被沽空50萬股以上、SEC數據曾經少過1億股嘅股票（⚠️已除牌公司冇價格數據，有倖存者偏差）。
測試期：2021-07至2026-09；前半2021-07至2023-12，後半2024-01至2026-09。
揀法：同short_term_gap.py一樣。
輸出：short_squeeze_results.csv、short_squeeze_trades.csv.gz
"""
import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from short_term_common import download, base_features, build_trades, report

PERIODS = [("全期", "2021-07-01", "2026-09-30"), ("前半", "2021-07-01", "2023-12-31"),
           ("後半", "2024-01-01", "2026-09-30")]
LAG_BDAYS = 9
PROGRESS = "short_squeeze_progress.txt"


def log(msg):
    import resource
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    line = f"{time.strftime('%H:%M:%S')} [記憶體 {mem:.0f}MB] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as fh:
        fh.write(line + "\n")
COOLDOWN = 20


def fetch_finra():
    days = pd.date_range("2021-06-01", "2026-09-30")

    def get(d):
        url = f"https://cdn.finra.org/equity/otcmarket/biweekly/shrt{d:%Y%m%d}.csv"
        for _ in range(3):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 10000:
                    return r.text
                if r.status_code in (403, 404):
                    return None
            except Exception:
                time.sleep(2)
        return None
    with ThreadPoolExecutor(12) as ex:
        texts = list(ex.map(get, days))
    frames = []
    for d, txt in zip(days, texts):
        if not txt:
            continue
        try:
            df = pd.read_csv(io.StringIO(txt), sep="|", quotechar='"', dtype=str, on_bad_lines="skip")
            df = df[["symbolCode", "marketClassCode", "currentShortPositionQuantity", "settlementDate"]]
            frames.append(df)
        except Exception as e:
            print(f"FINRA {d:%Y-%m-%d} 讀取失敗：{e}")
    si = pd.concat(frames, ignore_index=True)
    si["short"] = pd.to_numeric(si["currentShortPositionQuantity"], errors="coerce")
    si["settle"] = pd.to_datetime(si["settlementDate"])
    si = si[~si["marketClassCode"].astype(str).str.upper().str.contains("OTC|BB", na=False)]
    si = si[si["symbolCode"].astype(str).str.fullmatch(r"[A-Z]{1,5}")]
    log(f"FINRA：{si['settle'].nunique()}個結算日，{si['symbolCode'].nunique()}隻股票")
    return si[["symbolCode", "settle", "short"]].rename(columns={"symbolCode": "ticker"})


def parse_num(x):
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip().replace(",", "")
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(x[-1:].upper(), 1)
    try:
        return float(x[:-1] if mult != 1 else x) * mult
    except ValueError:
        return np.nan


SEC_HDR = {"User-Agent": "Stock-AI-Bot research script (GitHub Actions)", "Accept-Encoding": "gzip, deflate"}
SHARES_LAG_DAYS = 10


def fetch_shares_sec():
    """SEC EDGAR：每季公司封面上嘅「已發行股數」（point-in-time）。
    Finviz流通股冇歷史 + GitHub主機被Finviz封鎖，所以用已發行股數代替流通股。"""
    m = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HDR, timeout=30).json()
    cik2t = {}
    for v in m.values():
        cik2t.setdefault(int(v["cik_str"]), str(v["ticker"]).upper().replace(".", "-"))
    rows = []
    for y in range(2020, 2027):
        for q in range(1, 5):
            if (y, q) > (2026, 3):
                break
            url = f"https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY{y}Q{q}I.json"
            try:
                r = requests.get(url, headers=SEC_HDR, timeout=60)
                if not r.ok:
                    log(f"SEC {y}Q{q}：HTTP {r.status_code}")
                    continue
                for d in r.json().get("data", []):
                    t = cik2t.get(int(d["cik"]))
                    if t:
                        rows.append((t, pd.Timestamp(d["end"]), float(d["val"])))
            except Exception as e:
                log(f"SEC {y}Q{q} 出錯：{e}")
            time.sleep(0.3)
    sh = pd.DataFrame(rows, columns=["ticker", "end", "shares"])
    sh = sh.groupby(["ticker", "end"], as_index=False)["shares"].sum()   # 多類股份加埋
    log(f"SEC已發行股數：{sh['ticker'].nunique()}隻股票，{len(sh)}個數據點")
    return sh


def main():
    open(PROGRESS, "w").close()
    log("開始下載FINRA")
    si = fetch_finra()
    cand = si.groupby("ticker")["short"].max()
    cand = set(cand[cand >= 500_000].index)
    log(f"曾經被沽空50萬股以上：{len(cand)}隻")
    sh = fetch_shares_sec()
    small = set(sh.loc[sh["shares"] < 100_000_000, "ticker"])
    cand = sorted(cand & small)
    log(f"同時曾經已發行股數少過1億：{len(cand)}隻")
    spy = download(["SPY", "QQQ"], "2020-06-01", "2026-10-01")["SPY"]["Close"]
    prices = download(cand, "2020-06-01", "2026-10-01")
    log(f"有價格數據：{len(prices)}/{len(cand)}隻")
    prices = {t: d for t, d in prices.items() if (d["Close"] > 15).any()}
    bday = pd.offsets.BDay(LAG_BDAYS)

    signals = []
    sh_by_t = {t: g for t, g in sh.groupby("ticker")}
    for t, d in prices.items():
        g = sh_by_t.get(t)
        if g is None:
            continue
        g = g.assign(known=g["end"] + pd.Timedelta(days=SHARES_LAG_DAYS))
        g = g.sort_values("known").drop_duplicates("known", keep="last").set_index("known")["shares"]
        s = si[si["ticker"] == t].copy()
        if s.empty:
            continue
        s["known"] = s["settle"] + bday
        s = s.sort_values("known").drop_duplicates("known", keep="last").set_index("known")["short"]
        f = base_features(d)
        shares = g.reindex(f.index.union(g.index)).ffill().reindex(f.index)
        f["short_float"] = s.reindex(f.index.union(s.index)).ffill().reindex(f.index) / shares
        cond = ((f["short_float"] > 0.20) & (shares < 100_000_000) & (f["close"] > 15) & (f["close"] > f["sma200"]) &
                (f["rsi"] < 60) & (f["relvol"] >= 1.5) & (f["vol"] > 500_000))
        cond &= f.index >= pd.Timestamp(PERIODS[0][1])
        last = -10**9
        for i in [f.index.get_loc(x) for x in f.index[cond.fillna(False)]]:
            if i - last > COOLDOWN:
                signals.append((t, i, float(f["relvol"].iloc[i])))
                last = i
    log(f"信號數：{len(signals)}")
    if not signals:
        L = ["❌ 冇任何信號", f"股票池 {len(cand)}隻，有價格 {len(prices)}隻"]
        pd.DataFrame({"report": L}).to_csv("short_squeeze_results.csv", index=False, encoding="utf-8-sig")
        return
    trades = build_trades(signals, prices, spy)
    log(f"交易紀錄：{len(trades)}行，開始組合模擬")
    trades.to_csv("short_squeeze_trades.csv.gz", index=False, compression="gzip")
    notes = [f"股票池：FINRA曾被沽空50萬股以上、SEC曾經<1億股：{len(cand)}隻；有價格數據而且曾經高過$15：{len(prices)}隻；"
             f"信號共{len(signals)}個",
             "⚠️ 用SEC已發行股數代替流通股；機構持股/機構買賣/期權條件測試唔到",
             "成本每次買或賣0.15%；組合$10,000最多同時3隻"]
    L, _ = report(trades, prices, spy, PERIODS, "=== 短炒回測②：Short Squeeze（規則預先登記，見short_term_squeeze.py）===", notes)
    tops = trades[(trades["entry"] == "信號日收市買") & (trades["exit"] == "揸10日")]
    if len(tops):
        L += ["", "=== 出現最多信號嘅股票（信號日收市買、揸10日）==="]
        for t, g in tops.groupby("ticker").agg(n=("ret", "size"), avg=("ret", "mean")).sort_values("n", ascending=False).head(15).iterrows():
            L.append(f"　{t}：{int(g['n'])}次，平均 {g['avg'] * 100:+.2f}%")
    pd.DataFrame({"report": L}).to_csv("short_squeeze_results.csv", index=False, encoding="utf-8-sig")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        log("出錯：" + tb)
        print(tb)
        pd.DataFrame({"report": ["❌ 程式出錯：", *tb.splitlines()]}).to_csv(
            "short_squeeze_results.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv("short_squeeze_trades.csv.gz", compression="gzip")
        raise
