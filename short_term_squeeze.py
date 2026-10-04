"""
short_term_squeeze.py — 短炒回測②：Short Squeeze（照用戶Finviz篩選「Short squeeze 2.0」，做得到嘅部分）

⚠️ 預先登記（跑之前定好，跑完唔改）：
數據：
  - 沽空股數：FINRA每半個月公佈嘅數據（2021-06起先有交易所上市股票）。
    結算日之後9個交易日先當係「已經知道」（FINRA大約7個交易日後公佈，留多兩日）。
  - 流通股（Float）：只有今日數字（Finviz），當過去一直都係咁 —— ⚠️ 有偏差。
  - 機構持股 >50%、機構買賣 >0%：冇免費歷史數據，測試唔到，唔用。
  - 可以買期權（Optionable）：冇歷史數據，唔用。
信號（每日收市後）：
  - 沽空 / 流通股 > 20%（Short Float Over 20%）
  - 流通股 < 1億股（Float Under 100M）
  - 收市 > $15、收市高過200日線
  - RSI(14) < 60（Not Overbought）
  - 成交量 ≥ 過去3個月平均 × 1.5（Relative Volume Over 1.5）
  - 當日成交量 > 50萬股（Current Volume Over 500K）
  - 同一隻股票出咗信號之後20個交易日內唔再計（避免重複）
買入時間、賣出方法、組合、成本：同short_term_gap.py一樣（$10,000，最多3隻，每次0.15%）。
股票池：FINRA數據入面曾經被沽空50萬股以上、而今日Finviz流通股少過1億嘅股票（⚠️已除牌公司唔喺入面，有倖存者偏差）。
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


def fetch_float():
    """用Finviz（finvizfinance）攞今日流通股少過1億嘅股票同佢哋嘅流通股數。"""
    from finvizfinance.screener.ownership import Ownership
    sc = Ownership()
    sc.set_filter(filters_dict={"Float": "Under 100M"})
    df = sc.screener_view(verbose=0, sleep_sec=0.5)
    log(f"Finviz欄位：{list(df.columns)}；頭兩行：{df.head(2).to_dict('records')}")
    out = {str(t): parse_num(v) for t, v in zip(df["Ticker"], df["Float"])}
    # finvizfinance有時已經換算成「百萬」單位
    vals = np.array([v for v in out.values() if v == v])
    if len(vals) and np.nanmedian(vals) < 1000:
        out = {k: v * 1e6 for k, v in out.items()}
    log(f"Finviz：流通股少過1億嘅股票 {len(out)}隻（流通股中位數 {np.nanmedian(list(out.values())):,.0f}）")
    return out


def main():
    open(PROGRESS, "w").close()
    log("開始下載FINRA")
    si = fetch_finra()
    cand = si.groupby("ticker")["short"].max()
    cand = set(cand[cand >= 500_000].index)
    log(f"曾經被沽空50萬股以上：{len(cand)}隻")
    floats = fetch_float()
    cand = sorted(cand & set(floats))
    log(f"同時係今日流通股少過1億：{len(cand)}隻")
    spy = download(["SPY", "QQQ"], "2020-06-01", "2026-10-01")["SPY"]["Close"]
    prices = download(cand, "2020-06-01", "2026-10-01")
    log(f"有價格數據：{len(prices)}/{len(cand)}隻")
    prices = {t: d for t, d in prices.items() if (d["Close"] > 15).any()}
    bday = pd.offsets.BDay(LAG_BDAYS)

    signals = []
    for t, d in prices.items():
        fl = floats.get(t, np.nan)
        if not (fl > 0) or fl >= 100_000_000:
            continue
        s = si[si["ticker"] == t].copy()
        if s.empty:
            continue
        s["known"] = s["settle"] + bday
        s = s.sort_values("known").drop_duplicates("known", keep="last").set_index("known")["short"]
        f = base_features(d)
        f["short_float"] = s.reindex(f.index.union(s.index)).ffill().reindex(f.index) / fl
        cond = ((f["short_float"] > 0.20) & (f["close"] > 15) & (f["close"] > f["sma200"]) &
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
    notes = [f"股票池：FINRA曾被沽空50萬股以上、今日流通股<1億：{len(cand)}隻；有價格數據而且曾經高過$15：{len(prices)}隻；"
             f"信號共{len(signals)}個",
             "⚠️ 流通股用今日數字；機構持股/機構買賣/期權條件測試唔到",
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
