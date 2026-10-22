"""
pullback_scan.py — 每日「回調訊號」掃描（Esther 2026-10-23 要求）：每日收市後揀最多5隻符合條件嘅標普500股票，連重要新聞同公告發 Telegram。

條件（同第三十七輪入場規則一樣）：
  第1步 強勢股（同線E）：今年升 ≥ 30%；收市 ≥ 52週最高收市 × 0.7；收市 > 200日平均。
  第2步 回調訊號（當日收市後）：20日平均向上（今日 > 5日前）；收市 ≥ 20日平均；當日最低價 ≤ 10日平均 × 1.01。
  第3步：止蝕位 = 當日最低價；收市離最低價超過 8% 就剔走（止蝕太遠）。多過5隻就揀今年升最多嘅5隻。
  參考做法：下一個交易日收市買入；如果嗰日收市已經低過止蝕位就唔買。
⚠️ 第三十七輪回測（跌穿10日線賣）2015年起每年 -2.1%，未通過標準 —— 呢個掃描只係觀察，唔係投資建議，唔會自動模擬交易。

新聞：Yahoo Finance 新聞（最近3日，每隻最多2條）。公告：美國證監會 EDGAR（最近7日），附中文解釋。
下次業績日：Yahoo；10日內會特別提醒（第三十五輪：業績日一日跌≥10% 機會比平時高好多）。
紀錄：paper_log/pullback_signals.tsv（每日訊號）、paper_log/pullback_done.txt（防重複發）。
用法：python pullback_scan.py [--scheduled] [--final] [--dry]
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import main as bot
import momentum_monthly as MM
import prices as PR

DONE = "paper_log/pullback_done.txt"
LOG = "paper_log/pullback_signals.tsv"
MAX_PICKS = 5
SEC_UA = {"User-Agent": "Stock-AI-Bot-v2 research 233071485+MYMarketingandPR@users.noreply.github.com"}
FORM_ZH = {"4": "高管／大股東買賣股份", "144": "高管準備賣股", "8-K": "重大事件公告", "10-Q": "季度業績報告",
           "10-K": "年度報告", "SC 13G": "大股東持股變動", "SC 13D": "大股東持股變動（可能有意見）",
           "SC 13G/A": "大股東持股變動", "SC 13D/A": "大股東持股變動", "S-3": "準備發行新股／債券",
           "424B2": "發行證券", "424B5": "發行新股／債券", "DEF 14A": "股東大會文件", "3": "新高管首次申報持股"}
ITEM_ZH = {"1.01": "簽重大合約", "1.02": "終止重大合約", "2.01": "收購或出售資產", "2.02": "公布業績",
           "2.03": "新借貸", "2.05": "裁員／重組", "2.06": "資產減值", "3.01": "上市地位問題",
           "5.02": "高層人事變動", "5.03": "修改公司章程", "5.07": "股東投票結果", "7.01": "對外公布資料",
           "8.01": "其他重要事件", "9.01": "財務報表／附件"}


def ohlc(t, start, end):
    import yfinance as yf
    for attempt in range(3):
        try:
            h = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
            if h is None or h.empty:
                return None
            h = h[["Open", "High", "Low", "Close"]].astype(float)
            h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
            return h[~h.index.duplicated(keep="last")]
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


def check(h, day):
    """回傳 dict（符合條件）或者 None。h = 已調整 OHLC，最後一行要係 day。"""
    d = pd.Timestamp(day)
    if h is None or d not in h.index:
        return None
    h = h[h.index <= d]
    c = h["Close"]
    if len(c) < 253:
        return None
    last, low = float(c.iloc[-1]), float(h["Low"].iloc[-1])
    prev_year = c[c.index.year < d.year]
    if prev_year.empty:
        return None
    ytd = last / float(prev_year.iloc[-1]) - 1
    hi52 = float(c.iloc[-252:].max())
    ma10, ma20, ma200 = float(c.iloc[-10:].mean()), float(c.iloc[-20:].mean()), float(c.iloc[-200:].mean())
    ma20_5 = float(c.iloc[-25:-5].mean())
    if not (ytd >= 0.30 and last >= hi52 * 0.7 and last > ma200):
        return None
    if not (ma20 > ma20_5 and last >= ma20 and low <= ma10 * 1.01):
        return None
    risk = 1 - low / last
    return {"ytd": ytd, "close": last, "low": low, "ma10": ma10, "ma20": ma20, "ma200": ma200, "hi52": hi52,
            "risk": risk, "too_far": risk > 0.08}


def news(t, days=3, n=2):
    import yfinance as yf
    out = []
    try:
        items = yf.Ticker(t).news or []
    except Exception:
        return out
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    for it in items:
        c = it.get("content", it)
        title = c.get("title")
        when = None
        if c.get("pubDate"):
            try:
                when = datetime.fromisoformat(c["pubDate"].replace("Z", "+00:00"))
            except Exception:
                when = None
        elif it.get("providerPublishTime"):
            when = datetime.fromtimestamp(int(it["providerPublishTime"]), timezone.utc)
        pub = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else it.get("publisher")
        url = ((c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else None) or it.get("link")
        if title and (when is None or when >= cutoff):
            out.append({"title": title, "when": when, "pub": pub or "", "url": url or ""})
        if len(out) >= n:
            break
    return out


def next_earnings(t):
    import yfinance as yf
    try:
        cal = yf.Ticker(t).calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ed:
            ed = ed[0] if isinstance(ed, (list, tuple)) else ed
            return pd.Timestamp(ed).date()
    except Exception:
        pass
    return None


_CIK = None


def cik_of(t):
    global _CIK
    if _CIK is None:
        try:
            j = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_UA, timeout=30).json()
            _CIK = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in j.values()}
        except Exception as e:
            print(f"SEC 代號表攞唔到：{e}")
            _CIK = {}
    return _CIK.get(t.upper())


def filings(t, days=7):
    cik = cik_of(t)
    if not cik:
        return [], None
    link = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&owner=include&count=40"
    try:
        j = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=SEC_UA, timeout=30).json()
        r = j["filings"]["recent"]
    except Exception as e:
        print(f"{t} SEC 公告攞唔到：{e}")
        return [], link
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    out = []
    for form, fd, items in zip(r.get("form", []), r.get("filingDate", []), r.get("items", [""] * len(r.get("form", [])))):
        if fd < cutoff:
            break
        out.append({"form": form, "date": fd, "items": items or ""})
    time.sleep(0.2)
    return out, link


def filing_text(fl):
    if not fl:
        return "冇"
    cnt = {}
    for f in fl:
        cnt.setdefault(f["form"], []).append(f)
    parts = []
    for form, fs in cnt.items():
        zh = FORM_ZH.get(form, "其他文件")
        extra = ""
        if form.startswith("8-K"):
            its = sorted({i.strip() for f in fs for i in f["items"].split(",") if i.strip()})
            names = [ITEM_ZH.get(i, i) for i in its if i != "9.01"]
            if names:
                extra = "：" + "、".join(names)
        parts.append(f"{form} {zh}{extra}" + (f" ×{len(fs)}" if len(fs) > 1 else "") + f"（{max(f['date'] for f in fs)[5:]}）")
    return "；".join(parts)


def append_log(day, rows):
    os.makedirs("paper_log", exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a") as f:
        if new:
            f.write("date\tticker\tytd\tclose\tlow_stop\tma10\tma20\trisk_pct\tpicked\n")
        for t, s, picked in rows:
            f.write(f"{day}\t{t}\t{s['ytd'] * 100:.1f}\t{s['close']:.2f}\t{s['low']:.2f}\t{s['ma10']:.2f}\t{s['ma20']:.2f}\t"
                    f"{s['risk'] * 100:.1f}\t{int(picked)}\n")


def main():
    dry = "--dry" in sys.argv
    final = "--final" in sys.argv
    day = PR.market_day()
    if "--scheduled" in sys.argv and os.path.exists(DONE) and open(DONE).read().strip() == str(day):
        print(f"{day} 已經發過，唔再發。")
        return
    tickers, names = MM.current_sp500()
    start = (pd.Timestamp(day) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data, have = {}, 0
    for k, t in enumerate(tickers):
        h = ohlc(t, start, end)
        data[t] = h
        if h is not None and pd.Timestamp(day) in h.index:
            have += 1
        time.sleep(0.1)
        if (k + 1) % 100 == 0:
            print(f"下載 {k + 1}/{len(tickers)}，有今日數據 {have}")
    cover = have / max(len(tickers), 1)
    print(f"{day}：有今日（{day}）數據 {have}/{len(tickers)}（{cover * 100:.0f}%）")
    if cover < 0.5:
        print("大部分股票冇今日數據（可能休市或者 Yahoo 未更新），今次唔發。")
        return
    if cover < 0.95 and not final:
        print("今日數據未齊，等後備時間再試。")
        return
    sig = {}
    for t, h in data.items():
        s = check(h, day)
        if s:
            sig[t] = s
    ok = sorted([t for t, s in sig.items() if not s["too_far"]], key=lambda t: -sig[t]["ytd"])
    far = [t for t, s in sig.items() if s["too_far"]]
    picks = ok[:MAX_PICKS]
    append_log(day, [(t, sig[t], t in picks) for t in sorted(sig, key=lambda t: -sig[t]["ytd"])])

    lines = [f"🔍 *回調訊號掃描* {day}（標普500，收市後）", "",
             f"今日符合條件：{len(ok)} 隻" + (f"（另外 {len(far)} 隻止蝕太遠已剔走：{'、'.join(far[:6])}）" if far else "")
             + (f"；顯示今年升最多嘅 {len(picks)} 隻" if picks else ""),
             f"（今日數據齊全度 {cover * 100:.0f}%）", ""]
    if not picks:
        lines.append("⚪ 今日冇股票符合回調訊號。")
    for n, t in enumerate(picks, 1):
        s = sig[t]
        lines.append(f"*{n}. {t}*（{names.get(t, '')}）今年 {s['ytd'] * 100:+.0f}%")
        lines.append(f"　收市 ${s['close']:,.2f} ｜ 今日最低 ${s['low']:,.2f} ｜ 10日線 ${s['ma10']:,.2f} ｜ 20日線 ${s['ma20']:,.2f}")
        lines.append(f"　止蝕位 ${s['low']:,.2f}（離收市 -{s['risk'] * 100:.1f}%）｜ 下一個交易日收市低過 ${s['low']:,.2f} 就唔買")
        ed = next_earnings(t)
        if ed:
            gap = np.busday_count(day, ed)
            if 0 <= gap <= 10:
                lines.append(f"　⚠️ 下次業績：{ed}（{gap} 個交易日後）— 業績日有機會裂口大跌，止蝕可能失效")
            elif gap > 10:
                lines.append(f"　下次業績：{ed}")
        nw = news(t)
        if nw:
            for x in nw:
                w = x["when"].strftime("%m-%d") if x["when"] else ""
                lines.append(f"　📰 {w} {x['title']}（{x['pub']}）{x['url']}")
        else:
            lines.append("　📰 最近3日冇新聞")
        fl, link = filings(t)
        lines.append(f"　📄 公告（7日內）：{filing_text(fl)}" + (f"\n　　{link}" if fl and link else ""))
        lines.append("")
    lines += ["規則：今年升≥30% + 離52週最高位唔夠30% + 喺200日線上面；今日回落掂到10日線（最多高1%）+ 收市企喺20日線上面"
              " + 20日線向上；止蝕 = 今日最低價（太遠 >8% 唔要）。",
              "⚠️ 第三十七輪回測：呢個方法 2015 年起每年 -2.1%（跌穿10日線賣），未通過標準；只係觀察，唔會模擬交易。",
              "_唔係投資建議_"]
    text = "\n".join(lines)
    print(text)
    if not dry:
        # Telegram 每條上限約4000字：太長就分開發
        chunk, size = [], 0
        for ln in lines:
            if size + len(ln) + 1 > 3800 and chunk:
                bot.send_telegram("\n".join(chunk))
                chunk, size = [], 0
            chunk.append(ln)
            size += len(ln) + 1
        if chunk:
            bot.send_telegram("\n".join(chunk))
        os.makedirs("paper_log", exist_ok=True)       # 手動發都記低，避免排程再發一次
        with open(DONE, "w") as f:
            f.write(str(day))


if __name__ == "__main__":
    main()
