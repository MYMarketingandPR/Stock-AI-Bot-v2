"""
pullback_scan.py — 每日「回調訊號」掃描（Esther 2026-10-23 要求）：每日收市後揀最多5隻符合條件嘅標普500股票，連重要新聞同公告發 Telegram。

條件（同第三十七輪入場規則一樣）：
  第1步 強勢股（同線E）：今年升 ≥ 30%；收市 ≥ 52週最高收市 × 0.7；收市 > 200日平均。
  第2步 回調訊號（當日收市後）：20日平均向上（今日 > 5日前）；收市 ≥ 20日平均；當日最低價 ≤ 10日平均 × 1.01。
  第3步：止蝕位 = 當日最低價；收市離最低價超過 8% 就剔走（止蝕太遠）。多過5隻就揀今年升最多嘅5隻。
  參考做法：下一個交易日收市買入；如果嗰日收市已經低過止蝕位就唔買。
⚠️ 第三十七輪回測（跌穿10日線賣）2015年起每年 -2.1%，未通過標準 —— 呢個掃描只係觀察，唔係投資建議，唔會自動模擬交易。

新聞：Yahoo Finance RSS（攞唔到就用 Google News RSS），最近3日、每隻最多2條。公告：Yahoo 提供嘅美國證監會文件清單（最近7日；證監會網站拒絕 GitHub 連線；唔包括 Form 4）。
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
MAX_PICKS = 10                      # 2026-10-23 Esther：由5隻改做10隻
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
            h = h[["Open", "High", "Low", "Close", "Volume"]].astype(float)
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
    # ATR(14)（Wilder 平均真實波幅）：每日正常上落大約幾多錢
    pc = c.shift(1)
    tr = pd.concat([h["High"] - h["Low"], (h["High"] - pc).abs(), (h["Low"] - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    # 成交量：今日 ÷ 之前50日平均；近5日平均 ÷ 之前50日平均；今日喺過去60日排第幾（百分位）
    v = h["Volume"] if "Volume" in h else pd.Series(dtype=float)
    vol = vr = vr5 = vpct = vd = None
    if len(v) >= 60 and v.iloc[-55:-5].mean() > 0:
        vol = float(v.iloc[-1])
        base = float(v.iloc[-51:-1].mean())
        vr = vol / base if base > 0 else None
        vr5 = float(v.iloc[-5:].mean()) / base if base > 0 else None
        vpct = float((v.iloc[-60:] <= vol).mean() * 100)
        chg5 = c.iloc[-5:].values / c.iloc[-6:-1].values - 1                 # 最近5日每日升跌
        down = chg5 < 0
        vd = float(v.iloc[-5:].values[down].mean()) / base if base > 0 and down.any() else None
    return {"ytd": ytd, "close": last, "low": low, "ma10": ma10, "ma20": ma20, "ma200": ma200, "hi52": hi52,
            "risk": risk, "too_far": risk > 0.08, "atr": atr, "vol": vol, "vr": vr, "vr5": vr5, "vpct": vpct,
            "vd": vd, "chg": last / float(c.iloc[-2]) - 1}


def vol_word(r):
    if r is None:
        return "—"
    return "大幅多過平時" if r >= 2 else "多過平時" if r >= 1.2 else "差唔多" if r >= 0.8 else "少過平時"


def _rss(url, cutoff, n):
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=30)
    r.raise_for_status()
    out = []
    for it in ET.fromstring(r.content).findall(".//item"):
        title, link, pd_ = it.findtext("title"), it.findtext("link"), it.findtext("pubDate")
        try:
            when = parsedate_to_datetime(pd_) if pd_ else None
        except Exception:
            when = None
        if title and (when is None or when >= cutoff):
            out.append({"title": title.strip(), "when": when, "pub": "", "url": link or ""})
        if len(out) >= n:
            break
    return out


def news(t, name="", days=3, n=2):
    """最近幾日新聞標題（Yahoo Finance RSS；攞唔到就用 Google News RSS）。回傳 None = 兩個來源都攞唔到。"""
    from urllib.parse import quote
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        return _rss(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US", cutoff, n)
    except Exception as e:
        print(f"{t} Yahoo 新聞攞唔到：{e}")
    try:
        q = quote(f"{name} {t} stock".strip())
        return _rss(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en", cutoff, n)
    except Exception as e:
        print(f"{t} Google 新聞攞唔到：{e}")
    return None


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


SECTOR_ZH = {"Information Technology": "資訊科技", "Health Care": "醫療", "Financials": "金融", "Consumer Discretionary": "非必需消費",
             "Consumer Staples": "必需消費", "Industrials": "工業", "Materials": "原材料", "Energy": "能源", "Utilities": "公用",
             "Communication Services": "通訊服務", "Real Estate": "房地產"}


def sectors():
    """標普500成分股 → （行業中文, 細分行業英文）（Wikipedia GICS Sector / Sub-Industry）。攞唔到就回傳 {}, {}。"""
    import io
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        t = pd.read_html(io.StringIO(html))[0]
        keys = [str(a).replace(".", "-") for a in t["Symbol"]]
        return (dict(zip(keys, [SECTOR_ZH.get(b, b) for b in t["GICS Sector"]])),
                dict(zip(keys, t["GICS Sub-Industry"].astype(str))))
    except Exception as e:
        print(f"行業分類攞唔到：{e}")
        return {}, {}


def filings(t, days=7):
    """最近幾日向美國證監會提交嘅文件（Yahoo 提供嘅清單；證監會網站會拒絕 GitHub 連線）。
    回傳 None = 攞唔到。⚠️ Yahoo 清單唔包括高管買賣（Form 4）。"""
    import yfinance as yf
    try:
        fl = yf.Ticker(t).sec_filings or []
    except Exception as e:
        print(f"{t} 公告攞唔到：{e}")
        return None
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    out = []
    for f in fl:
        d = str(f.get("date", ""))[:10]
        if d >= cutoff:
            out.append({"form": str(f.get("type", "")), "date": d, "title": str(f.get("title", "")), "url": f.get("edgarUrl", "")})
    return out


def filing_text(fl):
    if fl is None:
        return "攞唔到（資料來源出錯）"
    if not fl:
        return "冇"
    parts = []
    for f in fl[:4]:
        zh = FORM_ZH.get(f["form"], "")
        parts.append(f"{f['date'][5:]} {f['form']}" + (f" {zh}" if zh else "") + (f"（{f['title']}）" if f["title"] else ""))
    return "；".join(parts) + (f" 等 {len(fl)} 份" if len(fl) > 4 else "")


def fmt2(x):
    return "" if x is None else f"{x:.2f}"


def append_log(day, rows, sec):
    os.makedirs("paper_log", exist_ok=True)
    header = "date\tticker\tytd\tclose\tlow_stop\tma10\tma20\trisk_pct\tpicked\tsector\tatr\tatr_pct\tvol_ratio\tvol5_ratio\n"
    ncol = header.count("\t") + 1
    if os.path.exists(LOG):
        old = open(LOG).read().splitlines()
        if old and old[0] + "\n" != header:                  # 欄位有加：舊紀錄補空格
            with open(LOG, "w") as f:
                f.write(header + "".join(ln + "\t" * (ncol - 1 - ln.count("\t")) + "\n" for ln in old[1:]))
        old_rows = [ln for ln in open(LOG).read().splitlines()[1:] if not ln.startswith(f"{day}\t")]
        with open(LOG, "w") as f:                          # 同一日重跑就覆蓋返嗰日
            f.write(header + "".join(ln + "\n" for ln in old_rows))
    else:
        with open(LOG, "w") as f:
            f.write(header)
    with open(LOG, "a") as f:
        for t, s, picked in rows:
            f.write(f"{day}\t{t}\t{s['ytd'] * 100:.1f}\t{s['close']:.2f}\t{s['low']:.2f}\t{s['ma10']:.2f}\t{s['ma20']:.2f}\t"
                    f"{s['risk'] * 100:.1f}\t{int(picked)}\t{sec.get(t, '')}\t{s['atr']:.2f}\t{s['atr'] / s['close'] * 100:.2f}\t"
                    f"{fmt2(s['vr'])}\t{fmt2(s['vr5'])}\n")


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
    sec, sub = sectors()
    append_log(day, [(t, sig[t], t in picks) for t in sorted(sig, key=lambda t: -sig[t]["ytd"])], sec)
    cnt = {}
    for t in picks:
        cnt[sec.get(t, "未知")] = cnt.get(sec.get(t, "未知"), 0) + 1

    lines = [f"🔍 *回調訊號掃描* {day}（標普500，收市後）", "",
             f"今日符合條件：{len(ok)} 隻" + (f"（另外 {len(far)} 隻止蝕太遠已剔走：{'、'.join(far[:6])}）" if far else "")
             + (f"；顯示今年升最多嘅 {len(picks)} 隻" if picks else ""),
             f"（今日數據齊全度 {cover * 100:.0f}%）",
             ("行業分佈：" + "、".join(f"{k} {v} 隻" for k, v in sorted(cnt.items(), key=lambda x: -x[1]))
              + ("（⚠️ 有行業佔一半或以上，比較集中）" if picks and max(cnt.values()) * 2 >= len(picks) else "")) if picks else "", ""]
    if not picks:
        lines.append("⚪ 今日冇股票符合回調訊號。")
    for n, t in enumerate(picks, 1):
        s = sig[t]
        stop_atr = (s["close"] - s["low"]) / s["atr"] if s["atr"] > 0 else None
        ok_atr = stop_atr is not None and 0.7 <= stop_atr <= 1.2           # 短炒條件①：止蝕 0.7–1.2 個 ATR
        ok_vol = s["vd"] is not None and s["vd"] < 1.2                       # 短炒條件②：最近5日跌市日成交量唔多過平時
        tag = " ✅✅" if ok_atr and ok_vol else (" ✅" if ok_atr or ok_vol else "")
        lines.append(f"*{n}. {t}*（{names.get(t, '')}｜{sec.get(t, '行業未知')}）今年 {s['ytd'] * 100:+.0f}%{tag}")
        same = [x for x in picks if x != t and sub.get(x) and sub.get(x) == sub.get(t)]
        if same:
            lines.append(f"　⚠️ 同類股：同 {'、'.join(same)} 同屬「{sub.get(t)}」，會一齊升跌，揀一隻就夠")
        lines.append(f"　收市 ${s['close']:,.2f} ｜ 今日最低 ${s['low']:,.2f} ｜ 10日線 ${s['ma10']:,.2f} ｜ 20日線 ${s['ma20']:,.2f}")
        lines.append(f"　離52週最高收市 ${s['hi52']:,.2f} 仲有 {(s['hi52'] / s['close'] - 1) * 100:+.1f}%（上面空間）")
        lines.append(f"　止蝕位 ${s['low']:,.2f}｜止蝕距離 -{s['risk'] * 100:.1f}%"
                     + (f"（約 {stop_atr:.1f} 個 ATR）" if stop_atr is not None else "") + (" ✅" if ok_atr else "")
                     + f"｜ 下一個交易日收市低過 ${s['low']:,.2f} 就唔買")
        lines.append(f"　ATR(14) ${s['atr']:,.2f}（每日正常上落約 {s['atr'] / s['close'] * 100:.1f}%）"
                     + ("｜ ⚠️ 止蝕少過半個 ATR，正常上落都可能掃到" if stop_atr is not None and stop_atr < 0.5 else ""))
        if s["vr"] is not None:
            up = s["chg"] >= 0
            mood = ("升市有買盤" if up else "跌市有沽壓") if s["vr"] >= 1.2 else ("升市但成交唔多" if up else "跌市但成交唔多")
            lines.append(f"　成交量：今日{'升' if up else '跌'} {s['chg'] * 100:+.1f}%，成交 {s['vol'] / 1e6:.2f} 百萬股 = 之前50日平均 {s['vr']:.1f} 倍"
                         f"（{vol_word(s['vr'])}，{mood}）｜ 近5日平均 {s['vr5']:.1f} 倍")
            lines.append("　回落日成交量：" + (f"最近5日入面下跌嗰幾日平均 {s['vd']:.1f} 倍（{vol_word(s['vd'])}）" + (" ✅" if ok_vol else " — 跌嗰陣成交多，有人沽貨")
                                              if s["vd"] is not None else "最近5日冇下跌日"))
        ed = next_earnings(t)
        if ed:
            gap = np.busday_count(day, ed)
            if gap == 0:
                lines.append(f"　⚠️ 業績：今日（{ed}）公布，留意明日開市反應")
            elif 0 < gap <= 10:
                lines.append(f"　⚠️ 下次業績：{ed}（{gap} 個交易日後）— 業績日有機會裂口大跌，止蝕可能失效")
            elif gap > 10:
                lines.append(f"　下次業績：{ed}")
        nw = news(t, names.get(t, ""))
        if nw is None:
            lines.append("　📰 新聞攞唔到（資料來源出錯）")
        elif nw:
            for x in nw:
                w = x["when"].strftime("%m-%d") if x["when"] else ""
                lines.append(f"　📰 {w} {x['title']}" + (f"（{x['pub']}）" if x["pub"] else "") + f" {x['url']}")
        else:
            lines.append("　📰 最近3日冇新聞")
        fl = filings(t)
        lines.append(f"　📄 公告（7日內）：{filing_text(fl)}" + (f"\n　　{fl[0]['url']}" if fl and fl[0]["url"] else ""))
        lines.append("")
    lines += ["✅ 短炒參考：止蝕距離 0.7 至 1.2 個 ATR（唔太近、唔太遠）✅；最近5日下跌嗰幾日成交量少過之前50日平均 1.2 倍 ✅；兩樣都符合 = ✅✅。"
              "呢兩個係常見短炒講法，我哋未回測過。",
              "公告來源：Yahoo 提供嘅美國證監會文件清單（唔包括高管買賣 Form 4）。新聞：Yahoo Finance（英文標題）。",
              "規則：今年升≥30% + 離52週最高位唔夠30% + 喺200日線上面；今日回落掂到10日線（最多高1%）+ 收市企喺20日線上面"
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
