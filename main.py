import os
import re
import time
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from finvizfinance.screener.technical import Technical
from groq import Groq

# ==========================================
# 0. 系統模式切換開關
# ==========================================
RUN_MODE = "AUTO"
SNIPER_TICKERS = ["SNDK", "SKHY"]

# ==========================================
# 1. 鑰匙設定
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
SHEET_WEBHOOK_URL = os.environ.get("SHEET_WEBHOOK_URL")

# ==========================================
# 2. 推送與紀錄模組（加咗 strategy 欄位）
# ==========================================
def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text[:4000], "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 400:
            payload["parse_mode"] = None
            requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram 推送失敗: {e}")


def log_to_google_sheet(ticker, strategy_name, vix, analysis_text):
    if not SHEET_WEBHOOK_URL:
        return
    ev_match = re.search(r"期望值.*?EV.*?([+-]?\d+\.?\d*)\s*%", analysis_text, re.IGNORECASE)
    ev = ev_match.group(1) if ev_match else "N/A"
    pivot_match = re.search(r"(?:Pivot|買入價|試探倉).*?\$?(\d+\.?\d*)", analysis_text, re.IGNORECASE)
    pivot = pivot_match.group(1) if pivot_match else "手動查閱"
    stop_match = re.search(r"(?:止損|Stop-Loss).*?\$?(\d+\.?\d*)", analysis_text, re.IGNORECASE)
    stop_loss = stop_match.group(1) if stop_match else "手動查閱"

    data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "ticker": ticker,
        "strategy": strategy_name,   # 新增：分開紀錄邊注係squeeze、邊注係minervini
        "vix": f"{vix:.2f}",
        "ev": ev,
        "pivot": pivot,
        "stop_loss": stop_loss,
    }
    try:
        requests.post(SHEET_WEBHOOK_URL, json=data, timeout=10)
    except Exception as e:
        print(f"Google Sheets 網路請求失敗: {e}")


# ==========================================
# 3. 市場數據、新聞、StockTwits情緒
# ==========================================
def get_vix():
    try:
        vix = yf.Ticker("^VIX")
        data = vix.history(period="1d")
        return float(data["Close"].iloc[-1])
    except Exception:
        return 20.0


def get_recent_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news[:5]
        news_text = ""
        for n in news_list:
            title = n.get("title", "")
            summary = n.get("summary", "")
            news_text += f"▪️ {title}: {summary}\n"
        return news_text if news_text else "近期無重大新聞或 SEC Filings。"
    except Exception as e:
        return f"新聞抓取失敗: {e}"


def get_stocktwits_sentiment(ticker):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        messages = data.get("messages", [])[:30]

        bullish, bearish, untagged = 0, 0, 0
        sample_texts = []
        for m in messages:
            sentiment = m.get("entities", {}).get("sentiment")
            label = sentiment.get("basic") if sentiment else None
            followers = m.get("user", {}).get("followers", 0)
            if followers < 20:
                continue
            if label == "Bullish":
                bullish += 1
            elif label == "Bearish":
                bearish += 1
            else:
                untagged += 1
            if len(sample_texts) < 3:
                sample_texts.append(m.get("body", "")[:100])

        total_tagged = bullish + bearish
        if total_tagged == 0:
            return "StockTwits：討論量太低或全部未標記情緒，無法判斷"
        bull_pct = bullish / total_tagged * 100
        return (
            f"StockTwits情緒：Bullish {bullish} / Bearish {bearish} "
            f"（看多比例 {bull_pct:.0f}%），未標記 {untagged} 則。討論樣本：\n"
            + "\n".join(f"- {t}" for t in sample_texts)
        )
    except Exception as e:
        return f"StockTwits抓取失敗: {e}"


def get_technical_snapshot(ticker):
    """攞返現價同你四個常用指標（MACD/KDJ/OBV/神奇九轉）嘅實際數值，
    塞入prompt俾AI用真實數字做判斷，而唔係淨係得個ticker名憑印象（甚至幻覺）砌內容。
    呢個function係直接回應KOD個案驗證咗嘅問題：AI之前完全冇真實數據，
    連公司屬於邊個行業都可以砌錯。"""
    try:
        df = yf.download(ticker, period="9mo", progress=False)
        if df.empty or len(df) < 60:
            return "技術數據不足（歷史數據少於60日），無法計算指標，分析時請特別保守。"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = (dif - dea) * 2

        # KDJ (9,3,3)，手動計算以確保公式可控
        low_n = low.rolling(9).min()
        high_n = high.rolling(9).max()
        rsv = (close - low_n) / (high_n - low_n) * 100
        k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        d = k.ewm(alpha=1 / 3, adjust=False).mean()
        j = 3 * k - 2 * d

        # OBV
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        obv_trend = "上升" if obv.iloc[-1] > obv.iloc[-6] else "下跌"

        # 神奇九轉（簡化版TD Sequential計數）：連續收市價高/低過4日前
        diff4 = close - close.shift(4)
        count, direction = 0, None
        for v in diff4.iloc[-15:]:
            if pd.isna(v):
                continue
            if v > 0:
                count = count + 1 if direction == "up" else 1
                direction = "up"
            elif v < 0:
                count = count + 1 if direction == "down" else 1
                direction = "down"
            else:
                count, direction = 0, None
        nine_turn_status = (
            f"{'向上' if direction == 'up' else '向下'}第{min(count, 9)}轉" if direction else "無明確計數"
        )

        sma50 = close.rolling(50).mean().iloc[-1]
        sma200_series = close.rolling(200).mean()
        sma200 = sma200_series.iloc[-1] if len(close) >= 200 else None
        high52w = close.rolling(min(len(close), 252)).max().iloc[-1]
        vol_ratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]

        lines = [
            f"- 現價：${close.iloc[-1]:.2f}（本次篩選/分析截至 {close.index[-1].date()} 收市，非即時報價）",
            f"- 50日均線：${sma50:.2f}（現價{'高於' if close.iloc[-1] > sma50 else '低於'}此線）",
        ]
        if sma200 is not None:
            lines.append(f"- 200日均線：${sma200:.2f}")
        lines += [
            f"- 距52週高位：{(close.iloc[-1] / high52w - 1) * 100:.1f}%",
            f"- MACD：DIF={dif.iloc[-1]:.3f}, DEA={dea.iloc[-1]:.3f}, Histogram={macd_hist.iloc[-1]:.3f}",
            f"- KDJ：K={k.iloc[-1]:.1f}, D={d.iloc[-1]:.1f}, J={j.iloc[-1]:.1f}",
            f"- OBV近5日走勢：{obv_trend}",
            f"- 神奇九轉：{nine_turn_status}",
            f"- 今日成交量 / 20日均量：{vol_ratio:.2f}x",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"技術指標計算失敗: {e}（此情況下AI分析不應視為有數據支持，應降級為觀望）"


# ==========================================
# 4. 兩套獨立篩選條件
# ==========================================
def get_short_squeeze_filters():
    # 已移除 'Short Ratio' —— 呢個filter key喺呢個finvizfinance版本根本唔存在，
    # 錯誤log直接確認咗；冇「days to cover」嘅原生filter，想要呢個數據要另外用finvizfinance
    # 嘅個股fundament查詢，呢度先簡化處理。
    return {
        "Float Short": "Over 15%",
        "Float": "Under 50M",
        "Average Volume": "Over 500K",
        "Relative Volume": "Over 2",
        "Price": "Over $3",
        "52-Week High/Low": "0-10% below High",  # 原本0-20%唔存在，Finviz貼近高位淨係有0-3/0-5/0-10%
        "InstitutionalOwnership": "Under 50%",  # 已修正：冇空格
    }


def get_minervini_filters():
    return {
        "Market Cap.": "+Small (over $300mln)",
        "Price": "Over $15",
        "Average Volume": "Over 500K",
        "50-Day Simple Moving Average": "Price above SMA50",
        "200-Day Simple Moving Average": "Price above SMA200",
        "52-Week High/Low": "0-10% below High",  # 原本0-25%唔存在；Finviz冇對應preset，
                                                    # 呢度用最接近、但比Minervini原意更嚴格嘅0-10%，
                                                    # 如果想要25%，要喺refine_minervini_candidates()
                                                    # 用實際價格/52週高位數值另外計，呢度未做呢part
        "EPS growththis year": "Positive (>0%)",   # 已修正：field名冇空格（錯誤log確認）；
                                                    # 但 "Positive (>0%)" 呢個數值字串未經錯誤log驗證，
                                                    # 跑之前請用下面嘅方法自己confirm一次
        "EPS growthqtr over qtr": "Positive (>0%)",
        "Sales growthqtr over qtr": "Positive (>0%)",
        "InstitutionalOwnership": "Over 10%",
    }


def run_screener(filters):
    try:
        screener = Technical()
        screener.set_filter(filters_dict=filters)
        df = screener.screener_view()
        if df is None or df.empty:
            return []
        return df["Ticker"].head(5).tolist()
    except Exception as e:
        print(f"Finviz 篩選失敗: {e}")
        return []


def refine_minervini_candidates(tickers):
    """Finviz篩完之後，額外補做150日均線同RS rank proxy，再篩一次"""
    results = []
    for t in tickers:
        try:
            df = yf.download(t, period="18mo", progress=False)
            if df.empty or len(df) < 260:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"]
            sma150 = close.rolling(150).mean()
            sma200 = close.rolling(200).mean()
            trend_ok = (
                close.iloc[-1] > sma150.iloc[-1]
                and sma150.iloc[-1] > sma200.iloc[-1]
                and sma200.iloc[-1] > sma200.iloc[-22]
            )
            r3 = close.iloc[-1] / close.iloc[-63] - 1
            r6 = close.iloc[-1] / close.iloc[-126] - 1
            r9 = close.iloc[-1] / close.iloc[-189] - 1
            r12 = close.iloc[-1] / close.iloc[-252] - 1
            rs_score = 2 * r3 + r6 + r9 + r12
            results.append({"ticker": t, "trend_template_ok": trend_ok, "rs_score": rs_score})
        except Exception as e:
            print(f"{t} refine失敗: {e}")
            continue

    df_result = pd.DataFrame(results)
    if df_result.empty:
        return df_result
    df_result["rs_rank_in_pool"] = df_result["rs_score"].rank(pct=True) * 100
    return df_result.sort_values("rs_rank_in_pool", ascending=False)


# ==========================================
# 5. AI 提示詞（按策略帶入唔同核心邏輯描述）
# ==========================================
STRATEGY_DESC = {
    "短擠壓Setup": "高空倉比率被逼平倉的軋空邏輯，唔靠基本面，靠空倉結構同成交量爆發判斷。",
    "Minervini Trend Template": "SEPA式強勢股邏輯：基本面(EPS/銷售增長)配合技術面趨勢排列(50>150>200日線)同相對強度，屬於趨勢延續交易，唔係逆勢博反彈。",
    "人手指定(雙框架評估)": "請同時用短擠壓同Minervini Trend Template兩套框架分別評估呢隻股票，並各自俾出判斷，唔好混為一談。",
}


def load_prompt(ticker, strategy_name, news_text, stocktwits_text, tech_text):
    date = datetime.now().strftime("%Y-%m-%d")
    strategy_logic = STRATEGY_DESC.get(strategy_name, "")
    return f"""
角色設定：
請你化身一位擁有 30 年華爾街對沖基金 CIO 的頂尖專家。

【嚴格規則 — 必須遵守】：
你只可以使用下面【實際技術數據】【即時新聞摘要】【StockTwits散戶情緒】三個區塊入面
提供嘅真實資料嚟分析。唔准自行創作、估計或憑訓練記憶補充任何未提供嘅具體數值 ——
包括但不限於：公司所屬行業/族群、同業比較對象、機構持股明細、具體均線數值、
空頭持倉變化百分比。如果某一項分析需要嘅信息冇喺下面提供，你必須喺對應部分
寫低「資料不足，無法判斷」，並喺【指令】降級為「防守觀望」，絕對不可以編造一個
聽落合理但冇根據嘅答案。

核心目標：為我剖析股票：{ticker}。
數據截止日期：{date}

【當前系統觸發的策略】：{strategy_name}
【策略核心邏輯】：{strategy_logic}
請根據呢個邏輯評估，但前提是唔違反上面「嚴格規則」。

【實際技術數據】：
{tech_text}

【即時新聞摘要】：
{news_text}

【StockTwits散戶情緒】：
{stocktwits_text}

階段一：終極決策前置
- 【指令】：(強烈買入 / 輕倉試探 / 防守觀望 / 嚴格避開)
- 【主觀勝率】：XX%
- 【期望值(EV)】：+/- X.XX%
- 【一句話核心邏輯】：精煉解釋決策原因，並註明係基於上面邊一項實際數據。

階段二：族群與基本面背景
1. 只根據【即時新聞摘要】入面實際提到嘅信息判斷公司所屬行業/背景；
   如果新聞冇提供足夠信息確認行業，明確寫低「無法從現有資料確認所屬行業」，
   不可以自行推斷或憑記憶填補。

階段三：技術面與極致風控（必須基於【實際技術數據】區塊嘅數字）
1. 確認技術型態是否符合【{strategy_name}】的邏輯，引用返上面實際嘅現價/均線/指標數值。
2. 標出最佳 Entry 買入價與硬性止損點位，價位必須以上面提供嘅「現價」為基礎推算。
3. 目標價 T1/T2，展示 RRR ≥ 3:1。

階段四：紅隊壓力測試
1. 這筆交易最大的陷阱是什麼？如果虧損，最可能是什麼原因？
"""


def risk_check_override(analysis_text, ticker):
    if "止損" not in analysis_text and "Stop-Loss" not in analysis_text:
        return False, "無止損設定"
    return True, "通過初步風控"


# ==========================================
# 6. 主程式執行
# ==========================================
def main():
    print(f"啟動 AI 交易員... 當前模式: {RUN_MODE}")
    if not GROQ_API_KEY:
        error_msg = "❌ 系統錯誤：找不到 GROQ_API_KEY。"
        print(error_msg)
        send_telegram(error_msg)
        return

    vix = get_vix()  # 保留VIX，但改做背景資訊，唔再用嚟揀策略
    candidates = []  # (ticker, strategy_name) 嘅list

    if RUN_MODE == "SNIPER":
        send_telegram(f"⏳ **AI交易員已上線**\n📊 VIX: `{vix:.2f}`\n🎯 單點狙擊模式（雙框架評估）")
        for t in SNIPER_TICKERS:
            candidates.append((t, "人手指定(雙框架評估)"))
    else:
        send_telegram(f"⏳ **AI交易員已上線**\n📊 VIX: `{vix:.2f}`\n🔍 同時掃描短擠壓 + Minervini 兩套候選...")

        squeeze_tickers = run_screener(get_short_squeeze_filters())
        for t in squeeze_tickers[:3]:
            candidates.append((t, "短擠壓Setup"))

        minervini_raw = run_screener(get_minervini_filters())
        if minervini_raw:
            refined = refine_minervini_candidates(minervini_raw)
            if not refined.empty:
                top = refined[refined["trend_template_ok"]].head(3)
                for t in top["ticker"]:
                    candidates.append((t, "Minervini Trend Template"))

    if not candidates:
        msg = "⚠️ 今日兩套策略都無符合條件的標的，AI 交易員提早下班。"
        print(msg)
        send_telegram(msg)
        return

    names = ", ".join(f"{t}({s})" for t, s in candidates)
    send_telegram(f"🎯 **今日候選: {names}**\n正在呼叫 AI 大腦逐一診斷...")

    client = Groq(api_key=GROQ_API_KEY)
    for ticker, strategy_name in candidates:
        try:
            print(f"正在分析 {ticker}（{strategy_name}）...")
            news_data = get_recent_news(ticker)
            stocktwits_data = get_stocktwits_sentiment(ticker)
            tech_data = get_technical_snapshot(ticker)
            prompt = load_prompt(ticker, strategy_name, news_data, stocktwits_data, tech_data)

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.2,
            )
            analysis = response.choices[0].message.content

            passed, reason = risk_check_override(analysis, ticker)
            if not passed:
                send_telegram(f"⚠️ **【{ticker}】遭紅隊否決**\n⛔ 原因: {reason}")
                continue

            send_telegram(f"🔥 **【{ticker}｜{strategy_name}】機密備忘錄** 🔥\n\n{analysis[:3800]}")
            log_to_google_sheet(ticker, strategy_name, vix, analysis)
            time.sleep(5)
        except Exception as e:
            error_msg = f"❌ 分析 {ticker} 時發生錯誤: {e}"
            print(error_msg)
            send_telegram(error_msg)


if __name__ == "__main__":
    main()
