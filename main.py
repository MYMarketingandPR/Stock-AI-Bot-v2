import os
import re
import time
from datetime import datetime
import requests
import pandas as pd
import yfinance as yf
from groq import Groq

from strategy_core import (
    compute_indicators,
    check_breakout_signal,
    candidate_snapshot_dict,
    format_quant_snapshot_text,
    MEGA_CAP_TICKERS,
    HIGH_BETA_TICKERS,
)

# ==========================================
# 0. 股票池（已搬去strategy_core.py，同backtest_screener.py共用同一份）
# ==========================================
WATCHLIST = MEGA_CAP_TICKERS + HIGH_BETA_TICKERS

# ==========================================
# 1. 鑰匙設定
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
SHEET_WEBHOOK_URL = os.environ.get("SHEET_WEBHOOK_URL")

# ==========================================
# 2. 推送與紀錄模組
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
    ev, pivot, stop_loss = "N/A", "手動查閱", "手動查閱"

    ev_match = re.search(r"(?:期望值|EV)[^|\n:]*[:|]?\s*([-+−‑]?\s*\d+\.?\d*)", analysis_text, re.IGNORECASE)
    if ev_match:
        ev = ev_match.group(1).replace(" ", "")

    for line in analysis_text.split("\n"):
        if pivot == "手動查閱" and re.search(r"(?:Entry|買入價|買入點|試探買入|進場)", line, re.IGNORECASE):
            prices = re.findall(r"\$\s*(\d+\.?\d*)", line)
            if prices:
                pivot = prices[0]
        if stop_loss == "手動查閱" and re.search(r"(?:Stop[-‑–—]?Loss|硬性止損|止損位|止損點|止損)", line, re.IGNORECASE):
            prices = re.findall(r"\$\s*(\d+\.?\d*)", line)
            if prices:
                stop_loss = prices[0]

    try:
        p_val, s_val = float(pivot), float(stop_loss)
        if s_val >= p_val:
            stop_loss = f"{s_val}(⚠️顛倒)"
    except Exception:
        pass

    data = {
        "date": datetime.now().strftime("%Y-%m-%d"), "ticker": ticker,
        "strategy": strategy_name, "vix": f"{vix:.2f}",
        "ev": ev, "pivot": pivot, "stop_loss": stop_loss,
    }
    try:
        requests.post(SHEET_WEBHOOK_URL, json=data, timeout=10)
    except Exception as e:
        print(f"Google Sheets 推送失敗: {e}")


# ==========================================
# 3. 市場環境 + 批次掃描（效能/穩定性修正）
# ==========================================
def get_market_context():
    try:
        vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]
        spy = yf.download("SPY", period="6mo", progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy_sma50 = spy["Close"].rolling(50).mean().iloc[-1]
        spy_bullish = spy["Close"].iloc[-1] > spy_sma50
        return float(vix), bool(spy_bullish)
    except Exception:
        return 20.0, True


def scan_breakout_candidates(spy_bullish):
    """改用批次下載（yf.download接受整個list），
    取代原本250次逐隻單獨download——大幅減少request數量，
    亦減低好似FI/MMC/SQ呢類因為rate limit而被誤判「delisted」嘅機會。"""
    print(f"正在批次下載 {len(WATCHLIST)} 隻股票嘅歷史數據...")
    try:
        batch = yf.download(WATCHLIST, period="1y", group_by="ticker", threads=True, progress=False)
    except Exception as e:
        print(f"批次下載失敗: {e}")
        return []

    candidates = []
    for t in WATCHLIST:
        try:
            if isinstance(batch.columns, pd.MultiIndex):
                if t not in batch.columns.get_level_values(0):
                    continue
                df = batch[t].dropna(how="all")
            else:
                df = batch  # 只有一隻ticker時yfinance唔會開MultiIndex，理論上WATCHLIST好多隻唔會行到呢支
            if df.empty or len(df) < 210:
                continue

            df = compute_indicators(df)
            row = df.iloc[-1]
            if check_breakout_signal(row, spy_is_bullish=spy_bullish):
                candidates.append(candidate_snapshot_dict(t, row))
        except Exception:
            continue
    return candidates


def get_fundamentals(ticker):
    """獨立處理.info呼叫 —— 同價格數據分開try/except，
    就算.info失敗(呢個endpoint出名唔穩定)，都唔會累到連現價/技術數據都攞唔到。"""
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "無法確認")
        eps_growth = info.get("earningsQuarterlyGrowth")
        eps_growth = f"{eps_growth * 100:.1f}%" if isinstance(eps_growth, (int, float)) else "資料不足"
        return sector, eps_growth
    except Exception:
        return "無法確認（.info抓取失敗）", "資料不足（.info抓取失敗）"


def get_recent_news(ticker):
    try:
        news_list = yf.Ticker(ticker).news[:3]
        return "\n".join([f"▪️ {n.get('title', '')}" for n in news_list]) if news_list else "近期無重大新聞。"
    except Exception:
        return "近期無重大新聞。"


# ==========================================
# 4. AI 提示詞
# ==========================================
def load_prompt(candidate, sector, eps_growth, news_text):
    ticker = candidate["ticker"]
    date = datetime.now().strftime("%Y-%m-%d")
    group_name = "Mega Cap（傳統巨無霸與穩健增長）" if ticker in MEGA_CAP_TICKERS else "High Beta（高動能與投機股）"
    quant_text = format_quant_snapshot_text(candidate)

    return f"""
角色設定：你是一位擁有 30 年華爾街對沖基金經驗的頂尖量化專家 (CIO)。

【格式鐵律】：
1. 禁止使用Markdown表格語法（帶 |---| 嘅表格），改用清單點(-)同粗體標題(###)。
2. 【嚴格規則】：下面【實際技術數據】已經係系統確認嘅真實數字，可以直接引用、唔使話「資料不足」。
   但超出呢個範圍嘅資訊（例如未提供嘅深入基本面、未來催化劑細節），如果真係冇根據，
   請老實寫「資料不足，無法判斷」，唔准為咗有個答案而自行創作或憑記憶估計一個數字——
   呢條規則過去令我哋避免咗一次嚴重嘅幻覺事故（將一間生科公司錯誤分類做半導體股），
   所以呢度係硬性要求，不可再被覆蓋。

分析標的：{ticker}（日期：{date} ｜ 分組：{group_name}）
【當前觸發策略】：爆量缺口突破 Volume Breakout
【本策略已通過嘅量化篩選條件】：
1. 短期均線多頭排列（SMA20 > SMA50，且 Close > SMA50 > SMA200）
2. 拒絕衰竭缺口（單日漲幅 < 10%）
3. 強勢收盤（收盤價企於全日波幅上半部）
4. RSI(14) < 85（避免追入極端超買）
此策略喺歷史回測中整體呈現正期望值，但個別交易嘅實際結果受市場環境影響，
不代表呢一注一定賺錢，請基於下面呢隻股票實際嘅數據獨立判斷，唔好純粹套用歷史平均數。

【實際技術數據（系統已確認，可直接引用）】：
{quant_text}

【基本面】：
- 所屬行業：{sector}
- EPS 季增長：{eps_growth}

【即時新聞摘要】：
{news_text}

---
### 階段一：終極決策前置
- 【指令】：(強烈買入 / 輕倉試探 / 防守觀望)
- 【主觀勝率】：基於上面實際數據嘅獨立判斷（唔係固定套用某個歷史平均數）
- 【期望值(EV)】：+/- X.XX%
- 【核心邏輯】：精煉解釋決策原因，並註明是基於哪一項實際數據或新聞。

### 階段二：族群與基本面催化劑
- 簡述呢間公司目前嘅營運或題材亮點，是否有足夠動力支撐呢波突破？如果新聞冇提供相關資訊，請老實講。

### 階段三：量化風控參數
- Entry（建議買入價）：$XX.XX
- Stop-Loss（硬性止損）：$XX.XX（定在買入價 -7% 處）
- Target T1（波段止盈）：$XX.XX（定在買入價 +10% 處）
- 資金保護策略：持倉期限最長 20 日；若獲利達 +5%，止損上移至買入價打和點。

### 階段四：紅隊壓力測試
- 致命陷阱：儘管技術面符合條件，呢筆突破最大嘅潛在風險或基本面盲區係咩？
"""


# ==========================================
# 5. 主程式執行
# ==========================================
def main():
    print("啟動 AI 量化交易員...")
    if not GROQ_API_KEY:
        print("❌ 找不到 GROQ_API_KEY")
        return

    vix, spy_bullish = get_market_context()

    if not spy_bullish:
        msg = "⚠️ **大盤濾網啟動**\n標普500跌破50日均線，目前禁止做多，AI 提早下班。"
        print(msg)
        send_telegram(msg)
        return

    send_telegram(f"⏳ **AI交易員已上線**\n📊 VIX: `{vix:.2f}`\n大盤趨勢: `多頭`\n正在批次掃描 {len(WATCHLIST)} 隻股票...")

    candidates = scan_breakout_candidates(spy_bullish)

    if not candidates:
        msg = f"⚠️ 今日 {len(WATCHLIST)} 隻核心股皆無觸發高品質爆量突破，守住資金，AI 提早下班。"
        print(msg)
        send_telegram(msg)
        return

    names = ", ".join(c["ticker"] for c in candidates)
    send_telegram(f"🎯 **今日精準突破名單: {names}**\n正在呼叫 AI 大腦診斷...")

    client = Groq(api_key=GROQ_API_KEY)

    for c in candidates:
        t = c["ticker"]
        try:
            print(f"正在分析 {t} ...")
            sector, eps_growth = get_fundamentals(t)
            news_data = get_recent_news(t)
            prompt = load_prompt(c, sector, eps_growth, news_data)

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.2,
            )
            analysis = response.choices[0].message.content

            group = "Mega Cap" if t in MEGA_CAP_TICKERS else "High Beta"
            send_telegram(f"🔥 **【{t}｜{group}｜爆量突破】機密備忘錄** 🔥\n\n{analysis[:3800]}")
            log_to_google_sheet(t, "Volume Breakout", vix, analysis)
            time.sleep(5)
        except Exception as e:
            print(f"❌ 分析 {t} 失敗: {e}")


if __name__ == "__main__":
    main()
