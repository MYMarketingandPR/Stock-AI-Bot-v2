import os
import re
import time
from datetime import datetime
import requests
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

def log_to_google_sheet(ticker, vix, screener_name, analysis_text):
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
        "vix": f"{vix:.2f}",
        "strategy": screener_name,
        "ev": ev,
        "pivot": pivot,
        "stop_loss": stop_loss
    }
    try:
        requests.post(SHEET_WEBHOOK_URL, json=data, timeout=10)
    except Exception as e:
        print(f"Google Sheets 網路請求失敗: {e}")

# ==========================================
# 3. 策略與 Finviz 篩選模組 (三套全天候策略)
# ==========================================
def get_vix():
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        data = vix.history(period="1d")
        return float(data['Close'].iloc[-1])
    except:
        return 20.0

# 策略一：動能突破 (適合 VIX < 18 穩定多頭)
def get_breakout_filters():
    return {
        'Market Cap.': '+Small (over $300mln)',
        'Price': 'Over $15',
        'Average Volume': 'Over 500K',
        '50-Day Simple Moving Average': 'Price above SMA50',
        '200-Day Simple Moving Average': 'Price above SMA200',
        '52-Week High/Low': '0-5% below High',
        'Gap': 'Up',
        'Change from Open': 'Up 2%',
        'Relative Volume': 'Over 2'
    }

# 策略二：高勝率回踩 (適合 18 <= VIX < 25 震盪市)
def get_pullback_filters():
    return {
        'Market Cap.': '+Small (over $300mln)',
        'Price': 'Over $15',
        'Average Volume': 'Over 500K',
        '200-Day Simple Moving Average': 'Price above SMA200', # 長期多頭護航
        '50-Day Simple Moving Average': 'Price above SMA50',   # 中期趨勢向上
        '20-Day Simple Moving Average': 'Price below SMA20'    # 短期跌破月線(回踩)
    }

# 策略三：恐慌抄底 (適合 VIX >= 25 崩盤市)
def get_panic_filters():
    return {
        'Market Cap.': '+Large (over $10bln)', # 只買百億美元以上大型安全權值股
        'Price': 'Over $15',
        'Average Volume': 'Over 500K',
        '50-Day Simple Moving Average': 'Price below SMA50', # 已經跌破季線
        'RSI (14)': 'Oversold (30)'                          # 短期極度超賣
    }

def select_screener_by_vix(vix):
    if vix < 18:
        return '🔥 爆量突破策略 (大盤穩定，強勢攻擊)', get_breakout_filters()
    elif 18 <= vix < 25:
        return '🛡️ 高勝率回踩策略 (大盤震盪，買入支撐)', get_pullback_filters()
    else:
        return '🩸 恐慌抄底策略 (大盤恐慌，撿漏權值股)', get_panic_filters()

def run_screener(filters):
    try:
        screener = Technical()
        screener.set_filter(filters_dict=filters)
        df = screener.screener_view()
        
        if df is None or df.empty:
            return []
        return df['Ticker'].head(3).tolist()
    except Exception as e:
        print(f"Finviz 篩選失敗: {e}")
        return []

# ==========================================
# 4. AI 提示詞與風控 (動態適應策略)
# ==========================================
def load_prompt(ticker, screener_name):
    date = datetime.now().strftime("%Y-%m-%d")
    return f"""
角色設定：
請你化身一位擁有 30 年華爾街對沖基金 CIO 的頂尖專家。

【當前系統觸發的量化策略】：{screener_name}
請務必根據此策略邏輯（突破 / 回踩支撐 / 恐慌超賣）來評估此股票，不要用錯誤的標準來評判。

核心目標：為我剖析股票：{ticker}。
數據截止日期：{date}

階段一：終極決策前置
- 【指令】：(強烈買入 / 輕倉試探 / 防守觀望 / 嚴格避開)
- 【主觀勝率】：XX% 
- 【期望值(EV)】：+/- X.XX% (EV = 勝率% × 平均盈利% - 敗率% × 平均虧損%)
- 【一句話核心邏輯】：精煉解釋決策原因。

階段二：戰術環境與同業族群掃描
1. 診斷目前該股在族群中的位階。

階段三：技術面與入場點
1. 確認技術型態是否符合【{screener_name}】的邏輯。
2. 標出最佳 Entry 買入價。

階段四：籌碼與資金流向
1. 評估機構持股趨勢、空頭結構。

階段五：極致風控與執行藍圖
1. 精確買賣與止損點位 (硬性止損)。
2. 目標價 T1/T2，展示 RRR ≥ 3:1。

階段六：紅隊壓力測試
1. 這筆交易最大的陷阱是什麼？如果虧損，最可能是什麼原因？
"""

def risk_check_override(analysis_text, ticker):
    if "止損" not in analysis_text and "Stop-Loss" not in analysis_text:
        return False, "無止損設定"
    return True, "通過初步風控"

# ==========================================
# 5. 主程式執行
# ==========================================
def main():
    print(f"啟動 AI 交易員... 當前模式: {RUN_MODE}")
    
    if not GROQ_API_KEY:
        error_msg = "❌ 系統錯誤：找不到 GROQ_API_KEY。"
        print(error_msg)
        send_telegram(error_msg)
        return

    vix = get_vix()
    
    if RUN_MODE == "SNIPER":
        screener_name = "🎯 單點狙擊模式"
        tickers = SNIPER_TICKERS
        send_telegram(f"⏳ **AI交易員已上線**\n📊 當前 VIX: `{vix:.2f}`\n🎯 啟動模式: {screener_name}\n🔍 已鎖定狙擊標的: {', '.join(tickers)}")
        
    else: 
        screener_name, filters = select_screener_by_vix(vix)
        send_telegram(f"⏳ **AI交易員已上線**\n📊 當前 VIX: `{vix:.2f}`\n⚙️ 觸發戰術: **{screener_name}**\n🔍 正在掃描全市場符合條件之標的...")
        tickers = run_screener(filters)

    if not tickers:
        msg = "⚠️ 今日無符合嚴格條件的標的，AI 交易員提早下班。"
        print(msg)
        send_telegram(msg)
        return
        
    send_telegram(f"🎯 **今日出線標的: {', '.join(tickers)}**\n正在呼叫 GPT-OSS-120B 大腦進行深度診斷...")
    
    client = Groq(api_key=GROQ_API_KEY)
    
    for ticker in tickers:
        try:
            print(f"正在分析 {ticker}...")
            # 這裡把策略名稱傳給 AI，讓它知道現在是在做什麼策略
            prompt = load_prompt(ticker, screener_name)
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.2
            )
            analysis = response.choices[0].message.content
            
            passed, reason = risk_check_override(analysis, ticker)
            if not passed:
                send_telegram(f"⚠️ **【{ticker}】 遭紅隊系統否決**\n⛔ 原因: {reason}\n_系統已自動捨棄該標的。_")
                continue
            
            send_telegram(f"🔥 **【{ticker}】 機密備忘錄** 🔥\n\n{analysis[:3800]}")
            log_to_google_sheet(ticker, vix, screener_name, analysis)
            time.sleep(5) 
            
        except Exception as e:
            error_msg = f"❌ 分析 {ticker} 時發生錯誤: {e}"
            print(error_msg)
            send_telegram(error_msg)

if __name__ == "__main__":
    main()
