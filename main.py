import os
import re
import time
from datetime import datetime
import requests
from finvizfinance.screener.technical import Technical
from google import genai
from google.genai import types

# ==========================================
# 1. 鑰匙設定 (新增了 Google Sheet Webhook)
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
SHEET_WEBHOOK_URL = os.environ.get("SHEET_WEBHOOK_URL")

# ==========================================
# 2. Telegram 與 Google Sheets 推送
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
    """將數據推送到 Google Sheets Webhook"""
    if not SHEET_WEBHOOK_URL:
        print("未設定 Google Sheets Webhook，跳過記錄。")
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
        response = requests.post(SHEET_WEBHOOK_URL, json=data, timeout=10)
        if response.status_code == 200:
            print(f"📝 {ticker} 已成功記錄至 Google Sheets")
        else:
            print(f"寫入 Google Sheets 失敗: {response.text}")
    except Exception as e:
        print(f"Google Sheets 網路請求失敗: {e}")

# ==========================================
# 3. 獲取 VIX 與 篩選器設定
# ==========================================
def get_vix():
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        data = vix.history(period="1d")
        return float(data['Close'].iloc[-1])
    except:
        return 20.0

def get_vcp_filters():
    return {
        'Market Cap.': '+Small (over $300mln)',
        'Price': 'Over $10',
        'Average Volume': 'Over 500K',
        '20-Day Simple Moving Average': 'Price above SMA20',
        '50-Day Simple Moving Average': 'Price above SMA50',
        '200-Day Simple Moving Average': 'Price above SMA200',
        '50-Day/200-Day SMA': 'SMA50 above SMA200',
        '52-Week High/Low': '0-10% below High',
        '52-Week Low': '30% or more above Low',
        'Relative Volume': 'Under 1',
        'Candlestick': 'Narrow Range',
        'Performance (Quarter)': 'Over +20%'
    }

def get_breakout_filters():
    return {
        'Market Cap.': '+Small (over $300mln)',
        'Price': 'Over $15',
        'Average Volume': 'Over 500K',
        '50-Day Simple Moving Average': 'Price above SMA50',
        '200-Day Simple Moving Average': 'Price above SMA200',
        '52-Week High/Low': '0-5% below High',
        'Gap': 'Over 0%',
        'Change from Open': 'Up 2%+',
        'Relative Volume': 'Over 2.5',
        'Performance (Week)': 'Over +5%'
    }

def get_short_squeeze_filters():
    return {
        'Market Cap.': '+Small (over $300mln)',
        'Price': 'Over $10',
        'Average Volume': 'Over 500K',
        'Short Float': 'Over 15%',
        'Beta': 'Over 1.5',
        '200-Day Simple Moving Average': 'Price above SMA200',
        'RSI (14)': 'Under 40',
        'Relative Volume': 'Over 1.5',
        'Change': 'Up 2%+'
    }

def select_screener_by_vix(vix):
    if vix < 18:
        return '🔥 右側爆量突破', get_breakout_filters()
    elif vix <= 25:
        return '📊 VCP蓄力準備', get_vcp_filters()
    else:
        return '🚀 高位逼空反彈', get_short_squeeze_filters()

def run_screener(filters):
    f = Technical()
    try:
        f.set_filter(filters_dict=filters)
        df = f.screener_view()
        if df.empty:
            return []
        return df['Ticker'].tolist()[:5] 
    except Exception as e:
        print(f"Finviz 篩選失敗: {e}")
        return []

# ==========================================
# 4. AI 提示詞與風控
# ==========================================
def load_prompt(ticker):
    date = datetime.now().strftime("%Y-%m-%d")
    return f"""
角色設定：
請你化身一位擁有 30 年華爾街對沖基金 CIO 與 Mark Minervini 風格動量交易大師的頂尖專家。你兼具機構籌碼解讀、期權流分析與深度基本面催化劑拆解能力。你的座右銘是：紀律勝過預測，風險控制先於利潤。

核心目標：為我剖析股票：{ticker}。

數據真實性與來源協議：
1. 強制開頭標註：「數據截止日期：{date}」和「主要數據來源：模型實時估算/歷史推演」。
2. 若缺乏精確實時數據，必須明確標註「(基於估算)」，嚴禁編造虛假數據。

階段一：終極決策前置
- 【指令】：(強烈買入-Pivot突破 / 輕倉試探-VCP建倉 / 防守觀望-等待財報 / 嚴格避開-破位風險)
- 【主觀勝率】：XX% (突破交易常規上限設為 65%)
- 【期望值(EV)】：+/- X.XX% (EV = 勝率% × 平均盈利% - 敗率% × 平均虧損%)
- 【一句話核心邏輯】：精煉解釋決策原因。

階段二：戰術環境與同業族群掃描
1. 大盤環境與策略匹配：診斷最適策略模式。若 VIX > 25，所有突破試探倉強制縮減 50%。
2. 財報 Expected Move 精算：估算財報日跳空幅度。若超出止損距離 60%，觸發財報避險警報。

階段三：Minervini 趨勢模板與技術「敲門磚」
1. Stage 2 趨勢過濾：檢驗均線多頭排列、距52週高低點位置。
2. 形態診斷：標出 VCP/高緊旗形 的 VDU 乾涸點與 Pivot Entry。若不符合，強制判定「形態破產」。

階段四：籌碼與資金流向
1. 評估機構持股趨勢、空頭結構。

階段五：極致風控與執行藍圖
1. 分數化凱利倉位：計算 Half-Kelly (Kelly % × 0.5) 作為單筆倉位上限。
2. 精確買賣與止損點位：
   - 試探倉位與量能確認條件。
   - 硬性止損 (單筆虧損控制 3%-6%)。
   - 目標價 T1/T2，展示 RRR ≥ 3:1。
3. 預設條件單指令：明確寫出「入場時同時掛好 GTC 止損單與 T1 止盈單」。

階段六：紅隊壓力測試與 Checklist
1. 交易邏輯作廢清單：列出 3 條無條件清倉的黑天鵝觸發條件。
2. 盤前 5 分鐘 Checklist：
   - 開盤前 15 分鐘價格在試探倉區間內？
   - 開盤後 30 分鐘爆量？
   - 無隔夜黑天鵝？

階段七：AI 自我質疑
1. 整體分析置信度評級：高 / 中 / 低。
2. AI 自我質疑：這筆交易最大的陷阱是什麼？
"""

def risk_check_override(analysis_text, ticker):
    if "強烈買入" not in analysis_text and "輕倉試探" not in analysis_text:
        return False, "非買入指令"
    if "止損" not in analysis_text and "Stop-Loss" not in analysis_text:
        return False, "無止損設定"
    ev_match = re.search(r"期望值.*?EV.*?([+-]?\d+\.?\d*)\s*%", analysis_text, re.IGNORECASE)
    if ev_match:
        ev = float(ev_match.group(1))
        if ev <= 0:
            return False, f"EV = {ev}% ≤ 0"
    return True, "通過風控"

# ==========================================
# 5. 主程式執行
# ==========================================
def main():
    print("啟動 AI 交易員...")
    vix = get_vix()
    screener_name, filters = select_screener_by_vix(vix)
    
    send_telegram(f"⏳ **AI交易員已上線**\n📊 當前 VIX: `{vix:.2f}`\n🎯 自動啟用策略: {screener_name}")
    
    tickers = ["SNDK", "SKHY"]
    if not tickers:
        send_telegram("📊 **今日掃描結果**\n\n今日無符合嚴格條件的標的。紀律勝過預測，建議空倉觀望。")
        return
    
    send_telegram(f"🎯 **今日精選: {', '.join(tickers)}**\n正在呼叫大腦進行 360 度深度診斷...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    for ticker in tickers:
        try:
            print(f"正在分析 {ticker}...")
            prompt = load_prompt(ticker)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            analysis = response.text
            
            passed, reason = risk_check_override(analysis, ticker)
            if not passed:
                send_telegram(f"⚠️ **【{ticker}】 遭紅隊系統否決**\n⛔ 原因: {reason}\n_系統已自動捨棄該標的。_")
                continue
            
            # 1. 發送 Telegram
            send_telegram(f"🔥 **【{ticker}】 機密備忘錄** 🔥\n\n{analysis[:3800]}")
            
            # 2. 自動寫入 Google Sheets
            log_to_google_sheet(ticker, vix, screener_name, analysis)
            
            time.sleep(8) 
            
        except Exception as e:
            error_msg = f"❌ 分析 {ticker} 時發生錯誤: {e}"
            print(error_msg)
            send_telegram(error_msg)

if __name__ == "__main__":
    main()
