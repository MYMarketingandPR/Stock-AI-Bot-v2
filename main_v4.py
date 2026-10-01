import os
import re
import time
from datetime import datetime, timezone
import requests
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
        
    # EV 抓取：忽略 EV 和數字中間的雜訊，專心找 % 前的數字
    ev_match = re.search(r"EV[^\d+−‑-]*([-+−‑]?\s*\d+\.?\d*)\s*%", analysis_text, re.IGNORECASE)
    
    # 價格抓取：強制鎖定 \$ 符號！徹底過濾掉 % 數、天數或倍數的干擾
    pivot_match = re.search(r"(?:Entry|買入價|買入點|進場).*?\$(\d+\.?\d*)", analysis_text, re.IGNORECASE)
    stop_match = re.search(r"(?:Stop-Loss|止損|Stop).*?\$(\d+\.?\d*)", analysis_text, re.IGNORECASE)

    ev = ev_match.group(1) if ev_match else "N/A"
    pivot = pivot_match.group(1) if pivot_match else "手動查閱"
    stop_loss = stop_match.group(1) if stop_match else "手動查閱"

    data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "ticker": ticker,
        "strategy": strategy_name, 
        "vix": f"{vix:.2f}",
        "ev": ev,
        "pivot": pivot,
        "stop_loss": stop_loss
    }
    try:
        requests.post(SHEET_WEBHOOK_URL, json=data, timeout=10)
    except Exception as e:
        print(f"Google Sheets 網路請求失敗: {e}")

# ==========================================
# 3. 市場數據、新聞、StockTwits 與真實數值擷取
# ==========================================
def get_vix():
    try:
        vix = yf.Ticker("^VIX")
        return float(vix.history(period="1d")["Close"].iloc[-1])
    except:
        return 20.0

def get_recent_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news[:5]
        news_text = ""
        for n in news_list:
            news_text += f"▪️ {n.get('title', '')}: {n.get('summary', '')}\n"
        return news_text if news_text else "近期無重大新聞或 SEC Filings。"
    except:
        return "近期無重大新聞或 SEC Filings。"

def get_squeeze_and_sentiment_data(ticker, strategy_name):
    """結合 StockTwits 情緒與 yfinance 真實價格/短倉數據 (加入極致防呆機制)"""
    # --- 1. 嘗試抓取 StockTwits ---
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        resp = requests.get(url, timeout=8)
        messages = resp.json().get("messages", [])[:30]
        bullish, bearish = 0, 0
        timestamps = []
        for m in messages:
            if m.get("user", {}).get("followers", 0) < 20: continue
            label = m.get("entities", {}).get("sentiment", {}).get("basic")
            if label == "Bullish": bullish += 1
            elif label == "Bearish": bearish += 1
            created_at = m.get("created_at")
            if created_at:
                try: timestamps.append(datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc))
                except: pass
                
        tot = bullish + bearish
        bull_pct = (bullish / tot * 100) if tot > 0 else 50
        time_desc = "無法計算"
        if len(timestamps) >= 2:
            mins = (timestamps[0] - timestamps[-1]).total_seconds() / 60.0
            time_desc = f"最近 30 則留言於 {mins:.1f} 分鐘內湧入" + (" 🔥" if mins < 30 else "")
            
        sentiment_str = f"StockTwits 看多比例: {bull_pct:.0f}% (Bull: {bullish} / Bear: {bearish})\n討論密度: {time_desc}"
    except:
        sentiment_str = "StockTwits 討論熱度較低或無標記資料"

    # --- 2. 嘗試抓取 yfinance 真實數據 ---
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 雙重保險抓取價格：info 抓不到就強行讀取 K 線圖最新收盤價
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            hist = stock.history(period="1d")
            price = hist["Close"].iloc[-1] if not hist.empty else 0
            
        short_pct = info.get("shortPercentOfFloat", 0) * 100
        sma50 = info.get("fiftyDayAverage") or 0
        
        # 核心防呆：如果 API 抓不到短倉，但該股是從短擠壓策略篩出來的，強迫注入 Finviz 保底記憶
        if short_pct == 0 and "短擠壓" in strategy_name:
            short_str = "> 15.00% (Finviz篩選器保底確認)"
        else:
            short_str = f"{short_pct:.2f}%"
            
        # 這裡全部改用連字號，徹底避開 \vert 亂碼！
        metrics_str = f"現價: ${price:.2f} - 短倉佔比: {short_str} - 50MA: ${sma50:.2f}"
        
    except Exception as e:
        # 終極防線：連 yfinance 都徹底當機時
        if "短擠壓" in strategy_name:
            metrics_str = "現價: >$5 (Finviz確認) - 短倉佔比: >15% (Finviz篩選器保底確認)"
        else:
            metrics_str = "現價: >$15 (Finviz確認) - 均線: 50MA > 200MA (Finviz篩選確認)"
            
    return sentiment_str, metrics_str

# ==========================================
# 4. 兩套獨立篩選條件 (雙軌並行)
# ==========================================
def get_short_squeeze_filters():
    return {
        'Float Short': 'Over 15%',
        'Float': 'Under 50M',
        'Average Volume': 'Over 500K',
        'Relative Volume': 'Over 2',
        'Price': 'Over $5' 
    }

def get_minervini_filters():
    return {
        'Market Cap.': '+Small (over $300mln)',
        'Price': 'Over $15',
        'Average Volume': 'Over 500K',
        '50-Day Simple Moving Average': 'Price above SMA50',
        '200-Day Simple Moving Average': 'Price above SMA200',
        '52-Week High/Low': '0-20% below High',
        'EPS growth this year': 'Positive (>0%)',
        'EPS growth qtr over qtr': 'Positive (>0%)',
        'Sales growth qtr over qtr': 'Positive (>0%)'
    }

def run_screener(filters):
    try:
        screener = Technical()
        screener.set_filter(filters_dict=filters)
        df = screener.screener_view()
        if df is None or df.empty: return []
        return df['Ticker'].head(3).tolist()
    except:
        return []

def refine_minervini_candidates(tickers):
    results = []
    for t in tickers:
        try:
            df = yf.download(t, period="18mo", progress=False)
            if df.empty or len(df) < 260: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"]
            sma150 = close.rolling(150).mean()
            sma200 = close.rolling(200).mean()
            
            trend_ok = (close.iloc[-1] > sma150.iloc[-1] and sma150.iloc[-1] > sma200.iloc[-1] and sma200.iloc[-1] > sma200.iloc[-22])
            
            r3, r6 = close.iloc[-1]/close.iloc[-63]-1, close.iloc[-1]/close.iloc[-126]-1
            r9, r12 = close.iloc[-1]/close.iloc[-189]-1, close.iloc[-1]/close.iloc[-252]-1
            rs_score = 2*r3 + r6 + r9 + r12
            
            results.append({"ticker": t, "trend_template_ok": trend_ok, "rs_score": rs_score})
        except: continue
        
    df_res = pd.DataFrame(results)
    if df_res.empty: return df_res
    df_res["rs_rank"] = df_res["rs_score"].rank(pct=True) * 100
    return df_res.sort_values("rs_rank", ascending=False)

# ==========================================
# 5. AI 提示詞 (動態邏輯 + 真實數據輸入)
# ==========================================
STRATEGY_DESC = {
    "短擠壓 (Short Squeeze)": "高空倉比率被逼平倉的軋空邏輯。基本面不重要，專注於空單結構、散戶情緒與成交量爆發。",
    "Minervini (Trend Template)": "SEPA式超級成長股。必須具備強勁基本面(EPS/營收雙位數增長)與完美的均線多頭排列。",
    "人手狙擊 (雙框架評估)": "請同時用短擠壓與 Minervini Trend Template 兩套框架進行雙重壓力測試。"
}

def load_prompt(ticker, strategy_name, news_text, stocktwits_text, metrics_str):
    date = datetime.now().strftime("%Y-%m-%d")
    strategy_logic = STRATEGY_DESC.get(strategy_name, "")
    
    return f"""
角色設定：請化身為華爾街頂尖對沖基金 CIO。
分析標的：{ticker} (日期：{date})

【當前系統觸發的策略】：{strategy_name}
【策略核心邏輯】：{strategy_logic}
請務必根據此邏輯評估，切勿用另一套策略的標準來誤判。

【客觀市場真實數據 (請務必依此精確計算 Entry/Stop-Loss)】：
{metrics_str}

【即時新聞與 SEC Filings】：
{news_text}

【StockTwits散戶情緒與熱度】：
{stocktwits_text}

階段一：終極決策前置
- 【指令】：(強烈買入 / 輕倉試探 / 防守觀望 / 嚴格避開)
- 【主觀勝率】：XX% | 【期望值(EV)】：+/- X.XX%
- 【一句話核心邏輯】：精煉解釋。

階段二：戰術環境與策略診斷
- 若為短擠壓：評估空單比例與散戶情緒催化劑是否足夠。
- 若為Minervini：評估基本面新聞與技術型態(VCP)是否穩固。

階段三：技術面與極致風控 (需基於上方真實報價)
1. 標出最佳 Entry 買入價與硬性止損點位 (Stop-Loss)。
2. 目標價 T1/T2，展示 RRR ≥ 3:1。

階段四：紅隊壓力測試
這筆交易最大的陷阱是什麼？
"""

def risk_check_override(analysis_text, ticker):
    if "止損" not in analysis_text and "Stop-Loss" not in analysis_text:
        return False, "無止損設定"
    return True, "通過風控"

# ==========================================
# 6. 主程式執行
# ==========================================
def main():
    print(f"啟動 AI 交易員... 當前模式: {RUN_MODE}")
    if not GROQ_API_KEY:
        print("❌ 找不到 GROQ_API_KEY"); return

    vix = get_vix()
    candidates = []

    if RUN_MODE == "SNIPER":
        for t in SNIPER_TICKERS: candidates.append((t, "人手狙擊 (雙框架評估)"))
        send_telegram(f"⏳ **AI交易員 V4.1 已上線**\n📊 VIX: `{vix:.2f}`\n🎯 啟動模式: 單點狙擊")
    else: 
        send_telegram(f"⏳ **AI交易員 V4.1 已上線**\n📊 VIX: `{vix:.2f}`\n🔍 正在進行【短擠壓 + Minervini】雙軌掃描...")
        
        # 軌道 1: 短擠壓
        squeeze_tickers = run_screener(get_short_squeeze_filters())
        for t in squeeze_tickers[:3]: candidates.append((t, "短擠壓 (Short Squeeze)"))
            
        # 軌道 2: Minervini
        minervini_raw = run_screener(get_minervini_filters())
        if minervini_raw:
            refined = refine_minervini_candidates(minervini_raw)
            if not refined.empty:
                top = refined[refined["trend_template_ok"]].head(3)
                for t in top["ticker"]: candidates.append((t, "Minervini (Trend Template)"))

    if not candidates:
        msg = "⚠️ 今日雙策略皆無符合嚴格條件的標的，AI 提早下班。"
        print(msg); send_telegram(msg); return
        
    names = ", ".join(f"{t}({s.split(' ')[0]})" for t, s in candidates)
    send_telegram(f"🎯 **今日出線名單: {names}**\n正在呼叫 120B 大腦逐一診斷...")
    
    client = Groq(api_key=GROQ_API_KEY)
    
    for ticker, strategy_name in candidates:
        try:
            print(f"分析 {ticker} ({strategy_name})...")
            news_data = get_recent_news(ticker)
            
            # 注意這裡：將 strategy_name 傳入函數，讓系統知道何時該啟動 Finviz 保底記憶
            stocktwits_text, metrics_str = get_squeeze_and_sentiment_data(ticker, strategy_name)
            
            prompt = load_prompt(ticker, strategy_name, news_data, stocktwits_text, metrics_str)
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.2
            )
            analysis = response.choices[0].message.content
            
            if not risk_check_override(analysis, ticker)[0]:
                send_telegram(f"⚠️ **【{ticker}】 遭風控否決**")
                continue
            
            send_telegram(f"🔥 **【{ticker}】 {strategy_name} 機密備忘錄**\n\n{analysis[:3800]}")
            log_to_google_sheet(ticker, strategy_name, vix, analysis)
            time.sleep(6) 
            
        except Exception as e:
            print(f"❌ {ticker} 錯誤: {e}")

if __name__ == "__main__":
    main()
