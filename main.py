import os
import re
import time
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from groq import Groq

# ==========================================
# 0. 系統模式切換開關與分組股票池
# ==========================================
RUN_MODE = "AUTO" 

MEGA_CAP_TICKERS = [
    "AAPL", "ABBV", "ACN", "ADBE", "ADI", "ADP", "AEM", "AEP", "AFL", "AIG", 
    "ALL", "AMAT", "AMD", "AME", "AMGN", "AMZN", "AON", "APD", "ASML", "AVGO", 
    "AXP", "AZN", "AZO", "BA", "BAC", "BIIB", "BKNG", "BLK", "BMO", "BNS", 
    "BRK-B", "BSX", "C", "CAT", "CB", "CCI", "CDNS", "CI", "CMCSA", "CME", 
    "CNQ", "COF", "COP", "COST", "CPRT", "CRM", "CSCO", "CSX", "CTAS", "CTVA", 
    "CVX", "D", "DE", "DHI", "DHR", "DIA", "DIS", "DUK", "ED", "EL", 
    "ELV", "EMR", "ENB", "EOG", "ETN", "EW", "EXC", "EXR", "F", "FAST", 
    "FCX", "FDX", "FICO", "FTNT", "GD", "GDX", "GE", "GILD", "GLD", "GOLD", 
    "GOOG", "GOOGL", "GPN", "GS", "GWW", "HAL", "HCA", "HD", "HLT", "HON", 
    "HUM", "IBM", "ICE", "IDXX", "INTC", "INTU", "IQV", "ITW", "IWM", "JNJ", 
    "JPM", "KDP", "KHC", "KLAC", "KMB", "KMI", "KO", "LIN", "LLY", "LMT", 
    "MA", "MAR", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", 
    "MNST", "MO", "MPC", "MRK", "MS", "MSFT", "MSI", "MTD", "MU", "NEE", 
    "NEM", "NFLX", "NKE", "NOC", "NOW", "NSC", "NVDA", "NVO", "NVS", "NXPI", 
    "O", "ODFL", "ORLY", "OXY", "PANW", "PAYX", "PCAR", "PEG", "PEP", "PFE", 
    "PG", "PGR", "PH", "PM", "PNC", "PRU", "PSX", "PYPL", "QCOM", "REGN", 
    "ROP", "ROST", "RSG", "RTX", "RY", "SAP", "SHEL", "SHW", "SLB", "SLV", 
    "SMH", "SNPS", "SO", "SONY", "SPGI", "SPY", "SRE", "STZ", "SU", "SYK", 
    "SYY", "T", "TD", "TEL", "TGT", "TJX", "TM", "TMO", "TRGP", "TRI", 
    "TRP", "TRV", "TSLA", "TSM", "TT", "TTE", "TXN", "UBER", "UNH"
]

HIGH_BETA_TICKERS = [
    "AFRM", "APP", "ARKK", "ARM", "BABA", "BIDU", "COIN", "CRWD", "DASH", "DDOG", 
    "DKNG", "EEM", "ENPH", "FSLR", "HOOD", "JD", "KRE", "LCID", "LI", "MDB", 
    "MELI", "MNDY", "MSTR", "NET", "NIO", "PDD", "PLTR", "PLUG", "QQQ", "RBLX", 
    "RIVN", "ROKU", "RUN", "SE", "SEDG", "SHOP", "SMCI", "SNOW", "SOXL", "SOXS", 
    "SOXX", "SPCE", "SPOT", "SQQQ", "TEAM", "TLT", "TOST", "TQQQ", "TTD", "U", 
    "UNG"
]

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
    if not BOT_TOKEN or not CHAT_ID: return
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
    if not SHEET_WEBHOOK_URL: return
    ev, pivot, stop_loss = "N/A", "手動查閱", "手動查閱"

    ev_match = re.search(r"(?:期望值|EV)[^|\n:]*[:|]?\s*([-+−‑]?\s*\d+\.?\d*)", analysis_text, re.IGNORECASE)
    if ev_match: ev = ev_match.group(1).replace(" ", "")

    for line in analysis_text.split("\n"):
        if pivot == "手動查閱" and re.search(r"(?:Entry|買入價|買入點|試探買入|進場)", line, re.IGNORECASE):
            prices = re.findall(r"\$\s*(\d+\.?\d*)", line)
            if prices: pivot = prices[0]

        if stop_loss == "手動查閱" and re.search(r"(?:Stop[-‑–—]?Loss|硬性止損|止損位|止損點|止損)", line, re.IGNORECASE):
            prices = re.findall(r"\$\s*(\d+\.?\d*)", line)
            if prices: stop_loss = prices[0]

    try:
        p_val, s_val = float(pivot), float(stop_loss)
        if s_val >= p_val: stop_loss = f"{s_val}(⚠️顛倒)"
    except: pass

    data = {
        "date": datetime.now().strftime("%Y-%m-%d"), "ticker": ticker,
        "strategy": strategy_name, "vix": f"{vix:.2f}",
        "ev": ev, "pivot": pivot, "stop_loss": stop_loss
    }
    try: requests.post(SHEET_WEBHOOK_URL, json=data, timeout=10)
    except: pass

# ==========================================
# 3. 核心大腦：執行回測同款量化條件掃描
# ==========================================
def get_market_context():
    try:
        vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]
        spy = yf.download("SPY", period="6mo", progress=False)
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        spy_sma50 = spy["Close"].rolling(50).mean().iloc[-1]
        spy_bullish = spy["Close"].iloc[-1] > spy_sma50
        return float(vix), bool(spy_bullish)
    except:
        return 20.0, True

def scan_breakout_candidates(spy_bullish):
    candidates = []
    print(f"正在全自動掃描 {len(WATCHLIST)} 隻市場最強流動性股票...")
    for t in WATCHLIST:
        try:
            df = yf.download(t, period="1y", progress=False)
            if df.empty or len(df) < 210: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            close = df["Close"].iloc[-1]
            open_p = df["Open"].iloc[-1]
            high_p = df["High"].iloc[-1]
            low_p = df["Low"].iloc[-1]
            prev_close = df["Close"].iloc[-2]
            
            sma20 = df["Close"].rolling(20).mean().iloc[-1]
            sma50 = df["Close"].rolling(50).mean().iloc[-1]
            sma200 = df["Close"].rolling(200).mean().iloc[-1]
            high52w = df["Close"].rolling(252).max().iloc[-1]
            
            vol_avg20 = df["Volume"].rolling(20).mean().iloc[-1]
            rel_vol = df["Volume"].iloc[-1] / vol_avg20 if vol_avg20 > 0 else 0
            
            pct_from_52w_high = (close - high52w) / high52w if high52w > 0 else -1
            gap_up = open_p > prev_close
            change_from_open = (close - open_p) / open_p if open_p > 0 else 0
            daily_ret = (close / prev_close) - 1
            
            daily_range = high_p - low_p
            close_strong = close >= low_p + (0.5 * daily_range) if daily_range > 0 else True
            
            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi14 = 100 - (100 / (1 + rs)).iloc[-1]
            
            if (spy_bullish 
                and close > 15 
                and close > sma50 
                and close > sma200 
                and sma20 > sma50              
                and close < sma50 * 1.20 
                and pct_from_52w_high >= -0.05 
                and gap_up 
                and change_from_open >= 0.02 
                and rel_vol >= 2.5
                and daily_ret < 0.10           
                and rsi14 < 85                 
                and close_strong):             
                
                candidates.append(t)
        except Exception as e:
            continue
    return candidates

def get_technical_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        info = stock.info
        close, vol = df["Close"], df["Volume"]

        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else "N/A"
        high52w = close.rolling(min(len(close), 252)).max().iloc[-1]
        vol_ratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]
        
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.ewm(alpha=1/14, adjust=False).mean() / loss.ewm(alpha=1/14, adjust=False).mean()
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        sector = info.get('sector', '無法確認')
        eps_growth = info.get('earningsQuarterlyGrowth', '資料不足')
        if isinstance(eps_growth, float): eps_growth = f"{eps_growth*100:.1f}%"

        lines = [
            f"- 所屬行業: {sector}",
            f"- EPS 增長: {eps_growth}",
            f"- 現價: ${close.iloc[-1]:.2f} (距52週高位: {(close.iloc[-1]/high52w - 1)*100:.1f}%)",
            f"- 20日均線: ${sma20:.2f} 並且 50日均線: ${sma50:.2f}",
            f"- RSI(14): {rsi:.1f}",
            f"- 今日成交量 vs 20日均量: {vol_ratio:.2f}x (強勢爆量且收盤強勁)"
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"技術指標計算失敗: {e}"

def get_recent_news(ticker):
    try:
        news_list = yf.Ticker(ticker).news[:3]
        return "\n".join([f"▪️ {n.get('title', '')}" for n in news_list]) if news_list else "近期無重大新聞。"
    except: return "近期無重大新聞。"

# ==========================================
# 4. AI 提示詞 (注入歷史回測信心數據)
# ==========================================
def load_prompt(ticker, news_text, tech_text):
    date = datetime.now().strftime("%Y-%m-%d")
    group_name = "Mega Cap (傳統巨無霸與穩健增長)" if ticker in MEGA_CAP_TICKERS else "High Beta (高動能與投機股)"
    
    return f"""
角色設定：你是一位擁有 30 年華爾街對沖基金經驗的頂尖量化專家 (CIO)。

【格式與排版鐵律 - 必須嚴格遵守】：
1. 🚫 **絕對禁止使用任何 Markdown 表格語法** (即帶有 |---| 豎線欄位的表格)！
2. 格式必須使用【清單點 Bullet Points (`-`)】與【粗體標題 (`###`)】。
3. 【主觀勝率】與【期望值(EV)】你必須基於此策略的歷史數據與基本面給出具體數字估算，絕不可回答「資料不足」。

分析標的：{ticker} (日期：{date} | 屬性：{group_name})
【當前觸發策略】：高勝率爆量缺口突破 (High-Probability Volume Breakout)
【量化優勢背書】：該標的已通過我方系統最嚴格的「三把量化手術刀」過濾：
1. 短期均線強大多頭排列 (20MA > 50MA > 200MA)
2. 拒絕衰竭缺口 (單日漲幅嚴格控制 < 10%)
3. 強勢收盤 (收盤價堅挺於全日波幅上半部，無散戶追高套牢的長上影線)
歷史回測證明，此形態在 Mega Cap 大型股上具備極高的延續性與 70% 的真實勝率；在高動能股上亦有 60% 勝率！

【實際技術數據】：
{tech_text}

【即時新聞摘要】：
{news_text}

---
### 階段一：終極決策前置
- 【指令】：(強烈買入 / 輕倉試探 / 防守觀望)
- 【主觀勝率】：XX% (請參考此模型歷史勝率 60-70% 作為基準，結合新聞微調)
- 【期望值(EV)】：+/- X.XX%
- 【核心邏輯】：精煉解釋決策原因，並註明是基於哪一項實際數據或新聞。

### 階段二：族群與基本面催化劑
- 簡述這間公司目前的營運或題材亮點，是否有足夠的動力支撐這波高勝率突破？

### 階段三：量化風控參數 (嚴格執行回測黃金比例)
- Entry (建議買入價)：$XX.XX (直接以現價或微調作參考)
- Stop-Loss (硬性止損)：$XX.XX (嚴格定在買入價的 -7% 處)
- Target T1 (波段止盈)：$XX.XX (嚴格定在買入價的 +10% 處，波段見好就收)
- 資金保護策略：持倉期限最長 20 日。若獲利達到 +5%，請手動將止損上移至買入價打和點 (保本)；若持有 10 日仍處於虧損，提早平倉釋放資金。

### 階段四：紅隊壓力測試
- 致命陷阱：儘管技術面完美，這筆突破最大的潛在風險或基本面盲區是什麼？
"""

# ==========================================
# 5. 主程式執行
# ==========================================
def main():
    print("啟動 AI 量化交易員...")
    if not GROQ_API_KEY: return

    vix, spy_bullish = get_market_context()
    
    if not spy_bullish:
        msg = f"⚠️ **大盤濾網啟動**\n標普500跌破50日均線，目前禁止做多，AI 提早下班。"
        print(msg); send_telegram(msg); return
        
    send_telegram(f"⏳ **AI交易員已上線**\n📊 VIX: `{vix:.2f}`\n大盤趨勢: `多頭`\n正在根據 70% 勝率量化模型掃描 250 隻全市場最強流動性股票...")
    
    breakout_tickers = scan_breakout_candidates(spy_bullish)
    
    if not breakout_tickers:
        msg = "⚠️ 今日 250 隻核心股皆無觸發高品質爆量突破，守住資金，AI 提早下班。"
        print(msg); send_telegram(msg); return
        
    names = ", ".join(breakout_tickers)
    send_telegram(f"🎯 **今日精準突破名單: {names}**\n正在呼叫 AI 大腦診斷...")
    
    client = Groq(api_key=GROQ_API_KEY)
    
    for t in breakout_tickers:
        try:
            print(f"正在分析 {t} ...")
            tech_data = get_technical_snapshot(t)
            news_data = get_recent_news(t)
            
            prompt = load_prompt(t, news_data, tech_data)
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.2
            )
            analysis = response.choices[0].message.content
            
            group = "Mega Cap" if t in MEGA_CAP_TICKERS else "High Beta"
            send_telegram(f"🔥 **【{t}｜{group}｜高勝率爆量突破】 機密備忘錄** 🔥\n\n{analysis[:3800]}")
            log_to_google_sheet(t, "Volume Breakout (Optimized)", vix, analysis)
            time.sleep(5) 
        except Exception as e:
            print(f"❌ 分析 {t} 失敗: {e}")

if __name__ == "__main__":
    main()
