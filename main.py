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
# 0. 系統模式切換開關與終極股票池
# ==========================================
RUN_MODE = "AUTO" 

# 終極流動性股票池 (標普500 + 納指100 + 頂級熱門股，共 250 隻)
WATCHLIST = [
    "AAPL", "ABBV", "ACN", "ADBE", "ADI", "ADP", "AEM", "AEP", "AFL", "AFRM", 
    "AIG", "ALL", "AMAT", "AMD", "AME", "AMGN", "AMZN", "AON", "APD", "ARKK", 
    "ARM", "ASML", "AVGO", "AXP", "AZN", "AZO", "BA", "BABA", "BAC", "BIDU", 
    "BIIB", "BKNG", "BLK", "BMO", "BNS", "BRK-B", "BSX", "C", "CAT", "CB", 
    "CCI", "CDNS", "CHM", "CI", "CMCSA", "CME", "CNQ", "COF", "COIN", "COP", 
    "COST", "CPRT", "CRM", "CRWD", "CSCO", "CSX", "CTAS", "CTVA", "CVX", "D", 
    "DASH", "DDOG", "DE", "DHI", "DHR", "DIA", "DIS", "DUK", "ED", "EEM", 
    "EL", "ELV", "EMR", "ENB", "ENPH", "EOG", "ETN", "EW", "EXC", "EXR", 
    "F", "FAST", "FCX", "FDX", "FI", "FICO", "FSLR", "FTNT", "GD", "GDX", 
    "GE", "GILD", "GLD", "GOLD", "GOOG", "GOOGL", "GPN", "GS", "GWW", "HAL", 
    "HCA", "HD", "HLT", "HON", "HOOD", "HUM", "IBM", "ICE", "IDXX", "INTC", 
    "INTU", "IQV", "ISG", "ITW", "IWM", "JD", "JNJ", "JPM", "KDP", "KHC", 
    "KLAC", "KMB", "KMI", "KO", "KRE", "LCID", "LCRX", "LI", "LIN", "LLY", 
    "LMT", "MA", "MAR", "MCD", "MCHP", "MCK", "MCO", "MDB", "MDLZ", "MDT", 
    "MELI", "MET", "META", "MMC", "MNST", "MO", "MPC", "MRK", "MS", "MSFT", 
    "MSI", "MSTR", "MTD", "MU", "NEE", "NEM", "NET", "NFLX", "NIO", "NKE", 
    "NOC", "NOW", "NSC", "NVDA", "NVO", "NVS", "NXPI", "O", "ODFL", "ORLY", 
    "OXY", "PANW", "PAYX", "PCAR", "PDD", "PEG", "PEP", "PFE", "PG", "PGR", 
    "PH", "PLTR", "PLUG", "PM", "PNC", "PRU", "PSX", "PYPL", "QCOM", "QQQ", 
    "REGN", "RIVN", "ROKU", "ROP", "ROST", "RSG", "RTX", "RUN", "RY", "SAP", 
    "SE", "SEDG", "SHEL", "SHOP", "SHW", "SLB", "SLV", "SMCI", "SMH", "SNOW", 
    "SNPS", "SO", "SONY", "SOXL", "SOXS", "SOXX", "SPCE", "SPGI", "SPY", "SQ", 
    "SQQQ", "SRE", "STZ", "SU", "SYK", "SYY", "T", "TD", "TEAM", "TEL", 
    "TGT", "TJX", "TLT", "TM", "TMO", "TOST", "TQQQ", "TRGP", "TRI", "TRP", 
    "TRV", "TSLA", "TSM", "TT", "TTE", "TXN", "U", "UBER", "UNG", "UNH"
]

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
            prev_close = df["Close"].iloc[-2]
            
            sma50 = df["Close"].rolling(50).mean().iloc[-1]
            sma200 = df["Close"].rolling(200).mean().iloc[-1]
            high52w = df["Close"].rolling(252).max().iloc[-1]
            
            vol_avg20 = df["Volume"].rolling(20).mean().iloc[-1]
            rel_vol = df["Volume"].iloc[-1] / vol_avg20 if vol_avg20 > 0 else 0
            
            pct_from_52w_high = (close - high52w) / high52w if high52w > 0 else -1
            gap_up = open_p > prev_close
            change_from_open = (close - open_p) / open_p if open_p > 0 else 0
            
            # RSI(14)
            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi14 = 100 - (100 / (1 + rs)).iloc[-1]
            
            # 回測完全一致之邏輯 (RSI < 85 防禦極端泡沫)
            if (spy_bullish 
                and close > 15 
                and close > sma50 
                and close > sma200 
                and close < sma50 * 1.20 
                and pct_from_52w_high >= -0.05 
                and gap_up 
                and change_from_open >= 0.02 
                and rel_vol >= 2.5
                and rsi14 < 85):
                
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

        sma50 = close.rolling(50).mean().iloc[-1]
        sma150 = close.rolling(150).mean().iloc[-1] if len(close) >= 150 else "N/A"
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else "N/A"
        high52w = close.rolling(min(len(close), 252)).max().iloc[-1]
        vol_ratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]

        sector = info.get('sector', '無法確認')
        eps_growth = info.get('earningsQuarterlyGrowth', '資料不足')
        if isinstance(eps_growth, float): eps_growth = f"{eps_growth*100:.1f}%"

        lines = [
            f"- 所屬行業: {sector}",
            f"- EPS 增長: {eps_growth}",
            f"- 現價: ${close.iloc[-1]:.2f} (距52週高位: {(close.iloc[-1]/high52w - 1)*100:.1f}%)",
            f"- 50日均線: ${sma50:.2f}",
            f"- 今日成交量 / 20日均量: {vol_ratio:.2f}x (強勢爆量)"
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
# 4. AI 提示詞 (無表格版 + 精準參數)
# ==========================================
def load_prompt(ticker, news_text, tech_text):
    date = datetime.now().strftime("%Y-%m-%d")
    return f"""
角色設定：你是一位擁有 30 年華爾街對沖基金 CIO 的頂尖專家。

【格式與排版鐵律 - 必須嚴格遵守】：
1. 🚫 **絕對禁止使用任何 Markdown 表格語法** (即帶有 |---| 豎線欄位的表格)！表格在 Telegram 手機介面會完全擠壓變形破版。
2. 格式必須使用【清單點 Bullet Points (`-`)】與【粗體標題 (`###`)】，各區塊之間請用「空行」清晰隔開，讓手機閱讀非常舒適。
3. 【主觀勝率】與【期望值(EV)】你必須基於現有數據進行專業估算，絕不可回答「資料不足」。

分析標的：{ticker} (日期：{date})
【當前觸發策略】：Mega-Cap 爆量缺口突破 (Volume Breakout)
【策略核心邏輯】：大盤處於牛市，巨型股出現跳空高開、盤中強勢拉升且成交量突破 2.5 倍均量，顯示機構資金強烈介入。

【實際技術數據】：
{tech_text}

【即時新聞摘要】：
{news_text}

---
### 階段一：終極決策前置
- 【指令】：(強烈買入 / 輕倉試探 / 防守觀望)
- 【主觀勝率】：XX%
- 【期望值(EV)】：+/- X.XX%
- 【核心邏輯】：精煉解釋決策原因，並註明是基於哪一項實際數據。

### 階段二：族群與基本面背景
- 根據新聞與技術數據，簡述這間公司目前的營運或題材亮點。

### 階段三：技術面與風控參數
- Entry (最佳買入價)：$XX.XX (直接以現價或微調作參考)
- Stop-Loss (硬性止損)：$XX.XX (必須定在買入價以下約 7% 處，作為保護)
- Target T1 (波段止盈)：$XX.XX (設定為買入價以上約 10%，波段見好就收)
- 保本策略指引：當獲利達到 +5% 時，請手動將止損上移至買入價。

### 階段四：紅隊壓力測試
- 致命陷阱：這筆突破最大的假動作風險或基本面盲區是什麼？
"""

# ==========================================
# 5. 主程式執行
# ==========================================
def main():
    print("啟動 AI 交易員...")
    if not GROQ_API_KEY: return

    vix, spy_bullish = get_market_context()
    
    if not spy_bullish:
        msg = f"⚠️ **大盤濾網啟動**\n標普500跌破50日均線，目前禁止做多，AI 提早下班。"
        print(msg); send_telegram(msg); return
        
    send_telegram(f"⏳ **AI交易員已上線**\n📊 VIX: `{vix:.2f}`\n大盤趨勢: `多頭`\n正在掃描 250 隻全市場最強流動性股票...")
    
    breakout_tickers = scan_breakout_candidates(spy_bullish)
    
    if not breakout_tickers:
        msg = "⚠️ 今日 250 隻核心股皆無觸發爆量突破條件，AI 提早下班。"
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
            
            send_telegram(f"🔥 **【{t}｜爆量缺口突破】 機密備忘錄** 🔥\n\n{analysis[:3800]}")
            log_to_google_sheet(t, "Volume Breakout", vix, analysis)
            time.sleep(5) 
        except Exception as e:
            print(f"❌ 分析 {t} 失敗: {e}")

if __name__ == "__main__":
    main()
