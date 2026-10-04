import os
import re
import time
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from groq import Groq

from strategy_core import (
    compute_indicators,
    check_breakout_signal,
    MEGA_CAP_TICKERS,
)

# ==========================================
# 0. 股票池（100% 集中火力於 72% 勝率的 Mega Cap）
# ==========================================
WATCHLIST = MEGA_CAP_TICKERS

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
    except: pass

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
        if float(stop_loss) >= float(pivot): stop_loss = f"{stop_loss}(⚠️顛倒)"
    except: pass
    data = {"date": datetime.now().strftime("%Y-%m-%d"), "ticker": ticker, "strategy": strategy_name, "vix": f"{vix:.2f}", "ev": ev, "pivot": pivot, "stop_loss": stop_loss}
    try: requests.post(SHEET_WEBHOOK_URL, json=data, timeout=10)
    except: pass

# ==========================================
# 3. 數據獲取與指標計算
# ==========================================
def get_market_context():
    try:
        vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]
        spy = yf.download("SPY", period="6mo", progress=False)
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        spy_bullish = spy["Close"].iloc[-1] > spy["Close"].rolling(50).mean().iloc[-1]
        return float(vix), bool(spy_bullish)
    except: return 20.0, True

def calculate_extra_indicators(df):
    """計算 MACD, KDJ, 神奇九轉 (TD Sequential)"""
    df = df.copy()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    df['TD_Setup'] = 0
    setup_count = 0
    for i in range(4, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-4]:
            setup_count += 1
        else:
            setup_count = 0
        df.iloc[i, df.columns.get_loc('TD_Setup')] = setup_count if setup_count <= 9 else 9
    return df

def scan_breakout_candidates(spy_bullish):
    print(f"正在批次下載 {len(WATCHLIST)} 隻 Mega Cap 股票歷史數據...")
    candidates = []
    try:
        # 要用2y：52週高位(High52w)要252個交易日先計到，1y只有約250日，
        # 會令PctFrom52wHigh永遠係NaN，check_breakout_signal永遠唔觸發。
        batch = yf.download(WATCHLIST, period="2y", group_by="ticker", threads=True, progress=False)
    except: return candidates

    for t in WATCHLIST:
        try:
            if isinstance(batch.columns, pd.MultiIndex):
                if t not in batch.columns.get_level_values(0): continue
                df = batch[t].dropna(how="all")
            else: df = batch
            if df.empty or len(df) < 210: continue

            df = compute_indicators(df)
            df = calculate_extra_indicators(df)
            row = df.iloc[-1]
            
            if check_breakout_signal(row, spy_is_bullish=spy_bullish):
                candidates.append({
                    "ticker": t,
                    "close": row["Close"], "sma20": row["SMA20"], "sma50": row["SMA50"], "sma200": row.get("SMA200", None),
                    "rsi14": row["RSI14"], "ema9": row["EMA9"], "ema21": row["EMA21"], "obv": row["OBV"],
                    "rel_vol": row["RelVol"], "pct_from_52w_high": row["PctFrom52wHigh"]*100, 
                    "gap_up": row["GapUp"], "change_from_open_pct": row["ChangeFromOpen"]*100, "daily_ret_pct": row["DailyRet"]*100,
                    "macd": row["MACD"], "macd_signal": row["Signal"], "macd_hist": row["Hist"],
                    "kdj_k": row["K"], "kdj_d": row["D"], "kdj_j": row["J"], "td_setup": row["TD_Setup"]
                })
        except: continue
    return candidates

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "無法確認")
        eps_growth = info.get("earningsQuarterlyGrowth")
        eps_growth = f"{eps_growth * 100:.1f}%" if isinstance(eps_growth, (int, float)) else "資料不足"
        return sector, eps_growth
    except: return "無法確認（抓取失敗）", "資料不足（抓取失敗）"

def get_recent_news(ticker):
    try:
        news_list = yf.Ticker(ticker).news[:3]
        return "\n".join([f"▪ {n.get('title', '')}" for n in news_list]) if news_list else "近期無重大新聞。"
    except: return "近期無重大新聞。"

def format_quant_snapshot_text(c):
    lines = [
        f"- 現價：${c['close']:.2f}",
        f"- 均線：SMA20={c['sma20']:.2f} / SMA50={c['sma50']:.2f}" + (f" / SMA200={c['sma200']:.2f}" if c.get("sma200") else ""),
        f"- 動能：EMA9={c['ema9']:.2f} / EMA21={c['ema21']:.2f}",
        f"- 震盪指標：RSI(14)={c['rsi14']:.1f} | KDJ(K={c['kdj_k']:.1f}, D={c['kdj_d']:.1f}, J={c['kdj_j']:.1f})",
        f"- 趨勢指標：MACD={c['macd']:.2f}, Signal={c['macd_signal']:.2f}, Hist={c['macd_hist']:.2f}",
        f"- 神奇九轉 (TD Setup)：連續上漲 {c['td_setup']} 天",
        f"- 量價：OBV={c['obv']:,.0f} | 相對成交量={c['rel_vol']:.2f}x",
        f"- 形態：距52週高位={c['pct_from_52w_high']:.1f}% | 今日跳空={'有' if c['gap_up'] else '無'}",
        f"- 日內表現：單日總漲幅={c['daily_ret_pct']:.2f}% | 開盤至今漲幅={c['change_from_open_pct']:.2f}%",
    ]
    return "\n".join(lines)

# ==========================================
# 4. AI 提示詞 (極致資金效率版)
# ==========================================
def load_prompt(candidate, sector, eps_growth, news_text):
    ticker = candidate["ticker"]
    date = datetime.now().strftime("%Y-%m-%d")
    quant_text = format_quant_snapshot_text(candidate)

    return f"""
角色設定：你是一位擁有 30 年華爾街經驗的頂尖量化專家 (CIO)。

【格式鐵律】：
1. 禁止使用Markdown表格語法（帶 |---|），改用清單點(-)同粗體標題(###)。
2. 【誠實原則】：量化數據已經確立，但若基本面或新聞資訊不足以支撐這筆突破，請老實註明「資料不足，風險較高」，絕不可捏造行業背景或未來催化劑。
3. 【模型背景】：此「爆量缺口突破」策略在歷史回測中具備極高正期望值（Mega Cap 大型股勝率高達 72%）。請參考此強大背景，但必須根據下方實際數據進行獨立評估。
4. 【輔助指標應用】：請觀察 MACD、KDJ 和 神奇九轉。若出現「TD Setup 達 8 或 9」或「KDJ 極度超買(>80)」等動能衰竭跡象，請在紅隊壓力測試中嚴厲警告。

分析標的：{ticker}（日期：{date} ｜ 分組：Mega Cap 傳統巨無霸）

【實際技術與量化數據】：
{quant_text}

【基本面與新聞】：
- 所屬行業：{sector}
- EPS 季增長：{eps_growth}
{news_text}

---
### 階段一：終極決策前置
- 【指令】：(強烈買入 / 輕倉試探 / 防守觀望)
- 【主觀勝率】：基於實際數據與行業板塊動能獨立判斷（基準為 72%，若欠缺資訊或出現衰竭訊號，請如實降低評分）
- 【期望值(EV)】：+/- X.XX%
- 【核心邏輯】：精煉解釋決策原因，基於哪一項數據？

### 階段二：族群與基本面催化劑
- 簡述目前營運亮點，是否有足夠動力支撐這波突破？(若資訊不足請直接回答「資料不足」)

### 階段三：量化風控參數
- Entry（買入價）：$XX.XX
- Stop-Loss（硬性止損）：$XX.XX（嚴格定於買入價 -7%）
- Target T1（波段止盈）：$XX.XX（嚴格定於買入價 +10%）
- 資金保護策略：最大持倉 20 日；【時間止損】若持有 10 日未達 +3% 利潤則強制平倉釋放資金；【保本機制】獲利達 +5% 將止損上移至買入價。

### 階段四：紅隊壓力測試
- 致命陷阱：這筆突破最大的潛在風險、超買跡象 (如 神奇九轉或 KDJ 鈍化) 或基本面盲區是什麼？
"""

# ==========================================
# 5. 主程式執行
# ==========================================
def main():
    print("啟動 AI 量化交易員...")
    if not GROQ_API_KEY: return
    vix, spy_bullish = get_market_context()
    if not spy_bullish:
        msg = "⚠️ **大盤濾網啟動**\n標普500跌破50日均線，目前禁止做多，AI 提早下班。"
        send_telegram(msg); return

    send_telegram(f"⏳ **AI交易員已上線**\n📊 VIX: `{vix:.2f}`\n正在批次掃描 {len(WATCHLIST)} 隻 Mega Cap 核心股...")
    candidates = scan_breakout_candidates(spy_bullish)

    if not candidates:
        msg = f"⚠️ 今日 {len(WATCHLIST)} 隻核心股皆無觸發爆量突破，守住資金賺取 IB01 無風險利息，提早下班。"
        send_telegram(msg); return

    names = ", ".join(c["ticker"] for c in candidates)
    send_telegram(f"🎯 **今日精準突破名單: {names}**\n正在呼叫 AI 大腦診斷...")
    client = Groq(api_key=GROQ_API_KEY)

    for c in candidates:
        t = c["ticker"]
        try:
            sector, eps_growth = get_fundamentals(t)
            news_data = get_recent_news(t)
            prompt = load_prompt(c, sector, eps_growth, news_data)

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.2,
            )
            analysis = response.choices[0].message.content
            send_telegram(f"🔥 **【{t}｜Mega Cap】機密備忘錄** 🔥\n\n{analysis[:3800]}")
            log_to_google_sheet(t, "Volume Breakout", vix, analysis)
            time.sleep(5)
        except Exception as e: print(f"❌ 分析 {t} 失敗: {e}")

if __name__ == "__main__":
    main()
