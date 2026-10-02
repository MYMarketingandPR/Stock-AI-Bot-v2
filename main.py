import os
import re
import time
from datetime import datetime, timezone
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
        
    ev = "N/A"
    pivot = "手動查閱"
    stop_loss = "手動查閱"

    # 1. 抓取 EV (期望值)
    ev_match = re.search(r"(?:期望值|EV)[^|\n:]*[:|]?\s*([-+−‑]?\s*\d+\.?\d*)", analysis_text, re.IGNORECASE)
    if ev_match:
        ev = ev_match.group(1).replace(" ", "")

    # 2. 逐行掃描 Entry 與 Stop-Loss
    for line in analysis_text.split("\n"):
        if pivot == "手動查閱" and re.search(r"(?:Entry|買入價|買入點|試探買入|進場)", line, re.IGNORECASE):
            prices = re.findall(r"\$\s*(\d+\.?\d*)", line)
            if prices:
                pivot = prices[0]

        if stop_loss == "手動查閱" and re.search(r"(?:Stop[-‑–—]?Loss|硬性止損|止損位|止損點|止損)", line, re.IGNORECASE):
            prices = re.findall(r"\$\s*(\d+\.?\d*)", line)
            if prices:
                stop_loss = prices[0]

    # 3. 風控守門員：檢查「做多策略」下，止損價是否高於入場價
    try:
        p_val = float(pivot)
        s_val = float(stop_loss)
        if s_val >= p_val:
            stop_loss = f"{s_val}(⚠️顛倒)"
            print(f"⚠️️ 警告：{ticker} 止損價 (${s_val}) 高於入場價 (${p_val})，AI 點位邏輯錯誤！")
    except:
        pass

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
# 3. 市場數據、新聞、StockTwits與技術指標
# ==========================================
def get_vix():
    try:
        vix = yf.Ticker("^VIX")
        data = vix.history(period="1d")
        return float(data["Close"].iloc[-1])
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
    except Exception as e:
        return f"新聞抓取失敗: {e}"

def get_stocktwits_sentiment(ticker):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        messages = data.get("messages", [])[:30]
        bullish, bearish, untagged = 0, 0, 0
        sample_texts = []
        for m in messages:
            sentiment = m.get("entities", {}).get("sentiment")
            label = sentiment.get("basic") if sentiment else None
            followers = m.get("user", {}).get("followers", 0)
            if followers < 20: continue
            if label == "Bullish": bullish += 1
            elif label == "Bearish": bearish += 1
            else: untagged += 1
            if len(sample_texts) < 3:
                sample_texts.append(m.get("body", "")[:100])
                
        total_tagged = bullish + bearish
        if total_tagged == 0:
            return "StockTwits: 討論量太低或全部未標記情緒，無法判斷"
        bull_pct = (bullish / total_tagged) * 100
        return (f"StockTwits情緒: Bullish {bullish} / Bearish {bearish} "
                f"(看多比例 {bull_pct:.0f}%)，未標記 {untagged} 則。\n")
    except Exception as e:
        return f"StockTwits抓取失敗: {e}"

def get_technical_snapshot(ticker):
    """加強版：抓取完整的技術與基本面數據，滿足 Minervini 策略需求"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y") 
        info = stock.info

        if df.empty or len(df) < 60:
            return "技術數據不足(歷史數據少於60日)，無法計算指標，分析時請特別保守。"

        close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = (dif - dea) * 2

        # KDJ
        low_n = low.rolling(9).min()
        high_n = high.rolling(9).max()
        rsv = (close - low_n) / (high_n - low_n) * 100
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3*k - 2*d

        # OBV
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        obv_trend = "上升" if obv.iloc[-1] > obv.iloc[-6] else "下跌"

        # 神奇九轉
        diff4 = close - close.shift(4)
        count, direction = 0, None
        for v in diff4.iloc[-15:]:
            if pd.isna(v): continue
            if v > 0:
                count = count + 1 if direction == "up" else 1
                direction = "up"
            elif v < 0:
                count = count + 1 if direction == "down" else 1
                direction = "down"
            else:
                count, direction = 0, None
        nine_turn_status = f"{'向上' if direction == 'up' else '向下'}第{min(count, 9)}轉" if direction else "無明顯九轉"

        # 1. 基本面與行業資料
        sector = info.get('sector', '無法確認')
        industry = info.get('industry', '無法確認')
        eps_growth = info.get('earningsQuarterlyGrowth', '資料不足')
        sales_growth = info.get('revenueGrowth', '資料不足')
        if isinstance(eps_growth, float): eps_growth = f"{eps_growth*100:.1f}%"
        if isinstance(sales_growth, float): sales_growth = f"{sales_growth*100:.1f}%"

        # 2. 短倉數據
        short_ratio = info.get('shortRatio', '資料不足')
        short_pct = info.get('shortPercentOfFloat', '資料不足')
        if isinstance(short_pct, float): short_pct = f"{short_pct*100:.1f}%"

        # 3. 完整均線系統
        sma50 = close.rolling(50).mean().iloc[-1]
        sma150 = close.rolling(150).mean().iloc[-1] if len(close) >= 150 else "數據不足"
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else "數據不足"
        high52w = close.rolling(min(len(close), 252)).max().iloc[-1]
        
        # 4. 相對強度 RS Proxy
        rs_1y = ((close.iloc[-1] / close.iloc[0]) - 1) * 100 if len(close) >= 200 else "數據不足"
        vol_ratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else 1.0

        lines = [
            f"- 所屬行業: {sector} / {industry}",
            f"- 基本面增長: 季度 EPS 增長={eps_growth}, 季度營收增長={sales_growth}",
            f"- 沽空數據: 短倉佔比(Short Float)={short_pct}, 沽空比率(Short Ratio)={short_ratio}",
            f"- 現價: ${close.iloc[-1]:.2f} (距52週高位: {(close.iloc[-1]/high52w - 1)*100:.1f}%)",
            f"- 50日均線: ${sma50:.2f}",
            f"- 150日均線: ${sma150:.2f}" if isinstance(sma150, float) else "- 150日均線: 數據不足",
            f"- 200日均線: ${sma200:.2f}" if isinstance(sma200, float) else "- 200日均線: 數據不足",
            f"- 個股一年期相對強度(RS Proxy): {rs_1y:.1f}%" if isinstance(rs_1y, float) else "- 相對強度: 數據不足",
            f"- MACD: DIF={dif.iloc[-1]:.3f}, DEA={dea.iloc[-1]:.3f}, Histogram={macd_hist.iloc[-1]:.3f}",
            f"- KDJ: K={k.iloc[-1]:.1f}, D={d.iloc[-1]:.1f}, J={j.iloc[-1]:.1f}",
            f"- OBV近5日走勢: {obv_trend}",
            f"- 神奇九轉: {nine_turn_status}",
            f"- 今日成交量 / 20日均量: {vol_ratio:.2f}x"
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"技術指標計算失敗: {e} (此情況下AI分析不應視為有數據支持)"

# ==========================================
# 4. 兩套獨立篩選條件
# ==========================================
def get_short_squeeze_filters():
    return {
        "Float Short": "Over 15%",
        "Float": "Under 50M",
        "Average Volume": "Over 500K",
        "Relative Volume": "Over 2",
        "Price": "Over $3",
        "52-Week High/Low": "0-10% below High",
        "InstitutionalOwnership": "Under 50%"
    }

def get_minervini_filters():
    return {
        "Market Cap.": "+Small (over $300mln)",
        "Price": "Over $15",
        "Average Volume": "Over 500K",
        "50-Day Simple Moving Average": "Price above SMA50",
        "200-Day Simple Moving Average": "Price above SMA200",
        "52-Week High/Low": "0-10% below High",
        "EPS growththis year": "Positive (>0%)",
        "EPS growthqtr over qtr": "Positive (>0%)",
        "Sales growthqtr over qtr": "Positive (>0%)",
        "InstitutionalOwnership": "Over 10%"
    }

def run_screener(filters):
    try:
        screener = Technical()
        screener.set_filter(filters_dict=filters)
        df = screener.screener_view()
        if df is None or df.empty: return []
        return df["Ticker"].head(5).tolist()
    except Exception as e:
        print(f"Finviz 篩選失敗: {e}")
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
        except Exception as e:
            continue
            
    df_result = pd.DataFrame(results)
    if df_result.empty: return df_result
    df_result["rs_rank_in_pool"] = df_result["rs_score"].rank(pct=True) * 100
    return df_result.sort_values("rs_rank_in_pool", ascending=False)

# ==========================================
# 5. AI 提示詞 (Telegram 專用清晰清單排版)
# ==========================================
STRATEGY_DESC = {
    "短擠壓Setup": "高空倉比率被逼平倉的軋空邏輯，不靠基本面，靠空倉結構與成交量爆發判斷。",
    "Minervini Trend Template": "SEPA式強勢股邏輯：基本面(EPS/銷售增長)配合技術面趨勢排列。",
    "人手指定(雙框架評估)": "請同時用短擠壓與Minervini Trend Template兩套框架分別評估這隻股票。"
}

def load_prompt(ticker, strategy_name, news_text, stocktwits_text, tech_text):
    date = datetime.now().strftime("%Y-%m-%d")
    strategy_logic = STRATEGY_DESC.get(strategy_name, "")
    
    return f"""
角色設定：
請你化身一位擁有 30 年華爾街對沖基金 CIO 的頂尖專家。

【格式與排版鐵律 - 必須嚴格遵守】：
1. 🚫 **絕對禁止使用任何 Markdown 表格語法** (即帶有 |---| 豎線欄位的表格)！表格在 Telegram 手機介面會完全擠壓變形破版。
2. 格式必須使用【清單點 Bullet Points (`-`)】與【粗體標題 (`###`)】，各區塊之間請用「空行」清晰隔開，讓手機閱讀非常舒適。
3. 客觀數據不准捏造，但【主觀勝率】與【期望值(EV)】你必須基於現有數據進行專業估算，絕不可回答「資料不足」。

核心目標：為我剖析股票：{ticker} (日期：{date})

【當前系統觸發策略】：{strategy_name}
【策略核心邏輯】：{strategy_logic}

【實際技術數據】：
{tech_text}

【即時新聞摘要】：
{news_text}

【StockTwits散戶情緒】：
{stocktwits_text}

---

### 階段一：終極決策前置
- 【指令】：(強烈買入 / 輕倉試探 / 防守觀望 / 嚴格避開)
- 【主觀勝率】：XX%
- 【期望值(EV)】：+/- X.XX%
- 【核心邏輯】：精煉解釋決策原因，並註明是基於哪一項實際數據。

### 階段二：族群與基本面背景
- 請根據【實際技術數據】與【即時新聞】總結公司所屬行業與營運背景。

### 階段三：技術面與極致風控 (條列式呈現)
- 技術型態診斷：引用實際現價、均線與指標逐一診斷。
- Entry (最佳買入價)：$XX.XX (簡述計算依據)
- Stop-Loss (硬性止損)：$XX.XX (注意：做多交易止損價必須低於買入價)
- Target T1：$XX.XX
- Target T2：$XX.XX
- 風險報酬比 (RRR)：展示 RRR ≥ 3:1 之計算。

### 階段四：紅隊壓力測試
- 致命陷阱：這筆交易最大的陷阱是什麼？
- 虧損主因：若虧損最可能的原因是什麼？
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
        print("❌ 系統錯誤:找不到 GROQ_API_KEY")
        return

    vix = get_vix()
    candidates = []

    if RUN_MODE == "SNIPER":
        send_telegram(f"⏳ **AI交易員已上線**\n📊 VIX: `{vix:.2f}`\n🎯 單點狙擊模式")
        for t in SNIPER_TICKERS: candidates.append((t, "人手指定(雙框架評估)"))
    else: 
        send_telegram(f"⏳ **AI交易員已上線**\n📊 VIX: `{vix:.2f}`\n🔍 同時掃描短擠壓 + Minervini 兩套候選...")
        
        squeeze_tickers = run_screener(get_short_squeeze_filters())
        for t in squeeze_tickers[:3]: candidates.append((t, "短擠壓Setup"))
            
        minervini_raw = run_screener(get_minervini_filters())
        if minervini_raw:
            refined = refine_minervini_candidates(minervini_raw)
            if not refined.empty:
                top = refined[refined["trend_template_ok"]].head(3)
                for t in top["ticker"]: candidates.append((t, "Minervini Trend Template"))

    if not candidates:
        msg = "⚠️ 今日兩套策略都無符合條件的標的，AI 交易員提早下班。"
        print(msg)
        send_telegram(msg)
        return
        
    names = ", ".join(f"{t} ({s})" for t, s in candidates)
    send_telegram(f"🎯 **今日候選: {names}**\n正在呼叫 AI 大腦逐一診斷...")
    
    client = Groq(api_key=GROQ_API_KEY)
    
    for ticker, strategy_name in candidates:
        try:
            print(f"正在分析 {ticker} ({strategy_name}) ...")
            news_data = get_recent_news(ticker)
            stocktwits_data = get_stocktwits_sentiment(ticker)
            tech_data = get_technical_snapshot(ticker)
            
            prompt = load_prompt(ticker, strategy_name, news_data, stocktwits_data, tech_data)
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.2
            )
            analysis = response.choices[0].message.content
            
            passed, reason = risk_check_override(analysis, ticker)
            if not passed:
                send_telegram(f"⚠️️ **【{ticker}】遭紅隊否決**\n原因: {reason}")
                continue
            
            send_telegram(f"🔥 **【{ticker}｜{strategy_name}】 機密備忘錄** 🔥\n\n{analysis[:3800]}")
            log_to_google_sheet(ticker, strategy_name, vix, analysis)
            time.sleep(5) 
            
        except Exception as e:
            error_msg = f"❌ 分析 {ticker} 時發生錯誤: {e}"
            print(error_msg)
            send_telegram(error_msg)

if __name__ == "__main__":
    main()
