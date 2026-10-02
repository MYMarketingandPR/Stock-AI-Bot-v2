import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# 第一步：設定你想測試嘅股票池 (57 隻熱門股)
# ==========================================
TICKERS = [
    "MU", "LITE", "VOO", "GOOG", "SPCX", "TSLA", "AVGO", 
    "NKE", "QQQ", "GOOGL", "NVDA", "ORCL", "SKHY", "SNDK", 
    "COHR", "SOXL", "INTC", "ACN", "NBIS", "RKLB", "MRVL", 
    "BE", "TQQQ", "AAOI", "TLT", "TSM", "MSFT", "MUU", 
    "AMD", "AAPL", "META", "NFLX", "AMZN", "MSTR", "SNXX", 
    "CRCL", "MCD", "AXTI", "CBRS", "DELL", "MRNA", "SPY", 
    "APP", "SNPS", "BABA", "GLW", "QQQM", "SOXS", "SQQQ", 
    "IBM", "NOK", "SOXX", "PLTR", "ON", "PDD", "TEM", "LITX"
]
START_DATE = "2023-01-01"
END_DATE = "2026-09-30"

# 進場/出場/風控規則 (已優化)
HOLD_DAYS = 10         # 買入後最多持有交易日
STOP_LOSS_PCT = -0.07  # 初始止損 -7%
TARGET_PCT = 0.15      # 止盈 +15%
BREAKEVEN_PCT = 0.05   # 保本觸發點：利潤達 +5% 時，止損上移至買入價

# ==========================================
# 第二步：下載大盤數據 (SPY) 作為環境濾網
# ==========================================
def get_spy_data():
    try:
        spy = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy["SPY_SMA50"] = spy["Close"].rolling(50).mean()
        # 建立一個 Series，日期對應大盤是否大於 50MA
        return spy["Close"] > spy["SPY_SMA50"]
    except:
        return pd.Series(dtype=bool)

# ==========================================
# 第三步：下載個股歷史數據 + 計算技術指標
# ==========================================
def prepare_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if df.empty or len(df) < 210:
        return None
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    
    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))
    
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["RelVol"] = df["Volume"] / df["VolAvg20"]
    df["High52w"] = df["Close"].rolling(252).max()
    df["GapUp"] = df["Open"] > df["Close"].shift(1)
    df["PctFrom52wHigh"] = (df["Close"] - df["High52w"]) / df["High52w"]
    df["ChangeFromOpen"] = (df["Close"] - df["Open"]) / df["Open"]
    
    return df

# ==========================================
# 第四步：設定爆量突破策略條件 (收緊爆量要求)
# ==========================================
def check_breakout_signal(row, spy_is_bullish):
    try:
        return (
            spy_is_bullish # 大盤濾網：SPY 必須在 50MA 之上
            and row["Close"] > 15
            and row["Close"] > row["SMA50"]
            and row["Close"] > row["SMA200"]
            and row["PctFrom52wHigh"] >= -0.05
            and bool(row["GapUp"])
            and row["ChangeFromOpen"] >= 0.02
            and row["RelVol"] >= 3 # 收緊：成交量必須爆發 3 倍以上
        )
    except Exception:
        return False

# ==========================================
# 第五步：逐日行走，模擬交易 (加入保本機制)
# ==========================================
def simulate_trades(df, ticker, spy_filter):
    trades = []
    i, n = 0, len(df)
    
    while i < n - HOLD_DAYS - 1:
        date_idx = df.index[i]
        # 檢查大盤濾網 (如果當日SPY數據不存在，預設允許)
        spy_bullish = spy_filter.get(date_idx, True) 
        
        row = df.iloc[i]
        if check_breakout_signal(row, spy_bullish):
            entry_price = df["Open"].iloc[i + 1] # 訊號出現後第二日開盤先入場
            entry_date = df.index[i + 1]
            exit_price, exit_date, exit_reason = None, None, "持有到期"
            
            # 動態止損變數
            current_stop_price = entry_price * (1 + STOP_LOSS_PCT)
            target_price = entry_price * (1 + TARGET_PCT)
            breakeven_trigger = entry_price * (1 + BREAKEVEN_PCT)
            
            for j in range(i + 1, min(i + 1 + HOLD_DAYS, n)):
                day_open = df["Open"].iloc[j]
                day_high = df["High"].iloc[j]
                day_low = df["Low"].iloc[j]
                
                # 保本機制：如果盤中最高價觸及 +5%，止損上移至買入價 (打和)
                if day_high >= breakeven_trigger and current_stop_price < entry_price:
                    current_stop_price = entry_price
                
                stop_hit = day_low <= current_stop_price
                target_hit = day_high >= target_price
                
                if stop_hit and target_hit:
                    exit_price = min(day_open, current_stop_price)
                    exit_date, exit_reason = df.index[j], "同日雙穿(保守判定止損)"
                    break
                elif stop_hit:
                    exit_price = min(day_open, current_stop_price)
                    # 判斷是真止損還是保本離場
                    reason_text = "保本離場" if current_stop_price == entry_price else "觸發止損"
                    exit_date, exit_reason = df.index[j], reason_text
                    break
                elif target_hit:
                    exit_price = target_price
                    exit_date, exit_reason = df.index[j], "觸發止盈"
                    break
                    
            if exit_price is None:
                last_j = min(i + HOLD_DAYS, n - 1)
                exit_price, exit_date = df["Close"].iloc[last_j], df.index[last_j]
                
            ret_pct = (exit_price - entry_price) / entry_price
            
            trades.append({
                "ticker": ticker,
                "entry_date": entry_date.date(),
                "entry_price": round(float(entry_price), 2),
                "exit_date": exit_date.date(),
                "exit_price": round(float(exit_price), 2),
                "return_pct": round(float(ret_pct) * 100, 2),
                "exit_reason": exit_reason,
            })
            i += HOLD_DAYS # 避免重疊開注
        else:
            i += 1
            
    return trades

# ==========================================
# 第六步：跑晒所有股票，並將總結寫入 Excel (CSV)
# ==========================================
def run_backtest():
    print("正在下載大盤 SPY 數據建立環境濾網...")
    spy_filter = get_spy_data()
    
    all_trades = []
    for t in TICKERS:
        print(f"處理緊 {t}...")
        df = prepare_data(t)
        if df is None:
            continue
        trades = simulate_trades(df, t, spy_filter)
        if trades:
            print(f" {t} 觸發咗 {len(trades)} 次訊號")
        all_trades.extend(trades)
        
    if not all_trades:
        print("\n完全冇觸發任何交易訊號,可能篩選條件太嚴...")
        pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "exit_date", "exit_price", "return_pct", "exit_reason"]).to_csv("backtest_results.csv", index=False, encoding='utf-8-sig')
        return

    result_df = pd.DataFrame(all_trades)
    win_trades = result_df[result_df["return_pct"] > 0]
    lose_trades = result_df[result_df["return_pct"] < 0] # 修改：剛好 0% 的保本單不當作輸單
    
    win_rate = len(win_trades) / len(result_df) * 100
    avg_win = win_trades["return_pct"].mean() if len(win_trades) else 0
    avg_loss = lose_trades["return_pct"].mean() if len(lose_trades) else 0
    expectancy = (win_rate / 100 * avg_win) + ((len(lose_trades) / len(result_df)) * avg_loss)

    print("\n========== 回測結果 ==========")
    print(f"總交易次數: {len(result_df)}")
    print(f"勝率: {win_rate:.1f}%")
    print(f"平均贏單: +{avg_win:.2f}%")
    print(f"平均輸單: {avg_loss:.2f}%")
    print(f"期望值(每注平均): {expectancy:.2f}%")
    print("==============================")
    
    # 將整體表現總結寫入 CSV 的頂部
    summary_data = [
        {"ticker": f"--- 回測總結 ---", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""},
        {"ticker": f"總交易次數: {len(result_df)}", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""},
        {"ticker": f"勝率: {win_rate:.1f}%", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""},
        {"ticker": f"期望值: {expectancy:.2f}%", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""},
        {"ticker": f"--- 詳細交易紀錄 ---", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""}
    ]
    
    # 合併總結與詳細紀錄
    summary_df = pd.DataFrame(summary_data)
    final_df = pd.concat([summary_df, result_df], ignore_index=True)
    
    # 輸出 CSV 時加入 encoding='utf-8-sig'，確保 Excel 開啟時中文字正常顯示
    final_df.to_csv("backtest_results.csv", index=False, encoding='utf-8-sig')
    print("詳細交易紀錄(附頂部總結)已存到 backtest_results.csv (Excel 亂碼已修復)")

if __name__ == "__main__":
    run_backtest()
