import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# 第一步：設定你想測試嘅股票池
# ==========================================
TICKERS = ["CRWV", "MP", "IONQ", "UUUU", "UMAC"]
START_DATE = "2023-01-01"
END_DATE = "2026-09-30"

# 進場/出場規則
HOLD_DAYS = 10         # 買入後最多持有幾多個交易日
STOP_LOSS_PCT = -0.07  # 止損 -7%
TARGET_PCT = 0.15      # 止盈 +15%

# ==========================================
# 第二步：下載歷史數據 + 計算技術指標
# ==========================================
def prepare_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if df.empty or len(df) < 210:
        return None
        
    # yfinance新版有時會回傳多層欄位名，強制拉平
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
# 第三步：設定爆量突破策略條件
# ==========================================
def check_breakout_signal(row):
    try:
        return (
            row["Close"] > 15
            and row["Close"] > row["SMA50"]
            and row["Close"] > row["SMA200"]
            and row["PctFrom52wHigh"] >= -0.05
            and bool(row["GapUp"])
            and row["ChangeFromOpen"] >= 0.02
            and row["RelVol"] >= 2
        )
    except Exception:
        return False

# ==========================================
# 第四步：逐日行走，模擬交易
# ==========================================
def simulate_trades(df, ticker):
    trades = []
    i, n = 0, len(df)
    
    while i < n - HOLD_DAYS - 1:
        row = df.iloc[i]
        if check_breakout_signal(row):
            entry_price = df["Open"].iloc[i + 1] # 訊號出現後第二日開盤先入場
            entry_date = df.index[i + 1]
            exit_price, exit_date, exit_reason = None, None, "持有到期"
            
            stop_price = entry_price * (1 + STOP_LOSS_PCT)
            target_price = entry_price * (1 + TARGET_PCT)
            
            for j in range(i + 1, min(i + 1 + HOLD_DAYS, n)):
                day_open = df["Open"].iloc[j]
                day_high = df["High"].iloc[j]
                day_low = df["Low"].iloc[j]
                
                stop_hit = day_low <= stop_price
                target_hit = day_high >= target_price
                
                if stop_hit and target_hit:
                    # 同一日「天地板」，保守假設止損先觸發
                    exit_price = min(day_open, stop_price)
                    exit_date, exit_reason = df.index[j], "同日雙穿(保守判定止損)"
                    break
                elif stop_hit:
                    exit_price = min(day_open, stop_price)
                    exit_date, exit_reason = df.index[j], "觸發止損"
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
            i += HOLD_DAYS # 呢注做完先睇下一注，避免重疊開太多注
        else:
            i += 1
            
    return trades

# ==========================================
# 第五步：跑晒所有股票，匯總結果
# ==========================================
def run_backtest():
    all_trades = []
    for t in TICKERS:
        print(f"處理緊 {t}...")
        df = prepare_data(t)
        if df is None:
            print(f" {t} 數據唔夠或攞唔到,跳過")
            continue
        trades = simulate_trades(df, t)
        print(f" {t} 觸發咗 {len(trades)} 次訊號")
        all_trades.extend(trades)
        
    # 如果無任何交易，生成空 CSV 以防止 GitHub Action Failed
    if not all_trades:
        print("\n完全冇觸發任何交易訊號,可能篩選條件太嚴...")
        pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "exit_date", "exit_price", "return_pct", "exit_reason"]).to_csv("backtest_results.csv", index=False)
        return

    result_df = pd.DataFrame(all_trades)
    win_trades = result_df[result_df["return_pct"] > 0]
    lose_trades = result_df[result_df["return_pct"] <= 0]
    
    win_rate = len(win_trades) / len(result_df) * 100
    avg_win = win_trades["return_pct"].mean() if len(win_trades) else 0
    avg_loss = lose_trades["return_pct"].mean() if len(lose_trades) else 0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    print("\n========== 回測結果 ==========")
    print(f"總交易次數: {len(result_df)}")
    print(f"勝率: {win_rate:.1f}%")
    print(f"平均贏單: +{avg_win:.2f}%")
    print(f"平均輸單: {avg_loss:.2f}%")
    print(f"期望值(每注平均): {expectancy:.2f}%")
    print("==============================")
    print("⚠️ 注意:交易次數太少(例如少過30-50注)嘅話,呢個勝率統計上唔夠穩定,")
    print("建議擴大 TICKERS 清單(例如你成個34隻watchlist) 先睇整體邊個結論。\n")
    
    result_df.to_csv("backtest_results.csv", index=False)
    print("詳細交易紀錄已存到 backtest_results.csv")

if __name__ == "__main__":
    run_backtest()
