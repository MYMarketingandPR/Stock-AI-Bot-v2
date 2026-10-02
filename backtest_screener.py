import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# 第一步：設定你想測試嘅股票池 (250 隻頂級熱門股)
# ==========================================
TICKERS = [
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
START_DATE = "2023-01-01"
END_DATE = "2026-09-30"

# 進場/出場/風控規則 
HOLD_DAYS = 20         
STOP_LOSS_PCT = -0.07  
TARGET_PCT = 0.10      
BREAKEVEN_PCT = 0.05   

# ==========================================
# 第二步：下載大盤數據 (SPY) 作為環境濾網
# ==========================================
def get_spy_data():
    try:
        spy = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy["SPY_SMA50"] = spy["Close"].rolling(50).mean()
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
        
    df["SMA20"] = df["Close"].rolling(20).mean()   # 💡 新增 20日均線
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["RelVol"] = df["Volume"] / df["VolAvg20"]
    df["High52w"] = df["Close"].rolling(252).max()
    df["GapUp"] = df["Open"] > df["Close"].shift(1)
    df["PctFrom52wHigh"] = (df["Close"] - df["High52w"]) / df["High52w"]
    df["ChangeFromOpen"] = (df["Close"] - df["Open"]) / df["Open"]
    
    # 計算當日升幅 (防衰竭缺口)
    df["DailyRet"] = df["Close"] / df["Close"].shift(1) - 1
    
    return df

# ==========================================
# 第四步：設定爆量突破策略條件 (終極勝率版)
# ==========================================
def check_breakout_signal(row, spy_is_bullish):
    try:
        # 💡 手術刀 1：計算K線實體，收盤價必須在全日波幅的「上半部」 (>50%)
        daily_range = row["High"] - row["Low"]
        close_strong = row["Close"] >= row["Low"] + (0.5 * daily_range) if daily_range > 0 else True
        
        return (
            spy_is_bullish
            and row["Close"] > 15
            and row["Close"] > row["SMA50"]
            and row["Close"] > row["SMA200"]
            and row["SMA20"] > row["SMA50"]        # 💡 手術刀 2：短期均線多頭排列，確保趨勢向上
            and row["Close"] < row["SMA50"] * 1.20 
            and row["PctFrom52wHigh"] >= -0.05
            and bool(row["GapUp"])
            and row["ChangeFromOpen"] >= 0.02
            and row["RelVol"] >= 2.5 
            and row["DailyRet"] < 0.10             # 💡 手術刀 3：防衰竭缺口，單日升幅不得大於 10%
            and close_strong                       # 💡 確保沒有長上影線的假突破
        )
    except Exception:
        return False

# ==========================================
# 第五步：逐日行走，模擬交易
# ==========================================
def simulate_trades(df, ticker, spy_filter):
    trades = []
    i, n = 0, len(df)
    
    while i < n - HOLD_DAYS - 1:
        date_idx = df.index[i]
        spy_bullish = spy_filter.get(date_idx, True) 
        
        row = df.iloc[i]
        if check_breakout_signal(row, spy_bullish):
            entry_price = df["Open"].iloc[i + 1] 
            entry_date = df.index[i + 1]
            exit_price, exit_date, exit_reason = None, None, "持有到期"
            
            current_stop_price = entry_price * (1 + STOP_LOSS_PCT)
            target_price = entry_price * (1 + TARGET_PCT)
            breakeven_trigger = entry_price * (1 + BREAKEVEN_PCT)
            
            for j in range(i + 1, min(i + 1 + HOLD_DAYS, n)):
                day_open = df["Open"].iloc[j]
                day_high = df["High"].iloc[j]
                day_low = df["Low"].iloc[j]
                
                # 保本機制
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
            i += HOLD_DAYS 
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
    print(f"準備掃描 {len(TICKERS)} 隻美股...")
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
        print("\n完全冇觸發任何交易訊號...")
        pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "exit_date", "exit_price", "return_pct", "exit_reason"]).to_csv("backtest_results.csv", index=False, encoding='utf-8-sig')
        return

    result_df = pd.DataFrame(all_trades)
    
    # 計算各項數據
    total_trades = len(result_df)
    win_trades = result_df[result_df["return_pct"] > 0]
    lose_trades = result_df[result_df["return_pct"] < 0]
    be_trades = result_df[result_df["return_pct"] == 0]
    
    wins = len(win_trades)
    losses = len(lose_trades)
    bes = len(be_trades)
    
    strict_win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    adj_win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
    
    avg_win = win_trades["return_pct"].mean() if wins > 0 else 0
    avg_loss = lose_trades["return_pct"].mean() if losses > 0 else 0

    # 建立總結表格內容
    summary_data = [
        {"ticker": "--- 回測表現總結 ---", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""},
        {"ticker": f"總交易次數：{total_trades} 次", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""},
        {"ticker": f"勝負分佈：贏 {wins} 次、輸 {losses} 次、打和(保本) {bes} 次", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""},
        {"ticker": f"嚴格勝率：{strict_win_rate:.2f}% (若撇除保本，真實勝率約為 {adj_win_rate:.2f}%)", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""},
        {"ticker": f"平均贏單利潤：+{avg_win:.2f}%", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""},
        {"ticker": f"平均輸單虧損：{avg_loss:.2f}%", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""},
        {"ticker": "", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""},
        {"ticker": "--- 詳細交易紀錄 ---", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""}
    ]
    
    summary_df = pd.DataFrame(summary_data)
    final_df = pd.concat([summary_df, result_df], ignore_index=True)
    
    # 輸出 CSV
    final_df.to_csv("backtest_results.csv", index=False, encoding='utf-8-sig')
    print("詳細交易紀錄(附一目了然總結)已存到 backtest_results.csv")

if __name__ == "__main__":
    run_backtest()
