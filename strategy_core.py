"""
strategy_core.py
核心量化引擎：結合 Paul Tudor Jones, Mark Minervini 及 Jesse Livermore 法則的終極版本。
"""
import numpy as np

# 擴充版 MEGA_CAP_TICKERS (近 300 隻核心藍籌與高流動性巨無霸)
MEGA_CAP_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "AEE", "AEM", "AEP", 
    "AFL", "AIG", "ALL", "AMAT", "AMD", "AME", "AMGN", "AMZN", "AON", "APD", 
    "ASML", "AVGO", "AWK", "AXP", "AZN", "AZO", "BA", "BAC", "BBY", "BIIB", 
    "BKNG", "BLK", "BMO", "BMY", "BNS", "BRK-B", "BSX", "BXP", "C", "CAG", 
    "CARR", "CAT", "CB", "CCI", "CDNS", "CDW", "CHD", "CI", "CMCSA", "CME", 
    "CMG", "CNC", "CNQ", "COF", "COP", "COST", "CPRT", "CRM", "CSCO", "CSX", 
    "CTAS", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "D", "DAL", "DE", "DFS", 
    "DHI", "DHR", "DIA", "DIS", "DLR", "DOV", "DTE", "DUK", "EBAY", "ECL", 
    "ED", "EIX", "EL", "ELV", "EMR", "ENB", "EOG", "EQR", "ESS", "ETN", 
    "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FAST", "FCX", "FDX", "FE", 
    "FICO", "FITB", "FTNT", "GD", "GDX", "GE", "GILD", "GIS", "GLD", "GM", 
    "GOLD", "GOOG", "GOOGL", "GPC", "GPN", "GS", "GWW", "HAL", "HCA", "HD", 
    "HIG", "HLT", "HON", "HRL", "HST", "HUM", "IBM", "ICE", "IDXX", "IFF", 
    "INTC", "INTU", "IP", "IQV", "IRM", "ISRG", "ITW", "IWM", "JNJ", "JPM", 
    "K", "KDP", "KEY", "KHC", "KIM", "KLAC", "KMB", "KMI", "KO", "KR", 
    "L", "LIN", "LLY", "LMT", "LOW", "LRCX", "LUV", "LYB", "MA", "MAR", 
    "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", 
    "MMC", "MMM", "MNST", "MO", "MPC", "MRK", "MS", "MSFT", "MSI", "MTB", 
    "MTD", "MU", "NEE", "NEM", "NFLX", "NKE", "NOC", "NOW", "NSC", "NTRS", 
    "NUE", "NVDA", "NVO", "NVS", "NXPI", "O", "ODFL", "OMC", "ORCL", "ORLY", 
    "OTIS", "OXY", "PANW", "PAYX", "PCAR", "PEG", "PEP", "PFE", "PG", "PGR", 
    "PH", "PM", "PNC", "PNR", "PPG", "PPL", "PRU", "PSX", "PYPL", "QCOM", 
    "REGN", "RF", "ROK", "ROP", "ROST", "RSG", "RTX", "RY", "SAP", "SBUX", 
    "SCHW", "SHEL", "SHW", "SLB", "SLV", "SMH", "SNPS", "SO", "SONY", "SPG", 
    "SPGI", "SPY", "SRE", "STT", "STZ", "SU", "SWK", "SYF", "SYK", "SYY", 
    "T", "TD", "TEL", "TGT", "TJX", "TM", "TMO", "TRGP", "TRI", "TRMB", 
    "TROW", "TRP", "TRV", "TSLA", "TSM", "TT", "TTE", "TXN", "TXT", "UBER", 
    "UDR", "UNH", "UNP", "UPS", "USB", "V", "VLO", "VMC", "VTR", "VTRS", 
    "WAB", "WAT", "WBA", "WDC", "WEC", "WELL", "WFC", "WM", "WMT", "WY", 
    "XEL", "XOM", "XYL", "YUM", "ZBH", "ZION"
]

def compute_indicators(df):
    """計算所有核心技術指標、MACD、KDJ 及 神奇九轉"""
    df = df.copy()
    
    # 基礎均線與量價
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["RelVol"] = df["Volume"] / df["VolAvg20"]
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    
    # 形態與乖離
    df["High52w"] = df["Close"].rolling(252).max()
    df["GapUp"] = df["Open"] > df["Close"].shift(1)
    df["PctFrom52wHigh"] = (df["Close"] - df["High52w"]) / df["High52w"]
    df["ChangeFromOpen"] = (df["Close"] - df["Open"]) / df["Open"]
    df["DailyRet"] = df["Close"] / df["Close"].shift(1) - 1

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1 / 14, adjust=False).mean() / loss.ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    # KDJ (9,3,3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # 神奇九轉 (TD Setup)
    df['TD_Setup'] = 0
    setup_up, setup_down = 0, 0
    for i in range(4, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-4]:
            setup_up += 1
            setup_down = 0
            df.iloc[i, df.columns.get_loc('TD_Setup')] = setup_up if setup_up <= 9 else 9
        elif df['Close'].iloc[i] < df['Close'].iloc[i-4]:
            setup_down -= 1
            setup_up = 0
            df.iloc[i, df.columns.get_loc('TD_Setup')] = setup_down if setup_down >= -9 else -9
        else:
            setup_up, setup_down = 0, 0
            
    return df

# ==========================================
# 引擎 A：爆量缺口突破 (Breakout) - 維持 60%+ 王牌
# ==========================================
def check_breakout_signal(row, spy_is_bullish):
    try:
        daily_range = row["High"] - row["Low"]
        close_strong = row["Close"] >= row["Low"] + (0.5 * daily_range) if daily_range > 0 else True
        return (
            spy_is_bullish
            and row["Close"] > 15
            and row["Close"] > row["SMA50"] > row["SMA200"]
            and row["SMA20"] > row["SMA50"]
            and row["Close"] < row["SMA50"] * 1.20
            and row["PctFrom52wHigh"] >= -0.05
            and bool(row["GapUp"])
            and row["ChangeFromOpen"] >= 0.02
            and row["RelVol"] >= 2.5
            and row["DailyRet"] < 0.10
            and row["RSI14"] < 85
            and close_strong
        )
    except Exception: return False

# ==========================================
# 引擎 B：強勢回踩 (Pullback) - 🚀 加入「波幅窒息」與「極致強勢」濾網
# ==========================================
def check_pullback_signal(row, spy_is_bullish, df_history):
    try:
        if not spy_is_bullish: return False
        
        # 1. 嚴格多頭：SMA20 > SMA50 > SMA200
        strict_trend_up = row["SMA20"] > row["SMA50"] > row["SMA200"]
        
        # 2. 🚀 極致相對強勢：距離 52 週新高不能超過 8% (只買最強領導者)
        super_strong = row["PctFrom52wHigh"] >= -0.08
        
        # 3. 🚀 波幅窒息 (Tight Area)：今日最高與最低的波幅，必須小於 2.5% (賣壓完全乾涸)
        daily_volatility = (row["High"] - row["Low"]) / row["Close"]
        volatility_contraction = daily_volatility < 0.025
        
        # 4. 縮量回調
        volume_drying = row["RelVol"] < 0.65
        
        # 5. 精準回踩支撐
        touch_sma20 = (row["Low"] <= row["SMA20"] * 1.01) and (row["Close"] > row["SMA20"])
        touch_sma50 = (row["Low"] <= row["SMA50"] * 1.01) and (row["Close"] > row["SMA50"])
        bounce_support = touch_sma20 or touch_sma50
        
        # 6. K線確認：收陽線或長下影線
        daily_range = row["High"] - row["Low"]
        lower_shadow = row["Close"] - row["Low"] if row["Close"] < row["Open"] else row["Open"] - row["Low"]
        has_lower_shadow = (lower_shadow / daily_range > 0.6) if daily_range > 0 else False
        bullish_candle = (row["Close"] > row["Open"]) or has_lower_shadow
        
        return (strict_trend_up and super_strong and volatility_contraction and 
                volume_drying and bounce_support and bullish_candle)
    except Exception: return False

# ==========================================
# 引擎 C：極度超賣「右側反轉」 (Dip Buy) - 🚀 加入「動能連續翻轉」確認
# ==========================================
def check_dip_buying_signal(row, df_history):
    try:
        # 1. 保命鐵律：必須在 SMA200 之上
        above_200 = row["Close"] > row["SMA200"]
        if not above_200: return False
        
        # 2. 曾經極度恐慌 (過去 5 日內 RSI < 30)
        recent_oversold = df_history["RSI14"].iloc[-5:].min() < 30
        
        # 3. 右側確認：強勢站回 EMA9
        reclaimed_ema9 = (row["Close"] > row["EMA9"]) and (df_history["Close"].iloc[-2] <= df_history["EMA9"].iloc[-2])
        
        # 4. 🚀 動能連續翻轉：MACD 柱狀圖必須「連續兩日」上升 (確保不是死貓反彈)
        macd_improving_2days = (row["Hist"] > df_history["Hist"].iloc[-2]) and \
                               (df_history["Hist"].iloc[-2] > df_history["Hist"].iloc[-3])
        
        # 5. 底部爆量接盤
        volume_support = row["RelVol"] > 1.2
        
        # 6. 當日必須收陽線
        buyers_stepped_in = row["Close"] > row["Open"]

        return (above_200 and recent_oversold and reclaimed_ema9 and 
                macd_improving_2days and volume_support and buyers_stepped_in)
    except Exception: return False
