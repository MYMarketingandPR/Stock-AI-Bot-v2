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
# 引擎 A：爆量缺口突破 (Breakout) - 維持原本的 60%+ 勝率王牌
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
# 引擎 B：強勢回踩 (Pullback) - Mark Minervini VCP 縮量法則
# ==========================================
def check_pullback_signal(row, spy_is_bullish, df_history):
    try:
        if not spy_is_bullish: return False
        
        # 1. 嚴格多頭排列 (Stage 2)：絕不買入短線已走弱的股票
        strict_trend_up = row["SMA20"] > row["SMA50"] > row["SMA200"]
        if not strict_trend_up: return False
        
        # 2. 健康的 RSI 休整：回踩時 RSI 不應崩盤
        healthy_rsi = 40 <= row["RSI14"] <= 60
        
        # 3. 🚀 Minervini VCP 縮量：成交量極度萎縮，代表賣壓完全乾涸
        volume_drying = row["RelVol"] < 0.65
        
        # 4. 精準回踩 SMA20 或 SMA50
        touch_sma20 = (row["Low"] <= row["SMA20"] * 1.015) and (row["Close"] > row["SMA20"])
        touch_sma50 = (row["Low"] <= row["SMA50"] * 1.015) and (row["Close"] > row["SMA50"])
        bounce_support = touch_sma20 or touch_sma50
        
        # 5. K線確認：必須收陽線或帶極長下影線 (拒絕下跌)
        daily_range = row["High"] - row["Low"]
        lower_shadow = row["Close"] - row["Low"] if row["Close"] < row["Open"] else row["Open"] - row["Low"]
        has_lower_shadow = (lower_shadow / daily_range > 0.6) if daily_range > 0 else False
        bullish_candle = (row["Close"] > row["Open"]) or has_lower_shadow
        
        return strict_trend_up and healthy_rsi and volume_drying and bounce_support and bullish_candle
    except Exception: return False

# ==========================================
# 引擎 C：極度超賣「右側反轉」 (Reversal) - Jesse Livermore + PTJ 法則
# ==========================================
def check_dip_buying_signal(row, df_history):
    try:
        # 1. 🚀 Paul Tudor Jones 鐵律：絕對不能跌穿 200 日線！
        above_200 = row["Close"] > row["SMA200"]
        if not above_200: return False
        
        # 2. 尋找「曾經」的極度恐慌：過去 5 天內，RSI 曾經跌破 30
        recent_oversold = df_history["RSI14"].iloc[-5:].min() < 30
        
        # 3. 🚀 Jesse Livermore 右側確認：今日收盤價必須強勢站回 EMA9 (9日均線) 之上！
        reclaimed_ema9 = (row["Close"] > row["EMA9"]) and (df_history["Close"].iloc[-2] <= df_history["EMA9"].iloc[-2])
        
        # 4. 底部爆量：證明大機構已經進場接盤
        volume_support = row["RelVol"] > 1.2
        
        # 5. 當日必須收陽線 (多頭絕對控制)
        buyers_stepped_in = row["Close"] > row["Open"]

        return above_200 and recent_oversold and reclaimed_ema9 and volume_support and buyers_stepped_in
    except Exception: return False

def candidate_snapshot_dict(ticker, row):
    """將技術指標打包，傳俾 AI"""
    return {
        "ticker": ticker, "close": float(row["Close"]), "sma20": float(row["SMA20"]),
        "sma50": float(row["SMA50"]), "sma200": float(row["SMA200"]) if not np.isnan(row["SMA200"]) else None,
        "rel_vol": float(row["RelVol"]), "gap_up": bool(row["GapUp"]),
        "change_from_open_pct": float(row["ChangeFromOpen"]) * 100, "daily_ret_pct": float(row["DailyRet"]) * 100,
        "pct_from_52w_high": float(row["PctFrom52wHigh"]) * 100, "rsi14": float(row["RSI14"]),
        "ema9": float(row["EMA9"]), "ema21": float(row["EMA21"]), "obv": float(row["OBV"]),
        "macd": float(row["MACD"]), "macd_signal": float(row["Signal"]), "macd_hist": float(row["Hist"]),
        "kdj_k": float(row["K"]), "kdj_d": float(row["D"]), "kdj_j": float(row["J"]), "td_setup": int(row["TD_Setup"])
    }

def format_quant_snapshot_text(c):
    lines = [
        f"- 現價：${c['close']:.2f} (距52週高位：{c['pct_from_52w_high']:.1f}%)",
        f"- 均線結構：SMA20={c['sma20']:.2f} | SMA50={c['sma50']:.2f}" + (f" | SMA200={c['sma200']:.2f}" if c.get("sma200") else ""),
        f"- 短期動能：EMA9={c['ema9']:.2f} | EMA21={c['ema21']:.2f}",
        f"- 震盪指標：RSI(14)={c['rsi14']:.1f} | KDJ(K={c['kdj_k']:.1f}, D={c['kdj_d']:.1f}, J={c['kdj_j']:.1f})",
        f"- 趨勢指標：MACD={c['macd']:.2f}, Hist={c['macd_hist']:.2f}",
        f"- 神奇九轉 (TD Setup)：{c['td_setup']} (正數為上漲結構，負數為下跌結構)",
        f"- 量價結構：相對均量={c['rel_vol']:.2f}x | OBV={c['obv']:,.0f}",
        f"- 日內表現：單日總跌漲={c['daily_ret_pct']:.2f}% | 開盤至今={c['change_from_open_pct']:.2f}% | 跳空={'有' if c['gap_up'] else '無'}",
    ]
    return "\n".join(lines)
