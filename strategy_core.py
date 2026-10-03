"""
strategy_core.py
核心量化引擎：結合 Paul Tudor Jones, Mark Minervini 及 Jesse Livermore 法則的終極狙擊手版本。
包含 Mega Cap (巨無霸) 與 High Beta (高動能妖股) 雙重股票池。
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

# 高動能與投機性科技妖股 (101 隻)
HIGH_BETA_TICKERS = [
    "AFRM", "APP", "ARKK", "ARM", "BABA", "BIDU", "COIN", "CRWD", "DASH", "DDOG",
    "DKNG", "EEM", "ENPH", "FSLR", "HOOD", "JD", "KRE", "LCID", "LI", "MDB",
    "MELI", "MNDY", "MSTR", "NET", "NIO", "PDD", "PLTR", "PLUG", "QQQ", "RBLX",
    "RIVN", "ROKU", "RUN", "SE", "SEDG", "SHOP", "SMCI", "SNOW", "SOXL", "SOXS",
    "SOXX", "SPCE", "SPOT", "SQQQ", "TEAM", "TLT", "TOST", "TQQQ", "TTD", "U",
    "UNG", "UPST", "AI", "IONQ", "CRWV", "RKLB", "OKLO", "SOUN", "BBAI", "RXRX", 
    "TEM", "ASTS", "JOBY", "ACHR", "SMR", "VST", "CEG", "GEV", "VRT", "IREN", 
    "CLSK", "MARA", "RIOT", "CVNA", "CELH", "ONON", "DUOL", "RDDT", "CART", "CAVA", 
    "HIMS", "W", "CHWY", "CRSP", "NVAX", "MRNA", "ZS", "OKTA", "PATH", "GTLB", 
    "TWLO", "PINS", "SNAP", "TTWO", "GME", "SOFI", "AXON", "APLD", "LUNR", "ETSY", "NBIS"
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
# 引擎 B：強勢回踩 (Pullback) - 🚀 終極狙擊：只選最強、波幅乾涸、右側啟動
# 🧪 volume_drying_max/volatility_max/pct_from_high_min 開放做參數，方便backtest
#    用train/validation out-of-sample方法去sweep「收緊幾多」至實際有幫助，
#    唔會影響main.py/main_v4.py現有live邏輯(佢哋冇用呢個engine，照用default值)。
# ==========================================
def check_pullback_signal(row, spy_is_bullish, df_history,
                           volume_drying_max=0.65, volatility_max=0.025, pct_from_high_min=-0.08):
    try:
        if not spy_is_bullish or len(df_history) < 3: return False

        # 1. 嚴格多頭：SMA20 > SMA50 > SMA200
        strict_trend_up = row["SMA20"] > row["SMA50"] > row["SMA200"]

        # 2. 🚀 極致相對強勢：距離 52 週新高不能超過 X% (只買市場領導者)
        super_strong = row["PctFrom52wHigh"] >= pct_from_high_min

        # 3. 🚀 波幅窒息 (Tight Area)：今日波幅細過X% (賣壓完全耗盡)
        daily_volatility = (row["High"] - row["Low"]) / row["Close"]
        volatility_contraction = daily_volatility < volatility_max

        # 4. 縮量回調：相對成交量細過X倍
        volume_drying = row["RelVol"] < volume_drying_max
        
        # 5. 精準回踩支撐
        touch_sma20 = (row["Low"] <= row["SMA20"] * 1.01) and (row["Close"] > row["SMA20"])
        touch_sma50 = (row["Low"] <= row["SMA50"] * 1.01) and (row["Close"] > row["SMA50"])
        bounce_support = touch_sma20 or touch_sma50
        
        # 6. K線確認：收陽線或長下影線
        daily_range = row["High"] - row["Low"]
        lower_shadow = row["Close"] - row["Low"] if row["Close"] < row["Open"] else row["Open"] - row["Low"]
        has_lower_shadow = (lower_shadow / daily_range > 0.6) if daily_range > 0 else False
        bullish_candle = (row["Close"] > row["Open"]) or has_lower_shadow
        
        # 7. 🚀 右側啟動確認：今日收盤價必須高於昨日最高價 (點火啟動)
        waking_up = row["Close"] > df_history["High"].iloc[-2]

        return (strict_trend_up and super_strong and volatility_contraction and 
                volume_drying and bounce_support and bullish_candle and waking_up)
    except Exception: return False

# ==========================================
# 引擎 C：極度超賣「右側反轉」 (Dip Buy) - 🚀 雙重動能翻轉確認
# ==========================================
def check_dip_buying_signal(row, df_history):
    try:
        if len(df_history) < 5: return False
        
        # 1. 保命鐵律：必須在 SMA200 之上
        above_200 = row["Close"] > row["SMA200"]
        if not above_200: return False
        
        # 2. 曾經極度恐慌 (過去 5 日內 RSI < 30)
        recent_oversold = df_history["RSI14"].iloc[-5:].min() < 30
        
        # 3. 右側確認：強勢站回 EMA9
        reclaimed_ema9 = (row["Close"] > row["EMA9"]) and (df_history["Close"].iloc[-2] <= df_history["EMA9"].iloc[-2])
        
        # 4. 🚀 動能連續翻轉：MACD 柱狀圖連續兩日上升
        macd_improving_2days = (row["Hist"] > df_history["Hist"].iloc[-2]) and \
                               (df_history["Hist"].iloc[-2] > df_history["Hist"].iloc[-3])
        
        # 5. 底部爆量接盤
        volume_support = row["RelVol"] > 1.2
        
        # 6. 當日必須收陽線
        buyers_stepped_in = row["Close"] > row["Open"]

        return (above_200 and recent_oversold and reclaimed_ema9 and 
                macd_improving_2days and volume_support and buyers_stepped_in)
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

# ==========================================
# 分級倉位建議 (Tiered Position Sizing) —— main.py(live提示)同backtest_screener.py共用，
# 避免兩邊各自維護一份又漂移。
#
# 依據：2023-01~2026-09 train/validation out-of-sample backtest 顯示——
#   - Breakout喺訓練/驗證兩期勝率都係三個引擎入面最高、最穩定(46.5%→43.5%)
#   - Pullback成交量最大，但勝率長期偏弱(~34-36%)
#   - Dip_Buy喺驗證期明顯轉差(34.5%→27.7%)
#   - Mega Cap組別驗證期企得住甚至改善(平均回報1.41%→1.94%)
#   - High Beta組別驗證期崩咗(勝率32.3%→21.2%，平均回報跌到接近0%)
# 所以用「策略 x 組別」嚟分級注碼，純粹係$80,000入面嘅資金分配差異，
# 絕對唔涉及槓桿——所有size都仍然喺總資產100%之內。
# ==========================================
STRATEGY_SIZE_MULTIPLIER = {
    "Breakout (突破)": 1.5,    # 兩期勝率最高、最穩定 → 加碼
    "Dip_Buy (撈底)": 1.0,     # 基準
    "Pullback (回踩)": 0.75,   # 成交量最大，但勝率/期望值相對較弱 → 減碼
}
GROUP_SIZE_MULTIPLIER = {
    "Mega Cap": 1.0,     # 驗證期企得住甚至改善 → 維持
    "High Beta": 0.5,    # 驗證期勝率/回報明顯轉差 → 減碼
}
BASE_POSITION_SIZE_PCT = 0.10  # 基準單位：Dip_Buy + Mega Cap = 1.0 x 1.0 = 10%


# ==========================================
# 策略 x 組別 可用性規則（結構性風控，唔係靠fit出嚫參數）
#
# 依據：2023-01~2026-09 train/validation拆期數據——
#   Dip_Buy+High Beta：訓練期33.33%勝率(3次，樣本太少)，驗證期崩到6.67%勝率(15次，平均回報-2.93%)。
#   經濟解釋：High Beta呢批高波動妖股，超賣之後未必會乖乖「右側確認反轉」，
#   跌落去可以繼續跌，Dip_Buy依賴嚫「抄底右側確認」邏輯對呢批股特別唔work。
#   ⚠️ 坦白講：呢個發現主要係由驗證期數據觀察到(訓練期樣本太少未能提早確認)，
#   嚴格嚫train-then-validate原則嚫角度,呢個唔算100%乾淨嚫出樣本外證據，
#   但佢係一個有經濟邏輯支持嚫結構性風控決定，唔係為追CAGR數字而fit嚫參數。
# main.py現時冇用Dip_Buy/High Beta(只用Breakout+Mega Cap)，所以呢個規則暫時只影響backtest_screener.py。
# ==========================================
DIP_BUY_EXCLUDED_GROUPS = {"High Beta"}


def is_strategy_allowed(strategy, group):
    """check呢個(策略,組別)組合係唔係被結構性規則禁止使用。"""
    if strategy == "Dip_Buy (撈底)" and group in DIP_BUY_EXCLUDED_GROUPS:
        return False
    return True


def get_tiered_position_size_pct(strategy, group, base=BASE_POSITION_SIZE_PCT,
                                  min_pct=0.03, max_pct=0.50):
    """根據策略類型(strategy)同股票組別(group)，計算呢次訊號應該用幾多%資金。
    例：Breakout+Mega Cap = 10% x 1.5 x 1.0 = 15%；Pullback+High Beta = 10% x 0.75 x 0.5 ≈ 3.75%。
    純粹資金分配嘅差異，所有size都喺總資產100%之內，唔涉及任何槓桿。
    main.py(live)同backtest_screener.py應該共用呢個function，先至唔會兩邊數字唔同步。"""
    s_mult = STRATEGY_SIZE_MULTIPLIER.get(strategy, 1.0)
    g_mult = GROUP_SIZE_MULTIPLIER.get(group, 1.0)
    pct = base * s_mult * g_mult
    return max(min_pct, min(max_pct, pct))


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
