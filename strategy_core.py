"""
strategy_core.py
main.py（live下單）同backtest_screener.py（歷史回測）共用嘅訊號邏輯。
日後要改策略條件，淨係需要改呢個檔案一個地方，兩邊會自動同步，
唔會再出現「live有RSI<85但backtest冇」呢類漂移。
"""
import numpy as np

# ==========================================
# 共用股票池 —— main.py同backtest_screener.py都應該喺呢度import，
# 唔好再各自維護一份，否則好快又會漂移（好似之前RSI<85果次）
# ==========================================
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

HIGH_BETA_TICKERS = [
    # 原有50隻
    "AFRM", "APP", "ARKK", "ARM", "BABA", "BIDU", "COIN", "CRWD", "DASH", "DDOG",
    "DKNG", "EEM", "ENPH", "FSLR", "HOOD", "JD", "KRE", "LCID", "LI", "MDB",
    "MELI", "MNDY", "MSTR", "NET", "NIO", "PDD", "PLTR", "PLUG", "QQQ", "RBLX",
    "RIVN", "ROKU", "RUN", "SE", "SEDG", "SHOP", "SMCI", "SNOW", "SOXL", "SOXS",
    "SOXX", "SPCE", "SPOT", "SQQQ", "TEAM", "TLT", "TOST", "TQQQ", "TTD", "U",
    "UNG",
    # 新增50隻，擴大High Beta組別樣本量
    "UPST", "AI", "IONQ", "CRWV", "RKLB", "OKLO", "SOUN", "BBAI", "RXRX", "TEM",
    "ASTS", "JOBY", "ACHR", "SMR", "VST", "CEG", "GEV", "VRT",
    "IREN", "CLSK", "MARA", "RIOT",
    "CVNA", "CELH", "ONON", "DUOL", "RDDT", "CART", "CAVA", "HIMS", "W", "CHWY",
    "CRSP", "NVAX", "MRNA",
    "ZS", "OKTA", "PATH", "GTLB", "TWLO",
    "PINS", "SNAP", "TTWO", "GME",
    "SOFI", "AXON", "APLD", "LUNR", "ETSY", "NBIS"
]


def compute_indicators(df):
    """喺一個OHLCV dataframe上面加返策略要用嘅所有技術指標欄位。
    df必須有 Open/High/Low/Close/Volume 欄位（index為日期）。"""
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["RelVol"] = df["Volume"] / df["VolAvg20"]
    df["High52w"] = df["Close"].rolling(252).max()
    df["GapUp"] = df["Open"] > df["Close"].shift(1)
    df["PctFrom52wHigh"] = (df["Close"] - df["High52w"]) / df["High52w"]
    df["ChangeFromOpen"] = (df["Close"] - df["Open"]) / df["Open"]
    df["DailyRet"] = df["Close"] / df["Close"].shift(1) - 1

    # RSI(14)，Wilder's smoothing（同大部份平台標準算法一致）
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    # 你而家主力用緊嘅 EMA9 / EMA21 / OBV
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # ⚠️ VWAP：呢度係用daily bar做嘅「近似值」（Typical Price按量加權20日），
    # 唔係你平時睇嘅真正intraday session VWAP。真VWAP要用intraday分鐘/tick數據，
    # 而yfinance嘅分鐘線淨係得返近60日歷史，冇辦法回測2年以上，
    # 所以呢度只做參考用，唔好同broker App度嗰個VWAP直接比較。

    return df


def check_breakout_signal(row, spy_is_bullish):
    """爆量突破策略嘅訊號判斷 —— main.py同backtest_screener.py都call呢個同一個function。"""
    try:
        daily_range = row["High"] - row["Low"]
        close_strong = row["Close"] >= row["Low"] + (0.5 * daily_range) if daily_range > 0 else True

        return (
            spy_is_bullish
            and row["Close"] > 15
            and row["Close"] > row["SMA50"]
            and row["Close"] > row["SMA200"]
            and row["SMA20"] > row["SMA50"]
            and row["Close"] < row["SMA50"] * 1.20
            and row["PctFrom52wHigh"] >= -0.05
            and bool(row["GapUp"])
            and row["ChangeFromOpen"] >= 0.02
            and row["RelVol"] >= 2.5
            and row["DailyRet"] < 0.10
            and row["RSI14"] < 85  # 原本只喺main.py live有，而家兩邊同步咗
            and close_strong
        )
    except Exception:
        return False


def candidate_snapshot_dict(ticker, row):
    """將screening階段已經計算好嘅數值，包裝做一個dict，
    方便直接傳俾AI prompt，唔使之後再重新計算一次（亦唔會再漏嘢）。"""
    return {
        "ticker": ticker,
        "close": float(row["Close"]),
        "sma20": float(row["SMA20"]),
        "sma50": float(row["SMA50"]),
        "sma200": float(row["SMA200"]) if not np.isnan(row["SMA200"]) else None,
        "rel_vol": float(row["RelVol"]),
        "gap_up": bool(row["GapUp"]),
        "change_from_open_pct": float(row["ChangeFromOpen"]) * 100,
        "daily_ret_pct": float(row["DailyRet"]) * 100,
        "pct_from_52w_high": float(row["PctFrom52wHigh"]) * 100,
        "rsi14": float(row["RSI14"]),
        "ema9": float(row["EMA9"]),
        "ema21": float(row["EMA21"]),
        "obv": float(row["OBV"]),
        "vwap_approx": float(row["VWAP_approx"]) if not np.isnan(row["VWAP_approx"]) else None,
    }


def format_quant_snapshot_text(c):
    """將candidate_snapshot_dict()嘅內容格式化做prompt入面用嘅文字區塊。"""
    lines = [
        f"- 現價：${c['close']:.2f}",
        f"- SMA20={c['sma20']:.2f} / SMA50={c['sma50']:.2f}" + (f" / SMA200={c['sma200']:.2f}" if c["sma200"] else ""),
        f"- RSI(14)：{c['rsi14']:.1f}",
        f"- EMA9={c['ema9']:.2f} / EMA21={c['ema21']:.2f}（{'EMA9>EMA21，短期動能偏多' if c['ema9'] > c['ema21'] else 'EMA9<EMA21，短期動能偏弱'}）",
        f"- OBV：{c['obv']:,.0f}",
        f"- VWAP(20日近似值，非即市)：{'${:.2f}'.format(c['vwap_approx']) if c['vwap_approx'] else '資料不足'}",
        f"- 距52週高位：{c['pct_from_52w_high']:.1f}%",
        f"- 相對成交量(對20日均量)：{c['rel_vol']:.2f}x",
        f"- 今日跳空：{'有' if c['gap_up'] else '無'}，開盤至今漲幅：{c['change_from_open_pct']:.2f}%，單日總漲幅：{c['daily_ret_pct']:.2f}%",
        "- 以上數據係系統篩選呢隻股票時已經確認符合突破條件嘅實際根據，唔需要再話「資料不足」。",
    ]
    return "\n".join(lines)
