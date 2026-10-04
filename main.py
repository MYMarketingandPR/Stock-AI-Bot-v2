"""
main.py — SPY每日訊號機械人（建議方案）

規則（同 spy_bottoms_v2.py 嘅「E4」完全一樣，研究結果見嗰個檔案）：
1. 基本規則：SPY收市低過200日線2% → 賣出；收市高過200日線2% → 買返。
2. 提早買入：手上係現金嘅時候，如果同一日出現「三合一」：
   RSI(14) < 30 + VIX ≥ 30 + 成交量 ≥ 50日平均2倍 → 全倉買入。
3. 提早買入之後：收市比買入價跌5% → 止損賣出；升到高過200日線2% → 變返正常持有。

回測（1994-2026，每次買賣0.1%成本）：每年 +11.5%，最大跌幅 -27%（SPY長揸 +10.9%，-55%）。
⚠️ 過去表現唔代表將來；呢個只係按規則提你，唔係投資建議。

點運作：每日美股收市後由GitHub Actions執行。程式用SPY由1993年至今嘅數據，由頭按規則行一次，
得出「今日應該持有定現金」，同昨日比較，有轉變就提你下一個交易日收市前買入/賣出。
（回測都係用「今日收市判斷、下一日收市交易」，所以咁做同回測一致。）

用法：python main.py               （正常，會發Telegram）
      python main.py --date 2020-03-12 --dry   （睇返某一日會出咩訊息，唔發Telegram）
"""
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

BAND = 0.02          # 200日線上下緩衝
RSI_MAX = 30
VIX_MIN = 30
VOL_MULT = 2.0
STOP = 0.05          # 提早買入後嘅止損

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
SHEET_WEBHOOK_URL = os.environ.get("SHEET_WEBHOOK_URL")


# ---------------- 數據 ----------------
def load_data():
    import yfinance as yf

    def dl(t, adjust):
        df = yf.download(t, start="1993-01-01", auto_adjust=adjust, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    adj = dl("SPY", True)
    raw = dl("SPY", False)
    vix = dl("^VIX", True)["Close"]
    df = pd.DataFrame({
        "adj": adj["Close"].astype(float),           # 已計息價格：用嚟計訊號（同回測一樣）
        "raw": raw["Close"].astype(float),           # 實際報價：用嚟話你知價位
        "volume": adj["Volume"].astype(float),
    }).dropna(subset=["adj"])
    df["vix"] = vix.astype(float).reindex(df.index).ffill()
    return df


def add_signals(df):
    c = df["adj"]
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn)
    df["ma200"] = c.rolling(200).mean()
    df["vol_mult"] = df["volume"] / df["volume"].rolling(50).mean()
    df["three"] = (df["rsi"] < RSI_MAX) & (df["vix"] >= VIX_MIN) & (df["vol_mult"] >= VOL_MULT)
    return df


# ---------------- 規則（由頭行一次）----------------
def replay(df):
    """回傳每日收市後嘅狀態：out=現金、in=正常持有、early=提早買入中；同提早買入價。"""
    state, entry_px = "out", np.nan
    states, entries = [], []
    for p, m, three in zip(df["adj"].values, df["ma200"].values, df["three"].values):
        if np.isnan(m):
            states.append("out")
            entries.append(np.nan)
            continue
        if state == "in":
            if p < m * (1 - BAND):
                state = "out"
        elif state == "out":
            if p > m * (1 + BAND):
                state = "in"
            elif bool(three):
                state, entry_px = "early", p
        elif state == "early":
            if p > m * (1 + BAND):
                state = "in"
            elif p < entry_px * (1 - STOP):
                state = "out"
        states.append(state)
        entries.append(entry_px if state == "early" else np.nan)
    df["state"] = states
    df["entry_adj"] = entries
    return df


# ---------------- 訊息 ----------------
NAMES = {"out": "現金（唔持有SPY）", "in": "持有SPY", "early": "持有SPY（恐慌時提早買入）"}


def build_message(df, i):
    row, prev = df.iloc[i], df.iloc[i - 1]
    date = df.index[i].date()
    k = row["raw"] / row["adj"]                    # 將已計息價格換返做實際報價
    sell_line = row["ma200"] * (1 - BAND) * k
    buy_line = row["ma200"] * (1 + BAND) * k
    ma_raw = row["ma200"] * k
    s0, s1 = prev["state"], row["state"]
    holding0, holding1 = s0 != "out", s1 != "out"

    if not holding0 and holding1:
        why = ("三合一恐慌信號出現（RSI、VIX、成交量全部達標）" if s1 == "early"
               else "SPY升返高過200日線2%")
        action = f"🟢 *買入*：下一個交易日收市前，用全部資金買入SPY\n原因：{why}"
    elif holding0 and not holding1:
        why = ("提早買入後跌咗5%，觸發止損" if s0 == "early"
               else "SPY跌穿200日線2%")
        action = f"🔴 *賣出*：下一個交易日收市前，賣晒SPY轉做現金\n原因：{why}"
    elif holding1:
        action = "⚪ *繼續持有*，唔使做嘢"
    else:
        action = "⚪ *繼續揸現金*，唔使做嘢"
    if s0 == "early" and s1 == "in":
        action += "\n（提早買入已經成功，由今日起變返正常持有）"

    lines = [
        f"📊 *SPY每日訊號* {date}",
        "",
        action,
        "",
        f"目前應該：{NAMES[s1]}",
        f"SPY收市：${row['raw']:.2f}",
        f"200日線：${ma_raw:.2f}（SPY {'高' if row['adj'] > row['ma200'] else '低'}過 {abs(row['adj'] / row['ma200'] - 1) * 100:.1f}%）",
    ]
    if s1 == "in":
        lines.append(f"賣出線：收市低過 ${sell_line:.2f} 就賣")
    elif s1 == "early":
        stop_raw = row["entry_adj"] * (1 - STOP) * k
        lines.append(f"止損線：收市低過 ${stop_raw:.2f} 就賣（買入價跌5%）")
        lines.append(f"變返正常持有：收市高過 ${buy_line:.2f}")
    else:
        lines.append(f"買入線：收市高過 ${buy_line:.2f} 就買返")
        three_txt = (f"RSI {row['rsi']:.0f}（要<30）{'✅' if row['rsi'] < RSI_MAX else '❌'} | "
                     f"VIX {row['vix']:.1f}（要≥30）{'✅' if row['vix'] >= VIX_MIN else '❌'} | "
                     f"成交量 {row['vol_mult']:.1f}倍（要≥2）{'✅' if row['vol_mult'] >= VOL_MULT else '❌'}")
        lines.append(f"三合一提早買入：{three_txt}")
    if s1 != "out":
        lines.append(f"參考：RSI {row['rsi']:.0f} | VIX {row['vix']:.1f} | 成交量 {row['vol_mult']:.1f}倍")

    chg = df["state"].ne(df["state"].shift()) & ((df["state"] == "out") != (df["state"].shift() == "out"))
    last = df.index[:i + 1][chg.iloc[:i + 1].values]
    if len(last):
        lines.append(f"上次買賣訊號：{last[-1].date()}")
    lines += ["", "_按回測規則自動提示，唔係投資建議_"]
    return "\n".join(lines), s1, sell_line, buy_line


def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("（冇Telegram設定，只印出嚟）")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text[:4000], "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 400:
            payload.pop("parse_mode")
            requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram 推送失敗: {e}")


def log_to_google_sheet(date, state, vix, close, line):
    if not SHEET_WEBHOOK_URL:
        return
    data = {"date": str(date), "ticker": "SPY", "strategy": f"SPY每日訊號：{NAMES[state]}",
            "vix": f"{vix:.2f}", "ev": "N/A", "pivot": f"{close:.2f}", "stop_loss": f"{line:.2f}"}
    try:
        requests.post(SHEET_WEBHOOK_URL, json=data, timeout=10)
    except Exception as e:
        print(f"Google Sheet 紀錄失敗: {e}")


def main(df=None):
    dry = "--dry" in sys.argv
    if df is None:
        df = load_data()
    df = replay(add_signals(df))

    if "--date" in sys.argv:
        target = pd.Timestamp(sys.argv[sys.argv.index("--date") + 1])
        i = int(df.index.searchsorted(target, side="right")) - 1
    else:
        i = len(df) - 1
    text, state, sell_line, buy_line = build_message(df, i)
    today_ny = datetime.now(ZoneInfo("America/New_York")).date()
    if "--date" not in sys.argv and df.index[i].date() != today_ny:
        text = (f"ℹ️ 今日（{today_ny}）未有新收市數據（可能係休市），"
                f"以下係最後交易日 {df.index[i].date()} 嘅訊號：\n\n") + text
    print(text)
    if not dry:
        send_telegram(text)
        row = df.iloc[i]
        log_to_google_sheet(df.index[i].date(), state, row["vix"], row["raw"],
                            sell_line if state != "out" else buy_line)


if __name__ == "__main__":
    main()
