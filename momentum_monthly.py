"""
momentum_monthly.py — 每月模擬交易提示：70% 動量前10隻 + 30% SPY訊號方案

規則（research_v6 預先定好揀法揀中嘅「原始動量前10隻 70% + SPY訊號 30%」）：
- 每月最後一個交易日收市後：將今日標普500成分股按「12-1動量」排名
  （由252個交易日前到21個交易日前嘅回報，已計股息），揀最高10隻。
- 70% 資金平均分落呢10隻（每隻7%）；下一個交易日收市前換倉。
- 30% 資金跟 main.py 嘅SPY訊號：訊號話持有就買SPY，話現金就揸現金。
- 回測（2018-2026驗證期）：每年 +18.6%，最大跌幅 -34.7%。⚠️ 呢個組合係睇過之前結果先提出，
  所以要靠模擬交易確認，唔好當保證。

運作：GitHub Actions 每個交易日收市後跑。月尾最後一個交易日：發換倉名單（或者用 --force）；
其他交易日：發模擬持倉由買入至今嘅表現。
每次結果記錄落 paper_log/YYYY-MM-DD.csv（workflow會放上 paper-trading 分支），同時寫入Google Sheet
（用main.py同一個webhook：strategy=動量模擬｜動作、ev=回報%或動量%、pivot=買入/參考價、stop_loss=今日價）。
用法：python momentum_monthly.py [--force] [--dry]
"""
import io
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import main as bot
import momentum_v1 as M

M.PROGRESS = "/tmp/momentum_monthly_progress.txt"
CAPITAL = float(os.environ.get("PAPER_CAPITAL", "10000"))
MOM_SHARE, TOPN, LOOK, SKIP = 0.70, 10, 252, 21


SHEET_WEBHOOK_URL = os.environ.get("SHEET_WEBHOOK_URL")


def sheet_row(date, ticker, action, ret_pct, entry_px, now_px, dry):
    """寫一行落Google Sheet（用返main.py同一個webhook同欄位）：
    strategy = 動量模擬｜動作、ev = 回報%、pivot = 買入/參考價、stop_loss = 今日價（呢個方案冇止損）。"""
    row = {"date": str(date), "ticker": ticker, "strategy": f"動量模擬｜{action}", "vix": "-",
           "ev": f"{ret_pct:+.2f}%" if ret_pct is not None else "-",
           "pivot": f"{entry_px:.2f}" if entry_px is not None else "-",
           "stop_loss": f"{now_px:.2f}" if now_px is not None else "-"}
    if dry or not SHEET_WEBHOOK_URL:
        print("Sheet:", row)
        return
    try:
        requests.post(SHEET_WEBHOOK_URL, json=row, timeout=10)
    except Exception as e:
        print(f"Google Sheet 紀錄失敗: {e}")


def current_sp500():
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=M.HDR, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    return sorted(set(M.norm(s) for s in t["Symbol"].dropna())), dict(zip(t["Symbol"].map(M.norm), t["Security"]))


def is_last_trading_day(day):
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.month != day.month


def rank_at(px, i):
    R = px.iloc[i - SKIP] / px.iloc[i - LOOK] - 1
    return R.dropna().sort_values(ascending=False)


def daily_status(today, dry):
    """平日：報告今個月模擬持倉由買入至今嘅表現。"""
    import glob
    import yfinance as yf
    logs = sorted(glob.glob("paper_log/*.csv"))
    logs = [f for f in logs if pd.read_csv(f)["date"].iloc[0] < str(today)]
    if not logs:
        print("未有模擬交易紀錄（要等第一次月尾換倉）。")
        return
    lg = pd.read_csv(logs[-1])
    sig_day = pd.Timestamp(lg["date"].iloc[0])
    top = list(lg["ticker"])
    h = yf.download(top + ["SPY"], start=(sig_day - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
                    end=(pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True, progress=False)["Close"]
    h = h[h.index > sig_day]
    if h.empty:
        return
    if h.index[-1].date() != today:
        print(f"{today} 未有收市數據，可能休市。")
        return
    per_stock = CAPITAL * MOM_SHARE / len(top)
    if len(h) == 1:
        text = (f"📅 *每月動量（模擬交易）* {today}\n\n今日係換倉日（應該喺今日收市前，每隻用大約 ${per_stock:,.0f} 買入）：\n"
                + "、".join(top) + "\n\n模擬成績由今日收市價開始計。\n_模擬交易，唔係投資建議_")
    else:
        entry = h.iloc[0]
        lines = [f"📊 *動量模擬持倉* {today}（{h.index[0].date()} 收市買入）", ""]
        total = 0.0
        for t in top:
            r = h[t].iloc[-1] / entry[t] - 1 if np.isfinite(entry.get(t, np.nan)) else 0.0
            total += per_stock * (1 + r)
            lines.append(f"{t}：{r * 100:+.1f}%（${per_stock * (1 + r):,.0f}）")
        spy_r = h["SPY"].iloc[-1] / entry["SPY"] - 1
        lines += ["", f"動量部分：${total:,.0f}（{(total / (per_stock * len(top)) - 1) * 100:+.1f}%）｜ 同期SPY {spy_r * 100:+.1f}%",
                  "今日唔使做嘢，月尾先換股。SPY部分睇另一個「SPY每日訊號」。", "_模擬交易，唔係投資建議_"]
        text = "\n".join(lines)
        for t in top:
            if np.isfinite(entry.get(t, np.nan)):
                sheet_row(today, t, "持倉", (h[t].iloc[-1] / entry[t] - 1) * 100, entry[t], h[t].iloc[-1], dry)
        sheet_row(today, "組合", f"總值 ${total:,.0f}", (total / (per_stock * len(top)) - 1) * 100, None, None, dry)
    print(text)
    if not dry:
        bot.send_telegram(text)


def main():
    force, dry = "--force" in sys.argv, "--dry" in sys.argv
    today = datetime.now(ZoneInfo("America/New_York")).date()
    if not force and not is_last_trading_day(today):
        daily_status(today, dry)
        return
    tickers, names = current_sp500()
    closes = M.download_close(tickers, (pd.Timestamp(today) - pd.Timedelta(days=500)).strftime("%Y-%m-%d"),
                              (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    px = pd.DataFrame(closes).sort_index()
    px = px[px.index.dayofweek < 5]
    last_day = px.index[-1].date()
    if not force and last_day != today:
        msg = f"ℹ️ 每月動量：今日（{today}）未有收市數據，可能休市，下個交易日再試。"
        print(msg)
        if not dry:
            bot.send_telegram(msg)
        return
    i = len(px) - 1
    now = rank_at(px, i)
    top = list(now.index[:TOPN])
    prev_i = px.index.searchsorted(px.index[i].replace(day=1), side="left") - 1
    prev_top = list(rank_at(px, prev_i).index[:TOPN]) if prev_i > LOOK else []
    add = [t for t in top if t not in prev_top]
    drop = [t for t in prev_top if t not in top]

    df = bot.replay(bot.add_signals(bot.load_data()))
    spy_hold = df["state"].iloc[-1] != "out"
    per_stock = CAPITAL * MOM_SHARE / TOPN
    spy_amt = CAPITAL * (1 - MOM_SHARE)

    lines = [f"📅 *每月動量（模擬交易）* {last_day}", "",
             f"模擬本金 ${CAPITAL:,.0f}：70% 動量股 + 30% SPY訊號", "",
             f"*動量前{TOPN}隻*（每隻大約 ${per_stock:,.0f}）："]
    for k, t in enumerate(top, 1):
        p = px[t].iloc[i]
        lines.append(f"{k}. {t}（{names.get(t, '')}）動量 {now[t] * 100:+.0f}% ｜ 參考價 ${p:,.2f} ｜ 大約 {per_stock / p:.2f} 股")
    lines += ["", f"🟢 新加入：{', '.join(add) if add else '冇'}",
              f"🔴 要賣出：{', '.join(drop) if drop else '冇'}",
              f"⚪ 繼續持有：{', '.join(t for t in top if t in prev_top) or '冇'}", "",
              f"SPY部分（${spy_amt:,.0f}）：{'持有SPY' if spy_hold else '揸現金'}（跟SPY每日訊號）", "",
              "做法：下一個交易日收市前換倉（用收市價買賣，IB可以買碎股）。",
              "止損：呢個方案冇止損（回測顯示止損令結果更差），每月尾先換股。",
              "_模擬交易，用嚟驗證回測，唔係投資建議_"]
    text = "\n".join(lines)
    print(text)
    os.makedirs("paper_log", exist_ok=True)
    pd.DataFrame({"date": str(last_day), "rank": range(1, TOPN + 1), "ticker": top,
                  "momentum": [round(now[t], 4) for t in top], "weight": MOM_SHARE / TOPN,
                  "spy_signal": "hold" if spy_hold else "cash"}).to_csv(
        f"paper_log/{last_day}.csv", index=False)
    for t in top:
        sheet_row(last_day, t, "繼續持有" if t in prev_top else "買入（下個交易日收市）", now[t] * 100, px[t].iloc[i], px[t].iloc[i], dry)
    for t in drop:
        p = px[t].iloc[i] if t in px.columns else None
        sheet_row(last_day, t, "賣出（下個交易日收市）", None, None, p, dry)
    if not dry:
        bot.send_telegram(text)


if __name__ == "__main__":
    main()
