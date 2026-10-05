"""
momentum_monthly.py — 每月模擬交易提示（兩條線）
  A：70% 動量前10隻 + 30% SPY訊號方案
  B：100% 動量前3隻（集中，跌幅大好多：回測驗證期 +22.7%，最大跌幅 -51%）

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
LOOK, SKIP = 252, 21
TRACKS = {"A": {"label": "線A：前10隻70% + SPY訊號30%", "topn": 10, "mom_share": 0.70},
          "B": {"label": "線B：前3隻100%", "topn": 3, "mom_share": 1.00}}
SHEET_WEBHOOK_URL = os.environ.get("SHEET_WEBHOOK_URL")


def sheet_row(date, track, ticker, action, ret_pct, entry_px, now_px, dry):
    """寫一行落Google Sheet（用返main.py同一個webhook同欄位）：
    strategy = 動量模擬線A/B｜動作、ev = 回報%、pivot = 買入/參考價、stop_loss = 今日價（呢個方案冇止損）。"""
    row = {"date": str(date), "ticker": ticker, "strategy": f"動量模擬線{track}｜{action}", "vix": "-",
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
    """平日：報告兩條線由買入至今嘅表現。"""
    import glob
    import yfinance as yf
    logs = sorted(glob.glob("paper_log/*.csv"))
    logs = [f for f in logs if pd.read_csv(f)["date"].iloc[0] < str(today)]
    if not logs:
        print("未有模擬交易紀錄（要等第一次月尾換倉）。")
        return
    lg = pd.read_csv(logs[-1])
    if "track" not in lg.columns:
        lg["track"] = "A"
    sig_day = pd.Timestamp(lg["date"].iloc[0])
    allt = sorted(set(lg["ticker"]))
    h = yf.download(allt + ["SPY"], start=(sig_day - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
                    end=(pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True, progress=False)["Close"]
    h = h[h.index > sig_day]
    if h.empty:
        return
    if h.index[-1].date() != today:
        print(f"{today} 未有收市數據，可能休市。")
        return
    if len(h) == 1:
        lines = [f"📅 *每月動量（模擬交易）* {today}", "", "今日係換倉日（應該喺今日收市前買入），模擬成績由今日收市價開始計：", ""]
        for tr, cfg in TRACKS.items():
            top = list(lg.loc[lg["track"] == tr, "ticker"])
            if top:
                per = CAPITAL * cfg["mom_share"] / len(top)
                lines.append(f"{cfg['label']}：每隻大約 ${per:,.0f} → " + "、".join(top))
        lines.append("_模擬交易，唔係投資建議_")
    else:
        entry = h.iloc[0]
        spy_r = h["SPY"].iloc[-1] / entry["SPY"] - 1
        lines = [f"📊 *動量模擬持倉* {today}（{h.index[0].date()} 收市買入）", ""]
        for tr, cfg in TRACKS.items():
            top = list(lg.loc[lg["track"] == tr, "ticker"])
            if not top:
                continue
            per = CAPITAL * cfg["mom_share"] / len(top)
            total, parts = 0.0, []
            for t in top:
                r = h[t].iloc[-1] / entry[t] - 1 if np.isfinite(entry.get(t, np.nan)) else 0.0
                total += per * (1 + r)
                parts.append(f"{t} {r * 100:+.1f}%")
                if np.isfinite(entry.get(t, np.nan)):
                    sheet_row(today, tr, t, "持倉", r * 100, entry[t], h[t].iloc[-1], dry)
            base = per * len(top)
            lines += [f"*{cfg['label']}*", "、".join(parts),
                      f"股票部分：${total:,.0f}（{(total / base - 1) * 100:+.1f}%）", ""]
            sheet_row(today, tr, "組合", f"股票部分總值 ${total:,.0f}", (total / base - 1) * 100, None, None, dry)
        lines += [f"同期SPY {spy_r * 100:+.1f}%", "今日唔使做嘢，月尾先換股。線A嘅SPY部分睇「SPY每日訊號」。",
                  "_模擬交易，唔係投資建議_"]
    text = "\n".join(lines)
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
    prev_i = px.index.searchsorted(px.index[i].replace(day=1), side="left") - 1
    prev = rank_at(px, prev_i) if prev_i > LOOK else pd.Series(dtype=float)
    df = bot.replay(bot.add_signals(bot.load_data()))
    spy_hold = df["state"].iloc[-1] != "out"

    lines = [f"📅 *每月動量（模擬交易）* {last_day}", f"每條線模擬本金 ${CAPITAL:,.0f}", ""]
    rows = []
    for tr, cfg in TRACKS.items():
        n = cfg["topn"]
        top, prev_top = list(now.index[:n]), list(prev.index[:n])
        add = [t for t in top if t not in prev_top]
        drop = [t for t in prev_top if t not in top]
        per = CAPITAL * cfg["mom_share"] / n
        lines.append(f"*{cfg['label']}*（每隻大約 ${per:,.0f}）")
        for k, t in enumerate(top, 1):
            p = px[t].iloc[i]
            lines.append(f"{k}. {t}（{names.get(t, '')}）動量 {now[t] * 100:+.0f}% ｜ 參考價 ${p:,.2f} ｜ 大約 {per / p:.2f} 股")
        lines += [f"🟢 新加入：{', '.join(add) if add else '冇'}",
                  f"🔴 要賣出：{', '.join(drop) if drop else '冇'}",
                  f"⚪ 繼續持有：{', '.join(t for t in top if t in prev_top) or '冇'}"]
        if cfg["mom_share"] < 1:
            lines.append(f"SPY部分（${CAPITAL * (1 - cfg['mom_share']):,.0f}）：{'持有SPY' if spy_hold else '揸現金'}（跟SPY每日訊號）")
        lines.append("")
        for k, t in enumerate(top, 1):
            rows.append({"date": str(last_day), "track": tr, "rank": k, "ticker": t, "momentum": round(now[t], 4),
                         "weight": cfg["mom_share"] / n, "spy_signal": "hold" if spy_hold else "cash"})
            sheet_row(last_day, tr, t, "繼續持有" if t in prev_top else "買入（下個交易日收市）",
                      now[t] * 100, px[t].iloc[i], px[t].iloc[i], dry)
        for t in drop:
            sheet_row(last_day, tr, t, "賣出（下個交易日收市）", None, None,
                      px[t].iloc[i] if t in px.columns else None, dry)
    lines += ["做法：下一個交易日收市前換倉（用收市價買賣，IB可以買碎股）。",
              "止損：冇止損，跌出排名先賣（回測顯示價錢止損令結果更差）。",
              "⚠️ 線B只揸3隻，回測最大跌幅大約 -51%。",
              "_模擬交易，用嚟驗證回測，唔係投資建議_"]
    text = "\n".join(lines)
    print(text)
    os.makedirs("paper_log", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"paper_log/{last_day}.csv", index=False)
    if not dry:
        bot.send_telegram(text)


if __name__ == "__main__":
    main()
