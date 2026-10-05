"""
momentum_monthly.py — 動量模擬交易（三條線 + SPY基準，各自獨立 $80,000）

  線A：70% 原始動量前10隻 + 30% SPY訊號方案，每半個月換（回測驗證期 +20.2%，最大跌幅 -35%）
  線B：100% 原始動量前3隻，每月換（回測驗證期 +22.7%，最大跌幅 -51%；只作研究）
  線C：70% 多週期動量前10隻 + 30% SPY訊號方案，每半個月換
       多週期 = 3個月、6個月、12-1個月三個排名取平均（第九輪C6：驗證期 +16.4%，最大跌幅 -31%）
  比較對象：SPY、QQQ、QQQM（QQQ平價版）、MTUM（動量ETF）——第一次換倉日買入長揸
       （第十二輪發現線A歷史上同QQQ差唔多、回報大部分係動量因子，所以要同QQQ同MTUM比）
  每月成績表：月尾最後一個交易日（或 --scoreboard）出一次：今年至今、開始至今、年化、最大跌幅、波幅、
       Sharpe、Calmar、回本時間、換手、近3/6/12個月、升市/跌市捕捉（對比QQQ）。
       存 paper_log/scoreboard_YYYY-MM.tsv + Google Sheet（strategy=動量模擬線X｜月度成績表）
  ⚠️ 模擬期間（至少3-6個月）唔改規則，否則模擬交易就驗證唔到回測。

規則：
- 換倉日收市後排名（今日標普500成分股，已計股息嘅價格），揀前N隻平均分配，下一個交易日收市價換倉。
  原始動量 = 252個交易日前到21個交易日前嘅回報。
- 線A/線C嘅30%跟 main.py SPY訊號：持有就揸SPY，現金就揸現金。
- 冇價錢止損，跌出排名先賣。每次換倉按實際買賣金額扣0.1%成本。
換倉日：線A、線C = 每月15號或之前最後一個平日 + 月尾最後一個平日；線B = 月尾。（--force 三條線一齊換）
其他交易日：計每條線由開始到今日嘅總值（NAV）、總回報、最大跌幅、換手，同SPY基準比較。
紀錄：paper_log/YYYY-MM-DD.csv（paper-trading分支）+ Google Sheet（用main.py同一個webhook）：
  換倉：strategy=動量模擬線X｜買入/賣出/繼續持有，ev=動量分數，pivot=參考價
  每日：strategy=動量模擬線X｜NAV，ticker=組合，ev=總回報%，pivot=總值，stop_loss=最大跌幅%
        strategy=動量模擬線X｜持倉，ev=今期回報%，pivot=今期買入價，stop_loss=今日價
用法：python momentum_monthly.py [--force] [--dry] [--tracks=A,C] [--scoreboard] [--scoreboard-only]
"""
import glob
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
CAPITAL = float(os.environ.get("PAPER_CAPITAL", "80000"))
COST = 0.001
LOOK, SKIP = 252, 21
TRACKS = {"A": {"label": "線A：原始前10隻70% + SPY30%（半月換）", "topn": 10, "mom_share": 0.70, "freq": "半月", "score": "原始"},
          "B": {"label": "線B：原始前3隻100%（每月換）", "topn": 3, "mom_share": 1.00, "freq": "月", "score": "原始"},
          "C": {"label": "線C：多週期前10隻70% + SPY30%（半月換）", "topn": 10, "mom_share": 0.70, "freq": "半月", "score": "多週期"}}
SHEET_WEBHOOK_URL = os.environ.get("SHEET_WEBHOOK_URL")


# ---------------- 工具 ----------------
def sheet_row(date, track, ticker, action, ev, pivot, stop, dry):
    row = {"date": str(date), "ticker": ticker, "strategy": f"動量模擬線{track}｜{action}", "vix": "-",
           "ev": ev if isinstance(ev, str) else (f"{ev:+.2f}%" if ev is not None else "-"),
           "pivot": f"{pivot:.2f}" if pivot is not None else "-",
           "stop_loss": stop if isinstance(stop, str) else (f"{stop:.2f}" if stop is not None else "-")}
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


def next_weekday(day):
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def is_last_trading_day(day):
    return next_weekday(day).month != day.month


def is_mid_month_day(day):
    return day.weekday() < 5 and day.day <= 15 and next_weekday(day).day > 15


def tracks_due(day):
    return [tr for tr, cfg in TRACKS.items()
            if is_last_trading_day(day) or (cfg["freq"] == "半月" and is_mid_month_day(day))]


def read_logs():
    out = []
    for f in sorted(glob.glob("paper_log/*.csv")):
        lg = pd.read_csv(f)
        if "track" not in lg.columns:
            lg["track"] = "A"
        out.append(lg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "track", "ticker"])


def scores_at(px, i):
    R121 = px.iloc[i - SKIP] / px.iloc[i - LOOK] - 1
    R3 = px.iloc[i] / px.iloc[i - 63] - 1
    R6 = px.iloc[i] / px.iloc[i - 126] - 1
    multi = pd.concat([R3.rank(pct=True), R6.rank(pct=True), R121.rank(pct=True)], axis=1).mean(axis=1, skipna=False)
    return {"原始": R121.dropna().sort_values(ascending=False), "多週期": multi.dropna().sort_values(ascending=False)}


# ---------------- 模擬總值（NAV）----------------
def simulate_nav(logs, h, spy_hold_eff):
    """按紀錄重播每條線：換倉日收市價重新分配，期間股票隨價浮動，SPY部分跟訊號。回傳 {線: DataFrame}。"""
    out = {}
    for tr, cfg in TRACKS.items():
        lt = logs[logs["track"] == tr]
        if lt.empty:
            continue
        entries = {}
        for dstr, g in lt.groupby("date"):
            after = h.index[h.index > pd.Timestamp(dstr)]
            if len(after):
                entries[after[0]] = list(g.sort_values("rank")["ticker"]) if "rank" in g else list(g["ticker"])
        stocks, spy_val, started, prev = {}, 0.0, False, None
        rows = []
        for d in h.index:
            if d < min(entries):
                continue
            if prev is not None and started:
                r = h.loc[d] / h.loc[prev] - 1
                for t in stocks:
                    x = r.get(t, np.nan)
                    if np.isfinite(x):
                        stocks[t] *= 1 + x
                if spy_hold_eff.get(d, 0.0) > 0 and np.isfinite(r.get("SPY", np.nan)):
                    spy_val *= 1 + r["SPY"]
            turn = 0.0
            if d in entries:
                total = (sum(stocks.values()) + spy_val) if started else CAPITAL
                tks = entries[d]
                new = {t: total * cfg["mom_share"] / len(tks) for t in tks}
                new_spy = total * (1 - cfg["mom_share"])
                turn = (sum(abs(new.get(t, 0) - stocks.get(t, 0)) for t in set(new) | set(stocks))
                        + abs(new_spy - spy_val)) if started else total
                total -= turn * COST
                stocks = {t: total * cfg["mom_share"] / len(tks) for t in tks}
                spy_val = total * (1 - cfg["mom_share"])
                started = True
            nav = sum(stocks.values()) + spy_val
            rows.append((d, nav, turn / max(nav, 1)))
            prev = d
        df = pd.DataFrame(rows, columns=["date", "nav", "turn"]).set_index("date")
        out[tr] = df
    return out


BENCH = {"SPY": "SPY基準（長揸）", "QQQ": "QQQ基準（長揸）", "QQQM": "QQQM基準（長揸，QQQ平價版）",
         "MTUM": "MTUM基準（長揸，動量ETF）"}


def load_state(today):
    """讀紀錄 + 下載價錢 + 重播每條線。回傳 dict；未有紀錄或今日未收市回傳 None。"""
    import yfinance as yf
    logs = read_logs()
    logs = logs[logs["date"] < str(today)]
    if logs.empty:
        print("未有模擬交易紀錄。")
        return None
    first = pd.Timestamp(logs["date"].min())
    allt = sorted(set(logs["ticker"]))
    h = yf.download(allt + list(BENCH), start=(first - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
                    end=(pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True, progress=False)["Close"]
    if isinstance(h, pd.Series):
        h = h.to_frame()
    h = h[h.index > first]
    if h.empty or h.index[-1].date() != today:
        print(f"{today} 未有收市數據，可能休市。")
        return None
    df = bot.replay(bot.add_signals(bot.load_data()))
    st = (df["state"] != "out").astype(float).shift(2)
    spy_hold_eff = st.reindex(h.index, method="ffill").fillna(0.0).to_dict()
    navs = simulate_nav(logs, h, spy_hold_eff)
    start_day = min(nv.index[0] for nv in navs.values())
    bench_navs = {}
    for b in BENCH:
        if b in h.columns and np.isfinite(h[b].get(start_day, np.nan)):
            s = h[b][h.index >= start_day].ffill()
            bench_navs[b] = CAPITAL * s / s.iloc[0]
    return {"logs": logs, "h": h, "navs": navs, "start_day": start_day, "bench": bench_navs}


def summary(nv):
    mdd = (nv / nv.cummax() - 1).min() * 100
    day_r = (nv.iloc[-1] / nv.iloc[-2] - 1) * 100 if len(nv) > 1 else 0.0
    return nv.iloc[-1], (nv.iloc[-1] / CAPITAL - 1) * 100, mdd, day_r


def daily_status(today, dry):
    S = load_state(today)
    if S is None:
        return
    logs, h, navs, start_day = S["logs"], S["h"], S["navs"], S["start_day"]
    lines = [f"📊 *動量模擬交易* {today}（由 {start_day.date()} 開始，每條線 ${CAPITAL:,.0f}）", ""]
    for tr, cfg in TRACKS.items():
        if tr not in navs:
            continue
        nv = navs[tr]["nav"]
        v, tot, mdd, dr = summary(nv)
        lines.append(f"*{cfg['label']}*")
        lines.append(f"總值 ${v:,.0f}（{tot:+.1f}%）｜ 今日 {dr:+.1f}% ｜ 最大跌幅 {mdd:.1f}%")
        sheet_row(today, tr, "組合", "NAV", tot, v, f"{mdd:.1f}%", dry)
        lt = logs[logs["track"] == tr]
        last = lt[lt["date"] == lt["date"].max()]
        sig_day = pd.Timestamp(last["date"].iloc[0])
        hh = h[h.index > sig_day]
        tks = list(last.sort_values("rank")["ticker"]) if "rank" in last else list(last["ticker"])
        if len(hh) == 1:
            lines.append(f"今日係換倉日（今日收市價買入）：{'、'.join(tks)}")
        elif len(hh) > 1:
            entry = hh.iloc[0]
            parts = []
            for t in tks:
                if np.isfinite(entry.get(t, np.nan)):
                    r = (hh[t].iloc[-1] / entry[t] - 1) * 100
                    parts.append(f"{t} {r:+.1f}%")
                    sheet_row(today, tr, t, "持倉", r, entry[t], hh[t].iloc[-1], dry)
            lines.append(f"今期（{hh.index[0].date()} 買入）：" + "、".join(parts))
        lines.append("")
    lines.append("*比較對象（第一日買入之後唔郁）*")
    for b, nv in S["bench"].items():
        bv, btot, bmdd, bdr = summary(nv)
        lines.append(f"{BENCH[b]}：${bv:,.0f}（{btot:+.1f}%）｜ 今日 {bdr:+.1f}% ｜ 最大跌幅 {bmdd:.1f}%")
        sheet_row(today, b, b, "NAV", btot, bv, f"{bmdd:.1f}%", dry)
    lines += ["", "冇止損，換倉日先換股。線A、線C嘅SPY部分跟「SPY每日訊號」。", "_模擬交易，唔係投資建議_"]
    text = "\n".join(lines)
    print(text)
    if not dry:
        bot.send_telegram(text)


# ---------------- 每月成績表 ----------------
def _w(s):
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s, width, left=False):
    gap = max(width - _w(s), 0)
    return s + " " * gap if left else " " * gap + s


def _fmt(v, kind):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if kind == "pct":
        return f"{v:+.1f}%"
    if kind == "dd":
        return f"{v:.1f}%"
    if kind == "num":
        return f"{v:.2f}"
    if kind == "cap":
        return f"{v:.0f}%"
    return str(v)


def get_rf(index):
    try:
        import yfinance as yf
        irx = yf.download("^IRX", start=(index[0] - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                          end=(index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), progress=False)["Close"]
        if isinstance(irx, pd.DataFrame):
            irx = irx.iloc[:, 0]
        return irx.reindex(index, method="ffill").fillna(0.0) / 100 / 252
    except Exception as e:
        print(f"攞唔到現金利率，Sharpe 當0%利率計：{e}")
        return pd.Series(0.0, index=index)


SCORE_ROWS = [("ytd", "今年至今", "pct"), ("total", "開始至今", "pct"), ("cagr", "年化回報", "pct"),
              ("mdd", "最大跌幅", "dd"), ("vol", "波幅", "dd"), ("sharpe", "Sharpe", "num"),
              ("calmar", "Calmar", "num"), ("rec", "回本時間", "txt"), ("turn", "換手(累計)", "dd"),
              ("r3m", "近3個月", "pct"), ("r6m", "近6個月", "pct"), ("r12m", "近12個月", "pct"),
              ("up_cap", "升市捕捉", "cap"), ("down_cap", "跌市捕捉", "cap")]


def monthly_scoreboard(today, dry):
    """月尾最後一個交易日出：A/B/C + SPY/QQQ/QQQM/MTUM 成績表（升跌市捕捉對比QQQ）。"""
    from perf import perf
    S = load_state(today)
    if S is None:
        return
    navs, start_day = S["navs"], S["start_day"]
    series = {}
    for tr in TRACKS:
        if tr in navs:
            nv = navs[tr]["nav"]
            r = nv.pct_change()
            r.iloc[0] = nv.iloc[0] / CAPITAL - 1
            t = navs[tr]["turn"].copy()
            t.iloc[0] = 0.0          # 第一次買入唔計換手
            series[f"線{tr}"] = (r, t)
    for b, nv in S["bench"].items():
        series[b] = (nv.pct_change().fillna(0.0), pd.Series(0.0, index=nv.index))
    idx = series[next(iter(series))][0].index
    rf = get_rf(idx)
    qqq_r = series["QQQ"][0] if "QQQ" in series else None
    stats = {}
    for name, (r, t) in series.items():
        p = perf(r, rf=rf, bench=qqq_r if name != "QQQ" else None, turn=t, asof_year=today.year)
        if p["days"] < 126:
            p["cagr"] = p["calmar"] = np.nan        # 未夠半年，年化數字會好誇張，唔顯示
        p["rec"] = ("冇跌過" if p["mdd"] >= 0 else (f"{p['rec_days']}日" if p["recovered"] else "未回本"))
        p["turn"] = p.get("turn_total", 0.0)
        stats[name] = p

    def table(cols):
        cols = [c for c in cols if c in stats]
        out = [_pad("", 10, True) + "".join(_pad(c, 8) for c in cols)]
        for key, lab, kind in SCORE_ROWS:
            out.append(_pad(lab, 10, True) + "".join(_pad(_fmt(stats[c].get(key), kind), 8) for c in cols))
        return "```\n" + "\n".join(out) + "\n```"

    days = stats[next(iter(stats))]["days"]
    lines = [f"🗓️ *每月成績表* {today}（由 {start_day.date()} 開始，{days} 個交易日）", "",
             "*我哋三條線*", table(["線A", "線B", "線C"]), "",
             "*比較對象*", table(["SPY", "QQQ", "QQQM", "MTUM"]), "",
             "點睇：",
             "• 未夠半年唔計年化回報同Calmar（數字會好誇張）；近3/6/12個月要夠日子先有。",
             "• 升市/跌市捕捉係同QQQ比：例如 120% / 100% = 升市多賺、跌市冇多蝕，好；120% / 150% = 多賺只係因為承受多咗跌。",
             "• Calmar = 年化回報 ÷ 最大跌幅，越高越好。回本時間 = 由最低位返到之前高位用咗幾耐。",
             "• A/B/C規則鎖死唔改。6個月後先用決策表判斷：A明顯贏QQQ同MTUM而跌幅唔差 → 繼續；"
             "A≈QQQ但跌幅更大 → 直接買QQQ更合理；A輸SPY → 停止。",
             "_模擬交易，唔係投資建議_"]
    text = "\n".join(lines)
    print(text)
    os.makedirs("paper_log", exist_ok=True)
    tab = pd.DataFrame({name: {lab: _fmt(p.get(key), kind) for key, lab, kind in SCORE_ROWS} for name, p in stats.items()})
    tab.to_csv(f"paper_log/scoreboard_{today:%Y-%m}.tsv", sep="\t", encoding="utf-8-sig")
    for name, p in stats.items():
        for key, lab, kind in SCORE_ROWS:
            sheet_row(today, name.replace("線", ""), lab, "月度成績表", _fmt(p.get(key), kind), None, "-", dry)
    if not dry:
        bot.send_telegram(text)


# ---------------- 換倉 ----------------
def main():
    force, dry = "--force" in sys.argv, "--dry" in sys.argv
    today = datetime.now(ZoneInfo("America/New_York")).date()
    want_board = "--scoreboard" in sys.argv or is_last_trading_day(today)
    if "--scoreboard-only" in sys.argv:
        monthly_scoreboard(today, dry)
        return
    due = list(TRACKS) if force else tracks_due(today)
    only = [a.split("=")[1] for a in sys.argv if a.startswith("--tracks=")]
    if only:
        due = [t for t in due if t in only[0].split(",")]
    if not due:
        daily_status(today, dry)
        if want_board:
            monthly_scoreboard(today, dry)
        return
    rebalance(today, due, force, dry)
    if want_board:
        monthly_scoreboard(today, dry)


def rebalance(today, due, force, dry):
    tickers, names = current_sp500()
    closes = M.download_close(tickers, (pd.Timestamp(today) - pd.Timedelta(days=500)).strftime("%Y-%m-%d"),
                              (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    px = pd.DataFrame(closes).sort_index()
    px = px[px.index.dayofweek < 5]
    last_day = px.index[-1].date()
    if not force and last_day != today:
        msg = f"ℹ️ 動量模擬：今日（{today}）未有收市數據，可能休市，下個交易日再試。"
        print(msg)
        if not dry:
            bot.send_telegram(msg)
        return
    i = len(px) - 1
    sc = scores_at(px, i)
    df = bot.replay(bot.add_signals(bot.load_data()))
    spy_hold = df["state"].iloc[-1] != "out"
    logs = read_logs()
    logs = logs[logs["date"] < str(last_day)]

    lines = [f"📅 *動量換倉（模擬交易）* {last_day}", f"每條線模擬本金 ${CAPITAL:,.0f}", ""]
    rows = []
    for tr, cfg in TRACKS.items():
        if tr not in due:
            continue
        n = cfg["topn"]
        ranked = sc[cfg["score"]]
        top = list(ranked.index[:n])
        lt = logs[logs["track"] == tr]
        prev_top = list(lt[lt["date"] == lt["date"].max()]["ticker"]) if not lt.empty else []
        add = [t for t in top if t not in prev_top]
        drop = [t for t in prev_top if t not in top]
        per = CAPITAL * cfg["mom_share"] / n
        lines.append(f"*{cfg['label']}*（每隻大約 ${per:,.0f}）")
        for k, t in enumerate(top, 1):
            p = px[t].iloc[i]
            show = f"動量 {sc['原始'].get(t, np.nan) * 100:+.0f}%" if cfg["score"] == "原始" else f"綜合排名分 {ranked[t]:.2f}"
            lines.append(f"{k}. {t}（{names.get(t, '')}）{show} ｜ 參考價 ${p:,.2f} ｜ 大約 {per / p:.2f} 股")
        lines += [f"🟢 新加入：{', '.join(add) if add else '冇'}",
                  f"🔴 要賣出：{', '.join(drop) if drop else '冇'}",
                  f"⚪ 繼續持有：{', '.join(t for t in top if t in prev_top) or '冇'}"]
        if cfg["mom_share"] < 1:
            lines.append(f"SPY部分（${CAPITAL * (1 - cfg['mom_share']):,.0f}）：{'持有SPY' if spy_hold else '揸現金'}（跟SPY每日訊號）")
        lines.append("")
        for k, t in enumerate(top, 1):
            rows.append({"date": str(last_day), "track": tr, "rank": k, "ticker": t, "score": round(float(ranked[t]), 4),
                         "weight": cfg["mom_share"] / n, "spy_signal": "hold" if spy_hold else "cash"})
            sheet_row(last_day, tr, t, "繼續持有" if t in prev_top else "買入（下個交易日收市）",
                      float(sc["原始"].get(t, np.nan)) * 100, px[t].iloc[i], px[t].iloc[i], dry)
        for t in drop:
            sheet_row(last_day, tr, t, "賣出（下個交易日收市）", None, None,
                      px[t].iloc[i] if t in px.columns else None, dry)
    lines += ["做法：下一個交易日收市前換倉（用收市價買賣，IB可以買碎股）。", "止損：冇止損，跌出排名先賣。"]
    if "B" in due:
        lines.append("⚠️ 線B只揸3隻，回測最大跌幅大約 -51%，只作研究。")
    lines.append("_模擬交易，用嚟驗證回測，唔係投資建議_")
    text = "\n".join(lines)
    print(text)
    os.makedirs("paper_log", exist_ok=True)
    out = pd.DataFrame(rows)
    path = f"paper_log/{last_day}.csv"
    if os.path.exists(path):          # 同一日重跑：保留冇換倉嗰啲線嘅紀錄
        old = pd.read_csv(path)
        if "track" not in old.columns:
            old["track"] = "A"
        out = pd.concat([old[~old["track"].isin(due)], out], ignore_index=True)
    out.to_csv(path, index=False)
    if not dry:
        bot.send_telegram(text)


if __name__ == "__main__":
    main()
