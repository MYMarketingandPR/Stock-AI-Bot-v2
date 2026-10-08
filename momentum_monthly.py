"""
momentum_monthly.py — 動量模擬交易（三條線 + SPY基準，各自獨立 $80,000）

  線A：70% 原始動量前10隻 + 30% SPY訊號方案，每半個月換（回測驗證期 +20.2%，最大跌幅 -35%）
  線B：100% 原始動量前3隻，每月換（回測驗證期 +22.7%，最大跌幅 -51%；只作研究）
  線C：70% 多週期動量前10隻 + 30% SPY訊號方案，每半個月換
       多週期 = 3個月、6個月、12-1個月三個排名取平均（第九輪C6：驗證期 +16.4%，最大跌幅 -31%）
  線D：倍升股 + 由高位跌20%賣（line_d.py，每日獨立跑同發訊息；月尾成績表會一齊列出）
  輪動：美股/納指/海外股/長債/黃金，高過10個月平均先揸，每樣20%（rotation_live.py；2026-10-07 開始）
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
import prices as PR
import rotation_live as RL
import sheet_v2 as SV

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
    if SV.available():            # Apps Script 已經有新格式（分頁）→ 唔再寫舊格式，避免第一頁太亂
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


def next_rebalance(day, freq):
    d = day
    for _ in range(45):
        d = next_weekday(d)
        if is_last_trading_day(d) or (freq == "半月" and is_mid_month_day(d)):
            return d
    return None


def safe_names():
    try:
        return current_sp500()[1]
    except Exception:
        return {}


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
    start_dl = min(first - pd.Timedelta(days=7), pd.Timestamp(today) - pd.Timedelta(days=400))
    h = yf.download(allt + list(BENCH), start=start_dl.strftime("%Y-%m-%d"),
                    end=(pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True, progress=False)["Close"]
    if isinstance(h, pd.Series):
        h = h.to_frame()
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    h = h[h.index <= pd.Timestamp(today)]
    # 逐隻檢查有冇今日收市價（2026-10-08 修正：之前有啲股票用咗前一日價）
    need = set(BENCH)
    for tr in TRACKS:
        lt = logs[logs["track"] == tr]
        if not lt.empty:
            need |= set(lt[lt["date"] == lt["date"].max()]["ticker"])
    h, miss = PR.ensure_fresh(h, sorted(need), today)
    if "SPY" in miss:
        print(f"{today} 未有收市數據（可能休市，或者 Yahoo 未更新）。")
        return None
    if miss and not PR.last_chance():
        print(f"{today} 收市價未齊（{miss}），今次唔出報告，等後備時間再跑。")
        return None
    h_all = h
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
    r12 = {}
    for t in h_all.columns:
        sr = h_all[t].dropna()
        sr = sr[sr.index <= pd.Timestamp(today)]
        if len(sr) >= 253:
            r12[t] = float(sr.iloc[-1] / sr.iloc[-253] - 1)
    return {"logs": logs, "h": h, "navs": navs, "start_day": start_day, "bench": bench_navs, "spy_hold": spy_hold_eff,
            "r12": r12, "h_all": h_all, "stale": miss}


def rotation_state(today):
    """輪動（rotation_live.py）：下載5個ETF、由開始日計到今日。出錯唔影響其他線。"""
    try:
        px = RL.download(today)
        px, miss = PR.ensure_fresh(px, list(RL.ASSETS), today, rounds=2)
        if miss:
            print(f"輪動：{miss} 未有 {today} 收市價，今日唔計。")
            return None
        return RL.simulate(px, get_rf(px.index), today, is_last_trading_day(today), CAPITAL)
    except Exception as e:
        print(f"輪動計唔到：{e}")
        return None


def summary(nv):
    mdd = (nv / nv.cummax() - 1).min() * 100
    day_r = (nv.iloc[-1] / nv.iloc[-2] - 1) * 100 if len(nv) > 1 else 0.0
    return nv.iloc[-1], (nv.iloc[-1] / CAPITAL - 1) * 100, mdd, day_r


def daily_status(today, dry, send=True):
    S = load_state(today)
    if S is None:
        return False
    logs, h, navs, start_day = S["logs"], S["h"], S["navs"], S["start_day"]
    r12 = S.get("r12", {})
    names = safe_names()
    hold_rows, daily_rows = [], []
    spy_now = S.get("spy_hold", {}).get(h.index[-1], 0.0) > 0
    sections = []
    head = f"📊 *動量模擬交易* {today}（由 {start_day.date()} 開始，每條線 ${CAPITAL:,.0f}）"
    if S.get("stale"):
        head += f"\n⚠️ 以下股票到最後都未有 {today} 收市價，用緊上一日價：{'、'.join(S['stale'])}"
    for tr, cfg in TRACKS.items():
        if tr not in navs:
            continue
        nv = navs[tr]["nav"]
        v, tot, mdd, dr = summary(nv)
        lines = [f"*{cfg['label']}*"]
        lines.append(f"總值 ${v:,.0f}（{tot:+.1f}%，{v - CAPITAL:+,.0f}）｜ 今日 {dr:+.1f}% ｜ 最大跌幅 {mdd:.1f}%")
        sheet_row(today, tr, "組合", "NAV", tot, v, f"{mdd:.1f}%", dry)
        lt = logs[logs["track"] == tr]
        last = lt[lt["date"] == lt["date"].max()]
        sig_day = pd.Timestamp(last["date"].iloc[0])
        hh = h[h.index > sig_day]
        tks = list(last.sort_values("rank")["ticker"]) if "rank" in last else list(last["ticker"])
        nxt = next_rebalance(today, cfg["freq"])
        stock_val = 0.0
        if len(hh) >= 1:
            ed = hh.index[0]
            nav_e = float(navs[tr]["nav"].get(ed, np.nan))
            okt = [t for t in tks if t in hh.columns and np.isfinite(hh[t].iloc[0])]
            alloc = nav_e * cfg["mom_share"] / max(len(okt), 1)
            first_day = len(hh) == 1
            lines.append(f"{'今日係換倉日，今日收市價買入' if first_day else f'{ed.date()} 收市買入'}（每隻大約 ${alloc:,.0f}）：")
            for k, t in enumerate(okt, 1):
                ep, cp = float(hh[t].iloc[0]), float(hh[t].ffill().iloc[-1])
                sh = alloc / ep
                val = sh * cp
                stock_val += val
                r_in = ""
                if "h_all" in S and t in S["h_all"].columns:
                    sr = S["h_all"][t].dropna()
                    sr = sr[sr.index <= ed]
                    if len(sr) >= 253:
                        r_in = round(float(sr.iloc[-1] / sr.iloc[-253] - 1) * 100, 1)
                g_now = f"{r12[t] * 100:+.0f}%" if t in r12 else "—"
                lines.append(f"{k}. {t}（{names.get(t, '')}）買入 ${ep:,.2f} → 今日 ${cp:,.2f} ｜ {sh:.2f} 股 ｜ "
                             f"{val - alloc:+,.0f}（{(cp / ep - 1) * 100:+.1f}%）｜ 過去12個月 {g_now}")
                hold_rows.append([f"線{tr}", t, names.get(t, ""), str(ed.date()), round(ep, 2), round(sh, 4), round(alloc, 2),
                                  round(cp, 2), round(val, 2), round(val - alloc, 2), round((cp / ep - 1) * 100, 2), "",
                                  f"冇止損；{nxt} 換倉日跌出頭{cfg['topn']}名就賣（下一個交易日收市）",
                                  r_in, round(r12[t] * 100, 1) if t in r12 else "", str(today)])
            if cfg["mom_share"] < 1:
                spy_val = float(nv.iloc[-1]) - stock_val
                base_spy = nav_e * (1 - cfg["mom_share"])
                lines.append(f"SPY部分：${base_spy:,.0f} → ${spy_val:,.0f}（{spy_val - base_spy:+,.0f}）｜ "
                             f"{'而家持有SPY' if spy_now else '而家揸現金'}（跟SPY每日訊號）")
                hold_rows.append([f"線{tr}", "SPY部分", "跟SPY每日訊號", str(ed.date()), "", "", round(nav_e * (1 - cfg["mom_share"]), 2),
                                  round(float(h["SPY"].iloc[-1]), 2) if "SPY" in h.columns else "", round(spy_val, 2),
                                  round(spy_val - nav_e * (1 - cfg["mom_share"]), 2), "", "",
                                  "持有SPY（訊號轉現金就賣）" if spy_now else "揸現金（訊號轉持有就買SPY）", "", "", str(today)])
        daily_rows.append([str(today), f"線{tr}", round(v, 2), round(tot, 2), round(dr, 2), round(mdd, 2),
                           round(float(nv.iloc[-1]) - stock_val, 2) if cfg["mom_share"] < 1 and not spy_now else 0,
                           len(tks), f"下次換倉 {nxt}"])
        if len(hh) > 1:
            entry = hh.iloc[0]
            for t in tks:
                if np.isfinite(entry.get(t, np.nan)):
                    r = (hh[t].iloc[-1] / entry[t] - 1) * 100
                    sheet_row(today, tr, t, "持倉", r, entry[t], hh[t].iloc[-1], dry)
        lines.append(f"幾時賣：冇止損；下次換倉日 {nxt} 收市後重新排名，跌出頭{cfg['topn']}名嘅下一個交易日收市賣出。")
        sections.append("\n".join(lines))
    R = rotation_state(today)
    if R is not None:
        try:
            sec, rrows, drow = RL.report(R, today, CAPITAL)
            sections.append(sec)
            hold_rows += rrows
            daily_rows.append(drow)
            if R["trades"]:
                SV.upsert("買賣紀錄", SV.TRADE_HEADER, [t + [f"{t[0]}|輪動|{t[3]}|{t[2]}"] for t in R["trades"]], dry)
        except Exception as e:
            print(f"輪動報告出錯：{e}")
    lines = ["*比較對象（第一日買入之後唔郁）*"]
    for b, nv in S["bench"].items():
        bv, btot, bmdd, bdr = summary(nv)
        lines.append(f"{BENCH[b]}：${bv:,.0f}（{btot:+.1f}%）｜ 今日 {bdr:+.1f}% ｜ 最大跌幅 {bmdd:.1f}%")
        sheet_row(today, b, b, "NAV", btot, bv, f"{bmdd:.1f}%", dry)
        daily_rows.append([str(today), f"比較：{b}", round(bv, 2), round(btot, 2), round(bdr, 2), round(bmdd, 2), 0, 1,
                           "第一日買入之後唔郁"])
    SV.replace_group("現時持倉", SV.HOLD_HEADER, 1, [f"線{t}" for t in TRACKS] + ["輪動"], hold_rows, dry)
    SV.upsert("每日總覽", SV.DAILY_HEADER, [r + [f"{r[0]}|{r[1]}"] for r in daily_rows], dry)
    lines += ["", "冇止損，換倉日先換股。線A、線C嘅SPY部分跟「SPY每日訊號」。", "_模擬交易，唔係投資建議_"]
    sections.append("\n".join(lines))
    send_sections(head, sections, dry or not send)
    return True


def send_sections(head, sections, dry, limit=3800):
    """Telegram 每個訊息最多大約4000字：按段落分開發。"""
    msgs, cur = [], head
    for sec in sections:
        if len(cur) + len(sec) + 2 > limit:
            msgs.append(cur)
            cur = head + "（續）"
        cur += "\n\n" + sec
    msgs.append(cur)
    for m in msgs:
        print(m)
        print("-" * 20)
        if not dry:
            bot.send_telegram(m)


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
    if os.path.exists("paper_log/lineD_nav.tsv"):           # 線D（line_d.py，每日自己記總值）
        dn = pd.read_csv("paper_log/lineD_nav.tsv", sep="\t", parse_dates=["date"]).drop_duplicates("date", keep="last")
        dn = dn[dn["date"] <= pd.Timestamp(today)].set_index("date")
        if len(dn) >= 2:
            r = dn["nav"].pct_change()
            r.iloc[0] = dn["nav"].iloc[0] / CAPITAL - 1
            series["線D"] = (r, dn["turn"].astype(float))
    R = rotation_state(today)                               # 輪動（rotation_live.py）
    if R is not None and len(R["nav"]) >= 1:
        nv = R["nav"]
        r = nv.pct_change()
        r.iloc[0] = nv.iloc[0] / CAPITAL - 1
        t = R["turn"].copy()
        t.iloc[0] = 0.0
        series["輪動"] = (r, t)
    for b, nv in S["bench"].items():
        series[b] = (nv.pct_change().fillna(0.0), pd.Series(0.0, index=nv.index))
    # SPY訊號方案（main.py：持有就揸SPY，現金就揸現金）同兩個組合（用模擬交易真實數字，每日固定比例）
    h, hold = S["h"], S.get("spy_hold", {})
    if "SPY" in h.columns:
        sr = h["SPY"].pct_change()
        sig = pd.Series([float(sr[d]) if hold.get(d, 0.0) > 0 and np.isfinite(sr[d]) else 0.0 for d in h.index], index=h.index)
        sig = sig[sig.index >= start_day]
        sig.iloc[0] = 0.0
        series["SPY訊號"] = (sig, pd.Series(0.0, index=sig.index))

    def combo(weights):
        df = pd.concat({k: series[k][0] for k in weights}, axis=1, join="inner").fillna(0.0)
        if len(df) < 2:
            return None
        return sum(w * df[k] for k, w in weights.items())
    for name, wts in (("組合甲", {"SPY訊號": 0.70, "線A": 0.15, "線D": 0.15}), ("組合乙", {"QQQ": 0.85, "線D": 0.15}),
                       ("組合丙", {"輪動": 0.50, "線D": 0.50})):
        if all(k in series for k in wts):
            r = combo(wts)
            if r is not None:
                series[name] = (r, pd.Series(0.0, index=r.index))
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
             "*我哋嘅線*", table(["線A", "線B", "線C", "線D"]), "",
             "*比較對象*", table(["SPY", "QQQ", "QQQM", "MTUM"]), "",
             "*三個組合（用模擬交易真實數字計，每日固定比例）*", table(["組合甲", "組合乙", "組合丙", "QQQ"]),
             "組合甲 = 70% SPY訊號 + 15% 線A + 15% 線D（回測守得住科網爆煲：2000-2009 最大跌幅 -31%）",
             "組合乙 = 85% QQQ + 15% 線D（回測2015年起回報較高，但2000-2009 最大跌幅 -77%）",
             "組合丙 = 50% 輪動 + 50% 線D（回測2015年起 +15.0% / 最大跌幅 -20.3%；2000-2009 最大跌幅 -22%）", "",
             "*核心部分*", table(["輪動", "SPY訊號", "SPY"]),
             "輪動 = 美股/納指/海外股/長債/黃金，月尾高過10個月平均先揸，每樣20%（2026-10-07 開始）", "",
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
    month = f"{today:%Y-%m}"
    score_rows = [[month, lab] + [_fmt(stats[c].get(key), kind) if c in stats else "" for c in SV.SCORE_HEADER[2:]]
                  for key, lab, kind in SCORE_ROWS]
    SV.replace_group("每月成績表", SV.SCORE_HEADER, 1, [month], score_rows, dry)
    if not dry:
        bot.send_telegram(text)


# ---------------- 換倉 ----------------
def main():
    force, dry = "--force" in sys.argv, "--dry" in sys.argv
    today = PR.market_day()                 # 最近一個已收市交易日（排程遲咗過半夜都計返啱嗰日）
    done_path = "paper_log/mm_done.txt"     # 排程每晚有後備時間：做咗就唔再做，避免重複發訊息
    if "--scheduled" in sys.argv and os.path.exists(done_path) and open(done_path).read().strip() == str(today):
        print(f"{today} 已經做咗，唔再跑。")
        return
    if "--sheet-only" in sys.argv:          # 只寫 Google Sheet（唔發 Telegram、唔換倉），日期 = 最近一個交易日
        d = [a.split("=")[1] for a in sys.argv if a.startswith("--date=") and len(a) > 7]
        day = pd.Timestamp(d[0]).date() if d else today
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        ok = SV.available()
        os.makedirs("paper_log", exist_ok=True)
        with open("paper_log/sheet_v2_status.txt", "w") as f:
            f.write(f"{datetime.now()} Apps Script 新格式：{'有' if ok else '冇（未部署新版本？）'}\n")
        print(f"Sheet v2：{'有' if ok else '冇'}；日期 {day}")
        daily_status(day, dry, send=False)
        return
    want_board = "--scoreboard" in sys.argv or is_last_trading_day(today)
    if "--scoreboard-only" in sys.argv:
        monthly_scoreboard(today, dry)
        return
    due = list(TRACKS) if force else tracks_due(today)
    only = [a.split("=")[1] for a in sys.argv if a.startswith("--tracks=")]
    if only:
        due = [t for t in due if t in only[0].split(",")]
    ok = daily_status(today, dry) if not due else rebalance(today, due, force, dry)
    if ok and want_board:
        monthly_scoreboard(today, dry)
    if ok and not dry:
        os.makedirs("paper_log", exist_ok=True)
        with open(done_path, "w") as f:
            f.write(str(today))


def rebalance(today, due, force, dry):
    tickers, names = current_sp500()
    closes = M.download_close(tickers, (pd.Timestamp(today) - pd.Timedelta(days=500)).strftime("%Y-%m-%d"),
                              (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    px = pd.DataFrame(closes).sort_index()
    px = px[(px.index.dayofweek < 5) & (px.index <= pd.Timestamp(today))]
    # 逐隻檢查有冇今日收市價（2026-10-08 修正）
    uni = [t for t in tickers if t in px.columns]
    px, miss = PR.ensure_fresh(px, uni, today, rounds=2)
    if len(miss) > 0.5 * max(len(uni), 1):
        print(f"ℹ️ 動量模擬：{today} 未有收市數據（可能休市，或者 Yahoo 未更新），遲啲再試。")
        return False
    if len(miss) > 0.02 * len(uni) and not PR.last_chance():
        print(f"動量模擬：{today} 仲有 {len(miss)} 隻未有收市價，等後備時間再跑。")
        return False
    stale_note = ""
    if miss:                                   # 少數仲未有：排名用返上一日價，並標明
        prev = px[px.index < pd.Timestamp(today)].ffill().iloc[-1]
        for t in miss:
            px.loc[pd.Timestamp(today), t] = prev.get(t, np.nan)
        stale_note = f"⚠️ {len(miss)} 隻未有今日收市價，排名用咗上一日價：{'、'.join(miss[:15])}"
    last_day = today
    i = len(px) - 1
    sc = scores_at(px, i)
    df = bot.replay(bot.add_signals(bot.load_data()))
    spy_hold = df["state"].iloc[-1] != "out"
    logs = read_logs()
    logs = logs[logs["date"] < str(last_day)]

    lines = [f"📅 *動量換倉（模擬交易）* {last_day}", f"每條線模擬本金 ${CAPITAL:,.0f}"] + ([stale_note] if stale_note else []) + [""]
    rows = []
    trade_rows = []

    def g12(t):
        s_ = px[t].dropna() if t in px.columns else pd.Series(dtype=float)
        return round(float(s_.iloc[-1] / s_.iloc[-253] - 1) * 100, 1) if len(s_) >= 253 else ""
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
        prev_day = pd.Timestamp(lt["date"].max()) if not lt.empty else None
        for t in top:
            p = float(px[t].iloc[i])
            act = "繼續持有" if t in prev_top else "建議買入（下一個交易日收市）"
            trade_rows.append([str(last_day), f"線{tr}", act, t, names.get(t, ""), round(p, 2), round(per / p, 4), round(per, 2),
                               "", g12(t), f"第{top.index(t) + 1}名；" + (f"動量 {sc['原始'].get(t, np.nan) * 100:+.0f}%"
                                                                     if cfg["score"] == "原始" else f"綜合排名分 {ranked[t]:.2f}")])
        for t in drop:
            p = float(px[t].iloc[i]) if t in px.columns else np.nan
            ret = ""
            if prev_day is not None and t in px.columns:
                after = px[t][px.index > prev_day].dropna()
                if len(after):
                    ret = round((p / float(after.iloc[0]) - 1) * 100, 2)
            trade_rows.append([str(last_day), f"線{tr}", "建議賣出（下一個交易日收市）", t, names.get(t, ""),
                               round(p, 2) if np.isfinite(p) else "", "", "", ret, g12(t), f"跌出頭{n}名"])
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
    SV.upsert("買賣紀錄", SV.TRADE_HEADER, [r + [f"{r[0]}|{r[1]}|{r[3]}|{r[2]}"] for r in trade_rows], dry)
    if not dry:
        bot.send_telegram(text)
    return True


if __name__ == "__main__":
    main()
