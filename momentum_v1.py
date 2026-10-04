"""
momentum_v1.py — 月度動量排名（學術做法）：每月買過去12個月（扣除最近1個月）最強嘅標普500股票

⚠️ 預先登記（跑之前定好，跑完唔改）：

股票池（盡量減少倖存者偏差）
- 每個月用「當時」嘅標普500成分股：首選GitHub公開數據fja05680/sp500（逐日成分股名單）；
  用唔到先用Wikipedia「成分股變動」表由今日名單倒推。
- 已經除牌/被收購而yfinance冇數據嘅股票會缺失 —— 每年報告數據覆蓋率；覆蓋率越低，結果越偏樂觀。

排名同持倉
- 每個月最後一個交易日收市後：計每隻股票「由252個交易日前到21個交易日前」嘅回報（12-1動量，已計股息）。
  要有足夠歷史先參加排名。
- 買排名最高嘅 N 隻，平均分配：N = 10、20、50。
- 下一個交易日收市交易（遲一日）。月中唔再平衡（隨股價浮動）；月中被剔出指數嘅照揸到下次換倉。
- 交易成本：每次買或賣0.1%（按實際換手計）。現金收益：13週國債息。
- 大市濾網（兩個版本都試）：冇濾網 / SPY 200日線緩衝2%（跌穿2%全部轉現金，升返高過2%用最新排名買返）。
  → 共6個組合。

測試期：2005-01至2026-09；前半2005-2015，後半2016-2026/09。
比較對象：
  1. SPY長揸（已計息）
  2. 你而家用緊嘅SPY每日訊號方案（main.py規則）
  3. 當時所有標普500成分股平均分配（每月再平衡）—— 用嚟分清楚「動量」定係「平均分配」嘅效果
揀法：前半段最大跌幅好過-35%入面回報最高，原封不動睇後半段；另外數每年贏SPY幾多年。
輸出：momentum_results.csv、momentum_progress.txt
"""
import io
import time

import numpy as np
import pandas as pd
import requests

START, SPLIT, END = "2005-01-01", "2015-12-31", "2026-09-30"
PERIODS = [("全期", START, END), ("前半", START, SPLIT), ("後半", "2016-01-01", END)]
TOPN = [10, 20, 50]
COST = 0.001
LOOK, SKIP = 252, 21
PROGRESS = "momentum_progress.txt"
HDR = {"User-Agent": "Mozilla/5.0"}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as fh:
        fh.write(line + "\n")


# ---------------- 歷史成分股 ----------------
def norm(t):
    return str(t).strip().upper().replace(".", "-")


def sp500_history():
    """回傳 members(d)：d嗰日嘅標普500成分股。
    首選：fja05680/sp500（GitHub公開數據，逐日列出當時成分股）；後備：Wikipedia變動表倒推。"""
    try:
        api = requests.get("https://api.github.com/repos/fja05680/sp500/contents/", headers=HDR, timeout=30).json()
        import re
        cands = []
        for f in api:
            n = f["name"]
            mm = re.search(r"\((\d{2})-(\d{2})-(\d{4})\)\.csv$", n)
            if n.startswith("S&P 500 Historical Components") and mm:
                cands.append((pd.Timestamp(f"{mm.group(3)}-{mm.group(1)}-{mm.group(2)}"), n))
        name = max(cands)[1]           # 檔名日期最新嗰個（唔好揀冇日期嘅舊檔）
        url = "https://raw.githubusercontent.com/fja05680/sp500/master/" + requests.utils.quote(name)
        df = pd.read_csv(io.StringIO(requests.get(url, headers=HDR, timeout=60).text))
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        snaps = [(d, set(norm(t) for t in str(x).split(","))) for d, x in zip(df["date"], df["tickers"])]
        log(f"成分股來源：fja05680/sp500 {name}，{len(snaps)}個日期，{snaps[0][0].date()} 至 {snaps[-1][0].date()}")
        if snaps[-1][0] < pd.Timestamp("2026-01-01"):
            raise RuntimeError(f"成分股數據只到 {snaps[-1][0].date()}，太舊")
        dates = [d for d, _ in snaps]

        def members(d):
            i = int(np.searchsorted(np.array(dates, dtype="datetime64[ns]"), np.datetime64(d), side="right")) - 1
            return snaps[max(i, 0)][1]
        return members, f"fja05680/sp500（{name}）"
    except Exception as e:
        log(f"fja05680數據用唔到（{e}），改用Wikipedia")
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=HDR, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    for k, t in enumerate(tables):
        log(f"Wikipedia表{k}欄位：{[str(c) for c in t.columns][:8]}")
    current = set(norm(t) for t in tables[0]["Symbol"].dropna())
    ch = None
    for t in tables[1:]:
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        low = [c.lower() for c in cols]
        if any("added" in c for c in low) and any("removed" in c for c in low):
            t.columns = cols
            ch = t
            break
    date_col = [c for c in ch.columns if "date" in c.lower()][0]
    add_col = [c for c in ch.columns if "added" in c.lower() and "ticker" in c.lower()][0]
    rem_col = [c for c in ch.columns if "removed" in c.lower() and "ticker" in c.lower()][0]
    ch = pd.DataFrame({"date": pd.to_datetime(ch[date_col], errors="coerce"),
                       "added": ch[add_col], "removed": ch[rem_col]}).dropna(subset=["date"])
    ch = ch.sort_values("date", ascending=False)
    log(f"Wikipedia：今日成分股{len(current)}隻；變動紀錄{len(ch)}條，最早 {ch['date'].min().date()}")

    def members(d):
        s = set(current)
        for _, r in ch[ch["date"] > d].iterrows():
            if isinstance(r["added"], str) and r["added"].strip():
                s.discard(norm(r["added"]))
            if isinstance(r["removed"], str) and r["removed"].strip():
                s.add(norm(r["removed"]))
        return s
    return members, "Wikipedia變動表倒推"


# ---------------- 數據 ----------------
def download_close(tickers, start, end):
    """一隻一隻慢慢下載（批量下載會被Yahoo限制，之前只攞到一半）。"""
    import yfinance as yf
    out, fails = {}, []
    for n, t in enumerate(tickers):
        for attempt in range(3):
            try:
                h = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
                if h is not None and len(h) > 20:
                    s = h["Close"].astype(float)
                    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
                    out[t] = s[~s.index.duplicated(keep="last")]
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        if t not in out:
            fails.append(t)
        time.sleep(0.3)
        if (n + 1) % 100 == 0:
            log(f"下載進度 {n + 1}/{len(tickers)}，有數據 {len(out)}")
    log(f"下載完成：有數據 {len(out)}/{len(tickers)}；冇數據例子：{fails[:40]}")
    return out


def get_cash(index):
    import yfinance as yf
    irx = yf.download("^IRX", start="2003-06-01", end="2026-10-01", progress=False)["Close"]
    if isinstance(irx, pd.DataFrame):
        irx = irx.iloc[:, 0]
    return irx.reindex(index).ffill().fillna(0.0) / 100 / 252


def spy_bot_curve(spy, cash_r):
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    st = (df["state"] != "out").astype(float).reindex(spy.index).fillna(0.0)
    r = spy.pct_change().fillna(0.0)
    w = st.shift(2).fillna(0.0)
    daily = w * r + (1 - w) * cash_r - w.diff().abs().fillna(0.0) * COST
    return (1 + daily).cumprod()


# ---------------- 模擬 ----------------
def simulate(px, targets, cash_r, filt=None):
    """px：每日收市（欄=股票）；targets：{換倉日: [股票]}（換倉日收市決定，下一日收市交易）；
    filt：每日True=可以持股（同樣遲一日生效）。回傳每日總資產（起步1.0）。"""
    days = px.index
    rets = px.pct_change()
    trade_on = {}
    for d, names in targets.items():
        i = days.searchsorted(d, side="right")
        if i < len(days):
            trade_on[days[i]] = names
    allow = filt.shift(1).reindex(days).fillna(False) if filt is not None else pd.Series(True, index=days)
    hold, cash, latest, vals = {}, 1.0, None, []
    for k, day in enumerate(days):
        if k > 0:
            r = rets.iloc[k]
            for t in list(hold):
                x = r.get(t, np.nan)
                if np.isfinite(x):
                    hold[t] *= 1 + x
            cash *= 1 + cash_r.iloc[k]
        if day in trade_on:
            latest = trade_on[day]
        want = latest if (latest and bool(allow.iloc[k])) else []
        if day in trade_on or set(want) != set(hold):
            total = cash + sum(hold.values())
            new = {t: total / len(want) for t in want} if want else {}
            turnover = sum(abs(new.get(t, 0) - hold.get(t, 0)) for t in set(new) | set(hold))
            total -= turnover * COST
            new = {t: total / len(want) for t in want} if want else {}
            hold, cash = new, (total if not want else 0.0)
        vals.append(cash + sum(hold.values()))
    return pd.Series(vals, index=days)


def stats(eq, a, b):
    s = eq[(eq.index >= pd.Timestamp(a)) & (eq.index <= pd.Timestamp(b))]
    before = eq[eq.index < pd.Timestamp(a)]
    base = float(before.iloc[-1]) if len(before) else float(s.iloc[0])
    yrs = (s.index[-1] - (before.index[-1] if len(before) else s.index[0])).days / 365.25
    v = np.concatenate([[base], s.values])
    cagr = ((s.iloc[-1] / base) ** (1 / yrs) - 1) * 100
    mdd = (v / np.maximum.accumulate(v) - 1).min() * 100
    return cagr, mdd, s.iloc[-1] / base


def yearly(eq):
    s = eq[eq.index >= pd.Timestamp(START)]
    y = s.groupby(s.index.year).last()
    prev = np.concatenate([[eq[eq.index < pd.Timestamp(START)].iloc[-1]], y.values[:-1]])
    return pd.Series(y.values / prev - 1, index=y.index)


def band_filter(spy, n=200, b=0.02):
    ma = spy.rolling(n).mean()
    pos, out = False, []
    for p, m in zip(spy.values, ma.values):
        if np.isnan(m):
            out.append(False)
            continue
        if pos and p < m * (1 - b):
            pos = False
        elif not pos and p > m * (1 + b):
            pos = True
        out.append(pos)
    return pd.Series(out, index=spy.index)


def main():
    open(PROGRESS, "w").close()
    members_fn, source = sp500_history()
    month_ends = pd.date_range("2004-12-31", END, freq="ME")
    members = {d: members_fn(d) for d in month_ends}
    everyone = sorted(set().union(*members.values()))
    log(f"2005-2026曾經係成分股：{len(everyone)}隻")
    closes = download_close(everyone + ["SPY"], "2003-06-01", "2026-10-01")
    spy = closes.pop("SPY")
    px = pd.DataFrame(closes).reindex(spy.index)
    px = px[px.index <= pd.Timestamp(END)]
    spy = spy[spy.index <= pd.Timestamp(END)]
    cash_r = get_cash(px.index)

    # 覆蓋率
    cov = []
    for d in month_ends[::12]:
        m = members[d]
        have = [t for t in m if t in px.columns and px[t].loc[:d].notna().any() and px[t].loc[:d].dropna().index[-1] >= d - pd.Timedelta(days=10)]
        cov.append(f"{d.year}：{len(have)}/{len(m)}（{len(have) / max(len(m), 1) * 100:.0f}%）")
    log("數據覆蓋率（每年1月前月尾）：" + "；".join(cov))

    # 每月排名
    days = px.index
    mom = px.shift(SKIP) / px.shift(LOOK) - 1
    targets = {n: {} for n in TOPN}
    eqw = {}
    for d in month_ends:
        i = days.searchsorted(d, side="right") - 1
        if i < LOOK:
            continue
        day = days[i]
        m = [t for t in members[d] if t in px.columns]
        alive = [t for t in m if np.isfinite(px[t].iloc[i])]
        sc = mom.loc[day, alive].dropna()
        for n in TOPN:
            targets[n][day] = list(sc.sort_values(ascending=False).index[:n])
        eqw[day] = alive
    log(f"換倉月份：{len(eqw)}")

    flt = band_filter(spy)
    curves = {}
    for n in TOPN:
        curves[f"動量前{n}隻（冇濾網）"] = simulate(px, targets[n], cash_r)
        curves[f"動量前{n}隻 + 200日線濾網"] = simulate(px, targets[n], cash_r, flt)
        log(f"完成 前{n}隻")
    bench = {"SPY長揸": spy / spy.iloc[0],
             "全部成分股平均分配": simulate(px, eqw, cash_r)}
    try:
        bench["你而家嘅SPY訊號方案"] = spy_bot_curve(spy, cash_r)
    except Exception as e:
        log(f"SPY訊號方案計唔到：{e}")

    L = ["=== 月度動量排名回測（規則預先登記，見momentum_v1.py開頭）===",
         f"股票池：每月用當時嘅標普500成分股（來源：{source}）；12-1動量；每月換倉；每次買賣0.1%",
         "數據覆蓋率：" + "；".join(cov), ""]
    L.append("=== 比較對象 ===")
    for k, eq in bench.items():
        L.append(f"　{k}：" + " | ".join(f"{pn} {stats(eq, a, b)[0]:+.2f}%（跌 {stats(eq, a, b)[1]:.1f}%）" for pn, a, b in PERIODS)
                 + f" | $10,000 → ${10000 * stats(eq, *PERIODS[0][1:])[2]:,.0f}")
    L.append("")
    L.append("=== 動量組合 ===")
    rows = []
    spy_y = yearly(bench["SPY長揸"])
    for k, eq in curves.items():
        s = {pn: stats(eq, a, b) for pn, a, b in PERIODS}
        yy = yearly(eq)
        win = int((yy > spy_y.reindex(yy.index)).sum())
        rows.append((k, s, win, len(yy)))
        L.append(f"　{k}：" + " | ".join(f"{pn} {s[pn][0]:+.2f}%（跌 {s[pn][1]:.1f}%）" for pn, _, _ in PERIODS)
                 + f" | $10,000 → ${10000 * s['全期'][2]:,.0f} | 贏SPY {win}/{len(yy)}年")
    L.append("")
    ok = [r for r in rows if r[1]["前半"][1] > -35]
    spy_s = {pn: stats(bench["SPY長揸"], a, b) for pn, a, b in PERIODS}
    L.append("=== 誠實測試：前半段（跌幅好過-35%）回報最高嘅組合，原封不動睇後半段 ===")
    if ok:
        p = max(ok, key=lambda r: r[1]["前半"][0])
        rank = sum(1 for r in rows if r[1]["後半"][0] > p[1]["後半"][0]) + 1
        L.append(f"　前半揀中：{p[0]}（前半 {p[1]['前半'][0]:+.2f}%，跌 {p[1]['前半'][1]:.1f}%）")
        L.append(f"　→ 後半：{p[1]['後半'][0]:+.2f}%（跌 {p[1]['後半'][1]:.1f}%），排第{rank}/{len(rows)}；"
                 f"SPY後半 {spy_s['後半'][0]:+.2f}%（跌 {spy_s['後半'][1]:.1f}%）")
        L.append(f"　結論：{'✅ 後半段贏SPY' if p[1]['後半'][0] > spy_s['後半'][0] else '❌ 後半段輸SPY'}")
    else:
        L.append("　前半段冇組合跌幅好過-35%")
    L.append("")
    L.append("=== 逐年回報 ===")
    names = list(bench) + list(curves)
    ys = {k: yearly(v) for k, v in {**bench, **curves}.items()}
    L.append("　年份 | " + " | ".join(names))
    for y in ys["SPY長揸"].index:
        L.append(f"　{y} | " + " | ".join(f"{ys[k].get(y, np.nan) * 100:+.1f}%" for k in names))
    pd.DataFrame({"report": L}).to_csv("momentum_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
