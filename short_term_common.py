"""
short_term_common.py — 短炒回測共用工具（跳空策略、Short Squeeze策略都用）

- 下載數據（yfinance，價格已按拆股調整，冇按股息調整；短炒影響好細）
- 每個信號計7種賣出方法 × 2種買入時間
- $10,000 組合模擬：最多同時揸3隻，每隻用當時總資產三分一；同一日信號太多就揀成交量倍數最大嗰啲
- 成本：每次買或賣0.15%（IB佣金 + 跳空股買賣差價，偏保守）
"""
import io
import numpy as np
import pandas as pd
import requests

COST = 0.0015
SLOTS = 3
CAPITAL = 10_000
MAX_HOLD = 20

EXITS = ["揸1日", "揸3日", "揸5日", "揸10日", "揸20日",
         "跌穿信號日低位止損（最多20日）", "跌穿信號日低位止損 + 升10%止賺（最多20日）"]
ENTRIES = ["信號日收市買", "第二日開市買"]


def sp_universe():
    """今日嘅標普500 + 標普400成分股（⚠️ 有倖存者偏差）。"""
    hdr = {"User-Agent": "Mozilla/5.0"}
    out = []
    for url in ["https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"]:
        html = requests.get(url, headers=hdr, timeout=30).text
        for t in pd.read_html(io.StringIO(html)):
            col = [c for c in t.columns if str(c).lower() in ("symbol", "ticker symbol", "ticker")]
            if col:
                out += [str(s).replace(".", "-").strip() for s in t[col[0]].dropna()]
                break
    return sorted(set(out))


def download(tickers, start, end, chunk=100, _retry=True):
    import time
    import yfinance as yf
    data = {}
    for i in range(0, len(tickers), chunk):
        time.sleep(1)
        part = tickers[i:i + chunk]
        try:
            df = yf.download(part, start=start, end=end, auto_adjust=False, group_by="ticker",
                             threads=True, progress=False)
        except Exception as e:
            print(f"下載失敗 {part[:3]}...: {e}")
            continue
        for t in part:
            try:
                d = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) > 250:
                    data[t] = d.astype(float)
            except Exception:
                pass
        print(f"已下載 {min(i + chunk, len(tickers))}/{len(tickers)}，有效 {len(data)}")
    missing = [t for t in tickers if t not in data]
    if _retry and missing:
        print(f"再試一次 {len(missing)}隻")
        time.sleep(20)
        data.update(download(missing, start, end, chunk=50, _retry=False))
    return data


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def base_features(d):
    f = pd.DataFrame(index=d.index)
    c = d["Close"]
    f["close"], f["open"], f["low"] = c, d["Open"], d["Low"]
    f["gap"] = d["Open"] / c.shift(1) - 1
    for n in (20, 50, 200):
        f[f"sma{n}"] = c.rolling(n).mean()
    f["avgvol"] = d["Volume"].rolling(63).mean().shift(1)     # 過去3個月平均（唔計當日）
    f["vol"] = d["Volume"]
    f["relvol"] = d["Volume"] / f["avgvol"]
    f["rsi"] = rsi(c)
    return f


def trade_outcomes(d, i, entry_mode):
    """信號喺第i日（收市後知道）。回傳 {賣出方法: (入場日, 出場日, 回報)}；數據唔夠就略過。"""
    o, c, lo = d["Open"].values, d["Close"].values, d["Low"].values
    idx = d.index
    if entry_mode == ENTRIES[0]:
        e_i, px = i, c[i]
    else:
        if i + 1 >= len(c):
            return {}
        e_i, px = i + 1, o[i + 1]
    stop = lo[i]
    res = {}
    for ex in EXITS:
        if ex.startswith("揸"):
            n = int(ex[1:-1])
            x_i = i + n
            if x_i >= len(c) or x_i < e_i:
                continue
        else:
            x_i = None
            last = min(i + MAX_HOLD, len(c) - 1)
            if last <= e_i:
                continue
            for k in range(e_i if entry_mode == ENTRIES[1] else e_i + 1, last + 1):
                if k == e_i and entry_mode == ENTRIES[0]:
                    continue
                if c[k] < stop or ("止賺" in ex and c[k] >= px * 1.10):
                    x_i = k
                    break
            if x_i is None:
                x_i = last
        r = (c[x_i] / px) * (1 - COST) / (1 + COST) - 1
        res[ex] = (idx[e_i], idx[x_i], r)
    return res


def simulate_portfolio(trades, prices, start, end):
    """trades：DataFrame（ticker, signal_date, entry_date, exit_date, ret, relvol）。
    每日按收市價計總資產。回傳每日總資產Series。"""
    t = trades[(trades["entry_date"] >= pd.Timestamp(start)) & (trades["entry_date"] <= pd.Timestamp(end))]
    t = t.sort_values(["entry_date", "relvol"], ascending=[True, False])
    by_entry = {k: g for k, g in t.groupby("entry_date")}
    days = pd.DatetimeIndex(sorted(set().union(*[set(p.index) for p in prices.values()])))
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    cash, open_pos, curve = float(CAPITAL), [], []
    for day in days:
        # 先處理今日出場
        still = []
        for p in open_pos:
            if p["exit_date"] == day:
                cash += p["value_at_entry"] * (1 + p["ret"])
            else:
                still.append(p)
        open_pos = still
        # 今日入場
        if day in by_entry:
            for _, tr in by_entry[day].iterrows():
                if len(open_pos) >= SLOTS:
                    break
                if any(p["ticker"] == tr["ticker"] for p in open_pos):
                    continue
                equity = cash + sum(_mtm(p, prices, day) for p in open_pos)
                size = min(cash, equity / SLOTS)
                if size <= 1:
                    break
                cash -= size
                open_pos.append({"ticker": tr["ticker"], "exit_date": tr["exit_date"], "ret": tr["ret"],
                                 "value_at_entry": size, "entry_px": tr["entry_px"]})
        # 同一日買入又止損嘅（第二日開市買、收市跌穿低位）
        still = []
        for p in open_pos:
            if p["exit_date"] == day:
                cash += p["value_at_entry"] * (1 + p["ret"])
            else:
                still.append(p)
        open_pos = still
        curve.append(cash + sum(_mtm(p, prices, day) for p in open_pos))
    return pd.Series(curve, index=days)


def _mtm(p, prices, day):
    s = prices[p["ticker"]]["Close"]
    px = s.get(day, np.nan)
    if np.isnan(px):
        px = s[:day].iloc[-1] if len(s[:day]) else p["entry_px"]
    return p["value_at_entry"] * px / p["entry_px"]


def curve_stats(curve):
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    v = np.concatenate([[CAPITAL], curve.values])
    cagr = ((curve.iloc[-1] / CAPITAL) ** (1 / yrs) - 1) * 100
    mdd = (v / np.maximum.accumulate(v) - 1).min() * 100
    return cagr, mdd, curve.iloc[-1]


def build_trades(signals, prices, spy_close):
    """signals：list of (ticker, signal_index, relvol)。每個信號計晒所有買入×賣出組合。"""
    rows = []
    for t, i, rv in signals:
        d = prices[t]
        for em in ENTRIES:
            for ex, (ed, xd, r) in trade_outcomes(d, i, em).items():
                epx = d["Close"].iloc[i] if em == ENTRIES[0] else d["Open"].iloc[i + 1]
                s0 = spy_close.asof(ed) if em == ENTRIES[0] else spy_close.asof(d.index[i])
                spy_r = spy_close.asof(xd) / s0 - 1
                rows.append({"ticker": t, "signal_date": d.index[i], "entry": em, "exit": ex,
                             "entry_date": ed, "exit_date": xd, "ret": r, "spy_ret": spy_r,
                             "relvol": rv, "entry_px": epx})
    return pd.DataFrame(rows)


def report(trades, prices, spy_close, periods, title, notes):
    L = [title, *notes, ""]
    L.append("=== 每筆交易（未計組合，只計每筆平均）===")
    L.append("　買入 | 賣出 | 交易數 | 賺錢比例 | 平均回報 | 中位數 | 同期SPY平均 | 平均贏SPY")
    for (em, ex), g in trades.groupby(["entry", "exit"], sort=False):
        L.append(f"　{em} | {ex} | {len(g)} | {(g['ret'] > 0).mean() * 100:.0f}% | {g['ret'].mean() * 100:+.2f}% | "
                 f"{g['ret'].median() * 100:+.2f}% | {g['spy_ret'].mean() * 100:+.2f}% | "
                 f"{(g['ret'] - g['spy_ret']).mean() * 100:+.2f}%")
    L.append("")
    L.append(f"=== $10,000組合（最多同時{SLOTS}隻）===")
    rows = []
    for (em, ex), g in trades.groupby(["entry", "exit"], sort=False):
        row = {"買入": em, "賣出": ex}
        for pn, a, b in periods:
            cv = simulate_portfolio(g, prices, a, b)
            cg, dd, fin = curve_stats(cv)
            row[f"{pn}回報%"], row[f"{pn}跌幅%"], row[f"{pn}最後$"] = cg, dd, fin
        rows.append(row)
    res = pd.DataFrame(rows)
    spy_line = []
    for pn, a, b in periods:
        s = spy_close[(spy_close.index >= pd.Timestamp(a)) & (spy_close.index <= pd.Timestamp(b))]
        cv = s / s.iloc[0] * CAPITAL
        cg, dd, fin = curve_stats(cv)
        spy_line.append((pn, cg, dd, fin))
    L.append("　比較：SPY長揸（價格，未計息）：" + " | ".join(f"{pn} {cg:+.2f}%（跌 {dd:.1f}%，${fin:,.0f}）" for pn, cg, dd, fin in spy_line))
    for _, r in res.iterrows():
        L.append(f"　{r['買入']} | {r['賣出']} | " + " | ".join(
            f"{pn} {r[f'{pn}回報%']:+.2f}%（跌 {r[f'{pn}跌幅%']:.1f}%，${r[f'{pn}最後$']:,.0f}）" for pn, _, _ in periods))
    L.append("")
    first, second = periods[1][0], periods[2][0]
    ok = res[res[f"{first}跌幅%"] > -35]
    L.append(f"=== 誠實測試：用{first}揀（跌幅好過-35%入面回報最高），原封不動睇{second} ===")
    if len(ok):
        p = ok.sort_values(f"{first}回報%", ascending=False).iloc[0]
        rank = int((res[f"{second}回報%"] > p[f"{second}回報%"]).sum()) + 1
        spy2 = [x for x in spy_line if x[0] == second][0]
        L.append(f"　{first}揀中：{p['買入']} | {p['賣出']}（{first} {p[f'{first}回報%']:+.2f}%，跌 {p[f'{first}跌幅%']:.1f}%）")
        L.append(f"　→ {second}：{p[f'{second}回報%']:+.2f}%（跌 {p[f'{second}跌幅%']:.1f}%），排第{rank}/{len(res)}；"
                 f"SPY同期 {spy2[1]:+.2f}%（跌 {spy2[2]:.1f}%）")
        beat = p[f"{second}回報%"] > spy2[1]
        L.append(f"　結論：{'✅ 後半段仍然贏SPY' if beat else '❌ 後半段輸畀SPY'}")
    else:
        L.append("　冇任何組合前半段跌幅好過-35%")
    return L, res
