"""
backtest_v2.py — v2 策略（Minervini趨勢模板 + RS + 突破）2010-2026 逐日組合模擬

同舊回測嘅分別：
- 逐日模擬：每日先處理出場、再處理入場、收市後先揀聽日嘅訊號；資產每日按收市價計（真實回撤）。
- 注碼按風險計（見 strategy_v2.position_value），唔係固定25%。
- 同日訊號按RS評級由高至低排隊，唔再係隨機次序。
- 平倉之後即刻可以再出訊號（冇40日鎖定）。
- 計交易成本（每邊0.10%）。
- 規則全部喺 strategy_v2.py 預先登記，呢個file淨係負責模擬同報告，唔調參數。

報告分期：2010-2017（從未用過）、2018-2022、2023-2026/09（舊系統用過，但v2規則冇喺上面調過）。

=== 第一次（亦係唯一一次）結果：2026-10-04，run 37192594694 ===
- 全期2010-2026/09：v2 CAGR +4.41%、最大回撤-39.3%（2021-01→2023-10，2024-11先返回高位）；
  SPY CAGR +14.17%、最大回撤-33.7%。$80,000 → v2 $164,625 vs SPY $735,402。
- 2010-2017（從未用過）：v2 +1.78% vs SPY +13.85%；2018-22：+5.25% vs +9.31%；2023-26：+9.05% vs +21.73%。
- 1272單，勝率34.4%，每單期望值+0.58%，Profit Factor 1.15；交易成本拖累約每年2.2%。
- 全部利潤嚟自10單（+$98,195，大部分係High Beta：NIO、RIOT、PLUG、SMCI），其餘1262單合共-$19,181。
  Mega Cap 931單 Profit Factor 1.07（接近零優勢）。High Beta名單係事後揀嘅，呢部分利潤有倖存者偏差。
- 結論：Minervini趨勢模板嘅「技術部分」機械化之後，2010-2026跑輸SPY好遠。規則已預先登記，唔會再改。
"""
from collections import Counter

import numpy as np
import pandas as pd
import yfinance as yf

import strategy_v2 as S

DOWNLOAD_START = "2008-01-01"       # 預留約2年暖身（52週、RS要252日）
TEST_START = "2010-01-01"
END = "2026-10-01"                  # yfinance的end唔包當日 → 數據到2026-09-30
INITIAL_CAPITAL = 80000.0
PERIODS = [
    ("2010-2017（從未用過嘅數據）", "2010-01-01", "2017-12-31"),
    ("2018-2022（舊系統用過）", "2018-01-01", "2022-12-31"),
    ("2023-2026/09（舊系統用過）", "2023-01-01", "2026-09-30"),
    ("全期 2010-2026/09", "2010-01-01", "2026-09-30"),
]
FIELDS = ["Open", "High", "Low", "Close", "Volume", "SMA50", "SMA150", "SMA200", "SMA200_prev",
          "Low52w", "High52w", "ATR", "VolAvg50", "Pivot", "RSRaw"]
RESULTS_FILE = "v2_results.csv"
EQUITY_FILE = "v2_equity.csv"


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_one(t):
    try:
        df = flatten(yf.download(t, start=DOWNLOAD_START, end=END, auto_adjust=True, progress=False))
        return df.dropna(how="all")
    except Exception:
        return None


def download_all(tickers):
    data = {}
    for k in range(0, len(tickers), 40):
        chunk = tickers[k:k + 40]
        print(f"下載 {k + 1}-{k + len(chunk)} / {len(tickers)} ...")
        try:
            batch = yf.download(chunk, start=DOWNLOAD_START, end=END, group_by="ticker",
                                auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print(f"  批次下載失敗（{e}），逐隻再試")
            batch = None
        for t in chunk:
            df = None
            if batch is not None and isinstance(batch.columns, pd.MultiIndex) \
                    and t in batch.columns.get_level_values(0):
                df = batch[t].dropna(how="all")
            if df is None or df.empty:
                df = download_one(t)
            if df is not None and len(df) >= 260 and {"Open", "High", "Low", "Close", "Volume"} <= set(df.columns):
                data[t] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return data


def max_drawdown(values):
    peak = np.maximum.accumulate(values)
    return float((values / peak - 1).min() * 100)


def trade_stats(tr):
    if len(tr) == 0:
        return "0單"
    wins, losses = tr[tr["pnl"] > 0], tr[tr["pnl"] <= 0]
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) and losses["pnl"].sum() != 0 else float("inf")
    return (f"{len(tr)}單，勝率{len(wins) / len(tr) * 100:.1f}%，平均贏{wins['ret_pct'].mean() if len(wins) else 0:+.2f}%，"
            f"平均輸{losses['ret_pct'].mean() if len(losses) else 0:+.2f}%，每單期望值{tr['ret_pct'].mean():+.2f}%，"
            f"Profit Factor {pf:.2f}，平均R {tr['r_mult'].mean():+.2f}")


def main():
    universe = S.v2_universe()
    group_of = dict(universe)
    tickers = [t for t, _ in universe]

    print("下載SPY ...")
    spy = flatten(yf.download("SPY", start=DOWNLOAD_START, end=END, auto_adjust=True, progress=False)).dropna(how="all")
    idx = spy.index
    spy_close = spy["Close"].astype(float)
    spy_ok = (spy_close > spy_close.rolling(S.MARKET_SMA).mean()).values

    raw = download_all(tickers)
    print(f"有足夠數據嘅股票：{len(raw)}/{len(tickers)}")
    cols = list(raw.keys())
    ind = {t: S.add_indicators(raw[t]).reindex(idx) for t in cols}
    P = {f: pd.DataFrame({t: ind[t][f] for t in cols}, index=idx) for f in FIELDS}
    rs = S.rs_rating(P["RSRaw"])
    sig = (S.trend_template(P, rs) & S.breakout(P)).fillna(False).values.astype(bool) & spy_ok[:, None]

    O, H, L, C = (P[f].values for f in ("Open", "High", "Low", "Close"))
    ATR, PIV, RS = P["ATR"].values, P["Pivot"].values, rs.values
    n, m = C.shape
    start_i = int(idx.searchsorted(pd.Timestamp(TEST_START)))

    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    eq = np.full(n, np.nan)
    expo = np.zeros(n)
    npos = np.zeros(n)
    last_close = np.full(m, np.nan)
    if start_i > 0:
        prev = C[start_i - 1]
        last_close[~np.isnan(prev)] = prev[~np.isnan(prev)]
    pending = []
    skips = Counter()
    signal_days = 0

    def close_position(j, px, i, reason):
        nonlocal cash
        p = positions.pop(j)
        proceeds = p["shares"] * px
        cost_out = proceeds * S.COST_PER_SIDE
        cash += proceeds - cost_out
        invested = p["shares"] * p["entry"] + p["cost_in"]
        pnl = proceeds - cost_out - invested
        trades.append({
            "ticker": cols[j], "group": group_of[cols[j]], "entry_date": idx[p["entry_i"]].date(),
            "exit_date": idx[i].date(), "entry_price": round(p["entry"], 4), "exit_price": round(px, 4),
            "init_stop": round(p["init_stop"], 4), "ret_pct": pnl / invested * 100, "pnl": pnl,
            "r_mult": (px - p["entry"]) / (p["entry"] - p["init_stop"]),
            "rs_at_signal": round(p["rs"], 1),
            "rs_at_exit": round(float(RS[i, j]), 1) if not np.isnan(RS[i, j]) else np.nan,  # 只係記錄，唔影響任何規則
            "days_held": i - p["entry_i"], "exit_reason": reason,
            "position_pct": p["pos_pct"] * 100,
        })

    def trail(j, i):
        p = positions[j]
        if not np.isnan(H[i, j]):
            p["hh"] = max(p["hh"], H[i, j])
        if not np.isnan(ATR[i, j]):
            ch = p["hh"] - S.TRAIL_ATR * ATR[i, j]
            if ch > p["stop"]:
                p["stop"], p["trailing"] = ch, True

    for i in range(start_i, n):
        ok = ~np.isnan(C[i])
        last_close[ok] = C[i][ok]

        # 1) 出場：用昨日收市後定好嘅止損位
        for j in list(positions):
            p = positions[j]
            if np.isnan(O[i, j]) or np.isnan(L[i, j]):
                continue
            label = "浮動止損" if p["trailing"] else "初始止損"
            if O[i, j] <= p["stop"]:
                close_position(j, O[i, j], i, label + "(跳空)")
            elif L[i, j] <= p["stop"]:
                close_position(j, p["stop"], i, label)
            else:
                trail(j, i)

        # 2) 入場：昨日收市出嘅訊號，按RS高至低
        equity_prev = eq[i - 1] if i > start_i else INITIAL_CAPITAL
        new_today = 0
        for j, rsv, pivot, atr in pending:
            if j in positions:
                continue
            if len(positions) >= S.MAX_POSITIONS:
                skips["已持有10個倉位"] += 1
                continue
            if new_today >= S.MAX_NEW_PER_DAY:
                skips["當日已開3個新倉"] += 1
                continue
            entry = O[i, j]
            if np.isnan(entry) or np.isnan(atr) or np.isnan(pivot):
                skips["冇數據"] += 1
                continue
            if entry > pivot * S.MAX_CHASE:
                skips["開市高過突破位5%，唔追"] += 1
                continue
            stop = S.initial_stop(entry, atr)
            value = S.position_value(equity_prev, entry, stop)
            cost_in = value * S.COST_PER_SIDE
            if value <= 0 or value + cost_in > cash:
                skips["現金唔夠"] += 1
                continue
            cash -= value + cost_in
            positions[j] = {"shares": value / entry, "entry": entry, "stop": stop, "init_stop": stop,
                            "hh": entry, "entry_i": i, "rs": rsv, "trailing": False, "cost_in": cost_in,
                            "pos_pct": value / equity_prev}
            new_today += 1
            if L[i, j] <= stop:                       # 入場當日已經跌穿止損
                close_position(j, stop, i, "初始止損(入場當日)")
            else:
                trail(j, i)

        # 3) 收市後揀聽日嘅候選：排除已持有，按RS由高至低
        cand = [j for j in np.where(sig[i])[0] if j not in positions]
        if cand:
            signal_days += 1
        cand.sort(key=lambda j: -RS[i, j])
        pending = [(j, RS[i, j], PIV[i, j], ATR[i, j]) for j in cand]

        # 4) 按市值計資產
        pos_val = sum(p["shares"] * last_close[j] for j, p in positions.items())
        eq[i] = cash + pos_val
        expo[i] = pos_val / eq[i]
        npos[i] = len(positions)

    # 期末未平倉：按最後收市價列出（唔計入已平倉統計）
    open_rows = []
    unrealized = 0.0
    for j, p in positions.items():
        px = last_close[j]
        unrealized += p["shares"] * px - (p["shares"] * p["entry"] + p["cost_in"])
        open_rows.append(f"　{cols[j]}：{idx[p['entry_i']].date()} 入場 ${p['entry']:.2f}，"
                         f"而家 ${px:.2f}（{(px / p['entry'] - 1) * 100:+.1f}%）")
    # 會計核對：期末資產 = 本金 + 已實現盈虧 + 未實現盈虧
    realized = sum(t["pnl"] for t in trades)
    diff = eq[n - 1] - (INITIAL_CAPITAL + realized + unrealized)
    accounting = f"會計核對：期末資產 − (本金+已實現+未實現) = {diff:.6f}（{'OK' if abs(diff) < 0.01 else '⚠️ 唔對數'}）"

    dates = idx[start_i:]
    strat = pd.Series(eq[start_i:], index=dates)
    spy_base = float(spy_close.iloc[start_i - 1]) if start_i > 0 else float(spy_close.iloc[0])
    bench = INITIAL_CAPITAL * spy_close.iloc[start_i:] / spy_base
    expo_s = pd.Series(expo[start_i:], index=dates)
    tr = pd.DataFrame(trades)
    if not tr.empty:
        tr["exit_date"] = pd.to_datetime(tr["exit_date"])

    def period_block(name, a, b):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        out = [f"--- {name} ---"]
        for label, s in (("v2策略", strat), ("SPY買入持有", bench)):
            seg = s[(s.index >= a) & (s.index <= b)]
            before = s[s.index < a]
            base = float(before.iloc[-1]) if len(before) else INITIAL_CAPITAL
            base_date = before.index[-1] if len(before) else dates[0] - pd.Timedelta(days=1)
            yrs = (seg.index[-1] - base_date).days / 365.25
            cagr = ((float(seg.iloc[-1]) / base) ** (1 / yrs) - 1) * 100
            mdd = max_drawdown(np.concatenate([[base], seg.values]))
            out.append(f"　{label}：CAGR {cagr:+.2f}%，最大回撤 {mdd:.1f}%，期末 ${float(seg.iloc[-1]):,.0f}")
        seg_e = expo_s[(expo_s.index >= a) & (expo_s.index <= b)]
        out.append(f"　平均資金使用率：{seg_e.mean() * 100:.0f}%")
        if not tr.empty:
            ptr = tr[(tr["exit_date"] >= a) & (tr["exit_date"] <= b)]
            out.append("　已平倉交易：" + trade_stats(ptr))
        return out

    lines = [
        "=== v2 回測：Minervini趨勢模板 + RS評級 + 突破入場（規則預先登記，見strategy_v2.py）===",
        f"本金${INITIAL_CAPITAL:,.0f}，冇槓桿，{TEST_START}起連續跑；每日按收市價計資產；每邊交易成本{S.COST_PER_SIDE * 100:.2f}%",
        f"股票池：{len(cols)}隻有數據（已剔除ETF）；大市濾網SPY>200日線；每單風險{S.RISK_PER_TRADE * 100:.2f}%，"
        f"單一倉位上限{S.MAX_POSITION_PCT * 100:.0f}%，最多{S.MAX_POSITIONS}個倉位，每日最多{S.MAX_NEW_PER_DAY}個新倉",
        "⚠️ 倖存者偏差：用今日嘅股票名單，愈早年份偏差愈大，結果要當偏樂觀上限。冇基本面、冇VCP偵測。",
        accounting,
        "",
    ]
    for name, a, b in PERIODS:
        lines += period_block(name, a, b) + [""]

    lines.append("=== 逐年回報（v2 vs SPY；2026係到9月底）===")
    for y in sorted(set(dates.year)):
        s_y, b_y = strat[strat.index.year == y], bench[bench.index.year == y]
        s_prev, b_prev = strat[strat.index.year < y], bench[bench.index.year < y]
        s0 = float(s_prev.iloc[-1]) if len(s_prev) else INITIAL_CAPITAL
        b0 = float(b_prev.iloc[-1]) if len(b_prev) else INITIAL_CAPITAL
        e_y = expo_s[expo_s.index.year == y].mean() * 100
        lines.append(f"　{y}：v2 {(float(s_y.iloc[-1]) / s0 - 1) * 100:+.2f}% | SPY {(float(b_y.iloc[-1]) / b0 - 1) * 100:+.2f}%"
                     f" | 平均資金使用率{e_y:.0f}%")
    lines.append("")

    if not tr.empty:
        lines.append("=== 分組（已平倉）===")
        for g in ("Mega Cap", "High Beta"):
            lines.append(f"　{g}：" + trade_stats(tr[tr["group"] == g]))
        lines.append("=== 出場原因 ===")
        for reason, cnt in tr["exit_reason"].value_counts().items():
            sub = tr[tr["exit_reason"] == reason]
            lines.append(f"　{reason}：{cnt}單，平均{sub['ret_pct'].mean():+.2f}%")
        top = tr.sort_values("pnl", ascending=False).head(5)
        lines.append("=== 賺得最多嘅5單 ===")
        for r in top.itertuples():
            lines.append(f"　{r.ticker} {r.entry_date}→{r.exit_date.date()}：{r.ret_pct:+.1f}%（{r.r_mult:+.1f}R）")
        lines.append("")
    lines.append(f"=== 訊號被跳過嘅原因（全期，共{signal_days}日有訊號）===")
    for k, v in skips.most_common():
        lines.append(f"　{k}：{v}次")
    lines.append(f"=== 期末未平倉（{len(open_rows)}個）===")
    lines += open_rows
    lines += ["", "--- 詳細交易紀錄（已平倉，ret_pct已扣交易成本）---"]

    cols_out = ["ticker", "group", "entry_date", "exit_date", "entry_price", "exit_price", "init_stop",
                "ret_pct", "pnl", "r_mult", "rs_at_signal", "rs_at_exit", "days_held", "position_pct", "exit_reason"]
    summary = pd.DataFrame({"ticker": lines})
    for c in cols_out[1:]:
        summary[c] = ""
    detail = tr[cols_out].copy() if not tr.empty else pd.DataFrame(columns=cols_out)
    if not detail.empty:
        detail["exit_date"] = detail["exit_date"].dt.date
        for c in ("ret_pct", "pnl", "r_mult", "position_pct"):
            detail[c] = detail[c].round(3)
    pd.concat([summary, detail], ignore_index=True).to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": dates.date, "v2_equity": strat.values.round(2), "spy_equity": bench.values.round(2),
                  "exposure_pct": (expo_s.values * 100).round(1), "positions": npos[start_i:].astype(int)}
                 ).to_csv(EQUITY_FILE, index=False)
    print("\n".join(lines))
    print(f"\n✅ 已存到 {RESULTS_FILE} 同 {EQUITY_FILE}")


if __name__ == "__main__":
    main()
