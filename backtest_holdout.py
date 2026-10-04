"""
backtest_holdout.py
2018-2022 出樣本外 holdout 測試（只跑一次，唔調參數）

目的：2023-2026呢段數據已經用嚟揀參數同驗證好多次，開始「用到殘」。
2018-2022我哋從來未睇過，而且包括2018年尾大跌、2020疫情崩盤同反彈、
2021大牛市、2022全年熊市——係真正嘅出樣本外測試。

規矩（跑之前已經鎖定，唔會因為結果再返去調）：
- 進場/出場規則：strategy_core.py 現有default（Pullback volume_drying_max=0.35、
  Dip_Buy剔除High Beta，其餘全部原設定）；出場規則同backtest_screener.py一樣
  （持有40日、-5%停損、+8%保本、+20%後浮動止賺回落1%、15日時間止損）。
- 倉位：
  1. Mega-Cap-only，單一25%（目前最穩陣嘅production候選）
  2. 全部股票池，單一25%
  3. 全部股票池，分級 base 32.5%（2023-24訓練期最新揀出嘅base）
  4. Mega-Cap-only，分級 base 30%（同上）
- 本金$80,000，冇槓桿，由2018-01-01連續跑到2022年尾（唔係每年重新起跑）。

已知偏差（結果要當「偏樂觀嘅上限」嚟睇）：
- 倖存者偏差：MEGA_CAP_TICKERS係今日嘅大藍籌名單，攞返去2018年用等於事後揀咗贏家。
- High Beta入面好多係近年先上市，早年冇數據（報告會列出覆蓋率）。
- 資金曲線用「已實現」計（持倉中按成本計），所以每年回報同最大回撤都係近似值，
  跨年持倉嘅盈虧計入平倉嗰年。

=== 結果（2026-10-04 跑咗一次，run 37189818824；呢段數據而家已經「用咗」）===
- 4個設定全部跑輸SPY買入持有(CAGR +9.31%)：Mega-Cap單一25% +1.67%、全部單一25% +1.35%、
  全部分級32.5% -1.11%、Mega分級30% +0.14%。即使有倖存者偏差幫手都係咁。
- 訊號質素同2023-26差好遠：Breakout 127次 勝率28.3% 期望值+0.02%；Dip_Buy 140次 勝率19.3%
  期望值-1.52%；Pullback 30次 勝率36.7% 期望值+0.52%。
- 防守有啲用：2022 Mega25 -6.53% vs SPY -18.18%（SPY 50日線濾網關咗64%交易日）。
  但升市大幅落後：2019 +7.54% vs +31.22%、2020 +0.85% vs +18.33%、2021 +7.23% vs +28.73%。
- 最大問題：Dip_Buy冇大市濾網，喺大跌市「成批觸發」——2020-03-05一日24個訊號全部止損(平均-5.85%)；
  一日>=5個Dip_Buy訊號嘅日子共51次訊號，平均-4.41%，其他日子+0.14%。28單跳空穿止損(最差-10.75%)。
- 回測方法本身：同一日多個訊號爭資金時，邊個攞到資金取決於dataframe行次序(冇意義嘅隨機性)。
  將同日次序打亂200次，Mega25 CAGR由-1.02%到+5.93%(中位數+2.31%)——最好情況都跑輸SPY。
  呢個隨機性亦解釋到之前run與run之間嘅部分波動同分級倉位嘅不穩定。
- 結論：2023-26嘅+17~20%好大程度係牛市環境+參數調整嘅產物，未有證據顯示策略跨市況有優勢。
  任何因為呢個結果而做嘅改動(例如Dip_Buy加大市濾網)都唔可以再用2018-22驗證，要用新數據。
"""
import pandas as pd
import yfinance as yf

import backtest_screener as bs
from strategy_core import MEGA_CAP_TICKERS, HIGH_BETA_TICKERS, get_tiered_position_size_pct

DOWNLOAD_START = "2017-01-01"   # 提早一年下載，俾SMA200/52週高位暖身
HOLDOUT_START = "2018-01-01"
HOLDOUT_END = "2022-12-31"
DOWNLOAD_END = "2023-03-31"     # 2022年尾開嘅倉要有數據行完持有期
YEARS = [2018, 2019, 2020, 2021, 2022]

INITIAL_CAPITAL = bs.INITIAL_CAPITAL
LOCKED_CONFIGS = [
    ("Mega-Cap-only，單一25%（production候選）", "mega", 0.25),
    ("全部股票池，單一25%", "full", 0.25),
    ("全部股票池，分級base 32.5%", "full", ("tier", 0.325)),
    ("Mega-Cap-only，分級base 30%", "mega", ("tier", 0.30)),
]
OUTPUT_FILE = "holdout_results.csv"


def get_spy(start, end):
    spy = yf.download("SPY", start=start, end=end, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    close = spy["Close"]
    bullish = close > close.rolling(50).mean()
    return close, bullish


def size_arg(spec):
    if isinstance(spec, tuple) and spec[0] == "tier":
        base = spec[1]
        return lambda strat, grp, b=base: get_tiered_position_size_pct(strat, grp, base=b)
    return spec


def yearly_equity(equity_log, initial):
    if not equity_log:
        return {y: initial for y in YEARS}, initial
    s = pd.Series([e for _, e in equity_log], index=pd.to_datetime([d for d, _ in equity_log]))
    s = s.groupby(level=0).last()
    ends = {}
    for y in YEARS:
        upto = s[s.index <= pd.Timestamp(f"{y}-12-31")]
        ends[y] = float(upto.iloc[-1]) if len(upto) else initial
    return ends, float(s.iloc[-1])


def max_drawdown(equity_log, initial):
    vals = [initial] + [e for _, e in equity_log]
    peak, mdd = vals[0], 0.0
    for v in vals:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return mdd * 100


def run_config(trades, label, universe, spec, spy_years):
    df = trades if universe == "full" else trades[trades["group"] == "Mega Cap"]
    df = df.reset_index(drop=True)
    lines = [f"=== {label} ==="]
    if df.empty:
        lines.append("⚠️ 冇交易訊號")
        return lines, None
    sim_df, final_eq, skipped, equity_log = bs.simulate_portfolio_with_capital_limit(
        df.copy(), INITIAL_CAPITAL, size_arg(spec))
    last_exit = max(df["exit_date"].max(), pd.Timestamp(HOLDOUT_END))
    years = (last_exit - pd.Timestamp(HOLDOUT_START)).days / 365.25
    cagr = ((final_eq / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    mdd = max_drawdown(equity_log, INITIAL_CAPITAL)
    taken = len(df) - skipped
    lines.append(f"👉 2018-01-01起連續跑：CAGR {cagr:+.2f}%，最終結餘${final_eq:,.2f}，"
                 f"開倉{taken}/{len(df)}次，最大回撤(已實現計) {mdd:.2f}%")
    ends, _ = yearly_equity(equity_log, INITIAL_CAPITAL)
    prev = INITIAL_CAPITAL
    lines.append("逐年（策略 vs SPY買入持有）：")
    for y in YEARS:
        ret = (ends[y] / prev - 1) * 100
        sim_y = sim_df[sim_df["entry_date"].dt.year == y]
        lines.append(f"　{y}年：策略 {ret:+.2f}%（年尾結餘${ends[y]:,.2f}，當年開倉{int(sim_y['position_taken'].sum())}次）"
                     f" | SPY {spy_years.get(y, float('nan')):+.2f}%")
        prev = ends[y]
    tail = (final_eq / ends[YEARS[-1]] - 1) * 100
    if abs(tail) > 0.005:
        lines.append(f"　2023年初（2022年尾開倉、2023先平倉）：{tail:+.2f}%")
    return lines, sim_df


def main():
    print(f"下載SPY（{DOWNLOAD_START} ~ {DOWNLOAD_END}）...")
    spy_close, spy_bullish = get_spy(DOWNLOAD_START, DOWNLOAD_END)
    spy_years = {}
    for y in YEARS:
        yc = spy_close[spy_close.index.year == y]
        prev_c = spy_close[spy_close.index < pd.Timestamp(f"{y}-01-01")]
        if len(yc) and len(prev_c):
            spy_years[y] = (float(yc.iloc[-1]) / float(prev_c.iloc[-1]) - 1) * 100
    h = spy_close[(spy_close.index >= pd.Timestamp(HOLDOUT_START)) & (spy_close.index <= pd.Timestamp(HOLDOUT_END))]
    pre = spy_close[spy_close.index < pd.Timestamp(HOLDOUT_START)]
    spy_total = float(h.iloc[-1]) / float(pre.iloc[-1])
    spy_cagr = (spy_total ** (1 / 5) - 1) * 100

    print(f"\n下載 {len(bs.WATCHLIST)} 隻股票...")
    data_cache, missing, late_start = {}, [], []
    for t in bs.WATCHLIST:
        print(f"處理緊 {t}...")
        df = bs.prepare_data(t, start=DOWNLOAD_START, end=DOWNLOAD_END)
        if df is None:
            missing.append(t)
            continue
        data_cache[t] = df
        if df.index[0] > pd.Timestamp("2017-03-31"):
            late_start.append(t)

    all_trades = []
    for t, df in data_cache.items():
        all_trades.extend(bs.simulate_trades(df, t, spy_bullish))
    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])
        trades = trades[(trades["entry_date"] >= pd.Timestamp(HOLDOUT_START)) &
                        (trades["entry_date"] <= pd.Timestamp(HOLDOUT_END))]
        trades = trades.sort_values("entry_date").reset_index(drop=True)

    def cov(group_list):
        have = [t for t in group_list if t in data_cache]
        late = [t for t in have if t in late_start]
        return len(have), len(group_list), late

    m_have, m_total, m_late = cov(MEGA_CAP_TICKERS)
    h_have, h_total, h_late = cov(HIGH_BETA_TICKERS)

    lines = [
        "=== 2018-2022 出樣本外 Holdout 測試（設定跑之前已鎖定，只跑一次，唔會因結果再調參數）===",
        f"本金${INITIAL_CAPITAL:,.0f}，冇槓桿，2018-01-01起連續跑；進出場規則=strategy_core現有default"
        "（Pullback volume_drying_max=0.35、Dip_Buy剔除High Beta）",
        "⚠️ 倖存者偏差：Mega Cap名單係今日嘅大藍籌，攞返去2018年用等於事後揀咗贏家，結果要當偏樂觀上限睇。",
        "⚠️ 資金曲線用已實現計（持倉按成本），每年回報/最大回撤係近似值，跨年持倉計入平倉嗰年。",
        f"數據覆蓋：Mega Cap {m_have}/{m_total}隻有數據（其中{len(m_late)}隻2017年Q1之後先有數據）；"
        f"High Beta {h_have}/{h_total}隻有數據（其中{len(h_late)}隻遲咗先有數據，多數係近年先上市）。",
        f"SPY買入持有 2018-2022：CAGR {spy_cagr:+.2f}%（總回報{(spy_total - 1) * 100:+.2f}%）",
        "",
    ]
    # SPY濾網：每年有幾多日SPY喺50日線之下（Breakout/Pullback呢啲日子唔會出訊號）
    lines.append("SPY 50日線濾網（SPY低過50日線嘅日子，Breakout/Pullback唔會觸發）：")
    for y in YEARS:
        yb = spy_bullish[spy_bullish.index.year == y]
        if len(yb):
            lines.append(f"　{y}年：{(1 - yb.mean()) * 100:.0f}%嘅交易日濾網關閉")
    lines.append("")

    if trades.empty:
        lines.append("⚠️ 2018-2022期間完全冇交易訊號")
        pd.DataFrame({"ticker": lines}).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print("\n".join(lines))
        return

    sims = {}
    for label, universe, spec in LOCKED_CONFIGS:
        cfg_lines, sim_df = run_config(trades, label, universe, spec, spy_years)
        lines += cfg_lines + [""]
        sims[label] = sim_df

    # 逐年 x 策略訊號質素（唔理資金夠唔夠，純粹睇訊號本身）
    lines.append("=== 逐年 x 策略：訊號質素（全部訊號，唔受資金限制）===")
    t2 = trades.copy()
    t2["year"] = t2["entry_date"].dt.year
    for y in YEARS:
        ty = t2[t2["year"] == y]
        parts = []
        for strat in ["Breakout (突破)", "Pullback (回踩)", "Dip_Buy (撈底)"]:
            s = ty[ty["strategy"] == strat]
            if len(s):
                parts.append(f"{strat} {len(s)}次 勝率{(s['return_pct'] > 0).mean() * 100:.0f}% 平均{s['return_pct'].mean():+.2f}%")
            else:
                parts.append(f"{strat} 0次")
        lines.append(f"　{y}年：" + "；".join(parts))
    for strat in ["Breakout (突破)", "Pullback (回踩)", "Dip_Buy (撈底)"]:
        st = bs.theoretical_stats(trades[trades["strategy"] == strat])
        lines.append(f"　5年合計 {strat}：{st['total']}次，勝率{st['strict_wr']:.1f}%，期望值{st['expectancy']:+.2f}%")
    lines += ["", "--- 詳細交易紀錄（全部訊號；taken_mega25/taken_full25 = 喺嗰個設定下有冇資金開倉）---"]

    detail = trades.copy()
    mega_sim = sims[LOCKED_CONFIGS[0][0]]
    full_sim = sims[LOCKED_CONFIGS[1][0]]
    detail["taken_full25"] = full_sim["position_taken"].values if full_sim is not None else False
    mega_taken = set()
    if mega_sim is not None:
        for _, r in mega_sim[mega_sim["position_taken"]].iterrows():
            mega_taken.add((r["ticker"], r["entry_date"]))
    detail["taken_mega25"] = [(r.ticker, r.entry_date) in mega_taken for r in detail.itertuples()]
    detail["year"] = detail["entry_date"].dt.year
    detail["entry_date"] = detail["entry_date"].dt.strftime("%Y-%m-%d")
    detail["exit_date"] = detail["exit_date"].dt.strftime("%Y-%m-%d")

    cols = ["ticker", "group", "strategy", "entry_date", "exit_date", "entry_price", "exit_price",
            "return_pct", "holding_days", "exit_reason", "year", "taken_mega25", "taken_full25"]
    summary_df = pd.DataFrame({"ticker": lines})
    for c in cols[1:]:
        summary_df[c] = ""
    out = pd.concat([summary_df, detail[cols]], ignore_index=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("\n".join(lines))
    print(f"\n✅ 已存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
