import pandas as pd
import yfinance as yf

from strategy_core import compute_indicators, check_breakout_signal, MEGA_CAP_TICKERS, HIGH_BETA_TICKERS

# ==========================================
# 第一步：股票池（已搬去strategy_core.py，兩個檔共用同一份，唔會再漂移）
# ==========================================
# ⚠️ 已知限制（未解決）：呢份名單用「依家」仲存在、仲算優質嘅公司揀出嚟，
# 但backtest卻用嚟測2023年開始嘅歷史——即係倖存者偏差(survivorship bias)，
# 凡係2023-2026年間已經除牌/淡出視野嘅股票，根本唔會出現喺呢個名單，
# 令勝率有機會系統性偏高。建議將呢份backtest嘅勝率當做「樂觀上限」。

TICKER_GROUPS = {
    "MEGA_CAP (傳統巨無霸與穩健增長)": MEGA_CAP_TICKERS,
    "HIGH_BETA (高動能與投機性科技股)": HIGH_BETA_TICKERS,
}

START_DATE = "2023-01-01"
END_DATE = "2026-09-30"

HOLD_DAYS = 20
STOP_LOSS_PCT = -0.07
TARGET_PCT = 0.10
BREAKEVEN_PCT = 0.05


def get_spy_data():
    try:
        spy = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy["SPY_SMA50"] = spy["Close"].rolling(50).mean()
        return spy["Close"] > spy["SPY_SMA50"]
    except Exception as e:
        print(f"⚠️ SPY數據下載失敗（{e}）")
        return pd.Series(dtype=bool)


def prepare_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if df.empty or len(df) < 210:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return compute_indicators(df)  # 改用strategy_core嘅共用function，同main.py保證一致


def simulate_trades(df, ticker, spy_filter):
    trades = []
    i, n = 0, len(df)

    while i < n - HOLD_DAYS - 1:
        date_idx = df.index[i]
        spy_bullish = spy_filter.get(date_idx, True)

        row = df.iloc[i]
        if check_breakout_signal(row, spy_bullish):  # 共用function，同main.py保證一致
            entry_price = df["Open"].iloc[i + 1]
            entry_date = df.index[i + 1]
            exit_price, exit_date, exit_reason = None, None, "持有到期"

            current_stop_price = entry_price * (1 + STOP_LOSS_PCT)
            target_price = entry_price * (1 + TARGET_PCT)
            breakeven_trigger = entry_price * (1 + BREAKEVEN_PCT)

            for j in range(i + 1, min(i + 1 + HOLD_DAYS, n)):
                day_open = df["Open"].iloc[j]
                day_high = df["High"].iloc[j]
                day_low = df["Low"].iloc[j]

                stop_hit = day_low <= current_stop_price
                target_hit = day_high >= target_price

                if stop_hit and target_hit:
                    exit_price = min(day_open, current_stop_price)
                    exit_date, exit_reason = df.index[j], "同日雙穿(保守判定止損)"
                    break
                elif stop_hit:
                    exit_price = min(day_open, current_stop_price)
                    reason_text = "保本離場" if current_stop_price == entry_price else "觸發止損"
                    exit_date, exit_reason = df.index[j], reason_text
                    break
                elif target_hit:
                    exit_price = target_price
                    exit_date, exit_reason = df.index[j], "觸發止盈"
                    break

                if day_high >= breakeven_trigger and current_stop_price < entry_price:
                    current_stop_price = entry_price

            if exit_price is None:
                last_j = min(i + HOLD_DAYS, n - 1)
                exit_price, exit_date = df["Close"].iloc[last_j], df.index[last_j]

            ret_pct = (exit_price - entry_price) / entry_price
            trades.append({
                "ticker": ticker,
                "entry_date": entry_date.date(),
                "entry_price": round(float(entry_price), 2),
                "exit_date": exit_date.date(),
                "exit_price": round(float(exit_price), 2),
                "return_pct": round(float(ret_pct) * 100, 2),
                "exit_reason": exit_reason,
            })
            i += HOLD_DAYS
        else:
            i += 1

    return trades


def run_backtest():
    print("正在下載大盤 SPY 數據建立環境濾網...")
    spy_filter = get_spy_data()
    if spy_filter.empty:
        print("⚠️ 警告：SPY濾網未能建立")

    all_groups_dataframes = []

    for group_name, tickers in TICKER_GROUPS.items():
        print(f"\n======================================")
        print(f"準備掃描 {group_name} 組別的 {len(tickers)} 隻美股...")
        print(f"======================================")

        all_trades = []
        for t in tickers:
            print(f"處理緊 {t}...")
            df = prepare_data(t)
            if df is None:
                continue
            trades = simulate_trades(df, t, spy_filter)
            if trades:
                print(f" {t} 觸發咗 {len(trades)} 次訊號")
            all_trades.extend(trades)

        if not all_trades:
            print(f"\n{group_name} 完全冇觸發任何交易訊號...")
            summary_data = [{"ticker": f"=== {group_name} 回測表現總結 ==="}, {"ticker": "無交易紀錄"}]
            for d in summary_data:
                for k in ["entry_date", "entry_price", "exit_date", "exit_price", "return_pct", "exit_reason"]:
                    if k not in d:
                        d[k] = ""
            all_groups_dataframes.append(pd.DataFrame(summary_data))
            all_groups_dataframes.append(pd.DataFrame(
                [{"ticker": "", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""} for _ in range(4)]
            ))
            continue

        result_df = pd.DataFrame(all_trades)

        total_trades = len(result_df)
        win_trades = result_df[result_df["return_pct"] > 0]
        lose_trades = result_df[result_df["return_pct"] < 0]
        be_trades = result_df[result_df["return_pct"] == 0]

        wins, losses, bes = len(win_trades), len(lose_trades), len(be_trades)
        strict_win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        adj_win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
        avg_win = win_trades["return_pct"].mean() if wins > 0 else 0
        avg_loss = lose_trades["return_pct"].mean() if losses > 0 else 0
        expectancy = (strict_win_rate / 100 * avg_win) + ((losses / total_trades) * avg_loss) if total_trades > 0 else 0

        reason_counts = result_df["exit_reason"].value_counts()
        reason_lines = [{"ticker": f" {reason}：{count} 次"} for reason, count in reason_counts.items()]

        summary_data = [
            {"ticker": f"=== {group_name} 回測表現總結 ==="},
            {"ticker": f"總交易次數：{total_trades} 次"},
            {"ticker": f"按實際盈虧：贏 {wins} 次、輸 {losses} 次、打和 {bes} 次"},
            {"ticker": f"嚴格勝率：{strict_win_rate:.2f}%（若撇除打和，真實勝率約為 {adj_win_rate:.2f}%）"},
            {"ticker": f"平均贏單利潤：+{avg_win:.2f}%"},
            {"ticker": f"平均輸單虧損：{avg_loss:.2f}%"},
            {"ticker": f"期望值（每注平均）：{expectancy:.2f}%"},
            {"ticker": ""},
            {"ticker": "出場原因分布："},
            *reason_lines,
            {"ticker": ""},
            {"ticker": f"--- {group_name} 詳細交易紀錄 ---"},
        ]
        for d in summary_data:
            for k in ["entry_date", "entry_price", "exit_date", "exit_price", "return_pct", "exit_reason"]:
                if k not in d:
                    d[k] = ""

        summary_df = pd.DataFrame(summary_data)
        group_final_df = pd.concat([summary_df, result_df], ignore_index=True)
        all_groups_dataframes.append(group_final_df)

        empty_lines = pd.DataFrame(
            [{"ticker": "", "entry_date": "", "entry_price": "", "exit_date": "", "exit_price": "", "return_pct": "", "exit_reason": ""} for _ in range(5)]
        )
        all_groups_dataframes.append(empty_lines)

    final_combined_df = pd.concat(all_groups_dataframes, ignore_index=True)
    final_combined_df.to_csv("backtest_results.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 雙組別詳細交易紀錄已整合並存到 backtest_results.csv")


if __name__ == "__main__":
    run_backtest()
