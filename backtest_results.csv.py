import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

from strategy_core import (
    compute_indicators, 
    check_breakout_signal, 
    check_pullback_signal, 
    check_dip_buying_signal, 
    MEGA_CAP_TICKERS,
    HIGH_BETA_TICKERS
)

WATCHLIST = MEGA_CAP_TICKERS + HIGH_BETA_TICKERS
START_DATE = "2023-01-01"
END_DATE = "2026-09-30"

# ==========================================
# 🚀 基礎量化風控參數 (高頻暴利模式)
# ==========================================
HOLD_DAYS = 15          # 大幅縮短持倉上限，拒絕資金站崗
STOP_LOSS_PCT = -0.06   # 微幅放寬止損空間，容忍洗盤
BREAKEVEN_PCT = 0.06    # 更快啟動保本機制

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
    if df.empty or len(df) < 210: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return compute_indicators(df)

def simulate_trades(df, ticker, spy_filter):
    trades = []
    i, n = 0, len(df)

    while i < n - HOLD_DAYS - 1:
        date_idx = df.index[i]
        spy_bullish = spy_filter.get(date_idx, True)
        row = df.iloc[i]
        df_history_up_to_today = df.iloc[:i+1] 

        strategy = None
        target_pct = 0.10
        time_stop_d = 15
        time_stop_tgt = 0.00

        # 動態分配各策略專屬的獲利目標與時間止損
        if check_breakout_signal(row, spy_bullish):
            strategy = "Breakout (突破)"
            target_pct = 0.15          # 15% 快速收割
            time_stop_d = 8            # 8 天不發動就砍
            time_stop_tgt = 0.02
        elif check_pullback_signal(row, spy_bullish, df_history_up_to_today):
            strategy = "Pullback (回踩)"
            target_pct = 0.12          # 12% 快速收割
            time_stop_d = 6            # 6 天不反彈就砍
            time_stop_tgt = 0.00
        elif check_dip_buying_signal(row, df_history_up_to_today):
            strategy = "Dip_Buy (撈底)"
            target_pct = 0.15
            time_stop_d = 10
            time_stop_tgt = 0.00

        if strategy:
            entry_price = df["Open"].iloc[i + 1]
            entry_date = df.index[i + 1]
            exit_price, exit_date, exit_reason = None, None, "持有到期"

            current_stop_price = entry_price * (1 + STOP_LOSS_PCT)
            target_price = entry_price * (1 + target_pct)  # 套用獨立止盈目標
            breakeven_trigger = entry_price * (1 + BREAKEVEN_PCT)

            for j in range(i + 1, min(i + 1 + HOLD_DAYS, n)):
                day_open = df["Open"].iloc[j]
                day_high = df["High"].iloc[j]
                day_low = df["Low"].iloc[j]
                day_close = df["Close"].iloc[j]

                trading_days_held = j - (i + 1)
                stop_hit = day_low <= current_stop_price
                target_hit = day_high >= target_price

                if stop_hit and target_hit:
                    exit_price = min(day_open, current_stop_price)
                    exit_date, exit_reason = df.index[j], "同日雙穿(保守判定止損)"
                    break
                elif stop_hit:
                    exit_price = min(day_open, current_stop_price)
                    reason_text = "保本離場" if current_stop_price == entry_price else f"觸發止損({STOP_LOSS_PCT*100:.0f}%)"
                    exit_date, exit_reason = df.index[j], reason_text
                    break
                elif target_hit:
                    exit_price = target_price
                    exit_date, exit_reason = df.index[j], f"觸發止盈(+{target_pct*100:.0f}%)"
                    break

                if trading_days_held == time_stop_d:
                    current_return = (day_close - entry_price) / entry_price
                    if current_return < time_stop_tgt:
                        exit_price = day_close
                        exit_date, exit_reason = df.index[j], f"時間止損(未達+{time_stop_tgt*100:.0f}%)"
                        break

                if day_high >= breakeven_trigger and current_stop_price < entry_price:
                    current_stop_price = entry_price

            if exit_price is None:
                last_j = min(i + HOLD_DAYS, n - 1)
                exit_price, exit_date = df["Close"].iloc[last_j], df.index[last_j]

            ret_pct = (exit_price - entry_price) / entry_price
            holding_days_calendar = (exit_date - entry_date).days
            group_tag = "High Beta" if ticker in HIGH_BETA_TICKERS else "Mega Cap"

            trades.append({
                "ticker": ticker,
                "group": group_tag,
                "strategy": strategy,
                "entry_date": entry_date.date(),
                "entry_price": round(float(entry_price), 2),
                "exit_date": exit_date.date(),
                "exit_price": round(float(exit_price), 2),
                "return_pct": round(float(ret_pct) * 100, 2),
                "holding_days": holding_days_calendar,
                "exit_reason": exit_reason,
            })
            i += HOLD_DAYS
        else:
            i += 1
    return trades

def run_backtest():
    print("正在下載大盤 SPY 數據建立環境濾網...")
    spy_filter = get_spy_data()
    if spy_filter.empty: print("⚠️ 警告：SPY濾網未能建立")

    print(f"\n======================================")
    print(f"準備掃描 {len(WATCHLIST)} 隻股票 (Mega Cap + High Beta)...")
    print(f"======================================")

    all_trades = []
    for t in WATCHLIST:
        print(f"處理緊 {t}...")
        df = prepare_data(t)
        if df is None: continue
        trades = simulate_trades(df, t, spy_filter)
        if trades: print(f" {t} 觸發咗 {len(trades)} 次訊號")
        all_trades.extend(trades)

    if not all_trades:
        print("\n完全冇觸發任何交易訊號...")
        return

    result_df = pd.DataFrame(all_trades)

    # 💰 綜合資金曲線模擬
    result_df['entry_date'] = pd.to_datetime(result_df['entry_date'])
    result_df['exit_date'] = pd.to_datetime(result_df['exit_date'])
    result_df = result_df.sort_values('entry_date').reset_index(drop=True)

    initial_capital = 80000.0
    current_capital = initial_capital
    position_size_pct = 0.20  

    equity_curve = []
    for idx, row in result_df.iterrows():
        trade_profit = (current_capital * position_size_pct) * (row['return_pct'] / 100.0)
        current_capital += trade_profit
        equity_curve.append(current_capital)

    result_df['portfolio_balance'] = [round(x, 2) for x in equity_curve]

    final_capital = current_capital
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100

    start_dt = result_df['entry_date'].min()
    end_dt = result_df['exit_date'].max()
    days_invested = (end_dt - start_dt).days
    years_invested = days_invested / 365.25
    annualized_return_pct = ((final_capital / initial_capital) ** (1 / years_invested) - 1) * 100 if years_invested > 0 else total_return_pct

    # 📊 各策略獨立複利與表現拆解
    strat_summary = []
    for strat in ["Breakout (突破)", "Pullback (回踩)", "Dip_Buy (撈底)"]:
        strat_df = result_df[result_df["strategy"] == strat].sort_values('entry_date')
        if len(strat_df) > 0:
            s_capital = initial_capital
            for _, r in strat_df.iterrows():
                s_capital += (s_capital * position_size_pct) * (r['return_pct'] / 100.0)

            s_wins = len(strat_df[strat_df["return_pct"] > 0])
            s_losses = len(strat_df[strat_df["return_pct"] < 0])
            s_total = len(strat_df)
            s_rate = (s_wins / (s_wins + s_losses)) * 100 if (s_wins + s_losses) > 0 else 0

            s_avg_win = strat_df[strat_df["return_pct"] > 0]["return_pct"].mean() if s_wins > 0 else 0
            s_avg_loss = strat_df[strat_df["return_pct"] < 0]["return_pct"].mean() if s_losses > 0 else 0
            s_net_profit = s_capital - initial_capital

            strat_summary.append({"ticker": f" {strat}: 淨賺 ${s_net_profit:,.0f} (最終資金 ${s_capital:,.0f}) | 觸發 {s_total}次 | 勝率 {s_rate:.1f}% | 均贏 +{s_avg_win:.2f}% | 均輸 {s_avg_loss:.2f}%"})
        else:
            strat_summary.append({"ticker": f" {strat}: 無觸發紀錄"})

    total_trades = len(result_df)
    win_trades = result_df[result_df["return_pct"] > 0]
    lose_trades = result_df[result_df["return_pct"] < 0]
    bes = len(result_df[result_df["return_pct"] == 0])

    wins, losses = len(win_trades), len(lose_trades)
    strict_win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    avg_win = win_trades["return_pct"].mean() if wins > 0 else 0
    avg_loss = lose_trades["return_pct"].mean() if losses > 0 else 0

    reason_counts = result_df["exit_reason"].value_counts()
    reason_lines = [{"ticker": f" {reason}：{count} 次"} for reason, count in reason_counts.items()]

    result_df['entry_date'] = result_df['entry_date'].dt.strftime('%Y-%m-%d')
    result_df['exit_date'] = result_df['exit_date'].dt.strftime('%Y-%m-%d')

    summary_data = [
        {"ticker": f"=== 全天候暴利印鈔機 (高頻游擊戰) 終極報告 ==="},
        {"ticker": f"🔥 模擬起始本金：${initial_capital:,.2f} (每注固定動用 20% 資金)"},
        {"ticker": f"🔥 模擬最終結餘：${final_capital:,.2f}"},
        {"ticker": f"📈 總體回報率 (Total Return)：+{total_return_pct:,.2f}%"},
        {"ticker": f"📈 年化回報率 (CAGR)：+{annualized_return_pct:,.2f}% (歷時 {years_invested:.1f} 年)"},
        {"ticker": ""},
        {"ticker": "👑 獨立策略表現排行榜 (假設獨立以 $80k 運行)："},
        *strat_summary,
        {"ticker": ""},
        {"ticker": f"總交易次數：{total_trades} 次 | 贏 {wins} 次、輸 {losses} 次、打和 {bes} 次"},
        {"ticker": f"嚴格勝率：{strict_win_rate:.2f}% | 平均贏單：+{avg_win:.2f}% | 平均輸單：{avg_loss:.2f}%"},
        {"ticker": ""},
        {"ticker": "出場原因分布："},
        *reason_lines,
        {"ticker": ""},
        {"ticker": f"--- 詳細交易紀錄 (依時間排序，展示綜合資金曲線) ---"},
    ]

    for d in summary_data:
        for k in ["group", "strategy", "entry_date", "entry_price", "exit_date", "exit_price", "return_pct", "holding_days", "exit_reason", "portfolio_balance"]:
            if k not in d: d[k] = ""

    summary_df = pd.DataFrame(summary_data)
    final_combined_df = pd.concat([summary_df, result_df], ignore_index=True)

    cols = ['ticker', 'group', 'strategy', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'return_pct', 'holding_days', 'exit_reason', 'portfolio_balance']
    final_combined_df = final_combined_df[cols]

final_combined_df.to_csv("backtest_results.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ 資金複利曲線與回報率已整合，已存到 backtest_results.csv")

if __name__ == "__main__":
    run_backtest()
