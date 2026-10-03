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

# ==========================================
# 第一步：股票池與環境設定
# ==========================================
WATCHLIST = MEGA_CAP_TICKERS + HIGH_BETA_TICKERS

START_DATE = "2023-01-01"
END_DATE = "2026-09-30"

HOLD_DAYS = 40
STOP_LOSS_PCT = -0.05
BREAKEVEN_PCT = 0.08

# 🚀 資金管理參數 —— 本金 $80,000，冇槓桿
INITIAL_CAPITAL = 80000.0
POSITION_SIZE_PCT = 0.20  # 每注動用「總資產」嘅20%，但受限於實際可用現金


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
    return compute_indicators(df)


# ==========================================
# 第二步：三引擎同步模擬交易（邏輯不變，只係訊號產生，唔處理資金）
# ==========================================
def simulate_trades(df, ticker, spy_filter):
    trades = []
    i, n = 0, len(df)

    while i < n - HOLD_DAYS - 1:
        date_idx = df.index[i]
        spy_bullish = spy_filter.get(date_idx, True)
        row = df.iloc[i]
        df_history_up_to_today = df.iloc[: i + 1]

        strategy = None
        if check_breakout_signal(row, spy_bullish):
            strategy = "Breakout (突破)"
        elif check_pullback_signal(row, spy_bullish, df_history_up_to_today):
            strategy = "Pullback (回踩)"
        elif check_dip_buying_signal(row, df_history_up_to_today):
            strategy = "Dip_Buy (撈底)"

        if strategy:
            entry_price = df["Open"].iloc[i + 1]
            entry_date = df.index[i + 1]
            exit_price, exit_date, exit_reason = None, None, "持有到期"

            current_stop_price = entry_price * (1 + STOP_LOSS_PCT)
            target_price = entry_price * (1 + 0.20)
            breakeven_trigger = entry_price * (1 + BREAKEVEN_PCT)
            time_stop_d, time_stop_tgt = 15, 0.00

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
                    exit_date, exit_reason = df.index[j], "觸發止盈(+20%)"
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
                "ticker": ticker, "group": group_tag, "strategy": strategy,
                "entry_date": entry_date, "entry_price": round(float(entry_price), 2),
                "exit_date": exit_date, "exit_price": round(float(exit_price), 2),
                "return_pct": round(float(ret_pct) * 100, 2),
                "holding_days": holding_days_calendar, "exit_reason": exit_reason,
            })
            i += HOLD_DAYS
        else:
            i += 1

    return trades


# ==========================================
# 第三步：事件驅動、有資金上限嘅組合模擬（取代原本有漏洞嘅序列式複利）
# ==========================================
def simulate_portfolio_with_capital_limit(result_df, initial_capital, position_size_pct):
    """
    冇槓桿嘅資金曲線模擬：
    - 將每注交易拆做「入場」同「離場」兩個事件，按日期行過去
    - 入場嗰刻，check返實際可用現金夠唔夠呢注應佔嘅資金（總資產 × position_size_pct）
    - 唔夠現金 → 呢注直接跳過（唔開倉），絕對唔會透支
    - 持倉中嘅部位，帳面值維持喺「入場時撥款金額」，離場先實現盈虧
      （簡化：唔做持倉期間嘅逐日mark-to-market，呢個唔影響「有冇用盡資金」呢個核心問題）
    """
    # 同一日有多過一個訊號爭緊同一舊資金時，優先俾返歷史期望值較高嘅引擎，
    # 唔再淨係跟入面原本嘅先後次序（呢個純粹係資金分配優先序，唔改任何訊號判斷邏輯）
    strategy_priority = {"Breakout (突破)": 0, "Dip_Buy (撈底)": 1, "Pullback (回踩)": 2}

    events = []
    for idx, row in result_df.iterrows():
        prio = strategy_priority.get(row.get("strategy"), 99)
        events.append((row["entry_date"], 1, prio, idx))  # 1=entry
        events.append((row["exit_date"], 0, -1, idx))      # 0=exit，同日排喺entry之前
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    events = [(d, t, idx) for d, t, _, idx in events]

    cash = initial_capital
    open_positions = {}  # idx -> 撥款金額
    taken = {}
    skipped_count = 0
    equity_log = []  # (date, total_equity) 用嚟畫資金曲線/debug

    for date, etype, idx in events:
        if etype == 0:  # 離場：實現盈虧，資金歸還現金
            if idx in open_positions:
                allocated = open_positions.pop(idx)
                ret_pct = result_df.at[idx, "return_pct"]
                cash += allocated * (1 + ret_pct / 100.0)
        else:  # 入場：check現金夠唔夠
            total_equity = cash + sum(open_positions.values())
            desired_size = total_equity * position_size_pct
            if desired_size <= cash + 1e-6:
                cash -= desired_size
                open_positions[idx] = desired_size
                taken[idx] = True
            else:
                taken[idx] = False
                skipped_count += 1
        equity_log.append((date, cash + sum(open_positions.values())))

    final_equity = cash + sum(open_positions.values())
    result_df = result_df.copy()
    result_df["position_taken"] = result_df.index.map(lambda i: taken.get(i, False))
    return result_df, final_equity, skipped_count, equity_log


# ==========================================
# 第四步：跑回測與生成報告
# ==========================================
def run_backtest():
    print("正在下載大盤 SPY 數據建立環境濾網...")
    spy_filter = get_spy_data()
    if spy_filter.empty:
        print("⚠️ 警告：SPY濾網未能建立")

    print(f"\n準備掃描 {len(WATCHLIST)} 隻股票 (Mega Cap + High Beta)...")

    all_trades = []
    for t in WATCHLIST:
        print(f"處理緊 {t}...")
        df = prepare_data(t)
        if df is None:
            continue
        trades = simulate_trades(df, t, spy_filter)
        if trades:
            print(f" {t} 觸發咗 {len(trades)} 次訊號")
        all_trades.extend(trades)

    if not all_trades:
        print("\n完全冇觸發任何交易訊號...")
        return

    result_df = pd.DataFrame(all_trades)
    result_df["entry_date"] = pd.to_datetime(result_df["entry_date"])
    result_df["exit_date"] = pd.to_datetime(result_df["exit_date"])
    result_df = result_df.sort_values("entry_date").reset_index(drop=True)

    # --- 理論勝率/期望值：假設每個訊號都做到（唔受資金限制），反映策略本身嘅質素 ---
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
    avg_holding_days = result_df["holding_days"].mean() if total_trades > 0 else 0

    # --- 實際資金曲線：有$80,000上限、冇槓桿、資金唔夠就跳過 ---
    result_df, final_capital, skipped_count, _ = simulate_portfolio_with_capital_limit(
        result_df, INITIAL_CAPITAL, POSITION_SIZE_PCT
    )
    taken_count = total_trades - skipped_count

    total_return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    start_dt, end_dt = result_df["entry_date"].min(), result_df["exit_date"].max()
    years_invested = (end_dt - start_dt).days / 365.25
    annualized_return_pct = (
        ((final_capital / INITIAL_CAPITAL) ** (1 / years_invested) - 1) * 100
        if years_invested > 0 else total_return_pct
    )

    result_df["entry_date"] = result_df["entry_date"].dt.strftime("%Y-%m-%d")
    result_df["exit_date"] = result_df["exit_date"].dt.strftime("%Y-%m-%d")

    summary_data = [
        {"ticker": "=== 回測報告（有資金上限、冇槓桿版本）==="},
        {"ticker": f"模擬起始本金：${INITIAL_CAPITAL:,.2f}（每注目標動用總資產嘅{POSITION_SIZE_PCT*100:.0f}%，但唔會透支）"},
        {"ticker": f"模擬最終結餘：${final_capital:,.2f}"},
        {"ticker": f"總體回報率：+{total_return_pct:,.2f}%" if total_return_pct >= 0 else f"總體回報率：{total_return_pct:,.2f}%"},
        {"ticker": f"年化回報率(CAGR)：+{annualized_return_pct:,.2f}%（歷時{years_invested:.1f}年）" if annualized_return_pct >= 0 else f"年化回報率(CAGR)：{annualized_return_pct:,.2f}%（歷時{years_invested:.1f}年）"},
        {"ticker": ""},
        {"ticker": f"系統總共發出 {total_trades} 次訊號，當中 {taken_count} 次實際開到倉，{skipped_count} 次因資金唔夠被跳過"},
        {"ticker": f"（資金使用率：{taken_count/total_trades*100:.1f}%——呢個比例愈低，代表$80,000呢個本金愈係跟唔上訊號流量，是真實嘅資金瓶頸）"},
        {"ticker": ""},
        {"ticker": "--- 下面係「理論」勝率/期望值，假設每個訊號都做到（唔受資金限制） ---"},
        {"ticker": f"總交易次數：{total_trades} 次"},
        {"ticker": f"按實際盈虧：贏 {wins} 次、輸 {losses} 次、打和 {bes} 次"},
        {"ticker": f"嚴格勝率：{strict_win_rate:.2f}%（若撇除打和，真實勝率約為 {adj_win_rate:.2f}%）"},
        {"ticker": f"平均贏單利潤：+{avg_win:.2f}%"},
        {"ticker": f"平均輸單虧損：{avg_loss:.2f}%"},
        {"ticker": f"期望值（每注平均）：{expectancy:.2f}%"},
        {"ticker": f"平均持股日數：{avg_holding_days:.1f} 日"},
        {"ticker": ""},
        {"ticker": "--- 詳細交易紀錄（position_taken=False 代表呢注因資金不足未能實際開倉）---"},
    ]

    for d in summary_data:
        for k in ["group", "strategy", "entry_date", "entry_price", "exit_date", "exit_price",
                  "return_pct", "holding_days", "exit_reason", "position_taken"]:
            if k not in d:
                d[k] = ""

    summary_df = pd.DataFrame(summary_data)
    final_combined_df = pd.concat([summary_df, result_df], ignore_index=True)
    cols = ["ticker", "group", "strategy", "entry_date", "exit_date", "entry_price", "exit_price",
            "return_pct", "holding_days", "exit_reason", "position_taken"]
    final_combined_df = final_combined_df[cols]

    final_combined_df.to_csv("backtest_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 已完成（有資金上限版本），最終結餘 ${final_capital:,.2f}，"
          f"{taken_count}/{total_trades} 注實際開到倉。已存到 backtest_results.csv")


if __name__ == "__main__":
    run_backtest()
