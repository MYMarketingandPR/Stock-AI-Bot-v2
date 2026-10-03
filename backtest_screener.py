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

# 基礎量化風控參數 (暴利模式)
HOLD_DAYS = 40          
STOP_LOSS_PCT = -0.05   
BREAKEVEN_PCT = 0.08    

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

# ==========================================
# 第二步：三引擎同步模擬交易 (暴利模式 20%止盈)
# ==========================================
def simulate_trades(df, ticker, spy_filter):
    trades = []
    i, n = 0, len(df)

    while i < n - HOLD_DAYS - 1:
        date_idx = df.index[i]
        spy_bullish = spy_filter.get(date_idx, True)
        row = df.iloc[i]
        
        df_history_up_to_today = df.iloc[:i+1] 
        
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
            target_price = entry_price * (1 + 0.20)  # 挑戰 20% 止盈
            breakeven_trigger = entry_price * (1 + BREAKEVEN_PCT)

            time_stop_d = 15
            time_stop_tgt = 0.00 

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

# ==========================================
# 第三步：跑回測與生成多維度報告 (加入資金複利及年化回報率模擬)
# ==========================================
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
    
    # 💰 模擬資金曲線
    result_df['entry_date'] = pd.to_datetime(result_df['entry_date'])
    result_df['exit_date'] = pd.to_datetime(result_df['exit_date'])
    result_df = result_df.sort_values('entry_date').reset_index(drop=True)
    
    # 🚀 用戶指定參數：起始資金 $80,000，每注 20%
    initial_capital = 80000.0
    current_capital = initial_capital
    position_size_pct = 0.20  
    
    equity_curve = []
    for idx, row in result_df.iterrows():
        # 每一注的盈虧 = (當前總資金 * 20%) * 該次交易回報率
        trade_profit = (current_capital * position_size_pct) * (row['return_pct'] / 100.0)
        current_capital += trade_profit
        equity_curve.append(current_capital)
        
    result_df['portfolio_balance'] = [round(x, 2) for x in equity_curve]
    
    # 🚀 計算總回報與年化回報 (CAGR)
    final_capital = current_capital
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    
    start_dt = result_df['entry_date'].min()
    end_dt = result_df['exit_date'].max()
    days_invested = (end_dt - start_dt).days
    years_invested = days_invested / 365.25
    
    if years_invested > 0:
        annualized_return_pct = ((final_capital / initial_capital) ** (1 / years_invested) - 1) * 100
    else:
        annualized_return_pct = total_return_pct

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

    # 確保日期變回字串方便存入 CSV
    result_df['entry_date'] = result_df['entry_date'].dt.strftime('%Y-%m-%d')
    result_df['exit_date'] = result_df['exit_date'].dt.strftime('%Y-%m-%d')

    summary_data = [
        {"ticker": f"=== 全天候暴利印鈔機 (20%止盈 / -5%止損) 終極報告 ==="},
        {"ticker": f"🔥 模擬起始本金：${initial_capital:,.2f} (每注固定動用 20% 資金)"},
        {"ticker": f"🔥 模擬最終結餘：${final_capital:,.2f}"},
        {"ticker": f"📈 總體回報率 (Total Return)：+{total_return_pct:,.2f}%"},
        {"ticker": f"📈 年化回報率 (CAGR)：+{annualized_return_pct:,.2f}% (歷時 {years_invested:.1f} 年)"},
        {"ticker": ""},
        {"ticker": f"總交易次數：{total_trades} 次"},
        {"ticker": f"按實際盈虧：贏 {wins} 次、輸 {losses} 次、打和 {bes} 次"},
        {"ticker": f"嚴格勝率：{strict_win_rate:.2f}%（若撇除打和，真實勝率約為 {adj_win_rate:.2f}%）"},
        {"ticker": f"平均贏單利潤：+{avg_win:.2f}%"},
        {"ticker": f"平均輸單虧損：{avg_loss:.2f}%"},
        {"ticker": f"期望值（每注平均）：{expectancy:.2f}%"},
        {"ticker": f"平均持股日數：{avg_holding_days:.1f} 日"},
        {"ticker": ""},
        {"ticker": f"--- 詳細交易紀錄 (依時間排序，展示資金曲線) ---"},
    ]
    
    for d in summary_data:
        for k in ["group", "strategy", "entry_date", "entry_price", "exit_date", "exit_price", "return_pct", "holding_days", "exit_reason", "portfolio_balance"]:
            if k not in d: d[k] = ""

    summary_df = pd.DataFrame(summary_data)
    final_combined_df = pd.concat([summary_df, result_df], ignore_index=True)
    
    cols = ['ticker', 'group', 'strategy', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'return_pct', 'holding_days', 'exit_reason', 'portfolio_balance']
    final_combined_df = final_combined_df[cols]
    
    final_combined_df.to_csv("backtest_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 資金複利曲線與回報率已整合，最終餘額 ${final_capital:,.2f}。已存到 backtest_results.csv")

if __name__ == "__main__":
    run_backtest()
