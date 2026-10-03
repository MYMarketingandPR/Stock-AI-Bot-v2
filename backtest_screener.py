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

# 🧪 訓練期 / 驗證期分段（out-of-sample驗證，防止為咗追一個CAGR數字而過擬合）：
# 淨係用訓練期嘅數據去揀邊個倉位比例最好，再原汁原味攞去驗證期測試，
# 驗證期嘅表現先係對策略「真實」靠唔靠得住嘅誠實檢查
TRAIN_START, TRAIN_END = "2023-01-01", "2024-12-31"
VALID_START, VALID_END = "2025-01-01", END_DATE

HOLD_DAYS = 40
STOP_LOSS_PCT = -0.05
BREAKEVEN_PCT = 0.08

# 🎯 浮動止賺（trailing stop）：獲利升到 TRAILING_TRIGGER_PCT 之後，
# 唔再硬性鎖定喺某個固定%數離場，淨係由最高位回落 TRAILING_PULLBACK_PCT 先沽
# （即係：賺夠20%之後，由高點跌1%就即刻走，盡量「捕捉」後面仲可能有嘅升幅）
TRAILING_TRIGGER_PCT = 0.20
TRAILING_PULLBACK_PCT = 0.01

# 🚀 資金管理參數 —— 本金 $80,000，冇槓桿
INITIAL_CAPITAL = 80000.0
# 一次過掃描幾個唔同嘅倉位比例，唔使再逐個改數字、逐次重新部署先知結果
POSITION_SIZE_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25]


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
            breakeven_trigger = entry_price * (1 + BREAKEVEN_PCT)
            trailing_trigger_price = entry_price * (1 + TRAILING_TRIGGER_PCT)
            time_stop_d, time_stop_tgt = 15, 0.00

            peak_price = entry_price      # 持倉期間見過嘅最高位
            trailing_active = False       # 有冇已經升穿 +20%，進入浮動止賺階段

            for j in range(i + 1, min(i + 1 + HOLD_DAYS, n)):
                day_open = df["Open"].iloc[j]
                day_high = df["High"].iloc[j]
                day_low = df["Low"].iloc[j]
                day_close = df["Close"].iloc[j]
                trading_days_held = j - (i + 1)

                # 先用今日高位更新歷史最高位（保守做法：假設創新高同跌穿停損可以同一日發生）
                peak_price = max(peak_price, day_high)
                if not trailing_active and peak_price >= trailing_trigger_price:
                    trailing_active = True

                if trailing_active:
                    # 進入浮動止賺後，實際生效嘅停損價 = max(原有止蝕/保本價, 由最高位回落1%)
                    trailing_stop_price = peak_price * (1 - TRAILING_PULLBACK_PCT)
                    effective_stop_price = max(current_stop_price, trailing_stop_price)
                else:
                    effective_stop_price = current_stop_price

                stop_hit = day_low <= effective_stop_price

                if stop_hit:
                    exit_price = min(day_open, effective_stop_price)
                    if trailing_active:
                        reason_text = f"浮動止賺離場(由高點${peak_price:.2f}回落{TRAILING_PULLBACK_PCT*100:.0f}%)"
                    elif current_stop_price == entry_price:
                        reason_text = "保本離場"
                    else:
                        reason_text = f"觸發止損({STOP_LOSS_PCT*100:.0f}%)"
                    exit_date, exit_reason = df.index[j], reason_text
                    break

                if trading_days_held == time_stop_d:
                    current_return = (day_close - entry_price) / entry_price
                    if current_return < time_stop_tgt:
                        exit_price = day_close
                        exit_date, exit_reason = df.index[j], f"時間止損(未達+{time_stop_tgt*100:.0f}%)"
                        break

                # 浮動止賺未觸發之前，先用原本嘅保本機制（升到BREAKEVEN_PCT就將停損提到入場價）
                if not trailing_active and day_high >= breakeven_trigger and current_stop_price < entry_price:
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
# 第三.五步：訓練/驗證輔助函數（out-of-sample驗證）
# ==========================================
def theoretical_stats(df):
    """唔受資金限制嘅理論勝率/期望值，純粹反映訊號質素（唔講緊實際開唔開到倉）。"""
    total = len(df)
    if total == 0:
        return dict(total=0, wins=0, losses=0, bes=0, strict_wr=0.0, adj_wr=0.0,
                     avg_win=0.0, avg_loss=0.0, expectancy=0.0, avg_hold=0.0)
    win = df[df["return_pct"] > 0]
    lose = df[df["return_pct"] < 0]
    be = df[df["return_pct"] == 0]
    wins, losses, bes = len(win), len(lose), len(be)
    strict_wr = wins / total * 100
    adj_wr = wins / (wins + losses) * 100 if (wins + losses) else 0.0
    avg_win = win["return_pct"].mean() if wins else 0.0
    avg_loss = lose["return_pct"].mean() if losses else 0.0
    expectancy = (strict_wr / 100 * avg_win) + (losses / total * avg_loss)
    avg_hold = df["holding_days"].mean()
    return dict(total=total, wins=wins, losses=losses, bes=bes, strict_wr=strict_wr, adj_wr=adj_wr,
                avg_win=avg_win, avg_loss=avg_loss, expectancy=expectancy, avg_hold=avg_hold)


def sweep_sizes(df_period, initial_capital):
    """喺一段期間入面，獨立（自己嗰段用返$initial_capital起跑）掃描POSITION_SIZE_SWEEP，揀CAGR最高嗰個。"""
    total = len(df_period)
    if total == 0:
        return [], None, None, None
    start_dt, end_dt = df_period["entry_date"].min(), df_period["exit_date"].max()
    years = (end_dt - start_dt).days / 365.25
    results, best_size, best_cagr, best_df = [], None, None, None
    for size in POSITION_SIZE_SWEEP:
        df_sim, final_cap, skipped, _ = simulate_portfolio_with_capital_limit(df_period.copy(), initial_capital, size)
        taken = total - skipped
        cagr = (
            ((final_cap / initial_capital) ** (1 / years) - 1) * 100
            if years > 0 else ((final_cap - initial_capital) / initial_capital) * 100
        )
        results.append({
            "size": size, "final_capital": final_cap, "cagr": cagr,
            "taken": taken, "skipped": skipped, "util_pct": taken / total * 100,
        })
        if best_cagr is None or cagr > best_cagr:
            best_cagr, best_size, best_df = cagr, size, df_sim
    return results, best_size, best_cagr, best_df


def apply_fixed_size(df_period, initial_capital, size):
    """用一個已經喺其他期間揀定嘅size,原汁原味放落呢段期間跑一次(唔會喺呢段再重新揀size，避免過擬合)。"""
    total = len(df_period)
    if total == 0 or size is None:
        return None
    start_dt, end_dt = df_period["entry_date"].min(), df_period["exit_date"].max()
    years = (end_dt - start_dt).days / 365.25
    df_sim, final_cap, skipped, _ = simulate_portfolio_with_capital_limit(df_period.copy(), initial_capital, size)
    taken = total - skipped
    cagr = (
        ((final_cap / initial_capital) ** (1 / years) - 1) * 100
        if years > 0 else ((final_cap - initial_capital) / initial_capital) * 100
    )
    return {
        "size": size, "final_capital": final_cap, "cagr": cagr, "taken": taken,
        "skipped": skipped, "util_pct": taken / total * 100 if total else 0.0,
        "years": years, "sim_df": df_sim,
    }


# ==========================================
# 第四步：跑回測與生成報告
# ==========================================
def analyze_universe(df_all, label):
    """對一組已經跑出嚟嘅trades(df_all)，做訓練/驗證拆期 + 倉位比例掃描分析。
    抽出嚟做獨立function，方便可以用相同嘅一批trades，比較「全部股票池」vs「只用Mega Cap」
    呢類「邊個股票池先靠得住」嘅問題，而唔使重新落yfinance攞多次data。"""
    train_mask = (df_all["entry_date"] >= pd.Timestamp(TRAIN_START)) & (df_all["entry_date"] <= pd.Timestamp(TRAIN_END))
    valid_mask = (df_all["entry_date"] >= pd.Timestamp(VALID_START)) & (df_all["entry_date"] <= pd.Timestamp(VALID_END))
    train_df = df_all[train_mask].reset_index(drop=True)
    valid_df = df_all[valid_mask].reset_index(drop=True)

    train_theo = theoretical_stats(train_df)
    valid_theo = theoretical_stats(valid_df)

    train_sweep, train_best_size, train_best_cagr, train_best_df = sweep_sizes(train_df, INITIAL_CAPITAL)
    valid_oos = apply_fixed_size(valid_df, INITIAL_CAPITAL, train_best_size)
    valid_sweep_ref, valid_own_best_size, valid_own_best_cagr, _ = sweep_sizes(valid_df, INITIAL_CAPITAL)

    lines = [f"=== {label} ==="]
    lines.append(f"【訓練期 {TRAIN_START} ~ {TRAIN_END}】訊號 {train_theo['total']} 次，倉位比例掃描結果（由細到大）：")
    for r in train_sweep:
        lines.append(f"　倉位{r['size']*100:.0f}%：CAGR {r['cagr']:+.2f}%，最終結餘${r['final_capital']:,.2f}，"
                      f"開倉{r['taken']}/{train_theo['total']}次(使用率{r['util_pct']:.1f}%)")
    if train_best_size is not None:
        lines.append(f"👉 訓練期揀出嘅最佳倉位：{train_best_size*100:.0f}%（CAGR {train_best_cagr:+.2f}%）——呢個size會攞去驗證期做出樣本外測試")
    lines.append(f"訓練期理論勝率：{train_theo['strict_wr']:.2f}%（撇除打和後約{train_theo['adj_wr']:.2f}%），期望值{train_theo['expectancy']:.2f}%，平均持股{train_theo['avg_hold']:.1f}日")
    lines.append("")
    lines.append(f"【驗證期 {VALID_START} ~ {VALID_END}】訊號 {valid_theo['total']} 次")
    if valid_oos is not None:
        lines.append(f"👉 用訓練期揀嘅倉位{train_best_size*100:.0f}%，喺驗證期（出樣本外）：CAGR {valid_oos['cagr']:+.2f}%，"
                      f"最終結餘${valid_oos['final_capital']:,.2f}，開倉{valid_oos['taken']}/{valid_theo['total']}次")
    else:
        lines.append("⚠️ 驗證期冇足夠交易訊號，未能計算")
    lines.append(f"驗證期理論勝率：{valid_theo['strict_wr']:.2f}%（撇除打和後約{valid_theo['adj_wr']:.2f}%），期望值{valid_theo['expectancy']:.2f}%，平均持股{valid_theo['avg_hold']:.1f}日")
    lines.append("")
    lines.append("--- 參考（唔應該用嚫揀參數！純粹量度過擬合風險）：如果直接用驗證期自己數據揀最啱size ---")
    for r in valid_sweep_ref:
        lines.append(f"　倉位{r['size']*100:.0f}%：CAGR {r['cagr']:+.2f}%")
    if valid_oos is not None and valid_own_best_cagr is not None:
        gap = valid_own_best_cagr - valid_oos["cagr"]
        lines.append(f"（驗證期自選最佳CAGR {valid_own_best_cagr:+.2f}% vs 誠實出樣本外CAGR {valid_oos['cagr']:+.2f}%，"
                      f"差距{gap:+.2f}個百分點——差距愈大，代表訓練期揀出嘅參數愈可能係撞中咗）")

    return {
        "label": label, "lines": lines, "train_df": train_df, "valid_df": valid_df,
        "train_theo": train_theo, "valid_theo": valid_theo,
        "train_best_size": train_best_size, "train_best_cagr": train_best_cagr,
        "train_best_df": train_best_df, "valid_oos": valid_oos,
    }


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

    # --- 用同一批已經跑出嚟嘅trades，分別比較「全部股票池」vs「只用Mega Cap(剔除High Beta)」---
    full_result = analyze_universe(result_df, "全部股票池 (Mega Cap + High Beta)")
    mega_only_df = result_df[result_df["group"] == "Mega Cap"].reset_index(drop=True)
    mega_result = analyze_universe(mega_only_df, "只用 Mega Cap (剔除 High Beta)")

    summary_lines = [
        "=== 回測報告：訓練期揀參數 → 驗證期做Out-of-Sample測試（防過擬合）===",
        f"每段獨立用返 ${INITIAL_CAPITAL:,.2f} 本金起跑（冇槓桿，資金唔夠嘅訊號會被跳過）",
        "",
    ]
    summary_lines += full_result["lines"]
    summary_lines += [""]
    summary_lines += mega_result["lines"]
    summary_lines += [
        "",
        "--- 詳細交易紀錄（下面全部用「全部股票池」版本；period=訓練/驗證；"
        "position_taken=False代表資金不足未開倉；全部用訓練期揀嘅size模擬）---",
    ]
    summary_data = [{"ticker": line} for line in summary_lines]

    train_best_df, valid_oos = full_result["train_best_df"], full_result["valid_oos"]
    detail_frames = []
    if train_best_df is not None:
        df_t = train_best_df.copy()
        df_t["period"] = "訓練期"
        detail_frames.append(df_t)
    if valid_oos is not None:
        df_v = valid_oos["sim_df"].copy()
        df_v["period"] = "驗證期"
        detail_frames.append(df_v)

    if detail_frames:
        detail_df = pd.concat(detail_frames, ignore_index=True)
        detail_df["entry_date"] = pd.to_datetime(detail_df["entry_date"]).dt.strftime("%Y-%m-%d")
        detail_df["exit_date"] = pd.to_datetime(detail_df["exit_date"]).dt.strftime("%Y-%m-%d")
    else:
        detail_df = pd.DataFrame(columns=["ticker", "group", "strategy", "entry_date", "exit_date",
                                           "entry_price", "exit_price", "return_pct", "holding_days",
                                           "exit_reason", "position_taken", "period"])

    for d in summary_data:
        for k in ["group", "strategy", "entry_date", "entry_price", "exit_date", "exit_price",
                  "return_pct", "holding_days", "exit_reason", "position_taken", "period"]:
            if k not in d:
                d[k] = ""

    summary_df = pd.DataFrame(summary_data)
    final_combined_df = pd.concat([summary_df, detail_df], ignore_index=True)
    cols = ["ticker", "group", "strategy", "entry_date", "exit_date", "entry_price", "exit_price",
            "return_pct", "holding_days", "exit_reason", "position_taken", "period"]
    final_combined_df = final_combined_df[cols]

    final_combined_df.to_csv("backtest_results.csv", index=False, encoding="utf-8-sig")

    valid_oos = full_result["valid_oos"]
    valid_cagr_print = f"{valid_oos['cagr']:+.2f}%" if valid_oos is not None else "N/A"
    print(f"\n✅ 已完成（訓練/驗證分段out-of-sample版本，加埋Mega-Cap-only對比）。"
          f"全部股票池：驗證期CAGR {valid_cagr_print}。"
          f"已存到 backtest_results.csv")


if __name__ == "__main__":
    run_backtest()
