"""
snapshot_estimates.py — 每月尾記低標普500每隻股票「分析師EPS預測」同「上調/下調次數」

點解：分析師預測變化（例如30日內今年EPS預測由$5.00加到$5.70）冇免費歷史數據，所以冇得回測。
自己由2026年10月起每月記一次，儲夠一段時間之後先可以測「分析師不斷上調預測嘅股票會唔會跑贏」。

每隻股票記：今季/下季/今年/明年 嘅 EPS預測（今日、7/30/60/90日前），過去7/30日上調同下調次數，分析師人數。
存：paper_log/estimates_YYYY-MM-DD.tsv（paper-trading 分支）。
只係月尾最後一個平日先跑（或者 --force）。唔會影響模擬交易。
"""
import os
import sys
import time

import pandas as pd

import momentum_monthly as MM
import prices as PR


def main():
    today = PR.market_day()
    if "--force" not in sys.argv and not MM.is_last_trading_day(today):
        print(f"{today} 唔係月尾，唔記分析師預測。")
        return
    if "--force" not in sys.argv and os.path.exists(f"paper_log/estimates_{today}.tsv"):
        print(f"{today} 已經記咗。")
        return
    import yfinance as yf
    tickers, _ = MM.current_sp500()
    rows, ok = [], 0
    for n, t in enumerate(tickers):
        row = {"date": str(today), "ticker": t}
        try:
            tk = yf.Ticker(t)
            tr = tk.eps_trend
            if tr is not None and len(tr):
                for per in tr.index:
                    for col in ["current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo"]:
                        if col in tr.columns:
                            row[f"eps_{per}_{col}"] = tr.at[per, col]
            rv = tk.eps_revisions
            if rv is not None and len(rv):
                for per in rv.index:
                    for col in ["upLast7days", "upLast30days", "downLast30days", "downLast7Days"]:
                        if col in rv.columns:
                            row[f"rev_{per}_{col}"] = rv.at[per, col]
            es = tk.earnings_estimate
            if es is not None and len(es) and "numberOfAnalysts" in es.columns:
                for per in es.index:
                    row[f"analysts_{per}"] = es.at[per, "numberOfAnalysts"]
            ok += len(row) > 2
        except Exception as e:
            row["error"] = str(e)[:100]
        rows.append(row)
        time.sleep(0.3)
        if (n + 1) % 100 == 0:
            print(f"進度 {n + 1}/{len(tickers)}，有數據 {ok}", flush=True)
    import os
    os.makedirs("paper_log", exist_ok=True)
    path = f"paper_log/estimates_{today}.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    print(f"完成：{ok}/{len(tickers)} 隻有數據 → {path}")


if __name__ == "__main__":
    main()
