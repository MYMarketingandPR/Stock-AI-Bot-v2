"""
warm_cache.py — 一次過下載研究常用資料入快取（data_cache），之後每輪研究直接用。
範圍：今日全美國上市普通股 + 1999年至今曾經入過標普500嘅股票 + 常用ETF；
  價錢+成交量 1998-01-01 至 2026-10-01；盈利紀錄；分析師紀錄（只限標普500）。
輸出：warm_cache_progress.txt
"""
import pandas as pd

import momentum_v1 as M
from research_v14 import fetch
from research_v15b import us_listed, download

M.PROGRESS = "warm_cache_progress.txt"


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    hist_sp = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", "2026-09-30", freq="ME")]))
    try:
        allus = us_listed()
    except Exception as e:
        M.log(f"全美國名單攞唔到：{e}")
        allus = []
    etfs = ["SPY", "QQQ", "QQQM", "MTUM", "SPHQ", "QUAL", "IWM", "MDY"]
    tick = sorted(set(allus) | set(hist_sp) | set(etfs))
    M.log(f"準備下載 {len(tick)} 隻")
    C, _ = download(tick, "1998-01-01", "2026-10-01")
    have = sorted(C)
    del C
    fetch(have, "earn")
    fetch([t for t in hist_sp if t in set(have)], "ana")
    M.log("完成")


if __name__ == "__main__":
    main()
