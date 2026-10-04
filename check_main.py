"""
check_main.py — 用真數據核對 main.py：
1. main.py 每日持倉 同 spy_bottoms_v2.py 嘅 E4 回測是否完全一樣
2. 印出幾個重要日子同今日嘅訊息
輸出：check_main.csv
"""
import pandas as pd

import main as bot
import spy_bottoms_v2 as v2

df = bot.replay(bot.add_signals(bot.load_data()))
spy, vol, vix, cash = v2.load()
f, ma = v2.add_features(spy, vol.reindex(spy.index), vix.reindex(spy.index))
sig = ((f["RSI14"] < 30) & (f["VIX"] >= 30) & (f["成交量倍數"] >= 2)).fillna(False)
bt = v2.base_plus(spy, ma, f, sig, "pct5")
pos = (df["state"] != "out").astype(float).reindex(bt.index)
L = [f"持倉唔同嘅日數：{int((pos != bt).sum())} / {len(bt)}（應該係0）", ""]
for d in ["2008-10-08", "2020-03-12", "2020-03-20", "2020-03-23", "2025-04-04", "2025-05-12"]:
    i = int(df.index.searchsorted(pd.Timestamp(d), side="right")) - 1
    L += [f"----- {d} -----", bot.build_message(df, i)[0], ""]
L += ["----- 最新 -----", bot.build_message(df, len(df) - 1)[0]]
pd.DataFrame({"report": "\n".join(L).split("\n")}).to_csv("check_main.csv", index=False, encoding="utf-8-sig")
print("\n".join(L))
