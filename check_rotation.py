"""check_rotation.py — 用真數據試跑 rotation_live（開始日改早，淨係核對程式，唔係模擬交易紀錄）。"""
import datetime as dt
import pandas as pd
import rotation_live as RL

out = []
today = dt.date(2026, 10, 6)
px = RL.download(today)
out.append(f"數據：{px.index[0].date()} 至 {px.index[-1].date()}，{len(px)} 日；缺數：{px.isna().sum().to_dict()}")
RL.START = pd.Timestamp("2026-08-03")
R = RL.simulate(px, pd.Series(0.0, index=px.index), today, False, 80000)
sec, rows, daily = RL.report(R, today, 80000)
out += [sec, str(daily)] + [str(t) for t in R["trades"]]
me = RL.month_ends(px.index, today, False)
out.append("月尾日子：" + "、".join(str(d.date()) for d in me[-11:]))
for a in RL.ASSETS:
    out.append(f"{a} 月尾價：" + "、".join(f"{v:.2f}" for v in px[a].reindex(me[-10:]).values))
open("check_rotation.txt", "w").write("\n".join(out))
print("\n".join(out))
