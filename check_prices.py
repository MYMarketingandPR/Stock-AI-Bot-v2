"""check_prices.py — MU 而家離線D、線E賣出線幾遠（日線收市價）。"""
import momentum_v1 as M
s = M.download_close(["MU"], "2025-01-01", "2026-10-15")["MU"]
s = s[s.index <= "2026-10-13"]
ma50, ma200 = s.iloc[-50:].mean(), s.iloc[-200:].mean()
hi = s.iloc[-252:].max()
hid = s.iloc[-252:].idxmax().date()
out = [f"MU 最後收市 {s.index[-1].date()} ${s.iloc[-1]:.2f}",
       f"日線 MA50 ${ma50:.2f} ｜ MA200 ${ma200:.2f} ｜ 52週最高收市 ${hi:.2f}（{hid}）｜ ×0.7 = ${hi * 0.7:.2f}",
       "最近10日收市：" + "、".join(f"{d.date()} {v:.2f}" for d, v in s.iloc[-10:].items())]
open("check_prices.txt", "w").write("\n".join(out))
print("\n".join(out))
