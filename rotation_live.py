"""
rotation_live.py — 模擬交易「輪動」：美股、納指、海外股、美國長債、黃金（第二十一輪 B4 趨勢平均分，Esther 2026-10-07：「要加，要試」）

規則（同回測一樣，鎖死唔改）：
  5個資產：SPY（美股）、QQQ（納指）、EFA（海外股）、TLT（美國長債）、GLD（黃金），每個最多 20%。
  每月最後一個交易日收市：資產月尾價 > 過去10個月月尾價平均（包埋今個月）→ 揸 20%；否則嗰 20% 揸現金。
  下一個交易日收市先買賣（同其他線一樣）；每月重新調返每個 20%；每次買賣扣 0.1%。
  開始日：2026-10-07 收市（用 2026-09 月尾嘅檢查結果開倉；之前冇回溯計）。
回測（第二十一輪，2015-2026）：+7.35%/年、最大跌幅 -11.5%；2000-2009 +8.19%/年、最大跌幅 -12.2%。
組合丙 = 50% 輪動 + 50% 線D（回測 2015-2026 +14.98% / -20.3%；2000-2009 +7.04% / -22.2%）—— 月尾成績表計。
"""
from datetime import timedelta

import numpy as np
import pandas as pd

ASSETS = {"SPY": "美股", "QQQ": "納指", "EFA": "海外股", "TLT": "美國長債", "GLD": "黃金"}
START = pd.Timestamp("2026-10-07")
COST = 0.001
W = 0.20


def download(today):
    import yfinance as yf
    h = yf.download(list(ASSETS), start=(pd.Timestamp(today) - pd.Timedelta(days=430)).strftime("%Y-%m-%d"),
                    end=(pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True, progress=False)["Close"]
    return h[list(ASSETS)].ffill()


def month_ends(idx, today, today_is_month_end):
    s = pd.Series(idx, index=idx)
    me = list(s.groupby([idx.year, idx.month]).max())
    t = pd.Timestamp(today)
    if me and me[-1].year == t.year and me[-1].month == t.month and not today_is_month_end:
        me = me[:-1]                      # 今個月未完，未算月尾
    return me


def check(px, me, d):
    """月尾 d 嘅檢查：回傳 {資產: 20%}（要揸嘅）同 {資產: (月尾價, 10個月平均)}。"""
    pts = [x for x in me if x <= d][-10:]
    w, info = {}, {}
    for a in ASSETS:
        v = px[a].reindex(pts).values.astype(float)
        if len(pts) == 10 and np.all(np.isfinite(v)):
            info[a] = (float(v[-1]), float(v.mean()))
            if v[-1] > v.mean():
                w[a] = W
        else:
            info[a] = (np.nan, np.nan)
    return w, info


def simulate(px, rf, today, today_is_month_end, capital):
    px = px[px.index <= pd.Timestamp(today)]
    days = px.index[px.index >= START]
    if len(days) == 0:
        return None
    me = month_ends(px.index, today, today_is_month_end)
    sched = {}
    prior = [x for x in me if x < days[0]]
    if prior:
        sched[days[0]] = prior[-1]
    for d in me:
        later = px.index[px.index > d]
        if d >= days[0] and len(later):
            sched[later[0]] = d
    cash, hold = capital, {}
    navs, turns, trades = [], [], []
    for i, d in enumerate(days):
        if i > 0:
            cash *= 1 + float(rf.get(d, 0.0))
        p = px.loc[d]
        nav = cash + sum(h["sh"] * p[a] for a, h in hold.items())
        traded = 0.0
        if d in sched:
            w, _ = check(px, me, sched[d])
            why = f"{sched[d].date()} 月尾檢查"
            for a in list(hold):
                val = hold[a]["sh"] * p[a]
                tgt = w.get(a, 0.0) * nav
                if tgt < val - 1:
                    amt = val - tgt
                    sh = amt / p[a]
                    full = a not in w
                    ret = (p[a] / (hold[a]["cost"] / hold[a]["sh"]) - 1) * 100
                    trades.append([str(d.date()), "輪動", "賣出" if full else "調返20%（賣少少）", a, ASSETS[a], round(float(p[a]), 2),
                                   round(sh, 4), round(amt, 2), round(ret, 2) if full else "", "",
                                   why + ("：低過10個月平均" if full else "")])
                    cash += amt * (1 - COST)
                    traded += amt
                    hold[a]["cost"] *= 1 - sh / hold[a]["sh"]
                    hold[a]["sh"] -= sh
                    if full or hold[a]["sh"] <= 1e-9:
                        hold.pop(a)
            for a, wt in w.items():
                cur = hold[a]["sh"] * p[a] if a in hold else 0.0
                tgt = wt * nav
                if tgt > cur + 1:
                    amt = min(tgt - cur, cash)
                    sh = amt * (1 - COST) / p[a]
                    trades.append([str(d.date()), "輪動", "買入" if a not in hold else "調返20%（買少少）", a, ASSETS[a],
                                   round(float(p[a]), 2), round(sh, 4), round(amt, 2), "", "", why + "：高過10個月平均"])
                    cash -= amt
                    traded += amt
                    if a in hold:
                        hold[a]["sh"] += sh
                        hold[a]["cost"] += amt
                    else:
                        hold[a] = {"sh": sh, "cost": amt, "date": d, "px": float(p[a])}
            nav = cash + sum(h["sh"] * p[a] for a, h in hold.items())
        navs.append(nav)
        turns.append(traded / nav if nav > 0 else 0.0)
    nav = pd.Series(navs, index=days)
    last_check = max(sched.values()) if sched else None
    pending = None
    if today_is_month_end and me and me[-1] == days[-1]:
        pending = check(px, me, me[-1])        # 今日月尾：下一個交易日收市執行
    return {"nav": nav, "turn": pd.Series(turns, index=days), "hold": hold, "cash": cash, "trades": trades,
            "px": px, "me": me, "last_check": last_check, "info": check(px, me, last_check)[1] if last_check is not None else {},
            "pending": pending}


def next_month_end(day):
    d = day
    for _ in range(40):
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        n = d + timedelta(days=1)
        while n.weekday() >= 5:
            n += timedelta(days=1)
        if n.month != d.month:
            return d
    return None


def report(R, today, capital):
    """回傳（Telegram 段落、現時持倉行、每日總覽行）。"""
    nav, px = R["nav"], R["px"]
    p = px.iloc[-1]
    v = float(nav.iloc[-1])
    mdd = (nav / nav.cummax() - 1).min() * 100
    dr = (nav.iloc[-1] / nav.iloc[-2] - 1) * 100 if len(nav) > 1 else 0.0
    nxt = next_month_end(today)
    lines = ["*輪動：美股／納指／海外股／長債／黃金（每樣最多20%）*",
             f"總值 ${v:,.0f}（{(v / capital - 1) * 100:+.1f}%，{v - capital:+,.0f}）｜ 今日 {dr:+.1f}% ｜ 最大跌幅 {mdd:.1f}%"]
    rows = []
    lc = R["last_check"]
    for a, name in ASSETS.items():
        mp, avg = R["info"].get(a, (np.nan, np.nan))
        cmp_txt = f"{lc.date() if lc is not None else ''}月尾 ${mp:,.2f} vs 10個月平均 ${avg:,.2f}" if np.isfinite(mp) else "數據唔夠"
        if a in R["hold"]:
            h = R["hold"][a]
            val = h["sh"] * p[a]
            lines.append(f"• {name} {a}：揸緊 ｜ 買入 ${h['px']:,.2f} → 今日 ${p[a]:,.2f} ｜ {h['sh']:.2f} 股 ｜ "
                         f"{val - h['cost']:+,.0f}（{(val / h['cost'] - 1) * 100:+.1f}%）｜ {cmp_txt}")
            rows.append(["輪動", a, name, str(h["date"].date()), round(h["px"], 2), round(h["sh"], 4), round(h["cost"], 2),
                         round(float(p[a]), 2), round(val, 2), round(val - h["cost"], 2), round((val / h["cost"] - 1) * 100, 2), "",
                         f"{nxt} 月尾收市低過10個月平均就賣（下一個交易日收市）；{cmp_txt}", "", "", str(today)])
        else:
            lines.append(f"• {name} {a}：冇揸（嗰20%揸現金）｜ {cmp_txt}")
    cash = float(R["cash"])
    lines.append(f"• 現金：${cash:,.0f}")
    rows.append(["輪動", "現金", "冇揸嗰幾份", "", "", "", "", "", round(cash, 2), "", "", "", "收國債息", "", "", str(today)])
    if R["pending"] is not None:
        w = R["pending"][0]
        buy = [a for a in w if a not in R["hold"]]
        sell = [a for a in R["hold"] if a not in w]
        lines.append(f"今日月尾檢查：下一個交易日收市 → 揸 {('、'.join(w) or '全部現金')}"
                     f"{'；新買 ' + '、'.join(buy) if buy else ''}{'；賣出 ' + '、'.join(sell) if sell else ''}")
    lines.append(f"幾時賣：每月最後一個交易日（下次 {nxt}）收市，低過過去10個月月尾平均就賣，高過就買返；下一個交易日收市執行。")
    daily = [str(today), "輪動", round(v, 2), round((v / capital - 1) * 100, 2), round(dr, 2), round(mdd, 2), round(cash, 2),
             len(R["hold"]), f"下次月尾檢查 {nxt}"]
    return "\n".join(lines), rows, daily
