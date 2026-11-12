"""
research_v52.py — 第五十二輪：CANSLIM 價量部分（N、L、S、M），20 隻規則揀出嘅股票，詳細交易日誌
  （Esther 2026-11-12：「過往12個月第一次升到30%-50%之間既頭20頭股票」；「係你再揀20隻，唔係我」；「確認」）

純研究，唔改任何模擬交易，唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：

一、揀股票（規則，唔係人手揀）：
  測試期 2015-01-01 至 2026-09-30。
  每隻股票第一次由「過去 252 日回報低過 +30%」升入「+30% 至 +50%」區間嘅日子 = 觸發日（只計測試期內第一次）。
  觸發日當日必須係當時標普500成分股。
  按觸發日由早到晚排，取頭 20 隻（每隻股票只計一次）。
  每隻股票**由觸發日開始**先可以買入（唔會用到觸發日之前嘅未來資料）。

二、買入信號（每日收市判斷，只用當日或之前嘅數據；四個條件全部符合先買）：
  N（新高）：收市 > 前 252 日最高收市（唔包括今日），而且 成交量 ≥ 前 50 日平均成交量 × 1.5。
  L（相對強弱）：3 個月回報 × 40% + 6 個月回報 × 30% + 12 個月回報 × 30%，減去同期 SPY 嘅加權回報，需要 > 0。
  S（供求）：過去 60 日，上升日平均成交量 ÷ 下跌日平均成交量 ≥ 1.2。
  M（大市）：SPY 收市 > SPY 50 日 EMA。
  同一日有多隻符合，按 L 嘅分數由高到低揀。

三、出場（收市判斷，以先到者為準）：
  收市 < 買入價 × 0.92（即由買入價跌 8%）→ 賣出；
  或者 收市 < 50 日平均線 → 賣出。
  觸發後一律下一個交易日收市成交。

四、資金：最多 5 隻，等額；每次買賣 0.1%；閒錢收 ^IRX。

五、記錄（每次買入、賣出）：日期、股票、動作、原因、N／L／S／M 數值、買入價、賣出價、回報、持有日數、賣出後 60 日最高／最低升跌。

六、判斷（本輪唔做合格判斷）：
  同 SPY 比較（驗證期 2015–2026）：年化回報、最大跌幅、Sharpe、交易次數、賺錢比例。
  每個條件（N、L、S、M）嘅買入後 20 日、60 日回報，用嚟睇邊個條件有幫助。

已知限制：
  - 只有 20 隻股票，樣本好細，結果只可以參考。
  - 冇盈利增長（C）同三年盈利（A），因為冇可靠嘅歷史盈利數據。
  - 成交量數據來自 Yahoo，可能有缺漏。
  - 倖存者偏差（已除牌嘅股票可能冇數據）。

輸出：v52_events.csv、v52_results.csv、v52_progress.txt
==== 結果（待跑）====
"""
import numpy as np
import pandas as pd
import yfinance as yf

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash
from research_v26 import COST

M.PROGRESS = "v52_progress.txt"
log = M.log
START, END = "2015-01-01", "2026-09-30"
HOLD = 5
STOP = 0.08


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("2013-12-31", END, freq="ME")]))
    closes = M.download_close(everyone + ["SPY", "QQQ"], "2013-01-01", "2026-10-01")
    spy, qqq = closes.pop("SPY"), closes.pop("QQQ")
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    px = pd.DataFrame(closes).reindex(days)
    cols = list(px.columns)
    log(f"收市價準備好：{len(cols)} 隻，{len(days)} 日")

    # 一、揀 20 隻（規則）
    r252 = px / px.shift(252) - 1
    inband = (r252 >= 0.30) & (r252 <= 0.50)
    first = inband & ~inband.shift(1, fill_value=False)
    win = (days >= pd.Timestamp(START)) & (days <= pd.Timestamp(END))
    events = []
    for t in cols:
        hits = first.index[first[t].values & win]
        if len(hits):
            events.append((hits[0], t))
    events.sort()
    picked, sel_date = [], {}
    for d, t in events:
        if len(picked) >= 20:
            break
        if t in members_fn(d):
            picked.append(t)
            sel_date[t] = d
    log(f"揀中 {len(picked)} 隻：{picked}")

    # 成交量（只需要揀中嘅股票同 SPY）
    vol = yf.download(picked + ["SPY"], start="2014-01-01", end="2026-10-02", progress=False, auto_adjust=False)["Volume"]
    vol = vol.reindex(days)
    px_s = px[picked]
    spy_s = spy.reindex(days)
    V = vol[picked].values
    P = px_s.values
    T, C = P.shape

    sp = spy_s.values
    spy_ema50 = spy_s.ewm(span=50, adjust=False).mean().values
    m_ok = sp > spy_ema50
    hi252 = px_s.rolling(252, min_periods=200).max().shift(1).values
    vavg50 = pd.DataFrame(V).rolling(50, min_periods=40).mean().shift(1).values
    vratio = V / vavg50
    N_ok = (P > hi252) & (vratio >= 1.5)
    rel = {}
    for k in (63, 126, 252):
        rs = px_s / px_s.shift(k) - 1
        rm = spy_s / spy_s.shift(k) - 1
        rel[k] = rs.values - rm.values.reshape(-1, 1)
    Lsc = 0.4 * rel[63] + 0.3 * rel[126] + 0.3 * rel[252]
    L_ok = Lsc > 0
    dP = px_s.diff().values
    upv = np.where(dP > 0, V, np.nan)
    dnv = np.where(dP < 0, V, np.nan)
    up_avg = pd.DataFrame(upv).rolling(60, min_periods=30).mean().values
    dn_avg = pd.DataFrame(dnv).rolling(60, min_periods=30).mean().values
    Sratio = up_avg / dn_avg
    S_ok = Sratio >= 1.2
    MA50 = px_s.rolling(50).mean().values
    rets = px_s.pct_change(fill_method=None).values
    elig = np.zeros((T, C), dtype=bool)
    for k, t in enumerate(picked):
        elig[np.searchsorted(days, sel_date[t]):, k] = True
    sig = N_ok & L_ok & S_ok & m_ok.reshape(-1, 1) & elig & np.isfinite(P)

    cr = get_cash(days).values
    cash_r = get_cash(days)
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = qqq.reindex(days).pct_change(fill_method=None).fillna(0.0)

    cash = 1.0
    pos = {}
    pend_exit, pend_buy = [], []
    navs, ev = [], []
    for i in range(T):
        for k in list(pos):
            r = rets[i, k]
            if np.isfinite(r):
                pos[k]["val"] *= 1 + r
        cash *= 1 + cr[i]
        for k, why in pend_exit:
            if k in pos:
                p = pos.pop(k)
                proceeds = p["val"] * (1 - COST)
                cash += proceeds
                ev.append(dict(動作="賣出", 股票=picked[k], 日期=str(days[i].date()), 原因=why, 買入日=p["date"],
                               買入價=p["entry"], 賣出價=P[i, k], 回報=proceeds / p["cost"] - 1, 持有日=i - p["day"],
                               _k=k, _i=i))
        pend_exit = []
        total = cash + sum(p["val"] for p in pos.values())
        for k, sc in pend_buy:
            if k in pos or len(pos) >= HOLD or not np.isfinite(P[i, k]):
                continue
            amt = min(total / HOLD, cash)
            if amt <= 0:
                continue
            cash -= amt
            pos[k] = {"val": amt * (1 - COST), "cost": amt, "entry": P[i, k], "day": i, "date": str(days[i].date())}
            ev.append(dict(動作="買入", 股票=picked[k], 日期=str(days[i].date()),
                           原因="N+L+S+M 全部符合（L分數 {:+.3f}）".format(sc), 買入價=P[i, k],
                           N_成交量倍數=vratio[i, k], L分數=Lsc[i, k], S比率=Sratio[i, k], M_SPY在50EMA上=bool(m_ok[i]),
                           _k=k, _i=i))
        pend_buy = []
        navs.append(cash + sum(p["val"] for p in pos.values()))
        for k in list(pos):
            p = P[i, k]
            if not np.isfinite(p):
                continue
            if p < pos[k]["entry"] * (1 - STOP):
                pend_exit.append((k, "跌穿買入價8%止損"))
            elif np.isfinite(MA50[i, k]) and p < MA50[i, k]:
                pend_exit.append((k, "收市跌穿50日線"))
        free = HOLD - len(pos)
        if free > 0:
            cands = [k for k in np.where(sig[i])[0] if k not in pos]
            cands.sort(key=lambda k: -Lsc[i, k])
            pend_buy = [(k, float(Lsc[i, k])) for k in cands[:free]]

    nav = np.array(navs)
    strat = pd.Series(np.r_[0.0, nav[1:] / nav[:-1] - 1], index=days)

    df = pd.DataFrame(ev)
    if len(df):
        post_max, post_min = [], []
        for _, e in df.iterrows():
            if e["動作"] == "賣出":
                i_, k_ = int(e["_i"]), int(e["_k"])
                seg = P[i_:min(i_ + 61, T), k_]
                seg = seg[np.isfinite(seg)]
                post_max.append(seg.max() / P[i_, k_] - 1 if len(seg) > 1 else np.nan)
                post_min.append(seg.min() / P[i_, k_] - 1 if len(seg) > 1 else np.nan)
            else:
                post_max.append(np.nan)
                post_min.append(np.nan)
        df["賣後60日最高"] = post_max
        df["賣後60日最低"] = post_min
        df = df.drop(columns=["_k", "_i"])
        df.to_csv("v52_events.csv", index=False, encoding="utf-8-sig")

    def stats(r):
        rr = r.loc[START:END].dropna()
        return perf(rr, rf=cash_r, bench=qqq_r.loc[START:END])

    s_strat = stats(strat)
    s_spy = stats(spy_r)
    s_qqq = stats(qqq_r)
    sells = df[df["動作"] == "賣出"] if len(df) else pd.DataFrame()
    L = ["=== 第五十二輪：CANSLIM 價量部分（N/L/S/M）+ 20 隻規則揀出股票（規則預先登記，見research_v52.py開頭）===",
         f"股票池：當時真實標普500成分股（{source}），由規則揀出 {len(picked)} 隻（觸發日，按日期排序）", ""]
    L.append("=== 揀出嘅股票（觸發日）===")
    for t in picked:
        L.append(f"　{t}：{sel_date[t].date()}")
    L.append("")
    L.append(f"=== 驗證期 {START} 至 {END} ===")
    L.append(f"　CANSLIM 價量策略：年化 {s_strat['cagr']:+.2f}% ｜ 最大跌幅 {s_strat['mdd']:.1f}% ｜ Sharpe {s_strat['sharpe']:.2f}")
    L.append(f"　SPY 長揸：年化 {s_spy['cagr']:+.2f}% ｜ 最大跌幅 {s_spy['mdd']:.1f}% ｜ Sharpe {s_spy['sharpe']:.2f}")
    L.append(f"　QQQ 長揸：年化 {s_qqq['cagr']:+.2f}% ｜ 最大跌幅 {s_qqq['mdd']:.1f}% ｜ Sharpe {s_qqq['sharpe']:.2f}")
    L.append("")
    L.append("=== 賣出統計 ===")
    if len(sells):
        L.append(f"　{len(sells)} 次賣出 ｜ 平均 {sells['回報'].mean() * 100:+.1f}% ｜ 賺錢 {(sells['回報'] > 0).mean() * 100:.0f}% ｜ "
                 f"持有平均 {sells['持有日'].mean():.0f} 日 ｜ 賣後60日最高平均 {sells['賣後60日最高'].mean() * 100:+.1f}%")
        for r_, g in sells.groupby("原因"):
            L.append(f"　{r_}：{len(g)} 次 ｜ 平均 {g['回報'].mean() * 100:+.1f}% ｜ 賺錢 {(g['回報'] > 0).mean() * 100:.0f}%")
    else:
        L.append("　冇賣出交易")
    L.append("")
    L.append("=== 注意：只有 20 隻股票，樣本細，結果只可以參考 ===")
    pd.DataFrame({"report": L}).to_csv("v52_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
