"""
research_v13.py — 第十三輪：「QQQ + SPY訊號保護」只測一次 + 分時代比較（唔再搵新條件、唔優化）

⚠️ 預先登記（2026-10-05 跑之前寫好，跑完只可以喺下面加結果，唔改規則、唔改判斷標準）：

策略（全部規則已經固定，冇任何參數要揀）：
  QQQ  = 100% QQQ 長揸（已計股息）
  Q1   = QQQ + SPY訊號保護：main.py SPY每日訊號話「持有」就揸100% QQQ，話「現金」就揸現金（^IRX利率）
         收市決定，第2個交易日開始生效（同之前所有測試一樣），每次轉換扣0.1%。
  Q2   = 70% QQQ + 30% SPY訊號方案（即係線A嘅結構，但將「自己揀10隻股票」換成QQQ）
  線A  = 70% 12-1動量前10隻（每半個月換）+ 30% SPY訊號方案
  線B  = 100% 12-1動量前3隻（每月換）
  線C  = 70% 多週期動量前10隻（每半個月換）+ 30% SPY訊號方案
  參考：SPY長揸、SPY訊號方案、MTUM長揸（2013年中先有）
  股票池 = 當時標普500成分股；已計股息；每次買賣0.1%；唔用槓桿。

時代（regime）：2000-2009（科網爆煲 + 金融海嘯）、2010-2019、2020-2026（到2026-09-30）；
  另外全期 2000-2026，同埋「揀規則期」2010-2017 / 「驗證期」2018-2026。
  ⚠️ 2000-2009 好多已除牌股票Yahoo冇數據，會列出每個時代嘅數據覆蓋率；覆蓋率低嘅時代結果要打折扣。
  ⚠️ SPY訊號規則當初係睇住SPY 1993年至今嘅歷史定出嚟，所以對SPY嚟講唔係完全「冇見過」；
     今次真正冇見過嘅係「將同一套訊號用落QQQ」。規則冇參數，所以「滾動揀選」等於每段都係直接用固定規則。

每個時代計：年化回報、最大跌幅、Sharpe（扣現金利率）、Calmar（年化÷最大跌幅）、回本時間（最低位返到之前高位）、
  升市/跌市捕捉（對比QQQ，用每月回報）、每年換手。

判斷標準（預先定，結果出嚟照跟）：
  Q1「有用」要同時符合：
    (1) 三個時代入面最少兩個：Calmar 高過 QQQ，而且年化回報唔低過 QQQ 2個百分點以上；
    (2) 全期最大跌幅好過 -35%。
  唔符合 → 接受「保護冇價值，直接揸QQQ就得」，以後唔再試其他保護條件。
  線A「穩定贏QQQ」要三個時代年化回報都高過QQQ；只係一兩個時代贏 → 優勢只屬某個時代。
  線A 對 Q2：如果線A唔係明顯（每年 >2個百分點）贏 Q2，即係「自己揀股」冇比直接買QQQ好。

輸出：v13_results.csv、v13_progress.txt
==== 結果（未跑）====
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v6 import effective_weights, LOOK, SKIP
from research_v8 import decision_days
from research_v10 import sleeve

M.PROGRESS = "v13_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"
REGIMES = [("2000-2009", "2000-01-01", "2009-12-31"), ("2010-2019", "2010-01-01", "2019-12-31"),
           ("2020-2026", "2020-01-01", END)]
EXTRA = [("全期2000-2026", "2000-01-01", END), ("揀規則期2010-2017", "2010-01-01", "2017-12-31"),
         ("驗證期2018-2026", "2018-01-01", END)]


def get_cash(index):
    import yfinance as yf
    irx = yf.download("^IRX", start="1998-01-01", end="2026-10-01", progress=False)["Close"]
    if isinstance(irx, pd.DataFrame):
        irx = irx.iloc[:, 0]
    return irx.reindex(index).ffill().fillna(0.0) / 100 / 252


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    month_ends = pd.date_range("1999-01-31", END, freq="ME")
    everyone = sorted(set().union(*[members_fn(d) for d in month_ends]))
    log(f"成分股來源：{source}；1999-2026 曾經入過標普500：{len(everyone)} 隻")
    closes = M.download_close(everyone + ["SPY", "QQQ", "MTUM"], "1998-06-01", "2026-10-01")
    spy = closes.pop("SPY")
    qqq = closes.pop("QQQ", None)
    mtum = closes.pop("MTUM", None)
    if qqq is None:
        raise RuntimeError("冇QQQ數據")
    spy = spy[spy.index <= pd.Timestamp(END)]
    px = pd.DataFrame(closes).reindex(spy.index)
    days = px.index
    rets = px.pct_change(fill_method=None)
    spy_r = spy.pct_change().fillna(0.0)
    qq = qqq.reindex(days)
    qqq_r = qq.pct_change(fill_method=None).fillna(0.0)
    qqq_ok = qq.notna()
    mtum_r = mtum.reindex(days).pct_change(fill_method=None) if mtum is not None else None
    cash_r = get_cash(days)

    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    st = (df["state"] != "out").astype(float).reindex(days).fillna(0.0).shift(2).fillna(0.0)
    sw = st.diff().abs().fillna(0.0) * COST
    bot_r = st * spy_r + (1 - st) * cash_r - sw
    q1 = st * qqq_r + (1 - st) * cash_r - sw
    q2 = 0.7 * qqq_r + 0.3 * bot_r
    q1_turn = st.diff().abs().fillna(0.0)
    q2_turn = 0.3 * q1_turn

    R121 = px.shift(SKIP) / px.shift(LOOK) - 1
    R3 = px / px.shift(63) - 1
    R6 = px / px.shift(126) - 1
    mem_cache, cover = {}, {}

    def members(d):
        k = (d.year, d.month)
        if k not in mem_cache:
            mem_cache[k] = list(members_fn(d))
        return mem_cache[k]

    def eligible(i):
        d = days[i]
        m = members(d)
        have = [t for t in m if t in px.columns]
        ok = [t for t in have if np.isfinite(px.iat[i, px.columns.get_loc(t)])
              and np.isfinite(px.iat[i - LOOK, px.columns.get_loc(t)])]
        cover[d] = len(ok) / max(len(m), 1)
        return ok

    def targets(dlist, kind, n):
        tg = {}
        for d in dlist:
            i = days.get_loc(d)
            if i < LOOK + 5 or d < pd.Timestamp("1999-12-01"):
                continue
            ok = eligible(i)
            if not ok:
                continue
            if kind == "原始":
                sc = R121.iloc[i][ok].dropna()
            else:
                sc = pd.concat([R3.iloc[i][ok].rank(pct=True), R6.iloc[i][ok].rank(pct=True),
                                R121.iloc[i][ok].rank(pct=True)], axis=1).mean(axis=1, skipna=False).dropna()
            top = list(sc.sort_values(ascending=False).index[:n])
            if top:
                tg[d] = pd.Series(1.0 / len(top), index=top)
        return tg

    half = decision_days(days, "每半個月")
    month = decision_days(days, "每月")
    WA, _ = effective_weights(rets, targets(half, "原始", 10))
    slA, tvA = sleeve(WA, rets, cash_r)
    lineA, turnA = 0.7 * slA + 0.3 * bot_r, 0.7 * tvA + 0.3 * q1_turn
    log("線A完成")
    WB, _ = effective_weights(rets, targets(month, "原始", 3))
    lineB, turnB = sleeve(WB, rets, cash_r)
    log("線B完成")
    WC, _ = effective_weights(rets, targets(half, "多週期", 10))
    slC, tvC = sleeve(WC, rets, cash_r)
    lineC, turnC = 0.7 * slC + 0.3 * bot_r, 0.7 * tvC + 0.3 * q1_turn
    log("線C完成")

    zero = pd.Series(0.0, index=days)
    strategies = [("QQQ長揸", qqq_r, zero), ("Q1 QQQ+SPY訊號保護", q1, q1_turn), ("Q2 70%QQQ+30%SPY訊號", q2, q2_turn),
                  ("線A", lineA, turnA), ("線B（前3隻）", lineB, turnB), ("線C（多週期）", lineC, turnC),
                  ("SPY長揸", spy_r, zero), ("SPY訊號方案", bot_r, q1_turn)]
    if mtum_r is not None:
        strategies.append(("MTUM長揸", mtum_r, zero))

    def stat(r, t, a, b, need_qqq=True):
        lo = max(pd.Timestamp(a), qqq_ok.idxmax()) if need_qqq else pd.Timestamp(a)
        rr = r.loc[lo:b].dropna()
        if len(rr) < 120:
            return None, lo
        p = perf(rr, rf=cash_r, bench=qqq_r.loc[lo:b], turn=t, cap_freq="ME")
        return p, rr.index[0]

    def line(name, p, start, a):
        rec = f"{p['rec_days'] / 30.4:.0f}個月" if p["recovered"] else f"未回本（{p['rec_days'] / 30.4:.0f}個月）"
        note = f"（由{start.date()}）" if start > pd.Timestamp(a) + pd.Timedelta(days=10) else ""
        cap = "" if name == "QQQ長揸" else f" ｜ 升市捕捉 {p['up_cap']:.0f}% / 跌市捕捉 {p['down_cap']:.0f}%"
        return (f"　{name}{note}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f} ｜ "
                f"Calmar {p['calmar']:.2f} ｜ 回本 {rec} ｜ 每年換手 {p['turn_year']:.0f}%{cap}")

    L = ["=== 第十三輪：QQQ + SPY訊號保護（只測一次）+ 分時代比較（規則預先登記，見research_v13.py開頭）===",
         f"股票池：當時標普500成分股（{source}）；已計股息；每次買賣0.1%；唔用槓桿；QQQ數據由 {qqq_ok.idxmax().date()} 開始", ""]
    cov = pd.Series(cover)
    L.append("=== 數據覆蓋率（換倉日：成分股入面有足夠價錢數據嘅比例）===")
    for wn, a, b in REGIMES:
        c = cov.loc[a:b]
        if len(c):
            L.append(f"　{wn}：平均 {c.mean() * 100:.0f}%（最低 {c.min() * 100:.0f}%）")
    L.append("")
    table = {}
    for wn, a, b in REGIMES + EXTRA:
        L.append(f"=== {wn} ===")
        for name, r, t in strategies:
            p, start = stat(r, t, a, b, need_qqq=True)
            if p is None:
                L.append(f"　{name}：數據唔夠")
                continue
            table[(wn, name)] = p
            L.append(line(name, p, start, a))
        L.append("")

    # 預先登記嘅判斷
    L.append("=== 預先登記判斷 ===")
    wins = []
    for wn, _, _ in REGIMES:
        q, p1 = table.get((wn, "QQQ長揸")), table.get((wn, "Q1 QQQ+SPY訊號保護"))
        if q and p1:
            ok = (p1["calmar"] > q["calmar"]) and (p1["cagr"] >= q["cagr"] - 2)
            wins.append(ok)
            L.append(f"　{wn}：Q1 Calmar {p1['calmar']:.2f} vs QQQ {q['calmar']:.2f}；年化 {p1['cagr']:+.2f}% vs {q['cagr']:+.2f}% → "
                     f"{'符合' if ok else '唔符合'}")
    full = table.get(("全期2000-2026", "Q1 QQQ+SPY訊號保護"))
    cond2 = full is not None and full["mdd"] > -35
    verdict = sum(wins) >= 2 and cond2
    L.append(f"　條件(1) 三個時代符合 {sum(wins)} 個（要≥2）；條件(2) 全期最大跌幅 "
             f"{full['mdd']:.1f}%（要好過-35%）→ {'符合' if cond2 else '唔符合'}" if full else "　全期數據唔夠")
    L.append(f"　⇒ Q1 {'有用：保護值得考慮' if verdict else '唔合格：接受「保護冇價值」，直接揸QQQ就得'}")
    a_wins = [table[(wn, "線A")]["cagr"] > table[(wn, "QQQ長揸")]["cagr"]
              for wn, _, _ in REGIMES if (wn, "線A") in table and (wn, "QQQ長揸") in table]
    L.append(f"　線A年化回報贏QQQ嘅時代：{sum(a_wins)}/{len(a_wins)} → "
             f"{'三個時代都贏，優勢比較穩定' if a_wins and all(a_wins) else '唔係每個時代都贏，優勢只屬某啲時代'}")
    for wn, _, _ in REGIMES + EXTRA[:1]:
        if (wn, "線A") in table and (wn, "Q2 70%QQQ+30%SPY訊號") in table:
            d = table[(wn, "線A")]["cagr"] - table[(wn, "Q2 70%QQQ+30%SPY訊號")]["cagr"]
            L.append(f"　{wn}：線A − Q2 = {d:+.2f}% 每年 → {'自己揀股明顯較好' if d > 2 else '自己揀股冇明顯好過直接買QQQ'}")
    pd.DataFrame({"report": L}).to_csv("v13_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
