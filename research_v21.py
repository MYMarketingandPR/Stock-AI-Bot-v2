"""
research_v21.py — 第二十一輪：四個新測試（Esther 2026-10-07：「以上四個建議全部都試」）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同：收市決定→下一個交易日收市買賣（第二日先開始計回報）；每次買賣0.1%；已計股息；冇投資嘅錢揸現金（^IRX國債息）。
  線D（同第十五輪、現時模擬交易一樣）：當時真實標普500；12個月升 ≥ +100% + 收市 > MA50 + 收市 > MA200；
  最多5隻、每隻 = 買入日總資產 ÷ 5；有空位揀12個月升幅最大；由買入後最高收市價跌20%就賣。
  SPY訊號方案 = main.py 每日訊號（持有SPY / 揸現金）。
  組合 = 每日固定比例（近似每日再平衡，同之前幾輪一樣）。
  時段：驗證期 2015-01 至 2026-09；另列 2000-2009、2010-2019、2020-2026、全期 2000-2026、三次大跌市。
  「四條問題」（同第十九輪）：驗證期 ① 有冇贏QQQ？ ② 贏嘅話最大跌幅好過-35%而且Sharpe ≥ QQQ？
    ③ 唔贏嘅話最大跌幅好過QQQ 5個百分點而且年化 ≥ SPY？ ④ 2000-2009 最大跌幅好過 -35%？
    (①且②) 或 ③ → 保留；④ 唔過 → 標明「科技牛市策略，唔係全天候」。
  「有改善」（同第十九輪）：最大跌幅改善 ≥ 3個百分點而且年化唔少過1個百分點；或者年化多 ≥ 2個百分點而且最大跌幅差唔超過2個百分點。

━━ 第一部分（建議3）：線D「靠唔靠運氣」━━
  先重做線D基準（應該同第十九輪「由高位跌20%・基準」驗證期 +21.38% / -33.8% 差唔多，用嚟核對程式）。
  A1 拎走賺最多5隻：基準入面「單次買賣回報最高」嘅5隻股票（唔同股票），成段時間當佢哋唔存在，重跑。
  A2 拎走賺最多10隻：同上，10隻。
  A3 手續費加倍：每次買賣 0.2%。  A4 手續費 0.5%（好差嘅成交）。
  A5 遲一日：收市決定 → 再下一個交易日收市先買賣（即係遲兩日）。
  A6 唔同開始日：由2000-01起每個月開始揸3年，計每段3年年化，對比同期SPY、QQQ。
  合格（全部要過先叫「穩健」）：
    A1 驗證期年化 ≥ SPY長揸驗證期年化，而且全期年化 ≥ SPY全期年化；
    A3 驗證期年化少咗 ≤ 2個百分點；A5 驗證期年化少咗 ≤ 3個百分點；
    A6 ≥ 60% 嘅3年段贏SPY。
  A2、A4 只作參考（唔計合格）。

━━ 第二部分（建議2）：股票、債券、黃金、海外股輪流揸 ━━
  資產（ETF未上市之前用同類基金/期貨代替，全部冇倖存者偏差）：
    美股 SPY；納指 QQQ（1999-03起）；海外股 EFA（之前用 VGTSX）；美國長債 TLT（之前用 VUSTX）；
    美國中期國債 IEF（之前用 VFITX）；黃金 GLD（之前用黃金期貨 GC=F，2000-08起，再之前冇數 = 唔揀）。
  每月最後一個交易日收市決定，下一個交易日收市換。
  B1 雙重動量（GEM，Antonacci 2014年已發表嘅規則）：SPY 12個月回報 > 國債息12個月 → 揸 SPY 同 EFA 入面12個月升得多嗰個；否則揸 IEF。
  B2 最強一個：SPY、QQQ、EFA、TLT、GLD，分數 = 1、3、6、12個月回報平均；揸分數最高嗰個；最高都 ≤ 0 → 現金。
  B3 最強兩個：同 B2，揸頭兩名，每個50%；分數 ≤ 0 嗰份揸現金。
  B4 趨勢平均分（Faber 2007年已發表）：5個資產各20%；月尾價 > 過去10個月月尾價平均 → 揸，否則嗰20%揸現金。
  揀核心：2000-2014（揀規則期）Sharpe 最高嗰個 = 「輪動核心」。
  組合：輪動核心 85%/70%/50% + 線D 15%/30%/50%；對照 SPY訊號 85%/70%/50% + 線D。
  判斷：B1-B4 同所有組合都用「四條問題」。

━━ 第三部分（建議1）：2倍ETF + SPY訊號（⚠️ 有槓桿成分；Esther 自己決定接唔接受）━━
  2倍ETF用每日模擬：每日回報 = 2 × 指數每日回報 − (國債息 + 0.5%/年) − 管理費0.95%/年。
  核對：模擬2倍QQQ vs 真實QLD（2006-06起）、模擬2倍SPY vs 真實SSO，列出每年回報差距。
  C1 SPY訊號持有 → 揸2倍SPY；現金 → 現金。  C2 SPY訊號持有 → 揸2倍QQQ；現金 → 現金。
  C3 SPY訊號持有 → 揸1倍QQQ（對照，冇槓桿）。  C4 2倍QQQ長揸（對照，睇下唔用訊號有幾危險）。
  C5 85% C1 + 15% 線D。  C6 85% C2 + 15% 線D。
  判斷：「四條問題」；另外列出有冇任何版本驗證期年化 ≥ 50% 而且最大跌幅好過 -35%（Esther 目標）。

━━ 第四部分（建議4）：線D分幾多錢，跟線D自己表現調 ━━
  每日收市：線D總值 > 線D總值200日平均 → 線D「順」；否則「唔順」（頭200日當唔順）。下一個交易日收市調整。
  D1 順 30% / 唔順 10%；D2 順 30% / 唔順 0%；其餘揸核心。核心 = SPY訊號；另外用第二部分揀中嘅輪動核心再做一次。
  每次調整收 0.1% × 調整比例 × 2（一邊賣一邊買）。
  對照：固定比例 = 同一個版本全期平均線D比例（四捨五入到1%）。
  判斷：同「固定比例對照」比用「有改善」標準（驗證期）；另外「四條問題」。

輸出：v21_results.csv、v21_progress.txt
==== 結果（2026-10-07 第一次跑，記錄低，唔再改）====
- 核對：線D基準驗證期 +21.38% / -33.8%，同第十九輪一模一樣；SPY訊號85%+線D15% +13.87% / -26.1% 亦同第十九輪P2一樣。
- 第一部分：唔算穩健（A1✘、A3✔、A5✔、A6✔）。
  單次賺最多：NVDA 2016-18 +453%、QCOM、TXN、GEV、PCAR、FICO、NTAP、CSCO、DVN、AMD。
  A1 拎走5隻：驗證期 +12.54%（基準 +21.38%，SPY +13.67%）、全期 +9.48%（SPY +8.24%）→ 驗證期輸SPY。A2 拎走10隻：+8.86%。
  A3 手續費0.2%：+21.00%；A4 0.5%：+19.86%；A5 遲一日：+20.65%（全期反而 +15.78%）。
  A6 285段3年：贏SPY 68%、贏QQQ 46%、中位數 +12.1%、最差 -12.0%、最好 +66.0%、≥50% 只有2%。
- 第二部分：揀規則期Sharpe最高 = B4 趨勢平均分（0.68）。B1-B4 單獨全部淘汰（全天候但賺少過SPY）：
  B4 驗證期 +7.35% / -11.5%、2000-2009 +8.19% / -12.2%；B3 +12.53% / -20.9%。
  輪動核心50%+線D50% +14.98% / -20.3%（2000-2009 +7.04% / -22.2%）→ 保留（全天候）；85/15、70/30 淘汰。
  SPY訊號70%+線D30% +15.51% / -24.9%、50/50 +17.50% / -23.7% → 保留（全天候）。
- 第三部分：模擬2倍同真實QLD/SSO每日相關0.996，每年差距平均 ±0.03 個百分點 → 模擬可信。
  C2 SPY訊號→2倍QQQ +30.74% / -48.7%（2000-2009 -13.42% / -94.6%）；C1 2倍SPY +19.19% / -49.2%；C6 +30.09% / -42.0% → 全部淘汰。
  C3 SPY訊號→1倍QQQ +18.18% / -27.3% → 保留（科技牛市策略，2000-2009 -69.5%）。
  Esther目標（驗證期 ≥50% 而且最大跌幅好過 -35%）：冇任何版本做到。
- 第四部分：4個動態版本全部「冇改善」：回報只多 0.1 至 0.5 個百分點，最大跌幅反而差 0.6 至 2.9 個百分點。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash

M.PROGRESS = "v21_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"
N = 5
LEV_FEE = 0.0095
LEV_SPREAD = 0.005


# ---------------- 線D 模擬（可以拎走股票、改手續費、改延遲）----------------
def sim_d(P, rets_v, cr, ok_base, r12n, banned=(), cost=COST, lag=1):
    ok = ok_base.copy()
    if len(banned):
        ok[:, list(banned)] = False
    T = P.shape[0]
    hold, info, cash = {}, {}, 1.0
    q_sell, q_buy, pend_s, pend_b = {}, {}, set(), set()
    navs, trades = [], []
    for i in range(T):
        for c in list(hold):
            r = rets_v[i, c]
            if np.isfinite(r):
                hold[c] *= 1 + r
        cash *= 1 + cr[i]
        for c in q_sell.pop(i, []):
            pend_s.discard(c)
            if c in hold:
                v = hold.pop(c)
                cash += v * (1 - cost)
                b = info.pop(c)
                sp = P[i, c] if np.isfinite(P[i, c]) else b["last"]
                trades.append((sp / b["px"] - 1, c, b["i"], i))
        total = cash + sum(hold.values())
        lst = q_buy.pop(i, [])
        for c in lst:
            pend_b.discard(c)
        for c in lst:
            if c in hold or len(hold) >= N or not np.isfinite(P[i, c]):
                continue
            amt = min(total / N, cash)
            if amt <= total * 0.01:
                break
            cash -= amt
            hold[c] = amt * (1 - cost)
            info[c] = {"px": float(P[i, c]), "peak": float(P[i, c]), "last": float(P[i, c]), "i": i}
        navs.append(cash + sum(hold.values()))
        for c in hold:
            if c in pend_s:
                continue
            p = P[i, c]
            if not np.isfinite(p):
                if i + 1 < T and not np.isfinite(P[i + 1, c]):
                    q_sell.setdefault(i + lag, []).append(c)
                    pend_s.add(c)
                continue
            info[c]["last"] = float(p)
            info[c]["peak"] = max(info[c]["peak"], float(p))
            if p <= info[c]["peak"] * 0.8:
                q_sell.setdefault(i + lag, []).append(c)
                pend_s.add(c)
        free = N - len(hold) + len([c for c in pend_s if c in hold]) - len(pend_b)
        if free > 0:
            cand = [c for c in np.where(ok[i])[0] if c not in hold and c not in pend_b]
            cand.sort(key=lambda c: -r12n[i, c])
            for c in cand[:free]:
                q_buy.setdefault(i + lag, []).append(c)
                pend_b.add(c)
    nav = np.array(navs)
    r = np.r_[0.0, nav[1:] / nav[:-1] - 1]
    return r, trades


# ---------------- 資產輪動 ----------------
def splice(primary, proxy, days):
    """ETF有價就用ETF每日回報，未有就用代替品。回傳每日回報（未有任何數據 = NaN）。"""
    def dr(s):
        if s is None:
            return pd.Series(np.nan, index=days)
        s = s.reindex(days).ffill(limit=5)
        return s.pct_change(fill_method=None)
    a, b = dr(primary), dr(proxy)
    return a.where(a.notna(), b)


def rotation(ar, cash_r, days, kind):
    """ar = DataFrame 每日回報（資產）；回傳（每日回報、每月目標權重 DataFrame）。"""
    idx = (1 + ar.fillna(0.0)).cumprod().where(ar.notna().cumsum() > 0)
    cash_idx = (1 + cash_r).cumprod()
    mdays = decision_days(days, "每月")
    rows = {}
    for d in mdays:
        i = days.get_loc(d)
        w = pd.Series(0.0, index=ar.columns)

        def ret(a, k):
            if i - k < 0:
                return np.nan
            x0, x1 = idx[a].iat[i - k], idx[a].iat[i]
            return x1 / x0 - 1 if np.isfinite(x0) and np.isfinite(x1) else np.nan

        if kind == "B1":
            s12, e12 = ret("SPY", 252), ret("EFA", 252)
            c12 = cash_idx.iat[i] / cash_idx.iat[max(i - 252, 0)] - 1
            if np.isfinite(s12) and s12 > c12:
                w["EFA" if np.isfinite(e12) and e12 > s12 else "SPY"] = 1.0
            elif np.isfinite(s12):
                w["IEF"] = 1.0
        elif kind in ("B2", "B3"):
            sc = {}
            for a in ("SPY", "QQQ", "EFA", "TLT", "GLD"):
                v = [ret(a, k) for k in (21, 63, 126, 252)]
                if all(np.isfinite(v)):
                    sc[a] = float(np.mean(v))
            top = sorted(sc, key=lambda a: -sc[a])[: (1 if kind == "B2" else 2)]
            for a in top:
                if sc[a] > 0:
                    w[a] = 1.0 / len(top) if kind == "B3" else 1.0
        elif kind == "B4":
            me = [days.get_loc(x) for x in mdays if x <= d][-10:]
            if len(me) == 10:
                for a in ("SPY", "QQQ", "EFA", "TLT", "GLD"):
                    v = idx[a].values[me]
                    if np.all(np.isfinite(v)) and v[-1] > v.mean():
                        w[a] = 0.2
        rows[d] = w
    tw = pd.DataFrame(rows).T.reindex(days).ffill().fillna(0.0)
    held = tw.shift(2).fillna(0.0)
    turn = held.diff().abs().sum(axis=1).fillna(0.0)
    r = (held * ar.fillna(0.0)).sum(axis=1) + (1 - held.sum(axis=1)) * cash_r - turn * COST
    return r, tw


def lev2(r, cash_r):
    return 2 * r - (cash_r + LEV_SPREAD / 252) - LEV_FEE / 252


def dyn_mix(core_r, d_r, hi, lo, cash_r):
    nav = (1 + d_r).cumprod()
    good = nav > nav.rolling(200).mean()
    w = pd.Series(np.where(good, hi, lo), index=d_r.index).shift(2).fillna(lo)
    cost = w.diff().abs().fillna(0.0) * COST * 2
    return w * d_r + (1 - w) * core_r - cost, w


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    etfs = ["SPY", "QQQ", "MTUM", "EFA", "VGTSX", "TLT", "VUSTX", "IEF", "VFITX", "GLD", "GC=F", "QLD", "SSO"]
    closes = M.download_close(everyone + etfs, "1998-01-01", "2026-10-01")
    E = {t: closes.pop(t, None) for t in etfs}
    spy = E["SPY"][E["SPY"].index <= pd.Timestamp(END)]
    days = spy.index
    px = pd.DataFrame(closes).reindex(days)
    cols = list(px.columns)
    P = px.values
    rets_v = px.pct_change(fill_method=None).values
    ma50, ma200 = px.rolling(50).mean().values, px.rolling(200).mean().values
    r12n = np.nan_to_num((px / px.shift(252) - 1).values, nan=-9)
    cash_r = get_cash(days)
    cr = cash_r.values
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = E["QQQ"].reindex(days).pct_change(fill_method=None).fillna(0.0)
    mtum_r = E["MTUM"].reindex(days).pct_change(fill_method=None) if E["MTUM"] is not None else None
    ix = {t: k for k, t in enumerate(cols)}
    mem = np.zeros(P.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True
    base_ok = mem & (r12n >= 1.0) & (P > ma50) & (P > ma200) & np.isfinite(P)
    log("數據準備好")

    def ser(r):
        return pd.Series(r, index=days)

    # ---------- 第一部分 ----------
    r0, tr0 = sim_d(P, rets_v, cr, base_ok, r12n)
    lineD = ser(r0)
    best = []
    for ret, c, a, b in sorted(tr0, key=lambda x: -x[0]):
        if c not in [x[1] for x in best]:
            best.append((ret, c, a, b))
        if len(best) >= 10:
            break
    A = {"線D基準": lineD}
    A["A1 拎走賺最多5隻"] = ser(sim_d(P, rets_v, cr, base_ok, r12n, banned=[x[1] for x in best[:5]])[0])
    A["A2 拎走賺最多10隻"] = ser(sim_d(P, rets_v, cr, base_ok, r12n, banned=[x[1] for x in best[:10]])[0])
    A["A3 手續費0.2%"] = ser(sim_d(P, rets_v, cr, base_ok, r12n, cost=0.002)[0])
    A["A4 手續費0.5%"] = ser(sim_d(P, rets_v, cr, base_ok, r12n, cost=0.005)[0])
    A["A5 遲一日買賣"] = ser(sim_d(P, rets_v, cr, base_ok, r12n, lag=2)[0])
    log("第一部分完成")

    # ---------- SPY訊號 ----------
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    stt = (df["state"] != "out").astype(float).reindex(days).fillna(0.0).shift(2).fillna(0.0)
    sw = stt.diff().abs().fillna(0.0) * COST
    bot_r = stt * spy_r + (1 - stt) * cash_r - sw

    # ---------- 第二部分 ----------
    ar = pd.DataFrame({"SPY": spy.pct_change(fill_method=None),
                       "QQQ": splice(E["QQQ"], None, days),
                       "EFA": splice(E["EFA"], E["VGTSX"], days),
                       "TLT": splice(E["TLT"], E["VUSTX"], days),
                       "IEF": splice(E["IEF"], E["VFITX"], days),
                       "GLD": splice(E["GLD"], E["GC=F"], days)}, index=days)
    first = {a: (ar[a].first_valid_index().date() if ar[a].first_valid_index() is not None else None) for a in ar}
    B, BW = {}, {}
    names = {"B1": "B1 雙重動量GEM", "B2": "B2 最強一個", "B3": "B3 最強兩個", "B4": "B4 趨勢平均分"}
    for k, nm in names.items():
        B[nm], BW[nm] = rotation(ar, cash_r, days, k)
    sel = {nm: perf(r.loc["2000-01-03":"2014-12-31"], rf=cash_r)["sharpe"] for nm, r in B.items()}
    core_nm = max(sel, key=sel.get)
    core = B[core_nm]
    combos = {}
    for wd in (0.15, 0.30, 0.50):
        combos[f"輪動核心{int((1 - wd) * 100)}%+線D{int(wd * 100)}%"] = (1 - wd) * core + wd * lineD
        combos[f"SPY訊號{int((1 - wd) * 100)}%+線D{int(wd * 100)}%"] = (1 - wd) * bot_r + wd * lineD
    log("第二部分完成")

    # ---------- 第三部分 ----------
    s2, q2 = lev2(spy_r, cash_r), lev2(qqq_r, cash_r)
    C = {"C1 SPY訊號→2倍SPY": stt * s2 + (1 - stt) * cash_r - sw,
         "C2 SPY訊號→2倍QQQ": stt * q2 + (1 - stt) * cash_r - sw,
         "C3 SPY訊號→1倍QQQ": stt * qqq_r + (1 - stt) * cash_r - sw,
         "C4 2倍QQQ長揸": q2.where(E["QQQ"].reindex(days).notna(), np.nan)}
    C["C5 85%C1+15%線D"] = 0.85 * C["C1 SPY訊號→2倍SPY"] + 0.15 * lineD
    C["C6 85%C2+15%線D"] = 0.85 * C["C2 SPY訊號→2倍QQQ"] + 0.15 * lineD
    chk = []
    for real, sim in (("QLD", q2), ("SSO", s2)):
        if E[real] is None:
            chk.append(f"　{real}：冇數據")
            continue
        rr = E[real].reindex(days).pct_change(fill_method=None).dropna()
        rr = rr[rr.index >= rr.index[0] + pd.Timedelta(days=30)]
        ss = sim.reindex(rr.index)
        ya = (1 + rr).groupby(rr.index.year).prod() - 1
        yb = (1 + ss).groupby(ss.index.year).prod() - 1
        gap = (yb - ya) * 100
        chk.append(f"　模擬 vs 真實 {real}（{rr.index[0].date()} 起）：每日相關 {rr.corr(ss):.4f} ｜ 年化 模擬 {perf(ss)['cagr']:+.2f}% vs 真實 {perf(rr)['cagr']:+.2f}% ｜ "
                   f"每年差距 平均 {gap.mean():+.2f}、最大 {gap.abs().max():.2f} 個百分點")
    log("第三部分完成")

    # ---------- 第四部分 ----------
    Dd, Dw = {}, {}
    for cn, cr_ in (("SPY訊號", bot_r), ("輪動核心", core)):
        for tag, hi, lo in (("D1 順30%/唔順10%", 0.30, 0.10), ("D2 順30%/唔順0%", 0.30, 0.0)):
            r, w = dyn_mix(cr_, lineD, hi, lo, cash_r)
            avg = round(float(w.loc["2000-01-03":END].mean()), 2)
            Dd[f"{cn}・{tag}"] = r
            Dd[f"{cn}・{tag}・對照固定{int(round(avg * 100))}%"] = (1 - avg) * cr_ + avg * lineD
            Dw[f"{cn}・{tag}"] = avg
    log("第四部分完成")

    # ---------- 報告 ----------
    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2019", "2010-01-01", "2019-12-31"),
               ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END), ("科網爆煲", "2000-03-01", "2002-12-31"),
               ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    if mtum_r is not None:
        allser["MTUM長揸"] = mtum_r
    allser["SPY訊號方案"] = bot_r
    allser.update(A)
    allser.update(B)
    allser.update(combos)
    allser.update(C)
    allser.update(Dd)
    tab = {}
    L = ["=== 第二十一輪：線D靠唔靠運氣 + 資產輪動 + 2倍ETF + 線D動態比例（規則預先登記，見research_v21.py開頭）===",
         f"股票池（線D）：當時真實標普500（{source}）；已計股息；每次買賣0.1%（另有註明除外）", ""]
    for wn, a, b in windows:
        L.append(f"=== {wn}（{a} 至 {b}）===")
        for nm, r in allser.items():
            rr = r.loc[a:b].dropna()
            if rr.shape[0] < 60:
                continue
            p = perf(rr, rf=cash_r, bench=qqq_r.loc[a:b])
            tab[(wn, nm)] = p
            note = f"（由{rr.index[0].date()}）" if rr.index[0] > pd.Timestamp(a) + pd.Timedelta(days=10) else ""
            L.append(f"　{nm}{note}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}")
        L.append("")

    def four(nm):
        q, s = tab[("驗證期", "QQQ長揸")], tab[("驗證期", "SPY長揸")]
        v, e = tab[("驗證期", nm)], tab.get(("2000-2009", nm))
        q1 = v["cagr"] > q["cagr"]
        q2_ = q1 and v["mdd"] > -35 and v["sharpe"] >= q["sharpe"]
        q3 = (not q1) and v["mdd"] >= q["mdd"] + 5 and v["cagr"] >= s["cagr"]
        q4 = e is not None and e["mdd"] > -35
        et = f"{e['cagr']:+.2f}% / {e['mdd']:.1f}%" if e else "冇數"
        return (f"　{nm}：驗證期 {v['cagr']:+.2f}% / {v['mdd']:.1f}% / Sharpe {v['sharpe']:.2f} ｜ 2000-2009 {et} → "
                f"①{'✔' if q1 else '✘'} ②{'✔' if q2_ else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} "
                f"④{'✔' if q4 else '✘'} ⇒ {'保留' if (q2_ or q3) else '淘汰'}{'（全天候）' if q4 else '（科技牛市策略，唔係全天候）'}")

    def better(x, y):
        return (x["mdd"] >= y["mdd"] + 3 and x["cagr"] >= y["cagr"] - 1) or (x["cagr"] >= y["cagr"] + 2 and x["mdd"] >= y["mdd"] - 2)

    q, s = tab[("驗證期", "QQQ長揸")], tab[("驗證期", "SPY長揸")]
    L.append(f"=== 比較基準（驗證期）：QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% / Sharpe {q['sharpe']:.2f} ｜ SPY {s['cagr']:+.2f}% / {s['mdd']:.1f}% ===")
    L.append("")
    L.append("=== 第一部分：線D靠唔靠運氣 ===")
    b0, f0 = tab[("驗證期", "線D基準")], tab[("全期", "線D基準")]
    L.append(f"　核對：線D基準 驗證期 {b0['cagr']:+.2f}% / {b0['mdd']:.1f}%（第十九輪 +21.38% / -33.8%）；買賣 {len(tr0)} 次")
    L.append("　單次賺最多嘅10隻（唔同股票）：" + "；".join(
        f"{cols[c]} {days[a].date()}→{days[b].date()} {ret * 100:+.0f}%" for ret, c, a, b in best))
    sf = tab[("全期", "SPY長揸")]
    res = {}
    for nm in ("A1 拎走賺最多5隻", "A2 拎走賺最多10隻", "A3 手續費0.2%", "A4 手續費0.5%", "A5 遲一日買賣"):
        x, xf = tab[("驗證期", nm)], tab[("全期", nm)]
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}%（基準 {b0['cagr']:+.2f}%，差 {x['cagr'] - b0['cagr']:+.2f}）｜ "
                 f"全期 {xf['cagr']:+.2f}% / {xf['mdd']:.1f}%（基準 {f0['cagr']:+.2f}%）")
    x, xf = tab[("驗證期", "A1 拎走賺最多5隻")], tab[("全期", "A1 拎走賺最多5隻")]
    res["A1"] = x["cagr"] >= s["cagr"] and xf["cagr"] >= sf["cagr"]
    res["A3"] = b0["cagr"] - tab[("驗證期", "A3 手續費0.2%")]["cagr"] <= 2
    res["A5"] = b0["cagr"] - tab[("驗證期", "A5 遲一日買賣")]["cagr"] <= 3
    # A6 滾動3年
    nav_d, nav_s, nav_q = [(1 + x.loc["2000-01-03":END].fillna(0.0)).cumprod() for x in (lineD, spy_r, qqq_r)]
    starts = [d for d in decision_days(nav_d.index, "每月")]
    win_s = win_q = tot = 0
    worst, cg = 9, []
    for d0 in starts:
        d1 = d0 + pd.DateOffset(years=3)
        if d1 > nav_d.index[-1]:
            break
        j0, j1 = nav_d.index.get_loc(d0), nav_d.index.searchsorted(d1) - 1
        g = lambda n: (n.iat[j1] / n.iat[j0]) ** (1 / 3) - 1
        gd, gs, gq = g(nav_d), g(nav_s), g(nav_q)
        tot += 1
        win_s += gd > gs
        win_q += gd > gq
        cg.append(gd)
        worst = min(worst, gd)
    cg = np.array(cg)
    res["A6"] = win_s / tot >= 0.6
    L.append(f"　A6 唔同開始日（{tot} 段3年）：贏SPY {win_s / tot * 100:.0f}% ｜ 贏QQQ {win_q / tot * 100:.0f}% ｜ 3年年化 中位數 {np.median(cg) * 100:+.1f}% ｜ "
             f"最差 {worst * 100:+.1f}% ｜ 最好 {cg.max() * 100:+.1f}% ｜ 有幾多段3年年化 ≥ 50%：{(cg >= 0.5).mean() * 100:.0f}%")
    L.append("　合格：" + "、".join(f"{k}{'✔' if v else '✘'}" for k, v in res.items()) + f" ⇒ {'穩健' if all(res.values()) else '唔算穩健'}")
    L.append("")
    L.append("=== 第二部分：資產輪動 ===")
    L.append("　數據開始日：" + "、".join(f"{a} {first[a]}" for a in first))
    L.append("　揀規則期 2000-2014 Sharpe：" + "、".join(f"{k} {v:.2f}" for k, v in sel.items()) + f" → 輪動核心 = {core_nm}")
    for nm in list(B) + list(combos):
        L.append(four(nm))
    held = BW[core_nm].loc["2000-01-03":END]
    L.append("　輪動核心平均持倉：" + "、".join(f"{a} {held[a].mean() * 100:.0f}%" for a in held.columns) + f"、現金 {(1 - held.sum(axis=1)).mean() * 100:.0f}%")
    L.append("")
    L.append("=== 第三部分：2倍ETF（⚠️ 有槓桿成分）===")
    L.extend(chk)
    for nm in C:
        L.append(four(nm))
    goal = [nm for nm in allser if ("驗證期", nm) in tab and tab[("驗證期", nm)]["cagr"] >= 50 and tab[("驗證期", nm)]["mdd"] > -35]
    top = max((nm for nm in allser if ("驗證期", nm) in tab), key=lambda n: tab[("驗證期", n)]["cagr"])
    L.append(f"　Esther目標（驗證期年化 ≥ 50% 而且最大跌幅好過 -35%）：{'、'.join(goal) if goal else '冇任何版本做到'} ｜ "
             f"本輪驗證期年化最高：{top} {tab[('驗證期', top)]['cagr']:+.2f}% / {tab[('驗證期', top)]['mdd']:.1f}%")
    L.append("")
    L.append("=== 第四部分：線D動態比例 ===")
    for nm, avg in Dw.items():
        ref = f"{nm}・對照固定{int(round(avg * 100))}%"
        x, y = tab[("驗證期", nm)], tab[("驗證期", ref)]
        L.append(f"　{nm}（全期平均線D {avg * 100:.0f}%）：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% vs 固定 {y['cagr']:+.2f}% / {y['mdd']:.1f}% → "
                 f"{'有改善' if better(x, y) else '冇改善'}")
        L.append("　" + four(nm))
    pd.DataFrame({"report": L}).to_csv("v21_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
