"""
research_v17.py — 第十七輪：倍升股生命週期 + 「突破→回調→再突破」+ QQQ核心加衛星倉（另一AI建議，2026-10-05）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：

股票池：U1 當時真實標普500（正式判斷）；U3 今日全美國上市普通股 + 50日平均成交額≥500萬美元（有倖存者偏差：
  已除牌股票冇價錢、中途除牌用最後價 → 結果偏好，只作參考）。數據1998起，結果2000起；已計股息；
  收市決定→下一日收市交易；每次買賣0.1%；唔用槓桿；比較對象 QQQ、SPY、MTUM（2013年中起）。

━━ 第一部分：倍升股生命週期（純描述）━━
  每月尾，股票池入面過去12個月升 ≥50% 嘅股票，按升幅分組：+50-100% / +100-200% / +200-300% / +300-500% / >+500%
  睇之後 1 / 3 / 6 / 12 / 24 個月：中位數、平均、贏SPY機率、跌≥30%機率。
  再將「升 ≥100%」嘅股票按以下拆開，睇之後6個月同12個月：
    時間：之前一年（12-24個月前）都升緊 vs 之前一年跌緊（「今年先爆」）
    路徑：過去一年最差嘅63日回調 <15% / 15-30% / >30%（健康趨勢 vs 爆炒）
    突破次數：過去一年創52週新高日數 ≥50日 vs <50日
    離52週高位：10%以內 vs 更遠
    MA50斜率、MA200斜率：向上 vs 向下（20日變化）
    波幅：60日波幅低過同組中位數 vs 高過
    成交量：20日平均 ÷ 120日平均 >1 vs ≤1

━━ 第二部分：突破 → 回調 → 再突破（「線E」候選）━━
  觸發（第 i 日收市，全部要符合）：
    A 過去12個月升 ≥ +100%   B MA50 > MA200   C MA200 向上（今日 > 20日前）
    E 前高 H = 之前60個交易日最高收市價，而 H 喺嗰日係52週新高；H 之後最低收市價比 H 低 10%至25%
    F 回調期間（H 之後到最低嗰日）平均成交量 < H 之前20日平均成交量
    G 今日收市 > H，而昨日收市 ≤ H（再突破）；D 自動符合（再突破時一定喺52週高位附近）
    同一隻股票20日內只計一次。下一日收市買入。
  事件研究：買入後 1 / 3 / 6 / 12 個月，對比 ① 同日「倍升股 + 突破60日高位但回調唔符合條件」② 每月所有「A+B+C」倍升股。
  組合：最多5隻、每隻 = 買入日總資產 ÷ 5；同時多過空位揀12個月升幅大嘅。
    E1 由買入後最高位跌20%賣（同線D一樣）；E2 揸足126個交易日（半年）先賣。

━━ 第三部分：QQQ核心 + 衛星倉（每日固定比例，近似每日再平衡）━━
  線A（70% 12-1動量前10隻半月換 + 30% SPY訊號）、線C（多週期）、線D（倍升股+跌20%賣，揸5隻），全部用U1：
    S1 70% QQQ + 15% 線A + 15% 線D     S2 70% QQQ + 15% 線C + 15% 線D     S3 85% QQQ + 15% 線D
    S4 80% QQQ + 20% E1                S5 70% QQQ + 15% 線A + 15% E1

判斷（驗證期 2015-01 至 2026-09，同期比較）—— 三條問題（建議文件提出）：
  ① 年化有冇贏QQQ？ ② 如果贏：最大跌幅好過 -35% 而且 Sharpe ≥ QQQ？ ③ 如果唔贏：最大跌幅最少好過QQQ 5個百分點，而且年化 ≥ SPY？
  (①且②) 或 ③ → 「保留」；否則淘汰。另外列出舊標準（年化 ≥ QQQ+3、Sharpe ≥ QQQ、最大跌幅好過-35%、三個年代贏兩個）。
  判斷對象：E1、E2（U1）、S1-S5。
輸出：v17_results.csv、v17_progress.txt
==== 結果（未跑）====
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v6 import effective_weights, LOOK, SKIP
from research_v8 import decision_days
from research_v10 import sleeve
from research_v13 import get_cash
from research_v15b import us_listed, download

M.PROGRESS = "v17_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    hist_sp = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    try:
        allus = us_listed()
    except Exception as e:
        log(f"全美國名單攞唔到：{e}")
        allus = []
    C, V = download(sorted(set(allus) | set(hist_sp) | {"SPY", "QQQ", "MTUM"}), "1998-01-01", "2026-10-01")
    spy = C.pop("SPY").astype(float)
    qqq = C.pop("QQQ").astype(float)
    mtum = C.pop("MTUM", None)
    for k in ("SPY", "QQQ", "MTUM"):
        V.pop(k, None)
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    cols = sorted(C)
    ix = {t: k for k, t in enumerate(cols)}
    Pdf = pd.DataFrame(C).reindex(days)[cols].astype("float32")
    Vdf = pd.DataFrame(V).reindex(days)[cols].astype("float32")
    del C, V
    f32 = lambda x: x.values.astype("float32")
    Pv, Pf = f32(Pdf), f32(Pdf.ffill())
    ma50, ma200 = f32(Pdf.rolling(50).mean()), f32(Pdf.rolling(200).mean())
    r12 = f32(Pdf / Pdf.shift(252) - 1)
    rprev = f32(Pdf.shift(252) / Pdf.shift(504) - 1)
    hi52 = f32(Pdf.rolling(252, min_periods=200).max())
    h60 = f32(Pdf.rolling(60, min_periods=40).max().shift(1))
    worst63 = f32((Pdf / Pdf.rolling(63, min_periods=20).max() - 1).rolling(252, min_periods=150).min())
    nhigh = f32((Pdf >= Pdf.rolling(252, min_periods=200).max()).astype("float32").rolling(252, min_periods=150).sum())
    s50, s200 = f32(Pdf.rolling(50).mean().pct_change(20, fill_method=None)), f32(Pdf.rolling(200).mean().pct_change(20, fill_method=None))
    vol60 = f32(Pdf.pct_change(fill_method=None).rolling(60).std() * np.sqrt(252))
    v20, v120 = f32(Vdf.rolling(20, min_periods=15).mean()), f32(Vdf.rolling(120, min_periods=90).mean())
    dv50 = f32((Pdf * Vdf).rolling(50, min_periods=40).mean())
    Vv = f32(Vdf)
    rets_v = f32(Pdf.pct_change(fill_method=None))
    del Vdf, Pdf
    log("指標計好")
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = qqq.reindex(days).pct_change(fill_method=None).fillna(0.0)
    mtum_r = mtum.astype(float).reindex(days).pct_change(fill_method=None) if mtum is not None else None
    cash_r = get_cash(days)
    cr = cash_r.values
    spy_v, qqq_v = spy.values, qqq.reindex(days).ffill().values

    U = {}
    m1 = np.zeros(Pv.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        m1[i, cache[key]] = True
    U["U1 當時真實標普500"] = m1
    if allus:
        m3 = np.zeros(Pv.shape, dtype=bool)
        m3[:, [ix[t] for t in allus if t in ix]] = True
        m3 &= np.nan_to_num(dv50, nan=0.0) >= 5e6
        U["U3 全美國（有偏差，只作參考）"] = m3

    L = ["=== 第十七輪：倍升股生命週期 + 突破回調再突破 + QQQ核心衛星（規則預先登記，見research_v17.py開頭）===",
         f"全美國 {len(allus)}；有價錢 {len(cols)}；已計股息；每次買賣0.1%；唔用槓桿", ""]

    def fwd_stats(rows, h):
        x = np.array([r for r in rows if np.isfinite(r[0]) and np.isfinite(r[1])])
        if len(x) < 20:
            return None
        return f"中位數 {np.median(x[:, 0]) * 100:+.0f}% 平均 {x[:, 0].mean() * 100:+.0f}% 贏SPY {(x[:, 0] > x[:, 1]).mean() * 100:.0f}% 跌≥30% {(x[:, 0] <= -0.3).mean() * 100:.0f}%"

    me = [days.searchsorted(d, side="right") - 1 for d in pd.date_range("2000-01-31", "2026-08-31", freq="ME")]
    HS = [(21, "1個月"), (63, "3個月"), (126, "6個月"), (252, "1年"), (504, "2年")]

    # ================= 第一部分 =================
    for un, mask in U.items():
        L.append(f"################ 第一部分：{un} ################")
        recs = []
        for i in me:
            cs = np.where(mask[i] & np.isfinite(Pv[i]) & (np.nan_to_num(r12[i], nan=-9) >= 0.5))[0]
            for c in cs:
                row = {"i": i, "c": c, "r12": r12[i, c], "prev": rprev[i, c], "w63": worst63[i, c], "nh": nhigh[i, c],
                       "off": Pv[i, c] / hi52[i, c] - 1 if np.isfinite(hi52[i, c]) else np.nan,
                       "s50": s50[i, c], "s200": s200[i, c], "vol": vol60[i, c],
                       "vx": v20[i, c] / v120[i, c] if v120[i, c] > 0 else np.nan}
                for h, _ in HS:
                    e = i + h
                    if e < len(days):
                        row[h] = (Pf[e, c] / Pv[i, c] - 1, spy_v[e] / spy_v[i] - 1)
                recs.append(row)
        D = pd.DataFrame(recs)
        D["g"] = pd.cut(D["r12"], [0.5, 1, 2, 3, 5, np.inf], right=False,
                        labels=["+50-100%", "+100-200%", "+200-300%", "+300-500%", ">+500%"])
        L.append("　按過去12個月升幅分組 → 之後表現")
        for g, sub in D.groupby("g", observed=True):
            L.append(f"　【{g}】（{len(sub)}）")
            for h, lab in HS:
                if h in sub:
                    s = fwd_stats([v for v in sub[h] if isinstance(v, tuple)], h)
                    if s:
                        L.append(f"　　{lab}：{s}")
        up = D[D["r12"] >= 1.0].copy()
        vmed = up["vol"].median()
        splits = [
            ("之前一年都升緊", up["prev"] > 0), ("之前一年跌緊（今年先爆）", up["prev"] <= 0),
            ("過去一年最差回調 <15%", up["w63"] > -0.15), ("過去一年最差回調 15-30%", (up["w63"] <= -0.15) & (up["w63"] > -0.30)),
            ("過去一年最差回調 >30%", up["w63"] <= -0.30),
            ("創52週新高 ≥50日", up["nh"] >= 50), ("創52週新高 <50日", up["nh"] < 50),
            ("離52週高位10%以內", up["off"] >= -0.10), ("離52週高位 >10%", up["off"] < -0.10),
            ("MA50向上", up["s50"] > 0), ("MA50向下", up["s50"] <= 0),
            ("MA200向上", up["s200"] > 0), ("MA200向下", up["s200"] <= 0),
            (f"波幅低（<{vmed * 100:.0f}%）", up["vol"] < vmed), ("波幅高", up["vol"] >= vmed),
            ("成交量增加 >1", up["vx"] > 1), ("成交量減少 ≤1", up["vx"] <= 1)]
        L.append("　—— 升 ≥100% 嘅股票再拆 ——")
        for lab, msk in splits:
            sub = up[msk]
            parts = []
            for h, hl in ((126, "6個月"), (252, "1年")):
                s = fwd_stats([v for v in sub[h] if isinstance(v, tuple)], h) if h in sub else None
                if s:
                    parts.append(f"{hl} {s}")
            L.append(f"　{lab}（{len(sub)}）：" + " ｜ ".join(parts))
        L.append("")
        log(f"第一部分 {un} 完成")

    # ================= 第二部分 =================
    def bpb_events(mask):
        ev, other = [], []
        last = {}
        abc = (np.nan_to_num(r12, nan=-9) >= 1.0) & (ma50 > ma200) & np.isfinite(ma200)
        for i in range(260, len(days) - 1):
            s20 = ma200[i - 20]
            cand = np.where(mask[i] & abc[i] & (ma200[i] > s20) & (Pv[i] > h60[i]) & (Pv[i - 1] <= h60[i]))[0]
            for c in cand:
                if c in last and i - last[c] < 20:
                    continue
                w = Pv[i - 60:i, c]
                if not np.isfinite(w).sum():
                    continue
                p = i - 60 + int(np.nanargmax(w))
                H = Pv[p, c]
                ok = False
                if p < i - 1 and np.isfinite(hi52[p, c]) and H >= hi52[p, c] * 0.999:
                    seg = Pv[p + 1:i, c]
                    if np.isfinite(seg).any():
                        t = p + 1 + int(np.nanargmin(seg))
                        depth = Pv[t, c] / H - 1
                        vb = np.nanmean(Vv[max(p - 20, 0):p, c])
                        va = np.nanmean(Vv[p + 1:t + 1, c])
                        ok = (-0.25 <= depth <= -0.10) and np.isfinite(va) and np.isfinite(vb) and va < vb
                last[c] = i
                (ev if ok else other).append((i, c))
        return ev, other

    def ev_stats(evs, label):
        out = [f"　{label}（{len(evs)} 次）"]
        for h, lab in HS[:4]:
            rows = []
            for i, c in evs:
                b, e = i + 1, i + 1 + h
                if e < len(days) and np.isfinite(Pf[b, c]):
                    rows.append((Pf[e, c] / Pf[b, c] - 1, qqq_v[e] / qqq_v[b] - 1, spy_v[e] / spy_v[b] - 1))
            x = np.array(rows)
            if len(x) >= 20:
                out.append(f"　　{lab}：中位數 {np.median(x[:, 0]) * 100:+.1f}% 平均 {x[:, 0].mean() * 100:+.1f}% ｜ "
                           f"贏QQQ {(x[:, 0] > x[:, 1]).mean() * 100:.0f}% ｜ 贏SPY {(x[:, 0] > x[:, 2]).mean() * 100:.0f}% ｜ "
                           f"跌≥30% {(x[:, 0] <= -0.3).mean() * 100:.0f}%")
        return out

    def simulate(cands_fn, N, exit_kind):
        hold, peak, ent = {}, {}, {}
        cash = 1.0
        ps, pb = [], []
        navs, turns = [], []
        for i in range(len(days)):
            for c in list(hold):
                r = rets_v[i, c]
                if np.isfinite(r):
                    hold[c] *= 1 + r
            cash *= 1 + cr[i]
            traded = 0.0
            for c in ps:
                if c in hold:
                    v = hold.pop(c)
                    traded += v
                    cash += v * (1 - COST)
            total = cash + sum(hold.values())
            for c in pb:
                if c in hold or len(hold) >= N or not np.isfinite(Pv[i, c]):
                    continue
                amt = min(total / N, cash)
                if amt <= total * 0.01:
                    break
                cash -= amt
                hold[c] = amt * (1 - COST)
                peak[c], ent[c] = float(Pv[i, c]), i
                traded += amt
            ps, pb = [], []
            total = cash + sum(hold.values())
            navs.append(total)
            turns.append(traded / total if total > 0 else 0.0)
            for c in hold:
                p = Pv[i, c]
                if not np.isfinite(p):
                    if i + 1 < len(days) and not np.isfinite(Pv[i + 1, c]):
                        ps.append(c)
                    continue
                peak[c] = max(peak[c], float(p))
                if exit_kind == "trail" and p <= peak[c] * 0.8:
                    ps.append(c)
                elif exit_kind == "hold" and i - ent[c] >= 126:
                    ps.append(c)
            free = N - (len(hold) - len(ps))
            if free > 0:
                cl = [c for c in cands_fn(i) if c not in hold]
                pb = cl[:free]
        nav = pd.Series(navs, index=days)
        return nav.pct_change().fillna(0.0), pd.Series(turns, index=days)

    strat = {}
    for un, mask in U.items():
        ev, other = bpb_events(mask)
        log(f"第二部分 {un}：觸發 {len(ev)}，其他突破 {len(other)}")
        L.append(f"################ 第二部分：{un} ################")
        base = [(i, c) for i in me for c in np.where(mask[i] & (np.nan_to_num(r12[i], nan=-9) >= 1.0) & (ma50[i] > ma200[i]))[0]]
        for a, b, lab in (("2000-01-01", "2014-12-31", "2000-2014"), ("2015-01-01", END, "2015-2026")):
            sel = lambda lst: [(i, c) for i, c in lst if pd.Timestamp(a) <= days[i] <= pd.Timestamp(b)]
            L.append(f"　=== {lab} ===")
            L += ev_stats(sel(ev), "突破→回調→再突破（符合全部條件）")
            L += ev_stats(sel(other), "倍升股突破60日高位，但回調唔符合條件")
            L += ev_stats(sel(base), "每月所有倍升股（12個月≥100% + MA50>MA200）")
        evd = {}
        for i, c in ev:
            evd.setdefault(i, []).append(c)
        fn = lambda i: sorted(evd.get(i, []), key=lambda c: -r12[i, c])
        for kind, nm in (("trail", "E1 突破回調再突破・跌20%賣"), ("hold", "E2 突破回調再突破・揸半年")):
            strat[(un, nm)] = simulate(fn, 5, kind)
        if un.startswith("U1"):
            ent_ok = mask & (np.nan_to_num(r12, nan=-9) >= 1.0) & (Pv > ma50) & (Pv > ma200)
            strat[(un, "線D 倍升股・跌20%賣")] = simulate(lambda i: sorted(np.where(ent_ok[i])[0], key=lambda c: -r12[i, c]), 5, "trail")
        L.append("")
        log(f"第二部分 {un} 組合完成")

    # ================= 第三部分：線A、線C =================
    sp_cols = [t for t in hist_sp if t in ix]
    px = pd.DataFrame(Pv[:, [ix[t] for t in sp_cols]], index=days, columns=sp_cols).astype(float)
    rets = px.pct_change(fill_method=None)
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    st = (df["state"] != "out").astype(float).reindex(days).fillna(0.0).shift(2).fillna(0.0)
    bot_r = st * spy_r + (1 - st) * cash_r - st.diff().abs().fillna(0.0) * COST
    R121 = px.shift(SKIP) / px.shift(LOOK) - 1
    R3, R6 = px / px.shift(63) - 1, px / px.shift(126) - 1
    half = decision_days(days, "每半個月")
    tgA, tgC = {}, {}
    for d in half:
        i = days.get_loc(d)
        if i < LOOK + 5 or d < pd.Timestamp("1999-12-01"):
            continue
        m = [t for t in members_fn(d) if t in px.columns]
        ok = [t for t in m if np.isfinite(px[t].iat[i]) and np.isfinite(px[t].iat[i - LOOK])]
        if not ok:
            continue
        a = R121.iloc[i][ok].dropna().sort_values(ascending=False).index[:10]
        tgA[d] = pd.Series(0.1, index=a)
        comp = pd.concat([R3.iloc[i][ok].rank(pct=True), R6.iloc[i][ok].rank(pct=True), R121.iloc[i][ok].rank(pct=True)],
                         axis=1).mean(axis=1, skipna=False).dropna()
        tgC[d] = pd.Series(0.1, index=comp.sort_values(ascending=False).index[:10])
    WA, _ = effective_weights(rets, tgA)
    slA, _ = sleeve(WA, rets, cash_r)
    WC, _ = effective_weights(rets, tgC)
    slC, _ = sleeve(WC, rets, cash_r)
    lineA, lineC = 0.7 * slA + 0.3 * bot_r, 0.7 * slC + 0.3 * bot_r
    log("線A、線C完成")
    u1 = "U1 當時真實標普500"
    lineD = strat[(u1, "線D 倍升股・跌20%賣")][0]
    E1 = strat[(u1, "E1 突破回調再突破・跌20%賣")][0]
    combos = {"S1 70%QQQ+15%線A+15%線D": 0.7 * qqq_r + 0.15 * lineA + 0.15 * lineD,
              "S2 70%QQQ+15%線C+15%線D": 0.7 * qqq_r + 0.15 * lineC + 0.15 * lineD,
              "S3 85%QQQ+15%線D": 0.85 * qqq_r + 0.15 * lineD,
              "S4 80%QQQ+20%E1": 0.8 * qqq_r + 0.2 * E1,
              "S5 70%QQQ+15%線A+15%E1": 0.7 * qqq_r + 0.15 * lineA + 0.15 * E1}

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2019", "2010-01-01", "2019-12-31"),
               ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END), ("科網爆煲", "2000-03-01", "2002-12-31"),
               ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]

    def show(name, r, a, b, t=None):
        rr = r.loc[a:b].dropna()
        p = perf(rr, rf=cash_r, bench=qqq_r.loc[a:b], turn=t, cap_freq="ME")
        extra = f" ｜ 每年換手 {p['turn_year']:.0f}%" if t is not None else ""
        note = f"（由{rr.index[0].date()}）" if rr.index[0] > pd.Timestamp(a) + pd.Timedelta(days=10) else ""
        return p, (f"　{name}{note}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f} ｜ "
                   f"Calmar {p['calmar']:.2f}{extra} ｜ 對QQQ 升市捕捉 {p['up_cap']:.0f}% / 跌市捕捉 {p['down_cap']:.0f}%")

    allser = {"QQQ長揸": (qqq_r, None), "SPY長揸": (spy_r, None)}
    if mtum_r is not None:
        allser["MTUM長揸"] = (mtum_r, None)
    allser["線A（70%動量+30%SPY訊號）"] = (lineA, None)
    allser["線C（多週期）"] = (lineC, None)
    for (un, nm), (r, t) in strat.items():
        allser[f"{nm}【{un[:2]}】"] = (r, t)
    for nm, r in combos.items():
        allser[nm] = (r, None)
    tabs = {}
    for wn, a, b in windows:
        L.append(f"=== 組合比較 {wn}（{a} 至 {b}）===")
        for nm, (r, t) in allser.items():
            if r.loc[a:b].dropna().shape[0] < 60:
                continue
            tabs[(wn, nm)], txt = show(nm, r, a, b, t)
            L.append(txt)
        L.append("")

    L.append("=== 預先登記判斷（驗證期 2015-2026；三條問題）===")
    q, s = tabs[("驗證期", "QQQ長揸")], tabs[("驗證期", "SPY長揸")]
    judge = [f"{nm}【U1】" for nm in ("E1 突破回調再突破・跌20%賣", "E2 突破回調再突破・揸半年")] + list(combos)
    for nm in judge:
        if ("驗證期", nm) not in tabs:
            continue
        v = tabs[("驗證期", nm)]
        q1 = v["cagr"] > q["cagr"]
        q2 = q1 and v["mdd"] > -35 and v["sharpe"] >= q["sharpe"]
        q3 = (not q1) and v["mdd"] >= q["mdd"] + 5 and v["cagr"] >= s["cagr"]
        keep = q2 or q3
        eras = sum(tabs.get((e, nm), {"cagr": -99})["cagr"] > tabs[(e, "QQQ長揸")]["cagr"] for e in ("2000-2009", "2010-2019", "2020-2026"))
        old = v["cagr"] >= q["cagr"] + 3 and v["sharpe"] >= q["sharpe"] and v["mdd"] > -35 and eras >= 2
        L.append(f"　{nm}：年化 {v['cagr']:+.2f}%（QQQ {q['cagr']:+.2f}%）、最大跌幅 {v['mdd']:.1f}%（QQQ {q['mdd']:.1f}%）、"
                 f"Sharpe {v['sharpe']:.2f}（{q['sharpe']:.2f}）→ ①{'✔' if q1 else '✘'} ②{'✔' if q2 else ('—' if not q1 else '✘')} "
                 f"③{'✔' if q3 else ('—' if q1 else '✘')} ⇒ {'保留' if keep else '淘汰'} ｜ 舊標準 {'合格' if old else '唔合格'}（三個年代贏 {eras}/3）")
    pd.DataFrame({"report": L}).to_csv("v17_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
