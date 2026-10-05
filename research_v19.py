"""
research_v19.py — 第十九輪：乖離止賺後回落再買 + 買入升幅上限 + SPY訊號核心（Esther 2026-10-05 確認）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同：當時真實標普500；已計股息；收市決定→下一日收市；每次買賣0.1%；最多5隻、每隻 = 買入日總資產 ÷ 5；冇股揸現金。
倍升股買入條件（同第十五輪）：12個月升 ≥ +100% + 收市 > MA50 + 收市 > MA200；有空位揀12個月升幅最大。
基本離場：M = 收市跌穿MA200；T = 由買入後最高位跌20%（線D）。基準 = M0、T0（冇任何新規則）。

━━ 第一部分：乖離止賺之後回落再買 ━━
  乖離 = 收市 ÷ MA50 − 1。乖離 ≥ X 就止賺賣出（X = 30% 或 50%；30% 係第十八輪揀規則期揀中、50% 係 Esther 例子）。
  新倍升股買入時乖離要 < X（同第十八輪）。
  乖離止賺賣咗嘅股票放入「觀察名單」：之後收市乖離回落到 -5% 至 +5%（MA50附近）而且收市 > MA200 → 優先買返（唔使再係倍升股）；
  觀察期間收市跌穿MA200 → 移出名單。有空位時先買觀察名單（多過空位揀12個月升幅大），再買新倍升股。
  版本：M/T × X30/X50 × 有/冇再買 = 8個（「冇再買」= 第十八輪做法，用嚟比較）。
━━ 第二部分：買入升幅上限 ━━
  新買入要 12個月升幅 ≤ 上限（200% 或 300%）。版本：M/T × 200%/300% = 4個。
━━ 第三部分：SPY訊號核心 + 衛星（每日固定比例）━━
  SPY訊號方案 = main.py 每日訊號（持有SPY / 揸現金）。線A = 70% 12-1動量前10半月換 + 30% SPY訊號。線D = T0。
  P1 70% SPY訊號 + 15% 線A + 15% 線D   P2 85% SPY訊號 + 15% 線D   P3 80% SPY訊號 + 10% 線A + 10% 線D（穩健版）
  對照：Q1 85% QQQ + 15% 線D   Q2 70% QQQ + 15% 線A + 15% 線D；基準 QQQ、SPY、MTUM、SPY訊號方案。

判斷：
  第一、二部分（同自己嘅基準 M0/T0 比，驗證期 2015-2026）：
    「有改善」= 最大跌幅改善 ≥ 3個百分點而且年化唔少過1個百分點；或者年化多 ≥ 2個百分點而且最大跌幅差唔超過2個百分點。
    另外列出 2000-2009 同全期。
  第三部分（四條問題）：驗證期 ① 有冇贏QQQ？ ② 贏嘅話最大跌幅好過-35%而且Sharpe ≥ QQQ？
    ③ 唔贏嘅話最大跌幅好過QQQ 5個百分點而且年化 ≥ SPY？ ④ 2000-2009 年代測試：嗰十年最大跌幅好過 -35%？
    (①且②) 或 ③ → 保留；④ 唔過 → 標明「科技牛市策略，唔係全天候」。
輸出：v19_results.csv、v19_progress.txt
==== 結果（2026-10-05 第一次跑，記錄低，唔再改）====
- 第一、二部分：12個新版本全部「冇改善」（驗證期同基準比）。
  回落MA50再買：觸發好少（4至34次），驗證期 +0.4至+1.2%、最大跌幅冇改善。
  買入升幅上限200%/300%：驗證期反而差（MA200組 +19.6%/+19.1% vs +21.0%；跌20%組 +16.9%/+19.6% vs +21.4%）→ 斬咗啲大贏家。
  例外：「跌穿MA200 + 乖離30%止賺」2000-2009 +11.3% / -35.7%（基準 +2.6% / -47.2%），同第十八輪一樣只係舊年代有用。
- 第三部分（四條問題）：
  P1 70%SPY訊號+15%線A+15%線D：驗證期 +14.83% / -27.1% / Sharpe 0.81；2000-2009 +5.16% / -30.6% → 保留（全天候）
  P2 85%SPY訊號+15%線D：+13.87% / -26.1%；2000-2009 +4.95% / -27.1% → 保留（全天候）
  P3 80%SPY訊號+10%A+10%D：+13.96% / -27.1%；2000-2009 +5.01% / -29.3% → 保留（全天候）
  Q1 85%QQQ+15%線D：+19.86% / -31.3%；2000-2009 -4.18% / -77.4% → 保留（科技牛市策略）
  Q2 70%QQQ+15%A+15%D：+19.73% / -27.6%；2000-2009 -2.31% / -72.4% → 保留（科技牛市策略）
  全期 2000-2026：P1 +11.6% / -30.6%、Q2 +11.0% / -72.4%、QQQ +8.8% / -83.0%、SPY訊號 +9.85% / -27.3%。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v6 import effective_weights, LOOK, SKIP
from research_v8 import decision_days
from research_v10 import sleeve
from research_v13 import get_cash

M.PROGRESS = "v19_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"
N = 5


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    closes = M.download_close(everyone + ["SPY", "QQQ", "MTUM"], "1998-01-01", "2026-10-01")
    spy, qqq, mtum = closes.pop("SPY"), closes.pop("QQQ"), closes.pop("MTUM", None)
    spy = spy[spy.index <= pd.Timestamp(END)]
    px = pd.DataFrame(closes).reindex(spy.index)
    days, cols = px.index, list(px.columns)
    P = px.values
    rets_v = px.pct_change(fill_method=None).values
    ma50, ma200 = px.rolling(50).mean().values, px.rolling(200).mean().values
    bias = P / ma50 - 1
    r12 = (px / px.shift(252) - 1).values
    cash_r = get_cash(days)
    cr = cash_r.values
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = qqq.reindex(days).pct_change(fill_method=None).fillna(0.0)
    mtum_r = mtum.reindex(days).pct_change(fill_method=None) if mtum is not None else None
    ix = {t: k for k, t in enumerate(cols)}
    mem = np.zeros(P.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True
    r12n = np.nan_to_num(r12, nan=-9)
    base_ok = mem & (r12n >= 1.0) & (P > ma50) & (P > ma200) & np.isfinite(P)
    log("數據準備好")

    def simulate(exit_kind, X=None, reentry=False, cap=None):
        ok = base_ok.copy()
        if X is not None:
            ok &= np.nan_to_num(bias, nan=9) < X
        if cap is not None:
            ok &= r12n <= cap
        hold, info, watch, cash = {}, {}, set(), 1.0
        ps, pb, navs, turns, trades = [], [], [], [], []
        n_re, n_tp = 0, 0
        for i in range(len(days)):
            for c in list(hold):
                r = rets_v[i, c]
                if np.isfinite(r):
                    hold[c] *= 1 + r
            cash *= 1 + cr[i]
            traded = 0.0
            for c, why in ps:
                if c in hold:
                    v = hold.pop(c)
                    traded += v
                    cash += v * (1 - COST)
                    b = info.pop(c)
                    sp = P[i, c] if np.isfinite(P[i, c]) else b["last"]
                    trades.append((sp / b["px"] - 1, b["r12"]))
                    if why == "tp" and reentry:
                        watch.add(c)
            total = cash + sum(hold.values())
            for c, is_re in pb:
                if c in hold or len(hold) >= N or not np.isfinite(P[i, c]):
                    continue
                amt = min(total / N, cash)
                if amt <= total * 0.01:
                    break
                cash -= amt
                hold[c] = amt * (1 - COST)
                info[c] = {"px": float(P[i, c]), "peak": float(P[i, c]), "last": float(P[i, c]), "r12": float(r12n[i - 1, c])}
                traded += amt
                if is_re:
                    n_re += 1
                    watch.discard(c)
            ps, pb = [], []
            total = cash + sum(hold.values())
            navs.append(total)
            turns.append(traded / total if total > 0 else 0.0)
            for c in hold:
                p = P[i, c]
                if not np.isfinite(p):
                    if i + 1 < len(days) and not np.isfinite(P[i + 1, c]):
                        ps.append((c, "gone"))
                    continue
                info[c]["last"] = float(p)
                info[c]["peak"] = max(info[c]["peak"], float(p))
                if X is not None and np.isfinite(bias[i, c]) and bias[i, c] >= X:
                    ps.append((c, "tp"))
                    n_tp += 1
                elif exit_kind == "M" and np.isfinite(ma200[i, c]) and p < ma200[i, c]:
                    ps.append((c, "x"))
                elif exit_kind == "T" and p <= info[c]["peak"] * 0.8:
                    ps.append((c, "x"))
            for c in list(watch):          # 觀察名單：跌穿MA200就移出
                if c in hold:
                    continue
                if not np.isfinite(P[i, c]) or (np.isfinite(ma200[i, c]) and P[i, c] < ma200[i, c]):
                    watch.discard(c)
            selling = {c for c, _ in ps}
            free = N - (len(hold) - len(selling))
            if free > 0:
                re_c = [c for c in watch if c not in hold and c not in selling and np.isfinite(bias[i, c])
                        and -0.05 <= bias[i, c] <= 0.05 and np.isfinite(ma200[i, c]) and P[i, c] > ma200[i, c]]
                re_c.sort(key=lambda c: -r12n[i, c])
                new_c = [c for c in np.where(ok[i])[0] if c not in hold and c not in selling and c not in re_c]
                new_c.sort(key=lambda c: -r12n[i, c])
                pb = [(c, True) for c in re_c] + [(c, False) for c in new_c]
                pb = pb[:free]
        nav = pd.Series(navs, index=days)
        return nav.pct_change().fillna(0.0), pd.Series(turns, index=days), trades, n_re, n_tp

    res = {}
    base = {"M": "跌穿MA200", "T": "由高位跌20%"}
    for ek, en in base.items():
        res[f"{en}・基準"] = (ek,) + simulate(ek)
        for X in (0.3, 0.5):
            res[f"{en}・乖離{int(X * 100)}%止賺・冇再買"] = (ek,) + simulate(ek, X=X)
            res[f"{en}・乖離{int(X * 100)}%止賺・回落MA50再買"] = (ek,) + simulate(ek, X=X, reentry=True)
        for cap in (2.0, 3.0):
            res[f"{en}・買入升幅上限{int(cap * 100)}%"] = (ek,) + simulate(ek, cap=cap)
        log(f"{en} 全部版本完成")

    # ---------- 線A、SPY訊號 ----------
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    stt = (df["state"] != "out").astype(float).reindex(days).fillna(0.0).shift(2).fillna(0.0)
    bot_r = stt * spy_r + (1 - stt) * cash_r - stt.diff().abs().fillna(0.0) * COST
    rets = px.pct_change(fill_method=None)
    R121 = px.shift(SKIP) / px.shift(LOOK) - 1
    tgA = {}
    for d in decision_days(days, "每半個月"):
        i = days.get_loc(d)
        if i < LOOK + 5 or d < pd.Timestamp("1999-12-01"):
            continue
        m = [t for t in members_fn(d) if t in px.columns]
        okk = [t for t in m if np.isfinite(px[t].iat[i]) and np.isfinite(px[t].iat[i - LOOK])]
        if okk:
            tgA[d] = pd.Series(0.1, index=R121.iloc[i][okk].dropna().sort_values(ascending=False).index[:10])
    WA, _ = effective_weights(rets, tgA)
    slA, _ = sleeve(WA, rets, cash_r)
    lineA = 0.7 * slA + 0.3 * bot_r
    lineD = res["由高位跌20%・基準"][1]
    log("線A、SPY訊號完成")
    port = {"P1 70%SPY訊號+15%線A+15%線D": 0.7 * bot_r + 0.15 * lineA + 0.15 * lineD,
            "P2 85%SPY訊號+15%線D": 0.85 * bot_r + 0.15 * lineD,
            "P3 80%SPY訊號+10%線A+10%線D（穩健版）": 0.8 * bot_r + 0.1 * lineA + 0.1 * lineD,
            "Q1 85%QQQ+15%線D": 0.85 * qqq_r + 0.15 * lineD,
            "Q2 70%QQQ+15%線A+15%線D": 0.7 * qqq_r + 0.15 * lineA + 0.15 * lineD}

    L = ["=== 第十九輪：乖離止賺後回落再買 + 買入升幅上限 + SPY訊號核心（規則預先登記，見research_v19.py開頭）===",
         f"股票池：當時真實標普500（{source}）；最多5隻；已計股息；每次買賣0.1%；唔用槓桿", ""]

    def show(name, r, a, b, t=None):
        rr = r.loc[a:b].dropna()
        p = perf(rr, rf=cash_r, bench=qqq_r.loc[a:b], turn=t, cap_freq="ME")
        ex = f" ｜ 每年換手 {p['turn_year']:.0f}%" if t is not None else ""
        note = f"（由{rr.index[0].date()}）" if rr.index[0] > pd.Timestamp(a) + pd.Timedelta(days=10) else ""
        return p, f"　{name}{note}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}{ex}"

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2019", "2010-01-01", "2019-12-31"),
               ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END), ("科網爆煲", "2000-03-01", "2002-12-31"),
               ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": (qqq_r, None), "SPY長揸": (spy_r, None)}
    if mtum_r is not None:
        allser["MTUM長揸"] = (mtum_r, None)
    allser["SPY訊號方案"] = (bot_r, None)
    allser["線A"] = (lineA, None)
    for nm, v in res.items():
        allser[nm] = (v[1], v[2])
    for nm, r in port.items():
        allser[nm] = (r, None)
    tab = {}
    for wn, a, b in windows:
        L.append(f"=== {wn}（{a} 至 {b}）===")
        for nm, (r, t) in allser.items():
            if r.loc[a:b].dropna().shape[0] < 60:
                continue
            tab[(wn, nm)], txt = show(nm, r, a, b, t)
            L.append(txt)
        L.append("")

    L.append("=== 每次買賣（1998-2026）===")
    for nm, v in res.items():
        tr = np.array([x[0] for x in v[3]])
        hi = np.array([x[1] for x in v[3]])
        if len(tr):
            L.append(f"　{nm}：{len(tr)} 次，賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ 蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ "
                     f"最差 {tr.min() * 100:+.0f}% ｜ 賺最多 {tr.max() * 100:+.0f}% ｜ 買入時已升>300%：{int((hi > 3).sum())} 次 ｜ "
                     f"乖離止賺 {v[5]} 次、回落再買 {v[4]} 次")
    L.append("")

    L.append("=== 預先登記判斷：第一、二部分（同基準比，驗證期 2015-2026）===")
    for nm, v in res.items():
        if nm.endswith("基準"):
            continue
        b0 = f"{base[v[0]]}・基準"
        x, y = tab[("驗證期", nm)], tab[("驗證期", b0)]
        better = (x["mdd"] >= y["mdd"] + 3 and x["cagr"] >= y["cagr"] - 1) or (x["cagr"] >= y["cagr"] + 2 and x["mdd"] >= y["mdd"] - 2)
        e1, e0 = tab[("2000-2009", nm)], tab[("2000-2009", b0)]
        f1, f0 = tab[("全期", nm)], tab[("全期", b0)]
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}%（基準 {y['cagr']:+.2f}% / {y['mdd']:.1f}%）→ {'有改善' if better else '冇改善'} ｜ "
                 f"2000-2009 {e1['cagr']:+.2f}% / {e1['mdd']:.1f}%（基準 {e0['cagr']:+.2f}% / {e0['mdd']:.1f}%）｜ "
                 f"全期 {f1['cagr']:+.2f}% / {f1['mdd']:.1f}%（基準 {f0['cagr']:+.2f}% / {f0['mdd']:.1f}%）")
    L.append("")
    L.append("=== 預先登記判斷：第三部分（四條問題）===")
    q, s = tab[("驗證期", "QQQ長揸")], tab[("驗證期", "SPY長揸")]
    for nm in port:
        v = tab[("驗證期", nm)]
        e = tab[("2000-2009", nm)]
        q1 = v["cagr"] > q["cagr"]
        q2 = q1 and v["mdd"] > -35 and v["sharpe"] >= q["sharpe"]
        q3 = (not q1) and v["mdd"] >= q["mdd"] + 5 and v["cagr"] >= s["cagr"]
        q4 = e["mdd"] > -35
        L.append(f"　{nm}：驗證期 {v['cagr']:+.2f}% / {v['mdd']:.1f}% / Sharpe {v['sharpe']:.2f}（QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% / {q['sharpe']:.2f}；"
                 f"SPY {s['cagr']:+.2f}%）｜ 2000-2009 {e['cagr']:+.2f}% / {e['mdd']:.1f}% → ①{'✔' if q1 else '✘'} "
                 f"②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'} ⇒ "
                 f"{'保留' if (q2 or q3) else '淘汰'}{'（全天候）' if q4 else '（科技牛市策略，唔係全天候）'}")
    pd.DataFrame({"report": L}).to_csv("v19_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
