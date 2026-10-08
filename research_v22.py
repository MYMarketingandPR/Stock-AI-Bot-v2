"""
research_v22.py — 第二十二輪：Jay Lo 講座（交易系統）四個可以測試嘅諗法（Esther 2026-10-08：「4個都試」）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同（同線D一樣）：當時真實標普500；已計股息；收市決定 → 下一個交易日收市買賣；每次買賣0.1%；
  最多5隻、每隻 = 買入日總資產 ÷ 5；冇投資嘅錢揸現金（^IRX）；有空位揀12個月升幅最大。
  基準 T0 = 線D：12個月升 ≥ +100% + 收市 > MA50 + 收市 > MA200 就買；由買入後最高收市價跌20%就賣。
  （核對：T0 驗證期應該同第十九、二十一輪一樣 +21.38% / -33.8%。）

新版本（每個只改一樣嘢）：
  X1 大市過濾（SPY）：SPY收市 > SPY 200日平均先開新倉；已經揸住嘅照舊用跌20%賣。
  X2 大市過濾（SPY訊號）：main.py SPY每日訊號 = 「持有」先開新倉；已經揸住嘅照舊。
  X3 第一次回調先買：倍升股（12個月 ≥ +100%）+ 收市 > MA200 + MA50向上（今日MA50 > 20日前MA50），
     而且 ① 過去5日（包今日）最低收市價喺 MA50 嘅 97%至103% 之間（回落到MA50附近，冇大幅跌穿）；
     ② 今日收市 > 昨日收市 而且 > MA50（重新轉強）；
     ③「第一次」：再之前60個交易日（第6至65日前）冇任何一日收市喺 MA50 嘅 97%至103% 之間。
     賣出照舊跌20%。
  X4 加注贏家：先買半份（買入日總資產 ÷ 10）；之後收市 ≥ 買入價 × 1.2 而且係買入後新高 → 下一個交易日收市
     加另一半（加嗰日總資產 ÷ 10，唔夠現金就有幾多加幾多），每隻只加一次；賣出照舊跌20%（全部一齊賣）。
  X5 多條均線一齊跌穿先賣：收市同時低過 MA10、MA20、MA50 就賣（冇跌20%規則）。買入同 T0。
  X6 跌20% 或者 多條均線一齊跌穿，邊個先到就賣。買入同 T0。

判斷（同 T0 比，驗證期 2015-01 至 2026-09，同之前一樣）：
  「有改善」= 最大跌幅改善 ≥ 3個百分點而且年化唔少過1個百分點；或者年化多 ≥ 2個百分點而且最大跌幅差唔超過2個百分點。
  另外列出 2000-2009、2010-2019、2020-2026、全期、三次大跌市；每個版本買賣次數、賺錢比例、平均、最差、最好；
  同埋「四條問題」（同第十九輪）作參考。
  6個版本，試得多總會有一兩個好彩：就算有改善，都要再用模擬交易驗證先可以用。
輸出：v22_results.csv、v22_progress.txt
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v13 import get_cash

M.PROGRESS = "v22_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"
N = 5


def sim(P, rets_v, cr, ok, r12n, exit_kind="T", mm=None, allow=None, pyramid=False):
    """線D 模擬。ok = 每日可以買嘅股票；exit_kind: T 跌20% / M 多條均線 / TM 邊個先到；
    mm = (ma10, ma20, ma50)；allow = 每日可唔可以開新倉；pyramid = 先買半份，升20%新高加另一半。"""
    T = P.shape[0]
    hold, info, cash = {}, {}, 1.0
    ps, pb, pa = [], [], []
    navs, trades = [], []
    n_add = 0
    for i in range(T):
        for c in list(hold):
            r = rets_v[i, c]
            if np.isfinite(r):
                hold[c] *= 1 + r
        cash *= 1 + cr[i]
        for c in ps:
            if c in hold:
                v = hold.pop(c)
                cash += v * (1 - COST)
                b = info.pop(c)
                sp = P[i, c] if np.isfinite(P[i, c]) else b["last"]
                trades.append(sp / b["px"] - 1)
        total = cash + sum(hold.values())
        for c in pa:                                   # 加注
            if c in hold and np.isfinite(P[i, c]) and not info[c]["added"]:
                amt = min(total / N / 2, cash)
                if amt > total * 0.005:
                    cash -= amt
                    hold[c] += amt * (1 - COST)
                    # 平均買入價（用嚟計每次買賣回報）
                    sh_old = info[c]["sh"]
                    sh_new = amt * (1 - COST) / P[i, c]
                    info[c]["px"] = (info[c]["px"] * sh_old + P[i, c] * sh_new) / (sh_old + sh_new)
                    info[c]["sh"] = sh_old + sh_new
                    info[c]["added"] = True
                    n_add += 1
        for c in pb:
            if c in hold or len(hold) >= N or not np.isfinite(P[i, c]):
                continue
            amt = min(total / N / (2 if pyramid else 1), cash)
            if amt <= total * 0.01:
                break
            cash -= amt
            hold[c] = amt * (1 - COST)
            info[c] = {"px": float(P[i, c]), "peak": float(P[i, c]), "last": float(P[i, c]), "entry": float(P[i, c]),
                       "sh": amt * (1 - COST) / P[i, c], "added": not pyramid}
        ps, pb, pa = [], [], []
        navs.append(cash + sum(hold.values()))
        for c in hold:
            p = P[i, c]
            if not np.isfinite(p):
                if i + 1 < T and not np.isfinite(P[i + 1, c]):
                    ps.append(c)
                continue
            new_high = p > info[c]["peak"]
            info[c]["last"] = float(p)
            info[c]["peak"] = max(info[c]["peak"], float(p))
            t_hit = p <= info[c]["peak"] * 0.8
            m_hit = False
            if mm is not None:
                a, b, d = mm[0][i, c], mm[1][i, c], mm[2][i, c]
                m_hit = np.isfinite(a) and np.isfinite(b) and np.isfinite(d) and p < a and p < b and p < d
            if (exit_kind == "T" and t_hit) or (exit_kind == "M" and m_hit) or (exit_kind == "TM" and (t_hit or m_hit)):
                ps.append(c)
            elif pyramid and not info[c]["added"] and new_high and p >= info[c]["entry"] * 1.2:
                pa.append(c)
        if allow is not None and not allow[i]:
            continue
        free = N - (len(hold) - len(ps))
        if free > 0:
            cand = [c for c in np.where(ok[i])[0] if c not in hold and c not in ps]
            cand.sort(key=lambda c: -r12n[i, c])
            pb = cand[:free]
    nav = np.array(navs)
    r = np.r_[0.0, nav[1:] / nav[:-1] - 1]
    return r, trades, n_add


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    closes = M.download_close(everyone + ["SPY", "QQQ"], "1998-01-01", "2026-10-01")
    spy, qqq = closes.pop("SPY"), closes.pop("QQQ")
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    px = pd.DataFrame(closes).reindex(days)
    cols = list(px.columns)
    P = px.values
    rets_v = px.pct_change(fill_method=None).values
    ma10, ma20 = px.rolling(10).mean().values, px.rolling(20).mean().values
    ma50, ma200 = px.rolling(50).mean().values, px.rolling(200).mean().values
    r12n = np.nan_to_num((px / px.shift(252) - 1).values, nan=-9)
    cash_r = get_cash(days)
    cr = cash_r.values
    spy_r = spy.pct_change().fillna(0.0)
    qqq_r = qqq.reindex(days).pct_change(fill_method=None).fillna(0.0)
    ix = {t: k for k, t in enumerate(cols)}
    mem = np.zeros(P.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True
    fin = np.isfinite(P)
    base_ok = mem & (r12n >= 1.0) & (P > ma50) & (P > ma200) & fin
    # X3 第一次回調
    ratio = P / ma50
    near = (ratio >= 0.97) & (ratio <= 1.03)
    near_df = pd.DataFrame(near, index=days)
    min5 = pd.DataFrame(ratio, index=days).rolling(5).min().values
    touched5 = (min5 >= 0.97) & (min5 <= 1.03)
    prior = near_df.shift(5).rolling(60).sum().values           # 第6至65日前
    ma50_up = ma50 > pd.DataFrame(ma50, index=days).shift(20).values
    up_today = P > pd.DataFrame(P, index=days).shift(1).values
    pull_ok = mem & (r12n >= 1.0) & (P > ma200) & ma50_up & touched5 & up_today & (P > ma50) & (prior == 0) & fin
    # 大市過濾
    spy_ok = (spy > spy.rolling(200).mean()).values
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    sig_ok = (df["state"] != "out").reindex(days).fillna(False).values
    log("數據準備好")

    mmv = (ma10, ma20, ma50)
    V = {}
    V["T0 線D基準"] = sim(P, rets_v, cr, base_ok, r12n)
    V["X1 SPY>MA200先開新倉"] = sim(P, rets_v, cr, base_ok, r12n, allow=spy_ok)
    V["X2 SPY訊號持有先開新倉"] = sim(P, rets_v, cr, base_ok, r12n, allow=sig_ok)
    V["X3 第一次回調先買"] = sim(P, rets_v, cr, pull_ok, r12n)
    V["X4 加注贏家"] = sim(P, rets_v, cr, base_ok, r12n, pyramid=True)
    V["X5 多條均線一齊跌穿先賣"] = sim(P, rets_v, cr, base_ok, r12n, exit_kind="M", mm=mmv)
    V["X6 跌20%或多條均線（邊個先到）"] = sim(P, rets_v, cr, base_ok, r12n, exit_kind="TM", mm=mmv)
    log("全部版本完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2019", "2010-01-01", "2019-12-31"),
               ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END), ("科網爆煲", "2000-03-01", "2002-12-31"),
               ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)
    tab = {}
    L = ["=== 第二十二輪：Jay Lo 講座四個諗法（大市過濾、第一次回調、加注贏家、多條均線離場）（規則預先登記，見research_v22.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻", ""]
    for wn, a, b in windows:
        L.append(f"=== {wn}（{a} 至 {b}）===")
        for nm, r in allser.items():
            rr = r.loc[a:b].dropna()
            p = perf(rr, rf=cash_r, bench=qqq_r.loc[a:b])
            tab[(wn, nm)] = p
            L.append(f"　{nm}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}")
        L.append("")
    L.append("=== 每次買賣（1999-2026）===")
    for nm, v in V.items():
        tr = np.array(v[1])
        if len(tr):
            ex = f" ｜ 加注 {v[2]} 次" if v[2] else ""
            L.append(f"　{nm}：{len(tr)} 次 ｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
                     f"蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}%{ex}")
    L.append("")
    q, s = tab[("驗證期", "QQQ長揸")], tab[("驗證期", "SPY長揸")]
    b0 = tab[("驗證期", "T0 線D基準")]
    L.append(f"=== 預先登記判斷（同 T0 比，驗證期）｜ T0 {b0['cagr']:+.2f}% / {b0['mdd']:.1f}%（第二十一輪 +21.38% / -33.8%）｜ "
             f"QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% ｜ SPY {s['cagr']:+.2f}% ===")
    for nm in V:
        if nm.startswith("T0"):
            continue
        x, e, e0 = tab[("驗證期", nm)], tab[("2000-2009", nm)], tab[("2000-2009", "T0 線D基準")]
        f, f0 = tab[("全期", nm)], tab[("全期", "T0 線D基準")]
        better = (x["mdd"] >= b0["mdd"] + 3 and x["cagr"] >= b0["cagr"] - 1) or (x["cagr"] >= b0["cagr"] + 2 and x["mdd"] >= b0["mdd"] - 2)
        q1 = x["cagr"] > q["cagr"]
        q2 = q1 and x["mdd"] > -35 and x["sharpe"] >= q["sharpe"]
        q3 = (not q1) and x["mdd"] >= q["mdd"] + 5 and x["cagr"] >= s["cagr"]
        q4 = e["mdd"] > -35
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} → {'有改善' if better else '冇改善'} ｜ "
                 f"2000-2009 {e['cagr']:+.2f}% / {e['mdd']:.1f}%（T0 {e0['cagr']:+.2f}% / {e0['mdd']:.1f}%）｜ "
                 f"全期 {f['cagr']:+.2f}% / {f['mdd']:.1f}%（T0 {f0['cagr']:+.2f}% / {f0['mdd']:.1f}%）｜ "
                 f"四條問題 ①{'✔' if q1 else '✘'} ②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'}")
    pd.DataFrame({"report": L}).to_csv("v22_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
