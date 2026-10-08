"""
research_v23.py — 第二十三輪：睇大市同 RS Rating 先決定賣唔賣（Esther 2026-10-08）

Esther：講座入面唔係「跌穿均線」或者「跌20%」就一定賣，仲會睇 SPY 喺200日線上定下、股價喺均線上定下、RS Rating。
問題：SPY 跌穿200日線，但自己隻股仲未跌穿 MA50，係咪可以繼續揸？

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同（同線D一樣）：當時真實標普500；已計股息；收市決定 → 下一個交易日收市買賣；每次買賣0.1%；
  最多5隻、每隻 = 買入日總資產 ÷ 5；冇投資嘅錢揸現金（^IRX）；
  買入 = 12個月升 ≥ +100% + 收市 > MA50 + 收市 > MA200；有空位揀12個月升幅最大。只改「賣出」。
定義：
  大市弱 = SPY收市 < SPY 200日平均。
  RS Rating（仿 IBD，1-99）：分數 = 0.4×(今日價÷63日前) + 0.2×(÷126日前) + 0.2×(÷189日前) + 0.2×(÷252日前)；
    同當日標普500成分股比，排第幾（百分位 × 99）。（IBD 用全美國股票同自己公式，呢個係近似版。）
賣出版本：
  E0 線D基準：由買入後最高收市價跌20%就賣。
  E1 大市弱但股強就揸：同E0，但如果「大市弱」而且股價收市仍然 > MA50 → 跌20%都唔賣；
     （大市強，或者股價已經跌穿MA50 → 照跌20%規則）。
  E2 RS 跌出領導股先賣：RS < 70 就賣；保險：由最高位跌30%一定賣。
  E3 狀態判斷：以下任何一樣就賣：① 收市 < MA50 而且 RS < 80（趨勢壞咗而且唔再係領導股）；
     ② 收市 < MA200（大趨勢壞咗）；③ 由最高位跌30%（保險）。
  E4 領導股俾多啲空間：跌20%先賣，但如果 RS ≥ 80 而且收市 > MA50 → 唔賣；保險：由最高位跌30%一定賣。
判斷（同 E0 比，驗證期 2015-01 至 2026-09）：
  「有改善」= 最大跌幅改善 ≥ 3個百分點而且年化唔少過1個百分點；或者年化多 ≥ 2個百分點而且最大跌幅差唔超過2個百分點。
  另外列 2000-2009、全期、三次大跌市、每次買賣統計、「四條問題」作參考。4個版本，有改善都要模擬交易再驗證。
另外（純描述，直接答 Esther 條問題）：
  標普500入面嘅倍升股（12個月 ≥ +100%），每5個交易日抽一次：
  「大市弱」嗰陣，分 ① 股價 > MA50 vs < MA50；② RS ≥ 90 vs < 90 → 之後 1個月 / 3個月：中位數、平均、跌 ≥20% 機率、贏SPY機率。
  同「大市強」嗰陣比較。
輸出：v23_results.csv、v23_progress.txt
==== 結果（2026-10-08 第一次跑，記錄低，唔再改）====
- 核對：E0 驗證期 +21.38% / -33.8%，同之前一樣。4個版本全部「冇改善」。
- E1 大市弱但股價 > MA50 就唔賣：驗證期 +21.52% / -33.8%（幾乎一樣）；2000-2009 +0.40% / -49.8%（E0 +4.55% / -46.1%）；
  金融海嘯 -20.9%（E0 -15.2%）；最差一次買賣 -63%（E0 -41%）。
- E4 RS≥80且>MA50唔理跌20%：驗證期 +21.93% / -33.8%；2000-2009 +0.48% / -49.8%；最差 -63%。
- E2 RS<70先賣：+18.94% / -42.6%；E3 狀態判斷：+14.65% / -41.7% → 兩個都差咗。
- 描述（倍升股，大市弱）：股價 > MA50 之後1個月中位數 +0.8%（贏SPY 58%），< MA50 -0.4%（50%）；
  3個月 +1.1% vs +0.3%，跌≥20% 機率 17% vs 18% → 喺MA50上面稍好，但分別細，六分一機會3個月內跌兩成。
  RS ≥90 vs <90：1個月 +1.1% vs -3.4%（RS<90 樣本只有101）。大市強時反而 < MA50 嘅之後升得多（+2.8% vs +0.8%）。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v13 import get_cash

M.PROGRESS = "v23_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"
N = 5


def sim(P, rets_v, cr, ok, r12n, sell_fn):
    T = P.shape[0]
    hold, info, cash = {}, {}, 1.0
    ps, pb, navs, trades = [], [], [], []
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
        for c in pb:
            if c in hold or len(hold) >= N or not np.isfinite(P[i, c]):
                continue
            amt = min(total / N, cash)
            if amt <= total * 0.01:
                break
            cash -= amt
            hold[c] = amt * (1 - COST)
            info[c] = {"px": float(P[i, c]), "peak": float(P[i, c]), "last": float(P[i, c])}
        ps, pb = [], []
        navs.append(cash + sum(hold.values()))
        for c in hold:
            p = P[i, c]
            if not np.isfinite(p):
                if i + 1 < T and not np.isfinite(P[i + 1, c]):
                    ps.append(c)
                continue
            info[c]["last"] = float(p)
            info[c]["peak"] = max(info[c]["peak"], float(p))
            if sell_fn(i, c, float(p), info[c]["peak"]):
                ps.append(c)
        free = N - (len(hold) - len(ps))
        if free > 0:
            cand = [c for c in np.where(ok[i])[0] if c not in hold and c not in ps]
            cand.sort(key=lambda c: -r12n[i, c])
            pb = cand[:free]
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1], trades


def rs_rating(px, mem):
    """仿 IBD RS Rating：同當日成分股比嘅百分位（1-99）。非成分股（例如揸住期間被剔出）都按成分股分佈計。"""
    score = (0.4 * px / px.shift(63) + 0.2 * px / px.shift(126) + 0.2 * px / px.shift(189) + 0.2 * px / px.shift(252)).values
    out = np.full(score.shape, np.nan)
    for i in range(score.shape[0]):
        ref = score[i][mem[i] & np.isfinite(score[i])]
        if len(ref) < 50:
            continue
        ref = np.sort(ref)
        row = score[i]
        f = np.isfinite(row)
        out[i, f] = np.clip(np.searchsorted(ref, row[f], side="right") / len(ref) * 99, 1, 99)
    return out


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
    base_ok = mem & (r12n >= 1.0) & (P > ma50) & (P > ma200) & np.isfinite(P)
    weak = (spy < spy.rolling(200).mean()).values
    rs = rs_rating(px, mem)
    log("數據準備好（包括RS Rating）")

    def fin(x):
        return np.isfinite(x)

    def e0(i, c, p, pk):
        return p <= pk * 0.8

    def e1(i, c, p, pk):
        if weak[i] and fin(ma50[i, c]) and p > ma50[i, c]:
            return False
        return p <= pk * 0.8

    def e2(i, c, p, pk):
        r = rs[i, c]
        return (fin(r) and r < 70) or p <= pk * 0.7

    def e3(i, c, p, pk):
        r = rs[i, c]
        a = fin(ma50[i, c]) and p < ma50[i, c] and fin(r) and r < 80
        b = fin(ma200[i, c]) and p < ma200[i, c]
        return a or b or p <= pk * 0.7

    def e4(i, c, p, pk):
        if p <= pk * 0.7:
            return True
        r = rs[i, c]
        if fin(r) and r >= 80 and fin(ma50[i, c]) and p > ma50[i, c]:
            return False
        return p <= pk * 0.8

    V = {}
    for nm, fn in (("E0 線D基準（跌20%賣）", e0), ("E1 大市弱但股價仲喺MA50上面就唔賣", e1), ("E2 RS<70先賣（保險跌30%）", e2),
                   ("E3 狀態判斷（MA50+RS<80 / MA200 / 跌30%）", e3), ("E4 領導股（RS≥80且>MA50）唔理跌20%（保險跌30%）", e4)):
        V[nm] = sim(P, rets_v, cr, base_ok, r12n, fn)
        log(f"{nm} 完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2019", "2010-01-01", "2019-12-31"),
               ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END), ("科網爆煲", "2000-03-01", "2002-12-31"),
               ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)
    tab = {}
    L = ["=== 第二十三輪：睇大市同 RS Rating 先決定賣唔賣（規則預先登記，見research_v23.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻；只改賣出", ""]
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
        L.append(f"　{nm}：{len(tr)} 次 ｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
                 f"蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}%")
    L.append("")
    q, s = tab[("驗證期", "QQQ長揸")], tab[("驗證期", "SPY長揸")]
    k0 = "E0 線D基準（跌20%賣）"
    b0, e00, f00 = tab[("驗證期", k0)], tab[("2000-2009", k0)], tab[("全期", k0)]
    L.append(f"=== 預先登記判斷（同 E0 比，驗證期）｜ E0 {b0['cagr']:+.2f}% / {b0['mdd']:.1f}%（應該 +21.38% / -33.8%）｜ "
             f"QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% ｜ SPY {s['cagr']:+.2f}% ===")
    for nm in V:
        if nm == k0:
            continue
        x, e, f = tab[("驗證期", nm)], tab[("2000-2009", nm)], tab[("全期", nm)]
        better = (x["mdd"] >= b0["mdd"] + 3 and x["cagr"] >= b0["cagr"] - 1) or (x["cagr"] >= b0["cagr"] + 2 and x["mdd"] >= b0["mdd"] - 2)
        q1 = x["cagr"] > q["cagr"]
        q2 = q1 and x["mdd"] > -35 and x["sharpe"] >= q["sharpe"]
        q3 = (not q1) and x["mdd"] >= q["mdd"] + 5 and x["cagr"] >= s["cagr"]
        q4 = e["mdd"] > -35
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} → {'有改善' if better else '冇改善'} ｜ "
                 f"2000-2009 {e['cagr']:+.2f}% / {e['mdd']:.1f}%（E0 {e00['cagr']:+.2f}% / {e00['mdd']:.1f}%）｜ "
                 f"全期 {f['cagr']:+.2f}% / {f['mdd']:.1f}%（E0 {f00['cagr']:+.2f}% / {f00['mdd']:.1f}%）｜ "
                 f"四條問題 ①{'✔' if q1 else '✘'} ②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'}")
    L.append("")

    # ---------- 描述：大市弱時，倍升股喺MA50上定下，之後點 ----------
    L.append("=== 描述：倍升股（12個月 ≥ +100%）之後表現，每5個交易日抽一次（2000-2026）===")
    S = spy.values
    rows = []
    for i in range(260, len(days) - 63, 5):
        if days[i] < pd.Timestamp("2000-01-01"):
            continue
        cs = np.where(mem[i] & (r12n[i] >= 1.0) & np.isfinite(P[i]) & np.isfinite(ma50[i]))[0]
        for c in cs:
            for h in (21, 63):
                fwd = P[i + h, c] / P[i, c] - 1 if np.isfinite(P[i + h, c]) else np.nan
                rows.append((bool(weak[i]), bool(P[i, c] > ma50[i, c]), bool(np.isfinite(rs[i, c]) and rs[i, c] >= 90), h, fwd,
                             S[i + h] / S[i] - 1))
    D = pd.DataFrame(rows, columns=["weak", "above50", "rs90", "h", "fwd", "spy"]).dropna()

    def desc(sub):
        if len(sub) < 30:
            return "樣本唔夠"
        return (f"樣本 {len(sub):,} ｜ 中位數 {sub['fwd'].median() * 100:+.1f}% ｜ 平均 {sub['fwd'].mean() * 100:+.1f}% ｜ "
                f"跌≥20% {(sub['fwd'] <= -0.2).mean() * 100:.0f}% ｜ 贏SPY {(sub['fwd'] > sub['spy']).mean() * 100:.0f}%")
    for h, lab in ((21, "之後1個月"), (63, "之後3個月")):
        L.append(f"　【{lab}】")
        for wk, wl in ((True, "大市弱（SPY < 200日線）"), (False, "大市強（SPY > 200日線）")):
            d0 = D[(D["h"] == h) & (D["weak"] == wk)]
            L.append(f"　　{wl}：股價 > MA50：{desc(d0[d0['above50']])}")
            L.append(f"　　{wl}：股價 < MA50：{desc(d0[~d0['above50']])}")
            L.append(f"　　{wl}：RS ≥ 90：{desc(d0[d0['rs90']])}")
            L.append(f"　　{wl}：RS < 90：{desc(d0[~d0['rs90']])}")
            L.append(f"　　{wl}：股價 > MA50 而且 RS ≥ 90：{desc(d0[d0['above50'] & d0['rs90']])}")
    L.append("　（同一隻股票每5日抽一次，樣本互相重疊，所以樣本數睇落多過實際；只作參考。）")
    pd.DataFrame({"report": L}).to_csv("v23_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
