"""
research_v24.py — 第二十四輪：ChatGPT 四個方向（倉位結構、倍升股甜蜜區、第一次 vs 第二次倍升、持有期）
（Esther 2026-10-08：「要做」）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同（同線D一樣，除非下面講明）：當時真實標普500；已計股息；收市決定 → 下一個交易日收市買賣；每次買賣0.1%；
  最多5隻；冇投資嘅錢揸現金（^IRX）；有空位揀12個月升幅最大。
  基準 T0 = 線D：12個月升 ≥ +100% + 收市 > MA50 + 收市 > MA200；每隻 = 買入日總資產 ÷ 5；由最高收市價跌20%就賣。

━━ A 倉位／風險結構（揀股規則完全唔改，只改點樣分錢）━━
  A1 按波幅分錢：每隻 = 總資產 × (12% ÷ 60日年化波幅)，限制喺 10%-30% 之間（唔夠現金就有幾多用幾多）。
  A2 行業上限：同一個 GICS 板塊最多揸2隻（板塊用今日分類：Wikipedia 標普500 + Nasdaq；已除牌冇分類 = 唔限制）。
  A3 相關性上限：新股同任何一隻持倉過去60日每日回報相關 > 0.7 → 唔買，揀下一隻。
  A4 線D2（SPY每日訊號 = 持有先開新倉；第二十二輪X2，作對照）。
  A5 線D2 + A2 + A3（大市過濾 + 行業上限 + 相關性上限一齊用）。
━━ B 倍升股甜蜜區（只改「12個月升幅」範圍，其他同T0）━━
  B1 +50%至+100%   B2 +100%至+150%   B3 +150%至+200%   B4 +200%至+300%
━━ C 第一次 vs 第二次倍升（T0 買入條件 + 「之前一年」嘅升幅，即 252至504日前）━━
  C1 之前一年跌（< 0%）：今年先爆   C2 之前一年 0%至+100%   C3 之前一年都 ≥ +100%（第二次倍升）
  另外（描述）：每月尾符合 T0 買入條件嘅股票，按 C1/C2/C3 分組，之後 3/6/12/24 個月：中位數、平均、贏SPY機率、跌≥30%機率。
━━ D 持有期（買入同T0，只改賣出）━━
  D1 跌25%、D2 跌30%（第十五輪C已經測過，今次只係用新標準再列一次）
  D3 固定揸3個月（63個交易日）  D4 固定揸6個月（126日）  D5 固定揸12個月（252日）（冇其他賣出規則）
  D6 跌20%，但買入後頭60個交易日唔理（最少揸60日）

判斷：
  ① 「有改善」（同T0比，驗證期 2015-01 至 2026-09，同之前一樣）：最大跌幅改善 ≥ 3個百分點而且年化唔少過1個百分點；
     或者年化多 ≥ 2個百分點而且最大跌幅差唔超過2個百分點。
  ② 新增「跨年代合格」（Esther 同意，由今輪開始）：2000-2009、2010-2014、2015-2019、2020-2026 四段，
     每段年化 ≥ 同期SPY長揸，而且每段最大跌幅好過 -35% → 四段全部過先叫「跨年代合格」。
  另外列：最差5年年化（由2000年起每個月開始揸5年）、每次買賣統計。
  18個版本：試得多總會有幾個好彩，有改善都要模擬交易再驗證。
輸出：v24_results.csv、v24_progress.txt
"""
import io

import numpy as np
import pandas as pd
import requests

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash

M.PROGRESS = "v24_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"
N = 5


def sectors():
    out = {}
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=M.HDR, timeout=30).text
        t = pd.read_html(io.StringIO(html))[0]
        for s, sec in zip(t["Symbol"], t["GICS Sector"]):
            out[M.norm(s)] = sec
        log(f"Wikipedia GICS 板塊：{len(out)} 隻")
    except Exception as e:
        log(f"Wikipedia 板塊攞唔到：{e}")
    try:
        r = requests.get("https://api.nasdaq.com/api/screener/stocks", params={"tableonly": "true", "download": "true"},
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"},
                         timeout=60)
        n = 0
        for row in r.json()["data"]["rows"]:
            k = M.norm(row["symbol"])
            if k not in out and row.get("sector"):
                out[k] = row["sector"]
                n += 1
        log(f"Nasdaq 板塊補充：{n} 隻")
    except Exception as e:
        log(f"Nasdaq 板塊攞唔到：{e}")
    return out


def sim(P, rets_v, cr, ok, rank, trail=0.2, fixed=None, min_hold=0, allow=None, weight_fn=None, sec=None, corr_cap=None):
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
            w = weight_fn(i - 1, c) if weight_fn else 1.0 / N
            amt = min(total * w, cash)
            if amt <= total * 0.01:
                break
            cash -= amt
            hold[c] = amt * (1 - COST)
            info[c] = {"px": float(P[i, c]), "peak": float(P[i, c]), "last": float(P[i, c]), "i": i}
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
            age = i - info[c]["i"]
            if fixed is not None:
                if age >= fixed:
                    ps.append(c)
            elif age >= min_hold and p <= info[c]["peak"] * (1 - trail):
                ps.append(c)
        if allow is not None and not allow[i]:
            continue
        free = N - (len(hold) - len(ps))
        if free > 0:
            cand = [c for c in np.where(ok[i])[0] if c not in hold and c not in ps]
            cand.sort(key=lambda c: -rank[i, c])
            keep = [c for c in hold if c not in ps]
            for c in cand:
                if len(pb) >= free:
                    break
                group = keep + pb
                if sec is not None and sec[c] is not None and sum(1 for x in group if sec[x] == sec[c]) >= 2:
                    continue
                if corr_cap is not None and group and i >= 60:
                    a = rets_v[i - 59:i + 1, c]
                    bad = False
                    for x in group:
                        b = rets_v[i - 59:i + 1, x]
                        m = np.isfinite(a) & np.isfinite(b)
                        if m.sum() >= 40 and np.corrcoef(a[m], b[m])[0, 1] > corr_cap:
                            bad = True
                            break
                    if bad:
                        continue
                pb.append(c)
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1], trades


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
    r12 = (px / px.shift(252) - 1).values
    r12n = np.nan_to_num(r12, nan=-9)
    prev = np.nan_to_num((px.shift(252) / px.shift(504) - 1).values, nan=np.nan)
    vol60 = (px.pct_change(fill_method=None).rolling(60).std() * np.sqrt(252)).values
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
    trend = mem & (P > ma50) & (P > ma200) & np.isfinite(P)
    base_ok = trend & (r12n >= 1.0)
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    sig_ok = (df["state"] != "out").reindex(days).fillna(False).values
    sm = sectors()
    sec = [sm.get(t) for t in cols]
    log(f"有板塊分類：{sum(s is not None for s in sec)}/{len(cols)} 隻")
    log("數據準備好")

    def wvol(i, c):
        v = vol60[i, c]
        return float(np.clip(0.12 / v, 0.10, 0.30)) if np.isfinite(v) and v > 0 else 1.0 / N

    pv = np.nan_to_num(prev, nan=np.nan)
    V = {}
    run = [("T0 線D基準", dict(ok=base_ok)),
           ("A1 按波幅分錢", dict(ok=base_ok, weight_fn=wvol)),
           ("A2 同板塊最多2隻", dict(ok=base_ok, sec=sec)),
           ("A3 相關>0.7唔買", dict(ok=base_ok, corr_cap=0.7)),
           ("A4 線D2（SPY訊號持有先開新倉）", dict(ok=base_ok, allow=sig_ok)),
           ("A5 線D2+板塊上限+相關上限", dict(ok=base_ok, allow=sig_ok, sec=sec, corr_cap=0.7)),
           ("B1 12個月升50%-100%", dict(ok=trend & (r12n >= 0.5) & (r12n < 1.0))),
           ("B2 12個月升100%-150%", dict(ok=trend & (r12n >= 1.0) & (r12n < 1.5))),
           ("B3 12個月升150%-200%", dict(ok=trend & (r12n >= 1.5) & (r12n < 2.0))),
           ("B4 12個月升200%-300%", dict(ok=trend & (r12n >= 2.0) & (r12n < 3.0))),
           ("C1 之前一年跌、今年先爆", dict(ok=base_ok & np.nan_to_num(pv < 0.0, nan=False).astype(bool))),
           ("C2 之前一年升0%-100%", dict(ok=base_ok & np.nan_to_num((pv >= 0.0) & (pv < 1.0), nan=False).astype(bool))),
           ("C3 之前一年都升≥100%（第二次倍升）", dict(ok=base_ok & np.nan_to_num(pv >= 1.0, nan=False).astype(bool))),
           ("D1 跌25%賣", dict(ok=base_ok, trail=0.25)),
           ("D2 跌30%賣", dict(ok=base_ok, trail=0.30)),
           ("D3 固定揸3個月", dict(ok=base_ok, fixed=63)),
           ("D4 固定揸6個月", dict(ok=base_ok, fixed=126)),
           ("D5 固定揸12個月", dict(ok=base_ok, fixed=252)),
           ("D6 跌20%但最少揸60日", dict(ok=base_ok, min_hold=60))]
    for nm, kw in run:
        ok = kw.pop("ok")
        V[nm] = sim(P, rets_v, cr, ok, r12n, **kw)
        log(f"{nm} 完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)
    tab = {}
    L = ["=== 第二十四輪：倉位結構、倍升股甜蜜區、第一次vs第二次倍升、持有期（規則預先登記，見research_v24.py開頭）===",
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
            L.append(f"　{nm}：{len(tr)} 次 ｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
                     f"蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}%")
        else:
            L.append(f"　{nm}：冇買賣")
    L.append("")

    def worst5(r):
        nav = (1 + r.loc["2000-01-03":END].fillna(0.0)).cumprod()
        out = []
        for d0 in decision_days(nav.index, "每月"):
            d1 = d0 + pd.DateOffset(years=5)
            if d1 > nav.index[-1]:
                break
            j0, j1 = nav.index.get_loc(d0), nav.index.searchsorted(d1) - 1
            out.append((nav.iat[j1] / nav.iat[j0]) ** (1 / 5) - 1)
        return min(out) * 100 if out else np.nan

    eras = ["2000-2009", "2010-2014", "2015-2019", "2020-2026"]
    b0 = tab[("驗證期", "T0 線D基準")]
    q = tab[("驗證期", "QQQ長揸")]
    L.append(f"=== 預先登記判斷 ｜ T0 驗證期 {b0['cagr']:+.2f}% / {b0['mdd']:.1f}%（應該 +21.38% / -33.8%）｜ QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% ===")
    L.append("　跨年代：每段年化 ≥ SPY 而且最大跌幅好過 -35%（✔/✘ 順序：2000-09、2010-14、2015-19、2020-26）")
    for nm in ["QQQ長揸", "SPY長揸"] + list(V):
        x = tab[("驗證期", nm)]
        marks = []
        for e in eras:
            v, s_ = tab[(e, nm)], tab[(e, "SPY長揸")]
            marks.append("✔" if (v["cagr"] >= s_["cagr"] and v["mdd"] > -35) else "✘")
        cross = all(m == "✔" for m in marks)
        if nm in V and nm != "T0 線D基準":
            better = (x["mdd"] >= b0["mdd"] + 3 and x["cagr"] >= b0["cagr"] - 1) or (x["cagr"] >= b0["cagr"] + 2 and x["mdd"] >= b0["mdd"] - 2)
            imp = "有改善" if better else "冇改善"
        else:
            imp = "—"
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} → {imp} ｜ 跨年代 {''.join(marks)} "
                 f"{'跨年代合格' if cross else '唔合格'} ｜ 各段年化 " + " / ".join(f"{tab[(e, nm)]['cagr']:+.1f}%" for e in eras)
                 + f" ｜ 各段最大跌幅 " + " / ".join(f"{tab[(e, nm)]['mdd']:.0f}%" for e in eras)
                 + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
    L.append("")

    # ---------- C 描述 ----------
    L.append("=== 描述：每月尾符合線D買入條件嘅股票，按「之前一年」分組，之後表現（2000-2026）===")
    S = spy.values
    rows = []
    for d in decision_days(days, "每月"):
        i = days.get_loc(d)
        if d < pd.Timestamp("2000-01-01"):
            continue
        for c in np.where(base_ok[i])[0]:
            pr = pv[i, c]
            if not np.isfinite(pr):
                continue
            g = "C1 之前一年跌" if pr < 0 else ("C2 之前一年升0-100%" if pr < 1.0 else "C3 第二次倍升")
            for h in (63, 126, 252, 504):
                if i + h < len(days) and np.isfinite(P[i + h, c]):
                    rows.append((g, h, P[i + h, c] / P[i, c] - 1, S[i + h] / S[i] - 1))
    D = pd.DataFrame(rows, columns=["g", "h", "fwd", "spy"])
    for h, lab in ((63, "3個月"), (126, "6個月"), (252, "12個月"), (504, "24個月")):
        L.append(f"　【之後{lab}】")
        for g in ("C1 之前一年跌", "C2 之前一年升0-100%", "C3 第二次倍升"):
            sub = D[(D["h"] == h) & (D["g"] == g)]
            if len(sub) < 20:
                L.append(f"　　{g}：樣本唔夠（{len(sub)}）")
                continue
            L.append(f"　　{g}：樣本 {len(sub)} ｜ 中位數 {sub['fwd'].median() * 100:+.1f}% ｜ 平均 {sub['fwd'].mean() * 100:+.1f}% ｜ "
                     f"贏SPY {(sub['fwd'] > sub['spy']).mean() * 100:.0f}% ｜ 跌≥30% {(sub['fwd'] <= -0.3).mean() * 100:.0f}%")
    L.append("　（每月抽一次，同一隻股票連續幾個月都會出現，樣本互相重疊；只作參考。）")
    pd.DataFrame({"report": L}).to_csv("v24_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
