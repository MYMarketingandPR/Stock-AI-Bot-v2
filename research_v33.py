"""
research_v33.py — 第三十三輪：等回落先買、同行業最多2隻、睇行業指標（Esther 2026-10-15）
  「想等股價回落先買」「我哋買嗰個行業嘅，可以同時參考吓嗰個行業嘅指標……升得太高係咪唔應該買入？或者佢開始轉弱？
   如果買入係咪應該要離場？而唔係淨係睇SPY」「同類股上限……呢個可以試埋」

純研究，唔改任何模擬交易（線A/B/C/D/D2/E、輪動、SPY訊號全部照舊），唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同設定同第二十七、二十八輪一樣：當時真實標普500（fja05680）、已計股息、收市決定→下一個交易日收市、每次買賣0.1%、
  揸5隻平均分、冇投資嘅錢收短期國債息（^IRX）。
兩個對照：線D（12個月 ≥ +100%、喺MA50同MA200上面，揀12個月升最多；由買入後最高收市跌20%賣）
         線E（今年 ≥ +30%、≥ 52週最高收市 × 0.7、喺MA200上面，揀今年升最多；跌穿MA200或者低過52週最高收市 × 0.7 就賣）

行業：用今日分類（Wikipedia 標普500 GICS Sector；冇嘅用 Nasdaq screener sector 對返 GICS）。已除牌冇分類 = 「唔知」。
行業指標 = SPDR 行業ETF（已計股息）：資訊科技 XLK、醫療 XLV、金融 XLF、非必需消費 XLY、必需消費 XLP、工業 XLI、
  原材料 XLB、能源 XLE、公用 XLU；通訊服務 XLC（2018-06前用XLK，因為當時大部分通訊股屬資訊科技）；
  房地產 XLRE（2015-10前用XLF，因為當時房託屬金融）。接駁方法：每日回報接駁成一條指數。

新版本（每個都分別套落線D同線E，一共10個；只有呢10個，跑完唔加唔減唔改數字）：
  P1 等回落先買：買入條件再加「收市 ≤ 20日平均 × 1.03」（即係回落到20日線附近先買；其他條件照舊）
  P2 同行業最多2隻：揀股時，同一行業（已揸住而唔係賣緊 + 今日已揀咗）夠2隻就跳過，揀下一隻。「唔知」行業唔限。
  P3 行業過熱唔買：行業指標收市 ≥ 行業指標200日平均 × 1.20 → 嗰個行業嘅股票唔買（已揸嘅照舊）。「唔知」當過關。
  P4 行業轉弱唔買：行業指標收市 < 行業指標200日平均 → 嗰個行業嘅股票唔買（已揸嘅照舊）。「唔知」當過關。
  P5 行業轉弱就賣：P4 + 已揸嘅股票，如果佢行業指標收市 < 200日平均，就賣（同原本賣出規則一齊用，邊個先到用邊個）。

判斷（同第二十八輪一樣，只係「有改善」改為同自己嗰條對照線比）：
  驗證期 2015-01 至 2026-09。
  有改善：最大跌幅好過對照 ≥3個百分點而年化唔低過對照超過1點；或者年化高過對照 ≥2點而最大跌幅唔差過對照超過2點。
  四條問題（同QQQ/SPY比）；跨年代合格（2000-09、2010-14、2015-19、2020-26 每段年化 ≥ SPY 而且最大跌幅好過 -35%）。
  另外列：最差5年年化、每次買賣統計、平均同一行業最多揸幾隻。
  10個版本一齊比，試得多總有好彩 → 要「有改善」+ 跨年代都企得穩先值得考慮；就算合格都唔會直接改模擬交易，要你同意先加新線。

已知限制：行業用今日分類（有啲公司幾十年前屬其他行業）；已除牌股多數冇分類；Yahoo 冇數據嘅已除牌股唔會入選。

輸出：v33_results.csv、v33_progress.txt
==== 結果（2026-10-15 第一次跑，run 38720392749，喺新 repo Stock-AI-Bot-v2 跑；記錄低，唔再改）====
- 數據：716隻股票有價（之前幾輪721隻，今次Yahoo少咗5隻）→ 線D對照變咗 +22.67% / -33.8%（之前記錄 +21.38%）；
  線E對照 +28.74% / -32.6% 同之前一樣。今輪全部版本用同一批數據，所以比較係公平嘅；之前結果唔改。
- 有行業分類 95%（677/716）。行業「過熱」（≥200日線×1.2）好少見：資訊科技 5% 日子、能源 7%，其他 0-3%。
- 線D（驗證期 年化 / 最大跌幅）：P1 等回落 +19.63% / -37.8%；P2 同行業≤2 +21.86% / -33.4%；P3 過熱唔買 +22.61% / -33.0%；
  P4 轉弱唔買 +21.12% / -35.3%；P5 轉弱就賣 +25.11% / -39.0%（買賣次數 240 → 397）。全部「冇改善」，跨年代全部唔合格。
- 線E：P1 +19.06% / -36.3%；P2 +25.75% / -35.7%；P3 +26.57% / -36.1%；P4 +29.02% / -32.9%（同對照差唔多）；
  P5 +16.05% / -35.0%（買賣次數 279 → 494）。全部「冇改善」，跨年代全部唔合格。
- 意思：等回落先買會錯過最強嘅股（最好一單由 +669% 跌到 +368%）；同行業上限、行業過熱/轉弱過濾效果好細或者更差；
  行業轉弱就賣令買賣次數多咗七成，賣完又升返（假訊號多）。平均同一行業最多揸 2.1-2.4 隻，過去集中一個行業唔係常態。
"""

import io

import numpy as np
import pandas as pd
import requests

import momentum_v1 as M
from perf import perf
from research_v8 import decision_days
from research_v13 import get_cash

M.PROGRESS = "v33_progress.txt"
log = M.log
END = "2026-09-30"
COST = 0.001
ETF = {"Information Technology": "XLK", "Health Care": "XLV", "Financials": "XLF", "Consumer Discretionary": "XLY",
       "Consumer Staples": "XLP", "Industrials": "XLI", "Materials": "XLB", "Energy": "XLE", "Utilities": "XLU",
       "Communication Services": "XLC", "Real Estate": "XLRE"}
FALLBACK = {"XLC": "XLK", "XLRE": "XLF"}
NASDAQ2GICS = {"Technology": "Information Technology", "Telecommunications": "Communication Services",
               "Health Care": "Health Care", "Finance": "Financials", "Real Estate": "Real Estate",
               "Consumer Discretionary": "Consumer Discretionary", "Consumer Staples": "Consumer Staples",
               "Industrials": "Industrials", "Basic Materials": "Materials", "Energy": "Energy", "Utilities": "Utilities"}


def sectors():
    out = {}
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=M.HDR, timeout=30).text
        t = pd.read_html(io.StringIO(html))[0]
        for s, sec in zip(t["Symbol"], t["GICS Sector"]):
            if sec in ETF:
                out[M.norm(s)] = sec
        log(f"Wikipedia GICS：{len(out)} 隻")
    except Exception as e:
        log(f"Wikipedia 攞唔到：{e}")
    try:
        r = requests.get("https://api.nasdaq.com/api/screener/stocks", params={"tableonly": "true", "download": "true"},
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"},
                         timeout=60)
        n = 0
        for row in r.json()["data"]["rows"]:
            k = M.norm(row["symbol"])
            g = NASDAQ2GICS.get(row.get("sector") or "")
            if g and k not in out:
                out[k] = g
                n += 1
        log(f"Nasdaq 補充：{n} 隻")
    except Exception as e:
        log(f"Nasdaq 攞唔到：{e}")
    return out


def sim(P, rets_v, cr, ok, rank, sell_fn, sec=None, cap=None, n=5):
    """同第二十六輪 sim 一樣；多咗 cap：同一行業最多揸 cap 隻（sec[c] = 行業編號，-1 = 唔知，唔限）。"""
    T = P.shape[0]
    hold, info, cash = {}, {}, 1.0
    ps, pb, navs, trades, maxsame = [], [], [], [], []
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
            if c in hold or len(hold) >= n or not np.isfinite(P[i, c]):
                continue
            amt = min(total / n, cash)
            if amt <= total * 0.01:
                break
            cash -= amt
            hold[c] = amt * (1 - COST)
            info[c] = {"px": float(P[i, c]), "peak": float(P[i, c]), "last": float(P[i, c])}
        ps, pb = [], []
        navs.append(cash + sum(hold.values()))
        if sec is not None and hold:
            ks = [sec[c] for c in hold if sec[c] >= 0]
            maxsame.append(max([ks.count(k) for k in set(ks)] or [1]))
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
        free = n - (len(hold) - len(ps))
        if free > 0:
            cand = [c for c in np.where(ok[i])[0] if c not in hold and c not in ps]
            cand.sort(key=lambda c: -rank[i, c])
            if cap is None:
                pb = cand[:free]
            else:
                cnt = {}
                for c in hold:
                    if c not in ps and sec[c] >= 0:
                        cnt[sec[c]] = cnt.get(sec[c], 0) + 1
                for c in cand:
                    if len(pb) >= free:
                        break
                    k = sec[c]
                    if k >= 0 and cnt.get(k, 0) >= cap:
                        continue
                    pb.append(c)
                    if k >= 0:
                        cnt[k] = cnt.get(k, 0) + 1
    nav = np.array(navs)
    return np.r_[0.0, nav[1:] / nav[:-1] - 1], trades, (float(np.mean(maxsame)) if maxsame else np.nan)


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    etfs = sorted(set(ETF.values()))
    closes = M.download_close(everyone + ["SPY", "QQQ"] + etfs, "1998-01-01", "2026-10-01")
    spy, qqq = closes.pop("SPY"), closes.pop("QQQ")
    etf_px = {e: closes.pop(e) for e in etfs if e in closes}
    log(f"行業ETF有數據：{sorted(etf_px)}")
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    # 行業指標：每日回報接駁（XLC/XLRE 未有之前用 XLK/XLF）
    sect_idx = {}
    for g, e in ETF.items():
        main_r = etf_px[e].reindex(days).pct_change(fill_method=None) if e in etf_px else pd.Series(np.nan, index=days)
        if e in FALLBACK:
            fb = etf_px[FALLBACK[e]].reindex(days).pct_change(fill_method=None)
            r = main_r.where(main_r.notna(), fb)
        else:
            r = main_r
        sect_idx[g] = (1 + r.fillna(0.0)).cumprod()
        first = etf_px.get(e if e not in FALLBACK else FALLBACK[e])
        if first is not None:
            sect_idx[g][days < first.dropna().index[0]] = np.nan
    px = pd.DataFrame(closes).reindex(days)
    cols = list(px.columns)
    P = px.values
    rets_v = px.pct_change(fill_method=None).values
    ma20, ma50, ma200 = px.rolling(20).mean().values, px.rolling(50).mean().values, px.rolling(200).mean().values
    r12n = np.nan_to_num((px / px.shift(252) - 1).values, nan=-9)
    ye = px.groupby(days.year).last()
    ytdn = np.nan_to_num(P / ye.shift(1).reindex(days.year).values - 1, nan=-9)
    hi52 = px.rolling(252, min_periods=200).max().values
    hi52f = np.nan_to_num(hi52, nan=np.inf)
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

    smap = sectors()
    names = list(ETF)
    sec = np.array([names.index(smap[t]) if t in smap else -1 for t in cols])
    known = (sec >= 0).mean() * 100
    log(f"有行業分類：{known:.0f}% 嘅股票（{(sec >= 0).sum()}/{len(cols)}）")
    S = np.column_stack([sect_idx[g].values for g in names])           # 日 × 行業
    Sma = pd.DataFrame(S).rolling(200).mean().values
    hot_s = np.nan_to_num(S >= Sma * 1.20, nan=False) & np.isfinite(Sma)
    weak_s = np.nan_to_num(S < Sma, nan=False) & np.isfinite(Sma)
    # 擴展到 日 × 股票（唔知行業 = False，即係過關）
    hot = np.zeros(P.shape, dtype=bool)
    weak = np.zeros(P.shape, dtype=bool)
    kn = sec >= 0
    hot[:, kn] = hot_s[:, sec[kn]]
    weak[:, kn] = weak_s[:, sec[kn]]

    d_ok = mem & (r12n >= 1.0) & (P > ma50) & (P > ma200) & fin
    e_ok = mem & (ytdn >= 0.30) & (P >= hi52f * 0.7) & (P > ma200) & fin
    pull = np.nan_to_num(P <= ma20 * 1.03, nan=False)
    log("數據準備好")

    def trail20(i, c, p, pk):
        return p <= pk * 0.8

    def broken(i, c, p, pk):
        a = np.isfinite(ma200[i, c]) and p < ma200[i, c]
        b = np.isfinite(hi52[i, c]) and p < hi52[i, c] * 0.7
        return a or b

    def plus_weak(f):
        return lambda i, c, p, pk: f(i, c, p, pk) or bool(weak[i, c])

    V = {}
    for base, ok, rank, sell in [("線D", d_ok, r12n, trail20), ("線E", e_ok, ytdn, broken)]:
        V[f"{base}（對照）"] = sim(P, rets_v, cr, ok, rank, sell, sec=sec)
        V[f"{base}・P1 等回落先買"] = sim(P, rets_v, cr, ok & pull, rank, sell, sec=sec)
        V[f"{base}・P2 同行業最多2隻"] = sim(P, rets_v, cr, ok, rank, sell, sec=sec, cap=2)
        V[f"{base}・P3 行業過熱唔買"] = sim(P, rets_v, cr, ok & ~hot, rank, sell, sec=sec)
        V[f"{base}・P4 行業轉弱唔買"] = sim(P, rets_v, cr, ok & ~weak, rank, sell, sec=sec)
        V[f"{base}・P5 行業轉弱就賣"] = sim(P, rets_v, cr, ok & ~weak, rank, plus_weak(sell), sec=sec)
        log(f"{base} 全部版本完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)
    tab = {}
    L = ["=== 第三十三輪：等回落先買、同行業最多2隻、行業指標（規則預先登記，見research_v33.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻；有行業分類 {known:.0f}%", ""]
    for g in names:
        s = pd.Series(S[:, names.index(g)], index=days).loc["2000-01-03":END]
        m = pd.Series(Sma[:, names.index(g)], index=days).loc["2000-01-03":END]
        ok_ = m.notna()
        L.append(f"　{g}：過熱（≥200日線×1.2）日子 {(s[ok_] >= m[ok_] * 1.2).mean() * 100:.0f}% ｜ 轉弱（<200日線）日子 {(s[ok_] < m[ok_]).mean() * 100:.0f}%")
    L.append("")
    for wn, a, b in windows:
        L.append(f"=== {wn}（{a} 至 {b}）===")
        for nm, r in allser.items():
            rr = r.loc[a:b].dropna()
            p = perf(rr, rf=cash_r, bench=qqq_r.loc[a:b])
            tab[(wn, nm)] = p
            L.append(f"　{nm}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f}")
        L.append("")
    L.append("=== 每次買賣（1999-2026）+ 平均同一行業最多揸幾隻 ===")
    for nm, v in V.items():
        tr = np.array(v[1])
        L.append(f"　{nm}：{len(tr)} 次 ｜ 賺錢 {(tr > 0).mean() * 100:.0f}% ｜ 平均 {tr.mean() * 100:+.1f}% ｜ "
                 f"蝕嘅平均 {tr[tr <= 0].mean() * 100:+.1f}% ｜ 最差 {tr.min() * 100:+.0f}% ｜ 最好 {tr.max() * 100:+.0f}% ｜ 同行業最多 {v[2]:.2f} 隻")
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
        return min(out) * 100

    eras = ["2000-2009", "2010-2014", "2015-2019", "2020-2026"]
    q, s = tab[("驗證期", "QQQ長揸")], tab[("驗證期", "SPY長揸")]
    for base in ["線D", "線E"]:
        b0 = tab[("驗證期", f"{base}（對照）")]
        L.append(f"=== 預先登記判斷：{base} ｜ 對照 {b0['cagr']:+.2f}% / {b0['mdd']:.1f}% ｜ QQQ {q['cagr']:+.2f}% / {q['mdd']:.1f}% / Sharpe {q['sharpe']:.2f} ｜ SPY {s['cagr']:+.2f}% ===")
        for nm in [k for k in V if k.startswith(base)]:
            x, e = tab[("驗證期", nm)], tab[("2000-2009", nm)]
            marks = "".join("✔" if (tab[(er, nm)]["cagr"] >= tab[(er, "SPY長揸")]["cagr"] and tab[(er, nm)]["mdd"] > -35) else "✘" for er in eras)
            imp = "—"
            if "對照" not in nm:
                better = (x["mdd"] >= b0["mdd"] + 3 and x["cagr"] >= b0["cagr"] - 1) or (x["cagr"] >= b0["cagr"] + 2 and x["mdd"] >= b0["mdd"] - 2)
                imp = "有改善" if better else "冇改善"
            q1 = x["cagr"] > q["cagr"]
            q2 = q1 and x["mdd"] > -35 and x["sharpe"] >= q["sharpe"]
            q3 = (not q1) and x["mdd"] >= q["mdd"] + 5 and x["cagr"] >= s["cagr"]
            q4 = e["mdd"] > -35
            L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} ｜ 同{base}比 {imp} ｜ "
                     f"四條問題 ①{'✔' if q1 else '✘'} ②{'✔' if q2 else ('—' if not q1 else '✘')} ③{'✔' if q3 else ('—' if q1 else '✘')} ④{'✔' if q4 else '✘'} ⇒ "
                     f"{'保留' if (q2 or q3) else '淘汰'} ｜ 跨年代 {marks} {'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 "
                     + " / ".join(f"{tab[(er, nm)]['cagr']:+.1f}%" for er in eras) + " ｜ 各段最大跌幅 "
                     + " / ".join(f"{tab[(er, nm)]['mdd']:.0f}%" for er in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
        L.append("")
    pd.DataFrame({"report": L}).to_csv("v33_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
