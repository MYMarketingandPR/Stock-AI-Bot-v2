"""
research_multibagger.py — 「幾倍股」研究（純描述，唔係揀策略；跑完唔調任何嘢）

問題（Esther 2026-10-05）：SNDK、MU、LITE 呢啲幾倍股係點搵出嚟？專門搵幾倍股，回報會唔會好過QQQ？

做法：每年1月第一個交易日（2003-2025），用當時標普500成分股，睇之後12個月（252個交易日）：
① 基本機率：每年有幾多%股票12個月內升1倍或以上（「翻倍股」）、升2倍或以上（「3倍股」）
② 事前睇唔睇到：翻倍股喺年頭嗰陣，之前12個月升咗幾多、波幅幾大、離52週高位幾遠；
   按「之前12-1個月動量」同「之前60日波幅」分10組，每組翻倍機率係平均嘅幾多倍
③ 動量前10隻入面，第二年有幾多隻真係翻倍
④ 「已經翻咗倍先買」：年頭時過去12個月已經升≥100%嘅股票，第二年平均/中位數回報、再翻倍機率、跌≥30%機率
⑤ 三個「專搵幾倍股」組合（每年1月平均買入，揸足一年）：
   P1 = 過去12個月已經升≥100%嘅全部股票
   P2 = 60日波幅最大10隻（幾倍股通常都係大上大落嘅股）
   P3 = 動量前10隻（100%股票，每年換一次）
   同QQQ、SPY長揸比較：年化回報、最大跌幅（每日計）。
⚠️ 中途除牌嘅股票：Yahoo冇價錢嘅會漏咗（多數係跌到除牌或者被收購），所以①④⑤嘅結果多數偏好。
輸出：multibagger_results.csv、multibagger_progress.txt
==== 結果（未跑）====
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf

M.PROGRESS = "multibagger_progress.txt"
log = M.log
END = "2026-09-30"
H = 252


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    starts_y = list(range(2003, 2026))
    everyone = sorted(set().union(*[members_fn(pd.Timestamp(f"{y}-01-05")) for y in starts_y]))
    closes = M.download_close(everyone + ["SPY", "QQQ"], "2001-06-01", "2026-10-01")
    spy = closes.pop("SPY")
    qqq = closes.pop("QQQ")
    spy = spy[spy.index <= pd.Timestamp(END)]
    px = pd.DataFrame(closes).reindex(spy.index)
    days = px.index
    qq = qqq.reindex(days)
    rets = px.pct_change(fill_method=None)
    vol60 = rets.rolling(60).std() * np.sqrt(252)
    R121 = px.shift(21) / px.shift(252) - 1
    R12 = px / px.shift(252) - 1
    hi52 = px.rolling(252).max()

    rows = []
    port = {"P1 已翻倍股": [], "P2 波幅最大10隻": [], "P3 動量前10隻": [], "QQQ": [], "SPY": []}
    missing = 0
    for y in starts_y:
        i = days.searchsorted(pd.Timestamp(f"{y}-01-01"))
        e = min(i + H, len(days) - 1)
        d = days[i]
        m = [t for t in members_fn(d) if t in px.columns and np.isfinite(px[t].iat[i]) and np.isfinite(px[t].iat[i - 252])]
        fwd = {}
        for t in m:
            seg = px[t].iloc[i:e + 1]
            if not np.isfinite(seg.iloc[-1]):
                missing += 1
                last = seg.dropna()
                fwd[t] = last.iloc[-1] / seg.iloc[0] - 1 if len(last) else np.nan   # 中途冇價：用最後價（多數偏好）
            else:
                fwd[t] = seg.iloc[-1] / seg.iloc[0] - 1
        for t in m:
            rows.append({"year": y, "t": t, "fwd": fwd[t], "mom": R121[t].iat[i], "r12": R12[t].iat[i],
                         "vol": vol60[t].iat[i], "off_hi": px[t].iat[i] / hi52[t].iat[i] - 1})
        df = pd.DataFrame([r for r in rows if r["year"] == y]).set_index("t")
        picks = {"P1 已翻倍股": list(df.index[df["r12"] >= 1.0]),
                 "P2 波幅最大10隻": list(df["vol"].dropna().sort_values(ascending=False).index[:10]),
                 "P3 動量前10隻": list(df["mom"].dropna().sort_values(ascending=False).index[:10])}
        for k, tk in picks.items():
            if not tk:
                port[k].append(pd.Series(0.0, index=days[i + 1:e + 1]))       # 冇股就揸現金（當0%）
                continue
            seg = px[tk].iloc[i:e + 1].ffill()
            nav = (seg / seg.iloc[0]).mean(axis=1)
            r = nav.pct_change().iloc[1:]
            r.iloc[0] -= 0.001
            port[k].append(r)
        for k, s in (("QQQ", qq), ("SPY", spy)):
            seg = s.iloc[i:e + 1]
            port[k].append((seg / seg.iloc[0]).pct_change().iloc[1:])
        log(f"{y} 完成：{len(m)} 隻")
    D = pd.DataFrame(rows).dropna(subset=["fwd"])
    D["x2"] = D["fwd"] >= 1.0
    D["x3"] = D["fwd"] >= 2.0

    L = [f"=== 幾倍股研究（{source}；每年1月買入揸12個月；2003-2025共23年）===",
         f"股票樣本 {len(D)} 個（股票×年）；中途冇價錢（除牌/收購）{missing} 個，用最後價計", ""]
    by = D.groupby("year").agg(n=("t", "size"), x2=("x2", "sum"), x3=("x3", "sum"))
    L.append("=== ① 每年有幾多隻翻倍 ===")
    L.append(f"　平均每年：{by['n'].mean():.0f} 隻入面，翻倍 {by['x2'].mean():.1f} 隻（{D['x2'].mean() * 100:.1f}%），"
             f"3倍 {by['x3'].mean():.1f} 隻（{D['x3'].mean() * 100:.2f}%）")
    L.append("　每年翻倍隻數：" + "，".join(f"{y}:{int(v)}" for y, v in by["x2"].items()))
    L.append("")

    L.append("=== ② 翻倍股事前有冇特徵（年頭嗰陣嘅數字，中位數）===")
    for lab, g in (("之後翻倍", D[D["x2"]]), ("其他", D[~D["x2"]])):
        L.append(f"　{lab}（{len(g)}）：之前12-1個月 {g['mom'].median() * 100:+.0f}% ｜ 60日波幅 {g['vol'].median() * 100:.0f}% ｜ "
                 f"離52週高位 {g['off_hi'].median() * 100:.0f}%")
    base = D["x2"].mean()
    for col, lab in (("mom", "之前12-1個月動量"), ("vol", "之前60日波幅")):
        q = D.dropna(subset=[col]).copy()
        q["dec"] = q.groupby("year")[col].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
        g = q.groupby("dec").agg(p=("x2", "mean"), med=("fwd", "median"), down=("fwd", lambda s: (s <= -0.3).mean()))
        L.append(f"　按{lab}分10組（1=最低，10=最高）：翻倍機率係平均嘅幾多倍 ｜ 第二年中位數回報 ｜ 跌≥30%機率")
        for k, r in g.iterrows():
            L.append(f"　　第{int(k)}組：{r['p'] / base:.1f}倍（{r['p'] * 100:.1f}%）｜ 中位數 {r['med'] * 100:+.1f}% ｜ 跌≥30% {r['down'] * 100:.0f}%")
    L.append("")

    L.append("=== ③ 動量前10隻，第二年有幾多隻翻倍 ===")
    top = D.dropna(subset=["mom"]).sort_values("mom", ascending=False).groupby("year").head(10)
    L.append(f"　23年 × 10隻 = {len(top)} 隻次：翻倍 {int(top['x2'].sum())} 隻次（{top['x2'].mean() * 100:.1f}%）｜ "
             f"跌≥30% {(top['fwd'] <= -0.3).mean() * 100:.1f}% ｜ 平均 {top['fwd'].mean() * 100:+.1f}% ｜ 中位數 {top['fwd'].median() * 100:+.1f}%")
    caught = D[D["x2"]].copy()
    caught["in_top10"] = caught.index.isin(top.index)
    L.append(f"　所有翻倍股入面，年頭已經喺動量前10嘅：{int(caught['in_top10'].sum())}/{len(caught)}")
    L.append("")

    L.append("=== ④ 已經翻咗倍（過去12個月 ≥ +100%）先買，第二年點？ ===")
    a = D[D["r12"] >= 1.0]
    L.append(f"　{len(a)} 隻次：平均 {a['fwd'].mean() * 100:+.1f}% ｜ 中位數 {a['fwd'].median() * 100:+.1f}% ｜ "
             f"再翻倍 {a['x2'].mean() * 100:.1f}% ｜ 跌≥30% {(a['fwd'] <= -0.3).mean() * 100:.1f}%")
    L.append(f"　（對比全部股票：平均 {D['fwd'].mean() * 100:+.1f}% ｜ 中位數 {D['fwd'].median() * 100:+.1f}% ｜ "
             f"翻倍 {base * 100:.1f}% ｜ 跌≥30% {(D['fwd'] <= -0.3).mean() * 100:.1f}%）")
    L.append("")

    L.append("=== ⑤ 「專搵幾倍股」組合 vs QQQ（每年1月買，揸一年；最大跌幅用每日計）===")
    for k, parts in port.items():
        r = pd.concat(parts).sort_index()
        r = r[~r.index.duplicated()]
        for lab, a0, b0 in (("2003-2026", "2003-01-01", END), ("2015-2026", "2015-01-01", END)):
            p = perf(r.loc[a0:b0])
            L.append(f"　{k} {lab}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe(未扣利率) {p['sharpe']:.2f}")
    yr = {k: (1 + pd.concat(v)).groupby(pd.concat(v).index.year).prod() - 1 for k, v in port.items()}
    L.append("　每年回報（%）：年份 | " + " | ".join(port))
    for y in sorted(yr["QQQ"].index):
        L.append(f"　　{y} | " + " | ".join(f"{yr[k].get(y, np.nan) * 100:+.0f}" for k in port))
    pd.DataFrame({"report": L}).to_csv("multibagger_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
