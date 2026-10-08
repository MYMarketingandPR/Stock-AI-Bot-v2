"""
research_v25.py — 第二十五輪：線D 唔買「第二次倍升」（之前一年都已經倍升）嘅股票（Esther 2026-10-08：「要做」）

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
⚠️ 要老實講：呢個諗法係由第二十四輪「描述」部分（同一批 2000-2026 歷史數據）得出嚟，所以今輪回測唔係獨立證據，
   睇落好都可能係「用答案出題」。真正驗證要靠之後嘅模擬交易。
共同（同線D一樣）：當時真實標普500；已計股息；收市決定 → 下一個交易日收市買賣；每次買賣0.1%；
  最多5隻、每隻 = 買入日總資產 ÷ 5；冇投資嘅錢揸現金（^IRX）；有空位揀12個月升幅最大；由最高收市價跌20%就賣。
  T0 = 線D；T2 = 線D2（SPY每日訊號持有先開新倉）。
新版本：
  N1 = 線D + 唔買「之前一年（252至504個交易日前）都升 ≥ +100%」嘅股票（冇504日數據嘅照買）。
  N2 = 線D2 + 同一個過濾。
判斷：
  ① 「有改善」（N1 同 T0 比、N2 同 T2 比，驗證期 2015-01 至 2026-09）：最大跌幅改善 ≥ 3個百分點而且年化唔少過1個百分點；
     或者年化多 ≥ 2個百分點而且最大跌幅差唔超過2個百分點。
  ② 跨年代合格（同第二十四輪）：2000-2009、2010-2014、2015-2019、2020-2026 每段年化 ≥ SPY 而且最大跌幅好過 -35%。
  另外列：被過濾走嘅買入次數、最差5年年化；同埋線D而家5隻持倉之前一年升幅（睇吓有冇「第二次倍升」）。
輸出：v25_results.csv、v25_progress.txt
==== 結果（2026-10-08 第一次跑，記錄低，唔再改）====
- N1 線D + 唔買第二次倍升：驗證期 +19.74% / -35.8%（T0 +21.38% / -33.8%）→ 冇改善；跨年代 ✘✔✔✘。
  但 2000-2009 +6.0% / -42%（T0 +4.5% / -46%）、最差5年 -1.1%（T0 -6.4%）。
- N2 線D2 + 唔買第二次倍升：+24.28% / -29.5%（T2 +25.64% / -29.5%）→ 冇改善；跨年代 ✘✔✔✔。
  2000-2009 +7.8% / -44%（T2 +5.2% / -51%）、最差5年 -4.0%（T2 -7.9%）。
- 只少咗約8次買入：跌20%賣出規則本身已經限制咗呢類股票嘅損失，所以描述統計嘅大分別，放落組合就細好多。
- 而家持倉：LITE 之前一年 +147%（第二次倍升）；MU +92%、DELL +31%、MRNA -56%；SNDK 上市唔夠兩年。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from perf import perf
from research_v13 import get_cash
from research_v24 import sim
from research_v8 import decision_days

M.PROGRESS = "v25_progress.txt"
log = M.log
END = "2026-09-30"


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
    prev = (px.shift(252) / px.shift(504) - 1).values
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
    second = np.isfinite(prev) & (prev >= 1.0)
    filt_ok = base_ok & ~second
    import main as bot
    df = bot.replay(bot.add_signals(bot.load_data()))
    sig_ok = (df["state"] != "out").reindex(days).fillna(False).values
    log("數據準備好")

    V = {"T0 線D": sim(P, rets_v, cr, base_ok, r12n),
         "N1 線D + 唔買第二次倍升": sim(P, rets_v, cr, filt_ok, r12n),
         "T2 線D2": sim(P, rets_v, cr, base_ok, r12n, allow=sig_ok),
         "N2 線D2 + 唔買第二次倍升": sim(P, rets_v, cr, filt_ok, r12n, allow=sig_ok)}
    log("全部版本完成")

    windows = [("驗證期", "2015-01-01", END), ("2000-2009", "2000-01-03", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
               ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", END), ("全期", "2000-01-03", END),
               ("科網爆煲", "2000-03-01", "2002-12-31"), ("金融海嘯", "2007-07-01", "2009-06-30"), ("2022熊市", "2022-01-01", "2022-12-31")]
    allser = {"QQQ長揸": qqq_r, "SPY長揸": spy_r}
    for nm, v in V.items():
        allser[nm] = pd.Series(v[0], index=days)
    tab = {}
    L = ["=== 第二十五輪：線D 唔買「第二次倍升」（規則預先登記，見research_v25.py開頭）===",
         f"股票池：當時真實標普500（{source}）；已計股息；每次買賣0.1%；最多5隻",
         "⚠️ 呢個諗法係由第二十四輪同一批歷史數據得出，回測唔係獨立證據，要靠模擬交易驗證。", ""]
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
    n_days = int((base_ok & second).any(axis=1).sum())
    L.append(f"　（有 {n_days} 個交易日，至少一隻符合線D條件嘅股票因為「第二次倍升」被過濾走）")
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
    L.append("=== 預先登記判斷 ===")
    for nm, ref in (("N1 線D + 唔買第二次倍升", "T0 線D"), ("N2 線D2 + 唔買第二次倍升", "T2 線D2"), ("T0 線D", None), ("T2 線D2", None)):
        x = tab[("驗證期", nm)]
        marks = "".join("✔" if (tab[(e, nm)]["cagr"] >= tab[(e, "SPY長揸")]["cagr"] and tab[(e, nm)]["mdd"] > -35) else "✘" for e in eras)
        imp = ""
        if ref:
            y = tab[("驗證期", ref)]
            better = (x["mdd"] >= y["mdd"] + 3 and x["cagr"] >= y["cagr"] - 1) or (x["cagr"] >= y["cagr"] + 2 and x["mdd"] >= y["mdd"] - 2)
            imp = f"同 {ref}（{y['cagr']:+.2f}% / {y['mdd']:.1f}%）比 → {'有改善' if better else '冇改善'} ｜ "
        L.append(f"　{nm}：驗證期 {x['cagr']:+.2f}% / {x['mdd']:.1f}% / Sharpe {x['sharpe']:.2f} ｜ {imp}跨年代 {marks} "
                 f"{'跨年代合格' if marks == '✔✔✔✔' else '唔合格'} ｜ 各段年化 " + " / ".join(f"{tab[(e, nm)]['cagr']:+.1f}%" for e in eras)
                 + " ｜ 各段最大跌幅 " + " / ".join(f"{tab[(e, nm)]['mdd']:.0f}%" for e in eras) + f" ｜ 最差5年年化 {worst5(allser[nm]):+.1f}%")
    L.append("")

    # 線D 而家持倉：之前一年升幅
    hold = ["SNDK", "MRNA", "LITE", "MU", "DELL"]
    try:
        hc = M.download_close(hold, "2024-01-01", "2026-10-09")
        L.append("=== 線D 而家持倉：過去12個月 vs 之前一年（252至504個交易日前）升幅（用最新收市價）===")
        for t in hold:
            s = hc.get(t)
            if s is None or len(s) < 253:
                L.append(f"　{t}：數據唔夠")
                continue
            r_now = s.iloc[-1] / s.iloc[-253] - 1
            r_prev = s.iloc[-253] / s.iloc[-505] - 1 if len(s) >= 505 else np.nan
            tag = "（上市唔夠兩年，冇得計）" if not np.isfinite(r_prev) else ("⚠️ 第二次倍升" if r_prev >= 1.0 else "")
            L.append(f"　{t}（{s.index[-1].date()}）：過去12個月 {r_now * 100:+.0f}% ｜ 之前一年 "
                     f"{'—' if not np.isfinite(r_prev) else f'{r_prev * 100:+.0f}%'} {tag}")
    except Exception as e:
        L.append(f"持倉檢查出錯：{e}")
    pd.DataFrame({"report": L}).to_csv("v25_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
