"""
research_gmma.py — 驗證用戶發現嘅規律：「GMMA（LWGMMA）向上、而且12條線高度重合之後，會出現一段升幅」

⚠️ 預先登記：以下定義、分組、通過標準喺第一次跑之前已經定好，跑完唔會為咗結果好睇而改。

GMMA（Daryl Guppy 顧比複合移動平均線）標準設定：
- 短期組：EMA 3、5、8、10、12、15
- 長期組：EMA 30、35、40、45、50、60
用戶App嘅LWGMMA有EMA1-EMA12共12條，截圖見到第6條同第7條之間明顯分開，同標準GMMA兩組結構一致。
如果App用嘅期數唔同，結果會有少少分別。

定義（全部用當日收市時已知嘅數據）：
- 重合度(%)   = 12條EMA最高 − 最低，除以收市價
- 重合度(ATR) = 12條EMA最高 − 最低，除以ATR(14)（考慮咗唔同股票本身波幅唔同）
- 「向上」     = 長期組6條EMA嘅平均值，高過10個交易日前
- 「多頭排列」 = 短期組最低嗰條 > 長期組最高嗰條（只作參考）

成功定義、抽樣方法、A/B股票分組：同 research_features.py 完全一樣
（每週抽樣；入場=下一日開市價；40日內先升10%而唔係先跌5% = 成功；另計60日超額回報；
 A半2010-17發現、B半2010-17同全部2018-26驗證）。

通過標準（比 research_features.py 更嚴）：三組同方向，而且「年份符號檢定」p < 0.05
（即係假設冇規律、每年同方向機會係一半，觀察到咁多年同方向嘅機會要細過5%；17年大約要13年同方向）。
原因：用純隨機數據試跑時，「最重合 vs 最分散」三組竟然都同方向（+1.4/+3.4/+2.9），只係年份65%先唔通過——
每週抽樣但睇40日，樣本大量重疊，3個百分點嘅差距可以純粹係噪音。原本70%標準嘅結果會一齊列出做對照。

預先定好嘅「最理想重合度」分組（喺「向上」嘅樣本入面比較）：
- 重合度(%)：<2%、2-3%、3-4%、4-6%、6-10%、≥10%
- 重合度(ATR)：<1、1-2、2-3、3-5、≥5 倍ATR

SNDK 個案：SNDK唔喺我哋股票池（2025年2月先由WDC分拆上市），而且係事後揀出嚟嘅大贏家，
所以只會逐週列出佢嘅數據做參考，唔計入統計。
"""
from math import comb

import numpy as np
import pandas as pd
import yfinance as yf

import research_features as RF
import strategy_v2 as S
from backtest_v2 import END, flatten

SHORT = [3, 5, 8, 10, 12, 15]
LONG = [30, 35, 40, 45, 50, 60]
SLOPE_DAYS = 10
PCT_EDGES = [0, 2, 3, 4, 6, 10, np.inf]
PCT_LABELS = ["<2%", "2-3%", "3-4%", "4-6%", "6-10%", "≥10%"]
ATR_EDGES = [0, 1, 2, 3, 5, np.inf]
ATR_LABELS = ["<1倍", "1-2倍", "2-3倍", "3-5倍", "≥5倍"]
GMMA_COLS = ["g_spread_pct", "g_spread_atr", "g_up", "g_bull_order", "g_spread_short_pct", "g_spread_long_pct"]
OUT_REPORT = "gmma_results.csv"
OUT_SAMPLES = "gmma_samples.csv.gz"
CASE_TICKER = "SNDK"


def add_gmma(d):
    c = d["Close"]
    s = pd.concat([c.ewm(span=p, adjust=False).mean() for p in SHORT], axis=1)
    lg = pd.concat([c.ewm(span=p, adjust=False).mean() for p in LONG], axis=1)
    allm = pd.concat([s, lg], axis=1)
    warm = c.notna().cumsum() >= max(LONG) * 2          # EMA要暖身，頭120日唔用
    spread = (allm.max(axis=1) - allm.min(axis=1)).where(warm)
    d["g_spread_pct"] = spread / c * 100
    d["g_spread_atr"] = spread / d["ATR"]
    long_mean = lg.mean(axis=1)
    d["g_up"] = (long_mean > long_mean.shift(SLOPE_DAYS)).astype(float).where(warm)
    d["g_bull_order"] = (s.min(axis=1) > lg.max(axis=1)).astype(float).where(warm)
    d["g_spread_short_pct"] = ((s.max(axis=1) - s.min(axis=1)) / c * 100).where(warm)
    d["g_spread_long_pct"] = ((lg.max(axis=1) - lg.min(axis=1)) / c * 100).where(warm)
    return d


def sign_test_p(agree, counted):
    """單邊二項檢定：假設每年同方向機會係50%，觀察到 >= agree 年同方向嘅機率。"""
    if counted == 0:
        return 1.0
    return sum(comb(counted, k) for k in range(agree, counted + 1)) / 2 ** counted


def verdict(sp, frac, counted):
    """回傳(嚴格標準結果文字, 舊70%標準是否通過)。"""
    signs = [np.sign(v) for v in sp if not np.isnan(v)]
    same = len(signs) == len(sp) and len(set(signs)) == 1 and signs[0] != 0
    agree = int(round(frac * counted))
    p = sign_test_p(agree, counted)
    strict = same and p < 0.05
    old = same and frac >= RF.YEAR_AGREEMENT
    return (f"{agree}/{counted}年同方向，符號檢定p={p:.3f} → 嚴格標準{'✅ 通過' if strict else '❌ 唔通過'}"
            f"（舊70%標準：{'通過' if old else '唔通過'}）")


def cells(d):
    if len(d) == 0:
        return "—"
    return f"{len(d)} / {d['success'].mean() * 100:.1f}% / {d['excess_60d'].mean() * 100:+.2f}%"


def sndk_case(spy):
    out = [f"=== {CASE_TICKER} 個案（只作參考，唔計入統計；佢係事後揀出嚟嘅大贏家）==="]
    try:
        df = flatten(yf.download(CASE_TICKER, start="2024-01-01", end=END, auto_adjust=True, progress=False)).dropna(how="all")
    except Exception as e:
        return out + [f"　下載失敗：{e}"]
    if len(df) < 200:
        return out + [f"　數據唔夠（{len(df)}日）"]
    d = add_gmma(S.add_indicators(df[["Open", "High", "Low", "Close", "Volume"]].astype(float)))
    idx = d.index
    spy = spy.reindex(idx)
    weekly = pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period("W-FRI")).last().values
    out.append("　日期 | 收市 | 重合度% | 重合度ATR | 向上 | 結果(40日) | 60日超額")
    for i in weekly:
        if i + 1 >= len(idx) or np.isnan(d["g_spread_pct"].iloc[i]):
            continue
        e = d["Open"].iloc[i + 1]
        fut = d.iloc[i + 1:i + 1 + RF.WINDOW]
        up, dn = fut["High"] >= e * (1 + RF.UP), fut["Low"] <= e * (1 - RF.DOWN)
        ui = up.values.argmax() if up.any() else 10 ** 6
        di = dn.values.argmax() if dn.any() else 10 ** 6
        if len(fut) < RF.WINDOW and ui == di == 10 ** 6:
            res = "未夠40日"
        else:
            res = "成功(先升10%)" if (up.any() and ui < di) else ("先跌5%" if dn.any() else "冇結果")
        if i + RF.FWD < len(idx):
            ex = (d["Close"].iloc[i + RF.FWD] / e - 1) - (spy["Close"].iloc[i + RF.FWD] / spy["Open"].iloc[i + 1] - 1)
            ex_s = f"{ex * 100:+.1f}%"
        else:
            ex_s = "未夠60日"
        out.append(f"　{idx[i].date()} | {d['Close'].iloc[i]:.2f} | {d['g_spread_pct'].iloc[i]:.1f}% | "
                   f"{d['g_spread_atr'].iloc[i]:.1f} | {'是' if d['g_up'].iloc[i] == 1 else '否'} | {res} | {ex_s}")
    return out


def main():
    s = RF.build_samples(extra=add_gmma, extra_cols=GMMA_COLS)
    s = s.dropna(subset=["g_spread_pct", "g_up"])
    sets = RF.sets_of(s)
    lines = [
        "=== GMMA 驗證：「向上 + 12條線高度重合」之後，係咪較大機會40日內先升10%？（設計預先登記）===",
        f"樣本：{len(s):,}個（{s['ticker'].nunique()}隻股票，每週一次，{s['date'].min().date()} 至 {s['date'].max().date()}）",
        "每格：樣本數 / 成功率 / 60日超額回報(對SPY)。⚠️ 倖存者偏差令絕對數字偏高，要睇組與組之間嘅分別。",
        "",
        "=== 基準（全部樣本）===",
        "　組別 | " + " | ".join(n for n, _ in sets),
        "　全部 | " + " | ".join(cells(d) for _, d in sets),
        "",
        "=== GMMA 向上 vs 唔向上 ===",
        "　組別 | " + " | ".join(n for n, _ in sets),
    ]
    for lab, val in (("向上", 1.0), ("唔向上", 0.0)):
        lines.append(f"　{lab} | " + " | ".join(cells(d[d["g_up"] == val]) for _, d in sets))

    def up_cut(d):
        return d["g_up"].map({1.0: "是", 0.0: "否"})
    sp, frac, counted, passed = RF.assess(s, sets, up_cut, "是", "否")
    lines.append("　「向上」−「唔向上」成功率差：" + "，".join(f"{n} {v:+.1f}" for (n, _), v in zip(sets, sp))
                 + "；" + verdict(sp, frac, counted))
    lines.append("")

    up = s[s["g_up"] == 1]
    up_sets = RF.sets_of(up)
    for title, col, edges, labels in (("重合度(%)", "g_spread_pct", PCT_EDGES, PCT_LABELS),
                                      ("重合度(ATR倍數)", "g_spread_atr", ATR_EDGES, ATR_LABELS)):
        lines.append(f"=== 只睇「向上」：{title}分組（邊個重合度最理想？）===")
        lines.append("　重合度 | " + " | ".join(n for n, _ in up_sets))
        for lab in labels:
            lines.append(f"　{lab} | " + " | ".join(
                cells(d[pd.cut(d[col], edges, labels=labels, right=False) == lab]) for _, d in up_sets))
        x = up_sets[0][1][col].dropna()
        q = np.unique(np.quantile(x, [0, .2, .4, .6, .8, 1]))
        if len(q) == 6:
            q[0], q[-1] = -np.inf, np.inf
            qlab = ["Q1(最重合)", "Q2", "Q3", "Q4", "Q5(最分散)"]

            def qcut(d, col=col, q=q, qlab=qlab):
                return pd.cut(d[col], q, labels=qlab)
            sp, frac, counted, passed = RF.assess(up, up_sets, qcut, "Q1(最重合)", "Q5(最分散)")
            lines.append(f"　五分位測試（切點由發現組計：" + "，".join(f"{qlab[k]}≤{q[k + 1]:.2f}" for k in range(4)) + "）")
            for lab in qlab:
                lines.append(f"　{lab} | " + " | ".join(cells(d[qcut(d) == lab]) for _, d in up_sets))
            lines.append("　「最重合」−「最分散」成功率差：" + "，".join(f"{n} {v:+.1f}" for (n, _), v in zip(up_sets, sp))
                         + "；" + verdict(sp, frac, counted))
        lines.append("")

    lines.append("=== 參考：向上 + 多頭排列（短期組全部高過長期組）===")
    lines.append("　組別 | " + " | ".join(n for n, _ in up_sets))
    for lab, val in (("多頭排列", 1.0), ("未排列", 0.0)):
        lines.append(f"　{lab} | " + " | ".join(cells(d[d["g_bull_order"] == val]) for _, d in up_sets))
    lines.append("")

    lines.append("=== 分組睇：Mega Cap / High Beta（向上 + 重合度<3%）===")
    tight = up[up["g_spread_pct"] < 3]
    for g, val in (("Mega Cap", False), ("High Beta", True)):
        lines.append(f"　{g} | " + " | ".join(cells(d[d["is_high_beta"] == val]) for _, d in RF.sets_of(tight)))
    lines.append("")

    spy = flatten(yf.download("SPY", start="2024-01-01", end=END, auto_adjust=True, progress=False)).dropna(how="all")
    lines += sndk_case(spy)

    pd.DataFrame({"report": lines}).to_csv(OUT_REPORT, index=False, encoding="utf-8-sig")
    keep = ["date", "ticker", "half", "is_high_beta", "outcome", "excess_60d", "rs", "spy_above_200"] + GMMA_COLS
    out = s[keep].copy()
    num = out.select_dtypes("float").columns
    out[num] = out[num].round(4)
    out["date"] = out["date"].dt.date
    out.to_csv(OUT_SAMPLES, index=False, compression="gzip")
    print("\n".join(lines))
    print(f"\n✅ 已存到 {OUT_REPORT} 同 {OUT_SAMPLES}")


if __name__ == "__main__":
    main()
