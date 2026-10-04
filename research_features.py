"""
research_features.py — 由我哋自己嘅數據搵規律：「出現咩條件之後，股票較大機會升10%？」

⚠️ 預先登記：以下設計（候選條件、成功定義、分組、通過標準）喺第一次跑之前已經定好，
   跑完唔會為咗令結果好睇而加條件或者改標準。

做法：
- 股票池：strategy_v2 嘅股票池（已剔除ETF，約373隻有數據），2010-01 至 2026-09。
- 每個星期最後一個交易日抽一次樣本（避免同一段升勢重複計好多次）。
- 每個樣本：用嗰日收市時已知嘅數據計條件；入場價 = 下一個交易日開市價。
- 「成功」= 之後40個交易日內，先升到入場價 +10%，而唔係先跌到 −5%
  （同一日兩樣都發生當失敗；40日內兩樣都冇 = 冇結果，唔當成功）。
- 另一個指標：60個交易日後回報減SPY同期回報（超額回報）。
- 唔睇「淨係贏嘅交易」：每個條件都分組，比較每組嘅成功率，同全部樣本嘅平均比。

候選條件（固定呢15個，唔再加）：
  趨勢：股價離200日線%、200日線過去1個月斜率
  動量：3個月回報、6個月回報、12個月回報（唔計最近1個月）、RS評級（IBD風格1-99）、RS過去20日變化
  位置：離52週高位%、離52週低位%
  短線：5日回報、RSI(14)
  波幅：ATR佔股價%
  成交量：20日平均 ÷ 50日平均
  大市：SPY是否高過200日線、股票池入面企喺50日線上嘅比例
  另外：Mega Cap / High Beta 分組

防止撞彩（通過標準）：
- 股票按代號固定分做A、B兩半（crc32(代號) 單雙數，唔係人手揀）。
- 發現組 = A半 2010-2017；驗證組1 = B半 2010-2017；驗證組2 = 全部股票 2018-2026/09。
- 連續型條件用「發現組」嘅五分位切點分5組，同一套切點套用落其他組。
- 一個條件算「通過」：最高組減最低組嘅成功率，喺三組數據都係同一個方向（正或負），
  而且喺至少70%嘅個別年份方向一致。

RS評級特別分析（用戶要求）：
- RS細分組：1-20、20-40、40-60、60-70、70-80、80-90、90-95、95-99。
- 入場RS vs 離場RS（離場 = 升到+10%、跌到−5%、或者第40日，以先到者為準）。
  ⚠️ 離場RS係「事後先知道」嘅資料，只可以用嚟理解，唔可以當入場條件。
  可以預先用嘅係「RS過去20日變化」，已經放咗喺候選條件入面。
- 如果有 v2_results.csv（同一個workflow先跑 backtest_v2.py），會加埋v2每單交易嘅入場/離場RS分析。
"""
import os
import zlib

import numpy as np
import pandas as pd
import yfinance as yf

import strategy_v2 as S
from backtest_v2 import DOWNLOAD_START, END, download_all, flatten

SAMPLE_START = "2010-01-01"
DISCOVERY_END = "2017-12-31"
UP, DOWN, WINDOW, FWD = 0.10, 0.05, 40, 60
YEAR_AGREEMENT = 0.70
RS_EDGES = [0, 20, 40, 60, 70, 80, 90, 95, 100]
RS_LABELS = ["1-20", "20-40", "40-60", "60-70", "70-80", "80-90", "90-95", "95-99"]
RS_COARSE_EDGES = [0, 40, 70, 80, 90, 100]
RS_COARSE_LABELS = ["<40", "40-70", "70-80", "80-90", "90-99"]
OUT_REPORT = "research_results.csv"
OUT_SAMPLES = "research_samples.csv.gz"

FEATURES = [
    ("f_dist_sma200", "股價離200日線"),
    ("f_sma200_slope", "200日線1個月斜率"),
    ("f_ret_3m", "3個月回報"),
    ("f_ret_6m", "6個月回報"),
    ("f_ret_12m_ex1m", "12個月回報(唔計最近1個月)"),
    ("rs", "RS評級"),
    ("rs_change_20", "RS過去20日變化"),
    ("f_dist_52w_high", "離52週高位"),
    ("f_dist_52w_low", "離52週低位"),
    ("f_ret_5d", "5日回報"),
    ("f_rsi14", "RSI(14)"),
    ("f_atr_pct", "ATR佔股價"),
    ("f_vol_trend", "成交量20日÷50日"),
    ("breadth", "股票池企喺50日線上嘅比例"),
]
BINARY = [("spy_above_200", "SPY高過200日線"), ("is_high_beta", "High Beta（對比Mega Cap）")]


def half_of(t):
    return "A" if zlib.crc32(t.encode()) % 2 == 0 else "B"


def ticker_features(df):
    d = S.add_indicators(df)
    c = d["Close"]
    d["f_dist_sma200"] = c / d["SMA200"] - 1
    d["f_sma200_slope"] = d["SMA200"] / d["SMA200_prev"] - 1
    d["f_ret_3m"] = c / c.shift(63) - 1
    d["f_ret_6m"] = c / c.shift(126) - 1
    d["f_ret_12m_ex1m"] = c.shift(21) / c.shift(252) - 1
    d["f_dist_52w_high"] = c / d["High52w"] - 1
    d["f_dist_52w_low"] = c / d["Low52w"] - 1
    d["f_ret_5d"] = c / c.shift(5) - 1
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["f_rsi14"] = 100 - 100 / (1 + gain / loss)
    d["f_atr_pct"] = d["ATR"] / c
    d["f_vol_trend"] = d["Volume"].rolling(20).mean() / d["VolAvg50"]
    d["above50"] = (c > d["SMA50"]).where(d["SMA50"].notna())
    return d


def build_samples():
    universe = S.v2_universe()
    group_of = dict(universe)
    print("下載SPY ...")
    spy = flatten(yf.download("SPY", start=DOWNLOAD_START, end=END, auto_adjust=True, progress=False)).dropna(how="all")
    idx = spy.index
    spy_o, spy_c = spy["Open"].astype(float).values, spy["Close"].astype(float).values
    spy_above = (spy["Close"] > spy["Close"].rolling(200).mean()).values

    raw = download_all([t for t, _ in universe])
    cols = list(raw.keys())
    print(f"有足夠數據嘅股票：{len(cols)}")
    feats = {t: ticker_features(raw[t]).reindex(idx) for t in cols}

    def panel(f):
        return pd.DataFrame({t: feats[t][f] for t in cols}, index=idx)

    O, H, L, C = (panel(f).values for f in ("Open", "High", "Low", "Close"))
    rs_df = S.rs_rating(panel("RSRaw"))
    RS = rs_df.values
    RSCHG = (rs_df - rs_df.shift(20)).values
    breadth = panel("above50").mean(axis=1).values
    F = {f: panel(f).values for f, _ in FEATURES if f.startswith("f_")}

    weekly_last = pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period("W-FRI")).last().values
    n, m = C.shape
    start_i = idx.searchsorted(pd.Timestamp(SAMPLE_START))
    sample_is = [i for i in weekly_last if i >= start_i and i + FWD < n]
    halves = np.array([half_of(t) for t in cols])
    is_hb = np.array([group_of[t] == "High Beta" for t in cols])
    tick = np.array(cols)
    cols_idx = np.arange(m)

    chunks = []
    for i in sample_is:
        e = O[i + 1]
        valid = ~np.isnan(e) & ~np.isnan(RS[i]) & ~np.isnan(C[i]) & (e > 0)
        if not valid.any():
            continue
        Hs, Ls = H[i + 1:i + 1 + WINDOW], L[i + 1:i + 1 + WINDOW]
        with np.errstate(invalid="ignore"):
            up, dn = Hs >= e * (1 + UP), Ls <= e * (1 - DOWN)
        up_any, dn_any = up.any(0), dn.any(0)
        big = 10 ** 6
        up_i = np.where(up_any, up.argmax(0), big)
        dn_i = np.where(dn_any, dn.argmax(0), big)
        outcome = np.where(up_any & (up_i < dn_i), 1, np.where(dn_any & (dn_i <= up_i), -1, 0))
        out_day = np.minimum(np.minimum(up_i, dn_i), WINDOW - 1)
        rs_exit = RS[i + 1 + out_day, cols_idx]
        excess = (C[i + FWD] / e - 1) - (spy_c[i + FWD] / spy_o[i + 1] - 1)
        v = np.where(valid)[0]
        d = {
            "date": np.repeat(idx[i].date(), len(v)), "ticker": tick[v], "half": halves[v],
            "is_high_beta": is_hb[v], "rs": RS[i, v], "rs_change_20": RSCHG[i, v], "rs_exit": rs_exit[v],
            "outcome": outcome[v], "days_to_outcome": out_day[v] + 1, "excess_60d": excess[v],
            "spy_above_200": np.repeat(bool(spy_above[i]), len(v)), "breadth": np.repeat(breadth[i], len(v)),
        }
        for f, arr in F.items():
            d[f] = arr[i, v]
        chunks.append(pd.DataFrame(d))
    s = pd.concat(chunks, ignore_index=True)
    s["date"] = pd.to_datetime(s["date"])
    s["year"] = s["date"].dt.year
    s["success"] = (s["outcome"] == 1).astype(float)
    return s


def sets_of(s):
    disc_end = pd.Timestamp(DISCOVERY_END)
    return [
        ("發現組(A半 2010-17)", s[(s["half"] == "A") & (s["date"] <= disc_end)]),
        ("驗證組1(B半 2010-17)", s[(s["half"] == "B") & (s["date"] <= disc_end)]),
        ("驗證組2(全部 2018-26)", s[s["date"] > disc_end]),
    ]


def bucket_table(sets, col, cut_fn, labels):
    lines = []
    header = "　組別 | " + " | ".join(name for name, _ in sets)
    lines.append(header + "   （每格：樣本數 / 成功率 / 60日超額回報）")
    for lab in labels:
        cells = []
        for _, d in sets:
            sub = d[cut_fn(d) == lab]
            if len(sub) == 0:
                cells.append("—")
            else:
                cells.append(f"{len(sub)} / {sub['success'].mean() * 100:.1f}% / {sub['excess_60d'].mean() * 100:+.2f}%")
        lines.append(f"　{lab} | " + " | ".join(cells))
    return lines


def spread(d, cut, top, bottom):
    a, b = d[cut(d) == top]["success"], d[cut(d) == bottom]["success"]
    if len(a) < 30 or len(b) < 30:
        return np.nan
    return (a.mean() - b.mean()) * 100


def assess(s, sets, cut, top, bottom):
    sp = [spread(d, cut, top, bottom) for _, d in sets]
    disc_sign = np.sign(sp[0]) if not np.isnan(sp[0]) else 0
    years = sorted(s["year"].unique())
    agree, counted = 0, 0
    for y in years:
        v = spread(s[s["year"] == y], cut, top, bottom)
        if np.isnan(v) or disc_sign == 0:
            continue
        counted += 1
        agree += int(np.sign(v) == disc_sign)
    same_sign = disc_sign != 0 and all(not np.isnan(x) and np.sign(x) == disc_sign for x in sp)
    frac = agree / counted if counted else 0
    passed = same_sign and frac >= YEAR_AGREEMENT
    return sp, frac, counted, passed


def main():
    s = build_samples()
    sets = sets_of(s)
    lines = [
        "=== 條件研究：出現咩條件之後，股票較大機會「40日內先升10%、而唔係先跌5%」？（設計預先登記）===",
        f"樣本：{len(s):,}個（{s['ticker'].nunique()}隻股票，每週一次，{s['date'].min().date()} 至 {s['date'].max().date()}）",
        "⚠️ 倖存者偏差：用今日嘅股票名單，所有成功率都會偏高；要睇嘅係「組與組之間嘅分別」，唔係絕對數字。",
        "",
        "=== 基準（唔加任何條件）===",
    ]
    for name, d in sets:
        lines.append(f"　{name}：樣本{len(d):,}，成功率{d['success'].mean() * 100:.1f}%，"
                     f"先跌5%比例{(d['outcome'] == -1).mean() * 100:.1f}%，60日超額回報平均{d['excess_60d'].mean() * 100:+.2f}%")
    lines.append("")

    summary = []
    disc = sets[0][1]
    q_labels = ["Q1(最低)", "Q2", "Q3", "Q4", "Q5(最高)"]
    for f, cn in FEATURES:
        x = disc[f].dropna()
        if x.nunique() < 5:
            continue
        edges = np.unique(np.quantile(x, [0, 0.2, 0.4, 0.6, 0.8, 1]))
        if len(edges) < 6:
            continue
        edges[0], edges[-1] = -np.inf, np.inf

        def cut(d, f=f, edges=edges):
            return pd.cut(d[f], edges, labels=q_labels)
        sp, frac, counted, passed = assess(s, sets, cut, "Q5(最高)", "Q1(最低)")
        rng = "，".join(f"{q_labels[k]}≤{edges[k + 1]:.3g}" for k in range(4))
        lines.append(f"=== {cn}（{f}）：五分位切點（由發現組計）{rng} ===")
        lines += bucket_table(sets, f, cut, q_labels)
        lines.append(f"　最高組−最低組 成功率差：" + "，".join(f"{n} {v:+.1f}" for (n, _), v in zip(sets, sp))
                     + f"；{counted}個年份入面{frac * 100:.0f}%同方向 → {'✅ 通過' if passed else '❌ 唔通過'}")
        lines.append("")
        summary.append((cn, sp, frac, passed))

    for f, cn in BINARY:
        def cut(d, f=f):
            return d[f].map({True: "是", False: "否"})
        sp, frac, counted, passed = assess(s, sets, cut, "是", "否")
        lines.append(f"=== {cn}（{f}）===")
        lines += bucket_table(sets, f, cut, ["是", "否"])
        lines.append(f"　「是」−「否」成功率差：" + "，".join(f"{n} {v:+.1f}" for (n, _), v in zip(sets, sp))
                     + f"；{counted}個年份入面{frac * 100:.0f}%同方向 → {'✅ 通過' if passed else '❌ 唔通過'}")
        lines.append("")
        summary.append((cn, sp, frac, passed))

    # RS 細分
    def rs_cut(d):
        return pd.cut(d["rs"], RS_EDGES, labels=RS_LABELS, include_lowest=True)
    lines.append("=== RS評級細分（入場時）===")
    lines += bucket_table(sets, "rs", rs_cut, RS_LABELS)
    lines.append("")

    # 入場RS -> 離場RS（描述性，離場RS係事後資料）
    s["rs_in_c"] = pd.cut(s["rs"], RS_COARSE_EDGES, labels=RS_COARSE_LABELS, include_lowest=True)
    s["rs_out_c"] = pd.cut(s["rs_exit"], RS_COARSE_EDGES, labels=RS_COARSE_LABELS, include_lowest=True)
    lines.append("=== 入場RS → 離場RS（全期；⚠️離場RS係事後先知，只用嚟理解，唔可以做入場條件）===")
    lines.append("　每格：樣本數 / 成功率。橫行=入場RS，直行=離場RS")
    lines.append("　入場\\離場 | " + " | ".join(RS_COARSE_LABELS))
    for a in RS_COARSE_LABELS:
        cells = []
        for b in RS_COARSE_LABELS:
            sub = s[(s["rs_in_c"] == a) & (s["rs_out_c"] == b)]
            cells.append(f"{len(sub)} / {sub['success'].mean() * 100:.0f}%" if len(sub) else "—")
        lines.append(f"　{a} | " + " | ".join(cells))
    for name, mask in (("成功", s["outcome"] == 1), ("先跌5%", s["outcome"] == -1), ("冇結果", s["outcome"] == 0)):
        sub = s[mask]
        lines.append(f"　{name}：入場RS中位數 {sub['rs'].median():.0f} → 離場RS中位數 {sub['rs_exit'].median():.0f}"
                     f"（樣本{len(sub):,}，平均{sub['days_to_outcome'].mean():.0f}日有結果）")
    lines.append("")

    # v2 交易嘅入場/離場RS
    if os.path.exists("v2_results.csv"):
        v2 = pd.read_csv("v2_results.csv")
        tr = v2[v2["group"].isin(["Mega Cap", "High Beta"])].copy()
        if "rs_at_exit" in tr.columns and len(tr):
            for c in ("ret_pct", "pnl", "rs_at_signal", "rs_at_exit"):
                tr[c] = pd.to_numeric(tr[c], errors="coerce")
            tr["in_c"] = pd.cut(tr["rs_at_signal"], RS_COARSE_EDGES, labels=RS_COARSE_LABELS, include_lowest=True)
            tr["out_c"] = pd.cut(tr["rs_at_exit"], RS_COARSE_EDGES, labels=RS_COARSE_LABELS, include_lowest=True)
            lines.append(f"=== v2交易（{len(tr)}單）：入場RS → 離場RS（每格：單數 / 勝率 / 平均回報）===")
            lines.append("　入場\\離場 | " + " | ".join(RS_COARSE_LABELS))
            for a in RS_COARSE_LABELS:
                cells = []
                for b in RS_COARSE_LABELS:
                    sub = tr[(tr["in_c"] == a) & (tr["out_c"] == b)]
                    cells.append(f"{len(sub)} / {(sub['pnl'] > 0).mean() * 100:.0f}% / {sub['ret_pct'].mean():+.1f}%"
                                 if len(sub) else "—")
                lines.append(f"　{a} | " + " | ".join(cells))
            w, l_ = tr[tr["pnl"] > 0], tr[tr["pnl"] <= 0]
            lines.append(f"　贏錢單：入場RS中位數{w['rs_at_signal'].median():.0f} → 離場{w['rs_at_exit'].median():.0f}；"
                         f"蝕錢單：入場{l_['rs_at_signal'].median():.0f} → 離場{l_['rs_at_exit'].median():.0f}")
            lines.append("")

    lines.append("=== 總結：邊啲條件通過咗三組數據 + 70%年份嘅測試？===")
    for cn, sp, frac, passed in summary:
        lines.append(f"　{'✅' if passed else '❌'} {cn}：成功率差 " + " / ".join(f"{v:+.1f}" for v in sp)
                     + f"，年份一致{frac * 100:.0f}%")

    pd.DataFrame({"report": lines}).to_csv(OUT_REPORT, index=False, encoding="utf-8-sig")
    keep = ["date", "ticker", "half", "is_high_beta", "rs", "rs_change_20", "rs_exit", "outcome", "days_to_outcome",
            "excess_60d", "spy_above_200", "breadth"] + [f for f, _ in FEATURES if f.startswith("f_")]
    out = s[keep].copy()
    num = out.select_dtypes("float").columns
    out[num] = out[num].round(4)
    out["date"] = out["date"].dt.date
    out.to_csv(OUT_SAMPLES, index=False, compression="gzip")
    print("\n".join(lines))
    print(f"\n✅ 已存到 {OUT_REPORT} 同 {OUT_SAMPLES}")


if __name__ == "__main__":
    main()
