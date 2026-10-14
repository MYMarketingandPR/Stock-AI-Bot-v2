"""
research_v31.py — 第三十一輪：邊種「排名方法」捉中大升股嘅命中率最高？（Esther 2026-10-14：「我地試，但唔改原本有既野」）

純研究，唔改任何模擬交易（線A/B/C/D/D2/E、輪動、SPY訊號全部照舊），唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
股票池：U1 當時真實標普500（fja05680），已計股息（Yahoo調整價）。
決定日：每月最後交易日，2000-01 至 2025-09（要有12個月結果）；「2年3倍」只計到 2024-09。
入場價：決定日下一個交易日收市（同模擬交易一樣）。
每個方法每個決定日揀分數最高5隻（同線D/E一樣揸5隻），要當日係成分股、有價、入場日有價。

方法（只有呢10個，跑完唔加唔減）：
  M0 全部成分股（基準：隨便揀嘅平均）
  M1 12個月升幅最高
  M2 今年升幅最高（YTD）
  M3 RS Rating 最高（第二十三輪同一計法）
  M4 6個月升幅最高
  M5 最接近52週最高收市價（同分按12個月升幅）
  M6 「去年跌、今年先升」：之前一年（252至504日前）升幅 < 0，再按12個月升幅揀最高
  M7 線D條件（12個月 ≥ +100%、喺MA50同MA200上面），按12個月升幅
  M8 線E條件（今年 ≥ +30%、≥ 52週最高收市 × 0.7、喺MA200上面），按今年升幅
  M9 離52週高位最遠（跌得最多，「抄底」對照；第十六輪見到佢3倍股機率高）

量度（每個方法 = 所有揀中嘅股票 × 月份）：
  ① 一年內倍升率：入場後252個交易日內，有冇任何一日收市 ≥ 入場價 × 2
  ② 兩年內3倍率：入場後504個交易日內最高收市 ≥ 入場價 × 3
  ③ 12個月後回報（中位數、平均）：第252個交易日收市÷入場價（中途除牌冇價 = 用最後有價嗰日）
  ④ 大跌率：12個月後回報 ≤ -30%
  ⑤ 倍數 = 方法嘅①÷同一批決定日 M0 嘅①
年代：2000-09、2010-14、2015-19、2020-25（決定日計）。

合格（「值得再做一輪交易測試」），全部要符合：
  (a) 全期倍數 ≥ 2
  (b) 四個年代每個倍數 ≥ 1.5
  (c) 12個月後回報中位數 ≥ M0 中位數
  (d) 大跌率 ≤ M0 大跌率 × 2
  就算合格，都只係「值得再試」，唔會直接加入模擬交易；要另外預先登記一輪完整交易測試先算。
  10個方法一齊比，試得多總有好彩 → 只信四個年代都企得穩嘅結果。

已知限制：Yahoo 冇數據嘅已除牌股（多數係爆煲股）唔會入選，令所有方法（包括M0）睇落好咗少少。

輸出：v31_results.csv（全期 + 各年代）、v31_progress.txt
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v8 import decision_days
from research_v23 import rs_rating

M.PROGRESS = "v31_progress.txt"
log = M.log
END = "2026-09-30"
LAST_DEC_12 = pd.Timestamp("2025-09-30")
LAST_DEC_24 = pd.Timestamp("2024-09-30")
ERAS = [("2000-09", 2000, 2009), ("2010-14", 2010, 2014), ("2015-19", 2015, 2019), ("2020-25", 2020, 2025)]
NAMES = {"M0": "全部成分股（基準）", "M1": "12個月升幅最高", "M2": "今年升幅最高", "M3": "RS Rating 最高",
         "M4": "6個月升幅最高", "M5": "最接近52週高位", "M6": "去年跌、今年先升", "M7": "線D條件",
         "M8": "線E條件", "M9": "離52週高位最遠（抄底）"}


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    everyone = sorted(set().union(*[members_fn(d) for d in pd.date_range("1999-01-31", END, freq="ME")]))
    closes = M.download_close(everyone + ["SPY"], "1998-01-01", "2026-10-01")
    spy = closes.pop("SPY")
    spy = spy[spy.index <= pd.Timestamp(END)]
    days = spy.index
    px = pd.DataFrame(closes).reindex(days)
    cols = list(px.columns)
    P = px.values
    n = len(days)
    ma50, ma200 = px.rolling(50).mean().values, px.rolling(200).mean().values
    r12 = (px / px.shift(252) - 1).values
    r6 = (px / px.shift(126) - 1).values
    prev = (px.shift(252) / px.shift(504) - 1).values
    ye = px.groupby(days.year).last()
    ytd = P / ye.shift(1).reindex(days.year).values - 1
    hi52 = px.rolling(252, min_periods=200).max().values
    near = P / hi52
    ix = {t: k for k, t in enumerate(cols)}
    mem = np.zeros(P.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True
    rs = rs_rating(px, mem)
    log(f"數據準備好：{len(cols)} 隻股票，來源 {source}")

    # 前瞻結果（每隻股票 × 決定日）
    decs = [d for d in decision_days(days, "每月") if pd.Timestamp("2000-01-01") <= d <= LAST_DEC_12]
    rows = []
    for d in decs:
        i = days.get_loc(d)
        if i + 1 >= n:
            continue
        ent = P[i + 1]
        ok = mem[i] & np.isfinite(P[i]) & np.isfinite(ent) & (ent > 0)
        w1 = P[i + 1:min(i + 1 + 253, n)]
        w2 = P[i + 1:min(i + 1 + 505, n)]
        with np.errstate(invalid="ignore", divide="ignore"):
            mx1 = np.nanmax(np.where(np.isfinite(w1), w1, -np.inf), axis=0) / ent
            mx2 = np.nanmax(np.where(np.isfinite(w2), w2, -np.inf), axis=0) / ent
            # 12個月後：第252日收市；冇價就用窗口內最後有價
            last = np.full(P.shape[1], np.nan)
            for k in range(w1.shape[0] - 1, -1, -1):
                fill = np.isnan(last) & np.isfinite(w1[k])
                last[fill] = w1[k][fill]
            end12 = last / ent - 1
        scores = {
            "M0": None,
            "M1": r12[i],
            "M2": ytd[i],
            "M3": rs[i],
            "M4": r6[i],
            "M5": near[i] + 1e-6 * np.nan_to_num(r12[i], nan=0),
            "M6": np.where(prev[i] < 0, r12[i], np.nan),
            "M7": np.where((r12[i] >= 1.0) & (P[i] > ma50[i]) & (P[i] > ma200[i]), r12[i], np.nan),
            "M8": np.where((ytd[i] >= 0.30) & (P[i] >= hi52[i] * 0.7) & (P[i] > ma200[i]), ytd[i], np.nan),
            "M9": -near[i],
        }
        for m, sc in scores.items():
            if sc is None:
                pick = np.where(ok)[0]
            else:
                cand = np.where(ok & np.isfinite(sc))[0]
                pick = cand[np.argsort(-sc[cand], kind="stable")[:5]]
            for c in pick:
                rows.append((m, d, cols[c], mx1[c] >= 2.0, mx2[c] >= 3.0 if d <= LAST_DEC_24 else np.nan, end12[c]))
        if d.month == 12:
            log(f"{d.date()} 完成")
    df = pd.DataFrame(rows, columns=["方法", "決定日", "股票", "一年內倍升", "兩年內3倍", "12個月後回報"])
    df["年"] = df["決定日"].dt.year

    def stats(g):
        e = g["12個月後回報"].dropna()
        return pd.Series({"樣本": len(g), "一年內倍升率%": g["一年內倍升"].mean() * 100,
                          "兩年內3倍率%": pd.to_numeric(g["兩年內3倍"], errors="coerce").mean() * 100,
                          "12個月中位數%": e.median() * 100, "12個月平均%": e.mean() * 100,
                          "大跌率%": (e <= -0.30).mean() * 100})

    out = []
    for label, a, b in [("全期", 2000, 2025)] + ERAS:
        sub = df[(df["年"] >= a) & (df["年"] <= b)]
        base = stats(sub[sub["方法"] == "M0"])
        for m in NAMES:
            s = stats(sub[sub["方法"] == m])
            s["倍數"] = s["一年內倍升率%"] / base["一年內倍升率%"] if base["一年內倍升率%"] > 0 else np.nan
            s["3倍倍數"] = s["兩年內3倍率%"] / base["兩年內3倍率%"] if base["兩年內3倍率%"] > 0 else np.nan
            out.append({"期間": label, "方法": m, "名稱": NAMES[m], **s.to_dict()})
    res = pd.DataFrame(out)

    # 合格判斷
    verdict = []
    full = res[res["期間"] == "全期"].set_index("方法")
    b0 = full.loc["M0"]
    for m in NAMES:
        if m == "M0":
            continue
        f = full.loc[m]
        era_x = [res[(res["期間"] == e[0]) & (res["方法"] == m)]["倍數"].iloc[0] for e in ERAS]
        a_ = f["倍數"] >= 2
        b_ = all(x >= 1.5 for x in era_x)
        c_ = f["12個月中位數%"] >= b0["12個月中位數%"]
        d_ = f["大跌率%"] <= b0["大跌率%"] * 2
        verdict.append({"期間": "判斷", "方法": m, "名稱": NAMES[m],
                        "結果": ("合格" if a_ and b_ and c_ and d_ else "唔合格") +
                                f"（a{'✔' if a_ else '✘'} b{'✔' if b_ else '✘'} c{'✔' if c_ else '✘'} d{'✔' if d_ else '✘'}；"
                                f"各年代倍數 {' / '.join(f'{x:.1f}' for x in era_x)}）"})
    res = pd.concat([res, pd.DataFrame(verdict)], ignore_index=True)
    res.to_csv("v31_results.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    log("\n" + res.to_string())


if __name__ == "__main__":
    main()
