"""
research_v32.py — 第三十二輪：跟升勢揀出嚟嘅股票，可唔可以事先認出邊啲之後會大跌？（Esther 2026-10-14：「好試吓做第32輪」）

純研究，唔改任何模擬交易（線A/B/C/D/D2/E、輪動、SPY訊號全部照舊），唔改之前任何結果。

⚠️ 預先登記（跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：
共同設定同第三十一輪一樣：U1 當時真實標普500、已計股息、每月最後交易日決定（2000-01 至 2025-09）、
  入場 = 下一個交易日收市、每個方法揀最高5隻。
股票池（「動量股」）：第三十一輪 M1 12個月、M2 今年、M3 RS、M4 6個月、M7 線D條件、M8 線E條件 揀中嘅股票，
  同一個月同一隻只計一次。另外分開列 M7（線D條件）同 M8（線E條件）做參考。
結果（同第三十一輪一樣）：大跌 = 12個月後回報 ≤ -30%；倍升 = 入場後252個交易日內任何一日收市 ≥ 入場價 × 2。

警號（全部只用決定日已知資料；只有呢6個，跑完唔加唔減唔改數字）：
  F1 升得太多：12個月升幅 > +300%
  F2 第二次倍升：之前一年（252至504日前）升幅 ≥ +100%
  F3 波動特別大：60日每日回報標準差，喺當日成分股排頭10%（百分位 ≥ 90）
  F4 離50日線太遠：收市 ÷ 50日平均 ≥ 1.25
  F5 大市弱：SPY 收市 < SPY 200日平均
  F6 任何一個（F1-F5 有一個或以上）

量度：每個警號 → 有警號 vs 冇警號嘅大跌率、倍升率、12個月中位數；剔走有警號之後，大跌股剩低幾多、倍升股剩低幾多。
合格（「值得再做一輪交易測試」），用動量股池，全部要符合：
  (a) 全期：剔走後大跌率 ≤ 原本大跌率 × 2/3（大跌率少咗最少三分一）
  (b) 全期：剔走後仲保留 ≥ 80% 嘅倍升股（按數目）
  (c) 四個年代（2000-09、2010-14、2015-19、2020-25）每個都符合 (a) 同 (b)
  就算合格，都只係「值得再試」，唔會直接加入模擬交易；要另外預先登記一輪完整交易測試先算。
  6個警號一齊比，試得多總有好彩 → 只信四個年代都企得穩嘅結果。

已知限制：同第三十一輪一樣，Yahoo 冇數據嘅已除牌股唔會入選。
  「12個月後」係假設揸足一年唔賣；真實線D/E有賣出線，實際蝕幅通常細過呢度。

輸出：v32_results.csv、v32_progress.txt
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v8 import decision_days
from research_v23 import rs_rating

M.PROGRESS = "v32_progress.txt"
log = M.log
END = "2026-09-30"
LAST_DEC_12 = pd.Timestamp("2025-09-30")
ERAS = [("2000-09", 2000, 2009), ("2010-14", 2010, 2014), ("2015-19", 2015, 2019), ("2020-25", 2020, 2025)]
FLAGS = {"F1": "升得太多（12個月>+300%）", "F2": "第二次倍升", "F3": "波動特別大（頭10%）",
         "F4": "離50日線太遠（≥1.25倍）", "F5": "大市弱（SPY<200日線）", "F6": "任何一個警號"}


def pct_rank_rows(a, mem):
    out = np.full(a.shape, np.nan)
    for i in range(a.shape[0]):
        ref = a[i][mem[i] & np.isfinite(a[i])]
        if len(ref) < 50:
            continue
        ref = np.sort(ref)
        f = np.isfinite(a[i])
        out[i, f] = np.searchsorted(ref, a[i][f], side="right") / len(ref) * 100
    return out


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
    vol60 = px.pct_change(fill_method=None).rolling(60, min_periods=50).std().values
    spy_weak = (spy < spy.rolling(200).mean()).values
    ix = {t: k for k, t in enumerate(cols)}
    mem = np.zeros(P.shape, dtype=bool)
    cache = {}
    for i, d in enumerate(days):
        key = (d.year, d.month)
        if key not in cache:
            cache[key] = [ix[t] for t in members_fn(d) if t in ix]
        mem[i, cache[key]] = True
    rs = rs_rating(px, mem)
    volpct = pct_rank_rows(vol60, mem)
    log(f"數據準備好：{len(cols)} 隻股票，來源 {source}")

    decs = [d for d in decision_days(days, "每月") if pd.Timestamp("2000-01-01") <= d <= LAST_DEC_12]
    rows = []
    for d in decs:
        i = days.get_loc(d)
        if i + 1 >= n:
            continue
        ent = P[i + 1]
        ok = mem[i] & np.isfinite(P[i]) & np.isfinite(ent) & (ent > 0)
        w1 = P[i + 1:min(i + 1 + 253, n)]
        with np.errstate(invalid="ignore", divide="ignore"):
            mx1 = np.max(np.where(np.isfinite(w1), w1, -np.inf), axis=0) / ent
            last = np.full(P.shape[1], np.nan)
            for k in range(w1.shape[0] - 1, -1, -1):
                fill = np.isnan(last) & np.isfinite(w1[k])
                last[fill] = w1[k][fill]
            end12 = last / ent - 1
        scores = {
            "M1": r12[i], "M2": ytd[i], "M3": rs[i], "M4": r6[i],
            "M7": np.where((r12[i] >= 1.0) & (P[i] > ma50[i]) & (P[i] > ma200[i]), r12[i], np.nan),
            "M8": np.where((ytd[i] >= 0.30) & (P[i] >= hi52[i] * 0.7) & (P[i] > ma200[i]), ytd[i], np.nan),
        }
        picked = {}
        for m, sc in scores.items():
            cand = np.where(ok & np.isfinite(sc))[0]
            for c in cand[np.argsort(-sc[cand], kind="stable")[:5]]:
                picked.setdefault(c, set()).add(m)
        for c, ms in picked.items():
            f1 = bool(r12[i, c] > 3.0)
            f2 = bool(np.isfinite(prev[i, c]) and prev[i, c] >= 1.0)
            f3 = bool(np.isfinite(volpct[i, c]) and volpct[i, c] >= 90)
            f4 = bool(np.isfinite(ma50[i, c]) and P[i, c] / ma50[i, c] >= 1.25)
            f5 = bool(spy_weak[i])
            rows.append({"決定日": d, "股票": cols[c], "M7": "M7" in ms, "M8": "M8" in ms,
                         "倍升": bool(mx1[c] >= 2.0), "大跌": bool(end12[c] <= -0.30), "回報": end12[c],
                         "F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F6": f1 or f2 or f3 or f4 or f5})
        if d.month == 12:
            log(f"{d.date()} 完成")
    df = pd.DataFrame(rows)
    df["年"] = df["決定日"].dt.year

    def summarise(sub, pool, period):
        out = []
        nd, nb = int(sub["大跌"].sum()), int(sub["倍升"].sum())
        cr = sub["大跌"].mean() * 100
        out.append({"股票池": pool, "期間": period, "警號": "（全部）", "樣本": len(sub), "有警號比例%": np.nan,
                    "有警號大跌率%": np.nan, "冇警號大跌率%": cr, "有警號倍升率%": np.nan, "冇警號倍升率%": sub["倍升"].mean() * 100,
                    "有警號中位數%": np.nan, "冇警號中位數%": sub["回報"].median() * 100,
                    "剔走後大跌率÷原本": 1.0, "保留倍升股%": 100.0, "保留大跌股%": 100.0})
        for f in FLAGS:
            y, k = sub[sub[f]], sub[~sub[f]]
            out.append({"股票池": pool, "期間": period, "警號": f"{f} {FLAGS[f]}", "樣本": len(sub),
                        "有警號比例%": len(y) / len(sub) * 100 if len(sub) else np.nan,
                        "有警號大跌率%": y["大跌"].mean() * 100 if len(y) else np.nan,
                        "冇警號大跌率%": k["大跌"].mean() * 100 if len(k) else np.nan,
                        "有警號倍升率%": y["倍升"].mean() * 100 if len(y) else np.nan,
                        "冇警號倍升率%": k["倍升"].mean() * 100 if len(k) else np.nan,
                        "有警號中位數%": y["回報"].median() * 100 if len(y) else np.nan,
                        "冇警號中位數%": k["回報"].median() * 100 if len(k) else np.nan,
                        "剔走後大跌率÷原本": (k["大跌"].mean() * 100 / cr) if len(k) and cr > 0 else np.nan,
                        "保留倍升股%": k["倍升"].sum() / nb * 100 if nb else np.nan,
                        "保留大跌股%": k["大跌"].sum() / nd * 100 if nd else np.nan})
        return out

    out = []
    for pool, mask in [("動量股", np.ones(len(df), bool)), ("線D條件", df["M7"].values), ("線E條件", df["M8"].values)]:
        sub0 = df[mask]
        for label, a, b in [("全期", 2000, 2025)] + ERAS:
            out += summarise(sub0[(sub0["年"] >= a) & (sub0["年"] <= b)], pool, label)
    res = pd.DataFrame(out)

    verdict = []
    mom = res[res["股票池"] == "動量股"]
    for f in FLAGS:
        name = f"{f} {FLAGS[f]}"
        g = mom[mom["警號"] == name].set_index("期間")

        def okp(p):
            return bool(g.loc[p, "剔走後大跌率÷原本"] <= 2 / 3 and g.loc[p, "保留倍升股%"] >= 80)
        a_ = bool(g.loc["全期", "剔走後大跌率÷原本"] <= 2 / 3)
        b_ = bool(g.loc["全期", "保留倍升股%"] >= 80)
        eras = [okp(e[0]) for e in ERAS]
        verdict.append({"股票池": "判斷", "期間": "", "警號": name,
                        "結果": ("合格" if a_ and b_ and all(eras) else "唔合格") +
                                f"（a{'✔' if a_ else '✘'} b{'✔' if b_ else '✘'} c {''.join('✔' if e else '✘' for e in eras)}）"})
    res = pd.concat([res, pd.DataFrame(verdict)], ignore_index=True)
    res.to_csv("v32_results.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    log("\n" + res.to_string())


if __name__ == "__main__":
    main()
