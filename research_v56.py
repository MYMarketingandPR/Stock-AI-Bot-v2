"""
research_v56.py — 第五十六輪：探索（搵賺錢股票嘅特徵，唔事先定義買入規則）
  （Esther 2026-11-19：「用2016-2020用撞大運既方法，搵岀賺錢股票既Pattern，唔需要先定義買入規矩。
     然後用撞大運既方法搵岀最佳止賺位同最佳止損位。」「唔想你事先寫好指標」）

性質：探索（discovery）。唔係通過測試。
  1. 只用 2016–2020 數據搵特徵。
  2. 驗證期（2021–2026）喺本腳本入面唔計算、唔輸出任何結果。
  3. 搵到嘅 pattern 要先寫成固定規則，之後再由另一輪腳本喺驗證期一次過測試。

搵嘅方法（事先寫定，跑之前已固定）：
  目標：信號日後 60 個交易日，超額回報（個股 − SPY）≥ +15% 記為「贏家」，否則「唔係」。
  特徵（全部只用當日或之前數據）：
    回報：5、20、60、120、250 日；超額回報（對 SPY）：60、250 日
    距離均線：MA20、MA50、MA200（%）
    距52週高位、距52週低位（%）；距60日高位（%）
    RSI(14)；MACD 柱（÷ 收市）；KDJ 的 K 值（9,3,3）
    20日波幅（標準差）；14日平均真實波幅（÷ 收市）
    量比（今日 ÷ 50日平均）；20日平均成交額（對數）
    今日收市在當日高低之間嘅位置（0–1）
  方法：
    A. 單個特徵：按十等分分組，睇贏家比例同平均超額回報，揀贏家比例差距最大嘅特徵。
    B. 兩個特徵：揀最有用嘅 5 個特徵，分成三等分做 3×3 格，列出贏家比例最高嘅格（樣本 ≥ 500）。
    C. 止損／止賺候選：對 B 嘅頂幾格，計算 60 日內嘅最大不利波動（MAE）同最大有利波動（MFE）分佈。
       候選止損 = 贏家 MAE 嘅第 20 百分位（即保留約 80% 贏家）；候選止賺 = 中位數 MFE。
       呢個只係候選，唔係已驗證嘅規則（事件次序同跳空未計）。

已知限制：
  - 多個特徵同多格比較，撞大運風險好高。
  - 樣本重疊（連續幾日嘅信號高度相關）。
  - 股票池係當時嘅標普500成分股，但價格數據可能缺漏。

輸出：v56_discovery.txt、v56_univariate.csv、v56_grid.csv、v56_stop_candidates.csv
==== 結果（2026-11-19 跑完，run 44533339951，記錄低，唔再改）====
基準贏家比例（60日超額 ≥ 15%）：8.9%（樣本 519,348）。
單個特徵頂 3：距52週高（贏家 4.2%→24.8%）；ATR14÷收市（3.6%→23.6%）；回報250日（4.6%→23.4%）。
  方向：最低組（即距高位最遠、250日跌最多、波動最大）嘅贏家比例最高。
兩個特徵頂格：ATR14÷收市第3組 + 超額回報250日第1組：樣本 86,776，贏家 20.9%，平均超額 +2.8%。
止損候選（頂格）：保留80%贏家＝跌約 −9%；保留90%＝跌約 −13%。止賺候選：中位 MFE 約 +16%，75% 分位約 +28%。
⚠️ 警告：2016–2020 包括 2020 年 3 月疫情急跌後嘅大反彈。「跌得最多、波動最大」嘅股票係咪主要係呢段反彈推動，要逐年檢查先知。
⚠️ 所有結果都係探索，驗證期未計算。
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v55 import load

M.PROGRESS = "v56_progress.txt"
log = M.log
DISC = ("2016-01-01", "2020-12-31")
HORIZON = 60
WIN = 0.15


def roll_fwd_min(C, h):
    # 未來 h 日（不含今日）嘅最低收市價
    return C.shift(-1).iloc[::-1].rolling(h, min_periods=h).min().iloc[::-1]


def roll_fwd_max(C, h):
    return C.shift(-1).iloc[::-1].rolling(h, min_periods=h).max().iloc[::-1]


def build(D):
    C, H, L, V = D["C"], D["H"], D["L"], D["V"]
    spy = D["spy"]
    F = {}
    ret1 = C.pct_change(fill_method=None)
    for k in (5, 20, 60, 120, 250):
        F[f"回報{k}日"] = C / C.shift(k) - 1
    spy_ret = {k: spy / spy.shift(k) - 1 for k in (60, 250)}
    F["超額回報60日"] = F["回報60日"].sub(spy_ret[60], axis=0)
    F["超額回報250日"] = F["回報250日"].sub(spy_ret[250], axis=0)
    for k in (20, 50, 200):
        F[f"距MA{k}"] = C / C.rolling(k).mean() - 1
    F["距52週高"] = C / H.rolling(250, min_periods=250).max() - 1
    F["距52週低"] = C / L.rolling(250, min_periods=250).min() - 1
    F["距60日高"] = C / C.rolling(60).max() - 1
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    F["RSI14"] = 100 - 100 / (1 + up / dn)
    ema12 = C.ewm(span=12, adjust=False).mean()
    ema26 = C.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    F["MACD柱÷收市"] = (macd - macd.ewm(span=9, adjust=False).mean()) / C
    ll = L.rolling(9).min()
    hh = H.rolling(9).max()
    rsv = (C - ll) / (hh - ll) * 100
    kk = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    F["KDJ_K"] = kk
    F["20日波幅"] = ret1.rolling(20).std()
    F["ATR14÷收市"] = (H - L).rolling(14).mean() / C
    F["量比50日"] = V / V.rolling(50).mean()
    F["成交額對數20日"] = np.log10((C * V).rolling(20).mean())
    F["收市位置"] = (C - L) / (H - L)
    return F


def main():
    open(M.PROGRESS, "w").close()
    D = load()
    days = D["days"]
    C, H, L = D["C"], D["H"], D["L"]
    mem = pd.DataFrame(D["mem"], index=days, columns=D["stocks"])
    F = build(D)
    log("特徵計算完成")
    spy = D["spy"]
    entry = C.shift(-1)
    exitp = C.shift(-(HORIZON + 1))
    fwd = exitp / entry - 1
    spy_f = spy.shift(-(HORIZON + 1)) / spy.shift(-1) - 1
    ex = fwd.sub(spy_f, axis=0)
    disc_mask = (days >= pd.Timestamp(DISC[0])) & (days <= pd.Timestamp(DISC[1]))
    m = mem & pd.DataFrame(np.repeat(disc_mask[:, None], C.shape[1], axis=1), index=days, columns=C.columns)
    m = m & ex.notna()
    base_vals = ex.values[m.values]
    base_n = int(base_vals.size)
    base_win = float(np.mean(base_vals >= WIN)) if base_n else np.nan
    L_out = ["=== 第五十六輪：探索（只用 2016–2020；驗證期唔計算）===",
             f"目標：60 日超額回報 ≥ {WIN * 100:.0f}% 為贏家；基準贏家比例 {base_win * 100:.1f}%（樣本 {base_n:,}）", ""]

    # A. 單個特徵十等分
    uni = []
    for name, X in F.items():
        x = X[m].values.ravel()
        y = ex[m].values.ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if x.size < 1000:
            continue
        qs = np.nanquantile(x, np.linspace(0, 1, 11))
        qs = np.unique(qs)
        if len(qs) < 3:
            continue
        bins = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, len(qs) - 2)
        rows = []
        for b in range(len(qs) - 1):
            sel = bins == b
            if sel.sum() == 0:
                continue
            rows.append(dict(特徵=name, 分組=b + 1, 下限=qs[b], 上限=qs[b + 1], 樣本=int(sel.sum()),
                             贏家比例=float(np.mean(y[sel] >= WIN)), 平均超額=float(np.mean(y[sel]))))
        if rows:
            wr = [r["贏家比例"] for r in rows]
            uni.append(dict(特徵=name, 最高贏家比例=max(wr), 最低贏家比例=min(wr), 差距=max(wr) - min(wr), _rows=rows))
    uni.sort(key=lambda r: -r["差距"])
    pd.DataFrame([{k: v for k, v in r.items() if k != "_rows"} for r in uni]).to_csv(
        "v56_univariate.csv", index=False, encoding="utf-8-sig")
    L_out.append("=== A. 單個特徵（按贏家比例差距排序，頂 10）===")
    for r in uni[:10]:
        L_out.append(f"　{r['特徵']}：贏家比例由 {r['最低贏家比例'] * 100:.1f}% 到 {r['最高贏家比例'] * 100:.1f}%（差 {r['差距'] * 100:.1f}%）")
    L_out.append("")
    for r in uni[:3]:
        L_out.append(f"　— {r['特徵']} 十等分：")
        for row in r["_rows"]:
            L_out.append(f"　　第{row['分組']}組 [{row['下限']:.3g}, {row['上限']:.3g}] 樣本{row['樣本']:,} 贏家{row['贏家比例'] * 100:.1f}% 平均超額{row['平均超額'] * 100:+.1f}%")
    L_out.append("")

    # B. 頂 5 個特徵做 3×3 格
    top5 = [r["特徵"] for r in uni[:5]]
    grid_rows = []
    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            fa, fb = top5[i], top5[j]
            xa = F[fa][m].values.ravel()
            xb = F[fb][m].values.ravel()
            y = ex[m].values.ravel()
            ok = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
            xa, xb, y = xa[ok], xb[ok], y[ok]
            qa = np.quantile(xa, [1 / 3, 2 / 3])
            qb = np.quantile(xb, [1 / 3, 2 / 3])
            ba = np.searchsorted(qa, xa, side="right")
            bb = np.searchsorted(qb, xb, side="right")
            for a in range(3):
                for b in range(3):
                    sel = (ba == a) & (bb == b)
                    n = int(sel.sum())
                    if n < 500:
                        continue
                    grid_rows.append(dict(特徵A=fa, 特徵B=fb, A分組=a + 1, B分組=b + 1, 樣本=n,
                                          贏家比例=float(np.mean(y[sel] >= WIN)), 平均超額=float(np.mean(y[sel]))))
    grid = pd.DataFrame(grid_rows).sort_values("贏家比例", ascending=False) if grid_rows else pd.DataFrame()
    grid.to_csv("v56_grid.csv", index=False, encoding="utf-8-sig")
    L_out.append("=== B. 兩個特徵 3×3 格（樣本 ≥ 500，贏家比例最高頂 8）===")
    L_out.append("　（1 = 特徵最低三分一，3 = 最高三分一）")
    for r in grid.head(8).to_dict("records"):
        L_out.append(f"　{r['特徵A']}第{r['A分組']}組 + {r['特徵B']}第{r['B分組']}組：樣本{r['樣本']:,} 贏家{r['贏家比例'] * 100:.1f}% 平均超額{r['平均超額'] * 100:+.1f}%")
    L_out.append("")

    # C. 頂格嘅 MAE／MFE（止損、止賺候選）
    mae = roll_fwd_min(C, HORIZON) / C - 1
    mfe = roll_fwd_max(C, HORIZON) / C - 1
    stop_rows = []
    for r in grid.head(3).to_dict("records"):
        xa = F[r["特徵A"]]
        xb = F[r["特徵B"]]
        qa = np.nanquantile(xa[m].values.ravel(), [1 / 3, 2 / 3])
        qb = np.nanquantile(xb[m].values.ravel(), [1 / 3, 2 / 3])
        ba = np.searchsorted(qa, xa.values, side="right") + 1
        bb = np.searchsorted(qb, xb.values, side="right") + 1
        fin = np.isfinite(xa.values) & np.isfinite(xb.values)
        sel = m.values & fin & (ba == r["A分組"]) & (bb == r["B分組"])
        win_sel = sel & (ex.values >= WIN)
        mv = mae.values[win_sel]
        fv = mfe.values[sel]
        mv = mv[np.isfinite(mv)]
        fv = fv[np.isfinite(fv)]
        if mv.size < 50:
            continue
        stop_rows.append(dict(格=f"{r['特徵A']}第{r['A分組']}組 + {r['特徵B']}第{r['B分組']}組", 贏家數=int(win_sel.sum()),
                              止損候選_保留80贏家=float(np.percentile(mv, 20)) * 100,
                              止損候選_保留90贏家=float(np.percentile(mv, 10)) * 100,
                              止賺候選_中位MFE=float(np.median(fv)) * 100,
                              止賺候選_75百分位MFE=float(np.percentile(fv, 75)) * 100))
    pd.DataFrame(stop_rows).to_csv("v56_stop_candidates.csv", index=False, encoding="utf-8-sig")
    L_out.append("=== C. 頂格嘅止損／止賺候選（只係候選，未經驗證）===")
    for r in stop_rows:
        L_out.append(f"　{r['格']}（贏家 {r['贏家數']:,}）：止損候選 保留80%贏家＝跌 {r['止損候選_保留80贏家']:.1f}%，保留90%＝跌 {r['止損候選_保留90贏家']:.1f}%；"
                     f"止賺候選 中位MFE＝升 {r['止賺候選_中位MFE']:.1f}%，75%分位＝升 {r['止賺候選_75百分位MFE']:.1f}%")
    L_out.append("")
    L_out.append("注意：以上係探索結果。驗證期（2021–2026）未計算。多個特徵同多格比較，撞大運風險高。")

    pd.DataFrame({"report": L_out}).to_csv("v56_discovery.csv", index=False, encoding="utf-8-sig")
    open("v56_discovery.txt", "w", encoding="utf-8").write("\n".join(L_out))
    log("完成")
    print("\n".join(L_out))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
