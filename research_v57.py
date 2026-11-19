"""
research_v57.py — 第五十七輪：擴大特徵探索（只用 2016–2020；驗證期唔計算）
  （Esther 2026-11-19：「特徵太少，要加多啲…揀最有用嘅 5–10 個特徵，2–5 個組合」；
     「我目的是搵到高獲利撞大運pattern，再分析，然後驗證。」）

性質：探索（discovery）。唔係通過測試。第56輪（20 個特徵）保留原樣，本輪係獨立一輪。

數據限制（要講清楚）：
  - 期權持倉量（OI）同最高持倉價位：免費數據冇 2016–2020 嘅歷史，所以**唔計算**。
    替代：按成交量加權平均價（VWAP）60 日、250 日，代表近期主要成交價位。
  - 週線：由日線計算，只用已經完結嘅週（週五收市後先生效，當週唔用），冇未來數據。

搵嘅步驟（事先寫定）：
  1. 目標：信號日後 60 個交易日，超額回報（個股 − SPY）≥ +15% 為「贏家」。
  2. 特徵約 40 個（見下），每個先做十等分，睇贏家比例差距，揀頂 10 個。
  3. 用頂 10 個特徵做組合：
       2 個特徵：三等分 3×3 格；
       3、4、5 個特徵：按中位數分兩邊（每格 2 個值）。
     每種組合列出贏家比例最高嘅格（樣本 ≥ 500）。
  4. 對頂格計算 60 日內最大不利波動（MAE）同最大有利波動（MFE），列出止損／止賺候選。
  5. 因為格數好多（約一萬多格），撞大運風險非常高。任何 pattern 都只係候選，必須之後用未見過嘅數據驗證。

已知限制：
  - 格數好多，而且唔同組合之間互相重疊。
  - 樣本重疊（連續幾日嘅信號高度相關）。
  - 2016–2020 數據已經俾第49至54輪用過，所以唔係真正未見過嘅數據。
  - 驗證只可以用紙上模擬之後嘅數據，或者之後新出現嘅數據。

輸出：v57_discovery.txt、v57_univariate.csv、v57_combos.csv、v57_stop_candidates.csv
==== 結果（待跑）====
"""
import itertools

import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v55 import load

M.PROGRESS = "v57_progress.txt"
log = M.log
DISC = ("2016-01-01", "2020-12-31")
HORIZON = 60
WIN = 0.15
TOPK = 10
MINN = 500


def weekly_to_daily(X, days):
    # 每週嘅值放喺該週五之後一日（週六）先生效；週五當日仍然用上一週
    X = X.copy()
    X.index = X.index + pd.Timedelta(days=1)
    return X.reindex(X.index.union(days)).ffill().reindex(days)


def build(D):
    C, H, L, V = D["C"], D["H"], D["L"], D["V"]
    spy = D["spy"]
    days = D["days"]
    F = {}
    ret1 = C.pct_change(fill_method=None)
    for k in (5, 20, 60, 120, 250):
        F[f"回報{k}日"] = C / C.shift(k) - 1
    F["超額回報60日"] = F["回報60日"].sub(spy / spy.shift(60) - 1, axis=0)
    F["超額回報250日"] = F["回報250日"].sub(spy / spy.shift(250) - 1, axis=0)
    sma = {k: C.rolling(k).mean() for k in (5, 10, 20, 50, 200)}
    for k in (5, 10, 20, 50, 200):
        F[f"距SMA{k}"] = C / sma[k] - 1
    F["SMA20_5日變化"] = sma[20] / sma[20].shift(5) - 1
    F["SMA50_20日變化"] = sma[50] / sma[50].shift(20) - 1
    F["距52週高"] = C / H.rolling(250, min_periods=250).max() - 1
    F["距52週低"] = C / L.rolling(250, min_periods=250).min() - 1
    F["距60日高"] = C / C.rolling(60).max() - 1

    def rsi(X, n):
        d = X.diff()
        up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        return 100 - 100 / (1 + up / dn)

    F["RSI14"] = rsi(C, 14)
    F["RSI5"] = rsi(C, 5)
    macd = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
    F["MACD柱÷收市"] = (macd - macd.ewm(span=9, adjust=False).mean()) / C
    ll = L.rolling(9).min()
    hh = H.rolling(9).max()
    F["KDJ_K"] = ((C - ll) / (hh - ll) * 100).ewm(alpha=1 / 3, adjust=False).mean()
    F["20日波幅"] = ret1.rolling(20).std()
    F["ATR14÷收市"] = (H - L).rolling(14).mean() / C
    m20 = C.rolling(20).mean()
    s20 = C.rolling(20).std()
    F["布林位置20日"] = (C - (m20 - 2 * s20)) / (4 * s20)
    F["量比5日"] = V / V.rolling(5).mean()
    F["量比20日"] = V / V.rolling(20).mean()
    F["量比50日"] = V / V.rolling(50).mean()
    F["成交額對數20日"] = np.log10((C * V).rolling(20).mean())
    up_v = V.where(ret1 > 0).rolling(60, min_periods=20).mean()
    dn_v = V.where(ret1 < 0).rolling(60, min_periods=20).mean()
    F["上升量÷下跌量60日"] = up_v / dn_v
    obv = (np.sign(ret1) * V).fillna(0).cumsum()
    F["OBV_20日變化÷平均量"] = (obv - obv.shift(20)) / V.rolling(20).mean() / 20
    F["收市位置"] = (C - L) / (H - L)
    vw60 = (C * V).rolling(60).sum() / V.rolling(60).sum()
    vw250 = (C * V).rolling(250).sum() / V.rolling(250).sum()
    F["距VWAP60"] = C / vw60 - 1
    F["距VWAP250"] = C / vw250 - 1
    # 週線（只用完結嘅週）
    Cw = C.resample("W-FRI").last()
    Hw = H.resample("W-FRI").max()
    Lw = L.resample("W-FRI").min()
    dw = Cw.diff()
    upw = dw.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dnw = (-dw.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi_w = 100 - 100 / (1 + upw / dnw)
    macd_w = Cw.ewm(span=12, adjust=False).mean() - Cw.ewm(span=26, adjust=False).mean()
    F["週線RSI14"] = weekly_to_daily(rsi_w, days)
    F["週線MACD柱÷收市"] = weekly_to_daily((macd_w - macd_w.ewm(span=9, adjust=False).mean()) / Cw, days)
    F["距週線MA10"] = weekly_to_daily(Cw / Cw.rolling(10).mean() - 1, days)
    F["距週線MA40"] = weekly_to_daily(Cw / Cw.rolling(40).mean() - 1, days)
    F["週線20週回報"] = weekly_to_daily(Cw / Cw.shift(20) - 1, days)
    F["距週線52週高"] = weekly_to_daily(Cw / Hw.rolling(52).max() - 1, days)
    return F


def main():
    open(M.PROGRESS, "w").close()
    D = load()
    days = D["days"]
    C = D["C"]
    mem = pd.DataFrame(D["mem"], index=days, columns=C.columns)
    F = build(D)
    log(f"特徵：{len(F)} 個")
    spy = D["spy"]
    ex = (C.shift(-(HORIZON + 1)) / C.shift(-1) - 1).sub(spy.shift(-(HORIZON + 1)) / spy.shift(-1) - 1, axis=0)
    disc = (days >= pd.Timestamp(DISC[0])) & (days <= pd.Timestamp(DISC[1]))
    m = mem.values & disc[:, None] & np.isfinite(ex.values)
    y = ex.values[m]
    base = float(np.mean(y >= WIN))
    L_out = ["=== 第五十七輪：擴大特徵探索（只用 2016–2020；驗證期唔計算）===",
             f"特徵數 {len(F)}；目標：60 日超額回報 ≥ {WIN * 100:.0f}% 為贏家；基準贏家比例 {base * 100:.1f}%（樣本 {y.size:,}）",
             "注意：期權持倉量無歷史數據，以 VWAP 代替。", ""]

    # A. 單個特徵十等分
    uni = []
    cols = {}
    for name, X in F.items():
        x = X.values[m]
        ok = np.isfinite(x)
        if ok.sum() < 5000:
            continue
        xx, yy = x[ok], y[ok]
        qs = np.unique(np.quantile(xx, np.linspace(0, 1, 11)))
        if len(qs) < 3:
            continue
        b = np.clip(np.searchsorted(qs, xx, side="right") - 1, 0, len(qs) - 2)
        wr = [float(np.mean(yy[b == k] >= WIN)) for k in range(len(qs) - 1) if (b == k).sum() > 0]
        uni.append(dict(特徵=name, 最高=max(wr), 最低=min(wr), 差距=max(wr) - min(wr)))
    uni.sort(key=lambda r: -r["差距"])
    pd.DataFrame(uni).to_csv("v57_univariate.csv", index=False, encoding="utf-8-sig")
    top = [r["特徵"] for r in uni[:TOPK]]
    L_out.append(f"=== A. 單個特徵（按贏家比例差距排序，頂 {TOPK} 個用於組合）===")
    for r in uni[:TOPK]:
        L_out.append(f"　{r['特徵']}：贏家比例 {r['最低'] * 100:.1f}% 至 {r['最高'] * 100:.1f}%（差 {r['差距'] * 100:.1f}%）")
    L_out.append("")

    # B. 組合（2–5 個特徵，頂 10 個之中）
    arr = {f: F[f].values[m] for f in top}
    ok_all = np.ones(y.size, dtype=bool)
    for f in top:
        ok_all &= np.isfinite(arr[f])
    tert = {}
    med = {}
    for f in top:
        x = arr[f]
        q = np.quantile(x[ok_all], [1 / 3, 2 / 3])
        tert[f] = np.searchsorted(q, x, side="right")
        med[f] = (x > np.median(x[ok_all])).astype(int)
    combos_out = []
    n_cells = 0
    for k in range(2, 6):
        best = []
        for combo in itertools.combinations(top, k):
            if k == 2:
                key = tert[combo[0]] * 3 + tert[combo[1]]
                nb = 9
            else:
                key = np.zeros(y.size, dtype=int)
                for f in combo:
                    key = key * 2 + med[f]
                nb = 2 ** k
            for cell in range(nb):
                sel = ok_all & (key == cell)
                n = int(sel.sum())
                n_cells += 1
                if n < MINN:
                    continue
                wr = float(np.mean(y[sel] >= WIN))
                best.append(dict(組合=" + ".join(combo), 格=cell, 樣本=n, 贏家比例=wr, 平均超額=float(np.mean(y[sel]))))
        best.sort(key=lambda r: -r["贏家比例"])
        combos_out += [dict(特徵數=k, **r) for r in best[:20]]
        L_out.append(f"=== B. {k} 個特徵組合（樣本 ≥ {MINN}，贏家比例頂 5）===")
        for r in best[:5]:
            L_out.append(f"　{r['組合']}（格 {r['格']}）：樣本 {r['樣本']:,} ｜ 贏家 {r['贏家比例'] * 100:.1f}% ｜ 平均超額 {r['平均超額'] * 100:+.1f}%")
        L_out.append("")
    pd.DataFrame(combos_out).to_csv("v57_combos.csv", index=False, encoding="utf-8-sig")
    L_out.append(f"　總共評估咗 {n_cells:,} 個格（撞大運風險非常高）")
    L_out.append("")

    # C. 頂格嘅 MAE／MFE（只對 2 個特徵組合頂 3 格做候選）
    mae = (C.shift(-1).iloc[::-1].rolling(HORIZON, min_periods=HORIZON).min().iloc[::-1] / C - 1).values[m]
    mfe = (C.shift(-1).iloc[::-1].rolling(HORIZON, min_periods=HORIZON).max().iloc[::-1] / C - 1).values[m]
    stop_rows = []
    best2 = [r for r in combos_out if r["特徵數"] == 2][:3]
    for r in best2:
        fa, fb = r["組合"].split(" + ")
        cell = r["格"]
        sel = ok_all & ((tert[fa] * 3 + tert[fb]) == cell)
        win_sel = sel & (y >= WIN) & np.isfinite(mae)
        mv = mae[win_sel]
        fv = mfe[sel & np.isfinite(mfe)]
        if mv.size < 50:
            continue
        stop_rows.append(dict(組合=r["組合"], 格=cell, 贏家數=int(win_sel.sum()),
                              止損_保留80贏家=float(np.percentile(mv, 20)) * 100,
                              止損_保留90贏家=float(np.percentile(mv, 10)) * 100,
                              止賺_中位MFE=float(np.median(fv)) * 100,
                              止賺_75百分位MFE=float(np.percentile(fv, 75)) * 100))
    pd.DataFrame(stop_rows).to_csv("v57_stop_candidates.csv", index=False, encoding="utf-8-sig")
    L_out.append("=== C. 頂格止損／止賺候選（只係候選，未驗證）===")
    for r in stop_rows:
        L_out.append(f"　{r['組合']}（格 {r['格']}，贏家 {r['贏家數']:,}）：止損候選 保留80%贏家＝跌 {r['止損_保留80贏家']:.1f}%，保留90%＝跌 {r['止損_保留90贏家']:.1f}%；"
                     f"止賺候選 中位MFE＝升 {r['止賺_中位MFE']:.1f}%，75%分位＝升 {r['止賺_75百分位MFE']:.1f}%")
    L_out.append("")
    L_out.append("注意：探索結果，驗證期未計算。搵到嘅 pattern 必須用未見過嘅數據驗證。")

    open("v57_discovery.txt", "w", encoding="utf-8").write("\n".join(L_out))
    pd.DataFrame({"report": L_out}).to_csv("v57_discovery.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L_out))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
