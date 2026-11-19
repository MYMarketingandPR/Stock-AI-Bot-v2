"""
research_v58.py — 第五十八輪：逐年檢查第56、57輪嘅探索頂格（只用 2016–2020；驗證期唔計算）
  （Esther 2026-11-19：「好」，即係同意先做「逐年檢查（尤其排除 2020 年）」。）

性質：探索嘅穩健性檢查。唔係通過測試，亦唔係驗證。
  目的：睇頂格嘅 pattern 係咪每年都有效，定係主要靠 2020 年疫情急跌之後嘅反彈。

檢查嘅 pattern（用第57輪嘅定義，格標籤照第57輪：三等分，0 = 最低三分一，2 = 最高三分一）：
  P1：距52週高 最低三分一 + 距VWAP250 最高三分一（第57輪 2 個特徵頂格）
  P2：距52週高 最低十分一（第56、57輪單個特徵頂）
  P3：ATR14÷收市 最高十分一（第56輪單個特徵頂，波動最大）
  基準：同一段期間所有成員股票日。

檢查內容（每個 pattern）：
  1. 逐年（2016、2017、2018、2019、2020）：樣本數、贏家比例（60日超額 ≥ 15%）、同年基準比例。
  2. 排除 2020 年 2 月 15 日至 6 月 30 日（疫情急跌同反彈期）之後：整體贏家比例、同基準比較。
  3. 只計 2020 年疫情期：贏家比例。
  4. 「升幅／反彈」集中度：頂 1% 贏家佔總超額回報嘅比例（睇係咪少數極端值拉高平均）。

判斷（事先寫定）：
  如果 P1 或 P2 或 P3 喺 5 年之中有 4 年或以上，贏家比例高過同年基準，而且排除疫情期之後仍然高過基準，
  先至值得寫成固定規則去做下一步驗證。否則視為主要係疫情反彈。

已知限制：
  - 格嘅分界點係用 2016–2020 全期計算，會有少少前瞻偏差（呢個同第57輪一樣）。
  - 每年樣本數唔同，2016 年唔同 2020 年。
  - 呢個唔係驗證。就算通過，都要用紙上模擬之後嘅數據再驗證。

輸出：v58_by_year.csv、v58_discovery.txt
==== 結果（待跑）====
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v55 import load
from research_v57 import build

M.PROGRESS = "v58_progress.txt"
log = M.log
DISC = ("2016-01-01", "2020-12-31")
HORIZON = 60
WIN = 0.15
CRISIS = (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-30"))


def main():
    open(M.PROGRESS, "w").close()
    D = load()
    days = D["days"]
    C = D["C"]
    mem = pd.DataFrame(D["mem"], index=days, columns=C.columns)
    F = build(D)
    spy = D["spy"]
    ex = (C.shift(-(HORIZON + 1)) / C.shift(-1) - 1).sub(spy.shift(-(HORIZON + 1)) / spy.shift(-1) - 1, axis=0)
    disc = (days >= pd.Timestamp(DISC[0])) & (days <= pd.Timestamp(DISC[1]))
    base_m = mem.values & disc[:, None] & np.isfinite(ex.values)

    def tert_of(name):
        x = F[name].values
        ok = base_m & np.isfinite(x)
        q = np.quantile(x[ok], [1 / 3, 2 / 3])
        return np.searchsorted(q, x, side="right")

    def dec_low(name):
        x = F[name].values
        ok = base_m & np.isfinite(x)
        return x <= np.quantile(x[ok], 0.10)

    def dec_high(name):
        x = F[name].values
        ok = base_m & np.isfinite(x)
        return x >= np.quantile(x[ok], 0.90)

    t1 = tert_of("距52週高")
    t2 = tert_of("距VWAP250")
    P = {
        "P1 距52週高最低三分一 + 距VWAP250最高三分一": (t1 == 0) & (t2 == 2),
        "P2 距52週高最低十分一": dec_low("距52週高"),
        "P3 ATR14÷收市最高十分一": dec_high("ATR14÷收市"),
    }
    yr = np.repeat(days.year.values[:, None], C.shape[1], axis=1)
    crisis = np.repeat(((days >= CRISIS[0]) & (days <= CRISIS[1]))[:, None], C.shape[1], axis=1)
    yv = ex.values
    rows = []
    L = ["=== 第五十八輪：逐年檢查頂格（只用 2016–2020）===",
         "（贏家 = 60 日超額回報 ≥ 15%）", ""]

    def wr(mask):
        v = yv[mask & np.isfinite(yv)]
        return (float(np.mean(v >= WIN)) if v.size else np.nan, int(v.size),
                float(np.mean(v)) if v.size else np.nan)

    for name, pm in P.items():
        L.append(f"--- {name} ---")
        yrs_ok = 0
        for y in range(2016, 2021):
            ym = base_m & (yr == y)
            pmask = ym & pm
            b_w, b_n, _ = wr(ym)
            p_w, p_n, p_mean = wr(pmask)
            beat = p_w > b_w
            yrs_ok += int(beat)
            L.append(f"　{y}：基準 {b_w * 100:.1f}%（{b_n:,}） ｜ 頂格 {p_w * 100:.1f}%（{p_n:,}） ｜ 頂格平均超額 {p_mean * 100:+.1f}% ｜ {'高過基準' if beat else '唔高過基準'}")
            rows.append(dict(模式=name, 範圍=str(y), 基準贏家=b_w, 基準樣本=b_n, 頂格贏家=p_w, 頂格樣本=p_n, 頂格平均超額=p_mean))
        # 排除疫情期
        ex_m = base_m & ~crisis
        b_w, b_n, _ = wr(ex_m)
        p_w, p_n, p_mean = wr(ex_m & pm)
        L.append(f"　排除 2020 年 2月15日–6月30日後：基準 {b_w * 100:.1f}%（{b_n:,}） ｜ 頂格 {p_w * 100:.1f}%（{p_n:,}） ｜ 頂格平均超額 {p_mean * 100:+.1f}%")
        rows.append(dict(模式=name, 範圍="排除疫情期", 基準贏家=b_w, 基準樣本=b_n, 頂格贏家=p_w, 頂格樣本=p_n, 頂格平均超額=p_mean))
        # 只計疫情期
        cr_m = base_m & crisis
        b_w, b_n, _ = wr(cr_m)
        p_w, p_n, p_mean = wr(cr_m & pm)
        L.append(f"　只計疫情期：基準 {b_w * 100:.1f}%（{b_n:,}） ｜ 頂格 {p_w * 100:.1f}%（{p_n:,}） ｜ 頂格平均超額 {p_mean * 100:+.1f}%")
        rows.append(dict(模式=name, 範圍="只計疫情期", 基準贏家=b_w, 基準樣本=b_n, 頂格贏家=p_w, 頂格樣本=p_n, 頂格平均超額=p_mean))
        # 集中度：頂 1% 超額佔總超額
        vals = yv[base_m & pm & np.isfinite(yv)]
        if vals.size:
            tot = float(np.sum(vals[vals > 0]))
            k = max(1, int(0.01 * vals.size))
            top = float(np.sum(np.sort(vals)[-k:]))
            L.append(f"　集中度：頂 1%（{k:,} 次）嘅超額佔所有正超額嘅 {top / tot * 100:.1f}%")
        L.append(f"　逐年高過基準：{yrs_ok}/5 年")
        L.append("")
    pd.DataFrame(rows).to_csv("v58_by_year.csv", index=False, encoding="utf-8-sig")
    L.append("注意：呢個係探索嘅穩健性檢查，唔係驗證。只用 2016–2020 數據。")
    open("v58_discovery.txt", "w", encoding="utf-8").write("\n".join(L))
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
