"""
research_v59.py — 第五十九輪：用第58輪嘅 P1/P2/P3 門檻，喺 2021–2026 做樣本外檢查（事先寫定）

背景：
  第58輪嘅 P1/P2/P3 係喺 2016–2020 搵出嚟嘅（探索）。Esther 想睇吓 2020–2026 會唔會一樣。
  但 2016–2026 數據已經用喺第47–55輪，所以 2021–2026 唔係「未見過」嘅數據。
  為咗盡量減少偏差，本輪做法：
    - 門檻（分界點）全部用 2016–2020 計算，然後凍結，唔用 2021–2026 數據調整。
    - 2021–2026 只用嚟計結果。
  即係話，本輪係「凍結門檻嘅再檢查」，仍然唔係真正驗證。真正驗證要用 2026-09-30 之後嘅紙上模擬數據。

Pattern（同第58輪一樣）：
  P1：距52週高 ≤ 2016–2020 三分之一分界點 且 距VWAP250 > 2016–2020 三分之二分界點
  P2：距52週高 ≤ 2016–2020 十分之一分界點
  P3：ATR14÷收市 > 2016–2020 九成分界點（最高十分一）
  基準：同一年所有成員股票日。
  贏家 = 60 日超額回報（相對 SPY）≥ 15%。

事先寫定嘅判斷（dispatch 前寫定，唔可以之後改）：
  一個模式「通過本輪」，要同時符合：
    (a) 2021–2026 六個日曆年（2026 為至 11-19 嘅部分年）中，≥ 4 年嘅贏家比例高過同年基準；
    (b) 整段 2021–2026 嘅贏家比例高過同期基準；
    (c) 整段 2021–2026 嘅平均超額回報 > 0。
  通過只代表「值得寫成固定規則，然後用紙上模擬數據再驗證」。唔係交易建議，亦唔係驗證。
  唔通過就視為第58輪嘅結果唔延續到 2021–2026。

已知限制：
  - 2021–2026 數據已用於第47–55輪，唔係獨立樣本。
  - 2026 年數據只到 11-19，而 60 日前瞻回報需要未來數據，最近嘅信號日會被剔除。
  - 冇計交易成本，亦冇計倉位管理；呢個係信號檢查，唔係策略回測。
  - 生存者偏差：成員資格用當日 S&P 500 成員名單（同之前一樣）。

輸出：v59_by_year.csv、v59_discovery.txt
==== 結果（2026-11-19 跑完，GitHub run 44614428489；輸出 v59_discovery.txt）====
（三個模式都符合 (a)(b)(c)，判斷為「通過（只係值得做紙上模擬驗證）」）
P1（距52週高最低三分一 + 距VWAP250最高三分一）：6/6 年高過基準；2021–2026 整段 24.0% vs 10.9%，平均超額 +3.3%；頂 1% 佔 13.9%。
    但 2023、2024 年平均超額係負數（−2.0%、−2.7%），即係高過基準嘅係贏家比例，唔係平均回報。
P2（距52週高最低十分一）：6/6 年高過基準；整段 18.6% vs 10.9%，平均超額只有 +0.2%；頂 1% 佔 11.2%。
    2023、2024 年平均超額亦係負數。
P3（ATR14÷收市最高十分一）：6/6 年高過基準；整段 22.0% vs 10.9%，平均超額 +1.4%；頂 1% 佔 12.4%。
    2023、2024 年平均超額係負數。
注意：
  - 2021–2026 數據已用於第47–55輪，所以呢個唔係獨立驗證。
  - (c) 只要求平均超額 > 0，P2 嘅 +0.2% 好細，幾乎等於冇差別；判斷規則寫得偏寬鬆，呢點要記低。
  - 下一步要用 2026-09-30 之後未見過嘅紙上模擬數據，以及加入交易成本同倉位管理再測。
==== 結果完 ====
"""
import numpy as np
import pandas as pd

import momentum_v1 as M
from research_v55 import load
from research_v57 import build

M.PROGRESS = "v59_progress.txt"
log = M.log
TRAIN = (pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31"))
TEST = (pd.Timestamp("2021-01-01"), pd.Timestamp("2026-11-19"))
HORIZON = 60
WIN = 0.15


def main():
    open(M.PROGRESS, "w").close()
    D = load()
    days = D["days"]
    C = D["C"]
    mem = pd.DataFrame(D["mem"], index=days, columns=C.columns).values.astype(bool)
    F = build(D)
    spy = D["spy"]
    ex = (C.shift(-(HORIZON + 1)) / C.shift(-1) - 1).sub(spy.shift(-(HORIZON + 1)) / spy.shift(-1) - 1, axis=0).values
    fin = np.isfinite(ex)
    in_train = ((days >= TRAIN[0]) & (days <= TRAIN[1]))[:, None]
    in_test = ((days >= TEST[0]) & (days <= TEST[1]))[:, None]
    tr_base = mem & in_train & fin
    te_base = mem & in_test & fin

    def cut(name, q):
        x = F[name].values
        return float(np.quantile(x[tr_base & np.isfinite(x)], q))

    # 門檻全部由 2016–2020 計算，然後凍結
    d52_q13 = cut("距52週高", 1 / 3)
    d52_q10 = cut("距52週高", 0.10)
    vw_q23 = cut("距VWAP250", 2 / 3)
    atr_q90 = cut("ATR14÷收市", 0.90)
    x52 = F["距52週高"].values
    xvw = F["距VWAP250"].values
    xatr = F["ATR14÷收市"].values
    P = {
        "P1 距52週高最低三分一（≤%.4f）+ 距VWAP250最高三分一（>%.4f）" % (d52_q13, vw_q23):
            (x52 <= d52_q13) & (xvw > vw_q23),
        "P2 距52週高最低十分一（≤%.4f）" % d52_q10: x52 <= d52_q10,
        "P3 ATR14÷收市最高十分一（>%.4f）" % atr_q90: xatr > atr_q90,
    }
    yr = np.repeat(days.year.values[:, None], C.shape[1], axis=1)
    L = ["=== 第五十九輪：凍結門檻（2016–2020）嘅 2021–2026 樣本外檢查 ===",
         "（贏家 = 60 日超額回報 ≥ 15%；唔係驗證；2021–2026 數據已用於早前輪次）", ""]
    rows = []

    def wr(mask):
        v = ex[mask & fin]
        return (float(np.mean(v >= WIN)) if v.size else np.nan, int(v.size),
                float(np.mean(v)) if v.size else np.nan)

    for name, pm in P.items():
        L.append(f"--- {name} ---")
        yrs_ok = 0
        for y in range(2021, 2027):
            ym = te_base & (yr == y)
            b_w, b_n, _ = wr(ym)
            p_w, p_n, p_mean = wr(ym & pm)
            beat = p_n > 0 and p_w > b_w
            yrs_ok += int(beat)
            L.append(f"　{y}：基準 {b_w * 100:.1f}%（{b_n:,}） ｜ 頂格 {p_w * 100:.1f}%（{p_n:,}） ｜ 頂格平均超額 {p_mean * 100:+.1f}% ｜ {'高過基準' if beat else '唔高過基準'}")
            rows.append(dict(模式=name, 範圍=str(y), 基準贏家=b_w, 基準樣本=b_n, 頂格贏家=p_w, 頂格樣本=p_n, 頂格平均超額=p_mean))
        b_w, b_n, _ = wr(te_base)
        p_w, p_n, p_mean = wr(te_base & pm)
        overall_beat = p_n > 0 and p_w > b_w
        mean_pos = p_n > 0 and p_mean > 0
        L.append(f"　整段 2021–2026：基準 {b_w * 100:.1f}%（{b_n:,}） ｜ 頂格 {p_w * 100:.1f}%（{p_n:,}） ｜ 頂格平均超額 {p_mean * 100:+.1f}%")
        rows.append(dict(模式=name, 範圍="2021–2026", 基準贏家=b_w, 基準樣本=b_n, 頂格贏家=p_w, 頂格樣本=p_n, 頂格平均超額=p_mean))
        vals = ex[te_base & pm & fin]
        if vals.size:
            pos = vals[vals > 0]
            k = max(1, int(0.01 * vals.size))
            top = float(np.sum(np.sort(vals)[-k:]))
            L.append(f"　集中度：頂 1%（{k:,} 次）嘅超額佔所有正超額嘅 {top / float(np.sum(pos)) * 100:.1f}%")
        passed = (yrs_ok >= 4) and overall_beat and mean_pos
        L.append(f"　逐年高過基準：{yrs_ok}/6 年（(a) 要求 ≥4）｜ (b) 整段高過基準：{'是' if overall_beat else '否'} ｜ (c) 平均超額 > 0：{'是' if mean_pos else '否'}")
        L.append(f"　本輪判斷：{'通過（只係值得做紙上模擬驗證）' if passed else '未通過'}")
        L.append("")
    pd.DataFrame(rows).to_csv("v59_by_year.csv", index=False, encoding="utf-8-sig")
    L.append("注意：凍結門檻嘅樣本外檢查，數據已用於早前輪次，唔係驗證。")
    open("v59_discovery.txt", "w", encoding="utf-8").write("\n".join(L))
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
