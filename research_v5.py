"""
research_v5.py — 第五輪研究：波幅控制、風險管理動量、剩餘動量、行業→個股動量

⚠️ 預先登記（跑之前定好，跑完唔改）：

共同設定
- 股票池：每月用當時嘅標普500成分股（fja05680/sp500，同momentum_v1.py一樣），價格由yfinance逐隻下載（已計股息）。
- 行業：用今日嘅GICS行業（Wikipedia現有成分股；其餘用yfinance）。⚠️ 公司轉行業嘅情況冇計。
- 每月最後一個交易日收市決定，下一個交易日收市交易，再下一日先計回報。
- 成本：每次買賣0.1%（按權重變化計）。現金：13週國債息。槓桿借錢成本：國債息 + 1.5%。
- 揀選期 2010-01至2017-12；驗證期 2018-01至2026-09。另外列出2020-2026（數據覆蓋最好）。
- 合格：驗證期回報高過SPY長揸，而且驗證期最大跌幅好過 -35%。
- 每個測試入面，用揀選期（最大跌幅好過-35%）回報最高嘅設定，原封不動睇驗證期。
- ⚠️ 用唔用槓桿分開列；用戶本身規則係唔用槓桿。

Test A：而家嘅SPY訊號方案 + 波幅控制
  持有時嘅倉位 = min(上限, 目標波幅 ÷ SPY過去60日實際波幅)
  目標波幅 12%、15%、18%、20%；上限 1.0倍（唔用槓桿）、1.25倍、1.5倍 → 12個 + 原本方案

Test B：風險管理動量（12-1動量前10隻，唔加200日線濾網）
  每隻權重 ∝ 1 ÷ 過去60日波幅；單一股票最多15%；單一行業最多30%（超出部分放現金）
  再做組合波幅控制：目標 12%、15%、18%；上限 1.0倍、1.5倍 → 6個 + 未做波幅控制

Test C：剩餘動量（12-1期間），前10隻、前20隻，平均分配
  原始：股票回報
  扣大市：股票回報 − β(對SPY) × SPY回報
  扣行業：股票回報 − 同行業平均回報
  扣行業β：股票回報 − β(對行業) × 行業平均回報
  → 8個

Test D：行業 → 個股
  行業按「成分股12-1動量平均」排名；揀頭3個行業每個最強2隻（6隻），或者頭4個行業每個最強3隻（12隻）
  平均分配 → 2個（冇質素篩選：冇免費嘅「當時」財務數據）

輸出：v5_results.csv、v5_progress.txt
"""
import io
import time

import numpy as np
import pandas as pd
import requests

import momentum_v1 as M

M.PROGRESS = "v5_progress.txt"
log = M.log
COST = 0.001
BORROW_SPREAD = 0.015
PICK = ("2010-01-01", "2017-12-31")
TEST = ("2018-01-01", "2026-09-30")
RECENT = ("2020-01-01", "2026-09-30")
FULL = ("2010-01-01", "2026-09-30")
PERIODS = [("全期2010-2026", *FULL), ("揀選期2010-2017", *PICK), ("驗證期2018-2026", *TEST), ("近期2020-2026", *RECENT)]


# ---------------- 行業 ----------------
def sectors_for(tickers):
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=M.HDR, timeout=30).text
    t0 = pd.read_html(io.StringIO(html))[0]
    sec = {M.norm(s): g for s, g in zip(t0["Symbol"], t0["GICS Sector"])}
    import yfinance as yf
    yf_map = {"Technology": "Information Technology", "Healthcare": "Health Care",
              "Financial Services": "Financials", "Consumer Cyclical": "Consumer Discretionary",
              "Consumer Defensive": "Consumer Staples", "Basic Materials": "Materials",
              "Communication Services": "Communication Services", "Industrials": "Industrials",
              "Energy": "Energy", "Utilities": "Utilities", "Real Estate": "Real Estate"}
    missing = [t for t in tickers if t not in sec]
    for t in missing:
        try:
            s = yf.Ticker(t).info.get("sector")
            if s:
                sec[t] = yf_map.get(s, s)
        except Exception:
            pass
        time.sleep(0.3)
    log(f"行業：{sum(1 for t in tickers if t in sec)}/{len(tickers)}隻有行業")
    return sec


# ---------------- 引擎 ----------------
def run_weights(rets, targets, cash_r, exposure=None):
    """targets：{決定日: pd.Series(權重)}；exposure：每日倍數（已經係「決定日」嘅數，入面會遲兩日生效）。
    每日按目標權重 × 倍數持有（簡化：唔計月中浮動）。回傳每日回報。"""
    days = rets.index
    W = pd.DataFrame(0.0, index=days, columns=rets.columns)
    rows = {}
    for d, w in targets.items():
        i = days.searchsorted(d, side="right") - 1
        if i + 2 < len(days):
            rows[days[i + 2]] = w
    filled = pd.DataFrame(rows).T.reindex(columns=rets.columns).fillna(0.0)
    W.loc[filled.index] = filled.values
    mask = pd.Series(False, index=days)
    mask[filled.index] = True
    W = W.where(mask, np.nan).ffill().fillna(0.0)
    e = pd.Series(1.0, index=days) if exposure is None else exposure.shift(2).reindex(days).fillna(0.0)
    unscaled = (W * rets.fillna(0.0)).sum(axis=1)
    invested = W.sum(axis=1)
    gross = e * invested
    port = e * unscaled + (1 - gross).clip(lower=0) * cash_r - (gross - 1).clip(lower=0) * (cash_r + BORROW_SPREAD / 252)
    eff = W.mul(e, axis=0)
    turnover = eff.diff().abs().sum(axis=1).fillna(0.0)
    return port - turnover * COST


def stats(daily, a, b):
    d = daily[(daily.index >= pd.Timestamp(a)) & (daily.index <= pd.Timestamp(b))]
    eq = (1 + d).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    v = np.concatenate([[1.0], eq.values])
    vol = d.std() * np.sqrt(252) * 100
    return {"cagr": (eq.iloc[-1] ** (1 / yrs) - 1) * 100, "mdd": (v / np.maximum.accumulate(v) - 1).min() * 100,
            "vol": vol, "mult": eq.iloc[-1]}


def vol_target(unscaled, target, cap, extra=None):
    vol = unscaled.rolling(60).std() * np.sqrt(252)
    e = (target / vol).clip(upper=cap).fillna(0.0)
    return e if extra is None else e * extra


def capped_inverse_vol(names, vols, sec, single=0.15, sector=0.30):
    """權重 ∝ 1/波幅；單一股票 ≤ single、單一行業 ≤ sector；分唔晒嘅放現金。"""
    raw = pd.Series({t: 1 / vols[t] for t in names if np.isfinite(vols.get(t, np.nan)) and vols[t] > 0})
    if raw.empty:
        return raw
    w = raw / raw.sum()
    g = pd.Series({t: sec.get(t, "未知") for t in w.index})
    for _ in range(100):
        w = w.clip(upper=single)
        for grp, idx in g.groupby(g).groups.items():
            tot = w[idx].sum()
            if tot > sector:
                w[idx] *= sector / tot
        free = 1 - w.sum()
        sec_sum = w.groupby(g).transform("sum")
        ok = (w < single - 1e-9) & (sec_sum < sector - 1e-9)
        if free < 1e-6 or not ok.any():
            break
        w[ok] += free * raw[ok] / raw[ok].sum()
    return w


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    month_ends = pd.date_range("2008-12-31", "2026-09-30", freq="ME")
    members = {d: members_fn(d) for d in month_ends}
    everyone = sorted(set().union(*members.values()))
    log(f"2009-2026曾經係成分股：{len(everyone)}隻")
    closes = M.download_close(everyone + ["SPY"], "2007-06-01", "2026-10-01")
    spy = closes.pop("SPY")
    px = pd.DataFrame(closes).reindex(spy.index)
    px, spy = px[px.index <= pd.Timestamp(FULL[1])], spy[spy.index <= pd.Timestamp(FULL[1])]
    rets = px.pct_change(fill_method=None)
    spy_r = spy.pct_change().fillna(0.0)
    cash_r = M.get_cash(px.index)
    sec = sectors_for(sorted(px.columns))

    cov = []
    for d in month_ends[12::12]:
        m = members[d]
        have = 0
        for t in m:
            if t in px.columns:
                x = px[t].loc[:d].dropna()
                have += int(len(x) > 0 and x.index[-1] >= d - pd.Timedelta(days=10))
        cov.append(f"{d.year}：{have / max(len(m), 1) * 100:.0f}%")
    log("數據覆蓋率：" + "；".join(cov))

    days = px.index
    vol60 = rets.rolling(60).std() * np.sqrt(252)
    results = {}   # 名 -> (測試, 有冇槓桿, 每日回報)

    # ---------- Test A ----------
    try:
        import main as bot
        df = bot.replay(bot.add_signals(bot.load_data()))
        state = (df["state"] != "out").astype(float).reindex(days).fillna(0.0)
        hold_spy = {days[0]: pd.Series({"SPY": 1.0})}
        spy_df = pd.DataFrame({"SPY": spy_r})
        base_e = state
        results["A0 原本SPY訊號方案"] = ("A", False, run_weights(spy_df, hold_spy, cash_r, base_e))
        spy_vol = spy_r.rolling(60).std() * np.sqrt(252)
        for tv in (0.12, 0.15, 0.18, 0.20):
            for cap in (1.0, 1.25, 1.5):
                e = (tv / spy_vol).clip(upper=cap).fillna(0.0) * state
                results[f"A 目標波幅{tv:.0%} 上限{cap}倍"] = ("A", cap > 1, run_weights(spy_df, hold_spy, cash_r, e))
        log("Test A 完成")
    except Exception as ex:
        log(f"Test A 出錯：{ex}")

    # ---------- 每月計分 ----------
    LOOK, SKIP = 252, 21
    tg = {k: {} for k in ["B", "C原始10", "C原始20", "C扣大市10", "C扣大市20", "C扣行業10", "C扣行業20",
                          "C扣行業β10", "C扣行業β20", "D3x2", "D4x3"]}
    for d in month_ends:
        i = days.searchsorted(d, side="right") - 1
        if i < LOOK + 5:
            continue
        day = days[i]
        alive = [t for t in members[d] if t in px.columns and np.isfinite(px[t].iloc[i])
                 and np.isfinite(px[t].iloc[i - LOOK])]
        if len(alive) < 50:
            continue
        R = (px[alive].iloc[i - SKIP] / px[alive].iloc[i - LOOK] - 1)
        Rm = spy.iloc[i - SKIP] / spy.iloc[i - LOOK] - 1
        win = rets[alive].iloc[i - LOOK + 1:i + 1]
        sw = spy_r.iloc[i - LOOK + 1:i + 1]
        beta_m = win.apply(lambda c: c.cov(sw) / sw.var())
        secs = pd.Series({t: sec.get(t, "未知") for t in alive})
        sec_R = R.groupby(secs).transform("mean")
        sec_daily = win.T.groupby(secs).transform("mean").T
        beta_s = pd.Series({t: win[t].cov(sec_daily[t]) / sec_daily[t].var() if sec_daily[t].var() > 0 else 1.0
                            for t in alive})
        scores = {"原始": R, "扣大市": R - beta_m * Rm, "扣行業": R - sec_R, "扣行業β": R - beta_s * sec_R}
        for k, sc in scores.items():
            sc = sc.dropna().sort_values(ascending=False)
            for n in (10, 20):
                top = sc.index[:n]
                tg[f"C{k}{n}"][day] = pd.Series(1 / len(top), index=top)
        top10 = R.dropna().sort_values(ascending=False).index[:10]
        tg["B"][day] = capped_inverse_vol(top10, vol60.iloc[i], sec)
        sec_rank = R.groupby(secs).mean().drop("未知", errors="ignore").sort_values(ascending=False)
        for key, (ns, nk) in {"D3x2": (3, 2), "D4x3": (4, 3)}.items():
            pick = []
            for g in sec_rank.index[:ns]:
                pick += list(R[secs == g].dropna().sort_values(ascending=False).index[:nk])
            tg[key][day] = pd.Series(1 / len(pick), index=pick)
    log("每月計分完成")

    # ---------- Test B ----------
    unB = run_weights(rets, tg["B"], cash_r * 0)     # 未縮放（冇現金）用嚟估波幅
    results["B 風險管理動量（冇波幅控制）"] = ("B", False, run_weights(rets, tg["B"], cash_r))
    for tv in (0.12, 0.15, 0.18):
        for cap in (1.0, 1.5):
            e = vol_target(unB, tv, cap)
            results[f"B 目標波幅{tv:.0%} 上限{cap}倍"] = ("B", cap > 1, run_weights(rets, tg["B"], cash_r, e))
    log("Test B 完成")
    for k in [k for k in tg if k.startswith("C") or k.startswith("D")]:
        results[k] = (k[0], False, run_weights(rets, tg[k], cash_r))
    log("Test C、D 完成")

    # ---------- 報告 ----------
    L = ["=== 第五輪研究（規則預先登記，見research_v5.py開頭）===",
         f"股票池：當時標普500成分股（{source}）；數據覆蓋率：" + "；".join(cov),
         "揀選期2010-2017；驗證期2018-2026/09；合格：驗證期回報 > SPY，而且驗證期最大跌幅好過-35%", ""]
    spy_d = spy_r.copy()
    spy_s = {pn: stats(spy_d, a, b) for pn, a, b in PERIODS}
    L.append("SPY長揸：" + " | ".join(f"{pn} {s['cagr']:+.2f}%（跌 {s['mdd']:.1f}%，波幅 {s['vol']:.0f}%）" for pn, s in spy_s.items()))
    L.append("")
    table = {}
    for name, (test, lev, daily) in results.items():
        table[name] = (test, lev, {pn: stats(daily, a, b) for pn, a, b in PERIODS})
    for test in ["A", "B", "C", "D"]:
        L.append(f"=== Test {test} ===")
        for name, (t, lev, s) in table.items():
            if t != test:
                continue
            ok = s["驗證期2018-2026"]["cagr"] > spy_s["驗證期2018-2026"]["cagr"] and s["驗證期2018-2026"]["mdd"] > -35
            L.append(f"　{'【槓桿】' if lev else ''}{name}：" + " | ".join(
                f"{pn} {x['cagr']:+.2f}%（跌 {x['mdd']:.1f}%）" for pn, x in s.items())
                + f" | $10,000(2010起)→${10000 * s['全期2010-2026']['mult']:,.0f} | {'✅合格' if ok else '❌'}")
        for lev_ok in (False, True):
            cand = [(n, s) for n, (t, lev, s) in table.items() if t == test and lev == lev_ok
                    and s["揀選期2010-2017"]["mdd"] > -35]
            if not cand:
                continue
            n, s = max(cand, key=lambda x: x[1]["揀選期2010-2017"]["cagr"])
            group = [x for x, (t, lev, _) in table.items() if t == test]
            rank = sum(1 for x in group if table[x][2]["驗證期2018-2026"]["cagr"] > s["驗證期2018-2026"]["cagr"]) + 1
            ok = s["驗證期2018-2026"]["cagr"] > spy_s["驗證期2018-2026"]["cagr"] and s["驗證期2018-2026"]["mdd"] > -35
            L.append(f"　→ 誠實測試（{'用槓桿' if lev_ok else '唔用槓桿'}）：揀選期揀中「{n}」（{s['揀選期2010-2017']['cagr']:+.2f}%）"
                     f"→ 驗證期 {s['驗證期2018-2026']['cagr']:+.2f}%（跌 {s['驗證期2018-2026']['mdd']:.1f}%），"
                     f"排{rank}/{len(group)}；SPY驗證期 {spy_s['驗證期2018-2026']['cagr']:+.2f}% → {'✅合格' if ok else '❌唔合格'}")
        L.append("")
    best = sorted(table.items(), key=lambda kv: -kv[1][2]["驗證期2018-2026"]["cagr"])[:5]
    L.append("=== 驗證期回報最高5個（⚠️事後睇，只作參考）===")
    for n, (t, lev, s) in best:
        L.append(f"　{'【槓桿】' if lev else ''}{n}：驗證期 {s['驗證期2018-2026']['cagr']:+.2f}%（跌 {s['驗證期2018-2026']['mdd']:.1f}%）")
    pd.DataFrame({"report": L}).to_csv("v5_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
