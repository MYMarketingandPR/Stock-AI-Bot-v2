"""
research_v14.py — 第十四輪：「公司資料」類策略（盈利驚喜、分析師上調）+ 質素ETF代替測試

為咩：第一至十三輪全部係「睇價錢」嘅方法（突破、指標、動量、止損、換倉日…），已經證明好難穩定贏QQQ。
今輪改測「市場係咪要用幾個月先完全消化新資訊」。

數據（probe_info.py 2026-10-05 查過）：
  ✔ 盈利公布紀錄（Yahoo）：預期EPS、實際EPS，大約2002年起；但已除牌公司冇數據（同價錢一樣有倖存者偏差）
  ✔ 分析師評級升降 + 目標價調整（Yahoo）：2012年起有，但2018年前好疏 → 下面會用覆蓋率決定測試由邊年開始
  ✘ 分析師EPS預測變化：只有今日，冇歷史 → 唔測；改為由今個月起每月自己記低（snapshot_estimates.py）
  ✘ 公司質素（ROE、現金流、負債）：SEC資料庫封鎖雲端伺服器 → 冇歷史；改用「質素ETF」做代替測試
  ✘ 羅素1000/3000歷史成分股：iShares網站攞唔到 → 股票池仍然係「當時標普500成分股」

⚠️ 預先登記（2026-10-05 跑之前寫好；跑完只可以喺下面加結果，唔改規則、唔改標準）：

共同設定：當時標普500成分股；已計股息；收市決定→下日收市交易（第2日起計回報）；每次買賣0.1%；唔用槓桿；
  100%股票（冇SPY訊號部分，方便直接同QQQ比）；每半個月換倉（15號或之前最後交易日 + 月尾）；前10隻平均分配。

「知道消息」嘅日子（避免偷睇未來）：
  公布時間 16:00 或之後、或者冇時間（00:00）→ 當作收市後公布 → 下一個交易日收市先算知道；
  公布時間 16:00 之前 → 當日收市算知道。之後再下一個收市先買入（即係唔食公布嗰下跳空，只測「之後慢慢升」）。

盈利驚喜（標準化）= (實際EPS − 預期EPS) ÷ 公布前一日股價。
  （唔用Yahoo嘅「驚喜%」，因為預期EPS好細嘅公司會出現1000%呢類誇張數字。）
  只用最近63個交易日內公布嘅驚喜（太舊當過期）。

分析師淨上調 = 過去63個交易日：(升級 + 加目標價 − 降級 − 減目標價) ÷ 總次數；最少要有3次動作先計。
  測試期由「當年標普500成分股有 ≥80% 最少有1次分析師動作」嗰年開始。

候選（只有呢5個，冇其他參數）：
  E1 = 盈利驚喜最高10隻
  E2 = 12-1動量最高30隻入面，再揀盈利驚喜最高10隻
  E3 = 動量排名同盈利驚喜排名平均，最高10隻
  R1 = 分析師淨上調最高10隻
  R2 = 12-1動量最高30隻入面，再揀分析師淨上調最高10隻
  參考：M0 = 純12-1動量前10隻（100%股票，同樣半月換）；QQQ、SPY、MTUM 長揸
  質素ETF代替測試：SPHQ（標普500質素，2005起）、QUAL（MSCI美國質素，2013起）、
    QUAL+MTUM 各50%（每月尾重新平衡）、SPHQ+QQQ 各50%（每月尾重新平衡）

滾動揀選（walk-forward）：
  盈利組：揀規則期 2003-2014 用評分（40%回報+30%最大跌幅+20%Sharpe+10%換手，只比最大跌幅好過-35%）喺E1-E3揀1個，
    驗證期 2015-2026 先判斷。
  分析師組：開始年至2026年對半切，前半揀R1/R2，後半判斷。

合格標準（全部要符合，同QQQ同一段時間比）：
  ① 驗證期年化回報 ≥ QQQ + 2個百分點
  ② 驗證期 Sharpe ≥ QQQ
  ③ 驗證期最大跌幅好過 -35%（Esther嘅底線；建議文件寫-40%，用較嚴嗰個）
  ④ 跌市測試：盈利組 2007-07至2009-06 嘅最大跌幅唔可以差過QQQ同期（分析師組數據太短，冇呢項）
  唔合格 → 記入避坑清單，唔再調。

另外（純描述，唔用嚟揀策略）：
  事件研究：每次公布按標準化驚喜分5組，睇買入後 1個月(21日)/3個月(63日)/6個月(126日) 平均跑贏SPY幾多
  → 直接答「市場係咪要幾個月先消化盈利驚喜」。

輸出：v14_results.csv、v14_progress.txt
==== 結果（2026-10-05 第一次跑，記錄低，唔再改）====
- 數據：盈利紀錄 660/689 隻、53,699 次公布；換倉日覆蓋率 2003-09 82%、2010-19 91%、2020-26 97%。
  分析師紀錄 656/689 隻；「每年最少1次動作」覆蓋率2012年已經87%，所以由2012開始（但2018年前每隻紀錄好疏，揀規則期要打折扣）。
- 盈利組：揀規則期揀咗E2（動量前30→驚喜前10）。驗證期2015-2026 +16.44%（QQQ +19.13%），Sharpe 0.64（QQQ 0.82），
  最大跌幅 -42.5%，金融海嘯 -55.1%（QQQ -53.4%）→ 四項全部唔合格。E1、E3 更差（E1 最大跌幅 -78.5%）。
- 分析師組：揀咗R1。驗證期2019-2026 +16.29%（QQQ +21.63%），Sharpe 0.70（0.82），最大跌幅 -36.8% → 唔合格。
- 質素ETF：QUAL +13.6%、QUAL+MTUM各半 +14.7%（同期QQQ +19.9%）；SPHQ 2005起 +9.8%（QQQ +15.7%）→ 全部輸QQQ。
- 事件研究：盈利驚喜最好組 vs 最差組，6個月差距 2003-14 +0.9%、2015-26 +2.9%；但最好組本身只係比SPY多 +0.8%至+3.2%，
  差距主要嚟自「驚喜差嘅股票之後繼續跑輸」。→ 市場大致即時消化咗盈利驚喜，剩低嘅「慢慢升」太細，蓋唔過QQQ。
- 參考：純動量前10（100%股票）驗證期 +18.67%（跌-40.1%），金融海嘯 -69.1%。
"""
import bisect
import time

import numpy as np
import pandas as pd

import data_cache as DC
import momentum_v1 as M
from perf import perf
from research_v6 import effective_weights, LOOK, SKIP
from research_v8 import decision_days
from research_v10 import sleeve, metrics, score_pick

M.PROGRESS = "v14_progress.txt"
log = M.log
COST = 0.001
END = "2026-09-30"
FRESH = 63


def known_index(days, ts):
    """公布時間 → 「知道消息」嗰日喺 days 入面嘅位置。"""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    d = ts.normalize()
    after_close = ts.hour >= 16 or (ts.hour == 0 and ts.minute == 0)
    i = days.searchsorted(d, side="left")          # 當日或之後第一個交易日
    if i < len(days) and days[i] == d and after_close:
        i += 1
    return i if i < len(days) else None


def fetch(tickers, what):
    import yfinance as yf
    out, fails = {}, 0
    for n, t in enumerate(tickers):
        for attempt in range(3):
            try:
                df = DC.earnings(t) if what == "earn" else DC.upgrades(t)
                if df is not None and len(df):
                    out[t] = df
                break
            except Exception as e:
                msg = str(e).lower()
                if not any(w in msg for w in ("rate", "too many", "timed out", "timeout", "connection", "429")):
                    break                      # 冇數據 / 已除牌：唔使再試
                time.sleep(4 * (attempt + 1))
        if t not in out:
            fails += 1
        time.sleep(0.4)
        if (n + 1) % 100 == 0:
            log(f"{'盈利紀錄' if what == 'earn' else '分析師紀錄'} 下載進度 {n + 1}/{len(tickers)}，有數據 {len(out)}")
    log(f"{'盈利紀錄' if what == 'earn' else '分析師紀錄'} 完成：{len(out)}/{len(tickers)}")
    return out


def main():
    open(M.PROGRESS, "w").close()
    members_fn, source = M.sp500_history()
    month_ends = pd.date_range("2001-12-31", END, freq="ME")
    everyone = sorted(set().union(*[members_fn(d) for d in month_ends]))
    log(f"成分股來源：{source}；2002-2026 曾經入過標普500：{len(everyone)} 隻")
    etfs = ["SPY", "QQQ", "MTUM", "SPHQ", "QUAL"]
    closes = M.download_close(everyone + etfs, "2001-01-01", "2026-10-01")
    E = {k: closes.pop(k, None) for k in etfs}
    spy = E["SPY"][E["SPY"].index <= pd.Timestamp(END)]
    px = pd.DataFrame(closes).reindex(spy.index)
    days = px.index
    rets = px.pct_change(fill_method=None)
    er = {k: (v.reindex(days).pct_change(fill_method=None) if v is not None else None) for k, v in E.items()}
    spy_r = er["SPY"].fillna(0.0)
    qqq_r = er["QQQ"].fillna(0.0)
    cash_r = M.get_cash(days)
    have = [t for t in everyone if t in px.columns]

    earn_raw = fetch(have, "earn")
    ana_raw = fetch(have, "ana")

    # ---------- 整理盈利事件 ----------
    events = []          # (ticker, k, surprise)
    ev_by = {}
    for t, df in earn_raw.items():
        df = df.copy()
        df = df[df["EPS Estimate"].notna() & df["Reported EPS"].notna()]
        df = df[~df.index.duplicated(keep="first")]
        col = px[t]
        ks, vs = [], []
        for ts, row in df.sort_index().iterrows():
            k = known_index(days, ts)
            if k is None or k < 1:
                continue
            ann_day = pd.Timestamp(ts).tz_convert("America/New_York").tz_localize(None).normalize() \
                if pd.Timestamp(ts).tzinfo is not None else pd.Timestamp(ts).normalize()
            j = days.searchsorted(ann_day, side="left") - 1     # 公布前一個交易日
            if j < 0:
                continue
            p = col.iat[j]
            if not np.isfinite(p) or p <= 0:
                continue
            s = (row["Reported EPS"] - row["EPS Estimate"]) / p
            ks.append(k)
            vs.append(float(s))
            events.append((t, k, float(s)))
        if ks:
            order = np.argsort(ks, kind="stable")
            ev_by[t] = (list(np.array(ks)[order]), list(np.array(vs)[order]))
    log(f"盈利事件：{len(events)} 次，{len(ev_by)} 隻股票")

    # ---------- 整理分析師事件 ----------
    an_by = {}
    year_has = {}
    for t, df in ana_raw.items():
        df = df.sort_index()
        sc = []
        for ts, row in df.iterrows():
            k = known_index(days, ts)
            if k is None:
                continue
            a = str(row.get("Action", "")).lower()
            pt = str(row.get("priceTargetAction", "")).lower()
            pos = (a == "up") + (pt.startswith("raise"))
            neg = (a == "down") + (pt.startswith("lower"))
            sc.append((k, pos - neg))
            year_has.setdefault(days[k].year, set()).add(t)
        if sc:
            ks = [x[0] for x in sc]
            cum_s = np.cumsum([x[1] for x in sc])
            an_by[t] = (ks, list(cum_s))

    R121 = px.shift(SKIP) / px.shift(LOOK) - 1
    mem_cache = {}

    def members(d):
        k = (d.year, d.month)
        if k not in mem_cache:
            mem_cache[k] = [t for t in members_fn(d) if t in px.columns]
        return mem_cache[k]

    def surprise_at(t, i):
        if t not in ev_by:
            return np.nan
        ks, vs = ev_by[t]
        p = bisect.bisect_right(ks, i) - 1
        if p < 0 or i - ks[p] > FRESH:
            return np.nan
        return vs[p]

    def revision_at(t, i):
        if t not in an_by:
            return np.nan
        ks, cs = an_by[t]
        hi = bisect.bisect_right(ks, i)
        lo = bisect.bisect_right(ks, i - FRESH)
        n = hi - lo
        if n < 3:
            return np.nan
        net = cs[hi - 1] - (cs[lo - 1] if lo > 0 else 0)
        return net / n + n * 1e-6        # 同分時動作多啲排前

    half = decision_days(days, "每半個月")
    cover = {"earn": {}, "ana": {}}

    def build(kind, start):
        tg = {}
        for d in half:
            if d < pd.Timestamp(start):
                continue
            i = days.get_loc(d)
            if i < LOOK + 5:
                continue
            m = members(d)
            ok = [t for t in m if np.isfinite(px[t].iat[i]) and np.isfinite(px[t].iat[i - LOOK])]
            if not ok:
                continue
            mom = R121.iloc[i][ok].dropna()
            if kind == "M0":
                top = list(mom.sort_values(ascending=False).index[:10])
            elif kind in ("E1", "E2", "E3"):
                sup = pd.Series({t: surprise_at(t, i) for t in ok}).dropna()
                cover["earn"][d] = len(sup) / max(len(m), 1)
                if kind == "E1":
                    top = list(sup.sort_values(ascending=False).index[:10])
                elif kind == "E2":
                    pool = [t for t in mom.sort_values(ascending=False).index[:30] if t in sup.index]
                    top = list(sup[pool].sort_values(ascending=False).index[:10])
                else:
                    both = pd.concat([mom.rank(pct=True), sup.rank(pct=True)], axis=1).dropna().mean(axis=1)
                    top = list(both.sort_values(ascending=False).index[:10])
            else:
                rev = pd.Series({t: revision_at(t, i) for t in ok}).dropna()
                cover["ana"][d] = len(rev) / max(len(m), 1)
                if kind == "R1":
                    top = list(rev.sort_values(ascending=False).index[:10])
                else:
                    pool = [t for t in mom.sort_values(ascending=False).index[:30] if t in rev.index]
                    top = list(rev[pool].sort_values(ascending=False).index[:10])
            if top:
                tg[d] = pd.Series(1.0 / len(top), index=top)
        W, _ = effective_weights(rets, tg)
        return sleeve(W, rets, cash_r)

    # 分析師測試開始年：當年成分股 ≥80% 有最少1次動作
    ana_start = None
    cov_lines = []
    for y in range(2012, 2027):
        m = members(pd.Timestamp(f"{y}-06-30"))
        c = len(year_has.get(y, set()) & set(m)) / max(len(m), 1)
        cov_lines.append(f"{y}:{c * 100:.0f}%")
        if ana_start is None and c >= 0.8:
            ana_start = y
    log(f"分析師覆蓋率：{' '.join(cov_lines)}；開始年 {ana_start}")

    L = ["=== 第十四輪：公司資料類策略（規則預先登記，見research_v14.py開頭）===",
         f"股票池：當時標普500成分股（{source}）；已計股息；每次買賣0.1%；唔用槓桿；100%股票；每半個月換",
         f"盈利紀錄：{len(earn_raw)}/{len(have)} 隻有數據；可用事件 {len(events)} 次",
         f"分析師紀錄：{len(ana_raw)}/{len(have)} 隻有數據；每年覆蓋率 {' '.join(cov_lines)}；測試開始年 {ana_start}", ""]

    res = {}
    for kind in ["M0", "E1", "E2", "E3"]:
        res[kind] = build(kind, "2002-12-01")
        log(f"{kind} 完成")
    if ana_start:
        for kind in ["R1", "R2"]:
            res[kind] = build(kind, f"{ana_start - 1}-12-01")
            log(f"{kind} 完成")

    ce = pd.Series(cover["earn"])
    if len(ce):
        L.append("盈利驚喜覆蓋率（換倉日有新鮮驚喜數據嘅成分股比例）：" + "，".join(
            f"{a[:4]}-{b[:4]} {ce.loc[a:b].mean() * 100:.0f}%" for a, b in
            [("2003-01-01", "2009-12-31"), ("2010-01-01", "2019-12-31"), ("2020-01-01", END)] if len(ce.loc[a:b])))
    L.append("")

    def show(name, r, t, a, b):
        p = perf(r.loc[a:b].dropna(), rf=cash_r, bench=qqq_r.loc[a:b], turn=t, cap_freq="ME")
        return p, (f"　{name}：年化 {p['cagr']:+.2f}% ｜ 最大跌幅 {p['mdd']:.1f}% ｜ Sharpe {p['sharpe']:.2f} ｜ "
                   f"Calmar {p['calmar']:.2f} ｜ 每年換手 {p['turn_year']:.0f}% ｜ "
                   f"對QQQ 升市捕捉 {p['up_cap']:.0f}% / 跌市捕捉 {p['down_cap']:.0f}%")

    zero = pd.Series(0.0, index=days)
    names = {"M0": "M0 純動量前10", "E1": "E1 盈利驚喜前10", "E2": "E2 動量前30→驚喜前10", "E3": "E3 動量+驚喜平均",
             "R1": "R1 分析師上調前10", "R2": "R2 動量前30→分析師上調前10"}

    def block(title, a, b, kinds):
        L.append(f"=== {title}（{a[:10]} 至 {b[:10]}）===")
        out = {}
        for nm, r in (("QQQ長揸", qqq_r), ("SPY長揸", spy_r)):
            out[nm], txt = show(nm, r, zero, a, b)
            L.append(txt)
        if er["MTUM"] is not None and er["MTUM"].loc[a:b].notna().sum() > 250:
            _, txt = show("MTUM長揸", er["MTUM"].loc[a:b].dropna(), zero, a, b)
            L.append(txt + "（MTUM 2013年中先有）")
        for k in kinds:
            if k in res:
                r, t = res[k]
                out[k], txt = show(names[k], r, t, a, b)
                L.append(txt)
        L.append("")
        return out

    # ---------- 盈利組 ----------
    pickE = block("盈利組 揀規則期", "2003-01-01", "2014-12-31", ["M0", "E1", "E2", "E3"])
    valE = block("盈利組 驗證期", "2015-01-01", END, ["M0", "E1", "E2", "E3"])
    block("盈利組 2020-2026", "2020-01-01", END, ["M0", "E1", "E2", "E3"])
    crashE = block("跌市測試 金融海嘯", "2007-07-01", "2009-06-30", ["M0", "E1", "E2", "E3"])
    block("全期", "2003-01-01", END, ["M0", "E1", "E2", "E3"])

    def judge(group, pick_tab, val_tab, crash_tab, a, b):
        cands = {k: metrics(res[k][0], res[k][1], a, b) for k in group if k in res}
        best, sc = score_pick(cands)
        L.append(f"　揀規則期評分：" + "，".join(f"{k} {v:.2f}" for k, v in sc.items()) + f" → 揀 {best}")
        v, q = val_tab[best], val_tab["QQQ長揸"]
        c1 = v["cagr"] >= q["cagr"] + 2
        c2 = v["sharpe"] >= q["sharpe"]
        c3 = v["mdd"] > -35
        L.append(f"　① 年化 {v['cagr']:+.2f}% vs QQQ {q['cagr']:+.2f}%+2 → {'✔' if c1 else '✘'}")
        L.append(f"　② Sharpe {v['sharpe']:.2f} vs QQQ {q['sharpe']:.2f} → {'✔' if c2 else '✘'}")
        L.append(f"　③ 最大跌幅 {v['mdd']:.1f}%（要好過-35%）→ {'✔' if c3 else '✘'}")
        ok = c1 and c2 and c3
        if crash_tab is not None:
            c4 = crash_tab[best]["mdd"] >= crash_tab["QQQ長揸"]["mdd"]
            L.append(f"　④ 金融海嘯最大跌幅 {crash_tab[best]['mdd']:.1f}% vs QQQ {crash_tab['QQQ長揸']['mdd']:.1f}% → {'✔' if c4 else '✘'}")
            ok = ok and c4
        L.append(f"　⇒ {best} {'合格' if ok else '唔合格'}")
        L.append("")

    L.append("=== 盈利組 預先登記判斷 ===")
    judge(["E1", "E2", "E3"], pickE, valE, crashE, "2003-01-01", "2014-12-31")

    # ---------- 分析師組 ----------
    if ana_start:
        s0 = pd.Timestamp(f"{ana_start}-01-01")
        mid = s0 + (pd.Timestamp(END) - s0) / 2
        a1, b1 = s0.strftime("%Y-%m-%d"), mid.strftime("%Y-%m-%d")
        a2 = (mid + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        pickR = block("分析師組 揀規則期", a1, b1, ["M0", "R1", "R2"])
        valR = block("分析師組 驗證期", a2, END, ["M0", "R1", "R2"])
        L.append("=== 分析師組 預先登記判斷 ===")
        judge(["R1", "R2"], pickR, valR, None, a1, b1)
    else:
        L.append("=== 分析師組：冇一年覆蓋率達80%，唔測 ===")
        L.append("")

    # ---------- 質素ETF代替測試 ----------
    L.append("=== 質素ETF代替測試（每月尾重新平衡；ETF管理費已反映喺價格）===")
    month_end = set(decision_days(days, "每月"))

    def blend(x, y):
        a_r, b_r = er[x], er[y]
        ok = a_r.notna() & b_r.notna()
        wa, wb, out = 0.5, 0.5, []
        for d in days:
            if not ok[d]:
                out.append(np.nan)
                continue
            ra, rb = a_r[d], b_r[d]
            r = wa * ra + wb * rb
            wa, wb = wa * (1 + ra) / (1 + r), wb * (1 + rb) / (1 + r)
            if d in month_end:
                turn = abs(wa - 0.5) + abs(wb - 0.5)
                r -= turn * COST
                wa, wb = 0.5, 0.5
            out.append(r)
        return pd.Series(out, index=days)

    etf_lines = []
    for nm, r in (("SPHQ長揸", er["SPHQ"]), ("QUAL長揸", er["QUAL"]), ("MTUM長揸", er["MTUM"]),
                  ("QUAL+MTUM 各半", blend("QUAL", "MTUM") if er["QUAL"] is not None and er["MTUM"] is not None else None),
                  ("SPHQ+QQQ 各半", blend("SPHQ", "QQQ") if er["SPHQ"] is not None else None)):
        if r is None or r.notna().sum() < 250:
            L.append(f"　{nm}：冇數據")
            continue
        rr = r.dropna()
        a = rr.index[0].strftime("%Y-%m-%d")
        p, txt = show(nm, rr, zero, a, END)
        q, _ = show("QQQ", qqq_r, zero, a, END)
        L.append(txt + f"（由{a}；同期QQQ 年化 {q['cagr']:+.2f}%、跌 {q['mdd']:.1f}%、Sharpe {q['sharpe']:.2f}）")
    L.append("")

    # ---------- 事件研究 ----------
    L.append("=== 事件研究：盈利驚喜分5組，買入後跑贏SPY幾多（平均，%）===")
    L.append("　（買入日 = 知道消息之後下一個收市；只計當時係標普500成分股）")
    spy_c = spy.reindex(days)
    rows = []
    for t, k, s in events:
        b = k + 1
        if b >= len(days) or t not in members(days[k]):
            continue
        row = {"t": t, "date": days[k], "s": s}
        for h in (21, 63, 126):
            e = b + h
            if e < len(days) and np.isfinite(px[t].iat[b]) and np.isfinite(px[t].iat[e]):
                row[h] = (px[t].iat[e] / px[t].iat[b] - spy_c.iat[e] / spy_c.iat[b]) * 100
        rows.append(row)
    ev = pd.DataFrame(rows)
    if len(ev):
        for a, b in (("2003-01-01", "2014-12-31"), ("2015-01-01", END)):
            sub = ev[(ev["date"] >= a) & (ev["date"] <= b)].copy()
            if len(sub) < 100:
                continue
            sub["q"] = pd.qcut(sub["s"].rank(method="first"), 5, labels=["1最差", "2", "3", "4", "5最好"])
            L.append(f"　{a[:4]}-{b[:4]}（{len(sub)} 次公布）")
            for qn, g in sub.groupby("q", observed=True):
                parts = []
                for h, lab in ((21, "1個月"), (63, "3個月"), (126, "6個月")):
                    x = g[h].dropna() if h in g else pd.Series(dtype=float)
                    if len(x):
                        parts.append(f"{lab} {x.mean():+.2f}%（贏SPY機率 {(x > 0).mean() * 100:.0f}%）")
                L.append(f"　　第{qn}組：" + " ｜ ".join(parts))
            top, bot = sub[sub["q"] == "5最好"], sub[sub["q"] == "1最差"]
            L.append("　　最好組 − 最差組：" + " ｜ ".join(
                f"{lab} {top[h].mean() - bot[h].mean():+.2f}%" for h, lab in ((21, "1個月"), (63, "3個月"), (126, "6個月"))
                if h in sub))
    pd.DataFrame({"report": L}).to_csv("v14_results.csv", index=False, encoding="utf-8-sig")
    log("完成")
    print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("出錯：" + traceback.format_exc())
        raise
