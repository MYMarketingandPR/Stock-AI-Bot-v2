"""
line_d.py — 模擬交易「線D」：倍升股輪換 + 由最高位跌20%賣（第十五輪C最接近合格嘅版本，2026-10-05 加入）

規則（鎖死，模擬期間唔改）：
- 股票池：今日標普500成分股；本金 $80,000（PAPER_CAPITAL），最多揸5隻，每隻 = 買入嗰日總資產 ÷ 5。
- 買入條件：過去252個交易日升 ≥ +100%（倍升股）+ 收市價 > MA50 + 收市價 > MA200。
  有空位就揀12個月升幅最大嘅；賣咗之後再符合條件可以再買。
- 賣出：買入之後記住最高收市價，收市價 ≤ 最高價 × 0.8（由最高位跌20%）就賣。冇其他止賺/止蝕。
- 時間：每個交易日收市後決定，下一個交易日收市價執行（同回測一樣）；每次買賣扣0.1%；現金唔計利息。
回測參考（第十五輪C，當時真實標普500）：2015-2026 每年 +21.4%（QQQ +19.1%），最大跌幅 -33.8%；2000-2009 每年 +5.0%。

紀錄（paper-trading 分支）：
  paper_log/lineD_state.json  — 現金、持倉（股數、買入日、買入價、最高價）、待執行指令
  paper_log/lineD_nav.tsv     — 每日總值同換手
  paper_log/lineD_trades.tsv  — 每次買賣
用法：python line_d.py [--dry]
"""
import json
import os
import sys
import numpy as np
import pandas as pd

import main as bot
import momentum_monthly as MM
import momentum_v1 as M
import prices as PR
import sheet_v2 as SV

M.PROGRESS = "/tmp/line_d_progress.txt"
CAPITAL = float(os.environ.get("PAPER_CAPITAL", "80000"))
COST = 0.001
SLOTS = 5
TRAIL = 0.20
# 線D2（2026-10-08 加，第二十二輪X2）：同線D一樣，但 SPY 每日訊號 = 「現金」嗰陣唔開新倉（已揸嘅照舊跌20%賣）
# 線E（2026-10-08 加，第二十六輪Y3）：今年升 ≥30% + 收市 ≥ 52週最高收市價 × 0.7 + 收市 > MA200 就買（揀今年升得最多）；
#   收市 < MA200 或者 收市 < 52週最高收市價 × 0.7 就賣；揸最多5隻。
# 線E2（2026-10-15 加，第三十三輪P4，Esther 要求觀察）：同線E一樣，但嗰隻股票所屬行業嘅 SPDR 行業ETF 收市 < 200日平均
#   就唔買（已揸嘅照線E規則賣）。行業 = Wikipedia 標普500 GICS Sector；冇分類 = 當過關。回測冇達到「有改善」標準，只係觀察。
# 線E3（2026-10-15 加，第三十四輪參考G0，Esther 要求觀察）：同線E一樣，但 SPY 每日訊號 = 「現金」就唔開新倉（同線D2一樣嘅閘門）。
#   回測 2015起 +28.1% / -32.6%（同線E差唔多），2000-2009 +12.1% / -46%（線E +4.5% / -62%）；未經預先登記判斷，只係觀察。
# 線E4（2026-10-28 加，第四十三輪W2，Esther 要求觀察）：買入同線E一樣；賣出改用「第二次周收市跌穿周線EMA21」（取代線E原本賣出條件）。
#   周收市 = 每個星期最後一個交易日收市；周EMA21 = 周收市嘅21週指數平均。買入之後第一個收市跌穿嘅星期 = 第一次，
#   之後要有最少一個星期收市喺上面，再跌穿 = 第二次 → 下一個交易日收市賣。賣出之後 20 個交易日內唔再買同一隻。
#   每個星期五收市後檢查；如果星期五休市，就喺下個星期第一個交易日先檢查（遲一日，回測冇呢個分別）。
#   回測 2015起 +31.8% / 最大跌幅 -39.9%（線E +28.7% / -32.6%），冇達「有改善」標準（跌幅差太多），只係觀察。
# 線E5（2026-10-28 加，第四十四輪C90，Esther 要求觀察）：同線E一樣，但只買今年升 +30% 至 +90% 嘅股（包埋兩邊；揀範圍入面升最多）；
#   賣出同線E一樣（買咗之後升過90%唔會因此賣）。回測 2015起 +26.9% / -31.3%（線E +28.7% / -32.6%），2000-2009 +12.3%（線E +4.5%）；
#   冇達「有改善」標準，而且 C110 差過 C90（可能有運氣成分），只係觀察。
TAG = ("E5" if "--variant=E5" in sys.argv else "E4" if "--variant=E4" in sys.argv else "E3" if "--variant=E3" in sys.argv else "E2" if "--variant=E2" in sys.argv else "E" if "--variant=E" in sys.argv
       else ("D2" if "--variant=D2" in sys.argv else "D"))
LABEL = f"線{TAG}"
SPY_GATE = TAG in ("D2", "E3")
IS_E = TAG in ("E", "E2", "E3", "E4", "E5")
YTD_CAP = 0.90 if TAG == "E5" else np.inf
W_EMA = TAG == "E4"
COOL_DAYS = 20
SECTOR_GATE = TAG == "E2"
SECTOR_ETF = {"Information Technology": "XLK", "Health Care": "XLV", "Financials": "XLF", "Consumer Discretionary": "XLY",
              "Consumer Staples": "XLP", "Industrials": "XLI", "Materials": "XLB", "Energy": "XLE", "Utilities": "XLU",
              "Communication Services": "XLC", "Real Estate": "XLRE"}
SECTOR_ZH = {"Information Technology": "資訊科技", "Health Care": "醫療", "Financials": "金融", "Consumer Discretionary": "非必需消費",
             "Consumer Staples": "必需消費", "Industrials": "工業", "Materials": "原材料", "Energy": "能源", "Utilities": "公用",
             "Communication Services": "通訊服務", "Real Estate": "房地產"}
LOOKBACK_DAYS = 760                 # 夠計「之前一年」升幅（504個交易日）
STATE = f"paper_log/line{TAG}_state.json"
NAV = f"paper_log/line{TAG}_nav.tsv"
TRADES = f"paper_log/line{TAG}_trades.tsv"


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"cash": CAPITAL, "hold": {}, "pending_sell": [], "pending_buy": [], "pending_r12": {}, "start": None, "last_date": None}


def prev_year(s):
    """之前一年升幅：252至504個交易日前（s = 已去除空值嘅收市價）。唔夠數據回傳 None。"""
    return float(s.iloc[-253] / s.iloc[-505] - 1) if len(s) >= 505 else None


def second_txt(g):
    """第二次倍升提示（第二十五輪：之前一年都已經升 ≥100% 嘅股票，之後12個月中位數 -14.8%）。只係提醒，唔影響買賣。"""
    return f"⚠️ 第二次倍升（之前一年 {g * 100:+.0f}%）" if g is not None and g >= 1.0 else ""


def ma200_txt(s):
    """2026-10-26 Esther 要求：股價 ≥ 200日線 × 2.6 就提醒（第四十一B輪：2000年起標普500只有15隻試過）。只係提醒，唔影響買賣。"""
    if len(s) < 200:
        return ""
    r = float(s.iloc[-1]) / float(s.iloc[-200:].mean())
    return f"⚠️ 股價係200日線 {r:.1f} 倍（≥2.6 倍，極罕見）" if r >= 2.6 else ""


def flags(s):
    return " ".join(x for x in [second_txt(prev_year(s)), ma200_txt(s)] if x)


def e_levels(s):
    """線E：回傳（200日平均, 52週最高收市價 × 0.7）。"""
    return (float(s.iloc[-200:].mean()) if len(s) >= 200 else None, float(s.iloc[-252:].max()) * 0.7 if len(s) >= 200 else None)


def weekly_ema(s):
    """線E4：回傳（周收市, 周EMA21）；index = 個星期嘅星期五。"""
    w = s.resample("W-FRI").last().dropna()
    return w, w.ewm(span=21, adjust=False, min_periods=21).mean()


def wema_update(h, s, today):
    """線E4：處理買入之後、已經完咗嘅星期（星期五 ≤ 今日），數「收市跌穿周EMA21」幾多次（連續幾個星期只算一次）。回傳次數。"""
    w, e = weekly_ema(s)
    start = pd.Timestamp(h["entry_date"]).to_period("W-FRI").end_time.normalize()
    done = pd.Timestamp(h["w_last"]) if h.get("w_last") else None
    for lab in w.index:
        if lab < start or lab > pd.Timestamp(today) or (done is not None and lab <= done) or not np.isfinite(e[lab]):
            continue
        below = bool(w[lab] < e[lab])
        if below and not h.get("w_prev", False):
            h["w_n"] = h.get("w_n", 0) + 1
        h["w_prev"] = below
        h["w_last"] = str(lab.date())
    return h.get("w_n", 0)


def wema_txt(h, s):
    w, e = weekly_ema(s)
    if not len(e) or not np.isfinite(e.iloc[-1]):
        return "周EMA21 —"
    wk = "今個星期暫計" if w.index[-1] > s.index[-1] else f"{w.index[-1].strftime('%m-%d')}嗰個星期"
    return (f"周EMA21 ${e.iloc[-1]:,.2f}（{wk}，收市 ${w.iloc[-1]:,.2f}）｜ "
            f"買入後收市跌穿 {h.get('w_n', 0)} 次（第2次就賣）")


def sell_reason(h=None):
    if W_EMA:
        return "第二次周收市跌穿周線EMA21"
    if IS_E:
        return "跌穿200日線或者離52週高位超過30%"
    return f"由買入後最高收市 ${h['peak']:,.2f} 跌 ≥20%" if h else "由高位跌20%"


def recent_high(t, day, n=42):
    """近期高位（只係顯示，唔影響買賣）：最近 n 個交易日（約兩個月，2026-10-14 用戶要求由一個月改做兩個月）嘅市中最高價，同埋係邊日。
    用未調整價（同券商App睇到嘅一樣）。攞唔到就回傳 (None, None)。"""
    try:
        import yfinance as yf
        d = pd.Timestamp(day)
        h = yf.Ticker(t).history(start=(d - pd.Timedelta(days=75)).strftime("%Y-%m-%d"),
                                 end=(d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), auto_adjust=False)
        hi = h["High"].astype(float).dropna()
        hi.index = pd.DatetimeIndex(hi.index).tz_localize(None).normalize()
        hi = hi[hi.index <= d].iloc[-n:]
        if hi.empty:
            return None, None
        return float(hi.max()), hi.idxmax().strftime("%m-%d")
    except Exception as e:
        print(f"{t} 近期高位攞唔到：{e}")
        return None, None


def sector_map():
    """線E2：標普500成分股 → GICS Sector（Wikipedia）。攞唔到就回傳 {}（全部當過關，訊息會註明）。"""
    import io
    import requests
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        t = pd.read_html(io.StringIO(html))[0]
        return {M.norm(a): b for a, b in zip(t["Symbol"], t["GICS Sector"]) if b in SECTOR_ETF}
    except Exception as e:
        print(f"行業分類攞唔到：{e}")
        return {}


def sell_rule_txt(h, s):
    if W_EMA:
        return wema_txt(h, s) + "；周收市第二次跌穿就賣（下一個交易日收市）"
    if not IS_E:
        return f"收市 ≤ ${h['peak'] * (1 - TRAIL):,.2f} 就賣（由最高位跌20%，下一個交易日收市）"
    m, k = e_levels(s)
    if m is None:
        return "收市跌穿200日線，或者低過52週最高位 × 0.7 就賣"
    return f"收市低過 ${max(m, k):,.2f} 就賣（200日線 ${m:,.2f}／52週最高位×0.7 ${k:,.2f}，邊個高用邊個；下一個交易日收市）"


def append_tsv(path, row):
    new = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False, mode="a", header=new)


def sheet_only(dry):
    """只將線D（或者線D2）而家嘅狀態寫入 Google Sheet（唔交易、唔改紀錄、唔發 Telegram）。"""
    st = load_state()
    day = st.get("last_date")
    if not day:
        print(f"{LABEL}：未有紀錄。")
        return
    tickers, names = MM.current_sp500()
    need = sorted(set(st["hold"]) | set(st["pending_buy"]))
    if not need:
        return
    closes = M.download_close(need, (pd.Timestamp(day) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d"),
                              (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    px = pd.DataFrame(closes).sort_index()
    r12 = {t: float(px[t].dropna().iloc[-1] / px[t].dropna().iloc[-253] - 1) for t in px.columns if px[t].notna().sum() >= 253}
    navs = pd.read_csv(NAV, sep="\t") if os.path.exists(NAV) else pd.DataFrame({"nav": [CAPITAL]})
    nav = float(navs["nav"].iloc[-1])
    hold_rows = []
    for t, h in st["hold"].items():
        cost, val = h["shares"] * h["entry_px"], h["shares"] * h["last_px"]
        sx = px[t].dropna() if t in px.columns else pd.Series(dtype=float)
        flag = flags(sx)
        hold_rows.append([LABEL, t, (names.get(t, "") + (" " + flag if flag else "")).strip(), h["entry_date"], round(h["entry_px"], 2),
                          round(h["shares"], 4), round(cost, 2),
                          round(h["last_px"], 2), round(val, 2), round(val - cost, 2), round((h["last_px"] / h["entry_px"] - 1) * 100, 2),
                          round(h["peak"], 2), sell_rule_txt(h, sx),
                          round(h["r12_entry"] * 100, 1) if h.get("r12_entry") is not None else "",
                          round(r12[t] * 100, 1) if t in r12 else "", day])
    SV.replace_group("現時持倉", SV.HOLD_HEADER, 1, [LABEL], hold_rows, dry)
    per = nav / SLOTS
    rows = []
    for t in st["pending_buy"]:
        if t not in px.columns:
            continue
        p = float(px[t].dropna().iloc[-1])
        g = st.get("pending_r12", {}).get(t, r12.get(t))
        rows.append([day, LABEL, "建議買入（下一個交易日收市）", t, names.get(t, ""), round(p, 2), round(per / p, 4), round(per, 2), "",
                     round(g * 100, 1) if g is not None else "",
                     "；".join(x for x in [("今年升≥30%，近52週高位，喺MA200上面" if IS_E else ("已升超過2倍（歷史上追入風險較高，規則照買）" if g and g > 2 else "倍升股")),
                                           flags(px[t].dropna())] if x)])
    if os.path.exists(TRADES):                    # 已執行嘅買賣：由紀錄檔重寫（價錢有改正都會更新）
        tr = pd.read_csv(TRADES, sep="\t")
        for _, x in tr.iterrows():
            act = "已買入" if x["action"] == "買入" else "已賣出"
            rows.append([x["date"], LABEL, act, x["ticker"], names.get(x["ticker"], ""), x["price"], x["shares"], x["value"],
                         x["return_pct"] if pd.notna(x["return_pct"]) else "", "", x["reason"]])
    SV.upsert("買賣紀錄", SV.TRADE_HEADER, [r + [f"{r[0]}|{r[1]}|{r[3]}|{r[2]}"] for r in rows], dry)
    SV.upsert("每日總覽", SV.DAILY_HEADER, [[day, LABEL, round(nav, 2), round((nav / CAPITAL - 1) * 100, 2), "", "",
                                          round(st["cash"], 2), len(st["hold"]),
                                          f"明日賣 {len(st['pending_sell'])} 隻、買 {len(st['pending_buy'])} 隻", f"{day}|{LABEL}"]], dry)


def main():
    dry = "--dry" in sys.argv
    if "--sheet-only" in sys.argv:
        sheet_only(dry)
        return
    today = PR.market_day()               # 最近一個已收市交易日（排程遲咗過半夜都啱）
    st = load_state()
    if st.get("last_date") == str(today):
        print(f"{LABEL}：今日已經跑過。")
        return
    tickers, names = MM.current_sp500()
    held = list(st["hold"])
    need = sorted(set(tickers) | set(held) | set(st["pending_buy"]) | {"QQQ"} | (set(SECTOR_ETF.values()) if SECTOR_GATE else set()))
    closes = M.download_close(need, (pd.Timestamp(today) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d"),
                              (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    px = pd.DataFrame(closes).sort_index()
    px = px[px.index.dayofweek < 5]
    # 逐隻檢查有冇今日收市價（2026-10-08 修正：之前有啲股票用咗前一日價）
    final = PR.last_chance()
    critical = sorted(set(held) | set(st["pending_buy"]) | set(st["pending_sell"]) | {"QQQ"}
                      | (set(SECTOR_ETF.values()) if SECTOR_GATE else set()))
    px, miss_c = PR.ensure_fresh(px, critical, today)
    if "QQQ" in miss_c:
        print(f"{LABEL}：{today} 未有收市數據（可能休市，或者 Yahoo 未更新），遲啲再試。")
        return
    uni_all = [t for t in tickers if t in px.columns]
    px, miss_u = PR.ensure_fresh(px, uni_all, today, rounds=2)
    too_many = len(miss_u) > 0.02 * max(len(uni_all), 1)
    if (miss_c or too_many) and not final:
        print(f"{LABEL}：{today} 收市價未齊（持倉/待執行 {miss_c}；股票池 {len(miss_u)} 隻），今次唔做，等後備時間再跑。")
        return
    stale = set(miss_c)
    px = px[px.index <= pd.Timestamp(today)]
    last = px.iloc[-1]
    enough = px.notna().sum() >= 253
    r12_all = (px.iloc[-1] / px.iloc[-253] - 1).where(enough)          # 過去12個月升幅（252個交易日）

    def r12txt(v):
        return "—" if v is None or not np.isfinite(v) else f"{v * 100:+.0f}%"

    def price(t):
        """只回傳今日收市價；未有就 None（唔會再靜靜雞用前一日價）。"""
        if t in px.columns:
            v = px[t].get(pd.Timestamp(today), np.nan)
            if np.isfinite(v):
                return float(v)
        return None

    title = ("線E5：線E + 只買今年升30%至90%" if TAG == "E5" else
             "線E4：線E買入 + 第二次周收市跌穿周EMA21先賣" if W_EMA else
             "線E2：線E + 行業轉弱唔開新倉" if SECTOR_GATE else
             "線E3：線E + SPY訊號「持有」先開新倉" if TAG == "E3" else
             "線E：今年升≥30% + 離高位<30% + 喺MA200上面" if IS_E else
             "線D2：線D + SPY訊號「持有」先開新倉" if SPY_GATE else "線D：倍升股 + 由高位跌20%賣")
    lines = [f"{'🅴' if IS_E else '🅳'} *{title}*（模擬本金 ${CAPITAL:,.0f}）{today}", ""]
    traded = 0.0
    trade_rows = []
    if stale:
        lines.append(f"⚠️ 以下股票到最後都未有 {today} 收市價，今日唔郁佢哋（價錢用返上一次）：{'、'.join(sorted(stale))}")
    carry_sell = []
    # 1) 執行昨日決定（今日收市價）
    for t in st["pending_sell"]:
        if t in st["hold"]:
            p = price(t)
            if not p:
                carry_sell.append(t)          # 未有今日價：留到下一個交易日先賣
                continue
            h = st["hold"].pop(t)
            v = h["shares"] * p
            traded += v
            st["cash"] += v * (1 - COST)
            ret = (p / h["entry_px"] - 1) * 100
            append_tsv(TRADES, {"date": str(today), "action": "賣出", "ticker": t, "price": round(p, 2),
                                "shares": round(h["shares"], 4), "value": round(v, 2), "return_pct": round(ret, 2),
                                "reason": sell_reason()})
            lines.append(f"✅ 已賣出 {t} @ ${p:,.2f}（今次 {ret:+.1f}%）")
            trade_rows.append([str(today), LABEL, "已賣出", t, names.get(t, ""), round(p, 2), round(h["shares"], 4), round(v, 2),
                               round(ret, 2), round(float(r12_all.get(t, np.nan)) * 100, 1) if np.isfinite(r12_all.get(t, np.nan)) else "",
                               sell_reason(h)
                               + f"（買入日 {h['entry_date']}，買入價 ${h['entry_px']:,.2f}）"])
            MM.sheet_row(today, TAG, t, "已賣出", ret, p, h["peak"], dry)
    total = st["cash"] + sum(h["shares"] * (price(t) or h["last_px"]) for t, h in st["hold"].items())
    for t in st["pending_buy"]:
        if t in st["hold"] or len(st["hold"]) >= SLOTS:
            continue
        p = price(t)
        if not p:
            continue
        amt = min(total / SLOTS, st["cash"])
        if amt <= total * 0.01:
            break
        st["cash"] -= amt
        sh = amt * (1 - COST) / p
        r_in = st.get("pending_r12", {}).get(t)
        st["hold"][t] = {"shares": sh, "entry_date": str(today), "entry_px": p, "peak": p, "last_px": p, "r12_entry": r_in}
        traded += amt
        append_tsv(TRADES, {"date": str(today), "action": "買入", "ticker": t, "price": round(p, 2),
                            "shares": round(sh, 4), "value": round(amt, 2), "return_pct": "",
                            "reason": (f"今年升 {r12txt(st.get('pending_ytd', {}).get(t))}（過去12個月 {r12txt(r_in)}）" if IS_E
                                       else f"倍升股（過去12個月 {r12txt(r_in)}）")})
        lines.append(f"✅ 已買入 {t} @ ${p:,.2f}（約 {sh:.2f} 股，${amt:,.0f}；買入時過去12個月已升 {r12txt(r_in)}）")
        trade_rows.append([str(today), LABEL, "已買入", t, names.get(t, ""), round(p, 2), round(sh, 4), round(amt, 2), "",
                           round(r_in * 100, 1) if r_in is not None else "",
                           "第二次周收市跌穿周線EMA21就賣" if W_EMA else "跌穿200日線或者低過52週最高位 × 0.7 就賣" if IS_E else f"賣出線 ${p * (1 - TRAIL):,.2f}（之後跟最高價 × 0.8 上移）"])
        MM.sheet_row(today, TAG, t, f"已買入（過去12個月已升{r12txt(r_in)}）",
                     r_in * 100 if r_in is not None else None, p, p * (1 - TRAIL), dry)
    st["pending_sell"], st["pending_buy"], st["pending_r12"], st["pending_ytd"] = list(carry_sell), [], {}, {}
    if st["start"] is None:
        st["start"] = str(today)
    # 2) 更新持倉、最高價
    for t, h in st["hold"].items():
        p = price(t)
        if p:
            h["last_px"] = p
            h["peak"] = max(h["peak"], p)
    nav = st["cash"] + sum(h["shares"] * h["last_px"] for h in st["hold"].values())
    append_tsv(NAV, {"date": str(today), "nav": round(nav, 2), "turn": round(traded / nav, 4) if nav > 0 else 0})
    # 3) 今日收市決定
    for t, h in st["hold"].items():
        if t in st["pending_sell"] or not price(t):
            continue
        if W_EMA:
            if wema_update(h, px[t].dropna(), today) >= 2:
                st["pending_sell"].append(t)
                st.setdefault("cool", {})[t] = str(today)
        elif IS_E:
            m, k = e_levels(px[t].dropna())
            if m is not None and (h["last_px"] < m or h["last_px"] < k):
                st["pending_sell"].append(t)
        elif h["last_px"] <= h["peak"] * (1 - TRAIL):
            st["pending_sell"].append(t)
    free = SLOTS - (len(st["hold"]) - len(st["pending_sell"]))
    cands = []
    gate_txt = ""
    if SPY_GATE:                              # 線D2／線E3：SPY 每日訊號 = 現金 → 唔開新倉
        sdf = bot.replay(bot.add_signals(bot.load_data()))
        sdf = sdf[sdf.index <= pd.Timestamp(today)]
        spy_hold = sdf["state"].iloc[-1] != "out"
        gate_txt = (f"SPY每日訊號（{sdf.index[-1].date()}）：{'持有 → 可以開新倉' if spy_hold else '現金 → 今日唔開新倉'}")
        if not spy_hold:
            free = 0
    if free > 0:
        uni = [t for t in tickers if t in px.columns]
        p0 = px[uni]
        ok = (p0.notna().sum() >= 253)
        uni = [t for t in uni if ok[t]]
        r12 = p0[uni].iloc[-1] / p0[uni].iloc[-253] - 1
        ma50 = p0[uni].iloc[-50:].mean()
        ma200 = p0[uni].iloc[-200:].mean()
        lastp = p0[uni].iloc[-1]
        if IS_E:
            ye = p0[uni][p0.index.year < today.year].ffill().iloc[-1]          # 上年最後收市價
            ytd = lastp / ye - 1
            hi52 = p0[uni].iloc[-252:].max()
            sel = ytd[(ytd >= 0.30) & (ytd <= YTD_CAP) & (lastp >= hi52 * 0.7) & (lastp > ma200)].sort_values(ascending=False)
            if SECTOR_GATE:                   # 線E2：行業ETF收市 < 200日平均 → 嗰個行業唔開新倉
                smap = sector_map()
                weak = []
                for g, e in SECTOR_ETF.items():
                    if e in px.columns:
                        se = px[e].dropna()
                        if len(se) >= 200 and se.iloc[-1] < se.iloc[-200:].mean():
                            weak.append(g)
                before = list(sel.index)
                sel = sel[[smap.get(t) not in weak for t in sel.index]]
                skipped = [t for t in before[:SLOTS + 5] if t not in sel.index and t not in st["hold"]]
                gate_txt = ("行業轉弱（行業ETF低過200日線）：" + ("、".join(SECTOR_ZH[g] for g in weak) if weak else "冇")
                            + ("；今日因此冇買：" + "、".join(skipped) if skipped else "")
                            + ("" if smap else "（⚠️ 今日攞唔到行業分類，全部當過關）"))
        else:
            sel = r12[(r12 >= 1.0) & (lastp > ma50) & (lastp > ma200)].sort_values(ascending=False)
        cool_txt = ""
        if W_EMA:                                  # 線E4：賣出決定之後 20 個交易日內唔再買同一隻
            blocked = [t for t, d in st.get("cool", {}).items() if (px.index > pd.Timestamp(d)).sum() <= COOL_DAYS]
            st["cool"] = {t: d for t, d in st.get("cool", {}).items() if t in blocked}
            sel = sel[[t not in blocked for t in sel.index]]
            if blocked:
                cool_txt = "冷靜期（賣出後20個交易日唔再買）：" + "、".join(blocked)
        cands = [t for t in sel.index if t not in st["hold"]]
        st["pending_buy"] = cands[:free]
        st["pending_r12"] = {t: float(r12[t]) for t in st["pending_buy"]}
        if IS_E:
            st["pending_ytd"] = {t: float(ytd[t]) for t in st["pending_buy"]}
        if cool_txt:
            gate_txt = (gate_txt + "\n" if gate_txt else "") + cool_txt
    # 4) 訊息
    navs = pd.read_csv(NAV, sep="\t")
    nv = navs["nav"].values
    mdd = (nv / np.maximum.accumulate(np.r_[CAPITAL, nv])[1:] - 1).min() * 100
    start = pd.Timestamp(st["start"])
    q = px["QQQ"].dropna() if "QQQ" in px.columns else None
    qtxt = ""
    if q is not None and len(q[q.index >= start]):
        qs = q[q.index >= start]
        qtxt = f"｜ 同期QQQ {(qs.iloc[-1] / qs.iloc[0] - 1) * 100:+.1f}%"
    lines.insert(1, f"總值 ${nav:,.0f}（{(nav / CAPITAL - 1) * 100:+.1f}%）｜ 最大跌幅 {mdd:.1f}% {qtxt}（由 {st['start']} 開始）")
    MM.sheet_row(today, TAG, "組合", "NAV", (nav / CAPITAL - 1) * 100, nav, f"{mdd:.1f}%", dry)
    dr_ = (nv[-1] / nv[-2] - 1) * 100 if len(nv) > 1 else 0.0
    hold_rows = []
    lines.append("")
    if st["hold"]:
        lines.append("*持倉*（" + ("賣出 = 周收市第二次跌穿周線EMA21" if W_EMA else "賣出線 = 200日線同52週最高位×0.7 邊個高用邊個" if IS_E else "止賺/止蝕線 = 買入後最高收市 × 0.8") + "）")
        for t, h in st["hold"].items():
            r = (h["last_px"] / h["entry_px"] - 1) * 100
            now12 = r12_all.get(t, np.nan)
            sx = px[t].dropna() if t in px.columns else pd.Series(dtype=float)
            flag = flags(sx)
            if W_EMA:
                sl = wema_txt(h, sx)
            elif IS_E:
                m, k = e_levels(sx)
                sl = f"賣出線 ${max(m, k):,.2f}（200日線 ${m:,.2f}／高位×0.7 ${k:,.2f}）" if m is not None else "賣出線 —"
            else:
                sl = f"賣出線 ${h['peak'] * (1 - TRAIL):,.2f}（買入後最高收市 ${h['peak']:,.2f} × 0.8）"
            rh, rd = recent_high(t, today)
            rh_txt = (f"近2個月高位 ${rh:,.2f}（{rd} 市中最高，今日 {(h['last_px'] / rh - 1) * 100:+.1f}%）｜ "
                      if rh else "")
            hi52 = float(sx.iloc[-252:].max()) if len(sx) else None
            hi52_txt = (f"52週最高收市 ${hi52:,.2f}（今日 {(h['last_px'] / hi52 - 1) * 100:+.1f}%）｜ "
                        if hi52 else "")
            lines.append(f"• {t}（{names.get(t, '')}）{r:+.1f}% ｜ 今日 ${h['last_px']:,.2f} ｜ "
                         f"{rh_txt}{hi52_txt}{sl} ｜ "
                         f"買入時12個月 {r12txt(h.get('r12_entry'))}、而家12個月 {r12txt(now12)}" + (f" ｜ {flag}" if flag else ""))
            MM.sheet_row(today, TAG, t, f"持倉（買入時12個月{r12txt(h.get('r12_entry'))}，而家{r12txt(now12)}）",
                         r, h["entry_px"], h["peak"] * (1 - TRAIL), dry)
            cost = h["shares"] * h["entry_px"]
            val = h["shares"] * h["last_px"]
            hold_rows.append([LABEL, t, (names.get(t, "") + (" " + flag if flag else "")).strip(), h["entry_date"], round(h["entry_px"], 2),
                              round(h["shares"], 4),
                              round(cost, 2), round(h["last_px"], 2), round(val, 2), round(val - cost, 2), round(r, 2),
                              round(h["peak"], 2), sell_rule_txt(h, sx),
                              round(h["r12_entry"] * 100, 1) if h.get("r12_entry") is not None else "",
                              round(float(now12) * 100, 1) if np.isfinite(now12) else "", str(today)])
    else:
        lines.append("而家冇持倉。")
    lines.append(f"現金 ${st['cash']:,.0f}")
    lines.append("")
    for t in st["pending_sell"]:
        hh_ = st["hold"][t]
        trade_rows.append([str(today), LABEL, "建議賣出（下一個交易日收市）", t, names.get(t, ""), round(hh_["last_px"], 2),
                           round(hh_["shares"], 4), round(hh_["shares"] * hh_["last_px"], 2),
                           round((hh_["last_px"] / hh_["entry_px"] - 1) * 100, 2), "",
                           "已" + sell_reason(hh_)])
    if st["pending_sell"]:
        lines.append("🔴 *下一個交易日收市前賣出*：" + "、".join(
            f"{t}（今日 ${st['hold'][t]['last_px']:,.2f}，已" + sell_reason(st['hold'][t]) + "）"
            for t in st["pending_sell"]))
    if st["pending_buy"]:
        per = nav / SLOTS
        lines.append("🟢 *下一個交易日收市前買入*（每隻大約 ${:,.0f}）：".format(per))
        for t in st["pending_buy"]:
            p = float(last[t])
            g = float(r12[t])
            warn = " ⚠️ 已升超過2倍（歷史上追入風險較高，規則照買）" if g > 2.0 else ""
            flag = flags(px[t].dropna())
            ytxt = f"今年升 {st['pending_ytd'][t] * 100:+.0f}% ｜ " if IS_E else ""
            lines.append(f"• {t}（{names.get(t, '')}）{ytxt}過去12個月已升 {g * 100:+.0f}% ｜ 參考價 ${p:,.2f} ｜ 大約 {per / p:.2f} 股{warn}"
                         + (f" ｜ {flag}" if flag else ""))
            MM.sheet_row(today, TAG, t, f"建議買入（下一個交易日收市；過去12個月已升{g * 100:+.0f}%）", g * 100, p, None, dry)
            trade_rows.append([str(today), LABEL, "建議買入（下一個交易日收市）", t, names.get(t, ""), round(p, 2),
                               round(per / p, 4), round(per, 2), "", round(g * 100, 1),
                               "；".join(x for x in [(f"今年升 {st['pending_ytd'][t] * 100:+.0f}%，近52週高位，喺MA200上面" if IS_E else
                                                      ("已升超過2倍（歷史上追入風險較高，規則照買）" if g > 2.0 else "倍升股")), flag] if x)])
    if not st["pending_sell"] and not st["pending_buy"]:
        lines.append("⚪ 明日唔使買賣。")
    if gate_txt:
        lines.append(gate_txt)
    rule = ("規則：今年升≥30% + 離52週最高位唔夠30% + 喺200日線上面先買（揀今年升最多）；周收市第二次跌穿周線EMA21就賣"
            "（買入後第一次跌穿之後，要有一個星期收返上去，再跌穿先算第二次）；賣出後20個交易日唔再買同一隻；揸最多5隻。"
            "（第四十三輪W2回測 2015起 +31.8%/年、最大跌幅 -39.9%；線E +28.7% / -32.6%；冇達改善標準，只係觀察）"
            if W_EMA else
            "規則：今年升≥30% + 離52週最高位唔夠30% + 喺200日線上面先買（揀今年升最多）；跌穿200日線或者離52週高位超過30%就賣；揸最多5隻。"
            + ("（線E2 再加：嗰隻股所屬行業嘅ETF低過200日線就唔買；第三十三輪回測 +29.0%/年、最大跌幅 -32.9%，同線E差唔多，未達改善標準，只係觀察）"
               if SECTOR_GATE else
               "（線E3 再加：SPY每日訊號係「現金」嗰陣唔開新倉；第三十四輪參考回測 2015起 +28.1%/年、2000-2009 +12.1% / 最大跌幅 -46%，只係觀察）"
               if TAG == "E3" else
               "（線E5 再加：今年升超過90%就唔買，已買嘅照線E規則賣；第四十四輪C90回測 2015起 +26.9%/年、最大跌幅 -31.3%，"
               "2000-2009 +12.3%；未達改善標準，只係觀察）"
               if TAG == "E5" else "（第二十六輪：2015年起回測 +28.7%/年，但 2000-2009 試過跌 -64%）") if IS_E else
            "規則：倍升股（12個月升≥100%）+ 喺MA50同MA200上面先買；由最高位跌20%就賣；揸最多5隻。"
            + ("（線D2：SPY每日訊號係「現金」嗰陣唔開新倉，用嚟同線D比較）" if SPY_GATE else ""))
    lines += ["", rule, "⚠️ 第二次倍升 = 之前一年都已經升 ≥100%，歷史上之後12個月中位數 -14.8%（只係提醒，唔影響買賣）。",
              "⚠️ 股價 ≥ 200日線 2.6 倍 = 極罕見（2000年起標普500只有15隻試過，第四十一B輪）；只係提醒，唔影響買賣，你自己決定點處理。",
              "_模擬交易，唔係投資建議_"]
    SV.replace_group("現時持倉", SV.HOLD_HEADER, 1, [LABEL], hold_rows, dry)
    SV.upsert("每日總覽", SV.DAILY_HEADER, [[str(today), LABEL, round(nav, 2), round((nav / CAPITAL - 1) * 100, 2), round(dr_, 2),
                                          round(mdd, 2), round(st["cash"], 2), len(st["hold"]),
                                          f"明日賣 {len(st['pending_sell'])} 隻、買 {len(st['pending_buy'])} 隻",
                                          f"{today}|{LABEL}"]], dry)
    SV.upsert("買賣紀錄", SV.TRADE_HEADER, [r + [f"{r[0]}|{r[1]}|{r[3]}|{r[2]}"] for r in trade_rows], dry)
    st["last_date"] = str(today)
    with open(STATE, "w") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    text = "\n".join(lines)
    print(text)
    if not dry:
        bot.send_telegram(text)


if __name__ == "__main__":
    os.makedirs("paper_log", exist_ok=True)
    main()
