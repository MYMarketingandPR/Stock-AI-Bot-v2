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
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import main as bot
import momentum_monthly as MM
import momentum_v1 as M

M.PROGRESS = "/tmp/line_d_progress.txt"
CAPITAL = float(os.environ.get("PAPER_CAPITAL", "80000"))
COST = 0.001
SLOTS = 5
TRAIL = 0.20
STATE = "paper_log/lineD_state.json"
NAV = "paper_log/lineD_nav.tsv"
TRADES = "paper_log/lineD_trades.tsv"


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"cash": CAPITAL, "hold": {}, "pending_sell": [], "pending_buy": [], "pending_r12": {}, "start": None, "last_date": None}


def append_tsv(path, row):
    new = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False, mode="a", header=new)


def main():
    dry = "--dry" in sys.argv
    today = datetime.now(ZoneInfo("America/New_York")).date()
    st = load_state()
    if st.get("last_date") == str(today):
        print("線D：今日已經跑過。")
        return
    tickers, names = MM.current_sp500()
    held = list(st["hold"])
    need = sorted(set(tickers) | set(held) | set(st["pending_buy"]) | {"QQQ"})
    closes = M.download_close(need, (pd.Timestamp(today) - pd.Timedelta(days=420)).strftime("%Y-%m-%d"),
                              (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    px = pd.DataFrame(closes).sort_index()
    px = px[px.index.dayofweek < 5]
    if px.empty or px.index[-1].date() != today:
        print(f"線D：{today} 未有收市數據，可能休市。")
        return
    last = px.iloc[-1]
    enough = px.notna().sum() >= 253
    r12_all = (px.iloc[-1] / px.iloc[-253] - 1).where(enough)          # 過去12個月升幅（252個交易日）

    def r12txt(v):
        return "—" if v is None or not np.isfinite(v) else f"{v * 100:+.0f}%"

    def price(t):
        if t in px.columns:
            s = px[t].dropna()
            if len(s):
                return float(s.iloc[-1])
        return None

    lines = [f"🅳 *線D：倍升股 + 由高位跌20%賣*（模擬本金 ${CAPITAL:,.0f}）{today}", ""]
    traded = 0.0
    # 1) 執行昨日決定（今日收市價）
    for t in st["pending_sell"]:
        if t in st["hold"]:
            p = price(t)
            h = st["hold"].pop(t)
            p = p if p else h["last_px"]
            v = h["shares"] * p
            traded += v
            st["cash"] += v * (1 - COST)
            ret = (p / h["entry_px"] - 1) * 100
            append_tsv(TRADES, {"date": str(today), "action": "賣出", "ticker": t, "price": round(p, 2),
                                "shares": round(h["shares"], 4), "value": round(v, 2), "return_pct": round(ret, 2),
                                "reason": "由高位跌20%"})
            lines.append(f"✅ 已賣出 {t} @ ${p:,.2f}（今次 {ret:+.1f}%）")
            MM.sheet_row(today, "D", t, "已賣出", ret, p, h["peak"], dry)
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
                            "reason": f"倍升股（過去12個月 {r12txt(r_in)}）"})
        lines.append(f"✅ 已買入 {t} @ ${p:,.2f}（約 {sh:.2f} 股，${amt:,.0f}；買入時過去12個月已升 {r12txt(r_in)}）")
        MM.sheet_row(today, "D", t, f"已買入（過去12個月已升{r12txt(r_in)}）",
                     r_in * 100 if r_in is not None else None, p, p * (1 - TRAIL), dry)
    st["pending_sell"], st["pending_buy"], st["pending_r12"] = [], [], {}
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
        if h["last_px"] <= h["peak"] * (1 - TRAIL):
            st["pending_sell"].append(t)
    free = SLOTS - (len(st["hold"]) - len(st["pending_sell"]))
    cands = []
    if free > 0:
        uni = [t for t in tickers if t in px.columns]
        p0 = px[uni]
        ok = (p0.notna().sum() >= 253)
        uni = [t for t in uni if ok[t]]
        r12 = p0[uni].iloc[-1] / p0[uni].iloc[-253] - 1
        ma50 = p0[uni].iloc[-50:].mean()
        ma200 = p0[uni].iloc[-200:].mean()
        lastp = p0[uni].iloc[-1]
        sel = r12[(r12 >= 1.0) & (lastp > ma50) & (lastp > ma200)].sort_values(ascending=False)
        cands = [t for t in sel.index if t not in st["hold"]]
        st["pending_buy"] = cands[:free]
        st["pending_r12"] = {t: float(r12[t]) for t in st["pending_buy"]}
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
    MM.sheet_row(today, "D", "組合", "NAV", (nav / CAPITAL - 1) * 100, nav, f"{mdd:.1f}%", dry)
    lines.append("")
    if st["hold"]:
        lines.append("*持倉*（止賺/止蝕線 = 最高價 × 0.8）")
        for t, h in st["hold"].items():
            r = (h["last_px"] / h["entry_px"] - 1) * 100
            off = (h["last_px"] / h["peak"] - 1) * 100
            now12 = r12_all.get(t, np.nan)
            lines.append(f"• {t}（{names.get(t, '')}）{r:+.1f}% ｜ 今日 ${h['last_px']:,.2f} ｜ 最高 ${h['peak']:,.2f}"
                         f"（離高位 {off:.1f}%）｜ 賣出線 ${h['peak'] * (1 - TRAIL):,.2f} ｜ "
                         f"買入時12個月 {r12txt(h.get('r12_entry'))}、而家12個月 {r12txt(now12)}")
            MM.sheet_row(today, "D", t, f"持倉（買入時12個月{r12txt(h.get('r12_entry'))}，而家{r12txt(now12)}）",
                         r, h["entry_px"], h["peak"] * (1 - TRAIL), dry)
    else:
        lines.append("而家冇持倉。")
    lines.append(f"現金 ${st['cash']:,.0f}")
    lines.append("")
    if st["pending_sell"]:
        lines.append("🔴 *下一個交易日收市前賣出*：" + "、".join(
            f"{t}（今日 ${st['hold'][t]['last_px']:,.2f}，已由高位跌 ≥20%）" for t in st["pending_sell"]))
    if st["pending_buy"]:
        per = nav / SLOTS
        lines.append("🟢 *下一個交易日收市前買入*（每隻大約 ${:,.0f}）：".format(per))
        for t in st["pending_buy"]:
            p = float(last[t])
            g = float(r12[t])
            warn = " ⚠️ 已升超過2倍（歷史上追入風險較高，規則照買）" if g > 2.0 else ""
            lines.append(f"• {t}（{names.get(t, '')}）過去12個月已升 {g * 100:+.0f}% ｜ 參考價 ${p:,.2f} ｜ 大約 {per / p:.2f} 股{warn}")
            MM.sheet_row(today, "D", t, f"建議買入（下一個交易日收市；過去12個月已升{g * 100:+.0f}%）", g * 100, p, None, dry)
    if not st["pending_sell"] and not st["pending_buy"]:
        lines.append("⚪ 明日唔使買賣。")
    lines += ["", "規則：倍升股（12個月升≥100%）+ 喺MA50同MA200上面先買；由最高位跌20%就賣；揸最多5隻。",
              "_模擬交易，唔係投資建議_"]
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
