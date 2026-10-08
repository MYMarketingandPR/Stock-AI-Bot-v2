"""
sheet_v2.py — Google Sheet 新格式（分頁、清楚欄位），模擬交易用（2026-10-06 Esther 要求：Sheet 要同 Telegram 一樣詳細）

用法：Apps Script（Google Sheet → 擴充功能 → Apps Script）要先加入 handleV2 / doGet（見 sheet_apps_script.gs）。
未加之前：available() 會回 False，bot 照舊用舊格式（一行一行加喺第一頁），唔會出錯。

分頁：
  現時持倉   — 每日更新：每條線持有咩股票、幾錢買、股數、今日價、賺蝕、幾時/幾錢要賣、過去12個月升幅
  每日總覽   — 每日加一行：每條線總值、總回報、今日、最大跌幅、現金
  買賣紀錄   — 有買賣/建議買賣先加：日期、線、動作、股票、價錢、股數、金額、回報、12個月升幅、原因
  每月成績表 — 月尾：每個指標一行，每條線一欄
"""
import os

import requests

URL = os.environ.get("SHEET_WEBHOOK_URL")
_OK = None

HOLD_HEADER = ["線", "股票", "公司", "買入日", "買入價$", "股數", "買入金額$", "今日價$", "市值$", "賺蝕$", "賺蝕%",
               "持有期間最高價$", "幾時賣／賣出價", "買入時過去12個月升幅%", "而家過去12個月升幅%", "更新日期"]
DAILY_HEADER = ["日期", "線", "總值$", "總回報%", "今日%", "最大跌幅%", "現金$", "持股數目", "備註", "紀錄編號"]
TRADE_HEADER = ["日期", "線", "動作", "股票", "公司", "價錢$", "股數", "金額$", "今次回報%", "過去12個月升幅%", "原因／備註",
                "紀錄編號"]
SCORE_HEADER = ["月份", "指標", "線A", "線B", "線C", "線D", "線D2", "組合甲", "組合乙", "組合丙", "輪動", "SPY訊號", "SPY", "QQQ", "QQQM", "MTUM"]


def available():
    """Apps Script 有冇新格式（GET ?ping=v2 → v2-ok）。"""
    global _OK
    if _OK is None:
        _OK = False
        if URL:
            try:
                r = requests.get(URL, params={"ping": "v2"}, timeout=20)
                _OK = "v2-ok" in r.text
            except Exception as e:
                print(f"Sheet v2 檢查失敗：{e}")
    return _OK


def _clean(v):
    try:
        import numpy as np
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else round(float(v), 4)
        if isinstance(v, np.integer):
            return int(v)
    except Exception:
        pass
    return "" if v is None else v


def _post(payload, dry):
    payload["rows"] = [[_clean(x) for x in r] for r in payload.get("rows", [])]
    if dry or not URL:
        print("Sheet v2:", payload.get("action"), payload.get("tab"), f"{len(payload['rows'])} 行")
        for r in payload["rows"][:8]:
            print("   ", r)
        return
    if not available():
        return
    try:
        requests.post(URL, json=payload, timeout=60)
    except Exception as e:
        print(f"Sheet v2 寫入失敗：{e}")


def replace_group(tab, header, key_col, keys, rows, dry=False):
    """刪走 key_col（由1數起）屬於 keys 嘅舊行，再加新行。"""
    _post({"v": 2, "action": "replace_group", "tab": tab, "header": header, "key_col": key_col,
           "keys": [str(k) for k in keys], "rows": rows}, dry)


def append(tab, header, rows, dry=False):
    if rows:
        _post({"v": 2, "action": "append", "tab": tab, "header": header, "rows": rows}, dry)


def upsert(tab, header, rows, dry=False):
    """每行最後一欄係「紀錄編號」：同一日重跑會覆蓋舊行，唔會重複。"""
    if rows:
        replace_group(tab, header, len(header), [r[-1] for r in rows], rows, dry)
