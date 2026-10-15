"""一次性：檢查新 repo（Stock-AI-Bot-v2）嘅 3 個 Secrets 啱唔啱。發一句測試 Telegram，ping Google Sheet（唔寫任何資料）。
任何一樣唔得就以錯誤結束，等 GitHub 顯示失敗。"""
import os
import sys

import requests

tok, chat, url = (os.environ.get(k, "") for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "SHEET_WEBHOOK_URL"))
bad = []
r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                  json={"chat_id": chat, "text": "✅ 測試：新 repo（Stock-AI-Bot-v2）連接 Telegram 成功。呢條唔使理。"}, timeout=20)
ok_tg = r.ok and r.json().get("ok")
print("Telegram:", "OK" if ok_tg else f"失敗 {r.status_code} {r.text[:200]}")
if not ok_tg:
    bad.append("Telegram")
try:
    s = requests.get(url, params={"ping": "v2"}, timeout=30)
    ok_sh = "v2-ok" in s.text
except Exception as e:
    ok_sh = False
    print("Sheet 錯誤:", e)
print("Google Sheet:", "OK" if ok_sh else "失敗")
if not ok_sh:
    bad.append("Sheet")
sys.exit(1 if bad else 0)
