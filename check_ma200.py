"""一次性檢查：用 --dry 睇每日訊息嘅 200日線2.6倍提醒（唔發Telegram、唔寫Sheet、唔改真紀錄）。"""
import glob
import json
import subprocess
import sys

out = []
r = subprocess.run("git fetch origin paper-trading && git checkout FETCH_HEAD -- paper_log", shell=True, capture_output=True, text=True)
out.append(f"取紀錄：{r.returncode}")
held = {"SNDK", "LITE", "MRNA", "MU", "DELL", "STX", "WDC", "INTC", "MRVL", "COHR", "MPC", "ILMN", "VLO", "HPE", "PSX", "TWLO", "CRWD"}
for f in sorted(glob.glob("paper_log/line*_state.json")):
    h = list(json.load(open(f)).get("hold", {}))
    out.append(f"{f}：{h}")
    held |= set(h)
import yfinance as yf  # noqa: E402
d = yf.download(sorted(held), period="2y", progress=False, auto_adjust=True)["Close"]
out.append("=== 每隻持股：最新收市 ÷ 200日平均 ===")
for t in sorted(held):
    s = d[t].dropna()
    if len(s) >= 200:
        rr = s.iloc[-1] / s.iloc[-200:].mean()
        out.append(f"{t} {s.index[-1].date()} 收市 {s.iloc[-1]:.2f} 200日平均 {s.iloc[-200:].mean():.2f} → {rr:.2f} 倍{'  ⚠️' if rr >= 2.6 else ''}")
cmds = [["line_d.py", f"--variant={v}", "--sheet-only", "--dry"] for v in ["D", "D2", "E", "E2", "E3"]]
for c in cmds:
    r = subprocess.run([sys.executable, "-u"] + c, capture_output=True, text=True)
    out.append(f"===== {' '.join(c)}（exit {r.returncode}）=====\n{r.stdout[-3000:]}\n{r.stderr[-1500:]}")
subprocess.run("git reset -q --hard HEAD && git clean -fdq paper_log", shell=True)   # 還原，等結果可以放上 research-results
open("check_ma200.txt", "w").write("\n".join(out))
print("\n".join(out)[-5000:])
