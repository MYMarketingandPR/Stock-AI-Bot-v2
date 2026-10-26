"""一次性檢查：用 --dry 跑線A/B/C、D、D2、E、E2、E3 每日訊息（唔發Telegram、唔寫Sheet、唔改真紀錄），睇 200日線2.6倍提醒。"""
import subprocess
import sys

out = []
r = subprocess.run("git fetch origin paper-trading && git checkout FETCH_HEAD -- paper_log", shell=True, capture_output=True, text=True)
out.append(f"取紀錄：{r.returncode} {r.stderr[-300:]}")
cmds = [["momentum_monthly.py", "--dry", "--tracks=NONE"]] + [["line_d.py", f"--variant={v}", "--dry"] for v in ["D", "D2", "E", "E2", "E3"]]
for c in cmds:
    r = subprocess.run([sys.executable, "-u"] + c, capture_output=True, text=True)
    out.append(f"===== {' '.join(c)}（exit {r.returncode}）=====\n{r.stdout}\n{r.stderr[-2000:]}")
open("check_ma200.txt", "w").write("\n".join(out))
print("\n".join(out)[-5000:])
