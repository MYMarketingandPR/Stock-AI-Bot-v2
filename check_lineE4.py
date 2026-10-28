"""一次性檢查：用 --dry 跑線E4（唔發Telegram、唔寫Sheet、唔改真紀錄）。"""
import subprocess
import sys

out = []
r = subprocess.run("git fetch origin paper-trading && git checkout FETCH_HEAD -- paper_log", shell=True, capture_output=True, text=True)
out.append(f"取紀錄：{r.returncode}")
for c in [["line_d.py", "--variant=E4", "--dry"], ["line_d.py", "--variant=E", "--sheet-only", "--dry"]]:
    r = subprocess.run([sys.executable, "-u"] + c, capture_output=True, text=True)
    out.append(f"===== {' '.join(c)}（exit {r.returncode}）=====\n{r.stdout[-6000:]}\n{r.stderr[-2000:]}")
subprocess.run("git reset -q --hard HEAD && git clean -fdq paper_log", shell=True)
open("check_lineE4.txt", "w").write("\n".join(out))
print("\n".join(out)[-5000:])
