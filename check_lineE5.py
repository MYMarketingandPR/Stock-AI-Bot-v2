"""一次性檢查：用 --dry 跑線E5（唔發Telegram、唔寫Sheet、唔改真紀錄）。"""
import subprocess
import sys

out = []
r = subprocess.run("git fetch origin paper-trading && git checkout FETCH_HEAD -- paper_log", shell=True, capture_output=True, text=True)
out.append(f"取紀錄：{r.returncode}")
r = subprocess.run([sys.executable, "-u", "line_d.py", "--variant=E5", "--dry"], capture_output=True, text=True)
out.append(f"===== E5（exit {r.returncode}）=====\n{r.stdout[-6000:]}\n{r.stderr[-2000:]}")
subprocess.run("git reset -q --hard HEAD && git clean -fdq paper_log", shell=True)
open("check_lineE5.txt", "w").write("\n".join(out))
print("\n".join(out)[-5000:])
