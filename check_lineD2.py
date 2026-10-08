"""check_lineD2.py — 試跑線D、線D2、線E（--dry，唔發Telegram、唔寫Sheet、唔推紀錄），核對程式冇錯。"""
import json
import subprocess
import sys

subprocess.run("git fetch origin paper-trading && git checkout origin/paper-trading -- paper_log", shell=True, check=True)
for f in ("paper_log/lineD_state.json", "paper_log/lineD2_state.json"):
    st = json.load(open(f))
    st["last_date"] = "2026-10-06"          # 只為試跑：當 10-07 未做，重做一次
    json.dump(st, open(f, "w"))
out = []
for v in ("--variant=D2", "--variant=E"):
    r = subprocess.run([sys.executable, "-u", "line_d.py", v, "--dry"], capture_output=True, text=True)
    out.append(f"===== {v} =====\n" + r.stdout[-5000:] + "\n---STDERR---\n" + r.stderr[-3000:])
open("check_lineD2.txt", "w").write("\n".join(out))
print("\n".join(out))
