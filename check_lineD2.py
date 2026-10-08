"""check_lineD2.py — 試跑線D2（--dry，唔發Telegram、唔寫Sheet、唔推紀錄），核對程式冇錯。"""
import json
import subprocess
import sys

subprocess.run("git fetch origin paper-trading && git checkout origin/paper-trading -- paper_log", shell=True, check=True)
st = json.load(open("paper_log/lineD2_state.json"))
st["last_date"] = "2026-10-06"          # 只為試跑：當 10-07 未做，重做一次
json.dump(st, open("paper_log/lineD2_state.json", "w"))
r = subprocess.run([sys.executable, "-u", "line_d.py", "--variant=D2", "--dry"], capture_output=True, text=True)
open("check_lineD2.txt", "w").write(r.stdout[-6000:] + "\n---STDERR---\n" + r.stderr[-4000:])
print(r.stdout[-6000:], r.stderr[-4000:])
