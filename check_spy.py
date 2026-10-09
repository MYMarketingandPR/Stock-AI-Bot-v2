"""check_spy.py — 試跑 main.py --dry（唔發Telegram），核對 SPY 每日訊號用緊最新收市價。"""
import subprocess
import sys

r = subprocess.run([sys.executable, "-u", "main.py", "--dry"], capture_output=True, text=True)
open("check_spy.txt", "w").write(r.stdout[-4000:] + "\n---STDERR---\n" + r.stderr[-3000:])
print(r.stdout, r.stderr)
