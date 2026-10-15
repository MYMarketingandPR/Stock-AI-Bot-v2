"""一次性檢查：用 --dry 跑線E3（唔發Telegram、唔寫Sheet、唔改真紀錄）。"""
import contextlib
import io
import os
import sys

sys.argv = ["line_d.py", "--variant=E3", "--dry"]
os.makedirs("paper_log", exist_ok=True)
import line_d as L  # noqa: E402

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try:
        L.main()
    except Exception:
        import traceback
        traceback.print_exc(file=buf)
open("check_lineE3.txt", "w").write(buf.getvalue())
print(buf.getvalue()[-3000:])
