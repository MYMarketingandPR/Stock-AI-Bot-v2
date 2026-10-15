"""一次性檢查：用 --dry 跑線E2（唔發Telegram、唔寫Sheet、唔改真紀錄），睇行業過濾有冇正常運作。"""
import contextlib
import io
import os
import sys

sys.argv = ["line_d.py", "--variant=E2", "--dry"]
os.makedirs("paper_log", exist_ok=True)
import line_d as L  # noqa: E402

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try:
        L.main()
    except Exception:
        import traceback
        traceback.print_exc(file=buf)
out = buf.getvalue()
smap = L.sector_map()
out += f"\n\n--- 行業分類數目：{len(smap)}；例子：" + ", ".join(f"{k}={smap[k]}" for k in list(smap)[:5])
open("check_lineE2.txt", "w").write(out)
print(out[-4000:])
