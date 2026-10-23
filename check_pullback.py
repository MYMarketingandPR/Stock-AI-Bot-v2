"""一次性檢查：pullback_scan.py --dry（唔發Telegram、唔寫紀錄），將訊息存落 check_pullback.txt。"""
import contextlib
import io
import sys

sys.argv = ["pullback_scan.py", "--dry"]
import pullback_scan as P  # noqa: E402

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try:
        P.main()
    except Exception:
        import traceback
        traceback.print_exc(file=buf)
open("check_pullback.txt", "w").write(buf.getvalue())
