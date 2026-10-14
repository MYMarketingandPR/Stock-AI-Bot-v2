"""一次性檢查：line_d.recent_high 攞到嘅近期高位（MRVL 應該係 10-06 市中 $301.20 左右）。"""
import line_d as L
out = []
for t in ["MRVL", "MU", "MRNA", "DELL", "BE", "SNDK", "LITE"]:
    for d in ["2026-10-13", "2026-10-14"]:
        out.append(f"{t}\t{d}\t{L.recent_high(t, d)}")
open("recent_high_check.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
