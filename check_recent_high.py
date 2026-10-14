"""一次性檢查：line_d.recent_high 近期高位。比較 1 個月（20日）同 2 個月（42日，現用預設）。"""
import inspect
import line_d as L
out = [f"預設日數 n = {inspect.signature(L.recent_high).parameters['n'].default}"]
for t in ["MRVL", "MU", "MRNA", "DELL", "BE", "SNDK", "LITE"]:
    out.append(f"{t}\t20日 {L.recent_high(t, '2026-10-14', 20)}\t預設 {L.recent_high(t, '2026-10-14')}")
open("recent_high_check.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
