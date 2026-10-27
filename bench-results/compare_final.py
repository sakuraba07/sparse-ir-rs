"""Compare the final pair selection (`final`) with `id` / `id-A` of the
benchmark branch, row by row, and summarize final / sign."""
import csv
import math
import sys
from collections import defaultdict

old = list(csv.DictReader(open(sys.argv[1])))
new = list(csv.DictReader(open(sys.argv[2])))
KEY = ("kernel", "stat", "lambda", "eps", "mitigate", "positive_only", "spectrum", "sigma")
METRICS = ("npts", "cond", "coef_mean", "tau_mean", "iw_mean")
ref = {}
for r in old:
    m = r["method"]
    if (m == "id" and r["mitigate"] == "false") or (m == "id-A" and r["mitigate"] == "true"):
        ref[tuple(r[k] for k in KEY)] = r
same = diff = 0
worst = []
for r in new:
    if r["method"] != "final":
        continue
    k = tuple(r[x] for x in KEY)
    o = ref[k]
    if all(r[m] == o[m] for m in METRICS):
        same += 1
    else:
        diff += 1
        ratio = max(abs(math.log(max(float(r[m]), 1e-300) / max(float(o[m]), 1e-300))) for m in METRICS[1:])
        worst.append((ratio, r["npts"], o["npts"], k))
print(f"final vs id/id-A: identical rows {same}, differing {diff}")
worst.sort(reverse=True)
for w in worst[:8]:
    print(f"  max |log ratio| {w[0]:.3f} npts {w[1]} vs {w[2]} {w[3]}")

by = defaultdict(dict)
for r in new:
    by[tuple(r[k] for k in KEY)][r["method"]] = r


def gm(xs):
    return math.exp(sum(math.log(max(x, 1e-300)) for x in xs) / len(xs))


print("final / sign (centrosymmetric):")
for mit in ("false", "true"):
    for label, pred in (("σ>0", lambda s: float(s) > 0), ("σ=0", lambda s: float(s) == 0)):
        out = [[float(v["final"][m]) / max(float(v["sign"][m]), 1e-300) for m in METRICS[1:]]
               for k, v in by.items() if "sign" in v and k[4] == mit and pred(k[7])]
        cells = " ".join(f"{gm([o[i] for o in out]):.2f}" for i in range(4))
        print(f"  mitigate={mit} {label}: cond coef tau iw = {cells}")
