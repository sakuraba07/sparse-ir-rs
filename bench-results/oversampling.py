"""Effect of mitigate (oversampling) per criterion: mitigated / plain ratios.

For the ID, `id` with mitigate uses the D-optimal oversampling, `id-A` the
A-optimal one and `id-E` the first-order E-optimal one; all are compared
with `id` without mitigate. `sign` (fencing) is the reference.
"""
import csv
import math
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
KEY = ("kernel", "stat", "lambda", "eps", "positive_only", "spectrum", "sigma")
by = defaultdict(dict)
for r in rows:
    tag = r["method"] + ("+m" if r["mitigate"] == "true" else "")
    by[tuple(r[k] for k in KEY)][tag] = r


def gm(xs):
    return math.exp(sum(math.log(max(x, 1e-300)) for x in xs) / len(xs))


METRICS = ("cond", "coef_mean", "tau_mean", "iw_mean", "npts")
print("| mitigated / plain | subset | cond | coef err | τ err | iν err | points |")
print("|---|---|---|---|---|---|---|")
for num, den in [("sign+m", "sign"), ("id+m", "id"), ("id-A+m", "id"), ("id-E+m", "id"),
                 ("id-A+m", "sign+m"), ("id-E+m", "sign+m")]:
    for label, pred in [("σ>0", lambda s: float(s) > 0), ("σ=0", lambda s: float(s) == 0)]:
        out = [[float(v[num][m]) / max(float(v[den][m]), 1e-300) for m in METRICS]
               for k, v in by.items() if num in v and den in v and pred(k[6])]
        if out:
            cells = " | ".join(f"{gm([o[i] for o in out]):.2f}" for i in range(len(METRICS)))
            print(f"| {num} / {den} | {label} | {cells} |")
