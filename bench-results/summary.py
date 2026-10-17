"""Compact summary tables of bench_matsubara_selection.rs output."""
import csv
import math
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
KEY = ("kernel", "stat", "lambda", "eps", "mitigate", "positive_only", "spectrum", "sigma")
by = defaultdict(dict)
for r in rows:
    by[tuple(r[k] for k in KEY)][r["method"]] = r


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def ratios(num, den, metric, pred):
    out = []
    for key, ms in by.items():
        if num in ms and den in ms and pred(dict(zip(KEY, key))):
            a, b = float(ms[num][metric]), float(ms[den][metric])
            out.append(max(a, 1e-300) / max(b, 1e-300))
    return out


def line(num, den, pred, label):
    cells = []
    for metric in ("cond", "coef_mean", "tau_mean", "iw_mean"):
        rs = sorted(ratios(num, den, metric, pred))
        if not rs:
            return
        cells.append(f"{gm(rs):.2f} [{rs[len(rs)//10]:.2f}, {rs[(9*len(rs))//10]:.2f}]")
    print(f"| {num}/{den} | {label} | " + " | ".join(cells) + " |")


print("ratio: geometric mean [10%, 90%] over conditions")
print("| methods | subset | cond | coef err | τ err | iν err |")
print("|---|---|---|---|---|---|")
noisy = lambda k: float(k["sigma"]) > 0
clean = lambda k: float(k["sigma"]) == 0
for num, den in [("full", "id"), ("notail", "id"), ("full", "notail")]:
    line(num, den, noisy, "σ>0")
    line(num, den, clean, "σ=0")
for num in ["id", "full"]:
    for mit in ["false", "true"]:
        line(num, "sign", lambda k, m=mit: noisy(k) and k["mitigate"] == m, f"σ>0, mitigate={mit}")
        line(num, "sign", lambda k, m=mit: clean(k) and k["mitigate"] == m, f"σ=0, mitigate={mit}")

# sigma = 0 absolute errors by method, for the aliasing floor
print()
print("σ=0 coefficient error relative to ε (geometric mean over conditions)")
acc = defaultdict(list)
for r in rows:
    if float(r["sigma"]) == 0:
        acc[(r["method"], r["mitigate"])].append(float(r["coef_mean"]) / float(r["eps"]))
for k in sorted(acc):
    xs = sorted(acc[k])
    print(f"  {k}: {gm(xs):.2f} (max {xs[-1]:.1f})")
