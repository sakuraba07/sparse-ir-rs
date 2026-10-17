"""Summary of bench_matsubara_selection.rs output and the pre-registered criterion.

Criterion (fixed before the run): keep the exchange step (method `full`)
only if, against `id`,
  (a) in >= 70% of the conditions with sigma > 0 the error ratio is <= 0.8, and
  (b) in no condition (sigma = 0 included) the ratio exceeds 1.25,
where a condition is (kernel, stat, lambda, eps, mitigate, positive_only,
spectrum, sigma) and the error ratio is the geometric mean of the ratios of
the mean coefficient error, the mean tau reconstruction error and the mean
Matsubara reconstruction error.
"""
import csv
import math
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
KEY = ("kernel", "stat", "lambda", "eps", "mitigate", "positive_only", "spectrum", "sigma")
METRICS = ("coef_mean", "tau_mean", "iw_mean")
by = defaultdict(dict)
for r in rows:
    by[tuple(r[k] for k in KEY)][r["method"]] = r


def ratio(a, b):
    """Geometric mean of the metric ratios a/b (floored at 1e-300)."""
    logs = [math.log(max(float(a[m]), 1e-300) / max(float(b[m]), 1e-300)) for m in METRICS]
    return math.exp(sum(logs) / len(logs))


def compare(num, den, label):
    noisy, better, worst = 0, 0, []
    for key, ms in by.items():
        if num not in ms or den not in ms:
            continue
        r = ratio(ms[num], ms[den])
        if float(key[7]) > 0:
            noisy += 1
            better += r <= 0.8
        worst.append((r, key))
    worst.sort(reverse=True)
    n_bad = sum(r > 1.25 for r, _ in worst)
    ratios = sorted(r for r, _ in worst)
    med = ratios[len(ratios) // 2] if ratios else float("nan")
    print(f"== {label}: {num} / {den}")
    print(f"  conditions {len(worst)}, sigma>0: {noisy}, ratio<=0.8: {better} "
          f"({100 * better / max(noisy, 1):.0f}%), ratio>1.25: {n_bad}, median ratio {med:.3f}")
    for r, key in worst[:8]:
        print(f"  worst {r:.3f} {dict(zip(KEY, key))}")
    for r, key in sorted(worst)[:3]:
        print(f"  best  {r:.3f} {dict(zip(KEY, key))}")
    return better >= 0.7 * noisy and n_bad == 0


def cond_table():
    """Median condition number per (kernel family, method, mitigate, positive_only)."""
    acc = defaultdict(list)
    seen = set()
    for r in rows:
        k = (r["kernel"], r["stat"], r["lambda"], r["eps"], r["method"], r["mitigate"], r["positive_only"])
        if k in seen:
            continue
        seen.add(k)
        fam = "logistic" if r["kernel"] == "logistic" else "windowed"
        acc[(fam, r["method"], r["mitigate"], r["positive_only"])].append(
            (float(r["cond"]), int(r["npts"]), int(r["L"])))
    print("== condition numbers (median / max over bases), points / L")
    for k in sorted(acc):
        cs = sorted(c for c, _, _ in acc[k])
        pts = sum(p / l for _, p, l in acc[k]) / len(acc[k])
        print(f"  {k}: cond median {cs[len(cs)//2]:.3g} max {cs[-1]:.3g}, points/L {pts:.2f}")


cond_table()
keep = compare("full", "id", "criterion")
print(f"CRITERION: keep exchange step = {keep}")
compare("notail", "id", "ablation")
compare("full", "notail", "tail effect")
compare("full", "sign", "new vs sign changes")
compare("id", "sign", "ID vs sign changes")
