"""Compare entry-pivot vs norm-pivot runs of examples/pivot_exp.rs.

usage: python3 pivot_analyze.py OUTDIR   (expects entry.txt/norm.txt/.err)
"""
import sys
from pathlib import Path


def load(p):
    d = {}
    for line in Path(p).read_text().splitlines():
        parts = line.split("\t")
        key = tuple(parts[:4])
        vals = [float(x) for x in parts[4].split()] if len(parts) > 4 and parts[4] else []
        d[key] = vals
    return d


def stops(p):
    out = []
    for line in Path(p).read_text().splitlines():
        if line.startswith("CPQR"):
            f = dict(kv.split("=") for kv in line.split()[1:])
            out.append((int(f["m"]), int(f["n"]), int(f["stop"])))
    return out


def main(outdir):
    od = Path(outdir)
    e, n = load(od / "entry.txt"), load(od / "norm.txt")
    refs = {(k[1], k[2]): v for k, v in e.items() if k[0] == "ref"}
    refs_n = {(k[1], k[2]): v for k, v in n.items() if k[0] == "ref"}
    same_ref = all(refs[k] == refs_n[k] for k in refs)
    print(f"Df64 reference identical in both runs: {same_ref}")
    print()
    hdr = f"{'path':10} {'kernel':9} {'lam':6} {'eps':6} {'n_e':>4} {'n_n':>4} {'max|de-dn|/s0':>14} {'err_e':>9} {'err_n':>9} {'err_e(>1e-10)':>13} {'err_n(>1e-10)':>13}"
    print(hdr)
    ndiff = 0
    nkeys = 0
    for key in e:
        path, kern, lam, eps = key
        if path in ("ref", "synthetic") or path.startswith("tau"):
            continue
        ve, vn = e[key], n[key]
        nkeys += 1
        m = min(len(ve), len(vn))
        s0 = ve[0]
        dmax = max(abs(ve[i] - vn[i]) / s0 for i in range(m)) if m else 0.0
        ref = refs.get((kern, lam))

        def err(v, floor):
            if ref is None:
                return float("nan")
            ks = [i for i in range(min(len(v), len(ref))) if ref[i] > floor * ref[0]]
            return max((abs(v[i] - ref[i]) / ref[0] for i in ks), default=0.0)

        if len(ve) != len(vn):
            ndiff += 1
        print(
            f"{path:10} {kern:9} {lam:6} {eps:6} {len(ve):4d} {len(vn):4d} {dmax:14.2e} "
            f"{err(ve, 0):9.2e} {err(vn, 0):9.2e} {err(ve, 1e-10):13.2e} {err(vn, 1e-10):13.2e}"
        )
    print(f"\ncases with different count: {ndiff}/{nkeys}")
    print("\ntau points (max |tau_e - tau_n| / beta, count):")
    for key in e:
        if key[0].startswith("tau"):
            ve, vn = e[key], n[key]
            beta = float(key[2])
            if len(ve) != len(vn):
                print(f"  {key}: COUNT DIFFERS {len(ve)} vs {len(vn)}")
            else:
                d = max(abs(a - b) for a, b in zip(ve, vn)) / beta
                print(f"  {key[0]} {key[1]} {key[2]} {key[3]}: n={len(ve)} maxdiff={d:.2e}")
    print("\nsynthetic singular values:")
    print("  entry:", e[("synthetic", "-", "-", "-")])
    print("  norm: ", n[("synthetic", "-", "-", "-")])

    se, sn = stops(od / "entry.err"), stops(od / "norm.err")
    print(f"\nCPQR calls: entry {len(se)}, norm {len(sn)}")
    if len(se) == len(sn):
        diffs = [(a, b) for a, b in zip(se, sn) if a[2] != b[2]]
        print(f"CPQR calls with a different stop index: {len(diffs)}")
        hist = {}
        for a, b in diffs:
            hist[b[2] - a[2]] = hist.get(b[2] - a[2], 0) + 1
        print("  (norm stop - entry stop): count =", dict(sorted(hist.items())))


if __name__ == "__main__":
    main(sys.argv[1])
