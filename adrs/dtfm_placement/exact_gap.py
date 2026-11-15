"""Gap between the evolved programs and the exact simulator optimum on the generalization topologies.

  skydiscover/.venv/bin/python adrs/dtfm_placement/exact_gap.py [--workers 10]

Uses the MILP staircase solver exact_milp.py (docs/milp-exact-optimum.md; the solver checks its model value
against REF.predict). Topologies: every entry of the generalization results (default output/generalization.json:
N = 16, 32, 64, 128 x seeds 5000-5004), so the program results there are compared directly. The results file must
come from the same simulator version as the current sim/. Output: --out (default output/exact_gap.json) + a table.
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import topologies as T  # noqa: E402

T.PARALLELISM.setdefault(128, (8, 16))


def one(name):
    os.environ["OMP_NUM_THREADS"] = "1"
    import exact_milp
    n, seed = name[1:].split("_s")
    r = exact_milp.solve(int(seed), int(n))
    return name, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--gen", default=os.path.join(HERE, "output", "generalization.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "output", "exact_gap.json"))
    a = ap.parse_args()
    gen = json.load(open(a.gen))
    names = sorted(gen, key=lambda k: (-int(k[1:].split("_s")[0]), k))
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(one, names))
    json.dump(res, open(a.out, "w"), indent=1)
    progs = [k for k in gen[names[0]] if k != "dtfm"]
    print(f"{'topology':12s} {'k':>2s} {'opt':>7s} {'model-sim':>9s} {'TO':>2s} " + " ".join(f"{p:>9s}" for p in progs))
    gaps = {p: {} for p in progs}
    for nm in sorted(names, key=lambda k: (int(k[1:].split("_s")[0]), k)):
        r = res[nm]
        row = []
        for p in progs:
            s = gen[nm][p]["sim_s"]
            g = s / r["opt"] if s else float("nan")
            gaps[p].setdefault(int(nm[1:].split("_s")[0]), []).append(g)
            row.append(f"{g:9.4f}")
        print(f"{nm:12s} {r['k']:2d} {r['opt']:7.2f} {r['model'] - r['opt']:9.1e} {r['timeouts']:2d} " + " ".join(row))
    print("\ngeomean sim(program) / sim(exact optimum), per N  (1.0 = optimal)")
    gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))  # noqa: E731
    for p in progs:
        print(f"  {p:10s} " + "  ".join(f"N{n}: {gm([x for x in v if x == x]):.4f}" for n, v in sorted(gaps[p].items())))


if __name__ == "__main__":
    main()
