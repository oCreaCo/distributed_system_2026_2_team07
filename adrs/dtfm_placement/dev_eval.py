"""Dev-set evaluator for N=64 program work: fresh topologies, compared with a fixed baseline program.

  skydiscover/.venv/bin/python adrs/dtfm_placement/dev_eval.py <program.py> [--n 64] [--seeds 6000-6019]

Train (seeds 0-2) is saturated (~1.277, see docs/worklog/2026-11-15_0340_*), so N=64 work is judged on a dev set:
seeds 6000-6019 by default (disjoint from train 0-2, held-out 100-102, the agent's study range 1000-4999 and the
generalization test 5000-5004). Score = geomean over the dev set of sim(baseline) / sim(candidate) (> 1: better than
the baseline). The baseline (output/agent_cc/best_program.py) is run once and cached. Same contract as the official
evaluator: 2000 cost_fn calls, 30 s per topology (a timeout or invalid output counts as ratio 0.1).
"""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import generalization as G
import reference as REF
import topologies as T

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "output", "agent_cc", "best_program.py")
CACHE = os.path.join(HERE, "output", "dev_eval_baseline.json")


def one(args):
    path, n, seed = args
    os.environ["OMP_NUM_THREADS"] = "1"
    topo = T.make_topology(n, seed)
    r = G.run_program(path, topo, REF.load_theta(), 30)
    return f"N{n}_s{seed}", r


def run_all(path, n, seeds, workers):
    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(one, [(path, n, s) for s in seeds]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("program")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seeds", default="6000-6019")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    key = f"N{a.n}:{a.seeds}"
    if key not in cache:
        cache[key] = run_all(BASELINE, a.n, seeds, a.workers)
        json.dump(cache, open(CACHE, "w"), indent=1, default=lambda o: o.item())
    base = cache[key]
    res = run_all(os.path.abspath(a.program), a.n, seeds, a.workers)
    ratios = []
    for name in sorted(res, key=lambda k: int(k.split("_s")[1])):
        b, c = base[name], res[name]
        ratio = b["sim_s"] / c["sim_s"] if c["sim_s"] else 0.1
        ratios.append(ratio)
        flag = "" if c["sim_s"] else f"  FAILED ({c['error']})"
        print(f"{name:10s} baseline {b['sim_s']:7.2f}  candidate {c['sim_s'] or float('nan'):7.2f}  ratio {ratio:6.3f}  "
              f"time {c['time_s']:5.1f}s  calls {c['calls']}{flag}")
    gm = math.exp(sum(map(math.log, ratios)) / len(ratios))
    print(json.dumps({"dev_score_vs_baseline": round(float(gm), 5), "better": int(sum(r > 1.0005 for r in ratios)),
                      "worse": int(sum(r < 0.9995 for r in ratios)), "n_topologies": len(ratios),
                      "max_time_s": max(r["time_s"] for r in res.values())}))


if __name__ == "__main__":
    main()
