"""Reference placements for every topology, cached in reference.json (python3 reference.py).

  paper_ga  : original DT-FM GCMA (population 100, 4900 trials) on its own communication cost model
  random    : best of BUDGET random placements under the simulator (same budget as candidates)
  optimum   : exhaustive search under the simulator (N <= 8 only)

All placements are scored with the calibrated simulator (sim/stepsim.py + sim/theta.json).
"""
import itertools
import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

_RANDOM_SEED, _NP_SEED = random.seed, np.random.seed  # DT-FM's GCMA overwrites both with ints

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, HERE)
import stepsim as SS  # noqa: E402
import topologies as T  # noqa: E402

BUDGET = 2000                       # simulator calls allowed per topology
CACHE = os.path.join(HERE, "reference.json")
MICRO_BATCHES, GA_STEPS = 64, 2


def load_theta():
    return list(json.load(open(os.path.join(ROOT, "sim", "theta.json"))).values())


def predict(theta, topo, rank_map):
    spec = SS.RunSpec(topo["delay_ms"], topo["bandwidth_gbps"], list(rank_map), pp=topo["pp"], dp=topo["dp"],
                      micro_batches=MICRO_BATCHES, ga_steps=GA_STEPS, layers=topo["layers"])
    return SS.predict(theta, spec)


def paper_ga(topo, seed=0):
    sys.path.insert(0, os.path.join(ROOT, "k8s"))
    import run_scheduler as R
    pp, dp = topo["pp"], topo["dp"]
    grad_gb = SS.LAYER_PARAMS * topo["layers"] * 2 / 1e9              # fp16 gradients of one stage
    act_gb = SS.ACT_BYTES * MICRO_BATCHES * GA_STEPS / 1e9            # activations per iteration
    R.configure(topo["delay_ms"], topo["bandwidth_gbps"], pp, dp, grad_gb, act_gb)
    _RANDOM_SEED(seed)
    _NP_SEED(seed)
    parts, scores, _ = R.S.GCMA(nodes=list(range(topo["n"])), population_size=100, trails=4900, mode="default")
    best = parts[int(np.argmin(scores))]
    partition = [tuple(best[i:i + dp]) for i in range(0, topo["n"], dp)]
    _, _, _, path, match = R.cost_of(partition)
    return R.to_rank_map(partition, path, match)


def reference_for(topo):
    theta = load_theta()
    n = topo["n"]
    out = {}
    ga = paper_ga(topo)
    out["paper_ga"] = {"rank_map": ga, "sim_s": predict(theta, topo, ga)}
    rng = np.random.default_rng(12345)
    best = min((predict(theta, topo, rm), rm) for rm in (rng.permutation(n).tolist() for _ in range(BUDGET)))
    out["random"] = {"rank_map": best[1], "sim_s": best[0]}
    if n <= 8:
        opt = min((predict(theta, topo, p), list(p)) for p in itertools.permutations(range(n)))
        out["optimum"] = {"rank_map": opt[1], "sim_s": opt[0]}
    print(f"{topo['name']:8s} paper_ga {out['paper_ga']['sim_s']:7.2f}s  random {out['random']['sim_s']:7.2f}s"
          + (f"  optimum {out['optimum']['sim_s']:7.2f}s" if "optimum" in out else ""), flush=True)
    return topo["name"], out


def build():
    topos = [t for split in ("train", "heldout") for t in T.topology_set(split)[0]]
    with Pool(len(topos)) as pool:
        ref = dict(pool.map(reference_for, topos))
    json.dump(ref, open(CACHE, "w"), indent=1)
    print(f"wrote {CACHE} ({len(ref)} topologies)")


def load():
    if not os.path.exists(CACHE):
        build()
    return json.load(open(CACHE))


if __name__ == "__main__":
    build()
