"""Same-objective, same-budget baselines, cached in baselines.json (python baselines.py).

reference.json's paper_ga searches DT-FM's own communication cost model, so the evaluator's
speedup mixes "better objective" with "better search". These baselines separate the two:

  ga_sim : original DT-FM GCMA (crossover + cyclic-partitioning local search, stage order and
           matching from the paper's TSP / bottleneck matching), but every candidate is scored by
           the calibrated simulator; population 100 + 1900 trials = BUDGET simulator calls.
  greedy : no simulator calls. DP groups grown greedily from the closest GPUs (gradient-size
           transfer time), then stage order / matching from the paper's TSP / bottleneck matching.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

import reference as REF  # noqa: E402  (sets up sys.path for stepsim / topologies)
import topologies as T

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "baselines.json")
GA_POPULATION = 100


def _configure(topo):
    sys.path.insert(0, os.path.join(REF.ROOT, "k8s"))
    import run_scheduler as R
    grad_gb = REF.SS.LAYER_PARAMS * topo["layers"] * 2 / 1e9
    act_gb = REF.SS.ACT_BYTES * REF.MICRO_BATCHES * REF.GA_STEPS / 1e9
    R.configure(topo["delay_ms"], topo["bandwidth_gbps"], topo["pp"], topo["dp"], grad_gb, act_gb)
    return R


def _partition_to_rank_map(R, partition, pp_cost_fn=None):
    """Stage groups -> rank_map, with stage order and cross-stage matching from the paper's TSP."""
    partition = [tuple(p) for p in partition]
    _, path, match = (pp_cost_fn or R.S.compute_pipeline_parallel_cost)(partition)
    return R.to_rank_map(partition, path, match)


def ga_sim(topo, theta, seed=0):
    R = _configure(topo)
    S = R.S
    calls = 0
    best = (float("inf"), None)
    orig_pp = S.compute_pipeline_parallel_cost

    def sim_dp_cost(candidate_partition=None):
        # GCMA scores a candidate as dp_cost + 2 * pp_cost; put the whole simulator time in dp_cost
        nonlocal calls, best
        calls += 1
        rm = _partition_to_rank_map(R, candidate_partition, orig_pp)
        c = REF.predict(theta, topo, rm)
        best = min(best, (c, rm))
        return c

    def zero_pp_cost(candidate_partition=None):
        _, path, match = orig_pp(candidate_partition)
        return 0.0, path, match

    REF._RANDOM_SEED(seed)
    REF._NP_SEED(seed)
    orig_dp = S.compute_data_parallel_cost
    S.compute_data_parallel_cost, S.compute_pipeline_parallel_cost = sim_dp_cost, zero_pp_cost
    try:
        S.GCMA(nodes=list(range(topo["n"])), population_size=GA_POPULATION,
               trails=REF.BUDGET - GA_POPULATION, mode="default")
    finally:
        S.compute_data_parallel_cost, S.compute_pipeline_parallel_cost = orig_dp, orig_pp
    assert calls == REF.BUDGET, calls
    return best[1], calls


def greedy(topo):
    R = _configure(topo)
    n, pp, dp = topo["n"], topo["pp"], topo["dp"]
    d, b = np.array(topo["delay_ms"]), np.array(topo["bandwidth_gbps"])
    grad_bytes = REF.SS.LAYER_PARAMS * topo["layers"] * 2
    t = d / 1e3 + grad_bytes * 8 / (b * 1e9 + 1e-9)
    t = np.maximum(t, t.T)                          # symmetric: push + pull
    np.fill_diagonal(t, np.inf)
    left, groups = set(range(n)), []
    while left:
        # seed the group with the GPU whose dp-1 nearest unassigned peers are closest
        def spread(i):
            peers = sorted(left - {i}, key=lambda j: t[i, j])[:dp - 1]
            return max((t[i, j] for j in peers), default=0.0), peers
        seed = min(left, key=lambda i: spread(i)[0])
        grp = [seed] + spread(seed)[1]
        groups.append(grp)
        left -= set(grp)
    return _partition_to_rank_map(R, groups)


def baselines_for(topo):
    theta = REF.load_theta()
    ga, calls = ga_sim(topo, theta)
    gr = greedy(topo)
    out = {"ga_sim": {"rank_map": ga, "sim_s": REF.predict(theta, topo, ga), "calls": calls},
           "greedy": {"rank_map": gr, "sim_s": REF.predict(theta, topo, gr), "calls": 0}}
    print(f"{topo['name']:8s} ga_sim {out['ga_sim']['sim_s']:7.2f}s  greedy {out['greedy']['sim_s']:7.2f}s", flush=True)
    return topo["name"], out


def build():
    topos = [t for split in ("train", "heldout") for t in T.topology_set(split)[0]]
    with Pool(len(topos)) as pool:
        out = dict(pool.map(baselines_for, topos))
    json.dump(out, open(CACHE, "w"), indent=1)
    print(f"wrote {CACHE} ({len(out)} topologies)")


def load():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else None


if __name__ == "__main__":
    build()
