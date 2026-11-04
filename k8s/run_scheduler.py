"""Run the DT-FM scheduler (GA + local search) on an N=8 topology and emit a rank_map.

The original scheduler.py is hard-wired to 64 GPUs (PP8 x DP8) through module globals.
Instead of editing it, this driver imports it and overrides those globals, so the cost
model and search code stay byte-identical to the paper's implementation.

Usage:
  python k8s/run_scheduler.py --config k8s/run_config.case3.json --pp 4 --dp 2 \
      --grad-gb 0.60 --act-gb 1.0 [--brute-force] [--random N] [--out k8s/run_config.case3_sched.json]

rank_map semantics (same as DT-FM): rank_map[pod] = rank, stage = rank % PP, dp group = rank // PP.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "DT-FM", "scheduler", "heuristic_evolutionary_solver"))
import scheduler as S  # noqa: E402  (imports with its 64-GPU defaults; overridden below)


def configure(delay, bandwidth, pp, dp, grad_gb, act_gb):
    S.num_devices = pp * dp
    S.way = pp                 # number of pipeline stages (partitions)
    S.partition_size = dp      # GPUs per stage (= DP degree)
    S.peer_delay = np.array(delay, dtype=float)
    S.peer_bandwidth = np.array(bandwidth, dtype=float)
    S.send_gradient_size = grad_gb
    S.send_activation_size = act_gb
    S.regions = None


def cost_of(partition):
    """partition: list of `way` tuples of node ids. Returns (total, dp, pp, path, match)."""
    dp_cost = S.compute_data_parallel_cost(candidate_partition=partition)
    pp_cost, path, match = S.compute_pipeline_parallel_cost(partition)
    return dp_cost + 2 * pp_cost, dp_cost, 2 * pp_cost, path, match


def to_rank_map(partition, path, match):
    pipeline = S.get_pipelines(partition, path, match)
    rank_map = [0] * S.num_devices
    for dp_idx in range(S.partition_size):
        for stage in range(S.way):
            rank_map[pipeline[stage, dp_idx]] = dp_idx * S.way + stage
    assert sorted(rank_map) == list(range(S.num_devices))
    return rank_map


def cost_of_rank_map(rank_map):
    """Evaluate an arbitrary rank_map with the same cost model (stage groups = partitions)."""
    partition = [tuple(pod for pod, r in enumerate(rank_map) if r % S.way == stage) for stage in range(S.way)]
    return cost_of(partition)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="run_config.json holding delay_ms / bandwidth_gbps")
    ap.add_argument("--pp", type=int, default=4)
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--grad-gb", type=float, required=True, help="gradient bytes synced per stage per iteration (GB)")
    ap.add_argument("--act-gb", type=float, required=True, help="activation bytes sent between stages per iteration (GB)")
    ap.add_argument("--population", type=int, default=100)
    ap.add_argument("--trials", type=int, default=4900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--brute-force", action="store_true", help="also enumerate every partition (N=8 only)")
    ap.add_argument("--random", type=int, default=0, help="also score this many random rank_maps")
    ap.add_argument("--out", help="write a run_config with the scheduled rank_map here")
    ap.add_argument("--tag", help="tag for --out (default: <config tag>_sched)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    configure(cfg["delay_ms"], cfg["bandwidth_gbps"], args.pp, args.dp, args.grad_gb, args.act_gb)
    random.seed(args.seed)
    np.random.seed(args.seed)
    n = S.num_devices

    print(f"identity rank_map cost: {cost_of_rank_map(list(range(n))):.4f} s")

    t0 = time.perf_counter()
    parts, scores, _ = S.GCMA(nodes=list(range(n)), population_size=args.population, trails=args.trials, mode="default")
    best = parts[int(np.argmin(scores))]
    partition = [tuple(best[i:i + S.partition_size]) for i in range(0, n, S.partition_size)]
    total, dpc, ppc, path, match = cost_of(partition)
    rank_map = to_rank_map(partition, path, match)
    print(f"GA: {time.perf_counter() - t0:.1f}s, {len(scores)} candidates evaluated")
    print(f"GA best cost: {total:.4f} s (dp {dpc:.4f} + pp {ppc:.4f})")
    print(f"GA partition (stage groups): {partition}, stage order: {path}")
    print(f"GA rank_map: {rank_map}")

    if args.brute_force:
        t0 = time.perf_counter()
        best_bf = (float("inf"), None)
        seen = set()
        for cand in S.all_candidate_partitions(list(range(n))):
            key = frozenset(frozenset(p) for p in cand)
            if key in seen:
                continue
            seen.add(key)
            c = cost_of(cand)[0]
            if c < best_bf[0]:
                best_bf = (c, cand)
        print(f"brute force: {len(seen)} distinct partitions in {time.perf_counter() - t0:.1f}s, "
              f"optimum cost: {best_bf[0]:.4f} s, partition: {best_bf[1]}")

    if args.random:
        costs = []
        for _ in range(args.random):
            rm = list(range(n))
            random.shuffle(rm)
            costs.append(cost_of_rank_map(rm))
        print(f"random rank_maps (n={args.random}): mean {np.mean(costs):.4f}, min {np.min(costs):.4f}, "
              f"max {np.max(costs):.4f} s")

    if args.out:
        out = dict(cfg)
        out["tag"] = args.tag or (cfg["tag"].replace("_identity", "") + "_sched")
        out["rank_map"] = rank_map
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out} (tag={out['tag']})")


if __name__ == "__main__":
    main()
