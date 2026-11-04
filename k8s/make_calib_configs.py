"""Generate the calibration run batches (k8s/calib/job*.json) and a manifest.

Topologies (N=8 pods):
  T1 two_regions  : pods 0-3 / 4-7 (DT-FM case 3: intra 1 ms / 5 Gbit, inter 11 ms / 1.12 Gbit)
  T2 four_regions : pairs (0,1)(2,3)(4,5)(6,7) in Virginia/Oregon/California/Ohio (DT-FM case 4 values)
  T3 worldwide    : one pod per city, values from DT-FM config.delay_bandwidth_dict
  homogeneous     : every link the same delay/bandwidth (sensitivity sweeps)
Placements per topology: identity, dplocal (fastest links as DP pairs), ga (paper cost model optimum),
worst (worst of 1000 random by paper model), rnd0, rnd1.
"""
import json
import os
import random
import sys

import numpy as np

# DT-FM's GCMA does `random.seed = i` / `np.random.seed = i` (assigns ints over the functions),
# so keep the real seed functions before the scheduler is imported.
_RANDOM_SEED, _NP_SEED = random.seed, np.random.seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "k8s"))
sys.path.insert(0, os.path.join(ROOT, "DT-FM", "scheduler"))
import run_scheduler as R  # noqa: E402
import generate_sim_com_matrices as G  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "DT-FM", "scheduler", "heuristic_evolutionary_solver"))
import config as C  # noqa: E402

N, PP, DP = 8, 4, 2
GRAD_GB, ACT_GB = 0.6, 1.0
OUT = os.path.join(ROOT, "k8s", "calib")

COMMON_ARGS = ["--pp-mode", "gpipe", "--dp-mode", "central_ps",
               "--pipeline-group-size", str(PP), "--data-group-size", str(DP),
               "--embedding-dim", "2048", "--num-heads", "16", "--fp16",
               "--num-layers", "6", "--batch-size", "64", "--micro-batch-size", "1",
               "--gradient-accumulate-step", "2", "--num-iters", "6",
               "--train-data", "./task_datasets/data/QQP/train_small.tsv"]


def homogeneous(delay, bw):
    d = np.full((N, N), float(delay)); b = np.full((N, N), float(bw))
    return d.tolist(), b.tolist()


def worldwide():
    cities = ["Oregon", "Virginia", "Ohio", "Tokyo", "Seoul", "London", "Frankfurt", "Ireland"]
    d = np.zeros((N, N)); b = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                d[i][j], b[i][j] = 5, 2  # diagonal is never a real link; paper's within-region default keeps costs finite
                continue
            key = f"{cities[i]}-{cities[j]}"
            key2 = f"{cities[j]}-{cities[i]}"
            dl, bw = C.delay_bandwidth_dict.get(key) or C.delay_bandwidth_dict[key2]
            d[i][j], b[i][j] = dl, bw
    return d.tolist(), b.tolist()


def topologies():
    d1, b1, _ = G.simulate_3_multi_universities(N)
    d2, b2, _ = G.simulate_4_regional_geo_distributed(N)
    d3, b3 = worldwide()
    return {"T1": (d1.tolist(), b1.tolist()), "T2": (d2.tolist(), b2.tolist()), "T3": (d3, b3)}


def dplocal_map(delay, bw):
    """Greedy: pair up pods along the cheapest links (DP pairs), one pair per stage."""
    cost = lambda i, j: delay[i][j] / 1e3 + GRAD_GB * 8 / bw[i][j]
    pairs = sorted(((cost(i, j), i, j) for i in range(N) for j in range(i + 1, N)))
    used, groups = set(), []
    for _, i, j in pairs:
        if i not in used and j not in used:
            groups.append((i, j)); used |= {i, j}
    rank_map = [0] * N
    for stage, (i, j) in enumerate(groups):
        rank_map[i] = stage; rank_map[j] = PP + stage
    return rank_map


def placements(delay, bw, seed):
    R.configure(delay, bw, PP, DP, GRAD_GB, ACT_GB)
    _RANDOM_SEED(seed); _NP_SEED(seed)
    parts, scores, _ = R.S.GCMA(nodes=list(range(N)), population_size=100, trails=2000, mode="default")
    best = parts[int(np.argmin(scores))]
    partition = [tuple(best[i:i + DP]) for i in range(0, N, DP)]
    _, _, _, path, match = R.cost_of(partition)
    ga = R.to_rank_map(partition, path, match)
    samples = []
    for _ in range(1000):
        rm = list(range(N)); random.shuffle(rm)
        samples.append((R.cost_of_rank_map(rm), rm))
    samples.sort()
    out = {"identity": list(range(N)), "dplocal": dplocal_map(delay, bw), "ga": ga, "worst": samples[-1][1]}
    # random placements from the 25th / 50th cost percentiles, skipping maps already in the set
    for name, q in (("rnd0", 0.25), ("rnd1", 0.5)):
        k = int(q * len(samples))
        while samples[k][1] in out.values():
            k += 1
        out[name] = samples[k][1]
    return {k: (v, R.cost_of_rank_map(v)) for k, v in out.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    topo = topologies()
    manifest = {}
    jobs = {1: [], 2: [], 3: []}

    def add(job, tag, d, b, rank_map=None, args=None, kind="train", **meta):
        r = {"tag": tag, "rank_map": rank_map, "delay_ms": d, "bandwidth_gbps": b, "args": args or []}
        if kind == "bench":
            r["type"] = "bench"
        jobs[job].append(r)
        manifest[tag] = dict(kind=kind, rank_map=rank_map, args=args or [], **meta)

    # --- Job 1: bench + homogeneous sweeps + T1 placements
    add(1, "bench_notc", None, None, kind="bench", topology="notc")
    for dl in (0, 10, 50, 100):
        d, b = homogeneous(dl, 1.0)
        add(1, f"homo_d{dl}_b1", d, b, topology="homo", delay=dl, bw=1.0, placement="identity")
    for bw in (0.3, 3.0):
        d, b = homogeneous(10, bw)
        add(1, f"homo_d10_b{bw}", d, b, topology="homo", delay=10, bw=bw, placement="identity")
    d, b = topo["T1"]
    add(1, "bench_T1", d, b, kind="bench", topology="T1")
    for name, (rm, cost) in placements(d, b, seed=1).items():
        add(1, f"T1_{name}", d, b, rm, topology="T1", placement=name, paper_cost=cost)

    # --- Job 2: T2 and T3 placements (+ bench each)
    for t in ("T2", "T3"):
        d, b = topo[t]
        add(2, f"bench_{t}", d, b, kind="bench", topology=t)
        for name, (rm, cost) in placements(d, b, seed=2).items():
            add(2, f"{t}_{name}", d, b, rm, topology=t, placement=name, paper_cost=cost)

    # --- Job 3: model-parameter variations (T1 identity / no tc) + repeats
    d, b = topo["T1"]
    dpl = manifest["T1_dplocal"]["rank_map"]
    add(3, "T1_identity_nooffload", d, b, None, ["--use-offload", "false"], topology="T1", placement="identity", variant="nooffload")
    add(3, "T1_dplocal_nooffload", d, b, dpl, ["--use-offload", "false"], topology="T1", placement="dplocal", variant="nooffload")
    add(3, "notc_nooffload", None, None, None, ["--use-offload", "false"], topology="notc", placement="identity", variant="nooffload")
    add(3, "T1_identity_b32", d, b, None, ["--batch-size", "32"], topology="T1", placement="identity", variant="b32")
    add(3, "T1_identity_l3", d, b, None, ["--num-layers", "3"], topology="T1", placement="identity", variant="l3")
    add(3, "notc_l3", None, None, None, ["--num-layers", "3"], topology="notc", placement="identity", variant="l3")
    add(3, "notc_rep1", None, None, None, topology="notc", placement="identity", rep=1)
    add(3, "T1_identity_rep1", d, b, None, topology="T1", placement="identity", rep=1)
    add(3, "T1_dplocal_rep1", d, b, dpl, topology="T1", placement="dplocal", rep=1)
    for t in ("T2", "T3"):
        add(3, f"{t}_ga_rep1", *topo[t], manifest[f"{t}_ga"]["rank_map"], topology=t, placement="ga", rep=1)

    for j, runs in jobs.items():
        json.dump({"common_args": COMMON_ARGS, "runs": runs}, open(os.path.join(OUT, f"job{j}.json"), "w"), indent=1)
        print(f"job{j}: {len(runs)} runs:", ", ".join(r["tag"] for r in runs))
    json.dump({"common_args": COMMON_ARGS, "topologies": {k: {"delay_ms": v[0], "bandwidth_gbps": v[1]} for k, v in topo.items()},
               "runs": manifest}, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    for tag, m in manifest.items():
        if "paper_cost" in m:
            print(f"  {tag:14s} rank_map={m['rank_map']} paper_cost={m['paper_cost']:.2f}")


if __name__ == "__main__":
    main()
