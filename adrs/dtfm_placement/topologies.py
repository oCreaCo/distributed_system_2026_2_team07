"""Seeded heterogeneous-network topologies for the DT-FM placement task.

Pods are grouped into regions. Intra-region links: 1-5 ms, 2-5 Gbit/s (the range measured on A100).
Inter-region links: real city pairs from DT-FM's scheduler/heuristic_evolutionary_solver/config.py,
with delays scaled by 0.2-0.5 so they stay inside the range the simulator was calibrated on (<= ~130 ms).

PP x DP per N follows a 24-layer GPT3-XL split; PP4 x DP2 at N=8 is the calibrated configuration.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "DT-FM", "scheduler", "heuristic_evolutionary_solver"))
from config import delay_bandwidth_dict  # noqa: E402

LAYERS = 24
PARALLELISM = {2: (2, 1), 4: (2, 2), 8: (4, 2), 16: (4, 4), 32: (8, 4), 64: (8, 8)}  # N -> (PP, DP)
CITIES = sorted({c for pair in delay_bandwidth_dict for c in pair.split("-")})

SCORED_N = (8, 16, 32, 64)     # N <= 4 has at most 24 placements: checked for validity only
TRAIN_SEEDS = (0, 1, 2)
HELDOUT_SEEDS = (100, 101, 102)


def city_link(a, b):
    return delay_bandwidth_dict.get(f"{a}-{b}") or delay_bandwidth_dict[f"{b}-{a}"]


def make_topology(n, seed):
    rng = np.random.default_rng(seed * 1000 + n)
    num_regions = int(rng.integers(2, min(8, max(2, n // 2)) + 1))
    cities = rng.choice(CITIES, size=num_regions, replace=False)
    # every region gets at least one pod, the rest are spread unevenly
    region = np.concatenate([np.arange(num_regions), rng.integers(0, num_regions, n - num_regions)])
    rng.shuffle(region)
    scale = rng.uniform(0.2, 0.5)
    intra = [(rng.uniform(1, 5), rng.uniform(2, 5)) for _ in range(num_regions)]

    delay = np.zeros((n, n))
    bw = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ri, rj = region[i], region[j]
            if ri == rj:
                delay[i, j], bw[i, j] = intra[ri]
            else:
                d, b = city_link(cities[ri], cities[rj])
                delay[i, j], bw[i, j] = d * scale, b
    pp, dp = PARALLELISM[n]
    return {"name": f"N{n}_s{seed}", "n": n, "pp": pp, "dp": dp, "layers": LAYERS // pp,
            "regions": [str(cities[r]) for r in region],
            "delay_ms": np.round(delay, 2).tolist(), "bandwidth_gbps": np.round(bw, 3).tolist()}


def topology_set(split):
    seeds = TRAIN_SEEDS if split == "train" else HELDOUT_SEEDS
    scored = [make_topology(n, s) for n in SCORED_N for s in seeds]
    checks = [make_topology(n, seeds[0]) for n in (2, 4)]
    return scored, checks
