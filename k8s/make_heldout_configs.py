"""Held-out validation batch (k8s/calib/job6.json): topologies never used for calibration.

  T4 uneven_two : 3 pods (0-2) in one region, 5 pods (3-7) in another; intra 2 ms / 4 Gbit, inter 30 ms / 0.8 Gbit
  T5 three      : regions {0,1,2} {3,4,5} {6,7}; intra 3 ms / 3 Gbit; inter (A,B) 20 ms / 1.5 Gbit,
                  (A,C) 45 ms / 0.9 Gbit, (B,C) 15 ms / 1.2 Gbit

Placements per topology: identity, paper (paper cost-model GA optimum), sim (best of all 8! rank maps
under the calibrated simulator sim/theta.json), simworst (worst under the simulator), rnd0.
"""
import itertools
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "k8s"))
sys.path.insert(0, os.path.join(ROOT, "sim"))
_RANDOM_SEED = random.seed
import run_scheduler as R  # noqa: E402
import stepsim as SS  # noqa: E402

N, PP, DP = 8, 4, 2


def regions_topology(region_of, intra, inter):
    d = np.zeros((N, N)); b = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ri, rj = region_of[i], region_of[j]
            if ri == rj:
                d[i][j], b[i][j] = intra
            else:
                d[i][j], b[i][j] = inter[tuple(sorted((ri, rj)))]
    return d.tolist(), b.tolist()


def topologies():
    t4 = regions_topology([0, 0, 0, 1, 1, 1, 1, 1], (2, 4.0), {(0, 1): (30, 0.8)})
    t5 = regions_topology([0, 0, 0, 1, 1, 1, 2, 2], (3, 3.0),
                          {(0, 1): (20, 1.5), (0, 2): (45, 0.9), (1, 2): (15, 1.2)})
    return {"T4": t4, "T5": t5}


def sim_search(theta, d, b, exclude=()):
    """Best and worst rank maps under the simulator; ties resolved away from maps in `exclude`."""
    best, worst = (float("inf"), None), (0, None)
    for perm in itertools.permutations(range(N)):
        rm = list(perm)
        t = SS.predict(theta, SS.RunSpec(d, b, rm))
        if t < best[0] or (t == best[0] and best[1] in exclude):
            best = (t, rm)
        if t > worst[0] or (t == worst[0] and worst[1] in exclude):
            worst = (t, rm)
    return best, worst


def main():
    theta = list(json.load(open(os.path.join(ROOT, "sim", "theta.json"))).values())
    manifest = json.load(open(os.path.join(ROOT, "k8s", "calib", "manifest.json")))
    common = [a for a in manifest["common_args"]]
    common[common.index("--num-iters") + 1] = "5"
    runs = []
    for name, (d, b) in topologies().items():
        manifest["topologies"][name] = {"delay_ms": d, "bandwidth_gbps": b}
        R.configure(d, b, PP, DP, 0.6, 1.0)
        _RANDOM_SEED(6)
        parts, scores, _ = R.S.GCMA(nodes=list(range(N)), population_size=100, trails=2000, mode="default")
        bestp = parts[int(np.argmin(scores))]
        partition = [tuple(bestp[i:i + DP]) for i in range(0, N, DP)]
        _, _, _, path, match = R.cost_of(partition)
        paper = R.to_rank_map(partition, path, match)
        (sim_t, sim_map), (worst_t, worst_map) = sim_search(theta, d, b, exclude=[list(range(N)), paper])
        # two more placements with simulator predictions distinct from the others (25th / 75th percentile
        # of 2000 random maps), so the held-out set spans the predicted range instead of one tie class
        samples = []
        for _ in range(2000):
            rm = list(range(N)); random.shuffle(rm)
            samples.append((SS.predict(theta, SS.RunSpec(d, b, rm)), rm))
        samples.sort()
        taken = {round(SS.predict(theta, SS.RunSpec(d, b, m)), 2) for m in (list(range(N)), paper, sim_map, worst_map)}
        extra = {}
        for pname, q in (("q25", 0.25), ("q75", 0.75)):
            k = int(q * len(samples))
            while k < len(samples) - 1 and round(samples[k][0], 2) in taken:
                k += 1
            taken.add(round(samples[k][0], 2)); extra[pname] = samples[k][1]
        placements = {"identity": list(range(N)), "paper": paper, "sim": sim_map, "simworst": worst_map, **extra}
        runs.append({"tag": f"bench_{name}", "type": "bench", "rank_map": None, "delay_ms": d, "bandwidth_gbps": b, "args": []})
        manifest["runs"][f"bench_{name}"] = {"kind": "bench", "topology": name}
        for pname, rm in placements.items():
            tag = f"{name}_{pname}"
            runs.append({"tag": tag, "rank_map": rm, "delay_ms": d, "bandwidth_gbps": b, "args": []})
            manifest["runs"][tag] = {"kind": "train", "rank_map": rm, "args": [], "topology": name, "placement": pname,
                                     "paper_cost": R.cost_of_rank_map(rm), "sim_pred": SS.predict(theta, SS.RunSpec(d, b, rm))}
            print(f"{tag:14s} rank_map={rm} paper={R.cost_of_rank_map(rm):6.2f} sim={SS.predict(theta, SS.RunSpec(d, b, rm)):6.2f}")
    # also test the simulator's claimed optimum on the calibration topologies T1 / T2
    for name in ("T1", "T2"):
        d, b = manifest["topologies"][name]["delay_ms"], manifest["topologies"][name]["bandwidth_gbps"]
        known = [r["rank_map"] for t, r in manifest["runs"].items() if r.get("topology") == name and r.get("rank_map")]
        (sim_t, sim_map), _ = sim_search(theta, d, b, exclude=known)
        R.configure(d, b, PP, DP, 0.6, 1.0)
        tag = f"{name}_simbest"
        runs.append({"tag": tag, "rank_map": sim_map, "delay_ms": d, "bandwidth_gbps": b, "args": []})
        manifest["runs"][tag] = {"kind": "train", "rank_map": sim_map, "args": [], "topology": name, "placement": "simbest",
                                 "paper_cost": R.cost_of_rank_map(sim_map), "sim_pred": sim_t}
        print(f"{tag:14s} rank_map={sim_map} paper={R.cost_of_rank_map(sim_map):6.2f} sim={sim_t:6.2f}")
    json.dump({"common_args": common, "runs": runs, "run_timeout_s": 1200},
              open(os.path.join(ROOT, "k8s", "calib", "job6.json"), "w"), indent=1)
    json.dump(manifest, open(os.path.join(ROOT, "k8s", "calib", "manifest.json"), "w"), indent=1)
    print(f"job6: {len(runs)} runs")


if __name__ == "__main__":
    main()
