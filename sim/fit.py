"""Fit the step-time simulator to measured runs and report its fidelity vs the paper cost model.

  python sim/fit.py                # fit on everything, leave-one-topology-out evaluation, write sim/theta.json

Data: A100_results/summary.json (from k8s/collect_results.py summarize) + k8s/calib/manifest.json.
The first single-run measurements (case3_*, full_notc) are mapped onto the T1 topology here.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, os.path.join(ROOT, "k8s"))
import stepsim as SS  # noqa: E402

LEGACY = {  # tag -> (topology, placement, rank_map); 10-iteration runs from the first reproduction
    "case3_identity_rep0": ("T1", "identity", list(range(8))),
    "case3_dplocal_rep0": ("T1", "dplocal", [0, 4, 1, 5, 2, 6, 3, 7]),
    "case3_worst_rep0": ("T1", "worst0", [4, 1, 5, 2, 0, 3, 7, 6]),
    "full_notc_rep0": ("notc", "identity", list(range(8))),
}


def load():
    summary = json.load(open(os.path.join(ROOT, "A100_results", "summary.json")))
    manifest = json.load(open(os.path.join(ROOT, "k8s", "calib", "manifest.json")))
    topologies = manifest["topologies"]
    runs = []
    for tag, r in summary.items():
        if tag in LEGACY:
            topo, placement, rm = LEGACY[tag]
            r = dict(r, topology=topo, placement=placement, rank_map=rm, args=[])
        if not r.get("ok") or r.get("iter_s") is None or r.get("kind") == "bench" or "topology" not in r:
            continue
        spec = SS.spec_from_summary(r, topologies)
        r0 = r.get("ranks", {}).get("0") or r.get("ranks", {}).get(0) or {}
        runs.append({"tag": tag, "topology": r["topology"], "placement": r.get("placement", ""),
                     "spec": spec, "measured": r["iter_s"], "paper_cost": paper_cost(r, topologies),
                     "breakdown": {k: r0.get(k) for k in ("fwd", "bwd", "opt")}})
    return runs


def paper_cost(r, topologies):
    """Paper communication-only cost for the same placement (None when no tc)."""
    import run_scheduler as R
    if r.get("topology") in (None, "notc"):
        return None
    if r.get("delay_ms") is not None:
        d, b = r["delay_ms"], r["bandwidth_gbps"]
    elif r["topology"] == "homo":
        d = np.full((8, 8), float(r["delay"])).tolist(); b = np.full((8, 8), float(r["bw"])).tolist()
    else:
        t = topologies[r["topology"]]; d, b = t["delay_ms"], t["bandwidth_gbps"]
    R.configure(d, b, 4, 2, 0.6, 1.0)
    return R.cost_of_rank_map(r.get("rank_map") or list(range(8)))


def report(name, runs, pred_key):
    groups = defaultdict(list)
    for r in runs:
        groups[r["topology"]].append(r)
    print(f"\n[{name}]")
    for topo, rs in sorted(groups.items()):
        pred = [r[pred_key] for r in rs]
        if any(p is None for p in pred) or len(rs) < 2:
            continue
        m = SS.ranking_metrics(pred, [r["measured"] for r in rs])
        print(f"  {topo:6s} n={m['n']:2d}  MAPE={m['mape']:.3f}  spearman={m.get('spearman', float('nan')):.2f}  "
              f"pairwise={m['pairwise_acc'] if m['pairwise_acc'] is not None else float('nan'):.2f}  top1_regret={m['top1_regret_s']:.2f}s")


def main():
    import argparse, re
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", help="regex on tags to leave out of the fit")
    ap.add_argument("--out", default=os.path.join(ROOT, "sim", "theta.json"))
    ap.add_argument("--init", help="theta json to start the least-squares fit from")
    a = ap.parse_args()
    runs = load()
    if a.exclude:
        runs = [r for r in runs if not re.search(a.exclude, r["tag"])]
    print(f"{len(runs)} measured runs")
    specs = [r["spec"] for r in runs]; meas = [r["measured"] for r in runs]
    theta0 = list(json.load(open(a.init)).values()) if a.init else None
    is_notc = [r["topology"] == "notc" for r in runs]
    theta = SS.fit_two_stage(specs, meas, [r['breakdown'] for r in runs], is_notc, theta0=theta0)
    for r in runs:
        r["sim_all"] = SS.predict(theta, r["spec"])
    print("theta (fit on all):", {k: round(v, 4) for k, v in zip(SS.PARAM_NAMES, theta)})
    print(f"{'tag':28s} {'meas':>7s} {'sim':>7s} {'err%':>6s} {'paper':>7s}")
    for r in sorted(runs, key=lambda x: (x['topology'], x['measured'])):
        pc = r["paper_cost"]
        print(f"{r['tag']:28s} {r['measured']:7.2f} {r['sim_all']:7.2f} {100 * (r['sim_all'] - r['measured']) / r['measured']:6.1f} "
              f"{pc if pc is not None else float('nan'):7.2f}")

    # leave-one-topology-out: fit on the other topologies, predict the held-out one
    for r in runs:
        r["sim_loo"] = None
    for topo in sorted({r["topology"] for r in runs}):
        train = [r for r in runs if r["topology"] != topo]
        test = [r for r in runs if r["topology"] == topo]
        if len(train) < 8 or not test:
            continue
        th = SS.fit_two_stage([r["spec"] for r in train], [r["measured"] for r in train], [r["breakdown"] for r in train],
                              [r["topology"] == "notc" for r in train], theta0=theta0)
        for r in test:
            r["sim_loo"] = SS.predict(th, r["spec"])
    report("simulator, fit on all runs", runs, "sim_all")
    report("simulator, leave-one-topology-out", [r for r in runs if r["sim_loo"] is not None], "sim_loo")
    report("paper cost model", [r for r in runs if r["paper_cost"] is not None], "paper_cost")
    json.dump(dict(zip(SS.PARAM_NAMES, theta)), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
