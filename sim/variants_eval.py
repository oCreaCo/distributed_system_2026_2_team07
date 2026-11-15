"""Evaluate candidate simulator changes against the 2026-11-15 A100 runs, without changing the simulator.

Note: P was adopted into sim/stepsim.py later on 2026-11-15 (docs/worklog/*_simulator-v2.md), so "base" now
already includes it; the numbers in docs/worklog/2026-11-15_0941_* were produced before that change.

  skydiscover/.venv/bin/python sim/variants_eval.py [--refit]

Variants are applied by monkeypatching sim/stepsim.py in this process only (sim/stepsim.py and sim/theta.json
stay as they are):
  P  last-stage parameters: the logs show stage 0 = layers*LP + 63.58M (embedding) but the last stage =
     layers*LP + 4.20M (classification head); the simulator gives the embedding to both. Affects gradient
     bytes (DP sync) and optimizer work.
  R  per-message overhead of N_RTT round trips on every link (NCCL bench 11-15: 64 MB transfers take
     size/bw + ~6 RTT; tc delay itself is exact).
  S  LINK_SUM = True (a stage's upstream and downstream activation transfers add up instead of max).
Data:
  old = runs used for the current theta (sim/fit.py load(): 11-03 runs),
  new = held-out runs from 11-15 (PP4xDP2 validation/ranking on N8_s100-102, calib2 PP8 / PP4xDP2 / PP2xDP4 on
        two new topologies). With --refit, every variant is refitted on old (sim.fit two-stage fit) and scored on new.
"""
import argparse
import itertools
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import fit as FIT  # noqa: E402
import stepsim as SS  # noqa: E402

HEAD_PARAMS = 306_350_082 - 302_149_632          # last stage minus 6 transformer layers (PP4 logs)
NEW_CONFIGS = ["adrs/dtfm_placement/output/a100_N8_heldout_val.json", "adrs/dtfm_placement/output/a100_N8_rank_a.json",
               "adrs/dtfm_placement/output/a100_N8_rank_b.json", "k8s/calib2/p8.json", "k8s/calib2/p4d2.json",
               "k8s/calib2/p2d4.json"]
ORIG = dict(params=SS.RunSpec.params, link=SS.RunSpec.link, link_sum=SS.LINK_SUM)


def set_variant(v, n_rtt):
    SS.LINK_SUM = "S" in v or ORIG["link_sum"]
    if "P" in v:
        SS.RunSpec.params = lambda self, s: self.layers * SS.LAYER_PARAMS + (
            SS.EMB_PARAMS if s == 0 else HEAD_PARAMS if s == self.pp - 1 else 0)
    else:
        SS.RunSpec.params = ORIG["params"]
    if "R" in v:
        def link(self, a, b, lat_fixed_ms, tcp_window_mb, bw_eff=1.0):
            lat, bw = ORIG["link"](self, a, b, lat_fixed_ms, tcp_window_mb, bw_eff)
            return lat + n_rtt * 2 * lat, bw
        SS.RunSpec.link = link
    else:
        SS.RunSpec.link = ORIG["link"]


def load_new():
    summary = json.load(open(os.path.join(ROOT, "A100_results", "summary.json")))
    runs = []
    for path in NEW_CONFIGS:
        cfg = json.load(open(os.path.join(ROOT, path)))
        for r in cfg["runs"]:
            if r.get("type") == "bench" or r["tag"] not in summary or not summary[r["tag"]].get("iter_s"):
                continue
            spec = SS.spec_from_summary(dict(r, args=cfg["common_args"] + r.get("args", [])), {})
            parts = r["tag"].split("_")
            group = f"{parts[0]}_{parts[1]}" if parts[0] in ("p8", "p4d2", "p2d4") else f"p4d2_{parts[2]}"
            runs.append({"tag": r["tag"], "group": group, "spec": spec, "measured": summary[r["tag"]]["iter_s"]})
    return runs


def metrics(runs, key):
    groups = defaultdict(list)
    for r in runs:
        groups[r["group"]].append(r)
    agree = tot = 0
    rhos, regrets = [], []
    for g, rs in groups.items():
        p = np.array([r[key] for r in rs]); m = np.array([r["measured"] for r in rs])
        rhos.append(spearmanr(p, m).statistic)
        regrets.append(m[int(np.argmin(p))] / m.min() - 1)
        for i, j in itertools.combinations(range(len(rs)), 2):
            if abs(m[i] - m[j]) / min(m[i], m[j]) < 0.01:
                continue
            tot += 1
            agree += (p[i] < p[j]) == (m[i] < m[j])
    errs = [r[key] / r["measured"] - 1 for r in runs]
    dirs = []
    for r in runs:
        if r["tag"].endswith("_bestrev"):
            b = next(x for x in runs if x["tag"] == r["tag"][:-3])
            dm = r["measured"] / b["measured"] - 1
            dp = r[key] / b[key] - 1
            dirs.append((dm, dp))
    dir_ok = sum((abs(dm) < 0.01) or (dp * dm > 0) for dm, dp in dirs)
    return {"mape": float(np.mean(np.abs(errs))), "bias": float(np.mean(errs)), "pairs": f"{agree}/{tot}",
            "pair_acc": agree / tot, "spearman_mean": float(np.mean(rhos)), "top1_regret_mean": float(np.mean(regrets)),
            "top1_regret_max": float(np.max(regrets)), "direction": f"{dir_ok}/{len(dirs)}",
            "per_group_err": {g: [round(100 * (r[key] / r["measured"] - 1), 1) for r in rs] for g, rs in groups.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--n-rtt", type=float, default=6.0)
    ap.add_argument("--variants", nargs="+", default=["base", "P", "R", "S", "PR", "PRS", "PS"])
    ap.add_argument("--out", default=os.path.join(ROOT, "sim", "variants_eval.json"))
    a = ap.parse_args()
    theta = list(json.load(open(os.path.join(ROOT, "sim", "theta.json"))).values())
    old, new = FIT.load(), load_new()
    print(f"old (fit) runs: {len(old)}, new (held-out) runs: {len(new)}")
    out = {}
    for v in a.variants:
        set_variant(v, a.n_rtt)
        th = theta
        if a.refit:
            th = SS.fit_two_stage([r["spec"] for r in old], [r["measured"] for r in old], [r["breakdown"] for r in old],
                                  [r["topology"] == "notc" for r in old], theta0=theta)
        for r in old + new:
            r["pred"] = SS.predict(th, r["spec"])
        old_mape = float(np.mean([abs(r["pred"] / r["measured"] - 1) for r in old]))
        m = metrics(new, "pred")
        out[v] = dict(theta=dict(zip(SS.PARAM_NAMES, th)), old_mape=old_mape, **m)
        print(f"{v:5s} old MAPE {100 * old_mape:5.1f}% | new MAPE {100 * m['mape']:5.1f}% bias {100 * m['bias']:+5.1f}% "
              f"pairs {m['pairs']} ({m['pair_acc']:.2f}) spearman {m['spearman_mean']:.2f} "
              f"top1 regret mean {100 * m['top1_regret_mean']:.1f}% max {100 * m['top1_regret_max']:.1f}% dir {m['direction']}")
        for g, e in m["per_group_err"].items():
            print(f"        {g:10s} err% {e}")
    set_variant("base", 0)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
