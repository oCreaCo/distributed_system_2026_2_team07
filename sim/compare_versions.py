"""Leave-one-topology-out comparison of simulator v1 (2026-11-03) and v2 (2026-11-15) on all measured runs.

  skydiscover/.venv/bin/python sim/compare_versions.py      # -> table + sim/loto_v1_v2.json

v1 = embedding parameters on the first and the last stage, no DP bandwidth efficiency (dp_bw_eff fixed at 1).
v2 = current sim/stepsim.py (embedding on stage 0 only, classification head on the last stage, dp_bw_eff fitted).
"P only" (v2 parameter counts, dp_bw_eff fixed at 1) is reported too. Every variant is fitted with
stepsim.fit_two_stage on the runs of all other topologies and scored on the held-out topology. Ranking metrics
compare placements within one (topology, PP x DP) group; pairs closer than 1% in measured time are skipped.
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import fit as FIT  # noqa: E402
import stepsim as SS  # noqa: E402

V2_PARAMS = SS.RunSpec.params
NET_V2 = list(SS.NET_IDX)


def v1_params(self, s):
    return self.layers * SS.LAYER_PARAMS + (SS.EMB_PARAMS if s in (0, self.pp - 1) else 0)


VARIANTS = {"v1": (v1_params, False), "P only": (V2_PARAMS, False), "v2": (V2_PARAMS, True)}


def fit(runs, theta0, dp_eff_free):
    SS.NET_IDX[:] = NET_V2 if dp_eff_free else [i for i in NET_V2 if i != 9]
    return SS.fit_two_stage([r["spec"] for r in runs], [r["measured"] for r in runs], [r["breakdown"] for r in runs],
                            [r["topology"] == "notc" for r in runs], theta0=theta0)


def main():
    runs = FIT.load()
    v1_theta = os.path.join(ROOT, "adrs", "dtfm_placement", "output", "sim_v1", "theta.json")
    theta0 = list(json.load(open(v1_theta)).values())[:9]
    topos = sorted({r["topology"] for r in runs} - {"notc"})
    out = {}
    for name, (params, eff_free) in VARIANTS.items():
        SS.RunSpec.params = params
        th0 = theta0 + [0.9 if eff_free else 1.0]
        th_all = fit(runs, th0, eff_free)
        for t in topos:
            th = fit([r for r in runs if r["topology"] != t], th_all, eff_free)
            for r in runs:
                if r["topology"] == t:
                    r["loo"] = SS.predict(th, r["spec"])
        ev = [r for r in runs if r["topology"] != "notc"]
        mape = lambda f: 100 * float(np.mean([abs(r["loo"] / r["measured"] - 1) for r in ev if f(r)]))  # noqa: E731
        groups = {}
        for r in ev:
            groups.setdefault(r.get("group", r["topology"]), []).append(r)
        agree = tot = 0
        regrets, rhos, misses = [], [], {}
        for g, rs in groups.items():
            if len(rs) < 3:
                continue
            p = np.array([r["loo"] for r in rs]); m = np.array([r["measured"] for r in rs])
            rhos.append(spearmanr(p, m).statistic)
            regrets.append(m[int(np.argmin(p))] / m.min() - 1)
            if regrets[-1] > 0.02:
                misses[g] = round(100 * regrets[-1], 1)
            for i, j in itertools.combinations(range(len(rs)), 2):
                if abs(m[i] - m[j]) / min(m[i], m[j]) < 0.01:
                    continue
                tot += 1
                agree += (p[i] < p[j]) == (m[i] < m[j])
        o = dict(mape_all=mape(lambda r: True), mape_1103=mape(lambda r: "group" not in r),
                 mape_1115=mape(lambda r: "group" in r), mape_pp8=mape(lambda r: r["spec"].pp == 8),
                 pairs=f"{agree}/{tot}", spearman=float(np.mean(rhos)), top1_regret_mean=100 * float(np.mean(regrets)),
                 top1_regret_max=100 * float(np.max(regrets)), top1_misses_over_2pct=misses,
                 theta_fit_all=dict(zip(SS.PARAM_NAMES, th_all)))
        out[name] = o
        print(f"{name:7s} LOTO MAPE all {o['mape_all']:.1f}% (11-03 {o['mape_1103']:.1f}%, 11-15 {o['mape_1115']:.1f}%, "
              f"PP8 {o['mape_pp8']:.1f}%) | pairs {o['pairs']} | Spearman {o['spearman']:.3f} | top-1 regret mean "
              f"{o['top1_regret_mean']:.2f}% max {o['top1_regret_max']:.2f}% {misses}")
    SS.RunSpec.params = V2_PARAMS
    SS.NET_IDX[:] = NET_V2
    json.dump(out, open(os.path.join(ROOT, "sim", "loto_v1_v2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
