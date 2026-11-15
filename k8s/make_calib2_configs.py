"""Second calibration data set (2026-11-15): batches that cover what the 11-03 data lacks.

  skydiscover/.venv/bin/python k8s/make_calib2_configs.py      # -> k8s/calib2/{p8,p2d4,p4d2}.json + manifest.json

New N=8 topologies from the ADRS generator (adrs/dtfm_placement/topologies.make_topology, seeds >= 300; not used
anywhere else). Per batch and topology, placements are picked from the simulator's ranking of all 8! placements
(best / median / worst) plus, for the pipeline-direction question, the best placement with its pipeline reversed
(same groups, stage order flipped; the simulator predicts the same time for both).
Two topologies (the first seeds >= 300 whose placements spread out under every config) are used by all batches:
  p8   : PP8 x DP1 (3 layers per stage), + NCCL pair benchmark of each topology
  p2d4 : PP2 x DP4 (12 layers per stage)
  p4d2 : PP4 x DP2 (6 layers, the calibrated config)
Each batch stays under ~8 runs so it finishes inside the job's 2 h deadline; batches are useful on their own.
"""
import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "adrs", "dtfm_placement"))
import reference as REF  # noqa: E402
import topologies as T  # noqa: E402

OUT = os.path.join(ROOT, "k8s", "calib2")
COMMON = json.load(open(os.path.join(ROOT, "k8s", "calib", "manifest.json")))["common_args"]
CONFIGS = {"p8": (8, 1, True), "p2d4": (2, 4, False), "p4d2": (4, 2, False)}   # (pp, dp, NCCL bench of the topologies)
N_TOPOS = 2


def spread_ok(theta, seed):
    """Keep topologies where every config has distinct best < median < worst (>= 3% apart) in a 3000-sample estimate."""
    import random
    rng = random.Random(seed)
    for pp, dp, _ in CONFIGS.values():
        topo = dict(T.make_topology(8, seed), pp=pp, dp=dp, layers=T.LAYERS // pp)
        xs = sorted(REF.predict(theta, topo, rng.sample(range(8), 8)) for _ in range(3000))
        lo, mid, hi = xs[0], xs[len(xs) // 2], xs[-1]
        if not (mid > lo * 1.03 and hi > mid * 1.03):
            return False
    return True


def reverse_pipeline(rm, pp):
    """Same GPU per (replica, stage) set, stage order flipped: rank r -> replica*pp + (pp-1-stage)."""
    return [(r // pp) * pp + (pp - 1 - r % pp) for r in rm]


def main():
    os.makedirs(OUT, exist_ok=True)
    theta = REF.load_theta()
    manifest = {"common_args": COMMON, "batches": {}}
    seeds = []
    for seed in range(300, 400):
        if spread_ok(theta, seed):
            seeds.append(seed)
        if len(seeds) == N_TOPOS:
            break
    print("topology seeds:", seeds)
    for name, (pp, dp, bench) in CONFIGS.items():
        args = ["--pipeline-group-size", str(pp), "--data-group-size", str(dp), "--num-layers", str(T.LAYERS // pp)]
        runs = []
        for seed in seeds:
            topo = dict(T.make_topology(8, seed), pp=pp, dp=dp, layers=T.LAYERS // pp)
            base = {"delay_ms": topo["delay_ms"], "bandwidth_gbps": topo["bandwidth_gbps"]}
            tname = f"c2s{seed}"
            if bench:
                runs.append({"tag": f"bench_{tname}", "type": "bench", "rank_map": None, **base, "args": []})
            allp = sorted((REF.predict(theta, topo, p), p) for p in itertools.permutations(range(8)))
            best, med, worst = list(allp[0][1]), list(allp[len(allp) // 2][1]), list(allp[-1][1])
            picks = {"best": best, "bestrev": reverse_pipeline(best, pp), "median": med, "worst": worst}
            for pname, rm in picks.items():
                pred = REF.predict(theta, topo, rm)
                runs.append({"tag": f"{name}_{tname}_{pname}", "rank_map": [int(x) for x in rm], **base, "args": args,
                             "pred_s": round(pred, 2)})
                print(f"{name:5s} {tname} {pname:8s} pred {pred:7.2f}  rank_map {rm}")
        cfg = {"common_args": COMMON, "run_timeout_s": 1500,
               "runs": [{k: v for k, v in r.items() if k != "pred_s"} for r in runs]}
        json.dump(cfg, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
        manifest["batches"][name] = {"pp": pp, "dp": dp, "seeds": list(seeds),
                                     "runs": [{"tag": r["tag"], "rank_map": r.get("rank_map"), "pred_s": r.get("pred_s")} for r in runs]}
        print(f"wrote k8s/calib2/{name}.json ({len(runs)} runs)")
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
