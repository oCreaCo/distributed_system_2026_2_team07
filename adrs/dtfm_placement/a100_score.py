"""Measured (A100) score of a placement policy vs DT-FM on N=8 topologies, with simulator v1/v2 predictions.

  skydiscover/.venv/bin/python adrs/dtfm_placement/a100_score.py [--label agentcc]

Reads A100_results/summary.json and the batch configs in output/a100_N8_*.json. For every N=8 topology with both a
`adrs_<topo>_dtfm` and an `adrs_<topo>_<label>` run, the measured ratio is iter(dtfm) / iter(label) (> 1: the
policy is faster). The measured score is the geometric mean of these ratios per split (train seeds 0-2,
held-out 100-102, fresh seeds >= 5000), i.e. combined_score's formula with measured times instead of simulated ones.
Simulator v1 (git 095eef5, theta in output/sim_v1/) and v2 (current sim/) predictions are shown next to it.
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import stepsim as SS  # noqa: E402

V2_PARAMS = SS.RunSpec.params


def v1_params(self, s):
    return self.layers * SS.LAYER_PARAMS + (SS.EMB_PARAMS if s in (0, self.pp - 1) else 0)


def predict(spec, version):
    if version == "v1":
        SS.RunSpec.params = v1_params
        th = list(json.load(open(os.path.join(HERE, "output", "sim_v1", "theta.json"))).values())[:9] + [1.0]
    else:
        SS.RunSpec.params = V2_PARAMS
        th = list(json.load(open(os.path.join(ROOT, "sim", "theta.json"))).values())
    v = SS.predict(th, spec)
    SS.RunSpec.params = V2_PARAMS
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="agentcc")
    ap.add_argument("--out", default=os.path.join(HERE, "output", "a100_score.json"))
    a = ap.parse_args()
    summary = json.load(open(os.path.join(ROOT, "A100_results", "summary.json")))
    runs = {}
    for path in sorted(glob.glob(os.path.join(HERE, "output", "a100_N8_*.json"))):
        cfg = json.load(open(path))
        for r in cfg["runs"]:
            runs.setdefault(r["tag"], dict(r, args=cfg["common_args"] + r.get("args", [])))
    rows = []
    for tag in sorted(runs):
        if not tag.endswith("_" + a.label):
            continue
        topo = tag[len("adrs_"):-len(a.label) - 1]
        dt = f"adrs_{topo}_dtfm"
        if tag not in summary or dt not in summary or dt not in runs:
            continue
        seed = int(topo.split("_s")[1])
        row = {"topology": topo, "split": "train" if seed < 100 else "held-out" if seed < 1000 else "fresh"}
        for name, t in (("dtfm", dt), ("policy", tag)):
            spec = SS.spec_from_summary(runs[t], {})
            row[name] = {"measured": summary[t]["iter_s"], "v1": predict(spec, "v1"), "v2": predict(spec, "v2")}
        for k in ("measured", "v1", "v2"):
            row["ratio_" + k] = row["dtfm"][k] / row["policy"][k]
        rows.append(row)
    print(f"{'topology':10s} {'split':8s} {'DT-FM meas':>10s} {'policy meas':>11s} | {'ratio meas':>10s} {'ratio v1':>8s} "
          f"{'ratio v2':>8s} | {'err v1 (dtfm/pol)':>17s} {'err v2 (dtfm/pol)':>17s}")
    order = {"train": 0, "held-out": 1, "fresh": 2}
    for r in sorted(rows, key=lambda r: (order[r["split"]], int(r["topology"].split("_s")[1]))):
        e = lambda k, v: 100 * (r[k][v] / r[k]["measured"] - 1)  # noqa: E731
        print(f"{r['topology']:10s} {r['split']:8s} {r['dtfm']['measured']:10.2f} {r['policy']['measured']:11.2f} | "
              f"{r['ratio_measured']:10.3f} {r['ratio_v1']:8.3f} {r['ratio_v2']:8.3f} | "
              f"{e('dtfm', 'v1'):+7.1f}/{e('policy', 'v1'):+6.1f}%  {e('dtfm', 'v2'):+7.1f}/{e('policy', 'v2'):+6.1f}%")
    gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))  # noqa: E731
    agg = {}
    for split in ("train", "held-out", "fresh", "all"):
        rs = [r for r in rows if split == "all" or r["split"] == split]
        if rs:
            agg[split] = {k: gm([r["ratio_" + k] for r in rs]) for k in ("measured", "v1", "v2")} | {"n": len(rs)}
            print(f"geomean {split:8s} (n={len(rs)}): measured {agg[split]['measured']:.3f}  sim v1 {agg[split]['v1']:.3f}  "
                  f"sim v2 {agg[split]['v2']:.3f}")
    json.dump({"rows": rows, "geomean": agg}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
