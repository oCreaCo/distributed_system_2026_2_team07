"""Build an A100 batch config that runs a found algorithm's placement next to DT-FM's (N = 8).

  python make_a100_config.py [program.py] [--out output/a100_N8.json] [--split heldout|train]

For each N=8 topology of the split, two runs share the same tc settings: the original DT-FM GA
placement and the placement returned by program.schedule(). Launch with:  k8s/sync.sh <out>
"""
import argparse
import importlib.util
import json
import os

import reference as REF
import topologies as T

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("program", nargs="?", default=os.path.join(HERE, "output", "best", "best_program.py"))
    ap.add_argument("--out", default=os.path.join(HERE, "output", "a100_N8.json"))
    ap.add_argument("--split", choices=("heldout", "train"), default="heldout")
    ap.add_argument("--seeds", type=int, nargs="+", help="N=8 topology seeds (overrides --split)")
    ap.add_argument("--label", default="found", help="tag suffix for the program's placement")
    ap.add_argument("--no-dtfm", action="store_true", help="only the program's placement")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("program", args.program)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    theta, ref = REF.load_theta(), REF.load()

    runs = []
    for seed in args.seeds or (T.HELDOUT_SEEDS if args.split == "heldout" else T.TRAIN_SEEDS):
        topo = T.make_topology(8, seed)
        found = program.schedule(topo["delay_ms"], topo["bandwidth_gbps"], topo["pp"], topo["dp"],
                                 lambda rm: REF.predict(theta, topo, rm), REF.BUDGET)
        dtfm = ref[topo["name"]]["paper_ga"]["rank_map"] if topo["name"] in ref else REF.paper_ga(topo)
        pairs = [] if args.no_dtfm else [("dtfm", [int(x) for x in dtfm])]
        for name, rm in pairs + [(args.label, [int(x) for x in found])]:
            runs.append({"tag": f"adrs_{topo['name']}_{name}", "rank_map": rm,
                         "delay_ms": topo["delay_ms"], "bandwidth_gbps": topo["bandwidth_gbps"], "args": []})
            print(f"{runs[-1]['tag']:22s} rank_map={rm}  predicted {REF.predict(theta, topo, rm):6.1f} s/iter")

    common = json.load(open(os.path.join(ROOT, "k8s", "calib", "manifest.json")))["common_args"]  # same as calibration
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"common_args": common, "runs": runs, "run_timeout_s": 1500}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({len(runs)} runs). Launch: k8s/sync.sh {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
