"""How much do the optimized placements depend on a simulator error? (analysis only; the simulator is unchanged)

Note: variant P was adopted into sim/stepsim.py later on 2026-11-15, so with the current simulator P equals base.

  skydiscover/.venv/bin/python adrs/dtfm_placement/model_sensitivity.py [--variant P] [--workers 12]

The variant (see sim/variants_eval.py; default P = correct last-stage parameter count) is applied by
monkeypatching sim/stepsim.py inside the worker processes. Per topology (train seeds 0-2 and held-out 100-102,
N = 8, 16, 32, 64) the best program (output/agent_cc/best_program.py) is run twice, once with the current
simulator as cost_fn and once with the variant, and the DT-FM paper_ga placement is computed once. Everything is
then scored under both models. Output: output/model_sensitivity_<variant>.json + a table.
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import generalization as G  # noqa: E402
import reference as REF  # noqa: E402
import topologies as T  # noqa: E402

PROGRAM = os.path.join(HERE, "output", "agent_cc", "best_program.py")


def job(args):
    n, seed, variant, which = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import variants_eval as VE
    theta = REF.load_theta()
    topo = T.make_topology(n, seed)
    if which == "dtfm":
        rm = REF.paper_ga(topo)
    else:
        VE.set_variant(variant if which == "variant" else "base", 0)
        captured = {}
        orig = REF.predict

        def spy(th, tp, r):                      # run_program's last predict call scores the returned placement
            captured["rm"] = list(r)
            return orig(th, tp, r)
        REF.predict = spy
        res = G.run_program(PROGRAM, topo, theta, 30)
        REF.predict = orig
        if res["sim_s"] is None:
            return (n, seed, which), None
        rm = captured["rm"]
    out = {}
    for model in ("base", variant):
        VE.set_variant(model, 0)
        out[model] = REF.predict(theta, topo, rm)
    return (n, seed, which), dict(rank_map=[int(x) for x in rm], **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="P")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    jobs = [(n, s, a.variant, w) for n in (64, 32, 16, 8) for s in (0, 1, 2, 100, 101, 102)
            for w in ("dtfm", "base", "variant")]
    with ProcessPoolExecutor(a.workers) as ex:
        res = {f"N{k[0]}_s{k[1]}:{k[2]}": v for k, v in ex.map(job, jobs)}
    json.dump(res, open(os.path.join(HERE, "output", f"model_sensitivity_{a.variant}.json"), "w"), indent=1)
    v = a.variant
    gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))  # noqa: E731
    print(f"{'topology':12s} {'same rm':>7s} {'loss under ' + v:>14s} {'loss under base':>15s} {'vsDTFM base':>11s} {'vsDTFM ' + v:>9s}")
    rows = []
    for n in (8, 16, 32, 64):
        for s in (0, 1, 2, 100, 101, 102):
            d, b, x = (res.get(f"N{n}_s{s}:{w}") for w in ("dtfm", "base", "variant"))
            if not (d and b and x):
                print(f"N{n}_s{s}: missing"); continue
            r = dict(t=f"N{n}_s{s}", n=n, train=s < 100, same=b["rank_map"] == x["rank_map"],
                     loss_v=b[v] / x[v], loss_base=x["base"] / b["base"], gain_base=d["base"] / b["base"],
                     gain_v_of_base=d[v] / b[v], gain_v=d[v] / x[v])
            rows.append(r)
            print(f"{r['t']:12s} {str(r['same']):>7s} {r['loss_v']:14.4f} {r['loss_base']:15.4f} {r['gain_base']:11.3f} {r['gain_v_of_base']:9.3f}")
    for split, sel in (("train", True), ("held-out", False)):
        rs = [r for r in rows if r["train"] == sel and r["n"] >= 8]
        print(f"{split:9s} combined_score-like geomean vs DT-FM: base-model {gm([r['gain_base'] for r in rs]):.4f} | "
              f"same placements under {v}: {gm([r['gain_v_of_base'] for r in rs]):.4f} | re-optimized under {v}: "
              f"{gm([r['gain_v'] for r in rs]):.4f} | mean loss from optimizing the wrong model {gm([r['loss_v'] for r in rs]):.4f}")


if __name__ == "__main__":
    main()
