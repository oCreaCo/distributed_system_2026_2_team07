"""Estimate headroom: how much better than the evolved programs can a placement get under the simulator?

  skydiscover/.venv/bin/python adrs/dtfm_placement/headroom.py [--calls 100000] [--split train] [--programs p1.py p2.py ...]

For every topology, start from the placements of the given programs (run with the normal 2000-call budget) and
run simulated annealing over GPU swaps with a much larger simulator budget. The result is still only a lower
bound on the true optimum (except N=8, where reference.json has the exhaustive optimum), but if it beats the
programs by a wide margin, the programs are stuck on a plateau rather than at the limit.
"""
import argparse
import importlib.util
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor

import reference as REF
import topologies as T

HERE = os.path.dirname(os.path.abspath(__file__))


def load_program(path):
    spec = importlib.util.spec_from_file_location(f"p{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def anneal(topo, theta, starts, calls, seed):
    rng = random.Random(seed)
    n, pp = topo["n"], topo["pp"]
    f = lambda rm: REF.predict(theta, topo, rm)
    best_rm, best = None, math.inf
    per = calls // len(starts)
    for rm0 in starts:
        cur = list(rm0); cc = f(cur)
        if cc < best:
            best, best_rm = cc, cur[:]
        t0 = 0.02 * cc                                   # initial temperature: 2% of the start cost
        for k in range(per):
            t = t0 * (1 - k / per) + 1e-9
            x, y = rng.sample(range(n), 2)
            if rng.random() < 0.3:                       # swap two whole replicas (keeps DP groups' stage sets)
                ra, rb = rng.sample(range(n // pp), 2)
                pos = {r: g for g, r in enumerate(cur)}
                cand = cur[:]
                for s in range(pp):
                    ga, gb = pos[ra * pp + s], pos[rb * pp + s]
                    cand[ga], cand[gb] = cand[gb], cand[ga]
            else:
                cand = cur[:]; cand[x], cand[y] = cand[y], cand[x]
            c = f(cand)
            if c <= cc or rng.random() < math.exp((cc - c) / t):
                cur, cc = cand, c
                if c < best:
                    best, best_rm = c, cand[:]
    return best, best_rm


def one(args):
    topo_name, split, programs, calls = args
    theta, ref = REF.load_theta(), REF.load()
    topo = next(t for t in T.topology_set(split)[0] if t["name"] == topo_name)
    starts, prog_cost = [], {}
    for p in programs:
        mod = load_program(p)
        rm = [int(x) for x in mod.schedule(topo["delay_ms"], topo["bandwidth_gbps"], topo["pp"], topo["dp"],
                                          lambda r: REF.predict(theta, topo, r), REF.BUDGET)]
        starts.append(rm)
        prog_cost[os.path.basename(os.path.dirname(os.path.dirname(p)))] = REF.predict(theta, topo, rm)
    best, rm = anneal(topo, theta, starts, calls, seed=0)
    dtfm = ref[topo_name]["paper_ga"]["sim_s"]
    return topo_name, dict(dtfm=dtfm, programs=prog_cost, headroom=best, rank_map=rm,
                           optimum=ref[topo_name].get("optimum", {}).get("sim_s"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=100000)
    ap.add_argument("--split", default="train")
    ap.add_argument("--programs", nargs="+", default=[os.path.join(HERE, "output", r, "best", "best_program.py")
                                                      for r in ("main_evox_100", "main_adaevolve_100", "cegas_r300_s0")])
    ap.add_argument("--out", default=os.path.join(HERE, "output", "headroom.json"))
    a = ap.parse_args()
    names = [t["name"] for t in T.topology_set(a.split)[0]]
    with ProcessPoolExecutor(min(len(names), os.cpu_count())) as ex:
        res = dict(ex.map(one, [(n, a.split, a.programs, a.calls) for n in names]))
    json.dump(res, open(a.out, "w"), indent=1)
    gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
    progs = list(next(iter(res.values()))["programs"])
    for n, r in res.items():
        print(f"{n:8s} dtfm {r['dtfm']:6.1f}  " + "  ".join(f"{k[:12]} {v:6.1f}" for k, v in r["programs"].items())
              + f"  headroom {r['headroom']:6.1f}" + (f"  optimum {r['optimum']:6.1f}" if r["optimum"] else ""))
    for k in progs:
        print(f"combined {k:20s} {gm([r['dtfm'] / r['programs'][k] for r in res.values()]):.4f}")
    print(f"combined {'best-of-programs':20s} {gm([r['dtfm'] / min(r['programs'].values()) for r in res.values()]):.4f}")
    print(f"combined {'headroom (SA)':20s} {gm([r['dtfm'] / r['headroom'] for r in res.values()]):.4f}")


if __name__ == "__main__":
    main()
