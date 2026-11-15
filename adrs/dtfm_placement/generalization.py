"""Check whether evolved programs generalize beyond the topologies they were scored on.

  skydiscover/.venv/bin/python adrs/dtfm_placement/generalization.py [--seeds 5000 5001 ...] [--sizes 16 32 64 128]
      [--programs name=path ...] [--time-limit 120]

Fresh topologies (seeds outside train 0-2, held-out 100-102 and the agent's study range 1000-4999), including
N=128 (PP8 x DP16, not used anywhere before). Per topology: original DT-FM GA (paper cost model), ga_sim (DT-FM GCMA
with the simulator as fitness, same 2000-call budget) and every program with the normal 2000-call budget. The time
limit is relaxed (default 120 s) so quality and speed are reported separately; runs over 30 s are flagged.
"""
import argparse
import importlib.util
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import baselines as BL
import reference as REF
import topologies as T

HERE = os.path.dirname(os.path.abspath(__file__))
T.PARALLELISM.setdefault(128, (8, 16))          # 24 layers / PP8 = 3 layers per stage
DEFAULT_PROGRAMS = {k: os.path.join(HERE, "output", v, "best", "best_program.py")
                    for k, v in (("evox", "main_evox_100"), ("adaevolve", "main_adaevolve_100"), ("cegas300", "cegas_r300_s0"))}
DEFAULT_PROGRAMS["seed"] = os.path.join(HERE, "initial_program.py")


class Budget(Exception):
    pass


def run_program(path, topo, theta, limit_s):
    import signal
    spec = importlib.util.spec_from_file_location("prog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = 0

    def cost_fn(rm):
        nonlocal calls
        calls += 1
        if calls > REF.BUDGET:
            raise Budget()
        return REF.predict(theta, topo, list(rm))

    def alarm(*_):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(limit_s)
    t0 = time.perf_counter()
    try:
        rm = [int(x) for x in mod.schedule(topo["delay_ms"], topo["bandwidth_gbps"], topo["pp"], topo["dp"], cost_fn, REF.BUDGET)]
        err = None if sorted(rm) == list(range(topo["n"])) else "not a permutation"
    except Exception as e:                       # noqa: BLE001
        rm, err = None, type(e).__name__
    finally:
        signal.alarm(0)
    dt = time.perf_counter() - t0
    return dict(sim_s=REF.predict(theta, topo, rm) if rm and not err else None, time_s=round(dt, 2), calls=min(calls, REF.BUDGET), error=err)


def job(args):
    kind, name, n, seed, limit_s = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    theta = REF.load_theta()
    topo = T.make_topology(n, seed)
    if kind == "dtfm":
        rm = REF.paper_ga(topo)
        return (topo["name"], kind), dict(sim_s=REF.predict(theta, topo, rm))
    if kind == "ga_sim":
        rm, _ = BL.ga_sim(topo, theta)
        return (topo["name"], kind), dict(sim_s=REF.predict(theta, topo, rm))
    return (topo["name"], kind), run_program(name, topo, theta, limit_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[5000, 5001, 5002, 5003, 5004])
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--programs", nargs="*", default=None, help="name=path pairs (default: evox, adaevolve, cegas300, seed)")
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "output", "generalization.json"))
    a = ap.parse_args()
    progs = dict(p.split("=", 1) for p in a.programs) if a.programs else DEFAULT_PROGRAMS
    jobs = [(k, None, n, s, a.time_limit) for n in a.sizes for s in a.seeds for k in ("dtfm", "ga_sim")]
    jobs += [(k, p, n, s, a.time_limit) for n in a.sizes for s in a.seeds for k, p in progs.items()]
    jobs.sort(key=lambda j: -j[2])               # big topologies first
    res = defaultdict(dict)
    with ProcessPoolExecutor(a.workers) as ex:
        for (topo, kind), r in ex.map(job, jobs):
            res[topo][kind] = r
    json.dump(res, open(a.out, "w"), indent=1)

    gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
    print(f"{'N':>4s} {'program':10s} {'vs DT-FM':>9s} {'vs ga_sim':>9s} {'fail':>4s} {'>30s':>4s} {'max time':>8s}")
    for n in a.sizes:
        names = [f"N{n}_s{s}" for s in a.seeds]
        for k in progs:
            ok = [t for t in names if res[t][k]["sim_s"]]
            vd = gm([res[t]["dtfm"]["sim_s"] / res[t][k]["sim_s"] for t in ok])
            vg = gm([res[t]["ga_sim"]["sim_s"] / res[t][k]["sim_s"] for t in ok])
            slow = sum(res[t][k]["time_s"] > 30 for t in names)
            print(f"{n:4d} {k:10s} {vd:9.3f} {vg:9.3f} {len(names) - len(ok):4d} {slow:4d} {max(res[t][k]['time_s'] for t in names):8.1f}")
        print(f"{n:4d} {'ga_sim':10s} {gm([res[t]['dtfm']['sim_s'] / res[t]['ga_sim']['sim_s'] for t in names]):9.3f}")


if __name__ == "__main__":
    main()
