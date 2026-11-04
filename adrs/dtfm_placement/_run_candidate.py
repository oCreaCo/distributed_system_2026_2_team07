"""Run a candidate program's schedule() on every topology of a split (subprocess of evaluator.py).

  python _run_candidate.py <program.py> <train|heldout> <out.json>

The simulator is only reachable through the budgeted cost_fn handed to schedule().
"""
import importlib.util
import json
import signal
import sys
import time
import traceback
from multiprocessing import Pool

import evaluator as E
import reference as REF
import topologies as T

TOPOLOGY_TIME_LIMIT_S = 30


class BudgetExceeded(Exception):
    pass


def on_alarm(signum, frame):
    raise TimeoutError(f"schedule() exceeded {TOPOLOGY_TIME_LIMIT_S}s")


def run_one(program_path, topo):
    """Run schedule() on one topology (in its own worker process, so SIGALRM limits it alone)."""
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    theta = REF.load_theta()
    calls = 0

    def cost_fn(rank_map):
        nonlocal calls
        calls += 1
        if calls > REF.BUDGET:
            raise BudgetExceeded(f"cost_fn called more than budget={REF.BUDGET} times")
        if not E.is_permutation(rank_map, topo["n"]):
            raise ValueError("cost_fn got an invalid rank_map (not a permutation of 0..N-1)")
        return REF.predict(theta, topo, rank_map)

    signal.signal(signal.SIGALRM, on_alarm)
    t0 = time.perf_counter()
    signal.alarm(TOPOLOGY_TIME_LIMIT_S)
    try:
        rm = program.schedule([row[:] for row in topo["delay_ms"]], [row[:] for row in topo["bandwidth_gbps"]],
                              topo["pp"], topo["dp"], cost_fn, REF.BUDGET)
        rm = [int(x) for x in rm]
        err = None
    except Exception as e:  # noqa: BLE001 - any failure of the candidate is reported, not raised
        rm, err = None, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=2).splitlines()[-1]}"
    finally:
        signal.alarm(0)
    return topo["name"], {"rank_map": rm, "calls": calls, "time_s": time.perf_counter() - t0, "error": err}


def main(program_path, split, out_path):
    scored, checks = T.topology_set(split)
    topos = scored + checks
    # topologies are independent: one process each (the simulator is pure CPU, so they scale linearly)
    with Pool(len(topos)) as pool:
        results = dict(pool.starmap(run_one, [(program_path, t) for t in topos]))
    json.dump(results, open(out_path, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
