"""Test-suite scorer for the DT-FM placement task (tcopt autopanel contract, hardened-v1).

  python measure.py --program <candidate.py> --input <topology.json> --time-limit <s>

Prints exactly one JSON object: {valid, reason, runtime_s, score, calls}. score is the simulated
seconds per training iteration of the returned placement (direction "min"), measured exactly as the
official evaluator does: schedule() runs in a fresh child process with a budgeted cost_fn
(2,000 calls) and a SIGALRM limit, and its rank_map is scored by the calibrated simulator.
"""
import argparse
import json
from pathlib import Path

from isolated import read_result, run_isolated

HERE = Path(__file__).resolve().parent
BUDGET = 2000

CALL = r'''
import json, signal, time
import stepsim

LIMIT = __LIMIT__
BUDGET = __BUDGET__
topo = json.load(open("input.dat"))
theta = list(json.load(open("theta.json")).values())
n = topo["n"]


def is_perm(rm):
    return isinstance(rm, (list, tuple)) and len(rm) == n and sorted(rm) == list(range(n))


def predict(rm):
    spec = stepsim.RunSpec(topo["delay_ms"], topo["bandwidth_gbps"], list(rm), pp=topo["pp"], dp=topo["dp"],
                           micro_batches=64, ga_steps=2, layers=topo["layers"])
    return stepsim.predict(theta, spec)


calls = 0


def cost_fn(rm):
    global calls
    calls += 1
    if calls > BUDGET:
        raise RuntimeError(f"cost_fn called more than budget={BUDGET} times")
    if not is_perm(rm):
        raise ValueError("cost_fn got an invalid rank_map (not a permutation of 0..N-1)")
    return predict(rm)


def on_alarm(signum, frame):
    raise TimeoutError(f"schedule() exceeded {LIMIT}s")


out = dict(valid=False, score=None, reason="", runtime_s=0.0, calls=0)
import program  # imported before the alarm, as in the official evaluator
signal.signal(signal.SIGALRM, on_alarm)
t0 = time.perf_counter()
signal.alarm(int(LIMIT))
try:
    rm = program.schedule([r[:] for r in topo["delay_ms"]], [r[:] for r in topo["bandwidth_gbps"]],
                          topo["pp"], topo["dp"], cost_fn, BUDGET)
    signal.alarm(0)
    rm = [int(x) for x in rm]
    if is_perm(rm):
        out.update(valid=True, score=predict(rm), reason="ok")
    else:
        out["reason"] = "returned rank_map is not a permutation of 0..N-1"
except BaseException as e:
    signal.alarm(0)
    out["reason"] = f"{type(e).__name__}: {e}"[:500]
out["runtime_s"] = time.perf_counter() - t0
out["calls"] = calls
json.dump(out, open("result.json", "w"))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--time-limit", type=float, required=True)
    a = ap.parse_args()
    code = CALL.replace("__LIMIT__", str(int(a.time_limit))).replace("__BUDGET__", str(BUDGET))
    run = run_isolated(a.program, code, a.input, deps=[HERE / "sim" / "stepsim.py", HERE / "sim" / "theta.json"],
                       timeout_s=a.time_limit + 30)
    res = read_result(run)
    if res is None:
        reason = "timeout" if run["timed_out"] else (run["stderr"] or run["stdout"])[-500:]
        res = dict(valid=False, score=None, reason=f"candidate process failed: {reason}", runtime_s=run["seconds"], calls=0)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
