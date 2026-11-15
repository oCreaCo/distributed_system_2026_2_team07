"""SkyDiscover evaluator for the DT-FM placement task.

Runs the candidate's schedule() on seeded topologies (N = 8, 16, 32, 64, plus N = 2, 4 validity checks),
scores each placement with the calibrated simulator and compares it with the original DT-FM GA.

  combined_score = geometric mean over scored topologies of  sim(DT-FM GA placement) / sim(candidate)
                   (> 1: faster than DT-FM; a failed or invalid topology counts as 0.1)

CLI (e.g. held-out check of the best program):  python evaluator.py <program.py> [train|heldout]
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import baselines as BL  # noqa: E402
import reference as REF  # noqa: E402
import topologies as T  # noqa: E402

FAIL_RATIO = 0.1
RUN_TIMEOUT_S = 600
# the simulator must only be reached through the budgeted cost_fn
FORBIDDEN = re.compile(r"\bstepsim\b|theta\.json|\b(import|from)\s+(reference|baselines|evaluator|_run_candidate|topologies)\b")


def is_permutation(rank_map, n):
    return isinstance(rank_map, (list, tuple)) and len(rank_map) == n and sorted(rank_map) == list(range(n))


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def run_candidate(program_path, split):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    try:
        # one thread per worker: the runner already uses one process per topology
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        proc = subprocess.run([sys.executable, os.path.join(HERE, "_run_candidate.py"),
                               os.path.abspath(program_path), split, out],
                              cwd=HERE, capture_output=True, text=True, timeout=RUN_TIMEOUT_S, env=env)
        if proc.returncode != 0:
            return None, (proc.stderr or proc.stdout)[-2000:]
        return json.load(open(out)), None
    except subprocess.TimeoutExpired:
        return None, f"candidate run exceeded {RUN_TIMEOUT_S}s"
    finally:
        if os.path.exists(out):
            os.remove(out)


def evaluate(program_path, split="train"):
    source = open(program_path).read()
    if FORBIDDEN.search(source):
        return {"combined_score": 0.0, "artifacts": {"feedback": "Do not import the simulator or evaluator "
                                                                 "modules; use only the cost_fn argument."}}
    results, err = run_candidate(program_path, split)
    if results is None:
        return {"combined_score": 0.0, "artifacts": {"feedback": f"Program failed to run:\n{err}"}}

    theta = REF.load_theta()
    ref = REF.load()
    bl = BL.load() or {}
    scored, checks = T.topology_set(split)
    ratios, vs_random, by_n, regret_n8, lines, failures = [], [], defaultdict(list), [], [], 0
    vs_bl = defaultdict(list)

    for topo in checks:
        r = results[topo["name"]]
        if r["error"] or not is_permutation(r["rank_map"], topo["n"]):
            failures += 1
            lines.append(f"{topo['name']}: INVALID ({r['error'] or 'not a permutation'})")

    for topo in scored:
        r, base = results[topo["name"]], ref[topo["name"]]
        if r["error"] or not is_permutation(r["rank_map"], topo["n"]):
            failures += 1
            ratio = FAIL_RATIO
            lines.append(f"{topo['name']}: FAILED ({r['error'] or 'not a permutation'})")
        else:
            sim_s = REF.predict(theta, topo, r["rank_map"])
            ratio = base["paper_ga"]["sim_s"] / sim_s
            vs_random.append(base["random"]["sim_s"] / sim_s)
            for k, v in bl.get(topo["name"], {}).items():
                vs_bl[k].append(v["sim_s"] / sim_s)
            if "optimum" in base:
                regret_n8.append(sim_s / base["optimum"]["sim_s"] - 1)
            lines.append(f"{topo['name']}: {sim_s:7.2f}s  (DT-FM GA {base['paper_ga']['sim_s']:7.2f}s, "
                         f"random {base['random']['sim_s']:7.2f}s"
                         + (f", optimum {base['optimum']['sim_s']:7.2f}s" if "optimum" in base else "")
                         + f")  calls={r['calls']}  time={r['time_s']:.1f}s")
        ratios.append(ratio)
        by_n[topo["n"]].append(ratio)

    # validity checks on N = 2, 4 gate the score: a program that cannot handle tiny inputs is not usable
    combined = geomean(ratios) * (FAIL_RATIO if any("INVALID" in line for line in lines) else 1.0)
    metrics = {"combined_score": combined, "failures": failures,
               "vs_random": geomean(vs_random),
               "mean_regret_vs_optimum_N8": sum(regret_n8) / len(regret_n8) if regret_n8 else None,
               "mean_time_s": sum(r["time_s"] for r in results.values()) / len(results)}
    for k, rs in sorted(vs_bl.items()):
        metrics[f"vs_{k}"] = geomean(rs)
    for n, rs in sorted(by_n.items()):
        metrics[f"speedup_vs_dtfm_N{n}"] = geomean(rs)
    metrics["artifacts"] = {"feedback": "Per topology (predicted seconds per iteration, lower is better):\n"
                                        + "\n".join(lines)}
    return metrics


if __name__ == "__main__":
    m = evaluate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "train")
    print(m.pop("artifacts")["feedback"])
    print(json.dumps(m, indent=1))
