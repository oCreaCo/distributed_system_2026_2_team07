"""Build the CEGAS test suite for the DT-FM placement task (tcopt autopanel format).

  skydiscover/.venv/bin/python adrs/dtfm_placement/cegas/build_panel.py [--out <dir>]

The suite has eight topologies from seeds disjoint from train (0-2) and held-out (100-102):
three small (N=16), three medium (N=32), two large (N=64), as in the paper's suite shape.
The per-test quality is the simulated iteration time of the candidate's placement (lower is better).
The scorer (measure.py) and frozen copies of the simulator and its calibration go next to panel.json;
validate.py then fills in the reference rows and the tolerance.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent
ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(TASK_DIR))
import topologies as T  # noqa: E402

SUITE = [(16, 200), (16, 201), (16, 202), (32, 203), (32, 204), (32, 205), (64, 206), (64, 207)]
TIME_LIMIT_S = 30           # same per-topology limit as the official evaluator
TASK = "dtfm_placement"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(out, reverse=False):
    out = Path(out).resolve()
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "sim").mkdir(exist_ok=True)
    shutil.copy2(HERE / "measure.py", out / "measure.py")
    shutil.copy2(ROOT / "sim" / "stepsim.py", out / "sim" / "stepsim.py")
    shutil.copy2(ROOT / "sim" / "theta.json", out / "sim" / "theta.json")
    inputs = []
    for n, seed in (SUITE[::-1] if reverse else SUITE):   # order = counterexample priority (first failing test)
        topo = T.make_topology(n, seed)
        topo.pop("regions")                       # schedule() never sees region labels
        path = out / "inputs" / f"{topo['name']}.json"
        path.write_text(json.dumps(topo))
        inputs.append(dict(id=topo["name"], path=f"inputs/{path.name}", sha256=sha(path), n=n, pp=topo["pp"], dp=topo["dp"]))
    manifest = dict(format="tcopt-autopanel-v1", task=TASK, direction="min", language="python", program_suffix=".py",
                    measure="measure.py", measure_sha256=sha(out / "measure.py"), isolation_contract="hardened-v1",
                    python=str(ROOT / "skydiscover" / ".venv" / "bin" / "python"),
                    time_limit_s=TIME_LIMIT_S, measurement_wall_limit_s=TIME_LIMIT_S + 90, tol=0.005,
                    quality="simulated seconds per training iteration (sim/stepsim.py + sim/theta.json); lower is better",
                    simulator_sha256={"stepsim.py": sha(out / "sim" / "stepsim.py"), "theta.json": sha(out / "sim" / "theta.json")},
                    inputs=inputs, status="pending")
    (out / "panel.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out / 'panel.json'} ({len(inputs)} inputs)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TASK_DIR / "output" / "cegas_panel"))
    ap.add_argument("--reverse", action="store_true", help="list large topologies first (N64 -> N16)")
    a = ap.parse_args()
    build(a.out, a.reverse)
