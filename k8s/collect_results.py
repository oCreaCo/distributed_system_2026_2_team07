"""Collect experiment logs into A100_results/<tag>/ and build A100_results/summary.json.

  python k8s/collect_results.py fetch <jobname>   # kubectl logs of the 8 experiment pods -> A100_results/jobs/<jobname>/pod{i}.log
  python k8s/collect_results.py split <jobname>   # split those logs per run tag -> A100_results/<tag>/rank{r}_pod{i}.log
  python k8s/collect_results.py summarize         # parse every A100_results/<tag>/ -> A100_results/summary.json (+ table)

Per run we keep: avg iteration time (rank 0, excl. first iter), per-rank phase means (fwd/bwd per ga-step,
optimizer step), per-rank parameter counts, and for bench runs the pairwise latency/bandwidth.
"""
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "A100_results")
RUN_MARK = re.compile(r"\[test\.py\] === run \d+/\d+ tag=(\S+) rank=(\d+) ")


def kubectl(*args):
    env = dict(os.environ, KUBECONFIG=os.path.join(ROOT, "team07.kubeconfig"))
    return subprocess.run(["kubectl", *args], env=env, check=True, capture_output=True, text=True).stdout


def fetch(jobname):
    out = os.path.join(RES, "jobs", jobname)
    os.makedirs(out, exist_ok=True)
    pods = kubectl("get", "pods", "-l", "batch.kubernetes.io/job-name=experiment", "-o", "name").split()
    for pod in pods:
        name = pod.split("/")[1]
        idx = int(name.split("-")[1])
        with open(os.path.join(out, f"pod{idx}.log"), "w") as f:
            f.write(kubectl("logs", name))
    print(f"fetched {len(pods)} pod logs into {out}")


def split(jobname):
    for path in sorted(glob.glob(os.path.join(RES, "jobs", jobname, "pod*.log"))):
        idx = int(re.search(r"pod(\d+)\.log", path).group(1))
        cur, buf = None, []
        for line in open(path):
            m = RUN_MARK.search(line)
            if m:
                flush(cur, buf, idx)
                cur, buf = (m.group(1), int(m.group(2))), []
            buf.append(line)
        flush(cur, buf, idx)
    print("split done")


def flush(cur, buf, idx):
    if cur is None:
        return
    tag, rank = cur
    d = os.path.join(RES, tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"rank{rank}_pod{idx}.log"), "w") as f:
        f.writelines(buf)


def mean_skip(vals, skip=1):
    v = vals[skip:] if len(vals) > skip else vals
    return st.mean(v) if v else None


def parse_run(tag):
    d = os.path.join(RES, tag)
    files = sorted(glob.glob(os.path.join(d, "rank*_pod*.log")) + glob.glob(os.path.join(d, "experiment-*.log")))
    if not files:
        return None
    info = {"tag": tag, "ranks": {}, "iter_s": None, "ok": True}
    pairs = {}
    for path in files:
        txt = open(path).read()
        m = re.search(r"rank(\d+)_pod(\d+)", path) or re.search(r"\[test\.py\] tag=\S+ pod=\d+ rank=(\d+)", txt)
        rank = int(m.group(1)) if m else None
        for pm in re.finditer(r"pair (\d+)-(\d+): latency ([\d.]+) ms, bandwidth ([\d.]+) Gbit/s", txt):
            pairs[f"{pm.group(1)}-{pm.group(2)}"] = {"latency_ms": float(pm.group(3)), "bandwidth_gbps": float(pm.group(4))}
        if "Traceback" in txt or re.search(r"exited with (?!0\b)", txt):
            info["ok"] = False
        fin = re.search(r"averaged \(exclude the first iter\) run time: ([\d.]+)", txt)
        if fin and info["iter_s"] is None:
            info["iter_s"] = float(fin.group(1))
        r = {}
        params = re.search(r"Total number of parameters: (\d+)", txt)
        if params:
            r["params"] = int(params.group(1))
        # the rank-0-of-pipeline prints per-phase times; other ranks print their own "Rank N node ..." lines too
        fw = [float(x) for x in re.findall(r"forward pass \d+/\d+ takes ([\d.]+)s", txt)]
        bw = [float(x) for x in re.findall(r"backward pass \d+/\d+ takes ([\d.]+)s", txt)]
        opt = [float(x) for x in re.findall(r"optimizer step takes ([\d.]+)s", txt)]
        it = [float(x) for x in re.findall(r"whole iteration takes ([\d.]+)s", txt)]
        for k, v, skip in (("fwd", fw, 2), ("bwd", bw, 2), ("opt", opt, 1), ("iter", it, 1)):
            if v:
                r[k] = mean_skip(v, skip)
        if rank is not None and r:
            info["ranks"][rank] = r
    if pairs:
        info["pairs"] = pairs
    if info["iter_s"] is None:  # pipeline-rank-0 log missing: fall back to any rank's per-iteration times
        its = [r["iter"] for r in info["ranks"].values() if r.get("iter")]
        if its:
            info["iter_s"] = st.mean(its)
    return info


def summarize():
    manifest = {}
    mp = os.path.join(ROOT, "k8s", "calib", "manifest.json")
    if os.path.exists(mp):
        manifest = json.load(open(mp))["runs"]
    runs = {}
    for d in sorted(glob.glob(os.path.join(RES, "*"))):
        tag = os.path.basename(d)
        if tag in ("jobs", "summary.json") or not os.path.isdir(d):
            continue
        info = parse_run(tag)
        if info:
            info.update({k: v for k, v in manifest.get(tag, {}).items()})
            runs[tag] = info
    json.dump(runs, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    print(f"{'tag':28s} {'ok':3s} {'iter_s':>7s} {'opt(r0)':>8s} {'fwd(r0)':>8s} {'paper':>7s}  topology/placement")
    for tag, r in runs.items():
        r0 = r["ranks"].get(0, {})
        pc = r.get("paper_cost")
        print(f"{tag:28s} {'ok' if r['ok'] else 'ERR':3s} {r['iter_s'] or float('nan'):7.2f} {r0.get('opt', float('nan')):8.2f} "
              f"{r0.get('fwd', float('nan')):8.2f} {pc if pc is not None else float('nan'):7.2f}  "
              f"{r.get('topology', '')}/{r.get('placement', '')}{' ' + r['variant'] if r.get('variant') else ''}"
              + (f"  pairs={len(r['pairs'])}" if "pairs" in r else ""))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        fetch(sys.argv[2])
    elif cmd == "split":
        split(sys.argv[2])
    elif cmd == "summarize":
        summarize()
    else:
        raise SystemExit(__doc__)
