"""DT-FM launcher for the TA-provided experiment.yaml.

experiment.yaml runs `python /workspace/test.py` on every pod. This script
  1. maps the pod index (JOB_COMPLETION_INDEX) to a DT-FM rank via rank_map,
  2. applies pairwise tc (netem) rules on eth0 from the delay/bandwidth matrices,
  3. runs dist_runner.py (or nccl_bench.py) and tees its output to /workspace/results/<tag>/.

/workspace/run_config.json holds either a single run or {"runs": [...], "common_args": [...]}
so many runs execute back to back in the same pods (no pod re-creation between runs).
Only the standard library is used here; DT-FM itself runs in /workspace/venv.
"""
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import time

CONFIG_PATH = os.environ.get("DTFM_CONFIG", "/workspace/run_config.json")
DTFM_DIR = "/workspace/DT-FM"
VENV_PYTHON = "/workspace/venv/bin/python"
NCCL_LIB = "/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib"
DEB_DIR = "/workspace/debs"
RESULTS_DIR = "/workspace/results"
BENCH_SCRIPT = "/workspace/nccl_bench.py"
IFACE = "eth0"
BASE_PORT = 29500
RUN_TIMEOUT_S = 900  # a hung rank must not block the rest of the batch
# prio bands 1:1-1:3 keep the default priomap (unfiltered traffic, e.g. DNS),
# so netem classes start at 1:4. prio supports at most 16 bands.
FIRST_NETEM_BAND = 4
MAX_PRIO_BANDS = 16


def log(msg):
    print(f"[test.py] {msg}", flush=True)


def run(cmd, check=True):
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check)


def resolve_peers(world_size, timeout_s=600):
    """Wait until every experiment-<i>.experiment has a DNS record; return their IPs."""
    ips = [None] * world_size
    deadline = time.time() + timeout_s
    while None in ips:
        for i in range(world_size):
            if ips[i] is None:
                try:
                    ips[i] = socket.gethostbyname(f"experiment-{i}.experiment")
                except socket.gaierror:
                    pass
        if None in ips:
            if time.time() > deadline:
                raise RuntimeError(f"could not resolve peers: {ips}")
            log("waiting for pods: " + ", ".join(f"experiment-{i}" for i, ip in enumerate(ips) if ip is None))
            time.sleep(10)
    return ips


def ensure_tc():
    if shutil.which("tc"):
        return
    debs = sorted(glob.glob(os.path.join(DEB_DIR, "*.deb")))
    if debs:
        run(["dpkg", "-i", *debs], check=False)
    if not shutil.which("tc"):
        run(["apt-get", "update"])
        run(["apt-get", "install", "-y", "--no-install-recommends", "iproute2"])


def build_tc_commands(my_idx, peer_ips, delay_ms, bandwidth_gbps):
    """Same scheme as DT-FM's scheduler/generate_heterogeneous_tc.py, adapted to pods:
    one netem class per distinct (delay, bandwidth) pair, u32 filter per destination IP.
    Rules shape egress only, as in the original."""
    classes = {}
    for j, ip in enumerate(peer_ips):
        if j != my_idx:
            key = (delay_ms[my_idx][j], bandwidth_gbps[my_idx][j])
            classes.setdefault(key, FIRST_NETEM_BAND + len(classes))
    bands = FIRST_NETEM_BAND - 1 + len(classes)
    assert bands <= MAX_PRIO_BANDS, f"too many distinct (delay, bandwidth) pairs: {len(classes)}"

    cmds = [["tc", "qdisc", "add", "dev", IFACE, "root", "handle", "1:", "prio", "bands", str(bands)]]
    for (delay, bw), band in classes.items():
        limit_pkts = max(1000, int(delay * 22500 * bw))
        # class / handle minor numbers are hexadecimal (band 10 is "1:a", not "1:10")
        cmds.append(["tc", "qdisc", "add", "dev", IFACE, "parent", f"1:{band:x}", "handle", f"{band:x}0:",
                     "netem", "delay", f"{delay}ms", "rate", f"{bw}Gbit", "limit", str(limit_pkts)])
    for j, ip in enumerate(peer_ips):
        if j != my_idx:
            band = classes[(delay_ms[my_idx][j], bandwidth_gbps[my_idx][j])]
            cmds.append(["tc", "filter", "add", "dev", IFACE, "parent", "1:0", "protocol", "ip", "prio", "1",
                         "u32", "match", "ip", "dst", f"{ip}/32", "flowid", f"1:{band:x}"])
    return cmds


def apply_tc(idx, peer_ips, delay_ms, bandwidth_gbps):
    run(["tc", "qdisc", "del", "dev", IFACE, "root"], check=False)
    if delay_ms is None or bandwidth_gbps is None:
        log("no delay/bandwidth matrices for this run: no tc")
        return
    ensure_tc()
    for cmd in build_tc_commands(idx, peer_ips, delay_ms, bandwidth_gbps):
        run(cmd)
    run(["tc", "qdisc", "show", "dev", IFACE])


def tee(cmd, cwd, env, log_path, timeout_s):
    """Run cmd, mirroring its output to stdout and log_path; kill it after timeout_s even if silent."""
    import threading
    log("$ " + " ".join(cmd))
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)

        def pump():
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
        t = threading.Thread(target=pump, daemon=True)
        t.start()
        try:
            ret = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log(f"run exceeded {timeout_s}s: killing")
            proc.kill()
            ret = proc.wait()
        t.join(timeout=30)
        return ret


def file_barrier(sync_dir, run_idx, idx, world_size, started, timeout_s=1500):
    """All pods wait here before each run, so a run that timed out on one pod cannot leave the
    pods on different run indices (which would deadlock every later run). Files older than this
    pod's start are ignored, so leftovers from an earlier job with the same config do not count."""
    d = os.path.join(sync_dir, f"run{run_idx}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, f"pod{idx}"), "w").write(str(time.time()))
    deadline = time.time() + timeout_s
    while True:
        present = [i for i in range(world_size)
                   if os.path.exists(os.path.join(d, f"pod{i}")) and os.path.getmtime(os.path.join(d, f"pod{i}")) >= started - 300]
        if len(present) == world_size:
            return
        if time.time() > deadline:
            raise RuntimeError(f"barrier timeout before run {run_idx}: only pods {present}")
        log(f"barrier run {run_idx}: waiting for pods {[i for i in range(world_size) if i not in present]}")
        time.sleep(5)


def main():
    started = time.time()
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    import hashlib
    sync_dir = os.path.join(RESULTS_DIR, "_sync", hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12])
    runs = cfg.get("runs") or [cfg]
    common_args = cfg.get("common_args", [])
    run_timeout_s = cfg.get("run_timeout_s", RUN_TIMEOUT_S)

    idx = int(os.environ["JOB_COMPLETION_INDEX"])
    world_size = int(os.environ["WORLD_SIZE"])
    log(f"pod={idx} host={socket.gethostname()} runs={len(runs)}")

    peer_ips = resolve_peers(world_size)
    log(f"peer ips (by pod index): {peer_ips}")
    if any(r.get("delay_ms") is not None for r in runs):
        ensure_tc()

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = NCCL_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    statuses = []
    for run_idx, r in enumerate(runs):
        tag = r["tag"]
        # rank_map[pod index] = DT-FM rank. Identity unless a scheduler assignment is given.
        rank_map = r.get("rank_map") or list(range(world_size))
        assert sorted(rank_map) == list(range(world_size)), f"{tag}: rank_map must be a permutation"
        rank = rank_map[idx]
        master_idx = rank_map.index(0)  # the pod holding rank 0 hosts the gloo TCP store
        port = BASE_PORT + run_idx  # fresh port per run avoids TIME_WAIT collisions on the master
        master = f"experiment-{master_idx}.experiment"
        file_barrier(sync_dir, run_idx, idx, world_size, started)
        log(f"=== run {run_idx + 1}/{len(runs)} tag={tag} rank={rank} master={master}:{port}")

        out_dir = os.path.join(RESULTS_DIR, tag)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, f"rank{rank}_pod{idx}.log")
        run_env = dict(env, RANK=str(rank), WORLD_SIZE=str(world_size), MASTER_ADDR=master, MASTER_PORT=str(port))
        try:
            apply_tc(idx, peer_ips, r.get("delay_ms"), r.get("bandwidth_gbps"))
            if r.get("type") == "bench":
                cmd = [VENV_PYTHON, "-u", BENCH_SCRIPT, "--out", os.path.join(out_dir, "bench.json"),
                       *r.get("args", [])]
                ret = tee(cmd, "/workspace", run_env, log_path, 300)
            else:
                os.makedirs(os.path.join(DTFM_DIR, "trace_json"), exist_ok=True)
                cmd = [VENV_PYTHON, "-u", "dist_runner.py",
                       "--dist-url", f"tcp://{master}:{port}",
                       "--world-size", str(world_size), "--rank", str(rank), "--cuda-id", "0",
                       "--trace-postfix", tag,
                       *common_args, *r.get("args", [])]
                ret = tee(cmd, DTFM_DIR, run_env, log_path, run_timeout_s)
        except Exception as e:  # e.g. a tc command rejected: record it and move on to the next run
            log(f"run {tag} failed before/while launching: {e!r}")
            ret = -1
        log(f"run {tag} exited with {ret}")
        statuses.append((tag, ret))
        with open(os.path.join(out_dir, f"status_pod{idx}.txt"), "w") as f:
            f.write(str(ret))

    run(["tc", "qdisc", "del", "dev", IFACE, "root"], check=False)
    log("batch done: " + ", ".join(f"{t}={c}" for t, c in statuses))
    sys.exit(0 if all(c == 0 for _, c in statuses) else 1)


if __name__ == "__main__":
    main()
