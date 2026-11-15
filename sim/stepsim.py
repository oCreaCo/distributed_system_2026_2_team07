"""Calibrated step-time simulator for DT-FM placements (GPipe + central_ps DP).

Model (per training iteration, S pipeline stages, M micro-batches per gradient-accumulation
step, G accumulation steps):

  f_s, b_s        = per-micro-batch forward / backward compute of stage s
                    = (f_layer, b_layer) * layers_s + (f_emb, b_emb) if the stage holds the embedding
  c_s             = activation transfer time on the link s -> s+1 = lat + act_bytes / bw
  overlap(x, y, k)= max(x, y) + k * min(x, y)        (k = 0: perfect overlap, k = 1: none)
  T_pipe          = G * (M + S - 1) * [ max_s overlap(f_s, c_s, alpha) + max_s overlap(b_s, c_s, alpha) ]
  T_dp            = max_s 2 * (lat_dp(s) + grad_bytes_s / bw_dp(s))             (central_ps push + pull)
  T_opt           = overlap(opt_per_gparam * params_max, T_dp, beta)             (CPU-offload optimizer)
  T_iter          = T_pipe + T_opt

Free parameters theta = [f_layer, b_layer, f_emb, b_emb, alpha, beta, opt_per_gparam, lat_fixed_ms, tcp_window_mb,
dp_bw_eff].
Effective link bandwidth is min(bw, tcp_window / RTT): measured iteration time doubled going from
10 ms to 50 ms one-way delay at 1 Gbit, which the paper's pure delay+size/bw model cannot produce.
alpha / beta in [0,1] are the non-overlapped fractions for PP activation transfer vs compute and
for DP gradient transfer vs optimizer work. dp_bw_eff scales the link bandwidth for the large DP gradient transfers
(NCCL pair bench 2026-11-15: 64 MB transfers reach 50-88% of the tc rate; v2, 2026-11-15, together with the
last-stage parameter count). The paper's communication-only cost model
is kept separately in k8s/run_scheduler.py for comparison.
"""
import itertools
import json
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

PARAM_NAMES = ["f_layer", "b_layer", "f_emb", "b_emb", "alpha", "beta", "opt_per_gparam", "lat_fixed_ms", "tcp_window_mb",
               "dp_bw_eff"]
LAYER_PARAMS = 50_358_272          # measured: 302,149,632 / 6 layers (hidden 2048)
EMB_PARAMS = 63_582_208            # measured: 365,731,840 - 302,149,632 (stage 0 only: word embedding)
HEAD_PARAMS = 4_200_450            # measured: 306,350,082 - 302,149,632 (last stage: classification head)
ACT_BYTES = 2048 * 2048 * 2        # seq x hidden x fp16 per micro-batch (micro-batch size 1)
LINK_SUM = False                   # stage comm = max(upstream, downstream) link time; True (sum) fits worse overall


class RunSpec:
    def __init__(self, delay_ms, bw_gbps, rank_map, pp=4, dp=2, micro_batches=64, ga_steps=2,
                 layers=6, offload=True):
        n = len(rank_map)
        assert n == pp * dp
        self.pp, self.dp, self.M, self.G, self.layers, self.offload = pp, dp, micro_batches, ga_steps, layers, offload
        self.delay = None if delay_ms is None else np.asarray(delay_ms, float)
        self.bw = None if bw_gbps is None else np.asarray(bw_gbps, float)
        # pod holding (stage, dp_idx)
        self.pod = {}
        for pod, rank in enumerate(rank_map):
            self.pod[(rank % pp, rank // pp)] = pod

    def link(self, a, b, lat_fixed_ms, tcp_window_mb, bw_eff=1.0):
        """(latency_s, effective bandwidth_bytes_per_s) for pod a -> pod b."""
        if self.delay is None:
            lat, bw = lat_fixed_ms / 1e3, 10e9 / 8     # measured no-tc baseline ~10 Gbit/s, 0.06 ms
        else:
            lat, bw = self.delay[a, b] / 1e3 + lat_fixed_ms / 1e3, self.bw[a, b] * 1e9 / 8 * bw_eff
        rtt = 2 * lat
        if rtt > 0:
            bw = min(bw, tcp_window_mb * 1e6 / rtt)
        return lat, bw

    def params(self, s):
        return self.layers * LAYER_PARAMS + (EMB_PARAMS if s == 0 else HEAD_PARAMS if s == self.pp - 1 else 0)


def overlap(compute, comm, k):
    """Time of two activities that can overlap: the longer one, plus a fraction k of the shorter."""
    return max(compute, comm) + k * min(compute, comm)


def predict_breakdown(theta, spec):
    f_layer, b_layer, f_emb, b_emb, alpha, beta, opt_per_gparam, lat_fixed_ms, tcp_window_mb, dp_bw_eff = theta
    S, M, G = spec.pp, spec.M, spec.G
    fwd_terms, bwd_terms, dp_terms = [], [], []
    for s in range(S):
        emb = s in (0, S - 1)
        f_s = f_layer * spec.layers + (f_emb if emb else 0)
        b_s = b_layer * spec.layers + (b_emb if emb else 0)
        # per micro-batch, a stage both receives on its upstream link and sends on its downstream link;
        # the two are serialized (PP8 zigzag measured 2x the single-crossing time), so they add up.
        # Each link: worst over the DP replicas (pipelines run in lockstep per replica).
        c_s = 0.0
        for t in (s - 1, s + 1):
            if 0 <= t < S:
                worst = 0.0
                for d in range(spec.dp):
                    lat, bw = spec.link(spec.pod[(s, d)], spec.pod[(t, d)], lat_fixed_ms, tcp_window_mb)
                    worst = max(worst, lat + ACT_BYTES / bw)
                c_s = c_s + worst if LINK_SUM else max(c_s, worst)
        # per micro-batch: activation transfer overlaps with compute of the neighbouring micro-batch
        fwd_terms.append(overlap(f_s, c_s, alpha))
        bwd_terms.append(overlap(b_s, c_s, alpha))
        # DP: every replica of stage s exchanges grads/params with the replica-0 "server"
        grad_bytes = spec.params(s) * 2
        for d in range(1, spec.dp):
            lat, bw = spec.link(spec.pod[(s, d)], spec.pod[(s, 0)], lat_fixed_ms, tcp_window_mb, dp_bw_eff)
            dp_terms.append(2 * (lat + grad_bytes / bw))
    t_fwd = G * (M + S - 1) * max(fwd_terms)
    t_bwd = G * (M + S - 1) * max(bwd_terms)
    t_dp = max(dp_terms) if dp_terms else 0.0
    t_opt0 = opt_per_gparam * max(spec.params(s) for s in range(S)) / 1e9 * (1.0 if spec.offload else 0.3)
    # gradient sync overlaps with the (CPU-offloaded) optimizer work; only the excess shows up
    t_opt = overlap(t_opt0, t_dp, beta)
    return {"iter": t_fwd + t_bwd + t_opt,
            "fwd": t_fwd / G, "bwd": t_bwd / G,      # per gradient-accumulation step, as rank 0 logs them
            "opt": t_opt}                            # rank 0's "optimizer step" covers DP sync + optimizer


def predict(theta, spec):
    return predict_breakdown(theta, spec)["iter"]


def fit(specs, measured, theta0=None, breakdowns=None):
    """measured: iteration times; breakdowns (optional): per-run dicts with any of fwd/bwd/opt
    (rank-0 phase means) used as extra relative residuals so the parameters are identifiable."""
    theta0 = theta0 or [0.05, 0.08, 0.01, 0.01, 0.3, 1.0, 8.0, 0.1, 6.0, 1.0]
    lo = [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.3]
    hi = [1, 1, 1, 1, 1.0, 3.0, 100, 5, 1000, 1.0]
    breakdowns = breakdowns or [{} for _ in specs]

    def residuals(th):
        out = []
        for s, m, bd in zip(specs, measured, breakdowns):
            pr = predict_breakdown(th, s)
            out.append((pr["iter"] - m) / m)
            for k in ("fwd", "bwd", "opt"):
                if bd.get(k):
                    out.append(0.5 * (pr[k] - bd[k]) / bd[k])
        return out
    res = least_squares(residuals, theta0, bounds=(lo, hi))
    return list(res.x)


def ranking_metrics(pred, meas):
    pred, meas = np.asarray(pred), np.asarray(meas)
    n = len(pred)
    out = {"n": n, "mape": float(np.mean(np.abs(pred - meas) / meas))}
    if n >= 3:
        out["spearman"] = float(spearmanr(pred, meas).correlation)
    agree = tot = 0
    for i, j in itertools.combinations(range(n), 2):
        if meas[i] == meas[j]:
            continue
        tot += 1
        agree += (pred[i] < pred[j]) == (meas[i] < meas[j])
    out["pairwise_acc"] = agree / tot if tot else None
    out["top1_regret_s"] = float(meas[int(np.argmin(pred))] - meas.min())
    return out


def spec_from_summary(run, topologies):
    args = run.get("args", [])
    def arg(name, default):  # last occurrence wins, as in argparse (common args come first, run args after)
        idx = [i for i, a in enumerate(args) if a == name]
        return type(default)(args[idx[-1] + 1]) if idx else default
    topo = topologies.get(run.get("topology"))
    if run.get("delay_ms") is not None:
        d, b = run["delay_ms"], run["bandwidth_gbps"]
    elif run.get("topology") == "homo":   # homogeneous sweep: every link delay/bw from the manifest meta
        d = np.full((8, 8), float(run["delay"])).tolist()
        b = np.full((8, 8), float(run["bw"])).tolist()
    elif topo:
        d, b = topo["delay_ms"], topo["bandwidth_gbps"]
    else:
        d = b = None
    batch = arg("--batch-size", 64)
    return RunSpec(d, b, run.get("rank_map") or list(range(8)),
                   pp=arg("--pipeline-group-size", 4), dp=arg("--data-group-size", 2),
                   micro_batches=batch, ga_steps=arg("--gradient-accumulate-step", 2),
                   layers=arg("--num-layers", 6), offload=arg("--use-offload", "true").lower() == "true")


COMPUTE_IDX = [0, 1, 2, 3, 6]      # f_layer, b_layer, f_emb, b_emb, opt_per_gparam
NET_IDX = [4, 5, 7, 8, 9]          # alpha, beta, lat_fixed_ms, tcp_window_mb, dp_bw_eff


def fit_two_stage(specs, measured, breakdowns, is_notc, theta0=None):
    """Stage 1: compute + optimizer parameters on runs without tc (network terms cannot matter there).
    Stage 2: network parameters on all runs with the compute parameters frozen. Avoids the trade-off
    where the joint fit shrinks compute to absorb network error."""
    theta = list(theta0 or [0.0066, 0.0095, 0.004, 0.025, 0.0, 0.13, 13.7, 0.0, 9.6, 0.9])
    lo = [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.3]
    hi = [1, 1, 1, 1, 1, 3, 100, 5, 1000, 1.0]

    def residuals(th, idxs):
        out = []
        for s, m, bd in zip(specs, measured, breakdowns):
            if s is None:
                continue
            pr = predict_breakdown(th, s)
            out.append((pr["iter"] - m) / m)
            for k in ("fwd", "bwd", "opt"):
                if bd.get(k):
                    out.append(0.5 * (pr[k] - bd[k]) / bd[k])
        return out

    def sub_fit(idxs, mask):
        sub_specs = [s if keep else None for s, keep in zip(specs, mask)]
        def r(x):
            th = list(theta)
            for i, v in zip(idxs, x):
                th[i] = v
            return [v for s_, m, bd in zip(sub_specs, measured, breakdowns) if s_ is not None
                    for v in _run_residuals(th, s_, m, bd)]
        res = least_squares(r, [theta[i] for i in idxs], bounds=([lo[i] for i in idxs], [hi[i] for i in idxs]), x_scale="jac")
        for i, v in zip(idxs, res.x):
            theta[i] = float(v)

    sub_fit(COMPUTE_IDX, is_notc)
    sub_fit(NET_IDX, [True] * len(specs))
    return theta


def _run_residuals(th, spec, m, bd):
    pr = predict_breakdown(th, spec)
    out = [(pr["iter"] - m) / m]
    for k in ("fwd", "bwd", "opt"):
        if bd.get(k):
            out.append(0.5 * (pr[k] - bd[k]) / bd[k])
    return out
