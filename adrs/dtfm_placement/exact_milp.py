"""Exact optimum of the simulator objective via MILP feasibility over (C, D) thresholds (any PP x DP).

Copy of output/agent_cc2/study/exact.py (written by the second agent for N=64), updated 2026-11-15 for the
corrected simulator: gradient size per stage comes from stepsim's RunSpec.params (embedding on stage 0 only,
classification head on the last stage), and N=128 is allowed. The objective depends only on C = slowest
pipeline link used and D = slowest stage-to-hub gradient link, so the optimum is a staircase walk over
(D descending, C ascending) with exact MILP feasibility checks (scipy HiGHS). Details: docs/milp-exact-optimum.md.
  python exact_milp.py N seed [seed ...]  -> prints JSON lines {seed, opt, model, k, timeouts, rank_map}
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reference as REF  # noqa: E402
import stepsim as SS  # noqa: E402  (on sys.path via reference)
import topologies as T  # noqa: E402

T.PARALLELISM.setdefault(128, (8, 16))

TH = dict(zip(["f_layer", "b_layer", "f_emb", "b_emb", "alpha", "beta", "opt", "latf", "win", "dp_eff"], REF.load_theta()))
ACT = 2048 * 2048 * 2


def link(topo, i, j, size, eff=1.0):
    lat = topo["delay_ms"][i][j] / 1e3
    bw = topo["bandwidth_gbps"][i][j] * 1e9 / 8 * eff
    bw = min(bw, TH["win"] * 1e6 / (2 * lat))
    return lat + size / bw


def stage_params(layers, e):
    """Parameters of a stage of class e (0 middle, 1 first, 2 last), as in stepsim.RunSpec.params."""
    return layers * SS.LAYER_PARAMS + (SS.EMB_PARAMS if e == 1 else SS.HEAD_PARAMS if e == 2 else 0)


def stage_class(s, pp):
    return 1 if s == 0 else 2 if s == pp - 1 else 0


def tables(topo):
    regs = topo["regions"]
    names = sorted(set(regs))
    k = len(names)
    members = [[g for g in range(topo["n"]) if regs[g] == nm] for nm in names]
    cnt = [len(m) for m in members]
    pp, layers = topo["pp"], topo["layers"]
    pipe = np.full((k, k), np.inf)
    tdp = np.full((3, k, k), np.inf)  # [stage class: 0 middle, 1 first, 2 last][member][hub]
    for a in range(k):
        for b in range(k):
            i = members[a][0]
            if a == b:
                if cnt[a] < 2:
                    continue
                j = members[a][1]
            else:
                j = members[b][0]
            pipe[a, b] = max(link(topo, i, j, ACT), link(topo, j, i, ACT))
            for e in (0, 1, 2):
                g = stage_params(layers, e) * 2
                tdp[e, a, b] = 2 * link(topo, i, j, g, TH["dp_eff"])
    return k, cnt, members, pipe, tdp


def objective_fn(topo):
    pp, layers = topo["pp"], topo["layers"]
    S = pp
    f = [TH["f_layer"] * layers + (TH["f_emb"] if s in (0, S - 1) else 0) for s in range(S)]
    b = [TH["b_layer"] * layers + (TH["b_emb"] if s in (0, S - 1) else 0) for s in range(S)]
    opt0 = TH["opt"] * max(stage_params(layers, stage_class(s, S)) for s in range(S)) / 1e9
    M = 2 * (64 + S - 1)

    def obj(C, D):
        t = M * (max(max(f), C) + max(max(b), C))
        return t + max(opt0, D) + TH["beta"] * min(opt0, D)
    return obj


def feasible(k, cnt, pipe, tdp, pp, dp, C, D, tl=20):
    nv = dp * pp * k
    idx = lambda d, s, r: (d * pp + s) * k + r  # noqa: E731
    rows = []
    # one region per cell
    A = lil_matrix((dp * pp + k + dp * (pp - 1) * k + pp * (dp - 1) * k, nv))
    lo, hi = [], []
    ri = 0
    for d in range(dp):
        for s in range(pp):
            for r in range(k):
                A[ri, idx(d, s, r)] = 1
            lo.append(1); hi.append(1); ri += 1
    for r in range(k):
        for d in range(dp):
            for s in range(pp):
                A[ri, idx(d, s, r)] = 1
        lo.append(cnt[r]); hi.append(cnt[r]); ri += 1
    for d in range(dp):
        for s in range(pp - 1):
            for a in range(k):
                bad = [b for b in range(k) if pipe[a, b] > C]
                if not bad:
                    continue
                A[ri, idx(d, s, a)] = 1
                for b in bad:
                    A[ri, idx(d, s + 1, b)] = 1
                lo.append(0); hi.append(1); ri += 1
    for s in range(pp):
        e = stage_class(s, pp)
        for d in range(1, dp):
            for h in range(k):
                bad = [r for r in range(k) if tdp[e, r, h] > D]
                if not bad:
                    continue
                A[ri, idx(0, s, h)] = 1
                for r in bad:
                    A[ri, idx(d, s, r)] = 1
                lo.append(0); hi.append(1); ri += 1
    A = A[:ri].tocsr()
    res = milp(np.zeros(nv), constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options={"time_limit": tl})
    if res.status == 0:
        x = res.x.reshape(dp, pp, k)
        return [[int(np.argmax(x[d, s])) for s in range(pp)] for d in range(dp)]
    if res.status == 1:  # time limit: unknown, treat as infeasible but flag
        return "timeout"
    return None


def solve(seed, n=64):
    topo = T.make_topology(n, seed)
    pp, dp = topo["pp"], topo["dp"]
    k, cnt, members, pipe, tdp = tables(topo)
    obj = objective_fn(topo)
    Cs = sorted({v for v in pipe.flatten() if np.isfinite(v)})
    Ds = sorted({v for v in tdp.flatten() if np.isfinite(v)}, reverse=True)
    best = (np.inf, None, None)
    ci = 0
    timeouts = 0
    for D in Ds:
        # smallest C feasible with this D; C index is nondecreasing as D decreases
        while ci < len(Cs):
            if obj(Cs[ci], D) >= best[0]:
                break
            g = feasible(k, cnt, pipe, tdp, pp, dp, Cs[ci], D)
            if g == "timeout":
                timeouts += 1
                g = None
            if g is not None:
                v = obj(Cs[ci], D)
                if v < best[0]:
                    best = (v, g, (Cs[ci], D))
                break
            ci += 1
        if ci >= len(Cs):
            break
    v, g, cd = best
    pools = [list(m) for m in members]
    rm = [0] * n
    for d in range(dp):
        for s in range(pp):
            rm[pools[g[d][s]].pop()] = d * pp + s
    sim = REF.predict(REF.load_theta(), topo, rm)
    return dict(seed=seed, opt=sim, model=v, k=k, timeouts=timeouts, rank_map=rm)


if __name__ == "__main__":
    for s in sys.argv[2:]:
        print(json.dumps(solve(int(s), int(sys.argv[1]))), flush=True)
