"""GPU placement for DT-FM: assign N GPUs to (pipeline stage, data-parallel replica).

Bottleneck-threshold construction on a class-level grid, then budgeted refinement with cost_fn.

1. GPUs with identical link profiles are grouped into classes (regions); any two GPUs of a class are
   interchangeable, so a placement is an dp x pp grid of class labels (row = replica, column = stage).
2. The iteration time is dominated by two bottlenecks: the slowest pipeline link used by any row
   (paid on every micro-batch) and the slowest gradient link from any GPU of a column to that
   column's hub (row 0). For every pipeline threshold C, a feasibility search (simulated annealing
   on violation counts) finds the smallest DP threshold D that admits a grid; each Pareto point
   (C, D) becomes a candidate.
3. Candidates are scored with cost_fn; the best ones are refined by cost_fn hill climbing over
   swaps between GPUs of different classes (swaps within a class are no-ops and are skipped).
"""
import math
import random
import time


# EVOLVE-BLOCK-START
def schedule(delay_ms, bandwidth_gbps, pp, dp, cost_fn, budget):
    n = pp * dp
    t_start = time.time()
    time_cap = 22.0
    rng = random.Random(12345)
    if n <= 1:
        return list(range(n))

    calls = 0
    best = [float("inf"), list(range(n))]
    seen = {}

    def cost(rm):
        nonlocal calls
        key = tuple(rm)
        if key in seen:
            return seen[key]
        if calls >= budget:
            return float("inf")
        calls += 1
        c = cost_fn(list(rm))
        seen[key] = c
        if c < best[0]:
            best[0] = c
            best[1] = list(rm)
        return c

    # ---- link model: effective bandwidth limited by a ~10 MB TCP window per RTT ----
    def link_time(i, j, size_bytes):
        lat = delay_ms[i][j] / 1e3
        bw = bandwidth_gbps[i][j] * 1e9 / 8
        if lat > 0:
            bw = min(bw, 10e6 / (2 * lat))
        return lat + size_bytes / max(bw, 1.0)

    # ---- classes of interchangeable GPUs ----
    def same(i, j):
        for k in range(n):
            if k == i or k == j:
                continue
            if (delay_ms[i][k] != delay_ms[j][k] or delay_ms[k][i] != delay_ms[k][j]
                    or bandwidth_gbps[i][k] != bandwidth_gbps[j][k] or bandwidth_gbps[k][i] != bandwidth_gbps[k][j]):
                return False
        return True

    classes = []
    for g in range(n):
        for cl in classes:
            r = cl[0]
            if same(g, r) and (len(cl) < 2 or (
                    delay_ms[g][r] == delay_ms[cl[1]][r] and delay_ms[r][g] == delay_ms[r][cl[1]]
                    and bandwidth_gbps[g][r] == bandwidth_gbps[cl[1]][r]
                    and bandwidth_gbps[r][g] == bandwidth_gbps[r][cl[1]])):
                cl.append(g)
                break
        else:
            classes.append([g])
    k = len(classes)
    lab = [0] * n
    for a, cl in enumerate(classes):
        for g in cl:
            lab[g] = a
    cnt = [len(cl) for cl in classes]

    ACT = 8e6
    STEPS = 20000
    SA_STEPS = 60000
    GRAD = 300e6
    INF = float("inf")
    pipe = [[INF] * k for _ in range(k)]
    dlat = [[INF] * k for _ in range(k)]
    dinv = [[INF] * k for _ in range(k)]
    for a in range(k):
        for b in range(k):
            i = classes[a][0]
            if a == b:
                if cnt[a] < 2:
                    continue
                j = classes[a][1]
            else:
                j = classes[b][0]
            pipe[a][b] = max(link_time(i, j, ACT), link_time(j, i, ACT))
            # replica -> hub: latency and seconds per byte
            dlat[a][b] = link_time(i, j, 0.0)
            dinv[a][b] = link_time(i, j, 1.0) - dlat[a][b]

    def grid_to_rm(grid):
        pools = [list(cl) for cl in classes]
        rm = [0] * n
        for d in range(dp):
            for s in range(pp):
                rm[pools[grid[d][s]].pop()] = d * pp + s
        return rm

    def rm_to_grid(rm):
        grid = [[0] * pp for _ in range(dp)]
        for g, r in enumerate(rm):
            grid[r // pp][r % pp] = lab[g]
        return grid

    # ---- feasibility search for thresholds (C, D) ----
    def row_viol(grid, d, C):
        row = grid[d]
        v = 0
        for s in range(pp - 1):
            if pipe[row[s]][row[s + 1]] > C:
                v += 1
        return v

    coltab = [None] * pp

    def col_viol(grid, s, D):
        h = grid[0][s]
        tab = coltab[s]
        v = 0
        for d in range(1, dp):
            if tab[grid[d][s]][h] > D:
                v += 1
        return v

    def total_viol(grid, C, D):
        return sum(row_viol(grid, d, C) for d in range(dp)) + sum(col_viol(grid, s, D) for s in range(pp))

    def feasible(grid0, C, D, steps):
        grid = [r[:] for r in grid0]
        v = total_viol(grid, C, D)
        if v == 0:
            return grid
        temp0 = 0.6
        for it in range(steps):
            temp = temp0 * (1 - it / steps) + 0.05
            if rng.random() < 0.05:
                # change the hub row: swap row 0 with another row
                d2 = rng.randrange(1, dp) if dp > 1 else 0
                if d2 == 0:
                    continue
                before = row_viol(grid, 0, C) + row_viol(grid, d2, C) + sum(col_viol(grid, s, D) for s in range(pp))
                grid[0], grid[d2] = grid[d2], grid[0]
                after = row_viol(grid, 0, C) + row_viol(grid, d2, C) + sum(col_viol(grid, s, D) for s in range(pp))
                delta = after - before
                if delta <= 0 or rng.random() < math.exp(-delta / temp):
                    v += delta
                    if v == 0:
                        return grid
                else:
                    grid[0], grid[d2] = grid[d2], grid[0]
                continue
            d1 = rng.randrange(dp); s1 = rng.randrange(pp)
            d2 = rng.randrange(dp); s2 = rng.randrange(pp)
            a, b = grid[d1][s1], grid[d2][s2]
            if a == b:
                continue
            rows = {d1, d2}
            if d1 == 0 or d2 == 0:
                cols = set(range(pp))
            else:
                cols = {s1, s2}
            before = sum(row_viol(grid, d, C) for d in rows) + sum(col_viol(grid, s, D) for s in cols)
            grid[d1][s1], grid[d2][s2] = b, a
            after = sum(row_viol(grid, d, C) for d in rows) + sum(col_viol(grid, s, D) for s in cols)
            delta = after - before
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                v += delta
                if v == 0:
                    return grid
            else:
                grid[d1][s1], grid[d2][s2] = a, b
        return None

    pipe_vals = sorted({pipe[x][y] for x in range(k) for y in range(k) if pipe[x][y] < INF})

    # initial grid: classes laid out row-major along a greedy nearest-neighbour order of classes,
    # so rows are mostly single-class and crossings join cheap pairs
    order = [max(range(k), key=lambda x: cnt[x])]
    while len(order) < k:
        last = order[-1]
        order.append(min((x for x in range(k) if x not in order), key=lambda x: pipe[last][x]))
    base = []
    for x in order:
        base += [x] * cnt[x]
    grid0 = [base[d * pp:(d + 1) * pp] for d in range(dp)]

    steps = STEPS if n >= 32 else 5000
    # smallest feasible pipeline threshold (no DP constraint)
    for s_ in range(pp):
        coltab[s_] = dlat
    ci0 = 0
    while ci0 < len(pipe_vals):
        g = feasible(grid0, pipe_vals[ci0], INF, steps)
        if g is not None:
            grid0 = g
            break
        ci0 += 1

    candidates = []
    # hypotheses on per-stage gradient size: the first and last stage may hold extra (embedding /
    # output) parameters; cost_fn decides which hypothesis fits
    for extra in (0.0, 0.5):
        if time.time() - t_start > time_cap * 0.3:
            break
        mid = [[dlat[x][y] + GRAD * dinv[x][y] for y in range(k)] for x in range(k)]
        end = [[dlat[x][y] + GRAD * (1 + extra) * dinv[x][y] for y in range(k)] for x in range(k)]
        for s_ in range(pp):
            coltab[s_] = end if s_ in (0, pp - 1) else mid
        dp_vals = sorted({t[x][y] for t in (mid, end) for x in range(k) for y in range(k) if t[x][y] < INF})
        grid = grid0
        ci = ci0
        di = len(dp_vals) - 1
        while ci < len(pipe_vals) and time.time() - t_start < time_cap * 0.3:
            C = pipe_vals[ci]
            improved = False
            while di >= 0:
                g = feasible(grid, C, dp_vals[di], steps)
                if g is None:
                    break
                grid = g
                improved = True
                di -= 1
            if improved:
                candidates.append((C, dp_vals[di + 1], [r[:] for r in grid]))
            if di < 0:
                break
            ci += 1

    # ---- score candidates with cost_fn (also row-reversed variants) ----
    scored = []
    for C, D, g in candidates:
        rm = grid_to_rm(g)
        c = cost(rm)
        scored.append((c, g))
        rg = [r[::-1] for r in g]
        rm2 = grid_to_rm(rg)
        c2 = cost(rm2)
        scored.append((c2, rg))
    if not scored:
        scored.append((cost(grid_to_rm(grid0)), grid0))
    scored.sort(key=lambda x: x[0])

    # ---- surrogate: T ~ a * (worst pipeline link) + b * (worst gradient link) + c, fitted to the
    # cost_fn values seen so far; the end-stage gradient factor is chosen by goodness of fit ----
    def tables_for(extra):
        mid = [[dlat[x][y] + GRAD * dinv[x][y] for y in range(k)] for x in range(k)]
        end = [[dlat[x][y] + GRAD * (1 + extra) * dinv[x][y] for y in range(k)] for x in range(k)]
        return [end if s_ in (0, pp - 1) else mid for s_ in range(pp)]

    def rowmax(row):
        m = 0.0
        for s_ in range(pp - 1):
            v = pipe[row[s_]][row[s_ + 1]]
            if v > m:
                m = v
        return m

    def colmax(g, s_, tabs):
        h = g[0][s_]
        tab = tabs[s_]
        m = 0.0
        for d in range(1, dp):
            v = tab[g[d][s_]][h]
            if v > m:
                m = v
        return m

    def parts(g, tabs):
        return max(rowmax(r) for r in g), max(colmax(g, s_, tabs) for s_ in range(pp))

    # sample a few perturbed candidates for a better-conditioned fit
    for c0, g0 in list(scored[:4]):
        for _ in range(4):
            g = [r[:] for r in g0]
            for _ in range(2):
                d1 = rng.randrange(dp); s1 = rng.randrange(pp); d2 = rng.randrange(dp); s2 = rng.randrange(pp)
                g[d1][s1], g[d2][s2] = g[d2][s2], g[d1][s1]
            scored.append((cost(grid_to_rm(g)), g))
    scored.sort(key=lambda x: x[0])

    def lstsq3(X, y):
        # normal equations for 3 parameters
        A = [[sum(X[i][p] * X[i][q] for i in range(len(X))) for q in range(3)] for p in range(3)]
        bvec = [sum(X[i][p] * y[i] for i in range(len(X))) for p in range(3)]
        M = [A[r][:] + [bvec[r]] for r in range(3)]
        for col in range(3):
            piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
            if abs(M[piv][col]) < 1e-12:
                return None
            M[col], M[piv] = M[piv], M[col]
            for r in range(3):
                if r != col:
                    f = M[r][col] / M[col][col]
                    for q in range(col, 4):
                        M[r][q] -= f * M[col][q]
        return [M[r][3] / M[r][r] for r in range(3)]

    fit = None
    for extra in (0.0, 0.25, 0.5, 0.75):
        tabs = tables_for(extra)
        X, y = [], []
        for c0, g in scored:
            if c0 < INF:
                C_, D_ = parts(g, tabs)
                X.append([C_, D_, 1.0]); y.append(c0)
        if len(X) < 4:
            continue
        w = lstsq3(X, y)
        if w is None or w[0] <= 0 or w[1] <= 0:
            continue
        res = sum((w[0] * X[i][0] + w[1] * X[i][1] + w[2] - y[i]) ** 2 for i in range(len(X)))
        if fit is None or res < fit[0]:
            fit = (res, w, tabs)
    if fit is None:
        fit = (0, [256.0, 2.0, 0.0], tables_for(0.5))
    _, (wa, wb, _), tabs = fit

    def sa_surrogate(g0, steps):
        g = [r[:] for r in g0]
        rmx = [rowmax(r) for r in g]
        cmx = [colmax(g, s_, tabs) for s_ in range(pp)]
        eps = 1e-3

        def f(rmx, cmx):
            return wa * max(rmx) + wb * max(cmx) + eps * (wa * sum(rmx) + wb * sum(cmx)) / (dp + pp)

        cur = f(rmx, cmx)
        bestv, bestg = cur, [r[:] for r in g]
        temp0 = 0.01 * cur
        for it in range(steps):
            temp = temp0 * (1 - it / steps) + 1e-9
            m = rng.random()
            if m < 0.85:
                d1 = rng.randrange(dp); s1 = rng.randrange(pp)
                d2 = rng.randrange(dp); s2 = rng.randrange(pp)
                a, b = g[d1][s1], g[d2][s2]
                if a == b:
                    continue
                g[d1][s1], g[d2][s2] = b, a
                nr = rmx[:]
                nr[d1] = rowmax(g[d1]); nr[d2] = rowmax(g[d2])
                nc = cmx[:]
                if d1 == 0 or d2 == 0:
                    nc = [colmax(g, s_, tabs) for s_ in range(pp)]
                else:
                    nc[s1] = colmax(g, s1, tabs); nc[s2] = colmax(g, s2, tabs)
                nv = f(nr, nc)
                if nv <= cur or rng.random() < math.exp((cur - nv) / temp):
                    cur, rmx, cmx = nv, nr, nc
                else:
                    g[d1][s1], g[d2][s2] = a, b
            else:
                d2 = rng.randrange(1, dp) if dp > 1 else 0
                if d2 == 0:
                    continue
                g[0], g[d2] = g[d2], g[0]
                nr = rmx[:]
                nr[0], nr[d2] = nr[d2], nr[0]
                nc = [colmax(g, s_, tabs) for s_ in range(pp)]
                nv = f(nr, nc)
                if nv <= cur or rng.random() < math.exp((cur - nv) / temp):
                    cur, rmx, cmx = nv, nr, nc
                else:
                    g[0], g[d2] = g[d2], g[0]
            if cur < bestv - 1e-12:
                bestv, bestg = cur, [r[:] for r in g]
        return bestv, bestg

    starts = []
    for c0, g in scored:
        if all(g != h for h in starts):
            starts.append(g)
        if len(starts) >= 3:
            break
    runs = 0
    while time.time() - t_start < time_cap * 0.7 and runs < 12:
        g0 = starts[runs % len(starts)]
        _, g = sa_surrogate(g0, SA_STEPS)
        scored.append((cost(grid_to_rm(g)), g))
        scored.append((cost(grid_to_rm([r[::-1] for r in g])), [r[::-1] for r in g]))
        runs += 1
    scored.sort(key=lambda x: x[0])

    # ---- refinement with cost_fn: class-level swaps, hub-row and column-order moves ----
    def refine(g, c, max_calls):
        g = [r[:] for r in g]
        stop = calls + max_calls
        stall = 0
        while calls < min(stop, budget) and time.time() - t_start < time_cap and stall < 4 * n:
            m = rng.random()
            if m < 0.8:
                d1 = rng.randrange(dp); s1 = rng.randrange(pp)
                d2 = rng.randrange(dp); s2 = rng.randrange(pp)
                if g[d1][s1] == g[d2][s2]:
                    continue
                g[d1][s1], g[d2][s2] = g[d2][s2], g[d1][s1]
                c2 = cost(grid_to_rm(g))
                if c2 < c:
                    c = c2; stall = 0
                else:
                    g[d1][s1], g[d2][s2] = g[d2][s2], g[d1][s1]; stall += 1
            elif m < 0.9:
                d1 = rng.randrange(1, dp) if dp > 1 else 0
                g[0], g[d1] = g[d1], g[0]
                c2 = cost(grid_to_rm(g))
                if c2 < c:
                    c = c2; stall = 0
                else:
                    g[0], g[d1] = g[d1], g[0]; stall += 1
            else:
                d1 = rng.randrange(dp)
                s1, s2 = sorted(rng.sample(range(pp), 2))
                g[d1][s1:s2 + 1] = g[d1][s1:s2 + 1][::-1]
                c2 = cost(grid_to_rm(g))
                if c2 < c:
                    c = c2; stall = 0
                else:
                    g[d1][s1:s2 + 1] = g[d1][s1:s2 + 1][::-1]; stall += 1
        return c, g

    top = scored[:3]
    remaining = budget - calls
    for i, (c, g) in enumerate(top):
        share = remaining // 2 if i == 0 else remaining // 4
        refine(g, c, share)
    if calls < budget:
        refine(rm_to_grid(best[1]), best[0], budget - calls)
    return best[1]
# EVOLVE-BLOCK-END
