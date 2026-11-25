"""GPU placement for DT-FM: assign N GPUs to (pipeline stage, data-parallel replica).

Multi-start structure-aware construction (cluster GPUs by region, match stages, order
pipelines), refined by a cheap TCP-window-aware proxy, then budgeted simulated annealing
with structure-preserving moves.
"""
import random
import math


# EVOLVE-BLOCK-START
def schedule(delay_ms, bandwidth_gbps, pp, dp, cost_fn, budget):
    """Return a rank_map (a permutation of 0..pp*dp-1) with low predicted iteration time."""
    n = pp * dp
    rng = random.Random(0x5EED1234)
    calls = 0
    best_rm = None
    best_c = float('inf')

    def cost(rm):
        nonlocal calls, best_rm, best_c
        calls += 1
        c = cost_fn(rm)
        if c < best_c:
            best_c = c
            best_rm = list(rm)
        return c

    if n <= 1:
        return list(range(n))

    # ---- link model: TCP-window limited effective bandwidth ----------------
    def ebw(i, j):
        if i == j:
            return 1e9
        rtt = delay_ms[i][j] + delay_ms[j][i]
        b = min(bandwidth_gbps[i][j], bandwidth_gbps[j][i])
        cap = 80.0 / max(rtt, 1e-6)   # Gbps: ~10 MB per round-trip time
        return max(min(b, cap), 1e-6)

    def ltime(i, j, size_mb):
        if i == j:
            return 0.0
        return 8.0 * size_mb / ebw(i, j)

    D = [[ltime(i, j, 8.0) for j in range(n)] for i in range(n)]

    def proxy(rm):
        r2g = [0] * n
        for g, r in enumerate(rm):
            r2g[r] = g
        tot = 0.0
        gsize = 400.0 / pp          # gradient bytes per stage
        for s in range(pp):
            agg = r2g[s]
            mx = 0.0
            for r in range(1, dp):
                t = ltime(r2g[r * pp + s], agg, gsize)
                if t > mx:
                    mx = t
            tot += mx               # workers send to replica 0 concurrently
        for s in range(pp - 1):
            mx = 0.0
            for r in range(dp):
                a = r2g[r * pp + s]
                b = r2g[r * pp + s + 1]
                t = ltime(a, b, 8.0)
                if t > mx:
                    mx = t
            tot += 4.0 * mx         # micro-batches mostly overlap with compute
        return tot

    def recenter(rm):
        # For each stage, choose the aggregator (replica 0) that minimizes the
        # full proxy, so DP gradient-sync and pipeline links are both considered.
        cur = list(rm)
        curp = proxy(cur)
        for s in range(pp):
            members = [g for g in range(n) if cur[g] % pp == s]
            if len(members) < 2:
                continue
            center = None
            for g in members:
                if cur[g] == s:
                    center = g
                    break
            if center is None:
                continue
            best_g = center
            best_p = curp
            for g in members:
                if g == center:
                    continue
                cur[g], cur[center] = cur[center], cur[g]
                p = proxy(cur)
                if p < best_p:
                    best_p = p
                    best_g = g
                cur[g], cur[center] = cur[center], cur[g]
            if best_g != center:
                cur[best_g], cur[center] = cur[center], cur[best_g]
                curp = best_p
        return cur

    # ---- clustering --------------------------------------------------------
    def cluster(gsize, start, jitter):
        unassigned = set(range(n))
        groups = []
        order = [start] + [x for x in range(n) if x != start]
        for s0 in order:
            if s0 not in unassigned:
                continue
            grp = [s0]
            unassigned.discard(s0)
            while len(grp) < gsize and unassigned:
                bestg = None
                bestd = float('inf')
                for u in unassigned:
                    dd = sum(D[u][g] for g in grp)
                    if jitter:
                        dd *= (1.0 + jitter * rng.random())
                    if dd < bestd:
                        bestd = dd
                        bestg = u
                grp.append(bestg)
                unassigned.discard(bestg)
            groups.append(grp)
        return groups

    def cluster_bn(gsize, start):
        # grow groups by minimizing the WORST link to the group (bottleneck),
        # which handles outlier GPUs better than summing all links.
        unassigned = set(range(n))
        groups = []
        order = [start] + [x for x in range(n) if x != start]
        for s0 in order:
            if s0 not in unassigned:
                continue
            grp = [s0]
            unassigned.discard(s0)
            while len(grp) < gsize and unassigned:
                bestg = None
                bestd = float('inf')
                for u in unassigned:
                    dd = max(D[u][g] for g in grp)
                    if dd < bestd:
                        bestd = dd
                        bestg = u
                grp.append(bestg)
                unassigned.discard(bestg)
            groups.append(grp)
        return groups

    def arrange_columns(groups):
        m = len(groups)

        def cross(a, b):
            return sum(D[x][y] for x in groups[a] for y in groups[b])

        remaining = list(range(m))
        order = [remaining.pop(0)]
        while remaining:
            last = order[-1]
            nxt = min(remaining, key=lambda g: cross(last, g))
            order.append(nxt)
            remaining.remove(nxt)
        gs = [groups[g] for g in order]
        col_of = {}
        assign = list(gs[0])
        for r in range(dp):
            col_of[assign[r]] = (0, r)
        for s in range(1, pp):
            rem = list(gs[s])
            newassign = [None] * dp
            # bottleneck matching: minimize the worst adjacent-stage link
            edges = []
            for r in range(dp):
                prev = assign[r]
                for t in range(len(rem)):
                    edges.append((D[prev][rem[t]], r, t))
            edges.sort(key=lambda e: e[0])
            used_r = [False] * dp
            used_t = [False] * len(rem)
            for d, r, t in edges:
                if not used_r[r] and not used_t[t]:
                    used_r[r] = True
                    used_t[t] = True
                    newassign[r] = rem[t]
                    col_of[rem[t]] = (s, r)
            assign = newassign
        rm = [0] * n
        for g, (s, r) in col_of.items():
            rm[g] = r * pp + s
        return rm

    def arrange_rows(groups):
        rows = []
        for grp in groups:
            rem = list(grp)
            start = rem.pop(0)
            path = [start]
            while rem:
                last = path[-1]
                bi = min(range(len(rem)), key=lambda t: D[last][rem[t]])
                path.append(rem.pop(bi))
            rows.append(path)

        def cross(a, b):
            return sum(D[x][y] for x in rows[a] for y in rows[b])

        m = len(rows)
        remaining = list(range(m))
        order = [remaining.pop(0)]
        while remaining:
            last = order[-1]
            nxt = min(remaining, key=lambda g: cross(last, g))
            order.append(nxt)
            remaining.remove(nxt)
        rm = [0] * n
        for replica, ri in enumerate(order):
            for s, g in enumerate(rows[ri]):
                rm[g] = replica * pp + s
        return rm

    # ---- candidate generation ---------------------------------------------
    cands = []
    seen = set()

    def add(rm):
        rm = recenter(rm)
        key = tuple(rm)
        if key not in seen:
            seen.add(key)
            cands.append(rm)

    use_bn = n <= 16   # bottleneck-clustering helps outlier-GPU small topologies
    add(arrange_columns(cluster(dp, 0, 0.0)))
    add(arrange_rows(cluster(pp, 0, 0.0)))
    if use_bn:
        add(arrange_columns(cluster_bn(dp, 0)))
        add(arrange_rows(cluster_bn(pp, 0)))
    nvar = 120
    for _ in range(nvar):
        st = rng.randrange(n)
        jit = 0.25
        add(arrange_columns(cluster(dp, st, jit)))
        add(arrange_rows(cluster(pp, st, jit)))
        if use_bn:
            add(arrange_columns(cluster_bn(dp, st)))
            add(arrange_rows(cluster_bn(pp, st)))

    # ---- cheap proxy refinement of candidates -----------------------------
    def proxy_polish(rm, iters):
        cur = list(rm)
        curp = proxy(cur)
        for _ in range(iters):
            x, y = rng.sample(range(n), 2)
            cur[x], cur[y] = cur[y], cur[x]
            p = proxy(cur)
            if p < curp:
                curp = p
            else:
                cur[x], cur[y] = cur[y], cur[x]
        return cur

    def proxy_polish_structured(rm, iters):
        cur = list(rm)
        curp = proxy(cur)
        for _ in range(iters):
            if rng.random() < 0.5:
                s = rng.randrange(pp)
                members = [g for g in range(n) if cur[g] % pp == s]
            else:
                r = rng.randrange(dp)
                members = [g for g in range(n) if cur[g] // pp == r]
            if len(members) < 2:
                continue
            x, y = rng.sample(members, 2)
            cur[x], cur[y] = cur[y], cur[x]
            p = proxy(cur)
            if p < curp:
                curp = p
            else:
                cur[x], cur[y] = cur[y], cur[x]
        return cur

    ref = [proxy_polish_structured(proxy_polish(rm, 300), 200) for rm in cands]
    cands = []
    seen = set()
    for rm in ref:
        add(rm)

    cands.sort(key=proxy)
    # extra proxy refinement focused on the most promising candidates
    extra = min(len(cands), max(8, 2 * (dp + pp)))
    for i in range(extra):
        cands[i] = proxy_polish_structured(cands[i], 900)
    cands.sort(key=proxy)
    polish = max(1, budget // 4)
    top = cands[:max(1, budget - polish)]
    scored = []
    for rm in top:
        if calls >= budget - polish:
            break
        c = cost(rm)
        scored.append((c, list(rm)))

    if best_rm is None:
        best_rm = list(range(n))
        if calls < budget:
            c = cost(best_rm)
            scored.append((c, list(best_rm)))

    # ---- local search: multi-start SA with structure-preserving moves ------
    reserve = max(1, budget // 8)

    def sa_run(start_rm, start_c, iters):
        nonlocal best_c, best_rm
        cur = list(start_rm)
        cur_c = start_c
        T0 = max(1e-9, start_c * 0.03)
        q = max(1, iters // 4)
        it = 0
        while it < iters and calls < budget - reserve:
            it += 1
            mode = rng.random()
            if mode < 0.5 or n < 4:
                x, y = rng.sample(range(n), 2)
            elif mode < 0.75:
                s = rng.randrange(pp)
                members = [g for g in range(n) if cur[g] % pp == s]
                if len(members) < 2:
                    continue
                x, y = rng.sample(members, 2)
            else:
                r = rng.randrange(dp)
                members = [g for g in range(n) if cur[g] // pp == r]
                if len(members) < 2:
                    continue
                x, y = rng.sample(members, 2)
            cur[x], cur[y] = cur[y], cur[x]
            c2 = cost(cur)
            accept = c2 <= cur_c
            if not accept:
                T = T0 * max(0.0, 1.0 - (it % q) / float(q))
                if T > 0 and rng.random() < math.exp(-(c2 - cur_c) / T):
                    accept = True
            if accept:
                cur_c = c2
            else:
                cur[x], cur[y] = cur[y], cur[x]
            if cur_c < best_c:
                best_c = cur_c
                best_rm = list(cur)
            if it % q == 0:
                cur = list(best_rm)
                cur_c = best_c

    scored.sort(key=lambda t: t[0])
    if not scored:
        scored = [(best_c, list(best_rm))]
    n_starts = min(3, len(scored))
    sa_total = max(1, budget - reserve - calls)
    per_run = max(1, sa_total // n_starts)
    for i in range(n_starts):
        sc, sr = scored[i]
        sa_run(sr, sc, per_run)

    # ---- final greedy hill-climb (structured moves, accept improvements only)
    reserve2 = max(pp * (dp - 1), 4)
    cur = list(best_rm)
    cur_c = best_c
    while calls < budget - reserve2:
        if rng.random() < 0.5 or n < 4:
            x, y = rng.sample(range(n), 2)
        elif rng.random() < 0.5:
            s = rng.randrange(pp)
            members = [g for g in range(n) if cur[g] % pp == s]
            if len(members) < 2:
                continue
            x, y = rng.sample(members, 2)
        else:
            r = rng.randrange(dp)
            members = [g for g in range(n) if cur[g] // pp == r]
            if len(members) < 2:
                continue
            x, y = rng.sample(members, 2)
        cur[x], cur[y] = cur[y], cur[x]
        c2 = cost(cur)
        if c2 <= cur_c:
            cur_c = c2
            if c2 < best_c:
                best_c = c2
                best_rm = list(cur)
        else:
            cur[x], cur[y] = cur[y], cur[x]

    # ---- true-cost recenter: pick the replica-0 aggregator per stage by the
    # real cost_fn (the proxy only approximates this choice). -----------------
    cur = list(best_rm)
    cur_c = best_c
    for s in range(pp):
        members = [g for g in range(n) if cur[g] % pp == s]
        if len(members) < 2:
            continue
        center = None
        for g in members:
            if cur[g] == s:
                center = g
                break
        if center is None:
            continue
        best_g = center
        best_val = cur_c
        for g in members:
            if g == center:
                continue
            if calls >= budget:
                break
            cur[g], cur[center] = cur[center], cur[g]
            c2 = cost(cur)
            if c2 < best_val:
                best_val = c2
                best_g = g
            cur[g], cur[center] = cur[center], cur[g]
        if best_g != center:
            cur[best_g], cur[center] = cur[center], cur[best_g]
            cur_c = best_val
            if cur_c < best_c:
                best_c = cur_c
                best_rm = list(cur)

    return best_rm
# EVOLVE-BLOCK-END
