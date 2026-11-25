"""GPU placement for DT-FM: assign N GPUs to (pipeline stage, data-parallel replica).

A placement is a rank_map: rank_map[gpu] = rank, with stage = rank % pp and replica = rank // pp.
GPUs that share a stage form one data-parallel group (gradient sync every iteration); GPUs with the
same replica index form one pipeline (activations flow stage 0 -> 1 -> ... -> pp-1).

Seed algorithm: a small genetic algorithm (order crossover + swap local search) in the spirit of
DT-FM's hybrid GA, using the calibrated step-time simulator as the objective.
"""
import random


# EVOLVE-BLOCK-START
def schedule(delay_ms, bandwidth_gbps, pp, dp, cost_fn, budget):
    """Bottleneck-aware GPU placement: structured construction + iterated local search.

    The simulator's per-iteration cost is bottleneck-dominated:
      * a stage's data-parallel gradient sync (hundreds of MB) is a star exchange
        through the stage's replica-0 GPU, so the slowest link out of that
        coordinator bounds the stage;
      * pipeline activations (8 MB x 128 micro-batches) are bounded by the slowest
        link adjacent to a stage.
    Effective bandwidth is TCP-window capped: eff = min(bw, 40 / delay_ms) Gbps.

    Steps:
      1. Build a symmetric effective-bandwidth matrix E.
      2. Structured candidates: partition the GPUs into pp stage groups of size dp
         around high-centrality hubs (grow each group by strongest link to its hub),
         then assign replicas so consecutive pipeline stages are joined by strong
         pairs (replica 0 = the most central GPU of each group).  Hub orders are
         drawn from centrality, nearest-neighbour chains and random shuffles.
         When dp == 1 a dedicated high-bottleneck pipeline path is also generated.
      3. Refine the incumbent with iterated local search: swaps are drawn from a
         mix of bottleneck-guided moves (touch the current worst E-matrix link),
         targeted-repair moves (swap a worst-link endpoint with a GPU that links
         strongly to the other endpoint), structure-preserving moves (swap two
         GPUs that share a pipeline or a stage) and random moves, with plateau
         acceptance and perturbation restarts.
      4. Final polish exhaustively tries swaps and relocations of the worst-link
         endpoints, keeping only strict improvements.
      5. Return the best placement seen.

    delay_ms[i][j], bandwidth_gbps[i][j]: link from GPU i to GPU j.
    cost_fn(rank_map) -> predicted seconds per training iteration. Each call counts toward `budget`.
    """
    n = pp * dp
    rng = random.Random(0x1F2E3D4C)
    calls = [0]

    def cost(rm):
        calls[0] += 1
        return cost_fn(rm)

    # ---- symmetric TCP-window-capped effective bandwidth (higher = better) ----
    E = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ei = E[i]
        di = delay_ms[i]
        bi = bandwidth_gbps[i]
        for j in range(n):
            if i == j:
                continue
            bw = bi[j]
            b2 = bandwidth_gbps[j][i]
            if b2 < bw:
                bw = b2
            d = di[j]
            d2 = delay_ms[j][i]
            if d2 > d:
                d = d2
            if d < 1e-6:
                d = 1e-6
            eff = 40.0 / d
            if bw < eff:
                eff = bw
            Ei[j] = eff

    centrality = sorted(range(n),
                        key=lambda g: -sum(E[g][x] for x in range(n) if x != g))

    # ---- link-time matrix (lower = better) --------------------------------
    T = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ti = T[i]
        Ei = E[i]
        for j in range(n):
            if i == j:
                continue
            e = Ei[j]
            if e < 1e-9:
                e = 1e-9
            Ti[j] = 1.0 / e

    # ---- open-TSP stage ordering (Held-Karp exact for small pp) -----------
    # Once GPUs are partitioned into dp pipelines (replicas), the stage
    # ordering inside each pipeline is exactly an open Traveling Salesman
    # Problem: nodes are the pipeline's GPUs, edge weight is the
    # TCP-window-capped link time T[a][b], and we want the minimum-cost
    # Hamiltonian path.  The simulator is bottleneck-dominated (the slowest
    # adjacent link bounds every micro-batch), so we also solve the min-max
    # variant (minimize the maximum edge on the path).  Held-Karp DP is exact
    # in O(2^k * k^2); for larger pipelines we fall back to greedy
    # nearest-neighbour + 2-opt / Or-opt.
    def _path_value(path, minmax):
        if len(path) < 2:
            return 0.0
        if minmax:
            v = 0.0
            for i in range(len(path) - 1):
                t = T[path[i]][path[i + 1]]
                if t > v:
                    v = t
            return v
        return sum(T[path[i]][path[i + 1]] for i in range(len(path) - 1))

    def _two_opt(path, minmax):
        k = len(path)
        if k < 3:
            return path
        for _ in range(6):
            improved = False
            cur_val = _path_value(path, minmax)
            for i in range(k - 1):
                for j in range(i + 1, k):
                    new = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
                    nv = _path_value(new, minmax)
                    if nv < cur_val - 1e-12:
                        path = new
                        cur_val = nv
                        improved = True
            if minmax:
                # Or-opt: move a single node elsewhere (2-opt cannot change the
                # max edge in a symmetric path, so we need node relocations).
                for i in range(k):
                    for j in range(k):
                        if i == j:
                            continue
                        new = path[:]
                        v = new.pop(i)
                        new.insert(j, v)
                        nv = _path_value(new, minmax)
                        if nv < cur_val - 1e-12:
                            path = new
                            cur_val = nv
                            improved = True
            if not improved:
                break
        return path

    def tsp_path(nodes, minmax=True):
        """Exact/approximate open TSP over `nodes` (Hamiltonian path).
        minmax=True minimises the maximum edge (bottleneck); minmax=False
        minimises the total path length."""
        k = len(nodes)
        if k <= 2:
            return list(nodes)
        INF = float('inf')
        if k <= 10:
            # Held-Karp DP: dp[mask][j] = best cost of a path visiting exactly
            # `mask` and ending at node j (best = min sum or min max).
            size = 1 << k
            dp = [[INF] * k for _ in range(size)]
            par = [[-1] * k for _ in range(size)]
            for i in range(k):
                dp[1 << i][i] = 0.0
            for mask in range(1, size):
                dpm = dp[mask]
                parm = par[mask]
                for j in range(k):
                    dj = dpm[j]
                    if dj == INF:
                        continue
                    tj = T[nodes[j]]
                    for nxt in range(k):
                        if mask & (1 << nxt):
                            continue
                        w = tj[nodes[nxt]]
                        if minmax:
                            cand = dj if dj > w else w
                        else:
                            cand = dj + w
                        nm = mask | (1 << nxt)
                        if cand < dp[nm][nxt] - 1e-12:
                            dp[nm][nxt] = cand
                            par[nm][nxt] = j
            full = size - 1
            best_end = 0
            best_val = INF
            for j in range(k):
                if dp[full][j] < best_val:
                    best_val = dp[full][j]
                    best_end = j
            path = []
            mask = full
            j = best_end
            while j >= 0:
                path.append(nodes[j])
                pj = par[mask][j]
                mask ^= (1 << j)
                j = pj
            path.reverse()
            return path
        # Larger pipelines: greedy nearest-neighbour from every start + 2-opt.
        best_path = None
        best_val = None
        for start in nodes:
            unvisited = set(nodes)
            unvisited.discard(start)
            path = [start]
            while unvisited:
                cur = path[-1]
                tc = T[cur]
                nxt = min(unvisited, key=lambda g: tc[g])
                path.append(nxt)
                unvisited.discard(nxt)
            path = _two_opt(path, minmax)
            val = _path_value(path, minmax)
            if best_val is None or val < best_val - 1e-12:
                best_val = val
                best_path = path
        return best_path

    def construct_tsp_pipeline(center_order, minmax=True):
        """Partition GPUs into dp pipelines (grow each pipeline around a
        high-centrality center by strongest link), then order the stages of
        each pipeline by solving the open TSP on the link-time matrix.
        This directly minimises the pipeline backbone's bottleneck link."""
        remaining = set(range(n))
        groups = []
        for r in range(dp):
            center = None
            for c in center_order:
                if c in remaining:
                    center = c
                    break
            if center is None:
                center = max(remaining,
                             key=lambda g: min((E[g][x] for x in remaining if x != g),
                                               default=0.0))
            others = sorted(remaining - {center}, key=lambda g: -E[center][g])
            group = [center] + others[:max(0, pp - 1)]
            groups.append(group)
            for g in group:
                remaining.discard(g)
        rank_map = [0] * n
        for r, group in enumerate(groups):
            order = tsp_path(group, minmax=minmax)
            for s, g in enumerate(order):
                rank_map[g] = r * pp + s
        return rank_map

    # ---- structural feature extractor for the surrogate model ------------
    # The simulator's per-iteration time is bottleneck-dominated, so we
    # describe a placement by its link-time bottlenecks (1 / effective bw):
    #   * the data-parallel gradient sync of stage s is a star exchange
    #     through its replica-0 GPU, so the worst outgoing link time bounds it;
    #   * pipeline activations are bounded by the slowest adjacent link, and
    #     the per-replica total link time bounds the pipeline.
    # Features: [1, worst DP link time, worst pipeline path time,
    #            total DP link time, total pipeline link time].
    def features(rm):
        gpu_at = [0] * n
        for g in range(n):
            gpu_at[rm[g]] = g
        dp_max = 0.0
        dp_sum = 0.0
        for s in range(pp):
            coord = gpu_at[s]
            worst = 0.0
            for r in range(1, dp):
                g = gpu_at[r * pp + s]
                e = E[coord][g]
                if e < 1e-9:
                    e = 1e-9
                t = 1.0 / e
                if t > worst:
                    worst = t
            if worst > dp_max:
                dp_max = worst
            dp_sum += worst
        pipe_max = 0.0
        pipe_sum = 0.0
        pipe_worst = 0.0
        for r in range(dp):
            base = r * pp
            tot = 0.0
            for s in range(pp - 1):
                a = gpu_at[base + s]
                b = gpu_at[base + s + 1]
                e = E[a][b]
                if e < 1e-9:
                    e = 1e-9
                t = 1.0 / e
                if t > pipe_worst:
                    pipe_worst = t
                tot += t
            if tot > pipe_max:
                pipe_max = tot
            pipe_sum += tot
        overall = dp_max if dp_max > pipe_worst else pipe_worst
        # Physics-informed feature vector (each entry is a link-time bottleneck,
        # i.e. 1 / effective bandwidth):
        #   [1, worst DP star link, worst per-replica pipeline total,
        #    total DP link time, total pipeline link time,
        #    worst single pipeline link, overall bottleneck]
        # The simulator states that the slowest link adjacent to a stage bounds
        # every micro-batch, so pipe_worst / overall carry the dominant signal.
        return [1.0, dp_max, pipe_max, dp_sum, pipe_sum, pipe_worst, overall]

    # ---- pure-Python least squares (normal equations + Gauss-Jordan) -----
    # Solves min_w ||A w - b||^2 without numpy: form AtA, Atb and eliminate.
    def lstsq(A, b):
        m = len(A)
        if m == 0:
            return None
        k = len(A[0])
        AtA = [[0.0] * k for _ in range(k)]
        Atb = [0.0] * k
        for i in range(m):
            Ai = A[i]
            bi = b[i]
            for p in range(k):
                api = Ai[p]
                Atb[p] += api * bi
                row = AtA[p]
                for q in range(k):
                    row[q] += api * Ai[q]
        for col in range(k):
            piv = col
            for r in range(col + 1, k):
                if abs(AtA[r][col]) > abs(AtA[piv][col]):
                    piv = r
            if piv != col:
                AtA[col], AtA[piv] = AtA[piv], AtA[col]
                Atb[col], Atb[piv] = Atb[piv], Atb[col]
            d = AtA[col][col]
            if abs(d) < 1e-12:
                continue
            for r in range(k):
                if r == col:
                    continue
                f = AtA[r][col] / d
                if f == 0.0:
                    continue
                row = AtA[r]
                prow = AtA[col]
                for c in range(col, k):
                    row[c] -= f * prow[c]
                Atb[r] -= f * Atb[col]
        w = [0.0] * k
        for i in range(k):
            d = AtA[i][i]
            if abs(d) > 1e-12:
                w[i] = Atb[i] / d
        return w

    # ---- spectral (Fiedler) ordering -------------------------------------
    # Build a normalized similarity matrix W from E, form the combinatorial
    # Laplacian L = D - W, and extract the Fiedler vector (eigenvector of the
    # second-smallest eigenvalue) by power iteration on the shifted Laplacian
    # (cI - L), deflating the constant eigenvector.  Sorting GPUs by their
    # Fiedler value gives a 1D embedding that places strongly connected GPUs
    # adjacent - exactly the structure a pipeline backbone needs.
    def fiedler_order():
        maxw = 0.0
        for i in range(n):
            Ei = E[i]
            for j in range(n):
                if Ei[j] > maxw:
                    maxw = Ei[j]
        if maxw <= 1e-12:
            return list(range(n))
        W = [[E[i][j] / maxw for j in range(n)] for i in range(n)]
        deg = [sum(W[i]) for i in range(n)]
        c = 2.0 * (max(deg) if deg else 1.0) + 1e-9
        v = [rng.random() - 0.5 for _ in range(n)]
        for _ in range(200):
            w = [0.0] * n
            for i in range(n):
                Wi = W[i]
                s = 0.0
                for j in range(n):
                    s += Wi[j] * v[j]
                w[i] = c * v[i] - (deg[i] * v[i] - s)
            mean = sum(w) / n
            w = [x - mean for x in w]
            norm = sum(x * x for x in w) ** 0.5
            if norm < 1e-12:
                break
            v = [x / norm for x in w]
        return sorted(range(n), key=lambda g: v[g])

    # ---- structure-aware construction --------------------------------------
    def construct(center_order):
        # Phase 1: partition GPUs into pp stage groups of size dp (bottleneck-aware).
        remaining = set(range(n))
        groups = []
        for s in range(pp):
            center = None
            for c in center_order:
                if c in remaining:
                    center = c
                    break
            if center is None:
                center = max(remaining,
                             key=lambda g: min((E[g][x] for x in remaining if x != g),
                                               default=0.0))
            others = sorted(remaining - {center}, key=lambda g: -E[center][g])
            group = [center] + others[:max(0, dp - 1)]
            groups.append(group)
            for g in group:
                remaining.discard(g)
        # Phase 2: assign replicas so pipeline links are strong; replica 0 = central.
        rem = [set(g) for g in groups]
        rank_map = [0] * n
        for r in range(dp):
            prev = None
            for s in range(pp):
                cand = rem[s]
                if not cand:
                    continue
                if prev is None:
                    best = max(cand,
                               key=lambda g: min((E[g][x] for x in cand if x != g),
                                                 default=0.0))
                else:
                    best = max(cand, key=lambda g: E[prev][g])
                rank_map[best] = r * pp + s
                rem[s].discard(best)
                prev = best
        return rank_map

    # ---- transpose construction: build pipelines first, then stages ---------
    def construct_transpose(center_order):
        # Phase 1: partition GPUs into dp pipeline groups of size pp.
        remaining = set(range(n))
        groups = []
        for r in range(dp):
            center = None
            for c in center_order:
                if c in remaining:
                    center = c
                    break
            if center is None:
                center = max(remaining,
                             key=lambda g: min((E[g][x] for x in remaining if x != g),
                                               default=0.0))
            others = sorted(remaining - {center}, key=lambda g: -E[center][g])
            group = [center] + others[:max(0, pp - 1)]
            groups.append(group)
            for g in group:
                remaining.discard(g)
        # Phase 2: within each pipeline, order stages so consecutive stages are
        # joined by strong links; stage 0 = the most central GPU of the pipeline.
        rem = [set(g) for g in groups]
        rank_map = [0] * n
        for s in range(pp):
            prev = None
            for r in range(dp):
                cand = rem[r]
                if not cand:
                    continue
                if prev is None:
                    best = max(cand,
                               key=lambda g: min((E[g][x] for x in cand if x != g),
                                                 default=0.0))
                else:
                    best = max(cand, key=lambda g: E[prev][g])
                rank_map[best] = r * pp + s
                rem[r].discard(best)
                prev = best
        return rank_map

    # ---- dedicated nearest-neighbour pipeline path (useful when dp == 1) ----
    def path_candidate(start):
        unvisited = set(range(n))
        path = [start]
        unvisited.discard(start)
        while unvisited:
            cur = path[-1]
            nxt = max(unvisited, key=lambda g: E[cur][g])
            path.append(nxt)
            unvisited.discard(nxt)
        rank_map = [0] * n
        for idx, g in enumerate(path):
            rank_map[g] = idx
        return rank_map

    best_c = None
    best_rm = None
    samples_X = []
    samples_y = []

    def consider(rm):
        nonlocal best_c, best_rm
        c = cost(rm)
        # Record (features, real cost) so the surrogate can be calibrated on a
        # deliberately diverse set of placements (spectral / hub / path / random).
        samples_X.append(features(rm))
        samples_y.append(c)
        if best_c is None or c < best_c:
            best_c = c
            best_rm = rm[:]
        return c

    # ---- structured candidates ---------------------------------------------
    # Centrality-based center orders (rotate through the most central GPUs),
    # generating both the stage-first and the pipeline-first (transpose) layout.
    n_struct = min(16, max(2, budget // 8))
    for k in range(n_struct):
        if calls[0] >= budget:
            break
        order = [centrality[(k + t) % n] for t in range(pp)]
        consider(construct(order))
        if calls[0] >= budget:
            break
        consider(construct_transpose(order))

    # Path-based center orders: greedy nearest-neighbour chains among hubs,
    # so the pipeline backbone itself is bottleneck-friendly.
    n_path = min(8, max(1, budget // 15))
    for k in range(n_path):
        if calls[0] >= budget:
            break
        start = centrality[k % n]
        unvisited = set(range(n))
        path = [start]
        unvisited.discard(start)
        while len(path) < pp and unvisited:
            cur = path[-1]
            nxt = max(unvisited, key=lambda g: E[cur][g])
            path.append(nxt)
            unvisited.discard(nxt)
        consider(construct(path))
        if calls[0] >= budget:
            break
        consider(construct_transpose(path))

    # Random-order centers for extra diversity.
    n_rand = min(6, max(1, budget // 25))
    for _ in range(n_rand):
        if calls[0] >= budget:
            break
        order = list(range(n))
        rng.shuffle(order)
        consider(construct(order))
        if calls[0] >= budget:
            break
        consider(construct_transpose(order))

    # ---- spectral (Fiedler) candidates -----------------------------------
    # The Fiedler vector provides a global 1D embedding of the topology that
    # local greedy growth can miss.  We use it three ways: (a) as a hub order
    # for both structured constructions, (b) as a direct "pipeline-major" split
    # (dp pipelines of pp spectrally-consecutive GPUs, stages in spectral
    # order), and (c) as a direct "stage-major" split (pp stages of dp
    # spectrally-consecutive GPUs) which keeps same-stage GPUs adjacent.
    spec = fiedler_order()

    def spectral_split_pipeline():
        rank_map = [0] * n
        for r in range(dp):
            for s in range(pp):
                idx = r * pp + s
                if idx < n:
                    rank_map[spec[idx]] = r * pp + s
        return rank_map

    def spectral_split_stage():
        rank_map = [0] * n
        for s in range(pp):
            for r in range(dp):
                idx = s * dp + r
                if idx < n:
                    rank_map[spec[idx]] = r * pp + s
        return rank_map

    if calls[0] < budget:
        consider(construct(spec))
    if calls[0] < budget:
        consider(construct_transpose(spec))
    if calls[0] < budget:
        consider(spectral_split_pipeline())
    if calls[0] < budget:
        consider(spectral_split_stage())
    if calls[0] < budget:
        consider(construct(list(reversed(spec))))
    if calls[0] < budget:
        consider(construct_transpose(list(reversed(spec))))

    # ---- TSP-ordered pipeline candidates ---------------------------------
    # Solve the open TSP for the stage ordering inside each pipeline.  The
    # min-max variant directly minimises the worst adjacent pipeline link,
    # which is the quantity that bounds every micro-batch in the simulator;
    # the min-sum variant additionally reduces the per-replica total path time.
    for mm in (True, False):
        if calls[0] >= budget:
            break
        consider(construct_tsp_pipeline(centrality, minmax=mm))
        if calls[0] >= budget:
            break
        consider(construct_tsp_pipeline(spec, minmax=mm))
        if calls[0] >= budget:
            break
        consider(construct_tsp_pipeline(list(reversed(spec)), minmax=mm))
    for k in range(min(4, max(1, budget // 20))):
        if calls[0] >= budget:
            break
        order = list(range(n))
        rng.shuffle(order)
        consider(construct_tsp_pipeline(order, minmax=True))
        if calls[0] >= budget:
            break
        consider(construct_tsp_pipeline(order, minmax=False))

    if dp == 1:
        for k in range(min(4, max(1, budget // 20))):
            if calls[0] >= budget:
                break
            consider(path_candidate(centrality[k % n]))

    if best_rm is None:
        rm = list(range(n))
        rng.shuffle(rm)
        consider(rm)

    # ---- QAP-based placement (breakthrough idea) --------------------------
    # Placement is exactly a quadratic assignment problem:
    #     minimize trace(A X B X^T)
    # where A[i][j] is the communication volume between rank i and rank j
    # (same-stage ranks exchange gradients; consecutive-stage ranks in the same
    # replica exchange activations) and B[i][j] = 1 / E[gpu_i][gpu_j] is the
    # TCP-window-capped link time.  Solving the QAP jointly optimizes the DP
    # grouping and the pipeline ordering, unlike the greedy / spectral
    # constructions above which optimize the two structures separately.
    #
    # scipy is not available, so we solve it in pure Python with a swap-based
    # local search using an O(n) delta update, started from several initial
    # permutations (identity, reversed, current best, random).  The QAP
    # objective is sum-based while the simulator is bottleneck-dominated, so we
    # try several flow weightings and always verify the results with the true
    # cost_fn (the best placement is retained regardless).
    Bmat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Bi = Bmat[i]
        Ei = E[i]
        for j in range(n):
            if i == j:
                continue
            e = Ei[j]
            if e < 1e-9:
                e = 1e-9
            Bi[j] = 1.0 / e

    def build_flow(w_dp, w_pipe):
        A = [[0.0] * n for _ in range(n)]
        for r in range(dp):
            for r2 in range(r + 1, dp):
                for s in range(pp):
                    i = r * pp + s
                    j = r2 * pp + s
                    A[i][j] = w_dp
                    A[j][i] = w_dp
        for r in range(dp):
            for s in range(pp - 1):
                i = r * pp + s
                j = r * pp + s + 1
                A[i][j] = w_pipe
                A[j][i] = w_pipe
        return A

    def qap_solve(A, inits, max_passes=4):
        # Swap-based local search on the QAP objective
        #   f(X) = sum_{i,j} A[i][j] * B[X(i)][X(j)]
        # where X(rank) = gpu.  Swapping the GPUs at ranks x and y changes only
        # the terms touching x or y, giving the O(n) delta
        #   delta = sum_{k != x,y} (A[x][k] + A[k][x] - A[y][k] - A[k][y])
        #                          * (B[b][X(k)] - B[a][X(k)])
        # with a = X(x), b = X(y).  Since A is symmetric this is
        #   2 * sum_k (A[x][k] - A[y][k]) * (B[b][X(k)] - B[a][X(k)]).
        best_rm = None
        best_obj = None
        for init in inits:
            gpu_at = [0] * n
            for g in range(n):
                gpu_at[init[g]] = g
            obj = 0.0
            for i in range(n):
                Ai = A[i]
                Bi = Bmat[gpu_at[i]]
                for j in range(n):
                    if Ai[j] != 0.0:
                        obj += Ai[j] * Bi[gpu_at[j]]
            improved = True
            passes = 0
            while improved and passes < max_passes:
                improved = False
                passes += 1
                for x in range(n):
                    Ax = A[x]
                    for y in range(x + 1, n):
                        Ay = A[y]
                        a = gpu_at[x]
                        b = gpu_at[y]
                        da = Bmat[a]
                        db = Bmat[b]
                        delta = 0.0
                        for k in range(n):
                            if k == x or k == y:
                                continue
                            axk = Ax[k]
                            ayk = Ay[k]
                            if axk == 0.0 and ayk == 0.0:
                                continue
                            delta += (axk - ayk) * (db[gpu_at[k]] - da[gpu_at[k]])
                        delta *= 2.0
                        if delta < -1e-12:
                            gpu_at[x], gpu_at[y] = b, a
                            obj += delta
                            improved = True
            rm = [0] * n
            for rank in range(n):
                rm[gpu_at[rank]] = rank
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_rm = rm
        return best_rm

    qap_inits = [list(range(n)), list(reversed(range(n)))]
    if best_rm is not None:
        qap_inits.append(best_rm[:])
    for _ in range(2):
        rm = list(range(n))
        rng.shuffle(rm)
        qap_inits.append(rm)

    qap_budget = min(6, max(2, budget // 16))
    qap_count = 0
    for w_dp, w_pipe in ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0),
                         (1.0, 4.0), (4.0, 1.0), (1.0, 8.0)):
        if qap_count >= qap_budget or calls[0] >= budget:
            break
        A = build_flow(w_dp, w_pipe)
        rm = qap_solve(A, qap_inits)
        if rm is not None:
            consider(rm)
            qap_count += 1

    # ---- calibrated white-box surrogate ----------------------------------
    # Fit cost(rm) ~ w . features(rm) on the diverse real evaluations gathered
    # above.  The simulator's cost is a bottleneck-dominated function of the
    # effective bandwidths, so this linear model in the bottleneck features is
    # highly predictive.  We then run thousands of *free* surrogate evaluations
    # to locate strong placements and only spend real budget verifying the
    # top-K surrogate candidates (the true best is always retained).
    # ---- physics-informed parametric model fit --------------------------
    # The simulator's per-iteration time is, to first order,
    #   cost = c0 + c1*max_dp + c2*max_pipe + c3*dp_sum + c4*pipe_sum
    #          + c5*pipe_worst + c6*max(max_dp, pipe_worst)
    # where every feature is a link-time (1 / effective bandwidth) bottleneck.
    # This model is linear in its coefficients, so a scipy.optimize.curve_fit
    # Gauss-Newton iteration reduces exactly to the normal equations solved by
    # lstsq.  We (a) clamp coefficients to be non-negative (all physics terms
    # are costs), (b) refit the intercept so the clamped model stays unbiased,
    # and (c) validate on a held-out split of the diverse probes via R^2 before
    # trusting the surrogate as a free oracle.
    def fit_weights(X, y):
        ww = lstsq(X, y)
        if ww is None:
            return None
        for i in range(len(ww)):
            if ww[i] < 0.0:
                ww[i] = 0.0
        preds = [sum(wi * fi for wi, fi in zip(ww, x)) for x in X]
        resid = [yi - pi for yi, pi in zip(y, preds)]
        ww[0] += sum(resid) / len(resid)
        return ww

    def r_squared(ww, X, y):
        preds = [sum(wi * fi for wi, fi in zip(ww, x)) for x in X]
        ymean = sum(y) / len(y)
        ss_tot = sum((yi - ymean) ** 2 for yi in y)
        ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, preds))
        if ss_tot < 1e-12:
            return 1.0
        return 1.0 - ss_res / ss_tot

    w = None
    if len(samples_X) >= 8:
        half = len(samples_X) // 2
        w_tr = fit_weights(samples_X[:half], samples_y[:half])
        r2_val = 1.0
        if w_tr is not None:
            r2_val = r_squared(w_tr, samples_X[half:], samples_y[half:])
        # Trust the fitted physics model only when it generalises to held-out
        # probes; otherwise fall back to the true-cost local search below.
        if r2_val > 0.5:
            w = fit_weights(samples_X, samples_y)

    if w is not None:
        def predict(rm):
            f = features(rm)
            return sum(wi * fi for wi, fi in zip(w, f))

        def surrogate_ils(start, iters):
            cur = start[:]
            cur_c = predict(cur)
            b = cur[:]
            b_c = cur_c
            stall = 0
            for _ in range(iters):
                x = rng.randrange(n)
                y = rng.randrange(n)
                if x == y:
                    continue
                cur[x], cur[y] = cur[y], cur[x]
                c2 = predict(cur)
                if c2 <= cur_c:
                    cur_c = c2
                    if c2 < b_c:
                        b_c = c2
                        b = cur[:]
                    stall = 0
                else:
                    cur[x], cur[y] = cur[y], cur[x]
                    stall += 1
                if stall > 40:
                    cur = b[:]
                    for _ in range(3):
                        a = rng.randrange(n)
                        c = rng.randrange(n)
                        cur[a], cur[c] = cur[c], cur[a]
                    cur_c = predict(cur)
                    stall = 0
            return b

        verify_n = max(1, budget // 8)
        starts = [best_rm]
        for _ in range(3):
            rm = list(range(n))
            rng.shuffle(rm)
            starts.append(rm)
        iters = max(400, min(20 * n, 1000))
        n_pool = max(4, min(verify_n * 2, 60))
        pool = []
        for i in range(n_pool):
            st = starts[i % len(starts)]
            cand = surrogate_ils(st, iters)
            pool.append((predict(cand), cand))
        # Verify only the top-K surrogate candidates with the true cost_fn.
        pool.sort(key=lambda t: t[0])
        for i in range(min(verify_n, len(pool))):
            if calls[0] >= budget:
                break
            consider(pool[i][1])

    # ---- proxy bottleneck (E-matrix) used to guide perturbations ------------
    def worst_link_gpus(rm):
        gpu_at = [0] * n
        for g in range(n):
            gpu_at[rm[g]] = g
        best_v = -1.0
        pair = None
        for s in range(pp):
            coord = gpu_at[s]
            for r in range(1, dp):
                g = gpu_at[r * pp + s]
                e = E[coord][g]
                if e < 1e-9:
                    e = 1e-9
                v = 1.0 / e
                if v > best_v:
                    best_v = v
                    pair = (coord, g)
        for r in range(dp):
            base = r * pp
            for s in range(pp - 1):
                a = gpu_at[base + s]
                b = gpu_at[base + s + 1]
                e = E[a][b]
                if e < 1e-9:
                    e = 1e-9
                v = 1.0 / e
                if v > best_v:
                    best_v = v
                    pair = (a, b)
        return pair

    # ---- structured move generator ------------------------------------------
    # A "structured" swap exchanges two GPUs that share a replica (same pipeline,
    # different stages -> reshuffles the DP grouping) or share a stage (same DP
    # group, different replicas -> reshuffles the pipeline ordering).  These
    # moves perturb the actual bottleneck links directly, unlike a purely random
    # swap, so they spend the cost_fn budget far more efficiently.
    def structured_pair(rm):
        gpu_at = [0] * n
        for g in range(n):
            gpu_at[rm[g]] = g
        choices = []
        if pp > 1:
            choices.append('pipeline')
        if dp > 1:
            choices.append('stage')
        if not choices:
            return None
        kind = choices[0] if rng.random() < 0.5 else choices[-1]
        if kind == 'pipeline':
            r = rng.randrange(dp)
            s1, s2 = rng.sample(range(pp), 2)
            return gpu_at[r * pp + s1], gpu_at[r * pp + s2]
        s = rng.randrange(pp)
        r1, r2 = rng.sample(range(dp), 2)
        return gpu_at[r1 * pp + s], gpu_at[r2 * pp + s]

    # ---- iterated local search (targeted swaps + plateau moves) ------------
    cur_rm = best_rm[:]
    cur_c = best_c
    stall_limit = max(8, n)
    kick = 2
    while calls[0] < budget:
        # hill climbing: mix bottleneck-guided swaps, structure-preserving
        # swaps and random swaps so the search focuses on the actual worst
        # link of the current placement while still exploring the layout.
        stall = 0
        while calls[0] < budget and stall < stall_limit:
            r = rng.random()
            if r < 0.25:
                # bottleneck-guided: move one endpoint of the worst E-matrix link
                pair = worst_link_gpus(cur_rm)
                if pair is not None:
                    x = pair[rng.randrange(2)]
                else:
                    x = rng.randrange(n)
                y = rng.randrange(n)
            elif r < 0.45:
                # targeted repair: swap one endpoint of the worst link with a
                # GPU that has a strong link to the other endpoint, so the
                # repaired link is likely to become fast.
                pair = worst_link_gpus(cur_rm)
                if pair is not None:
                    a, b = pair
                    if rng.random() < 0.5:
                        x, other = a, b
                    else:
                        x, other = b, a
                    cands = sorted((g for g in range(n) if g != x),
                                   key=lambda g: -E[other][g])[:3]
                    y = cands[rng.randrange(len(cands))]
                else:
                    x = rng.randrange(n)
                    y = rng.randrange(n)
            elif r < 0.70:
                # structure-preserving: swap two GPUs that share a pipeline or
                # a stage, which perturbs the DP grouping or the pipeline order
                # without destroying the overall layout.
                sp = structured_pair(cur_rm)
                if sp is not None:
                    x, y = sp
                else:
                    x = rng.randrange(n)
                    y = rng.randrange(n)
            else:
                x = rng.randrange(n)
                y = rng.randrange(n)
            if x == y:
                continue
            cur_rm[x], cur_rm[y] = cur_rm[y], cur_rm[x]
            c2 = cost(cur_rm)
            if c2 <= cur_c:
                cur_c = c2
                stall = 0
                if c2 < best_c:
                    best_c = c2
                    best_rm = cur_rm[:]
            else:
                cur_rm[x], cur_rm[y] = cur_rm[y], cur_rm[x]
                stall += 1
        if calls[0] >= budget:
            break
        # perturbation restart from the incumbent best
        cur_rm = best_rm[:]
        for _ in range(kick):
            rr = rng.random()
            if rr < 0.4:
                pair = worst_link_gpus(cur_rm)
                if pair is not None:
                    x = pair[rng.randrange(2)]
                else:
                    x = rng.randrange(n)
                y = rng.randrange(n)
            elif rr < 0.7:
                sp = structured_pair(cur_rm)
                if sp is not None:
                    x, y = sp
                else:
                    x = rng.randrange(n)
                    y = rng.randrange(n)
            else:
                x = rng.randrange(n)
                y = rng.randrange(n)
            cur_rm[x], cur_rm[y] = cur_rm[y], cur_rm[x]
        cur_c = cost(cur_rm)
        # adapt perturbation strength: grow when the incumbent stops improving
        kick = min(max(2, n // 2), kick + 1) if rng.random() < 0.5 else kick

    # ---- TSP repair of the worst pipeline ---------------------------------
    # Re-solve the open TSP for the stage ordering of the pipeline that
    # currently contains the worst adjacent link.  This is a strong, targeted
    # repair: the TSP minimises the pipeline backbone's bottleneck directly,
    # which single swaps often cannot fix (a whole pipeline may need re-ordering).
    def _pipeline_of_worst(rm):
        gpu_at = [0] * n
        for g in range(n):
            gpu_at[rm[g]] = g
        worst = -1.0
        wr = -1
        for r in range(dp):
            for s in range(pp - 1):
                t = T[gpu_at[r * pp + s]][gpu_at[r * pp + s + 1]]
                if t > worst:
                    worst = t
                    wr = r
        return wr

    if pp >= 3:
        for _ in range(min(3, max(1, budget // 30))):
            if calls[0] >= budget:
                break
            wr = _pipeline_of_worst(best_rm)
            if wr < 0:
                break
            gpu_at = [0] * n
            for g in range(n):
                gpu_at[best_rm[g]] = g
            group = [gpu_at[wr * pp + s] for s in range(pp)]
            improved_tsp = False
            for mm in (True, False):
                if calls[0] >= budget:
                    break
                order = tsp_path(group, minmax=mm)
                cand = best_rm[:]
                for s, g in enumerate(order):
                    cand[g] = wr * pp + s
                c2 = cost(cand)
                if c2 < best_c:
                    best_c = c2
                    best_rm = cand[:]
                    improved_tsp = True
            if not improved_tsp:
                break

    # ---- final polish: exhaustively try swaps AND relocations of the worst-link endpoints --
    # Swaps exchange two GPUs' ranks; relocations move one GPU to another rank and
    # shift the displaced GPUs, which can repair a bottleneck link that a single
    # swap cannot (e.g. when a whole pipeline needs re-ordering).
    while calls[0] < budget:
        pair = worst_link_gpus(best_rm)
        if pair is None:
            break
        improved = False
        for x in pair:
            for y in range(n):
                if x == y or calls[0] >= budget:
                    continue
                best_rm[x], best_rm[y] = best_rm[y], best_rm[x]
                c2 = cost(best_rm)
                if c2 < best_c:
                    best_c = c2
                    improved = True
                else:
                    best_rm[x], best_rm[y] = best_rm[y], best_rm[x]
        if not improved:
            # relocation: move endpoint x to every other position in the rank list
            for x in pair:
                for y in range(n):
                    if x == y or calls[0] >= budget:
                        continue
                    val = best_rm.pop(x)
                    best_rm.insert(y, val)
                    c2 = cost(best_rm)
                    if c2 < best_c:
                        best_c = c2
                        improved = True
                    else:
                        best_rm.pop(y)
                        best_rm.insert(x, val)
            if not improved:
                break

    return best_rm
# EVOLVE-BLOCK-END
