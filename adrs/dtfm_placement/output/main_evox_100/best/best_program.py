"""GPU placement for DT-FM: assign N GPUs to (pipeline stage, data-parallel replica).

A placement is a rank_map: rank_map[gpu] = rank, with stage = rank % pp and replica = rank // pp.
GPUs that share a stage form one data-parallel group (gradient sync every iteration); GPUs with the
same replica index form one pipeline (activations flow stage 0 -> 1 -> ... -> pp-1).

Seed algorithm: a small genetic algorithm (order crossover + swap local search) in the spirit of
DT-FM's hybrid GA, using the calibrated step-time simulator as the objective.
"""
import math
import random


# EVOLVE-BLOCK-START
def schedule(delay_ms, bandwidth_gbps, pp, dp, cost_fn, budget):
    """Structure-aware + cost_fn-based GPU placement.

    Approach:
    1. Tiny N (<=4): exhaustively enumerate all permutations.
    2. Estimate effective link bandwidth with TCP-window model
       min(bw, 40/delay_ms) Gbps.
    3. Build surrogate costs for pipeline (8 MB activation x 128 micro-batches)
       and data-parallel gradient sync.
    4. Run simulated annealing with multiple surrogate weight settings and
       structure-aware moves (same-stage swaps for DP grouping, same-replica
       swaps for pipeline ordering, random swaps for exploration) to generate
       diverse candidates (no cost_fn calls).
    5. Evaluate top candidates with cost_fn.
    6. Refine with a cost_fn-based genetic algorithm (order crossover +
       shallow random-swap local search) using most of the remaining budget.
    7. Final targeted hill-climb (swaps within stages, between adjacent
       stages, and within the same replica across non-adjacent stages) using
       any leftover budget.
    """
    import itertools
    n = pp * dp
    rng = random.Random(42)
    calls = 0

    # Memoize cost_fn: the GA / hill-climb frequently revisit identical
    # rank_maps (e.g. after a rejected swap), so caching makes each
    # duplicate evaluation free and effectively multiplies the budget.
    _cache = {}
    def cost(rm):
        nonlocal calls
        key = tuple(rm)
        if key in _cache:
            return _cache[key]
        calls += 1
        c = cost_fn(rm)
        _cache[key] = c
        return c

    # --- Exhaustive search for tiny N ---
    if n <= 4:
        best_rm = None
        best_cost = float('inf')
        for perm in itertools.permutations(range(n)):
            if calls >= budget:
                break
            c = cost(list(perm))
            if c < best_cost:
                best_cost = c
                best_rm = list(perm)
        return best_rm

    # --- Effective bandwidth (Gbps) ---
    ebw = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                ebw[i][j] = 1e9
            else:
                d = delay_ms[i][j]
                b = bandwidth_gbps[i][j]
                ebw[i][j] = min(b, 40.0 / d) if d > 0 else b

    # --- Base link costs (bidirectional) ---
    # Pipeline activations flow forward (s -> s+1) while gradients flow
    # backward, and DP gradient sync both sends to and receives from the
    # coordinator.  Taking the max of the two directions captures the
    # effective cost of a link on an asymmetric topology.
    base_pipe = [[0.0] * n for _ in range(n)]
    base_dp = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                base_pipe[i][j] = max(0.064 / ebw[i][j], 0.064 / ebw[j][i])
                base_dp[i][j] = max(1.0 / ebw[i][j], 1.0 / ebw[j][i])

    def make_surrogate(dp_w, pipe_w):
        def surrogate_cost(rank_map):
            inv = [0] * n
            for gpu, rank in enumerate(rank_map):
                inv[rank] = gpu
            total = 0.0
            # DP gradient sync: bottleneck link to coordinator
            for s in range(pp):
                coord = inv[s]
                dp_max = 0.0
                for r in range(1, dp):
                    gpu = inv[s + pp * r]
                    if base_dp[gpu][coord] > dp_max:
                        dp_max = base_dp[gpu][coord]
                total += dp_w * dp_max
            # Pipeline activation transfers: slowest link bounds each micro-batch
            for r in range(dp):
                pipe_max = 0.0
                for s in range(pp - 1):
                    g1 = inv[s + pp * r]
                    g2 = inv[(s + 1) + pp * r]
                    if base_pipe[g1][g2] > pipe_max:
                        pipe_max = base_pipe[g1][g2]
                total += pipe_w * 128.0 * pipe_max
            return total
        return surrogate_cost

    def make_surrogate_max(dp_w, pipe_w):
        """Max-based surrogate: the real simulator's pipeline cost is bounded
        by the slowest link (a max over stages / replicas), not a sum.  A
        max-over-stages objective often guides the search better for
        pipeline-heavy topologies because a single slow edge dominates the
        whole iteration rather than being diluted by averaging."""
        def surrogate_cost(rank_map):
            inv = [0] * n
            for gpu, rank in enumerate(rank_map):
                inv[rank] = gpu
            dp_max = 0.0
            for s in range(pp):
                coord = inv[s]
                stage_max = 0.0
                for r in range(1, dp):
                    gpu = inv[s + pp * r]
                    if base_dp[gpu][coord] > stage_max:
                        stage_max = base_dp[gpu][coord]
                if stage_max > dp_max:
                    dp_max = stage_max
            pipe_max = 0.0
            for r in range(dp):
                replica_max = 0.0
                for s in range(pp - 1):
                    g1 = inv[s + pp * r]
                    g2 = inv[(s + 1) + pp * r]
                    if base_pipe[g1][g2] > replica_max:
                        replica_max = base_pipe[g1][g2]
                if replica_max > pipe_max:
                    pipe_max = replica_max
            return dp_w * dp_max + pipe_w * 128.0 * pipe_max
        return surrogate_cost

    def constructive_init():
        remaining = set(range(n))
        sr = [[None] * dp for _ in range(pp)]
        # Stage 0: central seed and close GPUs
        centralities = {}
        for i in remaining:
            centralities[i] = sum(base_dp[i][j] for j in remaining if j != i)
        sorted_gpus = sorted(remaining, key=lambda i: centralities[i])
        seed = rng.choice(sorted_gpus[:min(3, len(sorted_gpus))])
        sr[0][0] = seed
        remaining.remove(seed)
        for r in range(1, dp):
            costs = {}
            for i in remaining:
                costs[i] = sum(base_dp[i][sr[0][k]] for k in range(r))
            sorted_gpus = sorted(remaining, key=lambda i: costs[i])
            best = rng.choice(sorted_gpus[:min(3, len(sorted_gpus))])
            sr[0][r] = best
            remaining.remove(best)
        # Subsequent stages: chain from previous stage
        for s in range(1, pp):
            for r in range(dp):
                prev_gpu = sr[s - 1][r]
                costs = {}
                for gpu in remaining:
                    c = base_pipe[prev_gpu][gpu]
                    for k in range(r):
                        c += base_dp[gpu][sr[s][k]]
                    costs[gpu] = c
                sorted_gpus = sorted(remaining, key=lambda i: costs[i])
                best = rng.choice(sorted_gpus[:min(3, len(sorted_gpus))])
                sr[s][r] = best
                remaining.remove(best)
        rank_map = [0] * n
        for s in range(pp):
            for r in range(dp):
                rank_map[sr[s][r]] = s + pp * r
        return rank_map

    def run_sa(surrogate_cost, restarts, iters):
        candidates = []
        for _ in range(restarts):
            current = constructive_init()
            inv = [0] * n
            for gpu, rank in enumerate(current):
                inv[rank] = gpu
            cur_s = surrogate_cost(current)
            best_s = cur_s
            best_c = list(current)
            for it in range(iters):
                T = max(0.005, 1.0 - it / iters)
                # Structure-aware move selection:
                #  - same-stage swap (changes DP grouping within a stage)
                #  - same-replica swap (changes pipeline ordering)
                #  - random swap (general exploration)
                m = rng.random()
                if m < 0.35 and dp >= 2:
                    s = rng.randrange(pp)
                    r1, r2 = rng.sample(range(dp), 2)
                    i = inv[s + pp * r1]
                    j = inv[s + pp * r2]
                elif m < 0.7 and pp >= 2:
                    r = rng.randrange(dp)
                    s1, s2 = rng.sample(range(pp), 2)
                    i = inv[s1 + pp * r]
                    j = inv[s2 + pp * r]
                else:
                    i, j = rng.sample(range(n), 2)
                ri, rj = current[i], current[j]
                current[i], current[j] = rj, ri
                inv[ri], inv[rj] = j, i
                new_s = surrogate_cost(current)
                diff = cur_s - new_s
                if diff > 0 or rng.random() < math.exp(diff / T):
                    cur_s = new_s
                    if new_s < best_s:
                        best_s = new_s
                        best_c = list(current)
                else:
                    current[i], current[j] = ri, rj
                    inv[ri], inv[rj] = i, j
            candidates.append((best_s, best_c))
        candidates.sort(key=lambda x: x[0])
        return candidates

    # --- Multiple weight settings for diversity ---
    # A spread of (dp_w, pipe_w) trade-offs: the first five are pipe-dominant
    # (the real simulator heavily weights pipeline bottlenecks), while the
    # last two explore DP-dominant regimes so the cost_fn evaluation gets a
    # structurally diverse candidate pool.
    weight_sets = [(1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.0, 2.0),
                   (5.0, 1.0), (1.0, 0.25), (10.0, 1.0), (0.25, 1.0)]
    candidate_groups = []
    for dp_w, pipe_w in weight_sets:
        sc = make_surrogate(dp_w, pipe_w)
        cands = run_sa(sc, restarts=5, iters=15000)
        candidate_groups.append([(s, rm) for s, rm in cands[:2]])
    # Max-based surrogates: the real simulator's pipeline cost is bounded by
    # the slowest link, so a max-over-stages objective often guides the
    # search better for pipeline-heavy topologies.
    for dp_w, pipe_w in [(1.0, 1.0), (5.0, 1.0), (1.0, 2.0), (10.0, 1.0)]:
        sc = make_surrogate_max(dp_w, pipe_w)
        cands = run_sa(sc, restarts=5, iters=15000)
        candidate_groups.append([(s, rm) for s, rm in cands[:2]])

    # Deduplicate and evaluate the best few candidates from every surrogate
    # group, so the cost_fn sees structurally diverse candidates rather than
    # only the ones a single surrogate happens to rank highest.  Sorting the
    # groups together would be misleading because sum- and max-based scores
    # live on different scales.
    evaluated = []
    seen = set()
    for group in candidate_groups:
        group.sort(key=lambda x: x[0])
        for _, rm in group[:2]:
            if calls >= budget or len(evaluated) >= 30:
                break
            key = tuple(rm)
            if key in seen:
                continue
            seen.add(key)
            c = cost(rm)
            evaluated.append((c, list(rm)))
        if calls >= budget or len(evaluated) >= 30:
            break

    if not evaluated:
        rm = list(range(n))
        rng.shuffle(rm)
        evaluated.append((cost(rm), rm))

    evaluated.sort(key=lambda x: x[0])
    best_cost = evaluated[0][0]
    best_rm = list(evaluated[0][1])

    # --- cost_fn-based genetic algorithm ---
    def order_crossover(a, b):
        i, j = sorted(rng.sample(range(n), 2))
        child = [None] * n
        child[i:j] = a[i:j]
        rest = [x for x in b if x not in child[i:j]]
        for k in list(range(0, i)) + list(range(j, n)):
            child[k] = rest.pop(0)
        return child

    def structure_crossover(a, b):
        # Position-based crossover on the inverse permutation (rank -> gpu).
        # rank = s + pp * r, so a contiguous rank range is a whole replica
        # (pipeline) and the rank set {s, s+pp, s+2pp, ...} is a whole stage
        # (data-parallel group).  Copying whole units from each parent keeps
        # the DP grouping / pipeline chaining that the simulator cares about.
        a_inv = [0] * n
        b_inv = [0] * n
        for gpu, rank in enumerate(a):
            a_inv[rank] = gpu
        for gpu, rank in enumerate(b):
            b_inv[rank] = gpu
        if dp > 1 and (pp == 1 or rng.random() < 0.5):
            units = [[s + pp * r for s in range(pp)] for r in range(dp)]
        else:
            units = [[s + pp * r for r in range(dp)] for s in range(pp)]
        take = [rng.random() < 0.5 for _ in units]
        if not any(take):
            take[rng.randrange(len(units))] = True
        if all(take):
            take[rng.randrange(len(units))] = False
        child_inv = [None] * n
        for u, unit in enumerate(units):
            src = a_inv if take[u] else b_inv
            for rank in unit:
                child_inv[rank] = src[rank]
        # Repair duplicates: GPUs that appear twice get replaced by the GPUs
        # that were dropped by the unit-level selection.
        cnt = {}
        for g in child_inv:
            cnt[g] = cnt.get(g, 0) + 1
        missing = [g for g in range(n) if cnt.get(g, 0) == 0]
        rng.shuffle(missing)
        for r in range(n):
            if cnt[child_inv[r]] > 1:
                cnt[child_inv[r]] -= 1
                child_inv[r] = missing.pop()
        child_rm = [0] * n
        for r, gpu in enumerate(child_inv):
            child_rm[gpu] = r
        return child_rm

    def local_search(rm, c, tries):
        # Structured local search: bias toward swaps that preserve / improve
        # the DP grouping (same-stage swaps) and pipeline chaining
        # (same-replica swaps), with occasional random swaps for exploration.
        inv = [0] * n
        for gpu, rank in enumerate(rm):
            inv[rank] = gpu
        for _ in range(tries):
            if calls >= budget:
                break
            m = rng.random()
            if m < 0.4 and dp >= 2:
                s = rng.randrange(pp)
                r1, r2 = rng.sample(range(dp), 2)
                rk1, rk2 = s + pp * r1, s + pp * r2
            elif m < 0.8 and pp >= 2:
                r = rng.randrange(dp)
                s1, s2 = rng.sample(range(pp), 2)
                rk1, rk2 = s1 + pp * r, s2 + pp * r
            else:
                rk1, rk2 = rng.sample(range(n), 2)
            g1, g2 = inv[rk1], inv[rk2]
            rm[g1], rm[g2] = rk2, rk1
            inv[rk1], inv[rk2] = g2, g1
            c2 = cost(rm)
            if c2 < c:
                c = c2
            else:
                rm[g1], rm[g2] = rk1, rk2
                inv[rk1], inv[rk2] = g1, g2
        return rm, c

    while len(evaluated) < 4 and calls < budget:
        rm = list(range(n))
        rng.shuffle(rm)
        evaluated.append((cost(rm), rm))
    evaluated.sort(key=lambda x: x[0])
    pop_size = len(evaluated)

    # Shallow local search to allow more generations
    ls_tries = max(2, min(8, n // 4))

    # Reserve 40% of remaining budget for final hill-climb
    ga_budget = calls + int((budget - calls) * 0.6)

    ga_iters = 0
    ga_max_iters = 10 * max(1, budget - calls) + 2000

    def _tournament(k=3):
        # k-way tournament selection: stronger selection pressure than
        # uniform sampling, so good structural traits propagate faster.
        bi = rng.randrange(pop_size)
        for _ in range(k - 1):
            idx = rng.randrange(pop_size)
            if evaluated[idx][0] < evaluated[bi][0]:
                bi = idx
        return evaluated[bi][1]

    def _mutate(rm, strength):
        # Structure-aware mutation: bias swaps toward DP grouping
        # (same-stage) and pipeline ordering (same-replica), with random
        # swaps for exploration.  Keeps children diverse without destroying
        # the layout that the simulator rewards.
        rm = list(rm)
        inv = [0] * n
        for gpu, rank in enumerate(rm):
            inv[rank] = gpu
        for _ in range(strength):
            m = rng.random()
            if m < 0.4 and dp >= 2:
                s = rng.randrange(pp)
                r1, r2 = rng.sample(range(dp), 2)
                rk1, rk2 = s + pp * r1, s + pp * r2
            elif m < 0.8 and pp >= 2:
                r = rng.randrange(dp)
                s1, s2 = rng.sample(range(pp), 2)
                rk1, rk2 = s1 + pp * r, s2 + pp * r
            else:
                rk1, rk2 = rng.sample(range(n), 2)
            g1, g2 = inv[rk1], inv[rk2]
            rm[g1], rm[g2] = rk2, rk1
            inv[rk1], inv[rk2] = g2, g1
        return rm

    while calls < ga_budget and ga_iters < ga_max_iters:
        ga_iters += 1
        a = _tournament()
        b = _tournament()
        # Mix the two crossover operators: order crossover explores the
        # generic permutation space, while structure_crossover keeps whole
        # replicas / stages intact so children retain the DP/pipeline layout.
        if rng.random() < 0.5:
            child = order_crossover(a, b)
        else:
            child = structure_crossover(a, b)
        # Occasional structure-aware mutation to escape local optima.
        if rng.random() < 0.3:
            child = _mutate(child, rng.randint(1, max(1, n // 8)))
        c = cost(child)
        child, c = local_search(child, c, tries=ls_tries)
        worst = max(range(pop_size), key=lambda k: evaluated[k][0])
        if c < evaluated[worst][0]:
            evaluated[worst] = (c, child)
            if c < best_cost:
                best_cost = c
                best_rm = list(child)

    # --- Final targeted hill-climb wrapped in iterated local search ---
    def hill_climb(current, current_cost):
        """Exhaustive structure-aware descent on one incumbent.

        Repeatedly sweeps four structured neighbourhoods until no move
        improves: (1) swaps within a stage (re-groups the data-parallel
        members), (2) swaps between adjacent stages (stage reassignment),
        (3) swaps within a replica across non-adjacent stages (pipeline
        re-ordering), and (4) cyclic rotations of a whole replica's stage
        order or a whole stage's replica order.  Returns the improved
        permutation and its cost; the caller tracks the global best.
        """
        inv = [0] * n
        for gpu, rank in enumerate(current):
            inv[rank] = gpu

        def get_gpu(s, r):
            return inv[s + pp * r]

        improved = True
        while calls < budget and improved:
            improved = False
            # Swaps within each stage (replica reassignment)
            for s in range(pp):
                for r1 in range(dp):
                    for r2 in range(r1 + 1, dp):
                        if calls >= budget:
                            break
                        g1 = get_gpu(s, r1)
                        g2 = get_gpu(s, r2)
                        rank1 = current[g1]
                        rank2 = current[g2]
                        current[g1], current[g2] = rank2, rank1
                        inv[rank1], inv[rank2] = g2, g1
                        c = cost(current)
                        if c < current_cost:
                            current_cost = c
                            improved = True
                        else:
                            current[g1], current[g2] = rank1, rank2
                            inv[rank1], inv[rank2] = g1, g2
            # Swaps between adjacent stages (stage reassignment)
            for s in range(pp - 1):
                for r1 in range(dp):
                    for r2 in range(dp):
                        if calls >= budget:
                            break
                        g1 = get_gpu(s, r1)
                        g2 = get_gpu(s + 1, r2)
                        rank1 = current[g1]
                        rank2 = current[g2]
                        current[g1], current[g2] = rank2, rank1
                        inv[rank1], inv[rank2] = g2, g1
                        c = cost(current)
                        if c < current_cost:
                            current_cost = c
                            improved = True
                        else:
                            current[g1], current[g2] = rank1, rank2
                            inv[rank1], inv[rank2] = g1, g2
            # Swaps within same replica (pipeline reassignment, non-adjacent stages)
            for r in range(dp):
                for s1 in range(pp):
                    for s2 in range(s1 + 2, pp):
                        if calls >= budget:
                            break
                        g1 = get_gpu(s1, r)
                        g2 = get_gpu(s2, r)
                        rank1 = current[g1]
                        rank2 = current[g2]
                        current[g1], current[g2] = rank2, rank1
                        inv[rank1], inv[rank2] = g2, g1
                        c = cost(current)
                        if c < current_cost:
                            current_cost = c
                            improved = True
                        else:
                            current[g1], current[g2] = rank1, rank2
                            inv[rank1], inv[rank2] = g1, g2
            # Cyclic rotations of a replica's stage ordering.  A single swap
            # cannot express a full reorder of a pipeline, but a rotation can
            # move every stage at once without disturbing the DP grouping.
            for r in range(dp):
                if pp < 3 or calls >= budget:
                    continue
                ranks = [s + pp * r for s in range(pp)]
                gpus = [inv[rk] for rk in ranks]
                for k in range(pp):
                    current[gpus[k]] = ranks[(k + 1) % pp]
                    inv[ranks[(k + 1) % pp]] = gpus[k]
                c = cost(current)
                if c < current_cost:
                    current_cost = c
                    improved = True
                else:
                    for k in range(pp):
                        current[gpus[k]] = ranks[k]
                        inv[ranks[k]] = gpus[k]
            # Cyclic rotations of a stage's replica assignment.  Moves every
            # data-parallel slot at once, which pairwise swaps cannot do.
            for s in range(pp):
                if dp < 3 or calls >= budget:
                    continue
                ranks = [s + pp * r for r in range(dp)]
                gpus = [inv[rk] for rk in ranks]
                for k in range(dp):
                    current[gpus[k]] = ranks[(k + 1) % dp]
                    inv[ranks[(k + 1) % dp]] = gpus[k]
                c = cost(current)
                if c < current_cost:
                    current_cost = c
                    improved = True
                else:
                    for k in range(dp):
                        current[gpus[k]] = ranks[k]
                        inv[ranks[k]] = gpus[k]
            # Sampled swaps between non-adjacent stages of *different*
            # replicas.  The adjacent-stage and same-replica neighbourhoods
            # miss these moves, and they are the only way to re-assign a
            # stage slot to a GPU that currently lives far away in the
            # pipeline ordering.  Sampled to keep the sweep cheap.
            if pp >= 3 and dp >= 2 and calls < budget:
                for _ in range(min(pp * dp, 16)):
                    if calls >= budget:
                        break
                    s1 = rng.randrange(pp)
                    s2 = rng.randrange(pp)
                    while s2 == s1:
                        s2 = rng.randrange(pp)
                    r1 = rng.randrange(dp)
                    r2 = rng.randrange(dp)
                    g1 = get_gpu(s1, r1)
                    g2 = get_gpu(s2, r2)
                    rank1 = current[g1]
                    rank2 = current[g2]
                    current[g1], current[g2] = rank2, rank1
                    inv[rank1], inv[rank2] = g2, g1
                    c = cost(current)
                    if c < current_cost:
                        current_cost = c
                        improved = True
                    else:
                        current[g1], current[g2] = rank1, rank2
                        inv[rank1], inv[rank2] = g1, g2
            # Randomised 2-opt: reverse a contiguous segment of a pipeline.
            # Each pipeline's cost is the max edge along it, so the classic
            # escape move for bottleneck-path objectives is a segment
            # reversal (which can remove a slow edge from the middle of the
            # path while keeping the endpoints fixed).  Sampled per sweep.
            if pp >= 3 and calls < budget:
                for _ in range(min(2 * pp, 12)):
                    if calls >= budget:
                        break
                    r = rng.randrange(dp)
                    s1 = rng.randrange(pp - 1)
                    s2 = rng.randrange(s1 + 1, pp)
                    ranks = [s + pp * r for s in range(s1, s2 + 1)]
                    gpus = [inv[rk] for rk in ranks]
                    for k, rk in enumerate(reversed(ranks)):
                        current[gpus[k]] = rk
                        inv[rk] = gpus[k]
                    c = cost(current)
                    if c < current_cost:
                        current_cost = c
                        improved = True
                    else:
                        for k, rk in enumerate(ranks):
                            current[gpus[k]] = rk
                            inv[rk] = gpus[k]
        return current, current_cost

    if calls < budget:
        current, current_cost = hill_climb(list(best_rm), best_cost)
        if current_cost < best_cost:
            best_cost = current_cost
            best_rm = list(current)
        # Iterated local search: perturb the incumbent, re-optimise, and keep
        # the result only if it beats the global best.  This spends any
        # leftover budget escaping local optima instead of idling (small and
        # medium topologies converge long before the call budget is hit).
        while calls < budget:
            cand = _mutate(best_rm, rng.randint(1, max(2, n // 6)))
            cand_cost = cost(cand)
            cand, cand_cost = hill_climb(cand, cand_cost)
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_rm = list(cand)

    return best_rm
# EVOLVE-BLOCK-END
