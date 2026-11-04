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
    """Return a rank_map (a permutation of 0..pp*dp-1) with low predicted iteration time.

    delay_ms[i][j], bandwidth_gbps[i][j]: link from GPU i to GPU j.
    cost_fn(rank_map) -> predicted seconds per training iteration. Each call counts toward `budget`;
    calling it more than `budget` times raises an error, so stop before that.
    """
    n = pp * dp
    rng = random.Random(0)
    calls = 0

    def cost(rm):
        nonlocal calls
        calls += 1
        return cost_fn(rm)

    def order_crossover(a, b):
        i, j = sorted(rng.sample(range(n), 2))
        child = [None] * n
        child[i:j] = a[i:j]
        rest = [x for x in b if x not in child[i:j]]
        for k in list(range(0, i)) + list(range(j, n)):
            child[k] = rest.pop(0)
        return child

    def local_search(rm, c, tries):
        for _ in range(tries):
            if calls >= budget:
                break
            x, y = rng.sample(range(n), 2)
            rm[x], rm[y] = rm[y], rm[x]
            c2 = cost(rm)
            if c2 < c:
                c = c2
            else:
                rm[x], rm[y] = rm[y], rm[x]
        return rm, c

    pop_size = max(4, min(20, budget // 20))
    population = []
    for _ in range(pop_size):
        rm = list(range(n))
        rng.shuffle(rm)
        population.append((cost(rm), rm))

    while calls < budget:
        (_, a), (_, b) = rng.sample(population, 2)
        child = order_crossover(a, b)
        c = cost(child)
        child, c = local_search(child, c, tries=n)
        worst = max(range(pop_size), key=lambda k: population[k][0])
        if c < population[worst][0]:
            population[worst] = (c, child)

    return min(population)[1]
# EVOLVE-BLOCK-END
