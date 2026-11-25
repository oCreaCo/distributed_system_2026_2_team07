# agent_cc: DT-FM placement, can a program beat ~1.27 on train?

Best program: `best_program.py` (= `v7.py`). Train combined_score **1.27727** (N8 1.3358, N16 1.1258,
N32 1.2557, N64 1.4095). This equals the best-of-three-programs per-topology bound (1.2773); it does
not exceed it.

## Evaluator runs (train only)

| # | file | combined | N64_s0 | N64_s1 | N64_s2 | idea |
|---|------|----------|--------|--------|--------|------|
| 1 | v5.py | 1.27607 | 62.30 | 65.38 | 64.35 | class grid + threshold (C,D) feasibility SA + cost_fn-fitted surrogate SA + banded constructions + cost_fn hill climb |
| 2 | cegas_r300_s0 (reference) | 1.27055 | 62.30 | 64.65 | 68.55 | existing program, rerun for per-topology numbers |
| 3 | main_adaevolve_100 (reference) | 1.26964 | 62.30 | 69.47 | 64.35 | existing program |
| 4 | main_evox_100 (reference) | 1.26735 | 68.21 | 64.65 | 64.35 | existing program |
| 5 | v6.py | 1.27607 | 62.30 | 65.38 | 64.35 | v5 + more surrogate-SA restarts (40) at 3 temperatures, more starts |
| 6 | v7.py | **1.27727** | 62.30 | 64.65 | 64.35 | v6 + vary the tie-break weight (sum of row/column maxima) across restarts |
| 7 | v8.py | 1.27727 | 62.30 | 64.65 | 64.35 | probe: v7 with ~2x surrogate search (80k steps x up to 90 runs, 25 s cap); no change |

N8/N16/N32 per-topology values are identical across v5-v8 and the three reference programs
(N8 = exhaustive optimum; N16 58.00/64.11/63.77; N32 56.24/60.11/60.65; evox gets 63.96 on N16_s2).

## Method (v7)

1. GPUs with identical link rows/columns form classes (regions); GPUs in a class are interchangeable,
   so a placement is a dp x pp grid of class labels (row = replica/pipeline, column = stage,
   row 0 = DP hub row). Swaps inside a class are no-ops and are never spent on cost_fn.
2. Link model from the task statement only: effective bw = min(bw, 10 MB / RTT), 8 MB activations,
   300 MB gradients. Pipeline bottleneck C = max link used along rows; DP bottleneck D = max over
   columns of link(member -> hub).
3. Threshold search: for increasing C, find the smallest D for which a feasible grid exists
   (SA on violation counts), under two hypotheses for end-stage gradient size; ~10 candidates.
4. Score candidates (+ row-reversed variants, + a few perturbations) with cost_fn and fit
   T ~ a*C + b*D + c by least squares; the end-stage gradient factor (0/0.25/0.5/0.75 extra) is
   picked by fit residual. Typical fit: a ~ 290, b ~ 1.8 (i.e. one 10 ms step of the pipeline
   bottleneck costs ~3 s; DP sync is roughly additive).
5. Banded constructions: classes in an order, laid out column-major (each stage ~ one class, every
   pipeline walks the same class order); all orders for k <= 7 classes, ranked by the surrogate.
6. Surrogate SA (cell swaps + hub-row swaps) from the best ~8 starts, 40 restarts with varied
   temperature and tie-break weight; every result scored with cost_fn.
7. Remaining budget: cost_fn hill climbing (class-level swaps, hub-row swaps, row segment reversal).
   Uses ~150-950 of the 2000 calls; 8-17 s per topology under the evaluator.

## What the N=64 objective looks like (offline, seeds >= 1000 only)

- With theta: T = 142*(max(f,C)+max(b,C)) + max(2.93, T_dp) + 0.166*min(2.93, T_dp), where
  C = slowest pipeline link time per micro-batch (both directions, all replicas) and
  T_dp = slowest 2*(lat + grad/bw) from a stage member to that stage's replica-0 GPU. For C above
  ~0.053 s this is ~284*C + T_dp + 0.5. Inter-region C is 0.07-0.33 s (20-90 s of iteration time);
  inter-region T_dp is 4-21 s. So iteration time is a two-bottleneck problem: the first priority is
  the cheapest inter-region pair the pipelines can be forced to cross, the second is how far the
  farthest stage member is from its hub.
- Stages 0 and pp-1 carry ~42% larger gradients (embedding), so the best layouts keep regions that
  are far from the hub out of the end columns (e.g. seed 1000: Ireland only in stages 1-6 -> 48.25
  vs 51.31 for the best of the three programs).
- Two layout families win: (a) rows mostly single-region with a few mixed rows that cross only at
  the cheapest allowed pairs, and DP crossing far regions; (b) when a far crossing is forced anyway
  (e.g. two regions with counts not divisible by 8), "banded" layouts where columns are single-region
  and all pipelines cross the same chain of regions, so DP stays local.
- Random-start swap SA (even with the exact cost) gets stuck at 60-100 s; from good starts it moves
  only a little. The landscape has plateaus (max of maxima) and deep local optima.

## Offline validation (N=64, seeds 1000-1029, simulator called directly in scratch scripts)

- v7 vs per-seed best of the three existing programs: geomean 1.0104 (v7 never worse, better on
  12/30 seeds by up to 7%: e.g. 1000 47.84 vs 51.31, 1004 50.55 vs 52.58, 1026 56.44 vs 59.21).
- Heavy check: exact-cost SA (2 temperatures x 250k steps) started from v7's result and from each of
  the three programs' results, 8 runs per seed. It improved v7 on only 1/30 seeds (1005: 51.63 ->
  49.21, a value v6 had itself found in an earlier run, i.e. run-to-run variance). On the other 29
  seeds no start, including the three programs' placements, beat v7.

## Judgment

On train, N8 is exhaustively optimal, and N16/N32/N64 values are identical (to 2 decimals) across
four structurally different searches (three evolved programs + this threshold/surrogate method),
and doubling this method's search (v8) changes nothing. On fresh seeds the same method does find
the headroom the old programs leave (~1% at N64 on average), and heavy exact-cost search rarely
improves on it. So the train split's ~1.2773 looks like the simulator optimum for these 12
topologies (a ceiling), not a plateau of the search; it is not proven optimal (no exact solver).
The new program does generalise better than any single old program (it matches best-of-three on
train and beats it on fresh seeds), which may show up on held-out.
