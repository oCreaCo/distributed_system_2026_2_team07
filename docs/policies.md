# Best program (policy) 현재 상태

최종 갱신: 2026-11-25. 대상은 adaevolve, EvoX, agent_cc의 최고 프로그램이다. CEGAS는 이 문서에서 다루지 않는다
(결과는 [overview](overview.md)와 worklog 참고). 재현 절차는 [reproduce.md](reproduce.md)에 있다.

## 1. 파일

| 이름 | 만든 방법 | 프로그램 | 과정 기록 |
|---|---|---|---|
| **agent_cc** (현재 최고) | Claude Code 서브에이전트 (claude-opus-5-5), 약 23분, 평가기 7회 | `adrs/dtfm_placement/output/agent_cc/best_program.py` (483줄, = `v7.py`) | `agent_cc/log.md`, 중간 버전 `v1.py`–`v8.py`, 프롬프트 `repro/agent_cc_prompt.md` |
| **EvoX** | SkyDiscover `evox`, deepseek-v4.1-flash, 100 it., 88분, $2.72 | `output/main_evox_100/best/best_program.py` (634줄, iteration 58) | `best/best_program_info.json`, 설정 `repro/config_evox.yaml` |
| **adaevolve** | SkyDiscover `adaevolve`, deepseek-v4.1-flash, 100 it., 72.5분, $2.68 | `output/main_adaevolve_100/best/best_program.py` (1,091줄, iteration 65) | `best/best_program_info.json`, 설정 `config.yaml` |
| seed (비교용) | DT-FM 스타일 작은 GA | `adrs/dtfm_placement/initial_program.py` | – |
| CEGAS 300 (agent_cc 시작점) | CEGAS, LLM 300 요청 | `output/cegas_r300_s0/best/best_program.py` (434줄) | 프로그램만 커밋. 점수는 [overview](overview.md#주요-수치) |

- 경로는 모두 `adrs/dtfm_placement/` 기준이다.
- 커밋된 것은 위 표의 프로그램과 기록뿐이다. checkpoint와 SkyDiscover 로그(합계 약 210 MB)는 로컬 `output/`에만 있다.
- 세 프로그램 모두 **시뮬레이터 v1**을 `cost_fn`으로 해서 만들어졌다 ([simulator.md](simulator.md#6-버전)).

## 2. 점수

`combined_score` = topology별 `sim(DT-FM GA 배치) / sim(프로그램 배치)`의 기하평균이다. `vs_ga_sim`은 같은 시뮬레이터와
같은 2,000회 예산으로 DT-FM GCMA를 돌린 기준선 대비 값이다. train은 N = 8, 16, 32, 64 × seed 0–2, held-out은 seed 100–102다.

| 프로그램 | v1 train / held-out | **v2 train / held-out** | v2 vs_ga_sim (train / held-out) |
|---|---|---|---|
| **agent_cc** | 1.277 / 1.322 | **1.269 / 1.306** | 1.030 / 1.047 |
| EvoX | 1.267 / 1.308 | 1.259 / 1.300 | 1.022 / 1.043 |
| adaevolve | 1.270 / 1.288 | 1.251 / 1.278 | 1.015 / 1.025 |
| seed | 1.048 / – | 1.068 / 1.054 | 0.867 / 0.845 |

- v2 값은 2026-11-25에 프로그램을 하나씩 단독 실행해 다시 잰 값이다. 11-15 값과 비교하면 EvoX와 adaevolve는 소수 넷째 자리까지 같다.
- agent_cc는 1.2686 → 1.2694, 1.3068 → 1.3057로 조금 달랐다. 이 프로그램은 실행 시간(wall-clock) 상한을 써서 결과가 CPU 부하에
  따라 조금씩 바뀐다.
- **평가기를 여러 개 동시에 돌리면 점수가 떨어진다** (EvoX 0.65, agent_cc 1.247까지). 공식 점수는 단독 실행으로 잰다.

N별 DT-FM 대비 (v2, train):

| | N8 | N16 | N32 | N64 |
|---|---|---|---|---|
| agent_cc | 1.310 | 1.105 | 1.288 | 1.393 |
| EvoX | 1.310 | 1.103 | 1.286 | 1.353 |
| adaevolve | 1.310 | 1.089 | 1.262 | 1.359 |

N8은 세 프로그램 모두 전수 탐색 최적값이라 같다. 차이는 N ≥ 16에서만 난다.

## 3. 일반화 (처음 보는 topology, 시뮬레이터)

N = 16, 32, 64, 128 × seed 5000–5004에서 ga_sim 대비로 쟀다. v1 기준이다 ([0445](worklog/2026-11-15_0445_generalization-overfit-check.md)).

| 프로그램 | N16 | N32 | N64 | N128 | N128 30 s 초과 |
|---|---|---|---|---|---|
| **agent_cc** | 1.033 | 1.064 | 1.051 | 1.084 | 0/5 (최대 19.3 s) |
| EvoX | 1.028 | 1.046 | **0.927** | 1.062 | 1/5 (30.0 s) |
| adaevolve | 1.033 | 1.032 | 1.022 | 1.060 | 0/5 (11.9 s) |

- EvoX는 train N64에서는 ga_sim보다 5.3% 좋았지만 새 N64에서는 7.3% 나쁘다. train 12개에 맞춰진 흔적이다.
- agent_cc만 v2에서도 다시 쟀다 (`output/generalization_v2.txt`). ga_sim 대비 1.014 / 1.049 / 1.039 / 1.096이다.

## 4. 정확한 최적값과의 차이

MILP로 구한 시뮬레이터 정확 최적값 대비다. 값은 `sim(프로그램) / sim(최적)`의 기하평균이고, 1.0이면 최적이다.
자세한 내용은 [MILP 문서](milp-exact-optimum.md)에 있다.

| | N16 | N32 | N64 | N128 |
|---|---|---|---|---|
| agent_cc, v1 | 1.000 | 1.000 | 1.005 | 1.010 |
| agent_cc, v2 | 1.002 | 1.011 | **1.026** | 1.001 |
| ga_sim, v2 | 1.016 | 1.060 | 1.066 | 1.097 |

- v1에서는 배치 탐색이 사실상 끝났다.
- v2에서는 N64에 약 2.6%의 여지가 다시 생겼다. agent_cc가 "첫·마지막 stage gradient가 크다"는 v1 구조를 가정하기 때문이다.

## 5. A100 실측 (N8)

| | train (3) | held-out (3) | 처음 보는 topology (4) | 전체 |
|---|---|---|---|---|
| agent_cc vs DT-FM | 1.182 | 1.164 | 1.089 | **1.139** (10개) |
| EvoX vs DT-FM | – | 1.158 | – | – |
| adaevolve vs DT-FM | – | 1.163 | – | – |

- EvoX와 adaevolve 값은 11-15 held-out 측정(0320 worklog, `A100_results/summary.json`)으로 다시 계산했다.
- N8에서는 세 프로그램 모두 시뮬레이터 최적에 도달해서 실측도 1% 안에서 같다.
- agent_cc는 N8_s1에서 실측 0.822(22% 손해)다. 느린 파이프라인 hop이 4개인 배치를 골랐는데, 시뮬레이터는 이 누적을 보지 못한다.
- 세 구분의 뜻:
  - train: 탐색 중 채점에 쓴 topology다.
  - held-out: 탐색이 끝난 뒤 확인용으로만 쓴 topology다. 다만 이 N8 3개의 실측은 v2 보정 데이터에 들어갔다.
  - 처음 보는 topology: 탐색에도 보정에도 쓰지 않았다.
- 자세한 내용은 [1633 worklog](worklog/2026-11-15_1633_a100-measured-score-agent-cc.md).

## 6. 프로그램 구조

**agent_cc (`v7.py`)**: 시뮬레이터 구조를 직접 이용하는 임계값 탐색에 surrogate 정제를 더했다.
1. 링크 행·열이 같은 GPU를 region class로 묶는다. 배치는 dp × pp 격자의 class 배정으로 표현한다. 같은 class 안 swap은 평가하지 않는다.
2. 과제 설명만으로 링크 모델을 만든다: `min(bw, 10 MB / RTT)`, activation 8 MB, gradient 300 MB.
   파이프라인 병목 C와 DP 병목 D를 계산한다.
3. C를 늘려 가며 실현 가능한 가장 작은 D를 SA로 찾는다. 끝 stage gradient 크기는 두 가지 가정을 둔다.
   이렇게 후보 약 10개를 만든다.
4. 후보를 `cost_fn`으로 재고 `T ≈ a·C + b·D + c`를 최소제곱으로 적합한다. 끝 stage 가중치는 적합 잔차로 고른다.
5. banded 배치(stage마다 region 하나, 모든 파이프라인이 같은 region 순서)를 k ≤ 7 class에서 전부 만들어 surrogate로 순위를 매긴다.
6. 상위 시작점에서 surrogate SA를 40회 재시작한다. 온도와 tie-break 가중치를 바꿔 가며 돌리고, 결과는 모두 `cost_fn`으로 잰다.
7. 남은 예산은 `cost_fn` hill climbing(class swap, hub 행 swap, 구간 뒤집기)에 쓴다. 2,000회 중 150–950회를 쓰고, topology당 8–17 s다.

**EvoX**: 대리 목적함수로 다양한 후보를 만든 뒤 시뮬레이터로 정제한다.
- `cost_fn` 결과를 memoization한다.
- 작은 N은 전수 탐색한다.
- 양방향 대역폭의 max로 링크 비용을 잡는다.
- 여러 (DP, pipeline) 가중치와 sum/max 대리 목적함수로 후보를 만든다.
- `cost_fn` GA를 돌리고, 남은 예산의 40%로 병목을 겨냥한 iterated local search를 한다.

**adaevolve**: DT-FM의 "DP 분할 + TSP 순서" 구조를 병목(max) 목적함수에 맞게 바꿨다.
1. TCP window를 반영한 링크 시간 행렬을 만든다.
2. GPU를 DP 그룹으로 분할한다.
3. pipeline마다 stage 순서를 open TSP로 푼다. 작은 pp는 Held-Karp 정확해와 min-max 변형을 쓰고, 큰 pp는 nearest-neighbour + 2-opt/Or-opt를 쓴다.
4. 병목 특징 기반 surrogate로 후보를 골라 `cost_fn`으로 확인한다.

## 7. 한계

- **세 프로그램 모두 v1 목적함수에 맞춰져 있다.** v2에서는 N64 여지 2.6%가 있다.
- **실측 손해의 원인은 탐색이 아니라 시뮬레이터에 있다.** hop 누적과 PP8 오차가 그 예다. 시뮬레이터를 고치면 프로그램도 그에 맞춰 다시 최적화해야 한다.
- **agent_cc 결과는 CPU 부하에 따라 조금 변한다.** 실행 시간 상한(`time.time()`)을 쓰기 때문이다.
- **N128 실행 시간 여유가 작은 프로그램이 있다.** EvoX는 30 s 제한에 걸린 적이 있다.
