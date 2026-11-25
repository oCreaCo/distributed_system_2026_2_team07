# 프로젝트 현황 개요

최종 갱신: 2026-11-25. 이어받는 사람은 [handoff.md](handoff.md)부터 읽는다.

## 목표 (proposal 요약)

DT-FM의 GPU 배치 알고리즘(Hybrid GA + Local Search)을 ADRS(LLM 기반 프로그램 진화)로 개선한다.
후보 알고리즘은 A100 실측으로 보정한 CPU 시뮬레이터로 빠르게 채점하고(inner loop), 좋은 후보만 A100에서
검증한다(outer loop). 성공 기준은 시뮬레이터 점수가 아니라 **실측 학습 시간 개선**이다.

| 목표 | 상태 |
|---|---|
| G1. DT-FM 배치 탐색 재현 | 완료. 원 scheduler 호출(`k8s/run_scheduler.py`). A100 실측은 11-03에 74 run, 11-15에 63 run과 NCCL bench 2개 |
| G2. 유용한 시뮬레이터 | v2: 120 run LOTO MAPE 10.8%, 순서 쌍 391/438, Spearman 0.854. 느린 hop 누적과 PP8 오차(26%)는 미반영 → [simulator.md](simulator.md) |
| G3. 자동 탐색 성능 평가 | adaevolve·EvoX·CEGAS 모두 train 1.27 근처에서 포화. agent_cc가 최고이고, MILP 정확 최적값 대비 0.1–2.6% 차이 → [policies.md](policies.md), [MILP](milp-exact-optimum.md) |
| G4. 일반화·재현성 | 실측에서 agent_cc가 DT-FM보다 평균 1.139배 빠르다 (N8, 10 topology). 처음 보는 topology에서는 1.089배. 재현 절차와 검증 결과 → [reproduce.md](reproduce.md) |
| G5. 남은 여지 | 탐색 여지는 v2 기준 N64 2.6%. 실측 여지는 주로 시뮬레이터 정확도에 있다 → [handoff.md](handoff.md#3-다음-단계-우선순위-순) |

## 부분별 문서

| 문서 | 내용 |
|---|---|
| [simulator.md](simulator.md) | 시뮬레이터 v2: 모델, 파라미터, 보정 데이터, 정확도, 알려진 오차, 버전 |
| [milp-exact-optimum.md](milp-exact-optimum.md) | MILP 정확 최적값: 정식화, 결과(v1·v2), 가능한 이유와 범위, 한계 |
| [policies.md](policies.md) | best program 상태: 파일, 점수(v1·v2), 일반화, 최적값 차이, 실측, 구조 |
| [reproduce.md](reproduce.md) | 재현 절차: 보정, 캐시, v1 환경, 채점, adaevolve·EvoX·agent_cc 재실행, 분석, A100 |
| [handoff.md](handoff.md) | 현재 상태 요약, 다음 단계, 주의사항, 레포 지도 |

## 주요 수치

**점수 정의**
- `combined_score`는 topology별 `sim(DT-FM GA 배치) / sim(후보 배치)`의 기하평균이다.
- train은 N = 8, 16, 32, 64 × seed 0–2이고, held-out은 seed 100–102다.
- topology당 `cost_fn` 2,000회, 30초 예산을 쓴다.
- `vs_ga_sim`은 같은 시뮬레이터와 같은 예산으로 돌린 DT-FM GCMA 대비 값이다. 탐색 효과만 보는 주 지표다.

시뮬레이터 점수 (진화 당시 v1 기준; 현재 v2 기준 값은 [policies.md](policies.md#2-점수)):

| 방법 | train | held-out | 비고 |
|---|---|---|---|
| seed GA | 1.048 | – | 시뮬레이터 2,000회 |
| random | 1.050 | 1.044 | 시뮬레이터 2,000회 |
| greedy | 1.087 | 1.127 | 시뮬레이터 0회 |
| **ga_sim** (DT-FM GCMA + 시뮬레이터) | **1.217** | **1.256** | 실질적인 기준선. seed 3개 편차는 train 1.1%, held-out 0.2% |
| adaevolve 100 it. (deepseek-v4.1-flash) | 1.270 | 1.288 | $2.68, 25 it. 이후 포화 |
| EvoX 100 it. (deepseek-v4.1-flash) | 1.267 | 1.307 | $2.72, best는 58 it. |
| CEGAS, LLM 100회 / suite 순서 반전 / 300 요청 | 1.206 / 1.247 / 1.271 | 1.221 / 1.286 / 1.296 | $0.28 / $0.49 / $1.37. 실행 간 편차가 크다 |
| **agent_cc** (Claude Code 에이전트) | **1.277** | **1.322** | vs_ga_sim 1.050 / 1.052 |

A100 검증:

| 실험 | 결과 | 문서 |
|---|---|---|
| held-out N8 3개, 찾은 배치 vs DT-FM | 실측 1.29–1.31× / 1.09× / 1.11× 빠름. 시뮬레이터는 8–21% 과대 추정 | [0320](worklog/2026-11-15_0320_a100-validation-n8.md) |
| 순위 보존 (PP4×DP2, 21 run) | Spearman 0.85 / 0.93 / 0.34, top-1 2/3 | [0530](worklog/2026-11-15_0530_a100-ranking-check.md) |
| calib2 (PP8, PP4×DP2, PP2×DP4, 26 run) | 순서 6개 중 5개 유지. PP8은 12–47% 과소 추정. 방향 효과 최대 7.6%. 실측 대역폭은 tc 설정의 50–88% | [0858](worklog/2026-11-15_0858_a100-calib2-data.md) |
| agent_cc 실측 채점 (N8 10개, 17 run) | DT-FM 대비 1.139배 (train 1.18 / held-out 1.16 / 새 topology 1.09). N8_s1은 0.82 | [1633](worklog/2026-11-15_1633_a100-measured-score-agent-cc.md) |

## 해석상 주의점

1. **원 DT-FM GA와 비교하면 목적함수 효과와 탐색 효과가 섞인다.**
   - DT-FM GA는 자체 통신 비용 모델로 탐색하고, 후보는 시뮬레이터로 탐색한다.
   - 예를 들어 N=8의 1.34배는 전부 목적함수 효과다. random도 최적에 도달한다.
   - 그래서 탐색 효과는 `vs_ga_sim`으로 본다.
2. **N=8은 포화되어 있다.** 모든 합리적 후보가 최적값을 찾는다. 그래서 시뮬레이터 점수 차이는 N ≥ 16에서만 나는데,
   A100 실측은 N=8에서만 가능하다.
3. **시뮬레이터 점수는 실측 이득을 과장한다.** N8 10개 topology에서 v2는 1.175로 예측했고 실측은 1.139였다.
   오차 원인은 [simulator.md](simulator.md#5-알려진-오차와-원인)에 있다.
4. **진화와 에이전트 탐색은 모두 시뮬레이터 v1에서 이뤄졌다.** v1과 v2 점수를 섞어 비교하지 않는다.
5. **점수는 단독 실행으로 잰다.** 평가기를 동시에 여러 개 돌리면 시간 제한 때문에 점수가 떨어진다.
