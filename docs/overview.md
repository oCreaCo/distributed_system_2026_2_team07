# 프로젝트 현황 개요

최종 갱신: 2026-11-15 21:40 (작업 내역은 [`worklog/`](worklog/) 참고)

## 목표 (proposal 요약)

DT-FM의 GPU 배치 알고리즘(Hybrid GA + Local Search)을 ADRS(LLM 기반 프로그램 진화)로 개선한다.
후보 알고리즘은 A100 실측으로 보정한 CPU 시뮬레이터로 빠르게 채점하고(inner loop), 좋은 후보만 A100에서
검증한다(outer loop). 성공 기준은 시뮬레이터 점수가 아니라 **실측 학습 시간 개선**이다.

| 목표 | 상태 |
|---|---|
| G1. DT-FM 배치 탐색 재현 | 완료: 원 scheduler 호출(`k8s/run_scheduler.py`), A100 74 run 실측 (11-03) + 63 run과 NCCL bench 2개 (11-15) |
| G2. 유용한 시뮬레이터 | v2 (11-15): 120 run LOTO MAPE 10.8%, 순서 쌍 391/438, Spearman 0.854. PP8 오차 26%와 느린 hop 누적은 미반영 ([1500](worklog/2026-11-15_1500_simulator-v2.md)) |
| G3. 자동 탐색 성능 평가 | 진행 중: adaevolve 100 it.가 ga_sim 대비 train +4.4% / held-out +2.5% (ga_sim seed 편차 1.1%보다 큼). CEGAS는 1.206–1.271로 실행 간 편차가 큼. 좋은 실행은 adaevolve와 같은 점수를 1/5 비용으로 (순서 반전 효과는 확인 안 됨) |
| G4. 일반화·재현성 | **실측**: 현재 최적 policy(agent_cc)가 N8 10개 topology에서 DT-FM보다 평균 1.139배 빠름 (train 1.18 / held-out 1.16 / 처음 보는 topology 1.09). 1개는 22% 손해 ([1633](worklog/2026-11-15_1633_a100-measured-score-agent-cc.md)). 일부 프로그램은 train 12개에 과적합 |
| G5. 남은 여지 | v1 기준으로는 탐색이 사실상 최적(N64 0.5%). v2 기준으로는 N64 2.6%, N32 1.1% 여지 ([MILP 문서](milp-exact-optimum.md)). 실측 여지는 시뮬레이터(느린 hop 누적, PP8)에 있음 |

## 구성 요소

- `sim/stepsim.py`: 해석적 step-time 모델. `T_iter = G·(M+S−1)·[max_s overlap(f_s,c_s,α) + max_s overlap(b_s,c_s,α)] + overlap(T_opt, T_dp, β)`.
  링크 유효 대역폭 = `min(bw, tcp_window / RTT)`, DP gradient 링크는 여기에 `dp_bw_eff`(0.94)를 곱함 (v2).
  파라미터 10개를 `sim/fit.py`가 2단계(no-tc → 네트워크)로 적합. 데이터는 11-03 74 run + 11-15 46 run.
  v1(11-03)은 git `095eef5`, v1 캐시는 `adrs/dtfm_placement/output/sim_v1/`.
  - `sim/theta.json`(v2)은 레포에 포함한다. 적합 결과가 시작점에 따라 달라지기 때문이다. v1 theta + `dp_bw_eff` 0.9에서
    시작하면(`fit.py --init`) 그대로 재현되고, 기본 시작점에서는 `beta`가 3.5% 다른 값이 나온다.
- `adrs/dtfm_placement/`: SkyDiscover 과제. 진화 대상 `schedule(delay_ms, bandwidth_gbps, pp, dp, cost_fn, budget)`.
  topology당 cost_fn 2,000회, 30초. N = 8, 16, 32, 64 × train seed 3개 = 12개 채점, N = 2, 4는 유효성만 검사.
  `combined_score` = topology별 `sim(DT-FM GA 배치) / sim(후보 배치)`의 기하평균.
- `k8s/`: A100 실험 도구 (팀 할당 시간에만 사용 가능).
- `skydiscover/`: SkyDiscover 0.2.0. Python 환경은 uv로 만든 `skydiscover/.venv` (numpy 2.4.2, scipy 1.18.1, Python 3.12.11).

## 확인된 수치 (2026-11-14 재실행, 시뮬레이터 v1 기준)

보정 θ: `f_layer 0.0042, b_layer 0.0112, f_emb 0.0109, b_emb 0.0193, alpha 0.0, beta 0.166, opt_per_gparam 13.6, lat_fixed_ms 0.0, tcp_window_mb 10.1`

| 지표 (leave-one-topology-out) | 시뮬레이터 | 논문 통신 비용 모델 |
|---|---|---|
| MAPE | 3.2–14.6% | 13.7–40% |
| pairwise 순위 정확도 | 0.75–1.00 | 0.27–0.93 (T4, T5는 spearman 음수) |
| top-1 regret | 모든 topology에서 0 s | 최대 13 s |

seed 프로그램: `combined_score 1.048`, `vs_random 0.998`, N별 DT-FM 대비 속도 향상 N8 1.34 / N16 0.99 / N32 0.95 / N64 0.95.

## 기준선과 진화 결과 (DT-FM GA 대비 점수, 시뮬레이터 v1 기준. v2 재채점은 [1500](worklog/2026-11-15_1500_simulator-v2.md))

| 방법 | train | held-out | 비고 |
|---|---|---|---|
| seed GA | 1.048 | – | 시뮬레이터 2,000회 |
| random | 1.050 | 1.044 | 시뮬레이터 2,000회 |
| greedy | 1.087 | 1.127 | 시뮬레이터 0회 |
| **ga_sim** (DT-FM GCMA + 시뮬레이터) | **1.217** | **1.256** | 시뮬레이터 2,000회. 실질적인 기준선 |
| pilot best, deepseek-v4.1-flash (15 it.) | 1.249 | 1.277 | $0.41 |
| pilot best, gemini-3.8-flash (15 it.) | 1.250 | 1.255 | $1.32 |
| **adaevolve 100 it.** (deepseek-v4.1-flash) | **1.270** | **1.288** | $2.68, 25 it. 이후 포화 |
| **EvoX 100 it.** (deepseek-v4.1-flash) | 1.267 | **1.307** | $2.72, best는 58 it., 25 it. 이후 포화 |
| CEGAS, LLM 100회 (tcopt q_editeval) | 1.206 | 1.221 | $0.28, 평가 24회 |
| **CEGAS, suite 순서 반전** (N64 → N16) | 1.247 | 1.286 | $0.49, 평가 62회 |
| **CEGAS, 300 요청** (원래 순서) | 1.271 | 1.296 | $1.37. 요청 81에서 1.270 ($0.46). 같은 설정 1차(1.206)와 편차 큼 |
| **agent_cc** (Claude Code 에이전트, 구조 기반 + 시뮬레이터 정제) | **1.277** | **1.322** | vs_ga_sim 1.050 / 1.052. wall-clock 상한(`time.time()`)을 써서 실행 속도에 의존 |
| ga_sim seed 0 / 1 / 2 | 1.217 / 1.213 / 1.227 | 1.256 / 1.256 / 1.254 | 기준선 편차 |

LLM 설정: OpenRouter, 주력 `deepseek/deepseek-v4.1-flash` (key `$OPENROUTER_COURSE_PROJECT_KEY`, 한도 $50).

## 알려진 문제와 해석상 주의점

1. **기준선이 목적함수 효과와 탐색 효과를 섞는다.** DT-FM GA는 자체 통신 비용 모델로, 후보는 시뮬레이터로 탐색한다.
   N=8의 1.34배는 전부 목적함수 효과(random도 최적 도달). → `baselines.py`의 ga_sim으로 분리함. 주 지표는 `vs_ga_sim`.
   ga_sim seed 3개 편차는 train 1.1%, held-out 0.2%.
2. **N=8은 포화**: 모든 합리적 후보가 최적값을 찾으므로 점수 차이는 N ≥ 16의 9개 topology에서만 난다.
   그런데 A100 실측 검증은 N=8에서만 가능하다.
3. **seed GA ≈ random search.** N ≥ 32에서는 구조를 보는 DT-FM GA가 시뮬레이터 기반 random보다 크게 낫다
   (N64_s0: 74 s vs 92 s). 구조 기반 초기화 + 시뮬레이터 정제가 유망.
4. **plateau가 많은 목적함수.** DT-FM 배치에서 임의 swap의 35–94%가 비용 불변, 개선은 1–26%.
   병목 stage/링크를 겨냥한 이웃 연산이 유리할 것으로 예상.
5. **외삽.** 보정은 N=8(주로 PP4×DP2)뿐인데 PP8×DP8까지 채점. central_ps 서버의 동시 수신 경합은 모델에 없음(검증 안 됨).
6. **방향성 미반영 가능성.** `p8d1_T1_rev` 오차 −39%: 시뮬레이터는 identity와 역순을 같은 값(23.07 s)으로 예측, 실측은 21.16 s vs 37.77 s.
7. 사소: README의 `project_team07.pdf`는 실제로 `references/dist_project_proposal.pdf`.
   `stepsim.fit()`과 `fit_two_stage` 내부 `residuals`는 미사용 코드. "fit on all"에는 held-out T4–T6가 포함되므로 LOO 수치를 인용할 것.

## A100 검증 요약 (2026-11-15)

| 실험 | 결과 | 문서 |
|---|---|---|
| held-out N8 3개, 찾은 배치 vs DT-FM | 실측 1.29–1.31× / 1.09× / 1.11× 빠름. 시뮬레이터는 9 run 모두 8–21% 과대 추정 | [0320](worklog/2026-11-15_0320_a100-validation-n8.md) |
| 순위 보존 (PP4×DP2, 21 run) | Spearman 0.85 / 0.93 / 0.34, top-1 2/3. 일부 배치 7–15% 과소 추정 | [0530](worklog/2026-11-15_0530_a100-ranking-check.md) |
| calib2 (PP8, PP4×DP2, PP2×DP4, 26 run) | 순서 6개 중 5개 유지. PP8은 12–47% 과소 추정. 방향 효과 최대 7.6%. 실측 대역폭은 tc 설정의 50–88% | [0858](worklog/2026-11-15_0858_a100-calib2-data.md) |
| 최적 policy(agent_cc) 실측 채점 (N8 10개, 17 run) | DT-FM 대비 1.139배 (train 1.18 / held-out 1.16 / 새 topology 1.09). N8_s1은 0.82 (느린 hop 누적) | [1633](worklog/2026-11-15_1633_a100-measured-score-agent-cc.md) |

8. **링크 대역폭 과대 가정.** NCCL 실측은 tc 설정의 50–88%(지연이 클수록 낮음)인데, 시뮬레이터는 window 항이
   걸리지 않아 100%로 본다. v2에서 DP gradient 링크에만 효율 0.94를 반영 (모든 링크에 적용하면 PP4×DP2를 과대 추정).
9. **`k8s/sync.sh`가 커밋된 로그를 덮어씀.** 동기화 후 `git restore`로 복구해야 한다 (미수정).
10. **마지막 stage 파라미터 오류 (v2에서 수정).** embedding은 stage 0에만 있고 마지막 stage는 head 4.2M뿐인데
   시뮬레이터는 둘 다 embedding을 준다. 고치면 MAPE 14.5→13.8%, 순위 지표는 같음, DT-FM 대비 점수 −1.6~2.1%,
   최적화 손해 0.4–0.7% ([0941](worklog/2026-11-15_0941_calib2-analysis-and-optimization-status.md)).
11. **PP8 과소 추정 (최대 47%)은 파이프라인 단계에서 생김.** 병목 링크·bench 대역폭·링크 합 모델 모두 설명 못 함.
   N32·N64 채점이 PP8이라 목적함수에 직접 영향.
12. **느린 파이프라인 hop의 누적 (미반영).** 시뮬레이터는 가장 느린 링크 하나로 파이프라인 시간을 정하지만, 실측은
   느린 hop이 늘 때마다 micro-batch당 약 그 지연만큼 늘어난다. agent_cc가 N8_s1에서 실측 22% 손해를 본 원인
   ([1633](worklog/2026-11-15_1633_a100-measured-score-agent-cc.md)). λ·(hop 지연 합) 항으로 시험해 보니 순서 쌍
   387→394/438. 정식 검증 전.
13. **평가 시 CPU 부하에 따라 점수가 바뀜.** wall-clock 상한을 쓰는 프로그램(agent_cc)과 느린 프로그램(EvoX, CEGAS)은
   평가기를 동시에 여러 개 돌리면 점수가 떨어진다. 공식 점수는 단독 실행으로 잴 것.
