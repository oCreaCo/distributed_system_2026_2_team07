# [2026-11-14 23:18] CEGAS 100회 실행 결과와 기준선(ga_sim) seed 편차

## 1. CEGAS (LLM 100회)

```bash
adrs/dtfm_placement/cegas/run.sh r100_s0 100 100     # 2026-11-14 22:51, 786 s, $0.2825 (OpenRouter 사용량 차이)
skydiscover/.venv/bin/python adrs/dtfm_placement/evaluator.py adrs/dtfm_placement/output/cegas_r100_s0/best/best_program.py heldout
```
로그: `output/cegas_r100_s0/a1/` (`summary.json`, `requests.jsonl`, `archive.jsonl`, `quality.jsonl`).

| 지표 | 값 |
|---|---|
| LLM 요청 / 공식 평가 | 100 / 24 (요청당 평가 0.24) |
| 최고 train combined / vs_ga_sim | **1.2062** / 0.991 (마지막 평가 #24에서 갱신, 아직 상승 중) |
| held-out combined / vs_ga_sim | 1.2207 / 0.972 |
| 비용 (loop 집계 / 사용량 차이) | $0.298 / $0.283, prompt 2.65M + completion 0.18M 토큰 |
| 행동 | run_program 40 (전부 panel 입력), edit_draft 23, evaluate 11, checkout 7, show_input 6, check_draft 1, 파싱 실패 8 (lenient parse 15) |

평가별 점수 궤적: 1.048 → 1.111 (#1, surrogate SA) → 1.163 (#3) → 1.169 (#11) → 1.191 (#19, 구조 기반 greedy seed) → 1.195 (#23) → 1.206 (#24).
도중 회귀 3번 (#7 0.61, #9 0.10, #12 0.34: TLE·버그), 에이전트가 checkout으로 복구.

suite 피드백 사용: 응답 기록 200개 중 72개가 실패 입력·목표·reference를 언급
(예: "N64 and N16_s201 still failing … focus on N64"). 다만 반례 입력을 직접 실행해 보는 후속(`run_program_witness`)은 0.

### adaevolve와 비교
| 방법 | 비용 | train | held-out |
|---|---|---|---|
| CEGAS, LLM 100회 | $0.28 | 1.206 | 1.221 |
| adaevolve, 10 iteration 시점 (같은 비용대) | 약 $0.27 | 1.241 | – |
| adaevolve, 100 iteration | $2.68 | 1.270 | 1.288 |
| ga_sim (기준선, seed 0) | – | 1.217 | 1.256 |

- 이 1회 실행에서 CEGAS는 **같은 비용에서도, 같은 iteration 수에서도 adaevolve보다 낮다.** 기준선 ga_sim에도 못 미친다.
- 해석상 주의:
  1. adaevolve의 실제 LLM 호출 수는 기록되지 않아 "같은 호출 수" 비교는 근사다.
  2. seed 1회다. TCResearch 기록상 이 계열 결과는 seed 분산이 크다.
  3. CEGAS는 평가 24회만 썼고 끝까지 상승 중이었다. 요청의 47%가 무료 행동(로컬 실행 등)이었다.
  4. reference가 adaevolve 최고 프로그램이라 suite 목표가 처음부터 멀었다(seed 대비 6/8 입력 실패). 논문 설정(reference = 같은 예산의 기존 optimizer 최고)과 같지만, 이 과제에서는 목표가 멀어 반례가 "어디가 부족한지"보다 "전부 부족함"에 가까웠을 수 있다.
  5. 논문 실험은 gpt-5, 여기는 deepseek-v4.1-flash.

## 2. ga_sim 기준선의 seed 편차

같은 GCMA + 시뮬레이터를 RNG seed 1, 2로 재실행 (`output/ga_sim_seeds.json`):

| split | seed 0 | seed 1 | seed 2 | topology별 best-of-3 |
|---|---|---|---|---|
| train | 1.2166 | 1.2132 | 1.2265 | 1.2303 |
| held-out | 1.2561 | 1.2563 | 1.2540 | 1.2635 |

- seed 간 차이는 train 약 1.1%, held-out 0.2%.
- adaevolve 최고(1.270 / 1.288)는 seed 최댓값보다 train +3.5%, held-out +2.5%, 시뮬레이터 호출 3배를 쓴 best-of-3보다도 높다.
  → **adaevolve가 찾은 알고리즘이 DT-FM 탐색(시뮬레이터 목적함수)보다 낫다는 결론은 seed 잡음 이상이다** (시뮬레이터 기준; 실측 검증은 남음).

## 다음 단계 (제안)
- CEGAS를 판단하려면 seed를 늘리거나(요청 100회당 약 $0.3), 요청 예산을 늘린 실행(예: 300회)이 필요하다.
- reference를 덜 강한 프로그램(예: ga_sim 수준)으로 둔 변형으로 반례가 더 구체적일 때의 행동 확인.
