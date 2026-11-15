# [2026-11-15 00:48] EvoX 기준선 결과 (100 iteration)

```bash
CFG=<scratchpad>/config_evox.yaml bash <scratchpad>/run_main.sh evox 100   # config = 과제 config + search.share_llm: true
skydiscover/.venv/bin/python adrs/dtfm_placement/evaluator.py adrs/dtfm_placement/output/main_evox_100/best/best_program.py heldout
```
2026-11-14 23:19 시작, 5,280 s (88분), rc=0, OpenRouter 사용량 차이 **$2.72**. 출력 `output/main_evox_100`.

## 결과

| 방법 | train | vs_ga_sim (train) | held-out | vs_ga_sim (held-out) | 비용 |
|---|---|---|---|---|---|
| ga_sim seed 0 | 1.217 | 1.000 | 1.256 | 1.000 | – |
| CEGAS, LLM 100회 | 1.206 | 0.991 | 1.221 | 0.972 | $0.28 |
| adaevolve 100 it. | **1.2696** | 1.044 | 1.288 | 1.025 | $2.68 |
| **EvoX 100 it.** | 1.2674 | 1.042 | **1.3075** | **1.041** | $2.72 |

- 최고 프로그램은 iteration 58에서 나왔다. N별 DT-FM 대비: N8 1.336, N16 1.125, N32 1.256, N64 1.368. 실패 0.
- train은 adaevolve와 사실상 같다(−0.2%, ga_sim seed 편차 1.1%보다 작음). **held-out은 EvoX가 +1.5% 높다**
  (ga_sim seed 편차 0.2%보다 크다). 다만 1회 실행이다.

### 최고 점수 궤적 (checkpoint 기준)

| iteration | 5 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|
| adaevolve | 1.229 | 1.241 | 1.267 | 1.269 | 1.270 |
| EvoX | 1.254 | 1.261 | 1.266 | 1.266 | 1.267 |

- 두 방법 모두 25 iteration 근처에서 포화했다. EvoX는 초반(5–10 it.)이 더 빠르다.
- iteration당 시간은 EvoX가 길었지만(53 s vs 43 s 평균) 비용은 거의 같다.

### 최고 프로그램 구조 (`best/best_program.py`)
- cost_fn 결과 memoization (같은 rank_map 재평가를 무료로).
- 작은 N은 전수 탐색.
- 양방향 대역폭의 max로 링크 비용을 잡고, 여러 (DP, pipeline) 가중치와 sum/max 대리 목적함수로 구조적으로 다양한 후보를 생성 → 그룹별 상위만 cost_fn으로 평가.
- cost_fn 기반 GA, 남은 예산의 40%로 iterated local search(병목 겨냥 hill-climb).
- adaevolve 최고와 같은 계열(대리 목적함수 seed + 시뮬레이터 정제)이다.

## 해석
- 같은 iteration 수·비용에서 SkyDiscover의 두 탐색법은 train에선 구별되지 않는다. held-out 차이는 seed 반복 없이는 판단 유보.
- CEGAS(1회, 100 요청)는 둘보다 낮다. trace 분석(`2026-11-14_2325`)상 원인 후보는 반례 제시 순서(N16 우선)와 예산이다.

## 다음 후보 (미실행, 승인 필요)
1. CEGAS suite 순서 N64 → N32 → N16 (약 $0.3).
2. CEGAS 300 요청 (약 $0.9).
3. adaevolve / EvoX seed 반복 (각 약 $2.7; 25 it.로 줄이면 약 $0.7).

예산: 누적 약 $7.53 / $50.
