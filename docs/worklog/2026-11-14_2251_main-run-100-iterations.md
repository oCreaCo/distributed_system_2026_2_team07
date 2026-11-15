# [2026-11-14 22:51] 본 진화 실행 (adaevolve, 100 iteration): iteration을 늘리면 더 오르는가

## 목표
pilot(15 iteration)보다 iteration을 늘렸을 때 점수가 얼마나 더 오르는지 확인한다.

## 수행
```bash
# OpenRouter deepseek/deepseek-v4.1-flash, adrs/dtfm_placement/config.yaml 그대로
skydiscover optimize initial_program.py evaluator.py -c config.yaml -s adaevolve -i 100 -o output/main_adaevolve_100
skydiscover/.venv/bin/python adrs/dtfm_placement/evaluator.py output/main_adaevolve_100/best/best_program.py heldout
```
2026-11-14 21:32 시작, 4,351 s (72.5분), 비용 **$2.683** (iteration당 약 $0.027).

## 결과

checkpoint별 최고 점수 (train, `combined_score` / `vs_ga_sim`):

| iteration | 5 | 10 | 20 | 25 | 30 | 45 | 65 | 100 |
|---|---|---|---|---|---|---|---|---|
| combined | 1.2294 | 1.2407 | 1.2585 | 1.2667 | 1.2676 | 1.2693 | 1.2696 | 1.2696 |
| vs_ga_sim | 1.011 | 1.020 | 1.034 | 1.041 | 1.042 | 1.043 | 1.044 | 1.044 |

최종 최고 프로그램 (1,091줄):

| split | combined | vs_ga_sim | vs_greedy | N16 | N32 | N64 |
|---|---|---|---|---|---|---|
| train | 1.2696 | 1.044 | 1.168 | 1.126 | 1.256 | 1.376 |
| held-out | 1.2879 | 1.025 | 1.143 | – | – | – |

비교: pilot 15 iteration 최고 train 1.249 / held-out 1.277.

## 해석
- **iteration을 늘리면 오르지만 빠르게 포화한다.** 이득의 대부분은 25 iteration까지 나왔고(1.2667), 이후 75 iteration은 +0.003뿐이다.
  같은 run을 checkpoint에서 더 이어 돌리는 것은 비용 대비 이득이 거의 없을 것으로 본다(추가 실행은 하지 않음).
- held-out에서도 ga_sim 대비 +2.5%로 유지된다. 다만 ga_sim은 seed 하나 값이라 분산 확인이 남아 있다.
- 최고 프로그램 구조: TCP window를 반영한 링크 시간 행렬 → DP 그룹(replica) 분할 → 각 pipeline의 stage 순서를 open TSP로 풂
  (작은 pp는 Held-Karp 정확해, min-max 변형 포함; 큰 pp는 nearest-neighbour + 2-opt/Or-opt) → 병목 특징 기반 surrogate로 후보를 골라 실제 cost_fn으로 확인.
  원 DT-FM의 "DP 분할 + TSP 순서" 구조를 시뮬레이터의 병목(max) 특성에 맞게 바꾼 형태다.

## 다음 단계
- 포화의 원인 구분: 같은 2,000회 예산의 한계인지(큰 예산으로 상한 추정), 탐색 정체인지.
- ga_sim seed 분산.
- CEGAS 비교 ([`2026-11-14_2252_cegas-setup.md`](2026-11-14_2252_cegas-setup.md)).
