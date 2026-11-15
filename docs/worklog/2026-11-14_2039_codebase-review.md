# [2026-11-14 20:39] 코드베이스 파악과 파이프라인 재실행

## 목표
README, proposal, project description, 코드(`sim/`, `adrs/`, `k8s/`)를 읽고, A100 없이 돌릴 수 있는 부분을 재실행해 현재 상태를 확인한다.

## 수행
```bash
cd skydiscover && uv sync && uv pip install --python .venv/bin/python scipy && cd ..   # Python 3.12.11, numpy 2.4.2, scipy 1.18.1
skydiscover/.venv/bin/python sim/fit.py                         # 1.7 s -> sim/theta.json
skydiscover/.venv/bin/python adrs/dtfm_placement/reference.py   # 6 min 45 s (12 cores) -> reference.json
skydiscover/.venv/bin/python adrs/dtfm_placement/evaluator.py adrs/dtfm_placement/initial_program.py
```
추가 분석: DT-FM GA 배치의 시간 분해와, 임의 swap 300개의 비용 변화 비율.

## 결과
- 보정·reference·seed 점수 모두 README와 일치 (seed `combined_score 1.048`).
- reference (train): DT-FM GA vs 같은 예산 random(시뮬레이터)

| topology | DT-FM GA | random | 최적 |
|---|---|---|---|
| N8_s0/s1/s2 | 70.9 / 73.8 / 90.6 | 50.4 / 69.4 / 56.8 | random과 동일 |
| N16_s0/s1/s2 | 66.7 / 71.9 / 70.5 | 78.9 / 70.4 / 64.0 | – |
| N32_s0/s1/s2 | 80.7 / 71.5 / 70.3 | 93.8 / 71.5 / 70.9 | – |
| N64_s0/s1/s2 | 74.2 / 100.6 / 97.3 | 92.1 / 98.4 / 86.7 | – |

- DT-FM GA 배치에서 swap 비용 불변 비율 0.24–0.94, 개선 비율 0.01–0.26. 큰 N일수록 plateau가 넓음.
- 시간 분해: N8–N16은 DP 동기화(opt/dp) 비중 23–37 s, N32–N64는 파이프라인이 50–80 s로 지배.

해석과 문제점은 [`../overview.md`](../overview.md)의 "알려진 문제" 참고.

## 다음 단계
1. 같은 시뮬레이터·같은 예산 기준선 추가 (DT-FM GCMA + 시뮬레이터 fitness, Greedy).
2. 그 결과로 seed와 프롬프트 개선 여부 판단.
3. OpenRouter flash 모델로 SkyDiscover 진화 실행 (예산 $50).
