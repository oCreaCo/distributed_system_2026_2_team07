# [2026-11-15 04:45] 과적합 확인: 처음 보는 topology와 N=128

```bash
skydiscover/.venv/bin/python adrs/dtfm_placement/generalization.py --programs seed=initial_program.py \
    evox=output/main_evox_100/best/best_program.py adaevolve=output/main_adaevolve_100/best/best_program.py \
    cegas300=output/cegas_r300_s0/best/best_program.py agent_cc=output/agent_cc/best_program.py   # -> output/generalization.json, 36분
```
- **topology:** N = 16, 32, 64, 128 × seed 5000–5004.
  - train(0–2), held-out(100–102), 에이전트 연구용(1000–4999) 어디에도 쓰이지 않은 seed다.
  - N=128(PP8×DP16, stage당 3 layer)은 처음 쓰는 크기다. `topologies.py`는 고치지 않고 스크립트 안에서만 추가했다.
- **비교 대상:** topology마다 원 DT-FM GA, ga_sim(같은 시뮬레이터·같은 2,000회 예산의 GCMA), 프로그램 5개.
- **조건:** 2,000회 예산은 그대로다.
  - 시간 제한은 품질과 속도를 따로 보려고 120 s로 늘렸다. 30 s를 넘은 run은 따로 셌다(공식 채점이었다면 실패).
- **주의:** 12개 worker가 12코어를 꽉 채운 상태(load 13)였다.
  - 그래서 시간 수치는 단독 실행보다 길게 나왔을 수 있다.
  - agent_cc는 시간 기준으로 탐색을 멈추므로 품질에도 영향이 있었을 수 있다.

## 결과: ga_sim 대비 (topology별 비율의 기하평균, >1이면 ga_sim보다 좋음)

| 프로그램 | N16 train → 새 topology | N32 | N64 | N128 (새 topology만) | 30 s 초과 (N128) |
|---|---|---|---|---|---|
| seed | – → 0.952 | – → 0.625 | – → 0.691 | 0.759 | 0/5 |
| EvoX | 1.045 → 1.028 | 1.052 → 1.046 | 1.053 → **0.927** | 1.062 | 1/5 (최대 30.0 s) |
| adaevolve | 1.046 → 1.033 | 1.052 → 1.032 | 1.060 → 1.022 | 1.060 | 0/5 (최대 11.9 s) |
| CEGAS 300 | 1.046 → **0.929** | 1.052 → 1.015 | 1.063 → 1.021 | 1.026 | **5/5 (최대 43.8 s)** |
| agent_cc | – → 1.033 | – → 1.064 | – → 1.051 | **1.084** | 0/5 (최대 19.3 s) |

DT-FM 대비 ga_sim: N16 1.088, N32 1.339, N64 1.232, N128 1.121.
train 값은 `output/headroom.json`(프로그램별 train 배치 시간)과 `baselines.json`(ga_sim)에서 계산했다.

## 해석
- **"N ≤ 32에 과적합"이라는 형태는 아니다.**
  - 세 ADRS 프로그램 모두 처음 보는 N=128에서도 ga_sim보다 2.6–6.2% 좋다.
  - 큰 N에서 무너지는 패턴은 없다.
- **대신 train topology 12개 자체에 맞춰진 흔적이 있다.** 새 topology에서는 셋 다 ga_sim 대비 우위가 줄고, 두 경우는 기준선보다 나빠졌다.
  - EvoX N64: 1.053 → 0.927. 새 N64에서는 같은 예산의 DT-FM GA보다 7% 나쁘다.
  - CEGAS N16: 1.046 → 0.929.
  - 가장 고르게 일반화한 것은 adaevolve다(우위가 1.3–3.8% 줄지만 모든 N에서 ga_sim보다 좋음).
- **실행 시간 확장이 숨은 문제다.**
  - CEGAS 프로그램은 N128 5개 모두 30 s를 넘었다. 공식 채점이었다면 전부 실패다.
  - EvoX도 1개가 경계(30.0 s)에 걸렸다.
  - train에는 N ≤ 64만 있어서 진화 중에는 이 문제가 드러나지 않는다.
- **agent_cc가 모든 N에서 가장 좋다**(ga_sim 대비 1.033–1.084). 시간도 20 s 안이다.
- **한계:** seed 5개라 크기별 표본이 작다.
  - ga_sim의 seed 편차는 train 기준 약 1%다.
  - 2–3% 수준의 차이는 확정적이지 않다.
  - EvoX N64(−12%)와 CEGAS N16(−11%)은 그보다 훨씬 크다.

## 개선 방향
1. **채점 topology를 늘리거나 매 평가마다 바꾸기.** 고정된 12개는 포화됐고 과적합을 부른다.
   - 예: 매 평가 새 seed, 또는 N별 seed 수 확대.
2. **채점에 큰 N의 시간 검사 추가.** 예: N=128 validity·시간 검사를 N=2, 4 검사처럼 추가.
3. **진화 시스템 프롬프트에 객관적인 구조 정보 넣기.** agent_cc가 찾은 구조(파이프라인 병목 가중치, 끝 stage gradient)다.
   - 다만 이것은 사람이 정보를 준 것이므로 실험 조건으로 따로 표시해야 한다.
