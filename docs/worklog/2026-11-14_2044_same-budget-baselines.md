# [2026-11-14 20:44] 같은 목적함수·같은 예산 기준선 추가

## 목표
기존 `combined_score`의 분모(DT-FM GA)는 논문 통신 비용 모델로 탐색한 배치라서, 후보의 점수에 "목적함수 차이"와
"탐색 알고리즘 차이"가 섞인다. 둘을 분리할 기준선을 만든다 (proposal 3.2: Random·Greedy·DT-FM 비교, 동일 evaluator).

## 변경
- `adrs/dtfm_placement/baselines.py` (신규) → `baselines.json` (gitignore)
  - `ga_sim`: 원 DT-FM GCMA (crossover + cyclic partitioning local search + 논문 TSP/bottleneck matching) 그대로,
    후보 적합도만 시뮬레이터로 교체. population 100 + trials 1900 = 시뮬레이터 2,000회 (후보 예산과 동일).
  - `greedy`: 시뮬레이터 0회. gradient 전송 시간 기준으로 가까운 GPU끼리 DP 그룹을 만들고, stage 순서와 매칭은 논문 TSP.
- `evaluator.py`: `vs_ga_sim`, `vs_greedy` 지표 추가 (`baselines.json`이 있을 때만). **`combined_score` 정의는 그대로.**
  기준선은 topology별 상수이므로 바꿔도 후보 간 순위는 같다. `baselines` 모듈 import도 금지 목록에 추가.
- `config.yaml`: LLM을 OpenRouter `deepseek/deepseek-v4.1-flash`로 변경 (모델별 `api_base`/`api_key` 지정.
  지정하지 않으면 SkyDiscover가 `deepseek/` 접두사를 자체 DeepSeek provider로 해석해 endpoint를 바꿈).

```bash
skydiscover/.venv/bin/python adrs/dtfm_placement/baselines.py    # 약 6분 (24 topology 병렬)
```

## 결과

DT-FM GA(논문 모델) 대비 점수, 기하평균 (`combined_score`와 같은 정의):

| 방법 | 시뮬레이터 호출 | train 전체 | train N≥16 | held-out 전체 | held-out N≥16 |
|---|---|---|---|---|---|
| seed GA (`initial_program.py`) | 2,000 | 1.048 | 0.966 | – | – |
| random | 2,000 | 1.050 | 0.969 | 1.044 | 0.995 |
| greedy | 0 | 1.087 | 1.020 | 1.127 | 1.123 |
| **ga_sim** (DT-FM GCMA + 시뮬레이터) | 2,000 | **1.217** | **1.186** | **1.256** | **1.280** |

seed의 새 지표: `vs_ga_sim 0.861`, `vs_greedy 0.964`.

## 해석
- **진짜 넘어야 할 기준은 1.05가 아니라 ga_sim의 1.22**다. 그 미만의 진화 결과는 "목적함수를 시뮬레이터로 바꾼 효과"일 뿐
  DT-FM 탐색 알고리즘보다 낫다는 근거가 되지 못한다. 최종 보고서는 `vs_ga_sim`을 주 지표로 써야 한다.
- seed GA는 무작위 swap 기반이라 random과 같고, 구조(DP 그룹을 먼저 정하고 stage 순서·매칭을 조합적으로 푸는 것)를
  쓰는 GCMA가 같은 예산에서 훨씬 좋다. greedy는 호출 0회로 seed보다 낫지만 topology별 편차가 큼 (N64_s1 104.8 s vs ga_sim 69.6 s).
- ga_sim은 N=8에서 항상 최적에 도달하지 못한다 (N8_s1 72.86 vs 최적 69.42). stage 순서·매칭을 논문 통신 모델의
  TSP로 고정하기 때문으로 보인다(미검증).
- 원 GCMA 코드의 버그 (수정하지 않음, 기록만): `random.seed = i` / `np.random.seed = i`는 seed 설정이 아니라 함수를 덮어씀.
  교체 조건 `offspring_score > max(a, offspring_score > b)`는 비교식 오류.

## 다음 단계
- 모델 pilot (DeepSeek v4.1 flash vs Gemini 3.8 flash, adaevolve 15 iteration씩) → 주 실행 모델 결정.
- seed를 ga_sim 수준으로 바꿀지, 프롬프트에 기준선 수치를 알려줄지 결정 (pilot 결과 본 뒤).
