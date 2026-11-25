# Handoff

최종 갱신: 2026-11-25. 이 작업을 이어받는 사람이 처음 읽을 문서다.

## 1. 한눈에 보는 현재 상태

- **목표:** DT-FM GPU 배치 알고리즘보다 나은 알고리즘을 ADRS(LLM 프로그램 진화)로 찾고, A100 실측으로 확인한다.
- **최고 policy:** `adrs/dtfm_placement/output/agent_cc/best_program.py`.
  - A100 실측에서 DT-FM보다 평균 **1.139배** 빠르다 (N8, 10개 topology).
  - 처음 보는 topology에서는 1.089배이고, 10개 중 2개에서는 졌다.
- **시뮬레이터 점수:** v2 기준 train 1.269 / held-out 1.306. 같은 예산의 DT-FM 탐색(ga_sim)보다 3.0% / 4.7% 높다.
- **탐색은 시뮬레이터 기준으로 거의 최적이다.** MILP 정확 최적값 대비 N16–N128에서 0.1–2.6% 차이다.
- **남은 이득은 시뮬레이터 정확도에 있다.** 느린 hop 누적과 PP8 오차가 대표적이다. 실측에서 진 경우의 원인도 여기에 있다.

## 2. 문서 지도

| 문서 | 내용 |
|---|---|
| [overview.md](overview.md) | 목표별 상태와 주요 수치 요약 |
| [simulator.md](simulator.md) | 시뮬레이터 v2 모델, 파라미터, 정확도, 알려진 오차 |
| [milp-exact-optimum.md](milp-exact-optimum.md) | 정확한 최적값(MILP)의 정식화, 결과, 가능한 이유, 한계 |
| [policies.md](policies.md) | adaevolve, EvoX, agent_cc 최고 프로그램의 점수, 구조, 한계 |
| [reproduce.md](reproduce.md) | 보정, 채점, 탐색, 분석, A100 실측 재현 절차 |
| [worklog/](worklog/) | 작업 단위 기록 (시간순, 색인은 [README](README.md)) |
| [progress_page.html](progress_page.html) | 진행 현황 웹페이지 원본 |

## 3. 다음 단계 (우선순위 순)

1. **시뮬레이터 v3: 느린 hop 누적 항.**
   - 후보: micro-batch당 λ·(pipeline 한 줄의 hop 지연 합).
   - 시험 적합 결과: λ = 0.081, 순서 쌍 387 → 394/438, top-1 손해 1.73 → 1.31%.
   - 할 일:
     - `stepsim.py`에 넣는다.
     - `fit_two_stage`로 재적합한다.
     - LOTO(`compare_versions.py` 방식)와, 보정에 쓰지 않은 agent_cc 실측 17 run으로 검증한다.
   - PP8 과소 추정도 함께 설명되는지 본다.
2. **v3로 policy 다시 최적화.**
   - 지금 프로그램은 v1 구조(끝 stage gradient가 큼, C·D 두 병목만 봄)를 가정한다.
   - hop 합 항이 들어가면 MILP의 (C, D) 구조가 깨진다. 비교 기준을 다시 정식화해야 한다
     ([MILP 문서 5a절](milp-exact-optimum.md#5a-정확해가-가능한-이유와-범위)).
3. **다음 A100 슬롯에서 확인.**
   - 대상: 재최적화한 배치의 N8_s1, s5000(실측에서 진 topology)과 fresh seed 몇 개.
   - 가능하면 같은 topology에서 PP8과 PP4를 짝지어 잰다. PP8 메커니즘을 가르는 데 필요하다.
4. 그 밖에 남은 것:
   - 파이프라인 방향 효과(최대 7.6%)는 모델에 없다.
   - N64 dev set으로 탐색을 조금 더 개선할 여지가 있다 (v2 기준 2.6%).

## 4. 주의사항

- **비밀값**
  - `team07.kubeconfig`(루트 심볼릭 링크)에는 private key가 들어 있다. gitignore(`*.kubeconfig`) 대상이고, 내용을 출력하거나 공유하지 않는다.
  - OpenRouter 키는 환경변수 `OPENROUTER_COURSE_PROJECT_KEY`로만 쓴다. 스크립트는 키를 출력하지 않는다.
- **LLM 예산:** 과목 키 한도는 $50이고, 11-15 기준 약 $9.4를 썼다. flash급 모델만 쓴다. 주력은 `deepseek/deepseek-v4.1-flash`다.
- **점수 측정**
  - 평가기는 한 번에 하나만 돌린다. 동시에 10개를 돌렸을 때 EvoX가 0.65, agent_cc가 1.247까지 떨어졌다.
  - agent_cc는 실행 시간 상한을 써서 결과가 ±0.003 흔들린다.
- **`sim/fit.py`의 기본 출력은 `sim/theta.json`이다.** 실험할 때는 `--out`을 준다. 덮어썼다면 `git restore sim/theta.json`으로 되돌린다.
- **held-out N8 topology(s100–s102)는 v2 보정 데이터에 들어갔다.** 이 3개에 대한 v2 예측은 실측에 맞춰져 있다.
  공정한 일반화 지표는 seed 5000번대 topology다.
- **A100**
  - 팀 할당 시간에만 쓸 수 있다. GPU는 8장, Job은 최대 2시간이다.
  - `k8s/sync.sh`가 커밋된 `A100_results/` 로그를 덮어쓴다. 실행 후 `git status`로 확인하고 `git restore`로 되돌린다.
- **TCResearch**(CEGAS 원본)는 읽기 전용이다. `PYTHONPATH`로만 쓴다.
- **레포에 없는 것:** `adrs/dtfm_placement/output/`의 대부분(약 600 MB)은 로컬에만 있다.
  - checkpoint, SkyDiscover 로그, CEGAS 결과, agent_cc2, 분석 JSON 등이다.
  - 레포에 있는 것: A100 배치 구성(`a100_N8_*.json`), v1 θ와 캐시(`sim_v1/`), adaevolve·EvoX·agent_cc의 최고 프로그램과 기록.

## 5. 레포 지도 (작업 관련 부분)

| 경로 | 내용 |
|---|---|
| `sim/stepsim.py`, `sim/theta.json`, `sim/fit.py` | 시뮬레이터 v2와 보정 |
| `sim/compare_versions.py`, `sim/variants_eval.py` | 버전 비교 (LOTO), 수정 후보 평가 |
| `adrs/dtfm_placement/evaluator.py`, `config.yaml`, `initial_program.py` | SkyDiscover 과제 (채점, 설정, seed) |
| `adrs/dtfm_placement/reference.py`, `baselines.py`, `topologies.py` | 기준선(DT-FM GA, ga_sim, greedy)과 topology 생성 |
| `adrs/dtfm_placement/exact_milp.py`, `exact_gap.py` | MILP 정확 최적값 |
| `adrs/dtfm_placement/generalization.py`, `headroom.py`, `dev_eval.py`, `model_sensitivity.py` | 분석 도구 |
| `adrs/dtfm_placement/make_a100_config.py`, `a100_score.py` | A100 실측 구성 생성과 채점 |
| `adrs/dtfm_placement/repro/` | 재현 스크립트와 프롬프트 |
| `adrs/dtfm_placement/output/{agent_cc, main_evox_100/best, main_adaevolve_100/best}` | 최고 프로그램 |
| `adrs/dtfm_placement/cegas/` | CEGAS 이식 코드 (TCResearch 사용) |
| `k8s/`, `A100_results/` | A100 실험 도구와 측정 결과 (11-03 74 run, 11-15 63 run + bench 2) |
