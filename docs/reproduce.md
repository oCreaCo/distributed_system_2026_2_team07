# 재현 절차

최종 갱신: 2026-11-25. 대상은 시뮬레이터 보정, 채점, adaevolve·EvoX·agent_cc의 탐색 과정, 분석 결과, A100 실측이다.
명령은 모두 저장소 루트에서 실행한다. 재현용 스크립트는 `adrs/dtfm_placement/repro/`에 있다.

| 파일 | 용도 |
|---|---|
| `repro/run_skydiscover.sh` | SkyDiscover 실행 1회와 OpenRouter 비용 기록 |
| `repro/config_evox.yaml` | EvoX 설정 (과제 `config.yaml` + `search.share_llm: true`) |
| `repro/setup_v1_worktree.sh` | 진화 당시의 시뮬레이터 v1 환경을 git worktree로 다시 만든다 |
| `repro/agent_cc_prompt.md` | agent_cc를 만든 서브에이전트 프롬프트 (원문 그대로) |

**무엇이 정확히 재현되고 무엇이 안 되나**
- **결정적이라 그대로 재현된다:** 시뮬레이터 보정값, 기준선 캐시, 커밋된 프로그램의 점수, MILP 최적값.
- **재현되지 않는다:** LLM 탐색 과정(adaevolve, EvoX, agent_cc). LLM 샘플링에 seed가 없어서 다시 돌리면 다른 프로그램이 나온다.
  재현할 수 있는 것은 같은 설정으로 돌리는 절차와, 비슷한 점수대에 도달하는지 여부다.
- **조금씩 흔들린다:** agent_cc 점수. 실행 시간 상한을 쓰기 때문에 CPU 부하에 따라 ±0.003 정도 변한다.

## 0. 환경

```bash
cd skydiscover && uv sync && uv pip install --python .venv/bin/python scipy && cd ..
export OPENROUTER_COURSE_PROJECT_KEY=...     # LLM 실행에만 필요 (과목 키, 한도 $50)
```

- Python은 항상 `skydiscover/.venv/bin/python`을 쓴다 (Python 3.12.11, numpy 2.4.2, scipy 1.18.1).
- 평가기 하나가 프로세스 약 12개를 쓴다. 점수를 잴 때는 평가기를 한 번에 하나만 돌린다. 동시에 돌리면 시간 제한 때문에 점수가 떨어진다.

## 1. 시뮬레이터 보정

현재 v2 θ는 `sim/theta.json`으로 커밋되어 있어 보통 다시 만들 필요가 없다. 다시 만들 때는 다음과 같이 한다.

```bash
P=skydiscover/.venv/bin/python
$P -c "import json; t=json.load(open('adrs/dtfm_placement/output/sim_v1/theta.json')); t['dp_bw_eff']=0.9; \
json.dump(t, open('/tmp/init_v2.json','w'))"
$P sim/fit.py --init /tmp/init_v2.json --out /tmp/theta_v2.json     # sim/theta.json과 차이 0 (2026-11-15 확인)
$P sim/compare_versions.py                                          # v1 / v2 LOTO 비교 -> sim/loto_v1_v2.json
```

- `fit.py`의 기본 `--out`은 `sim/theta.json`이라 덮어쓴다. 비교할 때는 다른 경로를 준다.
- `--init` 없이 돌리면 `beta`가 3.5% 다른 값이 나온다 ([simulator.md](simulator.md#2-파라미터-v2-simthetajson)).
- 보정 데이터는 `A100_results/summary.json`과 `fit.py`의 `INLINE_CONFIGS`에 적힌 배치 구성 파일이다. 둘 다 커밋되어 있다.

## 2. 기준선 캐시

평가기는 `adrs/dtfm_placement/reference.json`(DT-FM GA, random, N8 전수 탐색 최적)과 `baselines.json`(ga_sim, greedy)을 읽는다.
이 파일들은 gitignore 대상이고 현재 시뮬레이터로 다시 만든다.

```bash
$P adrs/dtfm_placement/reference.py      # 약 7분, 프로세스 24개
$P adrs/dtfm_placement/baselines.py
```

v1 캐시는 `adrs/dtfm_placement/output/sim_v1/`에 커밋되어 있다.

## 3. v1 환경 (진화 당시 재현)

adaevolve, EvoX, agent_cc는 모두 시뮬레이터 v1에서 만들어졌다. v1 환경은 다음 명령으로 만든다.

```bash
adrs/dtfm_placement/repro/setup_v1_worktree.sh ../team07_v1
```

- `e07b345` 커밋으로 worktree를 만든다. 이 커밋에 v1 `stepsim.py`와 당시 과제 코드가 있다.
- 그 위에 다음을 복사한다.
  - 현재 `config.yaml`: OpenRouter 설정. system message는 당시와 같다.
  - `evaluator.py`와 `baselines.py`: `vs_ga_sim` 지표용. `combined_score`에는 영향이 없다.
  - v1 θ와 v1 캐시.
- 다 쓰면 `git worktree remove ../team07_v1`로 지운다.

검증 (2026-11-25에 실행):
- e07b345의 `sim/fit.py`를 다시 돌리면 `output/sim_v1/theta.json`과 차이 0이다.
- 세 프로그램을 이 worktree의 평가기로 채점한 결과:

| 프로그램 | 기록된 v1 train / held-out | 재채점 train / held-out |
|---|---|---|
| adaevolve | 1.2696 / 1.2879 | 1.26964 / 1.28793 (같음) |
| EvoX | 1.2674 / 1.3075 | 1.26735 / 1.30747 (같음) |
| agent_cc | 1.2773 / 1.322 | 1.27727 / 1.31904 (held-out은 wall-clock 영향) |

```bash
$P ../team07_v1/adrs/dtfm_placement/evaluator.py adrs/dtfm_placement/output/main_adaevolve_100/best/best_program.py train
```

## 4. 현재 시뮬레이터(v2)로 채점

```bash
for p in agent_cc/best_program.py main_evox_100/best/best_program.py main_adaevolve_100/best/best_program.py; do
  for s in train heldout; do $P adrs/dtfm_placement/evaluator.py adrs/dtfm_placement/output/$p $s; done
done
```

평가기 한 번에 1–2분이 걸린다. 기대값은 [policies.md](policies.md#2-점수)의 v2 열이다. 2026-11-25에 다시 재서 확인했다.

## 5. adaevolve와 EvoX 다시 돌리기

```bash
# 원래 실행과 같은 조건 (시뮬레이터 v1)
ROOT=$(realpath ../team07_v1) adrs/dtfm_placement/repro/run_skydiscover.sh adaevolve 100
ROOT=$(realpath ../team07_v1) adrs/dtfm_placement/repro/run_skydiscover.sh evox 100 adrs/dtfm_placement/repro/config_evox.yaml
# 현재 시뮬레이터(v2)로 새로 탐색할 때는 ROOT를 빼고 실행한다
```

- 출력은 `<ROOT>/adrs/dtfm_placement/output/main_<search>_<iters>/`에 생긴다. 같은 이름의 폴더가 있으면 실행하지 않는다.
  - `best/best_program.py`, `best/best_program_info.json`
  - `checkpoints/checkpoint_<k>/`: 5 iteration마다 저장된다. `skydiscover viewer`로 볼 수 있다.
  - `logs/`
- LLM은 OpenRouter `deepseek/deepseek-v4.1-flash`다. 모델 비교는 [2132 worklog](worklog/2026-11-14_2132_model-pilot.md)에 있다.
- 원래 실행의 기록:

| | adaevolve | EvoX |
|---|---|---|
| 시작 | 2026-11-14 21:32 | 2026-11-14 23:19 |
| 시간 / 비용 | 72.5분 / $2.68 | 88분 / $2.72 |
| 최고 프로그램 | iteration 65, 1.2696 | iteration 58, 1.2674 |
| 5 / 10 / 25 / 50 / 100 it. 최고 | 1.229 / 1.241 / 1.267 / 1.269 / 1.270 | 1.254 / 1.261 / 1.266 / 1.266 / 1.267 |

- 둘 다 25 iteration 근처에서 포화했다. 재현이 목적이라면 25–30 iteration(약 $0.7)이면 같은 점수대(약 1.267)에 도달하는지 볼 수 있다.
- 비용 기록은 실행 전후의 OpenRouter `GET /api/v1/key` 사용량 차이다. 키 자체는 출력하지 않는다.

## 6. agent_cc 다시 돌리기

agent_cc는 SkyDiscover가 아니라 Claude Code 서브에이전트가 직접 프로그램을 작성하고 평가기로 확인하는 방식으로 만들었다.

- 원래 실행:
  - 2026-11-15 03:37–04:00 KST, `general-purpose` 서브에이전트, 모델 claude-opus-5-5.
  - 평가기 7회. 그중 3회는 기존 프로그램 재측정이다.
  - 결과는 `output/agent_cc/`에 있다: `log.md`, `v1.py`–`v8.py`, `best_program.py` = `v7.py`.
- 프롬프트: `adrs/dtfm_placement/repro/agent_cc_prompt.md`.
  - 과제 계약을 설명한다.
  - 기존 최고 프로그램 3개를 시작점으로 준다.
  - 당시 알려진 사실을 알려 준다: N8·N16·N32는 포화, N64에만 여지.
  - 규칙을 준다:
    - `output/agent_cc/`에만 쓴다.
    - 시뮬레이터는 `cost_fn`으로만 쓴다.
    - held-out 평가는 금지한다.
    - 오프라인 연구는 seed ≥ 1000 topology에서만 한다.
    - 평가기는 한 번에 하나, 약 90분.
  - 산출물도 정한다.

다시 돌리는 방법:
1. v1 worktree를 만든다 (3절). 프롬프트의 시뮬레이터 설명은 v1 기준이다 (예: "끝 stage gradient가 크다").
2. 프롬프트의 저장소 경로 `/home/yeomjy/dist2026/distributed_system_2026_2_team07`를 worktree 경로로 바꾼다.
3. 시작점 프로그램 3개를 worktree의 `adrs/dtfm_placement/output/`에 같은 경로로 복사한다. 모두 레포에 있다.
   - `main_adaevolve_100/best/best_program.py`, `main_evox_100/best/best_program.py`
   - `cegas_r300_s0/best/best_program.py` (CEGAS 300 요청, v1 train 1.2706). CEGAS 결과 중 레포에 있는 것은 이 프로그램뿐이다.
4. worktree에서 Claude Code를 열고, 이 프롬프트로 `general-purpose` 서브에이전트를 띄운다.
   비대화형으로 돌리려면 `claude -p "$(cat prompt.md)"`를 쓴다.
5. 끝나면 `log.md` 표를 확인하고 3·4절 방식으로 채점한다. 원래 결과는 train 1.2773(N64 62.30 / 64.65 / 64.35)이다.

- 같은 프로그램은 나오지 않는다. 원래 실행은 "class 격자 + (C, D) 임계값 탐색 + surrogate SA" 방법에 도달했고, train에서
  기존 셋 중 최선의 합(1.2773)과 같았다.
- 같은 날 두 번째 에이전트(agent_cc2, N64 dev set 개선)도 돌렸다. 프로그램은 개선하지 못했고, 대신 MILP solver를 만들었다.
  결과는 로컬 `output/agent_cc2/`에 있고 레포에는 없다. solver는 `exact_milp.py`로 옮겨 커밋했다.

## 7. 분석 재현

| 결과 | 명령 | 출력 (`adrs/dtfm_placement/output/`) |
|---|---|---|
| 일반화 (새 topology, N16–N128) | `$P adrs/dtfm_placement/generalization.py --programs agent_cc=adrs/dtfm_placement/output/agent_cc/best_program.py --out adrs/dtfm_placement/output/generalization_v2.json` | `generalization_v2.json/.txt` |
| 정확한 최적값 차이 | `$P adrs/dtfm_placement/exact_gap.py --gen adrs/dtfm_placement/output/generalization_v2.json --out adrs/dtfm_placement/output/exact_gap_v2.json` | `exact_gap_v2.json/.txt` |
| 한 topology의 정확한 최적값 | `cd adrs/dtfm_placement && $P exact_milp.py 64 5000 5001` | 표준 출력 (JSON 줄) |
| A100 실측 점수 | `$P adrs/dtfm_placement/a100_score.py` | `a100_score.json` |
| 파라미터 오류 민감도 | `$P adrs/dtfm_placement/model_sensitivity.py` | `model_sensitivity_P.json` |
| 큰 예산 SA로 headroom 추정 | `$P adrs/dtfm_placement/headroom.py --calls 100000` | `headroom.json` |

- `generalization.py`는 topology마다 원 DT-FM GA와 ga_sim도 돌리므로 수십 분 걸린다. 프로세스는 12개를 쓴다.
- `exact_gap.py`의 입력(`--gen`)은 같은 시뮬레이터 버전으로 만든 결과여야 한다.
- MILP는 확인 하나에 20 s 제한이 있다. 시간을 넘긴 확인은 실현 불가로 처리하므로, 그 topology의 값은 상한이다.

## 8. A100 실측

절차는 루트 `README.md`의 "A100에서 DT-FM 실행하기"와 같다. 실측 채점에 쓴 구성은 다음과 같다.

```bash
P=skydiscover/.venv/bin/python
$P adrs/dtfm_placement/make_a100_config.py adrs/dtfm_placement/output/agent_cc/best_program.py \
    --seeds 5000 5001 5002 5003 --label agentcc --out adrs/dtfm_placement/output/a100_N8_agentcc_fresh.json
nohup k8s/run_batches.sh none acc_fresh:adrs/dtfm_placement/output/a100_N8_agentcc_fresh.json &
$P adrs/dtfm_placement/a100_score.py
```

- 실제로 실행한 구성 파일은 `adrs/dtfm_placement/output/a100_N8_*.json`과 `k8s/calib2/*.json`에 커밋되어 있다.
- run 하나에 약 8분 걸린다. Job 하나는 최대 2시간이다.
- `k8s/sync.sh`는 커밋된 `A100_results/` 로그를 덮어쓴다. 끝나면 `git status`로 확인하고, 기존 파일은 `git restore`로 되돌린다.
- 같은 배치를 반복해 잰 잡음은 0.01%였다.
