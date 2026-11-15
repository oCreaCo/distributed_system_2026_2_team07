# 2026-11-15 21:40 문서 정리와 레포 커밋

## 문서

- `milp-exact-optimum.md`: 5a절 "정확해가 가능한 이유와 범위"를 추가했다.
  - 핵심: 시뮬레이터가 있어서가 아니라 시뮬레이터가 식이 공개된 해석적 모델이고 시간이 (C, D) 두 값으로 정해지기
    때문이다.
  - 실측·블랙박스 시뮬레이터·과제 policy·DT-FM 비용 모델에는 같은 방법을 쓸 수 없고, hop 지연 합 항을 넣으면 구조가 깨진다.
  - 코드 목록을 현재 solver(`exact_milp.py`) 기준으로 고쳤다.
- `overview.md`: 갱신 시각, G1 run 수(11-15: 학습 63 run + NCCL bench 2), A100 검증 표에 실측 채점 행,
  theta 재현 방법을 반영했다.

## 커밋 범위

- 11-15 A100 결과(`A100_results/` 로그, `summary.json`, `jobs/`)와 calib2 구성(`k8s/calib2/`).
- 시뮬레이터 v2(`sim/stepsim.py`, `sim/fit.py`, `sim/theta.json`)와 검증 스크립트(`compare_versions.py`, `variants_eval.py`).
- ADRS 도구: `baselines.py`, `cegas/`, `dev_eval.py`, `headroom.py`, `generalization.py`, `exact_milp.py`, `exact_gap.py`,
  `model_sensitivity.py`, `a100_score.py`, `make_a100_config.py` 수정, evaluator의 `vs_<baseline>` 지표.
- `docs/` 전체.

`.gitignore` 변경:
- **`sim/theta.json`을 추적한다.** 기본 시작점으로 `fit.py`를 다시 돌리면 `beta` 0.128 → 0.124, `opt_per_gparam`
  13.48 → 13.64처럼 최대 3.5% 다른 값이 나왔다. v1 theta + `dp_bw_eff` 0.9에서 시작하면(`--init`) 차이 0으로 재현된다.
  평가기 점수가 이 값에 직접 의존하므로 파일 자체를 커밋했다.
- `adrs/dtfm_placement/output/`에서 다음만 추적한다.
  - `a100_N8_*.json`: A100 배치 구성. `fit.py`와 `a100_score.py`가 읽는다.
  - `sim_v1/theta.json`: v1 비교에 쓴다.
  - `agent_cc/best_program.py`: 실측 채점한 policy.
  - 나머지 진화 출력(약 600 MB)은 계속 제외한다.
- `A100_results/**/status_pod*.txt`는 제외한다. 이전 커밋도 이 파일을 넣지 않았다.

커밋 전에 추가 파일 전체에서 API key와 private key 패턴을 검사했다. 매치는 `cegas/run.sh`의 환경변수 참조 한 줄뿐이고,
키 값은 들어 있지 않다.
