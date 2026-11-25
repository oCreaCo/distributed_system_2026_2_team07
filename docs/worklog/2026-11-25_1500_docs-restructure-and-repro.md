# 2026-11-25 15:00 문서 재구성과 best program 재현 자료

## 한 일

- docs를 부분별 문서로 나눴다.
  - `simulator.md`: 시뮬레이터 v2
  - `milp-exact-optimum.md`: MILP 분석 (기존 문서, 연결만 추가)
  - `policies.md`: best program 상태
  - `reproduce.md`: 재현 절차
  - `handoff.md`: 인계용 요약
  - `overview.md`는 목표별 상태와 주요 수치 요약으로 줄였다. 시뮬레이터 오차 항목은 `simulator.md`로, 운영상 주의점은 `handoff.md`로 옮겼다.
- adaevolve, EvoX, agent_cc의 결과를 커밋 대상에 넣었다 (`.gitignore` 예외). CEGAS는 요청에 따라 제외했다.
  - adaevolve·EvoX: `best/best_program.py`, `best/best_program_info.json`
  - agent_cc: `best_program.py`, `log.md`, 중간 버전 `v1.py`–`v8.py`
  - checkpoint와 로그(약 210 MB)는 제외했다.
- 재현 자료 `adrs/dtfm_placement/repro/`:
  - `run_skydiscover.sh`: 11-14 실행 스크립트를 정리한 것이다. transcript에서 복원했다. 원본은 scratchpad에 있어 지워졌다.
  - `config_evox.yaml`: 과제 설정 + `search.share_llm: true`
  - `setup_v1_worktree.sh`: 시뮬레이터 v1 환경
  - `agent_cc_prompt.md`: 서브에이전트 프롬프트 원문. 프롬프트가 시작점으로 준 CEGAS 300 프로그램(`output/cegas_r300_s0/best/best_program.py`)도 함께 커밋했다 (11-25 추가)
- v1 캐시(`output/sim_v1/reference.json`, `baselines.json`)도 커밋했다.
- 루트 README의 오래된 안내를 고쳤다 (LLM 설정 gpt-5 → OpenRouter, `fit.py` 기본 실행 → 커밋된 θ).

## 검증

- **v1 환경** (`setup_v1_worktree.sh`로 만든 e07b345 worktree):
  - `sim/fit.py` 재적합 결과가 `output/sim_v1/theta.json`과 차이 0이다.
  - adaevolve는 1.26964 / 1.28793, EvoX는 1.26735 / 1.30747로 기록과 같다.
  - agent_cc train은 1.27727로 같다. held-out은 1.31904로, 기록된 1.322와 wall-clock 영향만큼 다르다.
- **v2 (현재)**, 단독 실행:
  - agent_cc 1.2694 / 1.3057 (11-15: 1.2686 / 1.3068)
  - EvoX 1.2592 / 1.3004, adaevolve 1.2506 / 1.2781 (11-15와 같음)
- **`run_skydiscover.sh`**: 출력 폴더가 이미 있을 때와 키가 없을 때 실행하지 않고 멈추는 것을 확인했다. LLM 실행은 비용이 들어 돌리지 않았다.
- **`config_evox.yaml`**: SkyDiscover `load_config`로 읽으면 `share_llm = True`다.
- **문서 링크**: 상대 경로 대상이 모두 있다.

## 남은 것

- LLM 탐색 과정 자체는 재현되지 않는다 (샘플링 seed 없음). 같은 설정으로 25–30 iteration을 돌려 같은 점수대에 도달하는지 확인하는 데
  약 $0.7가 든다. 실행하지 않았다.
