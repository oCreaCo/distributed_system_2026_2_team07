# [2026-11-15 03:40] 1.27이 상한인가, plateau인가: headroom 추정 + 코딩 에이전트 탐색

## 1. headroom 추정 (`adrs/dtfm_placement/headroom.py`, 신규)
```bash
skydiscover/.venv/bin/python adrs/dtfm_placement/headroom.py --calls 200000     # -> output/headroom.json, 79 s
```
- 방법: 최고 프로그램 3개(EvoX, adaevolve, CEGAS 300)의 train placement을 시작점으로, GPU swap / replica swap 이웃의
  simulated annealing을 topology당 시뮬레이터 200k회(정규 예산의 100배)로 실행.
- 결과: **개선 0.** combined: evox 1.2674, adaevolve 1.2696, cegas 1.2706, topology별 best-of-3 **1.2773**, SA 1.2773.
- SA의 탐색력 확인 (random 시작 4개, 400k회, scratchpad `sa_random.py`):
  N32_s0 56.24–56.39 (프로그램과 같음), **N64는 69–78 s (프로그램 62–64 s)**. 단순 SA는 N64에서 약하다.

| N | 판단 |
|---|---|
| 8 | 전수 최적값을 세 프로그램 모두 달성. 여유 없음 |
| 16, 32 | 프로그램·SA·best-of-3가 일치. 최적 근처로 보임 (N32는 SA random 시작으로도 같은 값) |
| 64 | 프로그램끼리 불일치 (s0 62.3 vs 68.2, s1 64.6 vs 69.5, s2 64.3 vs 68.6). **여유 미확인** |

- 각 프로그램은 적어도 부분적으로 자기 plateau에 갇혀 있다. topology별로 셋 중 최선만 골라도 1.277.
- N64 3개에서 10% 더 줄이면 combined가 약 2.4% 오른다 (약 1.31).

## 2. 코딩 에이전트 탐색 (Claude Code subagent, 실행 중)
- 시작점: 위 세 프로그램. 채점은 `evaluator.py <file> train`만 (held-out 금지, 끝나고 한 번만 확인).
- 규칙: `output/agent_cc/`에만 쓰기, cost_fn 2,000회 / 30 s 유지, 파일 읽기·topology별 하드코딩 금지,
  오프라인 시뮬레이터 연구는 seed ≥ 1000 topology만. evaluator 약 25회 또는 90분.
- Codex는 별도 OpenAI 계정 과금이라 이번에는 쓰지 않았다.
- 주의: DeepSeek flash보다 훨씬 강한 모델이라 ADRS 방법들과 같은 조건의 비교가 아니다. "더 나은 알고리즘이 존재하는가"를 보는 실험이다.
