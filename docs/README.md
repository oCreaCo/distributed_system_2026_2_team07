# docs

- [`handoff.md`](handoff.md): 이어받는 사람용. 현재 상태, 다음 단계, 주의사항, 레포 지도
- [`overview.md`](overview.md): 목표별 상태와 주요 수치 요약 (항상 최신 상태로 유지)
- [`simulator.md`](simulator.md): 시뮬레이터 v2 모델, 파라미터, 정확도, 알려진 오차
- [`milp-exact-optimum.md`](milp-exact-optimum.md): 시뮬레이터 목적함수의 정확한 최적값(MILP) 정식화, 결과, 한계
- [`policies.md`](policies.md): adaevolve·EvoX·agent_cc 최고 프로그램의 점수, 구조, 한계
- [`reproduce.md`](reproduce.md): 보정, 채점, 탐색, 분석, A100 실측 재현 절차
- [`progress_page.html`](progress_page.html): 진행 현황 웹페이지 원본
- [`worklog/`](worklog/): 작업 단위 기록. 파일명과 제목에 `YYYY-MM-DD_HHMM` timestamp를 붙인다.

## 작업 내역

| 시각 | 문서 | 요약 |
|---|---|---|
| 2026-11-14 20:39 | [codebase-review](worklog/2026-11-14_2039_codebase-review.md) | 코드 파악, uv venv 구성, 보정·reference·seed 재실행 |
| 2026-11-14 20:44 | [same-budget-baselines](worklog/2026-11-14_2044_same-budget-baselines.md) | DT-FM GCMA + 시뮬레이터(ga_sim), greedy 기준선. 넘어야 할 점수는 1.22 |
| 2026-11-14 21:32 | [model-pilot](worklog/2026-11-14_2132_model-pilot.md) | DeepSeek vs Gemini 15 iteration. 점수 동일, DeepSeek 3.3배 저렴 → 주력. 본 실행 시작 |
| 2026-11-14 22:51 | [main-run-100-iterations](worklog/2026-11-14_2251_main-run-100-iterations.md) | adaevolve 100 it.: 1.2696 (held-out 1.288), 25 it. 이후 포화, $2.68 |
| 2026-11-14 22:52 | [cegas-setup](worklog/2026-11-14_2252_cegas-setup.md) | TCResearch CEGAS 이식: test suite·검증(V2 ρ=0.67)·sanity 통과, 100회 실행 시작 |
| 2026-11-14 23:18 | [cegas-result-and-baseline-variance](worklog/2026-11-14_2318_cegas-result-and-baseline-variance.md) | CEGAS 100회: 1.206 / held-out 1.221, $0.28 (adaevolve보다 낮음). ga_sim seed 편차 1.1% → adaevolve 개선은 잡음 이상 |
| 2026-11-14 23:25 | [cegas-trace-hypotheses](worklog/2026-11-14_2325_cegas-trace-hypotheses.md) | trace 검증: 반례는 따라감(H5 기각), 첫 실패만 제시돼 N64 격차 방치, 끝까지 상승(예산 부족), DSML 파싱 실패 9%. EvoX 기준선 시작 |
| 2026-11-15 00:48 | [evox-baseline-result](worklog/2026-11-15_0048_evox-baseline-result.md) | EvoX 100 it.: train 1.267 (adaevolve 1.270과 같음), held-out 1.307 (+1.5%), $2.72. 둘 다 25 it. 근처 포화 |
| 2026-11-15 01:33 | [cegas-reversed-suite-order](worklog/2026-11-15_0133_cegas-reversed-suite-order.md) | suite 순서 N64→N16: CEGAS 1.206 → 1.247 (held-out 1.221 → 1.286, adaevolve 100 it.와 비슷), $0.49. 평가 24 → 62회 |
| 2026-11-15 03:00 | [cegas-300-requests-and-order-correction](worklog/2026-11-15_0300_cegas-300-requests-and-order-correction.md) | CEGAS 300 요청: 1.2706 / held-out 1.296, 요청 81에서 이미 1.270 ($0.46). 같은 설정끼리 1.206 vs 1.270 → 순서 반전 효과는 확인 안 됨 (정정) |
| 2026-11-15 03:20 | [a100-validation-n8](worklog/2026-11-15_0320_a100-validation-n8.md) | A100 실측: 찾은 placement이 held-out N8 3개 모두 DT-FM보다 1.09–1.31배 빠름. 시뮬레이터는 8–21% 과대 추정. test.py tc 버그 수정 |
| 2026-11-15 03:40 | [headroom-and-agent-search](worklog/2026-11-15_0340_headroom-and-agent-search.md) | headroom: N8 최적, N16/32 최적 근처, 여유는 N64에만 (best-of-3 1.277). Claude Code 에이전트 탐색 시작 |
| 2026-11-15 04:45 | [generalization-overfit-check](worklog/2026-11-15_0445_generalization-overfit-check.md) | 새 topology·N=128: 큰 N에서 무너지진 않지만 train 12개에 맞춰짐 (EvoX N64 1.053→0.927, CEGAS N16 1.046→0.929 vs ga_sim). CEGAS 프로그램은 N128 5/5가 30 s 초과. agent_cc가 모든 N에서 최고 |
| 2026-11-15 05:20 | [n64-exact-optimum](worklog/2026-11-15_0520_n64-exact-optimum.md) | N64 개선 시도는 개선 없음. MILP 정확 최적값: 현재 최고 프로그램이 새 N64 60개 중 55개에서 최적, 차이 0.3%. 배치 탐색은 사실상 포화, 남은 여지는 시뮬레이터 정확도 |
| 2026-11-15 05:30 | [a100-ranking-check](worklog/2026-11-15_0530_a100-ranking-check.md) | A100 순위 보존: Spearman 0.85 / 0.93 / 0.34, 쌍 48/61 일치, top-1은 2/3. 반복 잡음 0.01%. 일부 배치를 7–15% 과소 추정해 순서가 깨짐 (원인 미확인) |
| 2026-11-15 08:58 | [a100-calib2-data](worklog/2026-11-15_0858_a100-calib2-data.md) | 추가 보정 데이터 26 run (PP8×DP1, PP4×DP2, PP2×DP4 × 새 topology 2개 + NCCL 측정). 순서는 6개 중 5개 유지. PP8은 12–47% 과소 추정. 파이프라인 방향 효과 최대 7.6% (시뮬레이터는 0). 실측 대역폭은 설정의 50–88%. 시뮬레이터는 수정 안 함 |
| 2026-11-15 09:41 | [calib2-analysis-and-optimization-status](worklog/2026-11-15_0941_calib2-analysis-and-optimization-status.md) | 시뮬레이터 오류 확인: 마지막 stage에 embedding이 없음(로그). 수정 후보 P는 MAPE 14.5→13.8%지만 순위는 그대로, S·R·tc 공유는 기각. PP8 오차는 파이프라인 단계. 최적화: agent_cc가 N16–128 정확 최적값과 0–1% 차이, P 오류로 인한 최적화 손해 0.4–0.7%. 시뮬레이터는 수정 안 함 |
| 2026-11-15 15:00 | [simulator-v2](worklog/2026-11-15_1500_simulator-v2.md) | 시뮬레이터 v2: 마지막 stage 파라미터 수정 + DP 대역폭 효율 0.94, 120 run으로 재보정. LOTO 순서 쌍 388→391/438, Spearman 0.839→0.854, top-1 손해 동일, PP8은 그대로(26%). v2 재채점 agent_cc 1.269/1.307. v2에서는 N64 탐색 여지 2.6% |
| 2026-11-15 16:33 | [a100-measured-score-agent-cc](worklog/2026-11-15_1633_a100-measured-score-agent-cc.md) | 현재 최적 policy(agent_cc)의 A100 실측: N8 10개 topology에서 DT-FM보다 평균 1.139배 빠름 (train 1.18, held-out 1.16, 처음 보는 topology 1.09). N8_s1에서 22% 손해: 느린 파이프라인 hop 누적을 시뮬레이터가 못 봄. 시뮬 v2가 v1보다 실측에 가까움 |
| 2026-11-15 21:40 | [docs-update-and-commit](worklog/2026-11-15_2140_docs-update-and-commit.md) | MILP 문서에 "정확해가 가능한 이유와 범위" 추가, overview 갱신. 11-15 결과·시뮬레이터 v2·도구·문서 커밋. `sim/theta.json`은 시작점 의존성(기본 재적합 시 최대 3.5% 차이) 때문에 추적 |
| 2026-11-25 15:00 | [docs-restructure-and-repro](worklog/2026-11-25_1500_docs-restructure-and-repro.md) | docs를 simulator / MILP / policies / reproduce / handoff로 분리. adaevolve·EvoX·agent_cc 최고 프로그램과 과정 기록, 재현 스크립트(`repro/`) 커밋. v1 worktree에서 기록된 점수 재현 확인 (agent_cc held-out만 wall-clock 차이) |
