# [2026-11-15 01:33] CEGAS: suite 순서 반전 (N64 → N32 → N16)

> **정정 (2026-11-15 03:00):** 같은 원래 순서의 두 번째 표본(300 요청 실행의 처음 100 요청)이 1.270에 도달했다. 아래의 "순서 반전으로 점수가 올랐다"는 해석은 실행 간 편차로 설명되며, 순서 효과는 확인되지 않았다. `2026-11-15_0300_cegas-300-requests-and-order-correction.md` 참고.

## 왜
trace 분석(`2026-11-14_2325`)에서 checker가 **실패한 테스트 중 목록상 첫 번째**만 반례로 제시하고(`tcopt/common/autopanel/checker.py:106,210`,
`self.ids = list(self.inputs)` → `next(i for i in self.ids if i in ids)`), 목록이 N16부터라 격차가 가장 큰 N64가 한 번도 제시되지 않았다.
TCResearch 코드를 고치지 않고, 테스트 목록 순서만 뒤집어 이 효과를 시험했다.

## 설정 (순서 외에는 동일)
```bash
skydiscover/.venv/bin/python adrs/dtfm_placement/cegas/build_panel.py --out adrs/dtfm_placement/output/cegas_panel_rev --reverse
PYTHONPATH=$TCOPT skydiscover/.venv/bin/python adrs/dtfm_placement/cegas/prepare.py \
    --reference adrs/dtfm_placement/output/main_adaevolve_100/best/best_program.py \
    --panel adrs/dtfm_placement/output/cegas_panel_rev --runs 'output/pilot_*' 'output/main_adaevolve_100' \
    --config-out adrs/dtfm_placement/cegas/config_rev.yaml
CFG_SRC=adrs/dtfm_placement/cegas/config_rev.yaml adrs/dtfm_placement/cegas/run.sh rev_r100_s0 100 100
```
- 코드 변경: `build_panel.py --reverse`, `run.sh`가 `CFG_SRC`로 config 경로를 받음.
- 검증: 통과. V2 Spearman 0.679 (원래 0.671, 같은 52개 저장 프로그램; EvoX 결과는 넣지 않음), tol 0.005, V4 6/8.
- `config.yaml`과 `config_rev.yaml`의 차이는 panel 경로와 hash 두 개뿐(`diff`로 확인).
- 2026-11-15 01:03 시작, 1,803 s. 01:26부터 CEGAS 300회 실행과 동시에 돌았다(12코어, 실패/TLE 0).

## 결과

| | 원래 순서 (r100_s0) | **반전 (rev_r100_s0)** |
|---|---|---|
| LLM 요청 / 공식 평가 | 100 / 24 | 100 / **62** |
| 최고 train / vs_ga_sim | 1.206 / 0.991 | **1.247 / 1.025** |
| held-out / vs_ga_sim | 1.221 / 0.972 | **1.286 / 1.024** |
| N16 / N32 / N64 speedup (train) | 1.056 / 1.188 / 1.263 | 1.126 / 1.249 / 1.289 |
| 비용 (requests.jsonl 집계) | $0.298 | $0.491 |
| 토큰 prompt / completion | 2.65M / 183k | 2.89M / 257k |
| DSML 파싱 실패 | 9 | 2 |
| run_program (로컬 실험) | 40 | 0 |
| 제시된 반례 | N16_s200 → N16_s201 → N16_s200 → N32_s203 | N64_s207 ↔ N64_s206 (끝까지 교대) |
| 마지막 실패 테스트 | N32_s203, N32_s205, N64_s206, N64_s207 | N64_s206, N32_s203, N16_s201 |

점수 궤적: 1.048 → 1.100 (#1) → 1.171 (#2) → 1.207 (#7) → 1.227 (#9) → 1.233 (#23) → 1.238 (#34) → 1.244 (#52) → 1.247 (#53).

### 다른 방법과 비교
| 방법 | train | held-out | 비용 |
|---|---|---|---|
| ga_sim seed 0 | 1.217 | 1.256 | – |
| CEGAS 원래 순서, 100 요청 | 1.206 | 1.221 | $0.30 |
| **CEGAS 반전, 100 요청** | 1.247 | 1.286 | $0.49 |
| adaevolve 10 it. (비슷한 비용) | 1.241 | – | 약 $0.27 |
| adaevolve 100 it. | 1.270 | 1.288 | $2.68 |
| EvoX 100 it. | 1.267 | 1.307 | $2.72 |

## 해석
- 반례가 N64로 바뀌자 에이전트는 N64를 직접 겨냥했고(N64_s207은 통과, N64_s206은 끝까지 실패), 점수가 크게 올랐다.
  held-out은 adaevolve 100 it.와 비슷하고(1.286 vs 1.288) 비용은 약 1/5이다. train은 아직 2–3% 낮다.
- **행동 자체도 달라졌다**: 로컬 실험(run_program) 40 → 0, 평가 24 → 62. 원래 실행의 "N16 하나에 40회 실험"이 없어졌다.
  다만 이것이 순서 때문인지 실행 간 무작위성 때문인지는 실행 1회씩으로는 구별할 수 없다.
  → 원래 순서로 돌고 있는 300 요청 실행의 처음 100 요청이 같은 설정의 두 번째 표본이 된다.
- 반례가 N64_s207과 N64_s206 사이를 교대했다: 한쪽을 고치면 다른 쪽이 회귀하는 trade-off가 있다(마지막에 s207 통과, s206 실패).
- 주의: 실행 1회. 반전 순서가 이 과제에 유리했던 이유는 "공식 점수에 크게 기여하는 큰 입력의 격차"가 먼저 보였기 때문이라고 본다.
  방법 수준의 일반화(예: 격차가 가장 큰 실패를 제시)는 TCResearch 코드 수정이 필요해 하지 않았다.

예산: 누적 약 $8.1 / $50 (300 요청 실행 진행 중, 미포함).
