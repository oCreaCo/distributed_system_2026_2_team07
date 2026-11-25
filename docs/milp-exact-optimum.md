# 시뮬레이터 목적함수의 정확한 최적값 (MILP)

작성: 2026-11-15 (21:40 5a절 추가, 11-25 재현 절차 연결). 1–5절은 시뮬레이터 v1(11-03 보정) 기준, 5b절은 v2 기준. 관련 작업 기록:
[0520 n64-exact-optimum](worklog/2026-11-15_0520_n64-exact-optimum.md),
[0941 calib2-analysis-and-optimization-status](worklog/2026-11-15_0941_calib2-analysis-and-optimization-status.md).

코드:
- `adrs/dtfm_placement/exact_milp.py`: 현재 solver (v2 기준, 레포에 포함). `python exact_milp.py N seed [seed...]`.
- `adrs/dtfm_placement/output/agent_cc2/study/exact.py`: v1 기준 원본 solver. 두 번째 Claude Code 에이전트가 작성했고,
  결과는 직접 재검산했다. `output/`은 gitignore 대상이라 레포에는 없다.
- `adrs/dtfm_placement/exact_gap.py`: N16–N128 비교.
- 결과 파일(`output/`, 레포 밖): `agent_cc2/study/exact_1000_1059.json`, `exact_gap.json`(v1), `exact_gap_v2.json`(v2).
- 재현 명령: [reproduce.md 7절](reproduce.md#7-분석-재현). 프로그램별 비교는 [policies.md](policies.md#4-정확한-최적값과의-차이).

## 왜 필요했나

LLM 진화(adaevolve, EvoX, CEGAS)와 에이전트 탐색이 모두 train 점수 약 1.27에서 멈췄다. 두 경우를 구분해야 했다.

- (a) 이 시뮬레이터에서 더 좋은 배치가 존재하지 않는다 (상한).
- (b) 존재하지만 탐색이 못 찾는다 (plateau).

N8은 8! = 40,320개를 전수 탐색할 수 있어 이미 (a)로 확인됐다. N16 이상은 전수 탐색이 불가능해서, 목적함수의 구조를
이용해 정확한 최적값을 계산했다.

## 1. 목적함수의 구조

시뮬레이터(`sim/stepsim.py`)의 한 iteration 시간:

```
T = G·(M+S−1)·[ max_s overlap(f_s, c_s, α) + max_s overlap(b_s, c_s, α) ] + overlap(T_opt0, T_dp, β)
```

- 보정된 θ에서 α ≈ 8e−12(사실상 0), `lat_fixed_ms` ≈ 0이다. `LINK_SUM = False`이므로 stage의 통신 시간 c_s는 앞뒤
  링크 중 느린 쪽이고, 그 값도 DP replica들 중 최악이다. 그래서 `max_s overlap(f_s, c_s, 0) = max(F_max, C)`이다.
  - **C** = 배치가 실제로 쓰는 파이프라인 링크(인접 stage 사이, 모든 replica) 중 가장 느린 것의 activation 전송 시간.
  - F_max, B_max = stage별 forward/backward 계산 시간의 최댓값. 배치와 무관한 상수다.
- `T_dp = max_{s, d≥1} 2·(lat + grad_bytes_s / bw)`: replica d의 stage s GPU와 그 stage의 hub(replica 0) 사이 링크 중
  가장 느린 것이다. 이 값을 **D**라 한다.
- 따라서 `T(C, D) = K·(max(F_max, C) + max(B_max, C)) + max(T_opt0, D) + β·min(T_opt0, D)`이고, K = G·(M+S−1)이다.
  - N64(PP8): K = 2·(64+7) = 142. C가 계산보다 크면 T ≈ 284·C + T_dp가 된다.
  - **T는 C와 D 각각에 대해 단조 비감소다.** 배치는 (C, D) 두 값을 통해서만 시간에 영향을 준다.

topology 구조도 문제를 줄인다 (`topologies.py`).
- GPU는 region으로 묶이고, 링크의 지연·대역폭은 region 쌍에만 의존한다. region 안 링크는 region별 상수다.
- 그래서 같은 region의 GPU는 서로 바꿔도 시간이 같다. 배치는 "DP × PP 격자의 각 칸에 region을 하나씩 배정하되,
  region r을 정확히 cnt[r]번 쓰는 문제"가 된다. region은 2–8개다.
- 링크 시간표는 region 쌍마다 미리 계산한다.
  - `pipe[a, b]` = max(a→b, b→a) activation 전송 시간 (forward와 backward 양방향을 모두 씀). 같은 region 안은
    cnt ≥ 2일 때만 정의된다.
  - `tdp[e, r, h]` = 2·(lat + grad/bw), region r의 replica와 hub region h 사이. e는 stage가 embedding을 가졌는지다
    (v1 시뮬레이터는 첫 stage와 마지막 stage에 embedding을 준다. 1–5절은 v1 기준, v2는 5b절).

## 2. 실현 가능성 검사: 0-1 정수계획

임계값 (C, D)가 주어졌을 때 "모든 파이프라인 링크 ≤ C이고 모든 gradient 링크 ≤ D인 배치가 있는가?"를 0-1 정수계획으로
검사한다 (목적함수 0, 실현 가능성만).

변수: x[d, s, r] ∈ {0, 1}, 즉 replica d, stage s 칸에 region r을 두면 1. 변수 수는 dp·pp·k이고 N64에서 최대 512개다.

제약:
1. 칸마다 region 하나: Σ_r x[d,s,r] = 1.
2. region 개수: Σ_{d,s} x[d,s,r] = cnt[r].
3. 파이프라인 임계값 (d, s<pp−1, a마다): x[d,s,a] + Σ_{b : pipe[a,b] > C} x[d,s+1,b] ≤ 1.
   - stage s에 a를 두면, 다음 stage에는 a와의 링크가 C보다 느린 region을 둘 수 없다.
   - 제약 1 때문에 오른쪽 합은 최대 1이므로, 이 한 줄이 "a이면 나쁜 b 금지"를 정확히 표현한다.
4. gradient 임계값 (s, d≥1, h마다): x[0,s,h] + Σ_{r : tdp[e_s, r, h] > D} x[d,s,r] ≤ 1.
   - hub(replica 0)에 h를 두면, 같은 stage의 다른 replica에는 h와의 gradient 링크가 D보다 느린 region을 둘 수 없다.

solver는 `scipy.optimize.milp`(HiGHS)이고, 검사 한 번의 시간 상한은 20초다. 시간 초과는 "실현 불가"로 처리한다.
그래서 시간 초과가 있으면 결과는 최적이 아니라 **상한**(최적보다 나쁠 수 있음)이 된다.

## 3. (C, D) 계단 탐색

C 후보는 `pipe`의 서로 다른 값, D 후보는 `tdp`의 서로 다른 값이다. 각각 수십 개 수준이다.

```
best = ∞;  ci = 0
for D in D 후보 (내림차순):
    while ci < len(C 후보):
        if T(C[ci], D) ≥ best: break          # 가지치기: 이 D에서는 더 나아질 수 없음
        if feasible(C[ci], D): best = min(best, T(C[ci], D)); break
        ci += 1                                # 이 D에서 C[ci]는 불가 → 더 작은 D에서도 불가
```

- D가 작아질수록 가능한 최소 C는 커지기만 한다 (단조성). 그래서 C 인덱스를 되돌릴 필요가 없다.
  MILP 호출은 (C 후보 수 + D 후보 수) 정도로 끝난다.
- 찾은 격자는 region pool에서 GPU를 꺼내 `rank_map`으로 바꾼다 (rank = d·pp + s).
- **검산**: 모델 값과 `REF.predict`(실제 시뮬레이터)의 차이가 모든 topology에서 1e−9 이하다. exact_gap 실행에서는
  −1.4e−10 ~ −4.8e−11이었다.
- topology 하나에 0.6–60초가 걸린다. 20개(N16–N128)는 10 worker로 67초 걸렸다.

## 4. 결과

**N64, 연구용 seed 1000–1059 (60개)**, 최고 프로그램(agent_cc) 대비:

| | 값 |
|---|---|
| agent_cc가 정확한 최적값과 같은 topology | 55 / 60 |
| 기하평균 차이 | 0.29% |
| 놓친 5개 | 1005 +6.1%, 1027 +5.8%, 1044 +3.0%, 1045 +2.1%, 1007 +0.7% (모두 region 7–8개) |
| MILP 시간 초과로 최적이 증명되지 않은 seed | 1005, 1027, 1037, 1054 |

**일반화 topology (seed 5000–5004) × N16/32/64/128**: 값은 프로그램 배치 시간 / 정확한 최적 시간의 기하평균이다.

| 프로그램 | N16 | N32 | N64 | N128 |
|---|---|---|---|---|
| ga_sim (DT-FM GCMA + 시뮬레이터) | 1.033 | 1.064 | 1.056 | 1.094 |
| EvoX 100 it. | 1.005 | 1.018 | 1.139 | 1.030 |
| adaevolve 100 it. | 1.000 | 1.031 | 1.034 | 1.032 |
| CEGAS 300 | 1.112 | 1.049 | 1.034 | 1.066 |
| **agent_cc** | **1.000** | **1.000** | **1.005** | **1.010** |

- agent_cc는 20개 중 15개에서 정확한 최적값이고, 최대 차이는 N128_s5003의 4.8%다.
- 시간 초과 2건(N64_s5002, N128_s5004)은 최적이 증명되지 않았다.

**놓친 원인** (agent_cc2 분석):
- 놓친 경우 대부분은 C는 같고 D만 더 작은 배치가 최적이었다 (예: 1005에서 D 12.93 → 9.97 s).
- MILP는 (C*, D*)가 실현 가능함을 증명하지만, 그 임계값에서 실현 가능한 격자는 극히 드물고 서로 떨어져 있다.
- 그래서 다음 방법들 모두 찾지 못했다:
  - 위반 수를 줄이는 SA (2만–20만 step)
  - 열 단위 SA
  - Hall 조건으로 완화한 SA
  - 기존 결과에서 warm start
- 열 단위 정확 DP/BFS는 상태가 seed 1005의 2번째 열에서 이미 90만 개라, 30초 순수 Python 안에서는 불가능했다.

## 5. 해석

- **이 시뮬레이터와 이 topology 생성기에서는 배치 탐색이 사실상 끝났다.**
  - N8은 전수 탐색 최적, N16·N32는 정확한 최적, N64는 0.3–0.5%, N128은 1.0% 차이다.
  - 탐색을 개선해 얻을 수 있는 train 점수 상승은 1% 미만이다. 1.27은 plateau가 아니라 상한에 가깝다.
- **MILP는 과제 규칙 안의 정책이 아니라 비교용 기준이다.**
  - 시뮬레이터 내부 구조와 θ를 직접 사용하는 white-box 방법이다. 과제의 프로그램은 `cost_fn` 호출 2,000회만 쓰는
    black-box이고, `config.yaml`에 따라 표준 라이브러리만 쓸 수 있어 scipy MILP를 넣을 수 없다.
  - agent_cc가 black-box로 이 최적값에 거의 도달했다는 것이 의미 있는 결과다.

## 5a. 정확해가 가능한 이유와 범위

"시뮬레이터가 있어서 정확해를 구할 수 있다"는 절반만 맞다. 필요한 것은 시뮬레이터 실행이 아니라, **식이 공개된 단순한
해석적 모델**이다. 이 MILP는 시뮬레이터 내부 식(1절)을 직접 이용한다.

- **시간이 두 값으로 정해진다.** 계산 시간은 배치와 무관한 상수이고, 통신 시간은 가장 느린 파이프라인 링크 C와 가장
  느린 gradient 링크 D만 본다. T = f(C, D)이고 C·D에 대해 단조 증가한다.
- **그래서 최적화가 실현 가능성 질문이 된다.** "모든 파이프라인 링크 ≤ C, 모든 gradient 링크 ≤ D인 배치가 있는가"는
  0-1 변수의 선형 제약으로 정확히 쓸 수 있다(2절). (C, D) 후보는 링크 값 수십 개뿐이라 모두 확인할 수 있다(3절).
- **topology 구조가 문제를 줄인다.** 링크 성능이 region 쌍에만 의존하므로, GPU N개의 순열이 region 2–8개를 격자 칸에
  개수에 맞춰 배정하는 문제가 된다.

같은 방법을 쓸 수 없는 경우:

| 대상 | 이유 |
|---|---|
| A100 실측 | 식이 없다. 배치 하나 재는 데 약 8분이라 탐색 자체가 어렵다 |
| 블랙박스 시뮬레이터 (예: 패킷 단위) | 실행은 되지만 구조를 모른다 |
| 과제의 policy | `cost_fn` 2,000회 black-box 접근과 표준 라이브러리만 허용된다 |
| DT-FM 논문 비용 모델 | 해석적이지만 파이프라인 비용이 경로를 따라 더하는 TSP 형태라 두 값 구조가 아니다 (정식화는 더 어렵지만 불가능하지는 않다) |

함의:
- **"exact"는 시뮬레이터 목적함수에 대해서만 exact다.** N8_s1에서 시뮬레이터상 최적에 가까운 배치(agent_cc)가 실측으로는
  DT-FM 배치보다 22% 느렸다 ([1633 worklog](worklog/2026-11-15_1633_a100-measured-score-agent-cc.md)).
- **시뮬레이터를 고치면 구조가 깨질 수 있다.** 다음 후보인 "느린 hop 지연 합" 항을 넣으면 T가 (C, D)만으로 정해지지
  않는다. 그때는 인접 stage 배정 쌍의 곱을 선형화한 더 큰 MILP 등이 필요하고, N64에서 풀리는지는 확인해야 한다.

## 5b. 시뮬레이터 v2 기준 결과 (2026-11-15 15:00 이후)

시뮬레이터 v2([1500 worklog](worklog/2026-11-15_1500_simulator-v2.md))는 마지막 stage에 embedding을 두지 않고, DP 링크에
대역폭 효율을 곱한다. solver는 `adrs/dtfm_placement/exact_milp.py`로 옮겨 다음을 반영했다.

- stage를 3종류(첫, 중간, 마지막)로 나누고 각 stage의 gradient 크기를 `stepsim.RunSpec.params`에서 가져온다.
- DP 링크 시간표 `tdp`에 `dp_bw_eff`를 곱한다.
- N128(PP8×DP16)을 허용한다.

모델 값과 시뮬레이터 예측의 차이는 1e−10 수준이고, N8_s0에서는 전수 탐색 최적값과 같다.

| agent_cc / 정확한 최적값 | N16 | N32 | N64 | N128 |
|---|---|---|---|---|
| v1 | 1.000 | 1.000 | 1.005 | 1.010 |
| v2 | 1.002 | 1.011 | **1.026** | 1.001 |

- v2에서는 N64에 약 2.6%, N32에 약 1.1%의 탐색 여지가 생겼다. agent_cc는 v1 구조에 맞춰져 있다.
- 따라서 "탐색은 끝났다"는 결론은 v1 목적함수에 대한 것이다.

## 6. 한계와 가정

- 성립 조건: α ≈ 0, `lat_fixed_ms` ≈ 0, `LINK_SUM = False`, region 쌍마다 링크가 같은 topology.
  시뮬레이터가 바뀌면 다음처럼 해야 한다.
  - 마지막 stage 파라미터 수정과 DP 효율(v2): 반영했다 (`exact_milp.py`, 5b절). 구조는 그대로다.
  - α > 0, 링크 합산, 느린 hop 지연 합(1633 worklog의 후보) 등: (C, D) 두 값으로 줄어들지 않으므로 정식화를 다시 해야 한다.
- 시간 초과는 실현 불가로 처리하므로, 그 topology의 "최적값"은 상한이다.
- **시뮬레이터상의 최적일 뿐이다.** A100 실측과의 차이는 시뮬레이터 오차만큼 날 수 있다.
  - PP4×DP2 오차는 −16~+21%, PP8은 최대 −47%다 (0858, 0941 worklog).
  - 실측 기준의 최적성은 이 문서가 말하지 않는다.
