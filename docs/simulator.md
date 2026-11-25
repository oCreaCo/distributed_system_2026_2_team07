# 시뮬레이터 현재 상태 (v2)

최종 갱신: 2026-11-25. 코드 `sim/stepsim.py`, 파라미터 `sim/theta.json`, 보정 `sim/fit.py`.
변경 이력은 [1500 worklog](worklog/2026-11-15_1500_simulator-v2.md), v1 분석은 [0941](worklog/2026-11-15_0941_calib2-analysis-and-optimization-status.md).

## 1. 무엇을 예측하나

DT-FM 학습 설정 하나(GPipe + central_ps DP, GPT-3 XL 24 layer, QQP, CPU offload optimizer)에서, 배치(`rank_map`)와
네트워크 행렬(`delay_ms`, `bandwidth_gbps`)을 받아 **iteration당 시간(초)**을 예측한다.

```
c_s    = lat + 8 MB / bw_eff(s → s+1)                       파이프라인 activation 전송
T_pipe = G·(M+S−1)·[max_s overlap(f_s, c_s, α) + max_s overlap(b_s, c_s, α)]
T_dp   = max_s 2·(lat + grad_bytes_s / (dp_bw_eff · bw_eff(member → stage hub)))
T_iter = T_pipe + overlap(opt_per_gparam · params_max, T_dp, β)
bw_eff = min(bw, tcp_window / RTT),   overlap(x, y, k) = max(x, y) + k·min(x, y)
```

- stage hub는 그 stage의 replica 0 GPU다 (central_ps).
- stage별 파라미터 수는 로그에서 잰 값이다. 모든 stage에 layer당 50.36M, stage 0에만 embedding 63.58M,
  마지막 stage에만 분류 head 4.20M이 더해진다.
- 파이프라인 시간은 **가장 느린 링크 하나**(C)로, DP 시간은 **가장 느린 gradient 링크 하나**(D)로 정해진다.
  이 구조 덕분에 정확한 최적값을 MILP로 구할 수 있다 ([MILP 문서](milp-exact-optimum.md)).

## 2. 파라미터 (v2, `sim/theta.json`)

| 이름 | 값 | 뜻 |
|---|---|---|
| `f_layer`, `b_layer` | 0.00424, 0.01122 s | layer당 micro-batch forward / backward 계산 |
| `f_emb`, `b_emb` | 0.01095, 0.01942 s | embedding stage의 추가 계산 |
| `alpha` | ≈ 0 | 파이프라인 전송과 계산의 비중첩 비율 (사실상 완전 중첩) |
| `beta` | 0.128 | DP 전송과 optimizer의 비중첩 비율 |
| `opt_per_gparam` | 13.48 s | 파라미터 10억 개당 CPU optimizer 시간 |
| `lat_fixed_ms` | ≈ 0 | 링크당 고정 지연 |
| `tcp_window_mb` | 10.1 MB | 실효 대역폭 상한 `window / RTT` |
| `dp_bw_eff` | 0.940 | DP gradient 전송에만 곱하는 대역폭 효율 (v2에서 추가) |

- `sim/theta.json`은 레포에 커밋되어 있다. 최소제곱 적합이 시작점에 따라 달라지기 때문이다.
  - 재현: v1 theta(`adrs/dtfm_placement/output/sim_v1/theta.json`)에 `dp_bw_eff: 0.9`를 더한 파일을 `--init`으로 주면
    차이 0으로 같은 값이 나온다.
  - 기본 시작점으로 돌리면 `beta` 0.124, `opt_per_gparam` 13.64처럼 최대 3.5% 다른 값이 나온다.
  - 절차는 [reproduce.md](reproduce.md#1-시뮬레이터-보정).

## 3. 보정 데이터

- 11-03 A100 74 run: topology T1–T6, notc, homo. 대부분 PP4×DP2이고, PP8·PP2도 일부 있다.
- 11-15 A100 46 run (`sim/fit.py`의 `INLINE_CONFIGS`):
  - N8 held-out topology s100–s102, PP4×DP2 22 run (검증·순위 확인용으로 잰 것)
  - calib2 24 run: 새 topology c2s305, c2s323 × PP8×DP1 / PP4×DP2 / PP2×DP4 × 배치 4종
- 11-15 agent_cc 실측 17 run은 보정에 넣지 않았다. 다음 버전 검증에 쓸 데이터다.

## 4. 정확도 (120 run, leave-one-topology-out)

`sim/compare_versions.py` → `sim/loto_v1_v2.json`.

LOTO(leave-one-topology-out)는 topology 하나를 빼고 나머지 run으로 보정한 모델로, 뺀 topology의 run을 예측하는 검증이다.
topology마다 반복한다. 보정 데이터에 들어간 topology를 맞히는 것은 쉬우므로, 처음 보는 네트워크에서의 정확도를 보려고 이렇게 잰다.
순위 지표는 같은 topology·같은 병렬 구성 안에서 계산했고, 측정 차이가 1% 미만인 쌍은 뺐다.

| 지표 | v1 | **v2** |
|---|---|---|
| MAPE 전체 | 10.9% | 10.8% |
| MAPE PP8 | 26.0% | 26.0% |
| 순서 맞춘 쌍 | 388/438 | 391/438 |
| 그룹 평균 Spearman | 0.839 | 0.854 |
| top-1 손해 평균 / 최대 | 1.19% / 6.66% | 1.19% / 6.66% |

참고로 v1을 처음 보정했을 때(11-03, 74 run) DT-FM 논문의 통신 비용 모델과 LOTO로 비교한 결과는 다음과 같다.

| | 시뮬레이터 v1 | 논문 비용 모델 |
|---|---|---|
| MAPE | 3.2–14.6% | 13.7–40% |
| top-1 손해 | 모든 topology에서 0 s | 최대 13 s |

보정에 쓰지 않은 실측(agent_cc vs DT-FM, N8 10 topology, 20 run)과 비교한 결과:

| | v1 | v2 |
|---|---|---|
| 시간 MAPE | 12.5% | 11.9% |
| DT-FM 대비 비율 오차 (절대값 평균) | 8.3% | 7.1% |
| 이득 방향 일치 | 8/10 | 8/10 |
| 기하평균 비율 (실측 1.139) | 1.212 | 1.175 |

## 5. 알려진 오차와 원인

| 문제 | 크기 | 상태 |
|---|---|---|
| **느린 파이프라인 hop의 누적** | 느린 hop이 하나 늘 때마다 micro-batch당 약 그 링크의 단방향 지연만큼 느려진다. agent_cc가 N8_s1에서 실측 22% 손해를 본 원인이다 | 미반영. 시험 모델 λ·(hop 지연 합), λ = 0.081: 순서 쌍 387 → 394/438. 정식 검증 전 ([1633](worklog/2026-11-15_1633_a100-measured-score-agent-cc.md)) |
| **PP8 과소 추정** | 12–47%, LOTO MAPE 26% | 원인 미확인. 파이프라인 단계에서 생긴다. 한 TCP window로 PP8과 PP4를 함께 맞출 수 없다. hop 누적이 유력 후보다 |
| 파이프라인 방향 효과 | 같은 배치의 역순이 최대 7.6% 다르다. 시뮬레이터는 0 | 미반영 |
| 링크 대역폭 과대 가정 | NCCL 실측은 tc 설정의 50–88% (지연이 클수록 낮음) | DP 링크에만 0.94 반영. 모든 링크에 적용하면 PP4×DP2를 과대 추정한다 |
| 외삽 | 보정은 N = 8뿐인데 N = 64(PP8×DP8)까지 채점한다 | central_ps 서버의 동시 수신 경합은 모델에 없다 (검증 불가) |

- 이 오차들은 N32·N64 채점(PP8)에 직접 영향을 준다. 실측 확인은 GPU 8장 제약 때문에 N8에서만 가능하다.
- 수치를 인용할 때는 LOTO 값을 쓴다. `fit.py`의 "fit on all"은 검증용 topology까지 보정에 넣은 값이다.
  `stepsim.fit()`과 `fit_two_stage` 안의 `residuals`는 쓰이지 않는 코드다.
- 시도했지만 채택하지 않은 수정(파이프라인 전용 window, 모든 링크 효율, 링크 합산, 메시지당 RTT)은 [1500 worklog](worklog/2026-11-15_1500_simulator-v2.md)에 있다.

## 6. 버전

| 버전 | 위치 | 차이 |
|---|---|---|
| v1 (11-03 보정) | git `095eef5`·`e07b345`의 `sim/stepsim.py` + `output/sim_v1/theta.json` | 첫 stage와 마지막 stage 모두에 embedding 파라미터를 더한다. `dp_bw_eff` 없음 |
| **v2 (11-15, 현재)** | `sim/stepsim.py`, `sim/theta.json` | 위 오류 수정 + `dp_bw_eff`, 120 run으로 재보정 |

- **모든 진화·에이전트 탐색(adaevolve, EvoX, CEGAS, agent_cc)은 v1에서 이뤄졌다.**
- v2로 다시 채점한 값은 [policies.md](policies.md)에 있다.
- v1 환경을 다시 만드는 방법은 [reproduce.md](reproduce.md#3-v1-환경-진화-당시-재현)에 있다.
