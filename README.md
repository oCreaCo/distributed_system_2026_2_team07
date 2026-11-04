# DT-FM 재현과 ADRS 기반 GPU 배치 탐색 (team07)

[DT-FM](https://arxiv.org/abs/2206.01288)은 이종 네트워크에서 GPU를 DP 그룹과 PP stage에 배치하는 알고리즘(Hybrid GA + Local Search)을 제안합니다. 이 프로젝트는 DT-FM을 SGS A100 클러스터에서 재현한 뒤, ADRS(LLM 기반 프로그램 진화)로 **DT-FM의 배치 알고리즘보다 더 적합한 알고리즘이 있는지 탐색**합니다. 후보 알고리즘은 실측으로 보정한 시뮬레이터로 빠르게 채점하고, 좋은 후보만 A100에서 다시 검증합니다.

자세한 배경과 계획은 `project_description.pdf`, `project_team07.pdf`를 참고하세요.

---

## 폴더 구성

```
.
├── DT-FM/            원 논문 코드 (학습 runtime + scheduler). CuPy 14 호환 패치만 적용
├── dev.yaml          조교 제공: GPU 1장짜리 interactive pod
├── experiment.yaml   조교 제공: 8 pod 실험 Job (각 pod가 /workspace/test.py 실행)
├── k8s/              A100 클러스터에서 실험을 돌리고 결과를 모으는 도구
├── sim/              실측으로 보정한 step-time 시뮬레이터
├── adrs/             SkyDiscover에 넘기는 배치 알고리즘 탐색 과제 (시뮬레이터로 채점)
├── skydiscover/      SkyDiscover 0.2.0 원본 복사본 (LLM 기반 프로그램 진화 프레임워크)
└── A100_results/     A100에서 측정한 로그와 요약 (2026-11-03, 학습 74 run + NCCL bench)
```

### `k8s/` — 클러스터 실험 도구

| 파일 | 실행 위치 | 역할 |
|---|---|---|
| `test.py` | **pod 안** | `experiment.yaml`이 실행하는 런처. pod 번호→rank 매핑, pod 간 tc 설정, `dist_runner.py` 실행, 여러 run 연속 실행 |
| `nccl_bench.py` | **pod 안** | pod 쌍마다 실제 latency/bandwidth 측정 |
| `sync.sh` | 로컬 | 코드·설정을 `/workspace`에 올리고 실험 Job 시작 |
| `run_batches.sh` | 로컬 | 여러 배치를 차례로 실행하고 끝날 때마다 로그 수집 |
| `collect_results.py` | 로컬 | pod 로그를 run별로 나누고 `A100_results/summary.json` 생성 |
| `run_scheduler.py` | 로컬 | 원 DT-FM scheduler(GA)를 N=8 등 원하는 크기로 호출 |
| `make_calib_configs.py`, `make_heldout_configs.py` | 로컬 | 보정용·검증용 run 묶음 생성 |
| `run_config.*.json` | – | 단일 run 설정 예시 (`smoke`, `case3_*` 등) |
| `calib/` | – | 실제로 실행한 배치 설정(`job1~7.json`)과 전체 run 목록(`manifest.json`) |

### `sim/` — step-time 시뮬레이터

| 파일 | 역할 |
|---|---|
| `stepsim.py` | 시뮬레이터 모델(라이브러리, 직접 실행하지 않음). 배치와 네트워크 행렬을 받아 iteration 시간 예측 |
| `fit.py` | 실행 스크립트. `A100_results/summary.json`에 맞춰 파라미터를 적합하고 `sim/theta.json` 저장 |


```bash
python3 sim/fit.py                                                              # -> sim/theta.json (전체 74 run)
python3 sim/fit.py --exclude '^T[456]_|simbest' --out sim/theta_preheldout.json  # held-out 검증에 쓴 보정값
```

시뮬레이터 사용 예:

```python
import json, sys
sys.path.insert(0, "sim")
import stepsim as SS

theta = list(json.load(open("sim/theta.json")).values())
spec = SS.RunSpec(delay_ms, bandwidth_gbps, rank_map, pp=4, dp=2)   # 8x8 행렬, 길이 8 rank_map
print(SS.predict(theta, spec))                                      # 예측 iteration 시간(초)
```

### `adrs/` — 배치 알고리즘 탐색 과제

`adrs/dtfm_placement/`는 SkyDiscover가 읽는 과제 정의입니다. LLM이 `schedule()`을 고쳐 나가고, evaluator가 이를 시뮬레이터로 채점합니다. GPU와 tc는 쓰지 않고 전부 CPU에서 돌아갑니다.

| 파일 | 역할 |
|---|---|
| `initial_program.py` | 진화 대상 `schedule(delay_ms, bandwidth_gbps, pp, dp, cost_fn, budget) -> rank_map`. `EVOLVE-BLOCK` 안쪽만 LLM이 수정. seed는 DT-FM 스타일의 작은 GA |
| `evaluator.py` | SkyDiscover가 호출하는 `evaluate()`. 후보를 실행하고 시뮬레이터로 채점 |
| `_run_candidate.py` | evaluator가 후보를 별도 프로세스(topology마다 1개)에서 실행할 때 쓰는 보조 스크립트 |
| `topologies.py` | 가상 네트워크 생성. 지역 단위로 묶고, 지역 간 링크는 DT-FM의 실제 도시 간 지연·대역폭 사용. N별 학습용 3개, held-out 3개 |
| `reference.py` | 비교 기준: 원 DT-FM GA, 같은 예산의 random search, N=8 전수 탐색 최적 → `reference.json` 캐시 |
| `config.yaml` | SkyDiscover 설정. LLM 모델, 탐색 방식, LLM에게 보여 줄 문제 설명 |
| `make_a100_config.py` | 찾은 알고리즘의 N=8 배치를 DT-FM 배치와 나란히 A100에서 돌리는 `run_config` 생성 |

채점 방식
- N = 8, 16, 32, 64에서 학습 topology 3개씩, 총 12개를 채점합니다. N별 PP×DP는 `2: 2×1, 4: 2×2, 8: 4×2, 16: 4×4, 32: 8×4, 64: 8×8`입니다.
- **`combined_score` = topology별 `DT-FM GA 배치의 예측 시간 / 후보 배치의 예측 시간`의 기하평균**입니다. 1보다 크면 DT-FM보다 빠릅니다.
- N = 2, 4는 배치 경우의 수가 24개 이하라 점수에 넣지 않고, 올바른 rank_map을 내는지만 검사합니다. 실패하면 점수가 크게 깎입니다.
- 후보는 시뮬레이터를 `cost_fn`으로만 쓸 수 있고, topology당 **2,000회**, **30초**로 제한됩니다. 시뮬레이터 모듈을 직접 import하면 0점입니다.
- `reference.json`과 SkyDiscover 출력(`output/`)은 다시 만들 수 있으므로 `.gitignore` 대상입니다.

### `skydiscover/` — SkyDiscover 프레임워크

[SkyDiscover](https://github.com/skydiscover-ai/skydiscover)(UC Berkeley, Apache-2.0) 0.2.0을 `.git` 없이 그대로 복사해 둔 것입니다. 우리 과제 코드는 모두 `adrs/`에 있고, skydiscover 폴더는 수정하지 않았습니다. 탐색 알고리즘은 `-s` 옵션으로 고릅니다(`adaevolve`, `evox`, `topk`, `beam_search`, `best_of_n`, `openevolve`, `gepa` 등). 자세한 사용법은 `skydiscover/skydiscover/optimize/README.md`에 있습니다.

---

## 로컬 준비

- `kubectl`, `python3` (numpy, scipy)
- `team07.kubeconfig`: **저장소 루트**에 둡니다. 스크립트들이 이 경로를 씁니다.

```bash
# 터미널 1: Kubernetes API로 가는 SSH tunnel (비밀번호는 ETL 공지). 출력이 없는 게 정상
ssh -N -L 16443:127.0.0.1:6443 -p 53036 k8s-tunnel@147.47.206.149

# 터미널 2
export KUBECONFIG=$PWD/team07.kubeconfig
kubectl auth can-i create jobs     # yes 이면 사용 가능
```

A100은 할당 시간에만 쓸 수 있습니다. 시간이 끝나면 Job/Pod는 지워지지만 `/workspace`의 파일은 남습니다.

---

## A100에서 DT-FM 실행하기

### 동작 구조

```
로컬                                   클러스터 (/workspace = 팀 공용 PVC)
k8s/sync.sh <config.json>
  1. dev pod 생성 ───────────────────▶ /workspace/results/  ──▶ 로컬 A100_results/ 로 복사
  2. DT-FM/, test.py, nccl_bench.py ─▶ /workspace/DT-FM/, /workspace/test.py, ...
     config.json ────────────────────▶ /workspace/run_config.json
  3. dev pod 삭제 (GPU 1장 반환)
  4. kubectl apply -f experiment.yaml ▶ pod 8개가 각자 /workspace/test.py 실행
                                         ├ run_config.json 읽기
                                         ├ rank 결정, tc 설정
                                         └ dist_runner.py → /workspace/results/<tag>/rank{r}_pod{i}.log
```

`dev.yaml`과 `experiment.yaml`은 조교가 준 그대로 씁니다. 실험마다 바뀌는 값은 모두 `run_config.json`에 넣습니다.

### 1. `/workspace` 초기 설정 (한 번만, 이미 되어 있음)

2026-11-03에 아래 상태로 만들어 두었습니다. `/workspace`는 유지되므로 보통 다시 할 필요가 없습니다. 비어 있을 때만 dev pod 안에서 실행하세요.

```bash
kubectl apply -f dev.yaml
kubectl exec -it $(kubectl get pods -l batch.kubernetes.io/job-name=dev -o name | cut -d/ -f2) -- bash

# --- 이하 dev pod 안 ---
# 1) venv + CuPy (torch는 이미지의 것을 그대로 사용)
python -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/pip install cupy-cuda12x

# 2) tc 바이너리: 이미지에 없어서 .deb를 받아 두고 실험 pod가 시작할 때 설치
apt-get update && apt-get install -y --download-only iproute2
mkdir -p /workspace/debs && cp /var/cache/apt/archives/*.deb /workspace/debs/

# 3) QQP 데이터와 vocab (DT-FM이 기본으로 찾는 경로)
mkdir -p /workspace/DT-FM/task_datasets/data && cd /workspace/DT-FM/task_datasets/data
python -c "import urllib.request as u; u.urlretrieve('https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip','qqp.zip')"
python -c "import zipfile; zipfile.ZipFile('qqp.zip').extractall('.')"        # -> QQP/train.tsv ...
python -c "import urllib.request as u; u.urlretrieve('https://huggingface.co/bert-large-cased/resolve/main/vocab.txt','bert-large-cased-vocab.txt')"
head -n 2001 QQP/train.tsv > QQP/train_small.tsv                              # 빠른 시작용 2000줄
exit
```

```bash
kubectl delete job dev     # 끝나면 반드시 삭제 (GPU 1장을 차지함)
```

데이터는 저장소에 없고 `/workspace`에만 있습니다. `sync.sh`는 `DT-FM/`을 덮어쓰기만 하고 지우지 않으므로 데이터는 그대로 유지됩니다.

### 2. smoke test

```bash
k8s/sync.sh k8s/run_config.smoke.json      # tc 없음, layer 1개, 3 iteration
kubectl get pods                           # experiment-0 ~ 7, 8개가 Running인지 확인
kubectl logs -f <experiment-0-xxxxx>       # pod 이름은 get pods로 확인
```

`Finished running 3 iterations, averaged (exclude the first iter) run time: ...`이 나오고 모든 pod가 `exited with 0`이면 성공입니다. 결과 수집:

```bash
python3 k8s/collect_results.py fetch smoke     # kubectl logs -> A100_results/jobs/smoke/
python3 k8s/collect_results.py split smoke     # run별로 나눔 -> A100_results/<tag>/
python3 k8s/collect_results.py summarize       # A100_results/summary.json 갱신 + 표 출력
```

### 3. 실험 설정 (`run_config.json`)

**단일 run**

```json
{
  "tag": "case3_identity_rep0",
  "rank_map": [0, 1, 2, 3, 4, 5, 6, 7],
  "delay_ms":       [[...8개...], ...8행...],
  "bandwidth_gbps": [[...8개...], ...8행...],
  "args": ["--pp-mode", "gpipe", "--dp-mode", "central_ps",
           "--pipeline-group-size", "4", "--data-group-size", "2", "--num-layers", "6", "..."]
}
```

**여러 run 연속 실행** (pod를 다시 만들지 않아 run당 시간이 짧음)

```json
{
  "common_args": ["--pp-mode", "gpipe", "..."],
  "run_timeout_s": 1200,
  "runs": [
    {"tag": "bench_T1", "type": "bench", "rank_map": null, "delay_ms": [...], "bandwidth_gbps": [...], "args": []},
    {"tag": "T1_identity", "rank_map": null, "delay_ms": [...], "bandwidth_gbps": [...], "args": []}
  ]
}
```

| 키 | 의미 |
|---|---|
| `tag` | 결과 폴더 이름. 겹치면 덮어쓰므로 run마다 다르게 |
| `rank_map` | `rank_map[pod 번호] = DT-FM rank`. `null`이면 identity. stage = `rank % PP`, DP 그룹 = `rank // PP` |
| `delay_ms`, `bandwidth_gbps` | pod 번호 기준 8×8 행렬. `delay_ms[i][j]`는 pod i → j 방향에 거는 지연. `null`이면 tc 없음 |
| `args` | `dist_runner.py` 인자. 배치 설정에서는 `common_args` 뒤에 붙고, 같은 인자는 뒤의 값이 이김 |
| `type: "bench"` | 학습 대신 `nccl_bench.py`로 pod 쌍별 latency/bandwidth 측정 → `bench.json` |

자주 쓰는 `dist_runner.py` 인자: `--pipeline-group-size`(PP), `--data-group-size`(DP), `--num-layers`(**stage당** layer 수, 24층이면 PP4에서 6), `--batch-size`, `--micro-batch-size`, `--gradient-accumulate-step`, `--num-iters`, `--use-offload false`, `--train-data ./task_datasets/data/QQP/train_small.tsv`(시작이 빠름).

예시는 `k8s/run_config.*.json`과 `k8s/calib/job*.json`을 참고하세요.

### 4. 원 DT-FM scheduler로 배치 구하기

```bash
python3 k8s/run_scheduler.py --config k8s/run_config.case3.json --pp 4 --dp 2 \
    --grad-gb 0.6 --act-gb 1.0 --brute-force --random 1000 \
    --out k8s/run_config.case3_sched.json      # GA가 찾은 rank_map을 넣은 설정 생성
```

원 `scheduler.py`는 고치지 않고 64 GPU용 전역값만 덮어써서 호출합니다. N=8에서는 `--brute-force`로 전수 탐색 최적값과 비교할 수 있습니다.

### 5. 여러 배치 자동 실행

```bash
# job_a가 끝나면 로그를 모으고 job_b, job_c를 차례로 실행
nohup k8s/run_batches.sh none job_a:path/a.json job_b:path/b.json job_c:path/c.json &
tail -f A100_results/run_batches.log
```

첫 인자는 "지금 돌고 있는 Job의 이름"입니다(없으면 `none`). 각 배치가 끝날 때마다 `fetch` → `split` → `summarize`까지 해 줍니다.

---

## 시뮬레이터로 SkyDiscover 실행하기

A100 없이 이 서버(CPU)에서 돌립니다. 흐름은 다음과 같습니다.

```
A100_results/summary.json ──sim/fit.py──▶ sim/theta.json ──▶ adrs/dtfm_placement/evaluator.py
                                                                   ▲ 후보 채점        │ 점수
                                   LLM이 schedule() 수정 ◀── SkyDiscover ◀────────────┘
```

### 1. 준비 (한 번만)

```bash
cd skydiscover
uv sync                                            # skydiscover/.venv 생성
uv pip install --python .venv/bin/python scipy     # 시뮬레이터(sim/stepsim.py)가 scipy를 씀
cd ..

python3 sim/fit.py                                 # sim/theta.json
python3 adrs/dtfm_placement/reference.py           # 기준값 캐시, 약 7분 (생략하면 첫 평가 때 자동으로 만듦)
```

`reference.py`는 원 DT-FM GA를 N=64까지 돌리기 때문에 오래 걸립니다. topology 24개를 프로세스 24개로 병렬 실행합니다.

### 2. seed 프로그램 채점으로 동작 확인

```bash
skydiscover/.venv/bin/python adrs/dtfm_placement/evaluator.py adrs/dtfm_placement/initial_program.py
```

3초 정도 걸리고, topology별 예측 시간과 아래 지표가 출력됩니다. seed의 `combined_score`는 약 1.05입니다.

| 지표 | 의미 |
|---|---|
| `combined_score` | SkyDiscover가 최대화하는 값. DT-FM GA 대비 속도 향상의 기하평균 |
| `speedup_vs_dtfm_N{8,16,32,64}` | N별 DT-FM 대비 속도 향상 |
| `vs_random` | 같은 예산(2,000회)의 random search 대비. 목적함수 효과와 탐색 효과를 구분하는 용도 |
| `mean_regret_vs_optimum_N8` | N=8 전수 탐색 최적값 대비 손해 비율 (0이면 최적) |
| `failures`, `mean_time_s` | 실패한 topology 수, topology당 평균 실행 시간 |

N=8은 random search로도 최적값에 도달할 만큼 쉬워서, 알고리즘 간 차이는 주로 N ≥ 16에서 납니다.

### 3. LLM 설정

SkyDiscover는 OpenAI 호환 API를 씁니다. `config.yaml`의 기본값은 OpenAI `gpt-5`입니다.

```bash
export OPENAI_API_KEY=...         # 또는 ANTHROPIC_API_KEY / GEMINI_API_KEY
```

다른 모델은 실행할 때 `--model`(예: `anthropic/claude-sonnet-4-6`)로 바꾸고, 로컬 서버 같은 OpenAI 호환 주소는 `--api-base`로 지정합니다. 형식은 `skydiscover/skydiscover/optimize/README.md`의 모델 설정 부분을 참고하세요.

### 4. 실행

```bash
cd skydiscover
uv run skydiscover optimize ../adrs/dtfm_placement/initial_program.py ../adrs/dtfm_placement/evaluator.py \
    -c ../adrs/dtfm_placement/config.yaml -s adaevolve -i 100 -o ../adrs/dtfm_placement/output
```

- `-s`: 탐색 알고리즘. `adaevolve`부터 시작하고, 비교용으로 `evox`, `openevolve` 등을 같은 `-i`로 돌립니다.
- `-i`: iteration 수. 한 iteration = LLM이 후보 1개 생성 + 채점(약 3초). 실제 시간은 대부분 LLM 응답 시간입니다.
- 결과: `output/best/best_program.py`(가장 좋은 알고리즘), `output/best/best_program_info.json`(점수), `output/logs/`(실행 로그). checkpoint는 `checkpoint_interval`(5 iteration)마다 저장되며 `uv run skydiscover viewer <checkpoint 폴더>`로 다시 볼 수 있습니다.
- 병렬 실행: `config.yaml`에 `max_parallel_iterations: 4`처럼 쓰면 여러 iteration이 동시에 돌아 LLM 대기 시간이 겹칩니다. 다만 `topk`, `beam_search`, `best_of_n`에서만 적용되고, **`adaevolve`는 순차 실행이라 이 값을 무시합니다.**

### 5. held-out topology로 확인

진화에 쓰지 않은 topology(seed 100~102)에서 다시 채점합니다. 학습 topology에만 맞춘 알고리즘인지 확인하는 단계입니다.

```bash
skydiscover/.venv/bin/python adrs/dtfm_placement/evaluator.py adrs/dtfm_placement/output/best/best_program.py heldout
```

### 6. 찾은 알고리즘을 A100에서 검증 (N = 8)

GPU quota가 8장이라 실측은 N ≤ 8에서만 가능합니다. `make_a100_config.py`가 N=8 held-out topology 3개에 대해 **DT-FM GA 배치**와 **찾은 알고리즘의 배치**를 같은 tc 설정으로 돌리는 배치 설정을 만듭니다. 학습 설정(`common_args`)은 보정 실험과 같습니다.

```bash
skydiscover/.venv/bin/python adrs/dtfm_placement/make_a100_config.py
# 기본값: 입력 output/best/best_program.py, 출력 output/a100_N8.json, held-out topology
# 다른 프로그램이나 학습 topology: make_a100_config.py <program.py> --split train --out <path>
```

배치별 rank_map과 예측 iteration 시간이 출력됩니다. 그다음은 [A100에서 DT-FM 실행하기](#a100에서-dt-fm-실행하기)와 같습니다.

```bash
k8s/sync.sh adrs/dtfm_placement/output/a100_N8.json
```

- 이 topology들은 예측 iteration 시간이 50~105초라 run 하나에 10분 정도 걸립니다. 6개 run이 Job 제한(2시간) 안에 들어가지만 여유가 크지 않으니, 필요하면 `common_args`의 `--num-iters`를 줄이세요.
- 같은 배치를 2~3번 반복하면 run 간 편차(±1초)와 구분할 수 있습니다.

---

## 주의사항

- **GPU quota는 팀당 8장**입니다. dev pod가 떠 있으면 실험 pod가 7개만 생기고, 이 7개는 8번째 pod를 10분 동안 기다리다 실패합니다. `sync.sh`는 dev를 지운 뒤 실험을 띄웁니다. 지운 pod가 `Terminating` 상태로 몇 분 동안 quota를 차지할 수 있습니다.
- **Job 하나는 최대 2시간**(`activeDeadlineSeconds: 7200`)입니다. 24층 기준 run 하나에 3~6분, 느린 topology는 10분 이상 걸리니 배치 크기를 맞추세요.
- **Job이 Failed로 끝나면 `kubectl logs`가 사라집니다.** run별 로그는 `/workspace/results`에 남아 있으니, 다음 `sync.sh` 때(또는 `k8s/sync.sh <config> --no-run`) `A100_results/`로 받아옵니다.
- **`make_calib_configs.py`, `make_heldout_configs.py`는 `k8s/calib/`를 덮어씁니다.** 이 폴더는 실제로 실행한 실험의 기록이니 새 실험은 다른 경로에 만드세요.
- 폴더를 pod에 복사할 때는 `kubectl cp` 대신 `tar`를 쓰세요(`kubectl cp`는 대상 폴더가 있으면 한 겹 더 만들 수 있음).
- tc는 각 pod의 **송신(egress)** 에만 걸립니다. A→B와 B→A는 각자 자기 행렬 값을 씁니다.
- 실험 중 측정값의 run 간 편차는 약 ±1초입니다. 비교할 때는 반복 측정하세요.

## DT-FM에 적용한 수정

- `comm/nccl_backend.py`: CuPy 13+에서 `cupy.cuda.nccl`이 없어져 `cupy_backends.cuda.libs.nccl` 사용. NCCL unique id를 bytes로 전달
- `pipeline_parallel/dist_pp_utils.py`: 저장소에 없는 모듈 import 주석 처리

이미지(torch 2.4.1, CUDA 12.1)에서 돌리기 위한 최소 수정이며 학습 로직은 그대로입니다.
