# Harness-1 Local

**DeepSeek-V4-Flash-0731 (FP8) + LoRA 기반 로컬 RL 파이프라인. H200 ×8 대상. Tinker 없이 동작.**

원본 [Harness-1](https://arxiv.org/abs/2606.02373) 의 fork로, 호스팅 Tinker 학습 서비스를
**verl**(주 백엔드) 또는 **TRL GRPOTrainer**(폴백) 기반 로컬 RL 스택으로 교체했다.
Harness-1의 실제 핵심 자산 — stateful 검색 하네스, 도구 환경, 4-요소 recall 보상 — 은
그대로 보존했다.

---

## 이 fork가 하는 일

284B Mixture-of-Experts 검색 에이전트(`DeepSeek-V4-Flash-0731`, 13B activated)를
multi-turn 검색 에피소드 위에서 GRPO 강화학습으로 학습한다. 에이전트는 다음을 배운다:

- 검색 백엔드(Chroma)에 정확한 쿼리를 날린다
- 검색 결과 문서를 읽고 증거 큐레이션을 한다
- 검색된 문서 기반으로 claim 검증을 한다
- 증거가 충분하면 탐색을 중단한다

보상은 recall 기반이다: 에이전트가 gold evidence 문서를 찾아 큐레이션했는가?

---

## 목표

1. **호스팅 서비스 없는 재현성.** Tinker도 Modal도 없다 — 오직 당신의 GPU와 검색 백엔드.
2. **단일 노드에서 284B MoE 스케일.** vLLM + DSpark speculative decoding 기반 FP8
   롤아웃; FSDP 기반 BF16 LoRA 학습.
3. **베스트 아이디어 차용.** 보상 설계와 롤아웃 프로토콜은 아래 출처에서 기법을 가져온다:
   - **Harness-1** (원본 4-요소 recall 보상)
   - **Sid-1** (TI/TO 프로토콜, NDCG 보상, 길이 스케줄링, format 안정성을 위한 큰 group size)
   - **KARL** (arXiv:2603.05218) (OAPL two-KL GRPO, nugget 평가, self-compression,
     pass-rate 필터링된 SFT 데이터)

---

## Quickstart

### 1. 환경

Python 3.12 필요 (`.python-version` 파일이 pin; torch 2.11은 Python 3.12 ABI로 빌드돼
있어 3.11에서는 동작하지 않는다). base sync는 harness 의존성과 함께
`torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`을 pin한다 (`[tool.uv.sources]`가
PyTorch cu128 index에서 당겨옴). **중요:** driver 570/CUDA 12.9 호스트에서는 cu130
wheel이 첫 kernel op에서 죜 수 있으므로 반드시 cu128 stack을 써야 한다. NCCL 2.28.9는
torch+cu128 의존성으로 자동으로 따라온다.

RL extras는 별도 두 개의 requirements 파일로 관리한다 — vLLM이 CUDA-specific wheel을
배포하는데, PyPI 기본 wheel(0.25.0)은 cu130 빌드라 같은 이유로 동작하지 않는다.
`requirements.rl.txt`가 GitHub release의 cu129 wheel URL을 직접 가리킨다.

```bash
uv sync                                                    # base (Python 3.12, cu128 torch)
uv pip install --no-deps -r requirements.rl.txt           # verl, vllm cu129, deepspeed, ray
uv pip install -r requirements.rl.runtime.txt             # 위 패키지들의 runtime deps
```

Python 3.12가 없다면 `uv python install 3.12` 후 `uv venv --python 3.12`.
`.python-version` 파일이 `uv run`에 대해 이 버전을 고정한다.

**호스트 `PYTHONPATH` 주의:** 공용 머신 등에서 `PYTHONPATH`가 `~/.local/lib/...`
를 가리키고 있으면 venv torch를 shadowing한다. `launch_rl.sh`/`launch_sft.sh`/
`validate_rollout.sh`는 자동으로 `unset PYTHONPATH`하지만, 직접 `uv run`을 호출할
때는 `env -u PYTHONPATH uv run ...` 형태로 실행해야 한다.

### 2. 베이스 모델 다운로드

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731
# (~157 GB, FP8 + FP4 혼합 정밀도; $HF_HOME/hub/ 아래 캐시)
```

검증:

```bash
du -sh "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/"
# 예상: ~157G
```

### 3. 자격증명 + 검색 백엔드 설정

```bash
cp .env.example .env.local
# .env.local 편집: OPENAI_API_KEY, CHROMA_API_KEY, CHROMA_DATABASE
# (HUGGINGFACE_TOKEN, ANTHROPIC_API_KEY는 사용 도구에 따라 옵션)
```

검색 백엔드는 호스팅 Chroma 인스턴스다. 필요한 것:
- `OPENAI_API_KEY` — 쿼리/후보 임베딩(text-embedding-3-large)용
- `CHROMA_API_KEY`, `CHROMA_DATABASE` — 벡터 스토어용
- Chroma에 사전 색인된 코퍼스(BrowseComp+, SEC, 특허, web). chunk ID가
  각 데이터셋 adapter의 qrels과 일치해야 한다. 색인 파이프라인은
  `datagen/README.md` 참조. 빠른 건강 검사:

  ```bash
  uv run python -c "
  from datagen.search_dataset import get_dataset
  ds = get_dataset('sec')
  print('train/test/rl queries:',
        len(ds.get_train_query_ids()),
        len(ds.get_test_query_ids()),
        len(ds.get_rl_query_ids()))
  "
  ```

### 4. smoke test (GPU 불필요)

```bash
unset PYTHONPATH  # 호스트 site-packages의 호환 안되는 torch ABI 차단
uv run python -m training_local.smoke_test
```

다섯 가지를 검사한다: `training_local` 모듈 전체 import, config 기본값, 모델
스냅샷에서 `encoding_dsv4.py` 동적 로드, NDCG + GRPO advantage 수학, `QueryRecord` /
`EnvState` 생성 가능성. 전부 통과해야 하며, 하나라도 실패하면 non-zero로 종료한다.

### 4b. rollout 검증 (GPU 2대 필요, 실제 모델 로드)

smoke test가 지나면 실제 모델이 2대 GPU에서 로드되는지 확인한다. 검색 백엔드나
보상 신호 없이 최소 루프만 검사한다:

```bash
bash training_local/validate_rollout.sh
# 또는 직접 GPU 지정: CUDA_VISIBLE_DEVICES=0,1 bash training_local/validate_rollout.sh
```

4가지를 검사한다: (1) `encoding_dsv4` 모듈 로드 + 프롬프트 렌더 (2) vLLM TP=2 +
FP8 + DSpark 7-token speculative 시작 (3) 샘플링 1회 (4) `parse_completion`이
DSML tool call 포맷을 잘라낸다.

H200 ×2 (TP=2)에서의 실측: vLLM 시작 ~450초(모델 로드 + DeepGEMM warmup +
CUDA graph capture) + 샘플링 수 초. 검증이 끝나면 자동으로 vLLM을 종료한다.

### 4c. multi-turn rollout 검증 (GPU 2대 필요)

단일 턴 검증이 지나면, RL trainer가 실제로 구동할 multi-turn 루프를 검사한다:

```bash
bash training_local/validate_multiturn.sh
```

system + user + tools 프롬프트에서 출발해 (1) turn 1의 sample → parse → tool_call
추출, (2) tool_result 메시지를 append한 turn 2의 sample → 최종 답 추출까지 확인한다.
`tool_call_id` 가 인코딩/디코딩을 무사히 통과하는지, 다중 턴에서 프롬프트 길이가
`max_model_len` 안에 들어가는지를 잡는다. H200 ×2에서 vLLM 시작 ~450초 + 2턴
샘플링 ~15초.

### 현재 검증 상태 (2026-08-02, H200 ×2 GPU 6,7)

| 검증 | 상태 | 비고 |
|---|---|---|
| `smoke_test.py` (GPU 불필요) | PASS | import, config, encoding, reward, 데이터 구조 |
| `validate_rollout.sh` (단일 턴) | PASS | vLLM 0.25.0+cu129, 451.2s init, 453 토큰 샘플 |
| `validate_multiturn.sh` (multi-turn) | PASS | 453.6s init, web_search → tool_result → 최종 답 |
| SFT warm-start (`launch_sft.sh`) | 미검증 | 검색 백엔드 설정 후 실행 필요 |
| RL 학습 (`launch_rl.sh`) | 미검증 | 검색 백엔드 + SFT warm-start 후 실행 필요 |

단일 턴과 multi-turn은 모델 로딩과 DSv4 인코딩/파싱 파이프라인 전체가 실제
H200에서 동작함을 확인한다. RL 본학습은 검색 백엔드(Chroma) 설정이 추가로
필요해서 본 문서 범위 밖이다. vLLM 시작 시간의 대부분(380초+)은 48개 FP8
safetensors shard 로드와 51개 PIECEWISE + 48개 FULL CUDA graph capture에
소모된다.

### 5. SFT warm-start (선택이지만 권장)

GPT-5.4로 합성 검색 궤적을 생성하고(쿼리 50–200개), pass rate로 필터링한 뒤
정책을 warm-start한다:

```bash
bash training/launch_sft_generation.sh          # 출력 → tmp/sft_data/
bash training_local/launch_sft.sh               # 출력 → outputs/sft_runs/<run>/final/
```

GPU 투입 전 연결을 확인하는 smoke 버전:

```bash
SMOKE_TEST=1 SFT_BATCH_SIZE=1 bash training_local/launch_sft.sh
```

### 6. RL 학습

SFT에서 warm-start (권장):

```bash
INIT_FROM_CHECKPOINT=outputs/sft_runs/<run_name>/final \
    bash training_local/launch_rl.sh
```

베이스 가중치에서 cold-start (SFT 건너뛰기):

```bash
bash training_local/launch_rl.sh
```

자주 쓰는 override (전부 환경변수 — 전체 목록은 `training_local/config.py`):

```bash
GROUP_SIZE=16 BATCH_SIZE=64 RUN_NAME=rl_v1 bash training_local/launch_rl.sh
BACKEND=trl bash training_local/launch_rl.sh       # TRL 폴백 강제
SMOKE_TEST=1 bash training_local/launch_rl.sh      # 1 step, GPU 없음
ROLLOUT_TP_SIZE=8 bash training_local/launch_rl.sh # TP=4에서 weight load OOM 시
```

### 7. 모니터링 (선택)

`LOG_WANDB=1`(기본값)이면 wandb 프로젝트를 연다. Sid-1 기준으로 다음을 주시한다:
- `reward/ndcg` 상승 (랭킹 품질 개선)
- `reward/recall` 상승 (gold 문서 더 많이 큐레이션)
- `metrics/n_turns` 감소 (효율성 학습)
- format pass rate가 0.95 아래로 떨어지면 → `GROUP_SIZE` 증가

### 8. LoRA 병합 + 서빙

학습 후 어댑터를 베이스에 병합해서 vLLM 서빙용으로 만든다. 전체 스니펫은
`docs/training_guide.md` §9 참조, 그 후:

```bash
vllm serve outputs/merged \
    --trust-remote-code --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 --max-model-len 131072 \
    --enable-expert-parallel \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

### 9. 평가

```bash
uv run python inference/evaluate_harness1_vllm.py \
    --model outputs/merged \
    --dataset browsecompplus --split test \
    --output tmp/eval_results.json
```

전체 평가 레시피는 `docs/run_vllm_browsecompplus.md` 참조.

---

## 아키텍처

```
                    ┌─────────────────────────────────────────┐
                    │           training_local/                │
                    │                                          │
   queries ───────► │  data.py    (SearchDataset adapter)      │
                    │     │                                    │
                    │     ▼                                    │
                    │  env.py     (LocalSearchEnv)             │
                    │     │        ┌── WorkingMemory           │
                    │     │        ├── 9 tools (search, read,  │
                    │     │        │   curate, verify, ...)    │
                    │     │        └── 4-요소 recall            │
                    │     ▼                                    │
                    │  encoding.py (DeepSeek encoding_dsv4)    │
                    │     │                                    │
                    │     ▼                                    │
                    │  rollout.py (multi-turn episode driver)  │
                    │     │                                    │
                    │     ▼                                    │
                    │  backends/                               │
                    │     ├── verl_runner.py  (주 백엔드)       │
                    │     │     ├── Megatron actor              │
                    │     │     ├── vLLM rollout (FP8 + DSpark) │
                    │     │     └── GRPO + OAPL                 │
                    │     └── trl_runner.py   (폴백)            │
                    │           ├── TRL GRPOTrainer             │
                    │           ├── PEFT LoRA                   │
                    │           └── vLLM colocate               │
                    └─────────────────────────────────────────┘
                                      │
                                      ▼
                            LoRA adapter checkpoint
                            → 병합 → vLLM serve
```

---

## 핵심 기술 선택

### 인코딩

DeepSeek-V4-Flash는 모델 저장소에 포함된 **커스텀 인코딩**(`encoding_dsv4.py`)을
쓴다 — Jinja chat template이 아니다. DSML 포맷의 tool call을 내보내고,
`thinking_mode`와 `low`/`high`/`max` reasoning effort를 지원한다.

### 보상

sparse terminal 보상, 다음 구성요소들을 합친다:

| 구성요소 | 출처 | 가중치(기본) |
|---|---|---|
| 큐레이션 set의 recall | Harness-1 | 0.7 |
| trajectory pool의 recall | Harness-1 | 0.3 |
| 최종 답안 recall (큐레이션) | Harness-1 | 0.8 |
| 최종 답안 recall (trajectory) | Harness-1 | 0.4 |
| 큐레이션 랭킹의 NDCG | Sid-1 | 0.2 |
| 도구 다양성 보너스 | Harness-1 | 0.25 × (distinct / 6) |
| 턴 페널티 (20턴 이후) | Harness-1 | -0.02 × 초과분 |
| FA miss 페널티 | Harness-1 | -0.35 × miss_rate |
| 최종 답안 성공 보너스 | Harness-1 | +1.0 if FA recall = 1.0 |

### GRPO

group size 8 (Sid-1: step 0에서 format pass rate > 0.95를 주는 크기).
advantage는 OAPL soft-min (KARL)으로 계산, `β₁ = 1.0`:

```
V̂*(x) = β₁ · ln( (1/G) Σᵢ exp(rᵢ / β₁) )
Aᵢ    = rᵢ - V̂*(x)
```

PPO surrogate clipping at ε = 0.2. KL 계수 `0.005`.

### TI/TO 프로토콜 (Sid-1)

Tokens-In / Tokens-Out: 매 턴마다 대화 히스토리 전체를 구조화된 메시지 리스트에서
다시 인코딩한다. 모델 출력을 파싱해서 메시지로 재조립 후 다시 렌더하는 패턴은
tool call 경계 토큰에서 log-prob 붕괴를 일으키므로 쓰지 않는다.

### 길이 스케줄링 (Sid-1)

`max_turns`가 처음 500 step 동안 32 → 128로 증가한다. 초반에는 짧고 효율적인
궤적을 유도하고, 후반으로 갈수록 긴 horizon 검색 동작으로 전환한다.

### SFT용 pass-rate 필터링 (KARL)

합성 SFT 궤적 중 `pass_rate ≤ 0.1` 또는 `≥ 0.9`인 것은 필터링된다 — 학습
신호가 없다 (너무 쉽거나 너무 어려움).

---

## 원본 Harness-1에서 바뀐 점

| 측면 | 원본 | 이 fork |
|---|---|---|
| 연산 백엔드 | Tinker 호스팅 서비스 | 로컬 GPU (verl/TRL + vLLM) |
| 베이스 모델 | `openai/gpt-oss-20b` | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| 토큰화 | Harmony (gpt-oss 전용) | DeepSeek `encoding_dsv4` (커스텀) |
| 보상 | 4-요소 recall | 4-요소 recall + NDCG (Sid-1) |
| GRPO advantage | Tinker 내부 | OAPL soft-min (KARL) |
| 길이 커리큘럼 | 고정 `MAX_TURNS=128` | 500 step에 걸쳐 32 → 128 |
| multi-epoch 데이터 | (다루지 않음) | ID 난독화 (Sid-1) |
| 필수 API 키 | `TINKER_API_KEY` | 없음 (옵션 `OPENAI_API_KEY`, `CHROMA_API_KEY`) |

삭제된 파일: `tinker-cookbook/` (165개 파일), `training/train_rl.py`,
`training/train_sft.py`, `training/launch_rl.sh`,
`training/launch_sft_training.sh` (전부 `training_local_backup/`에 백업).

추가된 파일: `training_local/` (새 파이프라인 전체).

그대로 보존된 파일: `harness/ultra_core.py`, `harness/tools.py`,
`harness/trajectory.py`, `harness/rerank.py`, `datagen/`, `inference/`,
`harness/agent.py` (`TinkerAgentInferenceModel` /
`ModalHarmonyAgentInferenceModel` 클래스만 제거).

---

## 리포지토리 구조

```
.
├── README.md                    ← 이 파일
├── pyproject.toml               ← uv 프로젝트 (Tinker 제거)
├── requirements.rl.txt          ← verl/vllm/deepspeed/ray (--no-deps install)
├── requirements.rl.runtime.txt  ← 위 패키지들의 runtime deps
├── .env.example                 ← 자격증명 템플릿
├── .python-version              ← Python 3.12 고정
├── harness/                     ← stateful 검색 환경 + 도구 + 보상 (보존)
├── datagen/                     ← 데이터셋 adapter (보존)
├── inference/                   ← 평가 러너 (보존)
├── docs/
│   ├── architecture.md          ← 설계 심층
│   ├── training_guide.md        ← 실제 학습 실행 가이드
│   ├── changelog.md             ← 무엇이 바뀌었고 왜 바뀌었나
│   └── run_vllm_browsecompplus.md  ← BrowseComp+ 평가 레시피
├── training/                    ← SFT 데이터 생성 (보존)
│   └── generate_sft_data.py
├── training_local/              ← 신규: 로컬 RL/SFT 파이프라인
│   ├── README.md
│   ├── encoding.py              ← DeepSeek 인코딩 wrapper
│   ├── config.py                ← 모든 하이퍼파라미터
│   ├── env.py                   ← LocalSearchEnv (multi-turn)
│   ├── rewards.py               ← 4-요소 + NDCG + GRPO advantage
│   ├── rollout.py               ← episode 드라이버
│   ├── agent.py                 ← vLLM 기반 정책
│   ├── data.py                  ← SearchDataset adapter
│   ├── checkpoint.py            ← LoRA 저장/로드
│   ├── tools_adapter.py         ← harness ToolSet ↔ OpenAI tool-call 스키마
│   ├── train_rl.py              ← RL 진입점
│   ├── train_sft.py             ← SFT 진입점
│   ├── smoke_test.py            ← 연결 검사 (GPU 불필요)
│   ├── launch_rl.sh
│   ├── launch_sft.sh
│   └── backends/
│       ├── verl_runner.py       ← 주 RL 백엔드
│       └── trl_runner.py        ← 폴백
├── training_local_backup/       ← 원본 Tinker 기반 스크립트 (.bak)
└── tests/
```

---

## 하드웨어 타깃

- **학습:** 8× NVIDIA H200 (개당 141 GB HBM3e, 합산 ~1.15 TB)
- **메모리 예산:** 284B FP8 ≈ 157 GB 가중치 + LoRA + activation + KV cache가
  1.15 TB 안에 충분히 들어옴
- **롤아웃 TP size:** 4 (H200 쌍 단위). `enable_expert_parallel=True`
- **KV cache dtype:** FP8
- **speculative decoding:** DSpark, 7 draft token

다른 GPU 토폴로지에서는 `ROLLOUT_TP_SIZE`, `ROLLOUT_GPU_MEM_UTIL`,
`ROLLOUT_MAX_MODEL_LEN`을 조정한다.

---

## 지원 데이터셋

업스트림 Harness-1 + datagen에서 상속:

- `browsecompplus` — BrowseComp+ 웹 증거 검색
- `sec` — SEC 공시 재무 QA
- `patents` — 특허 검색
- `web` — 웹 검색
- `longsealqa`, `frames`, `hotpotqa_subset`, `seal0qa` — 보조

각각 별도 Chroma 컬렉션이 필요하다. 설정은 `datagen/README.md` 참조.

---

## 알려진 한계

- **TRL 폴백에서의 multi-turn 롤아웃은 single-turn이다.** TRL의 `GRPOTrainer`가
  stateful 환경을 직접 구동하지 않기 때문이다. verl 백엔드는 커스텀 롤아웃 드라이버로
  이를 지원한다. 운영용 RL에는 verl을 쓴다.
- **SFT 궤적을 DeepSeek 포맷으로 재생성해야 한다.** 원본 `generate_sft_data.py`는
  Harmony 포맷 JSON을 생성하며, 로드 시점에 변환하지만 네이티브 DeepSeek 생성기는
  로드맵에 있다.
- **verl LoRA + DSpark 가중치 동기화는 아직 검증되지 않았다.** verl의 DeepSeek V4
  연동 문서는 full-param 학습을 다루며, LoRA + FP4 expert 가중치 동기화는 이 스케일에서
  테스트되지 않았다.
- **Chroma 검색 백엔드 필수.** 번들로 제공되지 않으며 `CHROMA_API_KEY` /
  `CHROMA_DATABASE`로 설정한다.

---

## 참고문헌

- **Harness-1 논문:** [arXiv:2606.02373](https://arxiv.org/abs/2606.02373) —
  원본 stateful harness + 4-요소 보상
- **Sid-1 기술 보고서:** [sid.ai/research/sid-1-technical-report](https://www.sid.ai/research/sid-1-technical-report) —
  TI/TO, NDCG, 길이 스케줄링, ID 난독화
- **KARL:** [arXiv:2603.05218](https://arxiv.org/abs/2603.05218) — OAPL,
  nugget 평가, self-compression, pass-rate 필터링
- **DeepSeek-V4 논문:** [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) —
  MoE 토폴로지, MXFP4 RL 롤아웃
- **verl DeepSeek-V4 연동:** [verl docs](https://verl.readthedocs.io/en/latest/advance/deepseek_v4_integration.html)
- **vLLM DeepSeek-V4 블로그:** [vllm.ai/blog/2026-04-24-deepseek-v4](https://vllm.ai/blog/2026-04-24-deepseek-v4)

---

## 인용

이 fork를 사용할 경우 Harness-1과 DeepSeek-V4 둘 다 인용한다:

```bibtex
@article{jiang2026harness,
  title={Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses},
  author={Jiang, Pengcheng and Shi, Zhiyi and Hong, Kelly and Xu, Xueqiang and Sun, Jiashuo and Sun, Jimeng and Bashir, Hammad and Han, Jiawei},
  journal={arXiv preprint arXiv:2606.02373},
  year={2026}
}

@article{deepseek2026v4,
  title={DeepSeek-V4: Towards Highly Efficient Million-Token Context},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2606.19348},
  year={2026}
}
```
