# 학습 가이드 — Harness-1 Local

DeepSeek-V4-Flash-0731 검색 에이전트를 LoRA RL로 학습시키는 end-to-end 레시피.
H200 ×8 대상.

## 사전 요구사항

### 하드웨어
- 8× NVIDIA H200 (개당 141 GB), tensor parallel용 NVLink
- 체크포인트 + SFT 데이터 + 모델 캐시용 디스크 ~1 TB

### 소프트웨어
- CUDA 12.8+ (NCCL 2.29.7+ 필요, torch 2.11이 요구)
- Python 3.12+ (`.python-version`으로 고정; torch 2.11+는 3.12 ABI로 빌드)
- `uv` 0.11+ (환경 관리)
- vLLM 호환 GPU 드라이버

### API 자격증명 (`.env.local`에)
- `OPENAI_API_KEY` — 검색 임베딩 + claim 검증에 필수
- `CHROMA_API_KEY`, `CHROMA_DATABASE` — 검색 백엔드에 필수
- `HUGGINGFACE_TOKEN` — 모델이 인증을 필요로 할 때만 (DeepSeek-V4-Flash는 공개)
- `BASETEN_API_KEY`, `BASETEN_MODEL_URL` — 옵션, reranker
- `WANDB_API_KEY` — 옵션, 실험 추적

## 단계별 레시피

### 1. 환경 설정

```bash
git clone <this-repo> harness-1
cd harness-1

# Base harness 의존성 (Python 3.12, torch 2.11, NCCL 2.29.7 고정)
uv sync

# RL extras. 두 파일:
#   requirements.rl.txt          — verl, vllm, deepspeed, ray (--no-deps로
#                                  torch 다운그레이드 방지)
#   requirements.rl.runtime.txt  — --no-deps가 건너뛰는 runtime deps
uv pip install --no-deps -r requirements.rl.txt
uv pip install -r requirements.rl.runtime.txt
```

검증:

```bash
unset PYTHONPATH && uv run python -c "
import torch, verl, vllm, deepspeed, ray, peft, trl
print('torch', torch.__version__, 'verl', verl.__version__,
      'vllm', vllm.__version__)
"
```

### 2. 베이스 모델 다운로드

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731
# ~157 GB → $HF_HOME/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/
```

검증:
```bash
du -sh ~/.data/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/
# 예상: ~157G
```

### 3. 검색 백엔드 설정

`datagen/README.md`를 따라:
- BrowseComp+, SEC, 특허, web 코퍼스 다운로드
- 올바른 chunk ID로 Chroma 컬렉션 빌드
- qrels이 chunk ID와 일치하는지 확인

빠른 검사:
```bash
uv run python -c "
from datagen.search_dataset import get_dataset
ds = get_dataset('sec')
print('train queries:', len(ds.get_train_query_ids()))
print('test queries:', len(ds.get_test_query_ids()))
print('rl queries:', len(ds.get_rl_query_ids()))
"
```

### 4. smoke test 실행

```bash
unset PYTHONPATH && uv run python -m training_local.smoke_test
```

기대 결과: 5개 테스트 전부 통과 (import, config, encoding, reward, 데이터 구조).

### 5. (옵션) SFT 궤적 생성

```bash
bash training/launch_sft_generation.sh
# JSON 궤적을 tmp/sft_data/에 출력
```

KARL 방식의 필터링을 위해, 생성 중에 pass-rate 어노테이션을 설정하라. SFT 로더는
`SFT_MIN_PASS_RATE=0.1`, `SFT_MAX_PASS_RATE=0.9`로 자동 필터링한다.

### 6. SFT warm-start

```bash
bash training_local/launch_sft.sh
# 출력: outputs/sft_runs/<run_name>/final/
```

Smoke 버전:
```bash
SMOKE_TEST=1 SFT_BATCH_SIZE=1 bash training_local/launch_sft.sh
```

### 7. RL 학습 (메인 이벤트)

```bash
INIT_FROM_CHECKPOINT=outputs/sft_runs/<run_name>/final \
    bash training_local/launch_rl.sh
```

Smoke 버전 (1 step, GPU 없음):
```bash
SMOKE_TEST=1 bash training_local/launch_rl.sh
```

verl 백엔드 강제:
```bash
BACKEND=verl bash training_local/launch_rl.sh
```

TRL 폴백 강제:
```bash
BACKEND=trl bash training_local/launch_rl.sh
```

### 8. 학습 모니터링

`LOG_WANDB=1`(기본값)이면 wandb 프로젝트에서 다음을 볼 수 있다:
- `reward/total`, `reward/recall`, `reward/ndcg`, `reward/turn_penalty`
- `metrics/n_turns`, `metrics/n_curated`
- `loss/policy`, `loss/kl`, `loss/value`
- `rollout/tokens_per_second`

Sid-1 기준으로 주시할 것:
- `metrics/n_turns` 감소 (좋음 — 에이전트가 효율성을 학습 중)
- `reward/ndcg` 상승 (좋음 — 랭킹 품질 개선)
- `reward/recall` 상승 (좋음 — gold 문서 더 많이 큐레이션)
- Format pass rate가 0.95 아래로 추락 (나쁨 — `GROUP_SIZE`를 늘려라)

### 9. LoRA 병합 + 서빙

학습 후 LoRA를 베이스에 병합해서 vLLM 서빙용으로 만든다:

```bash
uv run python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from training_local.checkpoint import save_merged_model

base_path = 'deepseek-ai/DeepSeek-V4-Flash-0731'
adapter_path = 'outputs/rl_runs/<run_name>/final'

print('Loading base...')
base = AutoModelForCausalLM.from_pretrained(
    base_path, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map='cpu',
)
print('Applying adapter...')
model = PeftModel.from_pretrained(base, adapter_path)
tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

print('Merging and saving...')
save_merged_model(model, tok, 'outputs/merged')
"
```

서빙:
```bash
vllm serve outputs/merged \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --moe-backend deep_gemm_mega_moe \
    --attention-config '{"use_fp4_indexer_cache": true}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
    --tensor-parallel-size 8 \
    --max-model-len 131072
```

### 10. 평가

병합된 모델로 기존 `inference/evaluate_harness1_vllm.py`를 사용:

```bash
uv run python inference/evaluate_harness1_vllm.py \
    --model outputs/merged \
    --dataset browsecompplus \
    --split test \
    --output tmp/eval_results.json
```

## 자주 발생하는 문제

### 롤아웃 중 "CUDA out of memory"

`ROLLOUT_GPU_MEM_UTIL`(기본 0.45)을 낮추기:
```bash
ROLLOUT_GPU_MEM_UTIL=0.35 bash training_local/launch_rl.sh
```

또는 `ROLLOUT_MAX_MODEL_LEN`을 줄이기:
```bash
ROLLOUT_MAX_MODEL_LEN=65536 bash training_local/launch_rl.sh
```

### "vLLM OOM during weight loading"

284B MoE in FP8은 ~157 GB가 필요. 8× H200에서 TP=4면 카드당 ~40 GB.
KV cache 여유를 남겨라. 필요하면 TP=8:
```bash
ROLLOUT_TP_SIZE=8 bash training_local/launch_rl.sh
```

### 모든 롤아웃의 보상 = 0

확인:
1. 쿼리가 로드되었나? 로그에서 `n_queries > 0`.
2. gold_ids가 채워졌나? 데이터셋이 document_ids를 반환하는지 확인.
3. `chroma_collection_name`이 인덱스와 일치하나?

### Format pass rate < 0.95

Sid-1에 따라 group size를 늘려라:
```bash
GROUP_SIZE=16 bash training_local/launch_rl.sh
```

### 모델이 malformed tool call을 출력

`encoding.parse_completion`이 일반적인 출력에서 예외를 던지지 않는지 확인.
모델이 DSML이 아닌 tool call 포맷을 만들어낸다면, prompt에 tools 스키마가
빠져 있을 가능성이 높다. `env.render_prompt(state)`가 system 메시지의
`tools` 필드를 포함하는지 검증하라.

## 하이퍼파라미터 튜닝

### 너무 천천히 잊음 (KL이 너무 높음)

```bash
KL_PENALTY_COEF=0.001   # 기본 0.005
```

### 너무 빨리 잊음 (mode collapse)

```bash
KL_PENALTY_COEF=0.02    # 기본 0.005
```

### 궤적이 너무 김

```bash
TURN_PENALTY_MAX=0.05   # 기본 0.02
MAX_TURNS_END=64        # 기본 128
```

### 궤적이 너무 짧음 (일찍 포기)

```bash
TURN_PENALTY_MAX=0.0    # 비활성화
MAX_TURNS_END=200
TURN_PENALTY_MIN_TURNS=50
```

### 탐험을 더 원할 때

```bash
ROLLOUT_TEMPERATURE=1.2
ROLLOUT_TOP_P=0.98
```

## 비용 추정 (8× H200)

- **전력:** ~5 kW × 24h = 하루 ~120 kWh
- **실행 시간:** SEC 3 epoch에 ~3-7일 (3.5K 쿼리 × 8 group × 3 epoch)
- **체크포인트:** 50개 LoRA 어댑터 × ~50 MB = 2.5 GB
- **최종 병합 모델:** ~157 GB
- **wandb 저장소:** run당 ~100 MB

## 재현성

고정할 것:
- `seed=42` (기본값)
- `pyproject.toml`과 `uv.lock`의 numpy, torch, vllm, verl 버전
- DeepSeek-V4-Flash-0731 스냅샷 해시 (commit 기록)
- Chroma 인덱스 버전

논문 수준의 재현성을 위해, 학습된 체크포인트 옆에 `pyproject.toml`, `uv.lock`,
`.env.example` 전체를 스냅샷으로 보관하라.
