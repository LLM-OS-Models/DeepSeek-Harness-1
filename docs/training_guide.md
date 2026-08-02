# 학습 가이드 — Harness-1 Local

DeepSeek-V4-Flash-0731 검색 에이전트를 LoRA RL로 학습시키는 end-to-end 레시피.
H200 ×8 (권장) 또는 H100 ×8 (가능).

## 사전 요구사항

### 하드웨어

두 가지 구성이 지원된다:

| 구성 | HBM (카드당) | max_model_len | rollover TP | 비고 |
|---|---|---|---|---|
| H200 ×8 | 141 GB | 131072 | 4 또는 8 | 긴 컨텍스트, 큰 batch. 권장. |
| H100 ×8 | 80 GB | 32768–65536 | 8 | KV cache 빠듯. max_model_len 축소 필수. |

공통 요구사항:
- NVLink 기반 tensor parallel (PCIe 토폴로지는 TP=2까지만 안정)
- 체크포인트 + SFT 데이터 + 모델 캐시용 디스크 ~1 TB
- CUDA 12.8+ driver (driver ≥ 570.x)
- 모델 가중치 ~157 GB (FP8+FP4)

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

# Base harness 의존성 (Python 3.12, torch 2.11.0+cu128 고정).
# pyproject.toml의 [tool.uv.sources]가 PyTorch cu128 index에서 당겨옴.
# driver 570/CUDA 12.9 호스트에서는 cu130 wheel이 kernel op에서 죽으므로
# 반드시 cu128 stack을 써야 한다.
uv sync

# RL extras. 두 파일:
#   requirements.rl.txt          — verl, vllm(GitHub cu129 wheel), deepspeed,
#                                  ray (--no-deps로 torch 다운그레이드 방지)
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
# torch.cuda.is_available()만 True여선 부족하다 — 실제 kernel op이 돌아야 검증.
x = torch.zeros(8, device='cuda:0')
assert (x+1).sum().item() == 8
print('cuda kernel ok')
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

### 4b. 단일 턴 롤아웃 검증 (GPU 2대)

smoke test가 지난 후, 실제 모델 로딩과 DSv4 인코딩/파싱 루프를 2대 GPU에서
검증한다. 검색 백엔드나 보상은 필요 없다 — 최소 루프만:

```bash
bash training_local/validate_rollout.sh
# 기본값: CUDA_VISIBLE_DEVICES=6,7, ROLLOUT_MAX_MODEL_LEN=8192
# 다른 GPU: CUDA_VISIBLE_DEVICES=0,1 bash training_local/validate_rollout.sh
```

4단계를 검사한다:
1. `encoding_dsv4` 모듈 동적 로드 + 시스템/유저 메시지 렌더
2. vLLM TP=2, FP8 KV cache, DSpark 7-token speculative 시작
3. 1회 샘플링 (temperature=0.7, max_tokens=512)
4. `parse_completion` 이 content/reasoning/tool_calls 구조로 변환

H200 ×2 실측: vLLM 시작 ~450초(모델 로드 + DeepGEMM warmup + 51 PIECEWISE +
48 FULL + 48 dspark CUDA graph capture) + 샘플링 수 초.

실패 시: `training_local/patch_vllm.py`가 자동 호출됐는지 로그에서 확인. 스크립트
자체가 `uv run --no-sync python -m training_local.patch_vllm || true`를 먼저
실행하지만, venv를 재설치했다면 수동으로 한 번 더 호출해 본다.

### 4c. multi-turn 롤아웃 검증 (GPU 2대)

단일 턴 검증이 지난 후, RL trainer가 실제로 구동할 multi-turn 루프를 검증한다:

```bash
bash training_local/validate_multiturn.sh
```

3단계를 검사한다:
1. vLLM 시작 (4b 와 동일)
2. Turn 1: `web_search` tool을 포함한 system+user 프롬프트에서 sample →
   parse → `tool_calls`가 `web_search` 로 잘 뽑히는지
3. Turn 2: tool_call_id 에 가짜 tool_result 를 append 후 re-encode →
   sample → 최종 답안이 tool_result를 참조하는지

`tool_call_id` 가 인코딩/디코딩을 무사히 통과하는지, multi-turn 프롬프트 길이가
`max_model_len` 안에 들어가는지를 잡는다. H200 ×2 실측: vLLM 시작 ~450초 +
2턴 샘플링 ~15초.

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

### vLLM import 또는 worker init 실패 (NamespaceTool, ThrMma, fmax, tilelang)

vLLM 0.25 wheel이 PyPI의 openai/cutlass-dsl/tilelang 버전과 충돌한다.
`training_local/patch_vllm.py`가 자동으로 잡는다:

```bash
unset PYTHONPATH && uv run --no-sync python -m training_local.patch_vllm
```

이 스크립트는 5개 패치를 적용한다 (NamespaceTool alias, tilelang stub symlink,
MHC forward_cuda 게이트, cute.ThrMma alias, nvvm.fmax 강제 new API). 자세한
사유는 `docs/changelog.md` 와 `training_local/README.md` 의 "vLLM 호환성 패치"
섹션. idempotent하므로 안전하게 재실행 가능.

`launch_rl.sh`/`launch_sft.sh`/`validate_*.sh`는 모두 시작 전에 자동 호출한다.
venv를 reinstall한 직후에만 수동으로 한 번 실행해 주면 된다.

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

## 비용 추정

### 학습 시간 (SEC 3 epoch, GROUP_SIZE=8)

계산 근거: 3,500 queries × 8 group × 3 epoch = 84K episodes. 평균 64 turns/episode
× 약 2K tokens/turn = 약 10.7B tokens 생성 필요.

| 구성 | 롤아웃 속도 (추정) | 순 학습 시간 | 평가 포함 | 비고 |
|---|---|---|---|---|
| **H200 ×8** | ~35K–45K tok/s (FP8 + DSpark 7) | 70–90시간 | **3–4일** | max_model_len=131K |
| **H100 ×8** | ~25K–32K tok/s (대역폭 70%) | 95–120시간 | **5–6일** | max_model_len=32K–65K |

이 추정은 평균 episode 길이 64 turn을 가정한 것이다. Sid-1 길이 스케줄링
(`MAX_TURNS_START=32 → MAX_TURNS_END=128`)을 따르므로, 초반은 짧고 후반은 길다.
실제 쿼리 난이도에 따라 ±50% 편차가 날 수 있다.

**완료 시점 확인:** wandb의 `reward/ndcg`가 step 200 이후 안정화되고,
`metrics/format_pass_rate`가 0.95 이상을 유지하면 수렴으로 본다. 3 epoch
전에 수렴하면 `MAX_STEPS`로 조기 종료한다.

### 디스크 / 전력

| 항목 | H200 ×8 | H100 ×8 |
|---|---|---|
| 전력 (700W TDP × 8) | ~5.6 kW | ~5.6 kW |
| 일일 전력 | ~135 kWh | ~135 kWh |
| 체크포인트 (LoRA 50개) | 2.5 GB | 2.5 GB |
| 최종 병합 모델 | ~157 GB | ~157 GB |
| 모델 캐시 (HF hub) | ~157 GB | ~157 GB |
| wandb 저장소 (run당) | ~100 MB | ~100 MB |

## 예상 결과 / 기대 metric

학습이 잘 진행되면 wandb에서 다음 변화가 관찰된다. 이것은 Sid-1 / Harness-1
논문의 baseline 범위이지, 보장된 수치가 아니다.

### 정량 metric

| metric | step 0 | 학습 후 기대치 | 의미 |
|---|---|---|---|
| `reward/recall` | 0.05–0.15 | 0.45–0.65 | gold 문서를 큐레이션 셋에 담는 비율 |
| `reward/ndcg` | 0.10–0.20 | 0.50–0.70 | 랭킹 품질 (이른 위치에 정답) |
| `metrics/n_turns` | 80–128 | 30–50 | 에피소드당 평균 턴 (효율성) |
| `metrics/format_pass_rate` | 0.70–0.85 | >0.95 | DSML tool_call 포맷 준수율 |
| `metrics/fa_hit_rate` | 0.10 | 0.40–0.55 | 최종 답안이 gold를 포함 |
| 도구 다양성 (distinct / 6) | 0.2–0.4 | 0.7–1.0 | 반복 행동 (동일 검색) 방지 |

### 정성적 변화

- **검색 쿼리 구체화:** "ACME revenue" → "ACME Corp FY2024 10-K revenue billion"
- **큐레이션 선택성:** 검색 결과 전체 dump → top-K relevant만 선택
- **자발적 종료:** max_turn 도달 전 증거 충분하면 stop 호출
- **증거 인용:** 최종 답안이 큐레이션한 chunk ID를 참조 (환각 감소)

### 문제 발생 징후와 대응

| 징후 | 원인 | 대응 |
|---|---|---|
| `format_pass_rate` < 0.95 | group size 부족 | `GROUP_SIZE=16` |
| `reward/recall` 정체 | exploration 부족 | `ROLLOUT_TEMPERATURE=1.2`, `MAX_TURNS_END=200` |
| `n_turns` 안 줄음 | turn penalty 약함 | `TURN_PENALTY_MAX=0.05` |
| `kl` 폭등 | 학습률 너무 높음 | `LEARNING_RATE=5e-6` (기본 1e-5) |
| 도구 다양성 < 0.5 | reward hacking | `TOOL_DIVERSITY_BONUS=0.5` |

## 재현성

고정할 것:
- `seed=42` (기본값)
- `pyproject.toml`과 `uv.lock`의 numpy, torch, vllm, verl 버전
- DeepSeek-V4-Flash-0731 스냅샷 해시 (commit 기록)
- Chroma 인덱스 버전

논문 수준의 재현성을 위해, 학습된 체크포인트 옆에 `pyproject.toml`, `uv.lock`,
`.env.example` 전체를 스냅샷으로 보관하라.
