# training_local — 로컬 RL 파이프라인 (Tinker 없음)

DeepSeek-V4-Flash-0731 (FP8) + LoRA 위에서 Harness-1의 로컬 RL/SFT 학습
파이프라인. 고수준 개요는 메인 `README.md`를 볼 것.

## 모듈

| 파일 | 역할 |
|---|---|
| `__init__.py` | 패키지 init, 버전 |
| `config.py` | 모든 하이퍼파라미터를 dataclass로 (env override 가능) |
| `encoding.py` | DeepSeek-V4 `encoding_dsv4.py` wrapper |
| `rewards.py` | 4-요소 recall + NDCG + GRPO advantage (OAPL) |
| `tools_adapter.py` | harness `ToolSet` ↔ OpenAI tool-call 스키마 브릿지 |
| `env.py` | `LocalSearchEnv` — multi-turn agentic 검색 환경 |
| `agent.py` | `DeepSeekPolicyInferenceModel` — vLLM 기반 정책 |
| `data.py` | `SearchDataset`에서 쿼리 로드, Sid-1 ID 난독화 적용 |
| `checkpoint.py` | LoRA 어댑터 + 병합된 full-model save/load |
| `rollout.py` | episode 드라이버: `run_episode`, `run_group` |
| `train_rl.py` | 메인 RL 진입점 |
| `train_sft.py` | SFT warm-start 진입점 |
| `smoke_test.py` | 연결 검사 (GPU 불필요) |
| `validate_rollout.py` | 실제 모델 단일 턴 롤아웃 검증 (GPU 2대) |
| `validate_rollout.sh` | 위 스크립트의 bash wrapper (기본 GPU 6,7) |
| `validate_multiturn.py` | multi-turn + tool_call 검증 (GPU 2대) |
| `validate_multiturn.sh` | 위 스크립트의 bash wrapper |
| `patch_vllm.py` | vLLM 0.25 설치 후 적용하는 호환성 패치 (idempotent) |
| `_vllm_runtime_patches.py` | agent.py / validate_* 에서 import 시점에 호출되는 패치 훅 |
| `launch_rl.sh` | 안전한 기본값의 bash 런처 |
| `launch_sft.sh` | SFT용 bash 런처 |
| `backends/verl_runner.py` | verl GRPO + 커스텀 롤아웃 (주 백엔드) |
| `backends/trl_runner.py` | TRL GRPOTrainer 폴백 |

## 일반적인 워크플로우

```bash
# 1. smoke test (GPU 불필요)
uv run python -m training_local.smoke_test

# 2. 단일 턴 롤아웃 검증 (GPU 2대, 약 8분)
bash training_local/validate_rollout.sh

# 3. multi-turn 롤아웃 검증 (GPU 2대, 약 8분)
bash training_local/validate_multiturn.sh

# 4. SFT warm-start (outputs/sft_runs/...에 어댑터 생성)
bash training_local/launch_sft.sh

# 5. RL 학습 (SFT 어댑터 또는 베이스 모델에서 이어서)
INIT_FROM_CHECKPOINT=outputs/sft_runs/<run>/final \
    bash training_local/launch_rl.sh

# 6. LoRA 병합 후 vLLM으로 서빙
uv run python -c "
from training_local.checkpoint import save_merged_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    'deepseek-ai/DeepSeek-V4-Flash-0731',
    torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto',
)
model = PeftModel.from_pretrained(base, 'outputs/rl_runs/<run>/final')
tok = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V4-Flash-0731')
save_merged_model(model, tok, 'outputs/merged')
"

vllm serve outputs/merged \
    --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
    --enable-expert-parallel --tensor-parallel-size 8
```

## vLLM 호환성 패치

vLLM 0.25.0+cu129 wheel은 PyPI openai/cutlass/tilelang 버전과 충돌이 있다.
`patch_vllm.py`가 이를 자동으로 잡는다:

```bash
uv run --no-sync python -m training_local.patch_vllm
```

이 스크립트는 idempotent하다 (이미 패치된 파일은 skip). 적용하는 패치:

1. **NamespaceTool import alias** — vLLM 0.25 가 `openai.types.responses.NamespaceTool`
   를 참조하지만 stable openai wheel에는 없다. `try/except`로 `FunctionTool`에 alias.
2. **tilelang stub library symlink** — tilelang 0.1.11 stub(`libcudart_stub.so`
   등)가 깨진 symbol을 가짐. 실제 CUDA lib로 symlink.
3. **MHC `forward_cuda` 4종 게이트** — vLLM MHC op가 tilelang을 hardcode하지만,
   tilelang이 깨졌을 때 `forward_native` torch/triton 폴백으로 빠지도록.
4. **`cute.core.ThrMma = ThrMma` alias** — cutlass-dsl 4.6.x는 `cute.ThrMma`로만
   노출인데, vLLM fmha annotation이 `cute.core.ThrMma`를 eager evaluate.
5. **`nvvm.fmax` new-API 강제** — vLLM이 CUDA 12.9 분기에서 old API( positional
   3개)로 가는데, cutlass-dsl 4.6.1은 new API( positional 2개)만 받음. `if False:`로
   강제 skip.

`launch_rl.sh`/`launch_sft.sh`/`validate_rollout.sh`/`validate_multiturn.sh`
모두가 시작 전에 이 스크립트를 자동 호출한다. venv를 재설치한 후에도
새로 패치를 적용할 필요 없다.

## `encoding.py` wrapper의 런타임 보정

DSv4 인코딩 모듈은 모델 저장소의 `encoding_dsv4.py`를 동적 로드하는데,
vLLM 0.25 환경에서 두 가지 사소한 불일치가 있다. wrapper가 자동으로 보정한다:

1. **EOS 자동 보강** — vLLM이 `RequestOutput.text`에서 EOS sentinel을 떼어냄.
   DSv4 parser는 text 끝에 `<｜end▁of▁sentence｜>`가 있어야 파싱이 됨.
2. **다중 `</think>` 복구** — thinking 모드에서 모델이 가끔 `</think>`를 여러 번
   emit (특히 `reasoning_effort="high"` 일 때). DSv4 parser는 첫 번째만 인식하고
   나머지를 에러로 reject. wrapper가 첫 번째 이후 영역의 추가 `</think>`를 strip.

이 보정 덕분에 검증 스크립트는 물론 RL 롤아웃까지 모델 출력을 그대로
`parse_completion`에 넣어도 된다.

## 설정

모든 하이퍼파라미터는 env var로 override 가능. 전체 목록과 기본값은 `config.py`.
자주 쓰는 override:

```bash
# 스케일
GROUP_SIZE=16 BATCH_SIZE=64

# 백엔드
BACKEND=verl              # 또는 trl, 또는 auto

# 메모리
ROLLOUT_TP_SIZE=8         # 롤아웃에 8개 H200 전부 사용
ROLLOUT_GPU_MEM_UTIL=0.5  # vLLM이 쓸 GPU 메모리 비율

# 커리큘럼
MAX_TURNS_START=16 MAX_TURNS_END=64 MAX_TURNS_SCHEDULE_STEPS=200

# 보상 shaping
OUTCOME_WEIGHT=0.8 NDCG_WEIGHT=0.3 TURN_PENALTY_MAX=0.05

# 로깅
LOG_WANDB=0               # wandb 끄기
RUN_NAME=rl_experiment_1
```

## 백엔드 선택

파이프라인은 verl이 설치되어 있으면 자동으로 선택하고, 아니면 TRL로 폴백한다:

```bash
# 자동 (기본값)
bash training_local/launch_rl.sh

# verl 강제
BACKEND=verl bash training_local/launch_rl.sh

# TRL 강제 (디버깅 / 단일 GPU smoke용)
BACKEND=trl bash training_local/launch_rl.sh
```

**어떤 걸 언제 쓰나:**
- **verl:** multi-turn 롤아웃을 하는 운영용 284B RL. deepspeed + ray + vllm 필요.
  커스텀 롤아웃으로 우리의 `LocalSearchEnv`을 처리한다.
- **TRL:** 빠른 smoke 테스트 / 단일 GPU 디버깅. 표준 `GRPOTrainer`가 stateful
  multi-turn 환경을 네이티브로 구동하지 못해서, 폴백은 한 completion을 한 턴으로
  취급한다. 보상 함수가 end-to-end으로 동작하는지 검증하는 데 유용.

## 참고문헌

논문 참조 (Harness-1, Sid-1, KARL, DeepSeek-V4)는 메인 프로젝트 `README.md`에서.
