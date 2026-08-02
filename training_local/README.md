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
| `launch_rl.sh` | 안전한 기본값의 bash 런처 |
| `launch_sft.sh` | SFT용 bash 런처 |
| `backends/verl_runner.py` | verl GRPO + 커스텀 롤아웃 (주 백엔드) |
| `backends/trl_runner.py` | TRL GRPOTrainer 폴백 |

## 일반적인 워크플로우

```bash
# 1. smoke test (GPU 불필요)
uv run python -m training_local.smoke_test

# 2. SFT warm-start (outputs/sft_runs/...에 어댑터 생성)
bash training_local/launch_sft.sh

# 3. RL 학습 (SFT 어댑터 또는 베이스 모델에서 이어서)
INIT_FROM_CHECKPOINT=outputs/sft_runs/<run>/final \
    bash training_local/launch_rl.sh

# 4. LoRA 병합 후 vLLM으로 서빙
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
