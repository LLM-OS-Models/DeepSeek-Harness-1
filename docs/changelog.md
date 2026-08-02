# 변경 이력 — Harness-1 → Harness-1 Local

업스트림 Harness-1에서 이 Tinker 없는 fork까지의 모든 변경사항. 원본 파일
백업은 `training_local_backup/`에 있다.

## [0.2.0] — 2026-08-02 — Tinker 없는 로컬 RL

### 제거

- **`tinker-cookbook/`** (165개 파일, 2.3 MB) — 로컬에 vendoring된 Tinker cookbook.
  더 이상 사용하지 않음.
- **`tinker>=0.3.0`** 의존성 (`pyproject.toml`에서).
- **`tinker-cookbook`** editable 의존성 (`pyproject.toml`에서).
- **`openai-harmony>=0.0.8`** 의존성 (DeepSeek-V4는 커스텀 인코딩 사용).
- **`training/train_rl.py`** — `training_local/train_rl.py`로 교체.
- **`training/train_sft.py`** — `training_local/train_sft.py`로 교체.
- **`training/launch_rl.sh`** — `training_local/launch_rl.sh`로 교체.
- **`training/launch_sft_training.sh`** — `training_local/launch_sft.sh`로 교체.
- **`harness/agent.py`**: `TinkerAgentInferenceModel`과
  `ModalHarmonyAgentInferenceModel` 클래스 (388 라인, 파일의 약 25%).
- **`harness/agent.py`**: Tinker가 필요한 `__main__` 데모 블록 (143 라인).
- **`harness/config.py`**: `tinker_api_key` 필드, `tinker` import,
  `get_tinker_service_client()` 메서드, `TINKER_API_KEY` env export.

### 추가

- **`training_local/`** — 새 Tinker 없는 RL/SFT 파이프라인:
  - `encoding.py` — DeepSeek-V4 `encoding_dsv4.py` wrapper
  - `config.py` — `RLConfig`, `SFTConfig`, `ModelConfig`, `LoRAConfig`
  - `rewards.py` — 4-요소 recall + NDCG + GRPO advantage (OAPL)
  - `tools_adapter.py` — harness ToolSet ↔ OpenAI tool-call 스키마 브릿지
  - `env.py` — `LocalSearchEnv` (multi-turn agentic 검색)
  - `agent.py` — `DeepSeekPolicyInferenceModel` (vLLM 기반)
  - `data.py` — SearchDataset adapter, Sid-1 ID 난독화 내장
  - `checkpoint.py` — LoRA + 병합된 full-model save/load
  - `rollout.py` — multi-turn episode 드라이버 (`run_episode`, `run_group`)
  - `train_rl.py` — RL 진입점 (verl 또는 TRL 백엔드 자동 선택)
  - `train_sft.py` — SFT warm-start 진입점
  - `smoke_test.py` — 연결 검사 (GPU 불필요)
  - `launch_rl.sh`, `launch_sft.sh` — 안전한 기본값의 bash 런처
  - `backends/verl_runner.py` — 주 RL 백엔드
  - `backends/trl_runner.py` — 폴백 RL 백엔드

### 변경

- **`pyproject.toml`** — 전면 재작성: Tinker/Harmony 제거, harness 의존성 유지.
  `verl`/`deepspeed`/`ray`는 resolver 충돌을 피하기 위해 sync 후 수동으로
  설치 (`verl`과 `vllm`이 numpy pin을 서로 안 맞춤).
- **베이스 모델**: `openai/gpt-oss-20b` → `deepseek-ai/DeepSeek-V4-Flash-0731`
- **보상**: 순수 4-요소 recall → 4-요소 + NDCG (Sid-1) + 도구 다양성 + 턴 페널티
  + FA miss + FA 성공 보너스.
- **GRPO advantage**: Tinker 내부 mean-centering → OAPL soft-min (KARL β₁/β₂).
- **`max_turns`**: 고정 128 → 처음 500 step에 걸쳐 32 → 128로 ramp (Sid-1
  길이 스케줄링).
- **multi-epoch 학습**: Sid-1 ID 난독화 (`sha1(query_id|epoch)[:8]`)로
  암기를 방지.
- **SFT 데이터 필터링**: KARL pass-rate 필터 (0.1 ≤ pass_rate ≤ 0.9).
- **토크나이저**: `o200k_harmony` (tiktoken) → DeepSeek 토크나이저 (HF).
- **`README.md`**: 새 파이프라인으로 전면 재작성.

### 각 변경의 이유

- **Tinker 제거:** Tinker는 `TINKER_API_KEY`가 필요한 호스팅 서비스고, 엄선된
  모델 목록(gpt-oss 계열)만 받는다. DeepSeek-V4-Flash는 목록에 없다. 로컬로
  가는 길뿐이다.
- **TRL 대신 verl:** verl은 공식 DeepSeek-V4 연동
  (https://verl.readthedocs.io/en/latest/advance/deepseek_v4_integration.html)을
  갖췄다 — Megatron actor + vLLM 롤아웃. 300B+ MoE RL에 목적에 맞게 만들어졌다.
  TRL은 폴백인데, `GRPOTrainer`가 stateful multi-turn 환경을 네이티브로
  구동하지 않기 때문이다.
- **Harmony 대신 DeepSeek 인코딩:** Harmony는 gpt-oss 전용 (special token,
  채널 구조). DeepSeek-V4-Flash는 DSML 포맷의 tool call과 `<｜DSML｜...>` 토큰을
  쓴다 — Harmony와 호환되지 않는다.
- **NDCG 추가:** 순수 recall 보상은 과보고를 한다 (에이전트가 최대 recall을 위해
  검색된 모든 문서를 큐레이션 셋에 dump). NDCG는 이른 위치에 irrelevant 문서가
  있는 경우 페널티를 준다.
- **OAPL advantage:** 표준 mean-centered GRPO는 OAPL에서 β₁ → ∞인 특수 케이스다.
  유한 β₁에서 soft-min은 advantage 추정 분산을 줄이고 KARL에 따라
  sample-efficient 하다.
- **길이 스케줄링:** Sid-1은 짧은 롤아웃으로 시작해 ramp-up하는 것이 정책에게
  초기에 다루기 쉬운 학습 문제를 주고, 이후 긴 horizon 검색 동작으로 전환함을
  발견했다.
- **ID 난독화:** 이게 없으면 모델이 epoch에 걸쳐 `query_id → answer` 매핑을
  외워서, retrieval 기반 보상 신호를 무력화시킨다.

### 호환성

- 보존된 `harness/` 모듈들 (`ultra_core.py`, `tools.py`, `trajectory.py`,
  `rerank.py`, `tasks.py`)은 변경되지 않았고 import 호환성을 유지한다.
- `datagen/`, `inference/`는 변경되지 않았다.
- `tests/smoke_imports.py`는 업데이트가 필요할 수 있다 (`training/`에서
  import하는데, 더 이상 `train_rl` / `train_sft`가 없다). Phase 5 smoke test 참조.

### 설정 중에 적용된 환경 수정 (2026-08-02)

- **Python 3.11 → 3.12.** PyPI의 torch 2.11+는 Python 3.12 ABI로 빌드됨.
  3.11 venv는 `undefined symbol: ncclDevCommDestroy`로 실패 — `libtorch_cuda.so`가
  3.11 ABI NCCL wheel과 연결되지 못함. `.python-version`이 3.12를 pin.
- **NCCL 2.27.5 → 2.29.7.** torch 2.11이 `ncclDevCommDestroy`을 참조하는데,
  이건 NCCL ≥ 2.29에만 존재. `pyproject.toml`에 `nvidia-nccl-cu12==2.29.7`로 pin.
- **torchvision 0.24 → 0.26.** torch 2.11은 torchvision 0.26과 짝을 이룸.
  불일치는 PEFT의 lazy `BloomPreTrainedModel` import 깊은 곳에서
  `operator torchvision::nms does not exist`로 나타난다.
- **nvidia-nvshmem-cu13 3.4.5.** vLLM 0.14가 `libnvshmem_host.so.3`을 동적 로드;
  `--no-deps` 설치로는 들어오지 않는다.
- **RL extras를 pyproject.toml에서 분리.** `verl`, `vllm`, `deepspeed`, `ray`는
  `[project.dependencies]`에 들어갈 수 없다 — torch/torchvision/numpy/transformers
  메이저를 서로 다르게 pin해서. 이제 `requirements.rl.txt`(네 라이브러리,
  `--no-deps`로 설치)와 `requirements.rl.runtime.txt`(`--no-deps`가 건너뛰는
  runtime deps: nvshmem, msgpack, omegaconf, hydra, cpuinfo, joblib, hjson,
  einops, nvitop, pyarrow, tensordict, tensorboardX, codetiming, pylatexenc,
  dill, pybind11)로 나뉘었다.
- **`PYTHONPATH` 보호.** 두 런치 스크립트(`launch_rl.sh`, `launch_sft.sh`) 모두
  `cd "$REPO_ROOT"` 후 `unset PYTHONPATH`를 한다. 호스트의
  `/home/work/.local/lib/python3.12/site-packages`가 venv를 가리는 것을 막기 위해서
  (호스트에 다른 ABI의 dev torch가 있음).

### GPU 6,7 롤아웃 검증 중 발견된 추가 이슈 (2026-08-02)

이전 섹션의 설정은 `torch.cuda.is_available()`이 True를 반환하는 것까지만 확인하고
마친 상태였다. 실제 284B 모델 로드를 시도하자 다음 4개의 additional 이슈가 순차적으로
드러났다. 모두 고쳐서 end-to-end 검증 완료.

- **torch 2.11.0+cu130 → 2.11.0+cu128.** PyPI 기본 `torch==2.11.0`은 CUDA 13
  빌드다. 호스트 driver가 570.86.10 = CUDA 12.9라 `torch.cuda.is_available()`은
  True를 주지만 첫 kernel op에서 `RuntimeError: NVIDIA driver too old
  (found version 12090)`로 죽는다. `[tool.uv.sources]`로 PyTorch cu128 index를
  torch/torchvision/torchaudio에 물려서 `+cu128` local-version wheel을 강제.
- **vLLM 0.25 PyPI 기본 → 0.25.0+cu129 (GitHub release).** PyPI의 기본 vllm
  wheel은 cu130 빌드라 같은 이유로 worker init이 죽는다. GitHub release에 있는
  `vllm-0.25.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl`을 `--no-deps`로 받아서
  설치. `requirements.rl.txt`가 직접 GitHub URL을 가리킨다.
- **NCCL 2.29.7 → 2.28.9 (자동).** torch+cu128이 의존성으로 당김. 더 이상
  `pyproject.toml`에 명시적 pin 불필요. 제거.
- **NCCL wheel file 유실.** uv가 hardlink mode로 install할 때 같은 filesystem
  이슈로 400 MB `libnccl.so.2`가 누락됐다. `UV_LINK_MODE=copy uv pip install
  --reinstall nvidia-nccl-cu12`로 복구.
- **`dtype="fp8"` reject.** vLLM 0.25의 `LLM(dtype=...)` Literal이
  `auto/half/float16/bfloat16/float/float32`만 받는다. FP8은 quantization이라
  dtype이 아니다. `ModelConfig.rollout_dtype` 기본값을 `"auto"`로 바꿨다 (model
  config의 `quantization_config.quant_method=fp8`을 vLLM이 자동 인식).
- **CUDA forked subprocess.** vLLM의 import 자체가 CUDA를 건드려서, 이후 TP
  worker를 fork할 때 `Cannot re-initialize CUDA in forked subprocess` 발생.
  `validate_rollout.py`가 `VLLM_WORKER_MULTIPROC_METHOD=spawn`을 default로 set.
- **NamespaceTool import.** vLLM 0.25의 `tool_parsers/utils.py`가
  `from openai.types.responses import NamespaceTool`를 하는데, openai 2.51/2.52
  stable에는 이 심볼이 없다. venv-local patch: `try/except ImportError`로
  FunctionTool에 alias (`.venv/.../vllm/tool_parsers/utils.py`). 재설치시 매번
  다시 적용해야 함 — 자동화는 TODO.
- **vLLM runtime deps 누락.** `--no-deps` install이 torch/cuda 외의 모든 dep를
  건너뛴다. `pyzmq`, `prometheus-fastapi-instrumentator`, `openai-harmony`,
  `llguidance`, `xgrammar`, `partial-json-parser`, `compressed-tensors`,
  `lm-format-enforcer`, `mistral-common[image]`, `opencv-python-headless`,
  `fastsafetensors`, `flashinfer-python`, `flashinfer-cubin`,
  `model-hosting-container-standards`, `apache-tvm-ffi`, `tilelang`,
  `tokenspeed-mla`, `humming-kernels[cu12]`, `quack-kernels`, `depyf`,
  `numba`, `outlines-core`, `diskcache`, `lark`, `nvtx`, `pynvvideocodec`,
  `torchcodec`, `nvidia-cudnn-frontend`, `nvidia-cutlass-dsl`, `cbor2`,
  `ijson`, `pybase64`, `blake3`, `apache-tvm-ffi` 등. 수동으로 install.
- **`PYTHONNOUSERSITE`/`-s`/`-E` 불충분.** 호스트의 `PYTHONPATH` env var가
  `~/.local/lib/python3.12/site-packages`를 prepend한다. 이게 user-site
  mechanism이 아니라서 `-s`/`-E` 외에는 `env -u PYTHONPATH`로 unset해야 한다.
  `validate_rollout.sh` wrapper가 이걸 처리 (launch_rl.sh/sft.sh는 이미 처리 중).

### Run 14까지 롤아웃 검증 중 추가 패치 (2026-08-02)

`validate_rollout.py`가 vLLM init → sample → parse까지 가는 동안 추가로 6개
이슈가 드러났다. 모두 `training_local/patch_vllm.py`가 자동화했고 (vLLM 재설치
시마다 `python -m training_local.patch_vllm`로 재적용), `encoding.py` wrapper가
나머지를 처리한다.

- **tilelang 0.1.9 → 0.1.11 + apache-tvm-ffi 0.1.10 pin.** tilelang 0.1.9 stub
  library(`libcudart_stub.so` 등)가 `cudaDeviceReset` 같은 기본 symbol 빠져 있어,
  vLLM worker가 `tilelang is required for mhc but is not installed`으로 죽음.
  0.1.11로 올리고 stub을 실제 CUDA lib로 symlink. 단 apache-tvm-ffi는 0.1.13+가
  `TypeAttr '__ffi_repr__' is already registered` double-registration 에러를 내서
  0.1.10으로 pin.
- **MHC `forward_cuda` tilelang gate.** vLLM 0.25의 MHC op가 `forward_cuda`에서
  tilelang을 hardcode. tilelang이 깨졌을 때 (`HAS_TILELANG_MHC=False`) 자동으로
  `forward_native`(torch/triton fallback)로 빠지도록 4개 method에 gate 추가.
- **`cute.core.ThrMma` back-compat alias.** vLLM 0.25의 bundled
  `vllm_flash_attn/cute/*` 와 `third_party/fmha_sm100/cute/*` 가 function
  annotation으로 `cute.core.ThrMma`를 참조. `from __future__ import annotations`
  이 없어서 Python이 annotation을 eager evaluate → cutlass-dsl 4.6.x는
  `cute.ThrMma`로만 노출이라 `AttributeError`. `cutlass/cute/__init__.py` 끝에
  `core.ThrMma = ThrMma` 한 줄 추가로 해결.
- **`nvvm.fmax` old/new API 갈라진 branch 강제 skip.** vLLM의 cute utils가
  `CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9` 일 때 old API
  (positional 3개)로 갈라지는데, cutlass-dsl 4.6.1의 새 MLIR op는 positional
  2개만 받는다. 호스트 driver가 12.9라 이 branch가 활성화돼서
  `TypeError: fmax() takes 2 positional arguments but 3` 발생. 두 파일의
  conditional을 `if False:`로 교체해서 new-API else branch 강제.
- **`parse_completion` EOS 자동 보강.** vLLM이 `RequestOutput.text`에서 EOS를
  떼어냄(`finish_reason='stop'`이어도). DSv4 parser는 text 끝에 EOS sentinel이
  있어야 파싱이 됨. wrapper가 끝에 `<｜end▁of▁sentence｜>`가 없으면 붙여준다.
- **`parse_completion` 다중 `</think>` 복구.** DSv4 parser는 첫 `</think>`를
  thinking block의 끝으로 쓰고, 두 번째부터는 content 안의 special token으로
  간주해 reject. 모델이 reasoning_effort="high"일 때 "Absolute maximum" 프롬프트가
  물려서 가끔 `</think>`를 여러 번 emit. wrapper가 thinking 모드일 때 첫 번째
  `</think>` 이후 영역의 추가 `</think>`를 strip해서 parse를 안정화.
- **Smoke test `REASONING_EFFORT=low`.** 모델 snapshot의 encoding이
  `reasoning_effort="high"`를 "Reasoning Effort: Absolute maximum..." 프롬프트로
  매핑하는데, smoke test 검증 목적상 over-thinking보다 깔끔한 structured output가
  더 유용. `validate_rollout.sh`가 `REASONING_EFFORT` default를 `low`로.



