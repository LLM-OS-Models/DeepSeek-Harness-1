# 아키텍처 — Harness-1 Local

`training_local/` 내 Tinker 없는 로컬 RL 파이프라인의 설계를 다룬다.

## 설계 목표

1. **Harness-1의 핵심 자산을 보존.** stateful 검색 환경(WorkingMemory, 9개 도구,
   4-요소 recall 보상)이 이 연구의 실제 기여다. 그대로 재사용한다. 재작성하지 않는다.
2. **Tinker를 옵션으로.** 모든 학습은 호스팅 서비스 의존성 없이 로컬 GPU에서
   동작해야 한다.
3. **DeepSeek-V4-Flash를 직접 타깃.** native FP8+FP4 양자화, 커스텀 DSML tool-call
   포맷, DSpark speculative decoding을 그대로 쓴다. "일반 HF 모델" 추상화로
   폴백하지 않는다.
4. **발표된 기법을 차용.** Sid-1과 KARL에서 구현 가능한 모든 것을, 충돌이 없는 한
   편입시킨다.

## 컴포넌트 맵

```
┌─────────────────────────────────────────────────────────────────────┐
│                         harness/ (보존됨)                             │
│                                                                      │
│   ultra_core.py    WorkingMemory, compute_reward, system prompt     │
│   tools.py         SearchCorpusTool, GrepCorpusTool,                │
│                    ReadDocumentTool, PruneChunksTool 등              │
│   trajectory.py    Action, Observation, Trajectory                  │
│   rerank.py        BasetenReranker, ContextualReranker              │
│   agent.py         AgentInferenceModel (OpenAI/Anthropic/Moonshot)  │
│                    [TinkerAgentInferenceModel 제거됨]                │
└─────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ 사용
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                      training_local/ (신규)                          │
│                                                                      │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐   │
│   │  encoding    │────►│     env      │◄────│   tools_adapter  │   │
│   │ (DSV4 wrap)  │     │ (LocalSearch │     │ (ToolSet ↔ OA)   │   │
│   └──────────────┘     │  Env)        │     └──────────────────┘   │
│                        └──────┬───────┘                            │
│                               │                                      │
│                               ▼                                      │
│                        ┌──────────────┐                            │
│                        │   rewards    │                            │
│                        │ (4-요소 +     │                            │
│                        │  NDCG + OAPL)│                            │
│                        └──────────────┘                            │
│                               ▲                                      │
│                               │                                      │
│   ┌──────────────┐     ┌──────┴───────┐     ┌──────────────────┐   │
│   │   rollout    │────►│ train_rl.py  │────►│    backends/     │   │
│   │ (episode     │     │  (진입점)     │     │ verl_runner /    │   │
│   │  driver)     │     └──────────────┘     │ trl_runner       │   │
│   └──────────────┘                          └──────────────────┘   │
│         ▲                                                       │   │
│         │                                                       ▼   │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐   │
│   │    agent     │◄────│    config    │     │   checkpoint     │   │
│   │ (vLLM 정책)   │     │ (dataclass)  │     │ (LoRA 저장/로드)  │   │
│   └──────────────┘     └──────────────┘     └──────────────────┘   │
│         ▲                                                       │   │
│         │                                                       ▼   │
│   ┌──────────────┐                                     [LoRA adapter] │
│   │     data     │                                                   │
│   │ (SearchDS    │                                                   │
│   │  adapter)    │                                                   │
│   └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 학습 step당 데이터 흐름

```
1. SearchDataset에서 G개 쿼리를 샘플링
   │
   ▼
2. 각 쿼리마다 DeepSeekEncoding으로 초기 prompt 렌더
   │
   ▼
3. 각 (쿼리, group_index)마다 multi-turn episode 실행:
   a. env.render_prompt(state) → prompt 문자열
   b. policy.sample(prompt) → completion (vLLM FP8 + DSpark)
   c. encoding.parse_completion(completion) → {content, reasoning, tool_calls}
   d. env.step(state, parsed) → 새 state + done
   e. done이거나 max_turns까지 반복
   │
   ▼
4. env.compute_terminal_reward(state) → RewardBreakdown
   │
   ▼
5. 쿼리당 G개 보상을 그룹화, OAPL로 GRPO advantage 계산
   │
   ▼
6. (prompt_token_ids, completion_token_ids, advantages, logprobs)로 PPO 업데이트
   │
   ▼
7. N step마다 LoRA adapter 저장
```

## 인코딩 경계

DeepSeek-V4-Flash의 `encoding_dsv4.py`는 모델 스냅샷에 포함된 self-contained
Python 모듈이다. 동적으로 로드한다:

```python
encoding = DeepSeekEncoding(
    model_path="deepseek-ai/DeepSeek-V4-Flash-0731",
    thinking_mode="thinking",
    reasoning_effort="high",
)
prompt = encoding.encode_messages(messages)
parsed = encoding.parse_completion(model_output_text)
```

`parsed`의 형태는 `{role, content, reasoning_content, tool_calls}`이며,
`tool_calls`는 OpenAI 포맷이다. `env.step()`이 이걸 소비한다.

## 보상 계산

`compute_terminal_reward`는 가중 합산된 모든 구성요소를 담은 `RewardBreakdown`을
반환한다. 기본 가중치(`RLConfig`에 정의)는 원본 Harness-1 설정 + Sid-1 NDCG 0.2를
반영한다.

KARL에 따라 보상은 EPISODE 단위로(sparse) 계산된다. 턴 단위 shaping(현재까지의
큐레이션 셋에 대한 NDCG 등)은 튜닝 가능하지만 근시안적 행동을 유발할 수 있어
기본값은 꺼져 있다.

## GRPO + OAPL

`grpo_advantages(group_rewards, beta1)`가 그룹 멤버당 하나씩 advantage 리스트를
반환한다. `beta1 = ∞`면 표준 mean-centering, 유한한 `beta1`이면 KARL OAPL의
soft-min이 된다.

```
V̂*(x) = β₁ · ln( (1/G) Σᵢ exp(rᵢ / β₁) )
Aᵢ    = rᵢ - V̂*(x)
```

verl에서는 `algorithm.advantage_estimator = "grpo"` 슬롯에 커스텀 kwargs로
연결된다. TRL에서는 트레이너 내부 GRPO가 처리하며, 우리의 OAPL advantage는
독자적인 폴백 경로에서만 사용된다.

## multi-turn 롤아웃 프로토콜

Sid-1 TI/TO에 따라, 매 턴마다 구조화된 `messages` 리스트 전체에서 FULL 대화
히스토리를 다시 인코딩한다. 모델 출력을 파싱해서 메시지로 재조립 후 다시 렌더하는
패턴은 tool call 경계 토큰에서 log-probability 붕괴를 일으킨다.

구체적으로, `env.render_prompt(state)`는 `encoding.encode_messages(state.messages)`를
호출한다. 샘플링 후 어시스턴트 턴은 그대로 append된다:

```python
state.messages.append({
    "role": "assistant",
    "content": parsed["content"],
    "reasoning_content": parsed["reasoning_content"],
    "tool_calls": parsed["tool_calls"],
})
```

도구 결과는 `{"role": "tool", "name": ..., "content": ..., "tool_call_id": ...}`로
append된다.

## 압축 (KARL)

`state.messages`가 `compression_char_threshold`(기본 150K 자, KARL의 BrowseComp+
설정)를 넘으면 에이전트가 자신의 히스토리를 요약한다. 이것은 정책의 나머지 부분과
함께 end-to-end로 학습된다. 압축 단계도 RL 최적화에 포함된다(압축 요약이
gradient 계산에 참여).

현재는 별도 학습 스케줄이 필요해 `RLConfig.enable_compression`에서 기본 꺼짐.
로드맵 항목.

## 백엔드

### verl (주 백엔드)

```
Ray 클러스터 (자동 초기화, 보이는 모든 GPU 사용)
  ├── Actor worker (Megatron 또는 FSDP, BF16, LoRA rank 32)
  ├── Rollout worker (vLLM, FP8, DSpark 7-token speculation)
  ├── Reference worker (FSDP, frozen)
  └── Reward driver (커스텀: LocalSearchEnv + 4-요소 보상)
```

verl은 공식 DeepSeek-V4 연동을 갖추고 있다:
- `verl/workers/rollout/vllm_rollout/utils.py`가 MXFP4 가중치 동기화 처리
- R2/R3 router replay: actor와 rollout 사이의 expert routing 일관성

우리의 기여는 `LocalSearchEnv`를 감싸서 에피소드당 `RewardBreakdown.total`을
스칼라 보상으로 반환하는 **보상 함수**다.

### TRL (폴백)

TRL의 `GRPOTrainer`는 multi-turn stateful 환경을 네이티브로 지원하지 않는다.
폴백은 한 completion을 한 턴으로 취급해 부분 보상을 계산한다. 용도:
- 보상 함수 smoke 테스트
- 단일 GPU 디버깅
- LoRA 학습 자체가 수렴하는지 검증

실제 multi-turn RL에는 verl을 쓴다.

## 체크포인트 전략

세 가지 타입:

1. **LoRA 어댑터만** (작음, ~50 MB): `save_lora_adapter(model, dir, step)`
2. **병합된 full 모델** (큼, ~157 GB): `save_merged_model(model, tok, dir)`
3. **학습 상태** (재개용): `save_training_state(dir, step, opt, sched, rng)`

H200 ×8에서는 50 step마다 저장. 어댑터만 저장하는 건 저렴; 병합은 학습 종료 시
(또는 평가 시)만.

## H200 ×8에서의 메모리 예산

```
8 × H200 = 1,141 GB 총 HBM3e

카드당 할당 (각 143 GB):
  - 모델 가중치 (FP8):    ~20 GB  (157 GB / 8)
  - LoRA 어댑터 (BF16):   <1 GB
  - Optimizer state (Adam): ~3 GB  (2× LoRA 파라미터 × 4 bytes)
  - Activation:            ~30 GB (gradient checkpointing 켠 상태)
  - KV cache (FP8):         ~80 GB (긴 컨텍스트)
                            ─────
                            ~134 GB ← 143 GB 안에 들어옴
```

여유가 더 필요하면 `ROLLOUT_GPU_MEM_UTIL`을 조정한다.
