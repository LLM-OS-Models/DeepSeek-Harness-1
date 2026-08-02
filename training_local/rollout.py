"""Multi-turn rollout driver.

Orchestrates LocalSearchEnv + DeepSeekPolicyInferenceModel for one full search
episode. This is the rollout primitive consumed by both verl-based training
(verl/workers/rollout) and the standalone TRL-based fallback trainer.

Per Sid-1 (TI/TO):
  - Each turn re-encodes the full message history. We never reconstruct
    messages from parsed dicts across turns — we only append the new assistant
    turn and the new tool results.

Per KARL (compression):
  - When message history exceeds char threshold, compress older turns
    (model summarizes own history). Disabled by default; can be enabled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from training_local.config import RLConfig
from training_local.encoding import DeepSeekEncoding
from training_local.env import EnvState, LocalSearchEnv
from training_local.rewards import RewardBreakdown

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    """One model turn within an episode."""

    turn_idx: int
    prompt: str
    prompt_token_ids: List[int]
    completion_text: str
    completion_token_ids: List[int]
    logprobs: Optional[List[float]]
    finish_reason: str
    parsed_action: Dict[str, Any]


@dataclass
class EpisodeResult:
    """Full result of one multi-turn search episode."""

    query_id: str
    query_text: str
    turns: List[TurnRecord] = field(default_factory=list)
    final_state: Optional[EnvState] = None
    reward: Optional[RewardBreakdown] = None
    error: Optional[str] = None

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def is_success(self) -> bool:
        return self.error is None and self.reward is not None


def run_episode(
    env: LocalSearchEnv,
    policy,
    query,
    max_turns: int,
    encoding: Optional[DeepSeekEncoding] = None,
    config: Optional[RLConfig] = None,
) -> EpisodeResult:
    """Run one full search episode.

    Args:
        env: LocalSearchEnv instance.
        policy: object with .sample(prompt) -> List[Completion] (e.g.,
            DeepSeekPolicyInferenceModel, or a verl-wrapped rollout).
        query: QueryRecord with query_text, gold_doc_ids, fa_doc_ids.
        max_turns: max number of turns.
        encoding: optional DeepSeekEncoding for parsing completions.
        config: optional RLConfig (for compression settings).

    Returns:
        EpisodeResult with all turns + terminal reward.
    """
    result = EpisodeResult(query_id=query.query_id, query_text=query.query_text)
    encoding = encoding or env.encoding

    state = env.reset(
        query_id=query.query_id,
        query_text=query.query_text,
        gold_ids=query.gold_doc_ids,
        fa_gold_ids=query.fa_doc_ids,
    )

    for turn_idx in range(max_turns):
        try:
            prompt = env.render_prompt(state)
        except Exception as e:
            result.error = f"Turn {turn_idx} prompt render failed: {e}"
            break

        try:
            completions = policy.sample(prompt, n=1)
            if not completions:
                result.error = f"Turn {turn_idx}: no completions returned"
                break
            completion = completions[0]
        except Exception as e:
            result.error = f"Turn {turn_idx} sampling failed: {e}"
            break

        try:
            parsed = encoding.parse_completion(completion.text)
        except Exception as e:
            parsed = {"role": "assistant", "content": completion.text, "tool_calls": []}
            logger.warning(
                "Turn %d: parse failed (%s). Treating as plain content.", turn_idx, e
            )

        turn_record = TurnRecord(
            turn_idx=turn_idx,
            prompt=prompt,
            prompt_token_ids=completion.prompt_token_ids or [],
            completion_text=completion.text,
            completion_token_ids=completion.token_ids,
            logprobs=completion.logprobs,
            finish_reason=completion.finish_reason,
            parsed_action=parsed,
        )
        result.turns.append(turn_record)

        try:
            state, _, done, _ = env.step(state, parsed)
        except Exception as e:
            result.error = f"Turn {turn_idx} env.step failed: {e}"
            break

        if done:
            break

    result.final_state = state
    try:
        result.reward = env.compute_terminal_reward(state)
    except Exception as e:
        result.error = f"Reward computation failed: {e}"

    return result


def run_group(
    env: LocalSearchEnv,
    policy,
    query,
    group_size: int,
    max_turns: int,
    encoding: Optional[DeepSeekEncoding] = None,
    config: Optional[RLConfig] = None,
) -> List[EpisodeResult]:
    """Run a group of episodes for the same query (GRPO group)."""
    return [
        run_episode(env, policy, query, max_turns, encoding, config)
        for _ in range(group_size)
    ]


def collect_group_rewards(group: List[EpisodeResult]) -> List[float]:
    """Extract scalar rewards from a group of episodes (for advantage computation)."""
    return [
        ep.reward.total if (ep.reward is not None) else 0.0
        for ep in group
    ]
