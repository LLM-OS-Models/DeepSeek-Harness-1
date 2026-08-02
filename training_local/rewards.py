"""Reward computation for harness-1 local RL.

Combines:
  - Original harness-1 4-component recall reward (ultra_core.compute_reward)
  - Sid-1 NDCG-style ranking reward (discourages over-reporting)
  - Tool diversity bonus (anti-reward-hacking)
  - Turn penalty (encourage efficiency)

The reward is computed at the END of each multi-turn rollout. Optional per-turn
shaping uses NDCG of curated set so far.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from training_local.config import RLConfig


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown for logging/debugging."""

    recall: float = 0.0
    trajectory_recall: float = 0.0
    final_answer_recall: float = 0.0
    trajectory_fa_recall: float = 0.0
    ndcg: float = 0.0
    tool_diversity_bonus: float = 0.0
    turn_penalty: float = 0.0
    total: float = 0.0
    metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

    def to_dict(self) -> Dict[str, float]:
        return {
            "reward/total": self.total,
            "reward/recall": self.recall,
            "reward/trajectory_recall": self.trajectory_recall,
            "reward/final_answer_recall": self.final_answer_recall,
            "reward/trajectory_fa_recall": self.trajectory_fa_recall,
            "reward/ndcg": self.ndcg,
            "reward/tool_diversity_bonus": self.tool_diversity_bonus,
            "reward/turn_penalty": self.turn_penalty,
            **{f"metrics/{k}": v for k, v in self.metrics.items()},
        }


def compute_ndcg(
    ranked_doc_ids: List[str],
    gold_ids: Set[str],
    discount_base: int = 2,
) -> float:
    """NDCG with binary relevance (Sid-1 style).

    DCG  = sum_i rel_i / log_discount_base(i+1)  for i = 1..N
    IDCG = same for ideal ranking (all gold first)
    NDCG = DCG / IDCG (0 if no gold or no relevant)

    Discourages over-reporting better than recall because irrelevant items in
    early positions dilute DCG via larger denominator.
    """
    if not gold_ids or not ranked_doc_ids:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids):
        if doc_id in gold_ids:
            dcg += 1.0 / math.log(i + discount_base, discount_base)

    n_gold_in_ranked = min(len(gold_ids), len(ranked_doc_ids))
    idcg = sum(
        1.0 / math.log(i + discount_base, discount_base) for i in range(1, n_gold_in_ranked + 1)
    )
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_tool_diversity_bonus(
    tool_calls_made: Dict[str, int],
    target: int,
    bonus_weight: float,
) -> float:
    """Bonus for using diverse tools (anti-reward-hacking from original harness-1)."""
    if bonus_weight <= 0 or target <= 0:
        return 0.0
    n_distinct = sum(1 for c in tool_calls_made.values() if c > 0)
    return bonus_weight * min(n_distinct / target, 1.0)


def compute_turn_penalty(
    n_turns: int,
    min_turns: int,
    max_penalty: float,
) -> float:
    """Linear penalty after `min_turns` (encourage efficiency)."""
    if max_penalty <= 0 or n_turns <= min_turns:
        return 0.0
    return -max_penalty * (n_turns - min_turns) / max(n_turns, 1)


def compute_terminal_reward(
    curated_ids: List[str],
    pool_ids: List[str],
    gold_ids: Set[str],
    final_answer_gold_ids: Set[str],
    n_turns: int,
    tool_calls_made: Dict[str, int],
    config: RLConfig,
    eval_metrics: Optional[Dict[str, float]] = None,
) -> RewardBreakdown:
    """Compute terminal reward for a finished search episode.

    Args:
        curated_ids: final curated doc IDs (the answer set).
        pool_ids: all doc IDs seen during the trajectory.
        gold_ids: gold relevance doc IDs.
        final_answer_gold_ids: subset of gold tied to the final answer.
        n_turns: number of actions taken.
        tool_calls_made: counter of tool_name -> num_calls.
        config: RL hyperparameters.
        eval_metrics: optional dict of precomputed metrics from dataset evaluator.

    Returns:
        RewardBreakdown with per-component values and total.
    """
    breakdown = RewardBreakdown(metrics=eval_metrics or {})

    curated_set = set(curated_ids)
    pool_set = set(pool_ids)

    # ===== Recall components =====
    if gold_ids:
        recall = len(curated_set & gold_ids) / len(gold_ids)
        trajectory_recall = len(pool_set & gold_ids) / len(gold_ids)
    else:
        recall = trajectory_recall = 0.0

    if final_answer_gold_ids:
        fa_recall = len(curated_set & final_answer_gold_ids) / len(final_answer_gold_ids)
        trajectory_fa_recall = len(pool_set & final_answer_gold_ids) / len(final_answer_gold_ids)
    else:
        fa_recall = trajectory_fa_recall = 0.0

    breakdown.recall = recall
    breakdown.trajectory_recall = trajectory_recall
    breakdown.final_answer_recall = fa_recall
    breakdown.trajectory_fa_recall = trajectory_fa_recall

    # ===== NDCG (Sid-1) =====
    breakdown.ndcg = compute_ndcg(curated_ids, gold_ids)

    # ===== Tool diversity bonus =====
    breakdown.tool_diversity_bonus = compute_tool_diversity_bonus(
        tool_calls_made, config.tool_diversity_target, config.tool_diversity_bonus
    )

    # ===== Turn penalty =====
    breakdown.turn_penalty = compute_turn_penalty(
        n_turns, config.turn_penalty_min_turns, config.turn_penalty_max
    )

    # ===== FA miss penalty (curated missed, but trajectory saw) =====
    if final_answer_gold_ids:
        fa_seen_in_trajectory = pool_set & final_answer_gold_ids
        fa_in_curated = curated_set & final_answer_gold_ids
        fa_missed = fa_seen_in_trajectory - fa_in_curated
        fa_miss_penalty = -config.fa_miss_penalty_weight * (
            len(fa_missed) / len(final_answer_gold_ids)
        )
    else:
        fa_miss_penalty = 0.0

    # ===== Final answer bonus (sparse success) =====
    final_answer_success = config.final_answer_bonus if fa_recall >= 1.0 else 0.0

    # ===== Weighted total =====
    total = (
        config.outcome_weight * recall
        + config.trajectory_recall_weight * trajectory_recall
        + config.final_answer_recall_weight * fa_recall
        + config.trajectory_fa_recall_weight * trajectory_fa_recall
        + config.ndcg_weight * breakdown.ndcg
        + breakdown.tool_diversity_bonus
        + breakdown.turn_penalty
        + fa_miss_penalty
        + final_answer_success
    )

    breakdown.total = total
    breakdown.metrics.update(
        {
            "recall": recall,
            "trajectory_recall": trajectory_recall,
            "final_answer_recall": fa_recall,
            "trajectory_fa_recall": trajectory_fa_recall,
            "ndcg": breakdown.ndcg,
            "n_turns": n_turns,
            "n_curated": len(curated_set),
            "n_pool": len(pool_set),
            "n_gold": len(gold_ids),
            "fa_miss_penalty": fa_miss_penalty,
            "final_answer_success_bonus": final_answer_success,
        }
    )
    return breakdown


def grpo_advantages(
    group_rewards: List[float],
    beta1: float = 1.0,
) -> List[float]:
    """OAPL-style soft-min advantage (KARL arXiv:2603.05218).

    V̂*(x) = β₁ ln(1/G Σ exp(r/β₁))
    A_i   = r_i - V̂*(x)

    Setting β₁ → ∞ recovers standard mean-centered GRPO.
    """
    if not group_rewards:
        return []
    if beta1 <= 0 or math.isinf(beta1):
        mean = sum(group_rewards) / len(group_rewards)
        return [r - mean for r in group_rewards]

    max_r = max(group_rewards)
    exp_sum = sum(math.exp((r - max_r) / beta1) for r in group_rewards)
    log_sum_exp = max_r + beta1 * math.log(exp_sum / len(group_rewards))
    return [r - log_sum_exp for r in group_rewards]
