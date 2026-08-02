"""Smoke test — verifies the training_local pipeline imports and basic
config validation work WITHOUT a GPU.

Run:
    python -m training_local.smoke_test
"""

from __future__ import annotations

import sys
from pathlib import Path


def test_imports():
    """All training_local modules should import cleanly."""
    print("[1/5] Testing imports...")
    from training_local import config, encoding, rewards, tools_adapter
    from training_local import env, agent, data, checkpoint, rollout
    from training_local import train_rl, train_sft
    print("  ✓ All training_local modules imported")


def test_config():
    """Default config should load with no env vars."""
    print("[2/5] Testing config...")
    from training_local.config import RLConfig, SFTConfig, ModelConfig, LoRAConfig
    rl = RLConfig()
    sft = SFTConfig()
    assert rl.group_size > 0
    assert rl.batch_size > 0
    assert sft.batch_size > 0
    assert rl.model.model_path == "deepseek-ai/DeepSeek-V4-Flash-0731"
    print(f"  ✓ RLConfig: group_size={rl.group_size}, lr={rl.learning_rate}")
    print(f"  ✓ SFTConfig: epochs={sft.num_epochs}, lr={sft.learning_rate}")


def test_encoding_module_resolution():
    """Encoding module should resolve to the downloaded DeepSeek snapshot."""
    print("[3/5] Testing encoding module resolution...")
    try:
        from training_local.encoding import load_encoding_module
        mod = load_encoding_module("deepseek-ai/DeepSeek-V4-Flash-0731")
        assert hasattr(mod, "encode_messages")
        assert hasattr(mod, "parse_message_from_completion_text")
        print("  ✓ encoding_dsv4.py loaded from snapshot")
    except FileNotFoundError as e:
        print(f"  ⚠ Encoding not found: {e}")
        print("    (Run download first: hf download deepseek-ai/DeepSeek-V4-Flash-0731)")


def test_rewards():
    """Reward computation should work on synthetic inputs."""
    print("[4/5] Testing rewards...")
    from training_local.rewards import (
        compute_ndcg,
        compute_terminal_reward,
        grpo_advantages,
    )
    from training_local.config import RLConfig

    # NDCG with perfect ranking at top
    ndcg_perfect = compute_ndcg(["a", "b", "c"], {"a", "b"})
    assert ndcg_perfect > 0.5, f"NDCG should be high for perfect ranking: {ndcg_perfect}"

    # NDCG with over-reporting (relevant items late)
    ndcg_late = compute_ndcg(["x", "y", "z", "a", "b"], {"a", "b"})
    assert ndcg_late < ndcg_perfect, "Late-relevant NDCG should be lower"

    # Terminal reward
    cfg = RLConfig()
    bd = compute_terminal_reward(
        curated_ids=["a", "b"],
        pool_ids=["a", "b", "c"],
        gold_ids={"a", "b"},
        final_answer_gold_ids={"a"},
        n_turns=10,
        tool_calls_made={"search_corpus": 3, "read_document": 2},
        config=cfg,
    )
    assert bd.total > 0, f"Reward should be positive for finding all gold: {bd.total}"
    print(f"  ✓ NDCG (perfect): {ndcg_perfect:.3f}")
    print(f"  ✓ Terminal reward: {bd.total:.3f}")

    # GRPO advantages
    advs = grpo_advantages([1.0, 0.5, 0.8, 0.2])
    assert len(advs) == 4
    assert max(advs) > 0 and min(advs) < 0
    print(f"  ✓ GRPO advantages: {[round(a, 3) for a in advs]}")


def test_data_structure():
    """QueryRecord and basic data adapters should be constructable."""
    print("[5/5] Testing data structures...")
    from training_local.data import QueryRecord
    q = QueryRecord(
        query_id="test_q1",
        query_text="What is X?",
        dataset_name="test",
        gold_doc_ids={"doc1", "doc2"},
        fa_doc_ids={"doc1"},
    )
    assert len(q.gold_doc_ids) == 2

    from training_local.env import LocalSearchEnv, EnvState
    state = EnvState(query_id="test", query_text="test")
    assert state.n_turns == 0
    assert not state.is_terminal
    print("  ✓ QueryRecord constructable")
    print("  ✓ EnvState constructable")


def main():
    print("=" * 60)
    print("harness-1 local — smoke test")
    print("=" * 60)
    try:
        test_imports()
        test_config()
        test_encoding_module_resolution()
        test_rewards()
        test_data_structure()
        print()
        print("✓ All smoke tests passed.")
        return 0
    except Exception as e:
        print()
        print(f"✗ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
