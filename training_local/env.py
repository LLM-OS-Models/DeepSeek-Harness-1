"""Local multi-turn search environment for harness-1 RL.

This is the DeepSeek-V4-Flash-compatible replacement for SlidingWindowSearchEnv.
It reuses the harness WorkingMemory, ToolSet, and reward primitives but uses
DeepSeek encoding instead of Harmony, and structured actions instead of Tinker
ModelInput tokens.

Key design decisions (informed by Sid-1 + KARL):
  - TI/TO protocol: maintain conversation as a list of messages, encode the
    full history each turn. Never re-tokenize parsed messages.
  - Compression: when context exceeds char threshold, summarize older turns
    (KARL pattern).
  - Sparse terminal reward: 4-component recall computed only at end of episode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from training_local.config import RLConfig
from training_local.encoding import DeepSeekEncoding
from training_local.rewards import (
    RewardBreakdown,
    compute_terminal_reward,
)
from training_local.tools_adapter import (
    ToolCallResult,
    execute_openai_tool_call,
    toolset_to_openai_tools,
)


@dataclass
class EnvState:
    """Snapshot of env state for one episode."""

    query_id: str
    query_text: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    n_turns: int = 0
    is_terminal: bool = False
    terminal_reason: str = ""
    curated_ids: List[str] = field(default_factory=list)
    pool_ids: List[str] = field(default_factory=list)
    tool_calls_made: Dict[str, int] = field(default_factory=dict)
    tool_results: List[ToolCallResult] = field(default_factory=list)


class LocalSearchEnv:
    """Multi-turn search environment using DeepSeek-V4 encoding.

    Lifecycle:
        env = LocalSearchEnv(...)
        state = env.reset(query_id, query_text, gold_ids, fa_gold_ids)
        for turn in range(max_turns):
            prompt = env.render_prompt(state)
            model_output = policy.sample(prompt)
            parsed = encoding.parse_completion(model_output)
            state, reward_so_far, done, info = env.step(state, parsed)
            if done:
                break
        reward = env.compute_terminal_reward(state)
    """

    def __init__(
        self,
        toolset,
        encoding: DeepSeekEncoding,
        config: RLConfig,
        dataset=None,
    ):
        self.toolset = toolset
        self.encoding = encoding
        self.config = config
        self.dataset = dataset

        # v8d system prompt from harness
        try:
            from harness.ultra_core import get_system_prompt
            self._get_system_prompt = get_system_prompt
        except ImportError:
            self._get_system_prompt = lambda q: (
                "You are a search agent. Use the provided tools to find and "
                "curate documents that answer the user's query."
            )

    def reset(
        self,
        query_id: str,
        query_text: str,
        gold_ids: Optional[Set[str]] = None,
        fa_gold_ids: Optional[Set[str]] = None,
    ) -> EnvState:
        """Initialize environment for a new query."""
        state = EnvState(query_id=query_id, query_text=query_text)
        state.gold_ids = gold_ids or set()
        state.fa_gold_ids = fa_gold_ids or set()

        system_prompt = self._get_system_prompt(query_text)
        state.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text},
        ]
        return state

    def render_prompt(self, state: EnvState) -> str:
        """Encode the current message history into a DeepSeek prompt string."""
        tools_schema = toolset_to_openai_tools(self.toolset)
        messages_with_tools = [
            {**state.messages[0], "tools": tools_schema}
            if i == 0
            else m
            for i, m in enumerate(state.messages)
        ]
        return self.encoding.encode_messages(messages_with_tools)

    def step(
        self,
        state: EnvState,
        parsed_action: Dict[str, Any],
    ) -> Tuple[EnvState, float, bool, Dict[str, Any]]:
        """Apply one model action (already parsed) to the environment.

        Args:
            state: current EnvState.
            parsed_action: dict from DeepSeekEncoding.parse_completion with keys
                role, content, reasoning_content, tool_calls.

        Returns:
            (new_state, intermediate_reward, done, info)
        """
        # Append the assistant turn to history (TI/TO: preserve exact content)
        assistant_msg = {
            "role": "assistant",
            "content": parsed_action.get("content", ""),
            "reasoning_content": parsed_action.get("reasoning_content", ""),
            "tool_calls": parsed_action.get("tool_calls", []),
        }
        state.messages.append(assistant_msg)
        state.n_turns += 1

        tool_calls = parsed_action.get("tool_calls", [])
        info: Dict[str, Any] = {"tool_results": []}

        if not tool_calls:
            # No tool calls = either final answer or end_search via text
            state.is_terminal = True
            state.terminal_reason = "no_tool_calls"
            return state, 0.0, True, info

        # Execute each tool call
        for tc in tool_calls:
            result = execute_openai_tool_call(self.toolset, tc)
            state.tool_results.append(result)
            state.tool_calls_made[result.name] = state.tool_calls_made.get(result.name, 0) + 1
            info["tool_results"].append({
                "name": result.name,
                "success": result.success,
                "error": result.error,
            })

            if result.name == "end_search":
                state.is_terminal = True
                state.terminal_reason = "end_search_tool"
                break

            if result.name == "curate":
                _update_curated_from_args(state, result.arguments)

            # Track pool IDs from search results
            if result.metadata and hasattr(result.metadata, "returned_chunk_ids"):
                for did in result.metadata.returned_chunk_ids or []:
                    if did not in state.pool_ids:
                        state.pool_ids.append(did)

            # Append tool result as a tool message
            state.messages.append({
                "role": "tool",
                "name": result.name,
                "content": result.output if result.success else f"Error: {result.error}",
                "tool_call_id": tc.get("id", ""),
            })

        # Check max turns
        if state.n_turns >= self.config.max_turns_end:
            state.is_terminal = True
            state.terminal_reason = "max_turns"

        return state, 0.0, state.is_terminal, info

    def compute_terminal_reward(self, state: EnvState) -> RewardBreakdown:
        """Compute the terminal 4-component + NDCG reward."""
        return compute_terminal_reward(
            curated_ids=state.curated_ids,
            pool_ids=state.pool_ids,
            gold_ids=getattr(state, "gold_ids", set()),
            final_answer_gold_ids=getattr(state, "fa_gold_ids", set()),
            n_turns=state.n_turns,
            tool_calls_made=state.tool_calls_made,
            config=self.config,
        )


def _update_curated_from_args(state: EnvState, args: Dict[str, Any]) -> None:
    """Update curated_ids from curate tool arguments."""
    add_ids = args.get("add_ids", []) or []
    remove_ids = args.get("remove_ids", []) or []
    existing = set(state.curated_ids)
    existing.update(add_ids)
    existing.difference_update(remove_ids)
    state.curated_ids = sorted(existing)
