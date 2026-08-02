"""Adapter bridging harness ToolSet to verl/standard tool-calling format.

The harness ToolSet is model-agnostic but uses typed Action/Observation classes.
verl and most modern RL frameworks use OpenAI-compatible tool-call schema.
This module converts between the two.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from harness.tools import (
    Tool,
    ToolCallMetadata,
    ToolSet,
)


def tool_to_openai_schema(tool: Tool) -> Dict[str, Any]:
    """Convert a harness Tool to OpenAI function schema."""
    params = tool.parameters() if hasattr(tool, "parameters") else {}
    properties = {}
    required = []
    for pname, pschema in params.items():
        properties[pname] = {
            "type": pschema.get("type", "string"),
            "description": pschema.get("description", ""),
        }
        if pschema.get("required", False):
            required.append(pname)

    return {
        "type": "function",
        "function": {
            "name": tool.name(),
            "description": tool.description() if hasattr(tool, "description") else "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def toolset_to_openai_tools(toolset: ToolSet) -> List[Dict[str, Any]]:
    """Convert all tools in a ToolSet to OpenAI tool list."""
    schemas = []
    for tool in toolset.tools():
        try:
            schemas.append(tool_to_openai_schema(tool))
        except Exception:
            continue
    return schemas


@dataclass
class ToolCallResult:
    """Result of executing a single tool call."""

    name: str
    arguments: Dict[str, Any]
    success: bool
    output: str
    metadata: Optional[ToolCallMetadata] = None
    error: Optional[str] = None


def execute_openai_tool_call(
    toolset: ToolSet,
    tool_call: Dict[str, Any],
) -> ToolCallResult:
    """Execute an OpenAI-format tool call against a harness ToolSet.

    Args:
        toolset: harness ToolSet instance.
        tool_call: {id, type: "function", function: {name, arguments}} (JSON string).

    Returns:
        ToolCallResult with serialized output text.
    """
    fn = tool_call.get("function", {}) if "function" in tool_call else tool_call
    name = fn.get("name", "")
    raw_args = fn.get("arguments", "{}")

    try:
        if isinstance(raw_args, str):
            args = json.loads(raw_args) if raw_args else {}
        else:
            args = raw_args or {}
    except json.JSONDecodeError as e:
        return ToolCallResult(
            name=name,
            arguments={},
            success=False,
            output="",
            error=f"Invalid JSON arguments: {e}",
        )

    tool = toolset.get_tool(name) if hasattr(toolset, "get_tool") else None
    if tool is None:
        for t in toolset.tools():
            if t.name() == name:
                tool = t
                break

    if tool is None:
        return ToolCallResult(
            name=name,
            arguments=args,
            success=False,
            output="",
            error=f"Unknown tool: {name}",
        )

    try:
        result = tool(**args) if hasattr(tool, "__call__") else None
        output_text = _serialize_tool_result(result)
        metadata = result.metadata if hasattr(result, "metadata") else None
        return ToolCallResult(
            name=name,
            arguments=args,
            success=True,
            output=output_text,
            metadata=metadata,
        )
    except Exception as e:
        return ToolCallResult(
            name=name,
            arguments=args,
            success=False,
            output="",
            error=str(e),
        )


def _serialize_tool_result(result: Any, max_chars: int = 8000) -> str:
    """Serialize a tool result into a string for the model.

    Truncates very long outputs to keep context budget manageable.
    """
    if result is None:
        return ""
    if isinstance(result, str):
        text = result
    elif hasattr(result, "to_prompt_str"):
        text = result.to_prompt_str()
    elif hasattr(result, "text"):
        text = result.text
    else:
        try:
            text = json.dumps(result.__dict__, default=str, indent=2)
        except Exception:
            text = str(result)

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated, {len(text) - max_chars} chars omitted]"
    return text
