from typing import Any, Dict, List, Optional, Union

from typing_extensions import Required, TypedDict


class LLMGatewayMessageParam(TypedDict, total=False):
    """A chat message. Pass as a plain dict: `{"role": "user", "content": "..."}`."""

    role: Required[str]
    '"user", "assistant", "system", or "tool" (a tool call result, paired with `tool_call_id`)'

    content: Optional[Union[str, List[Dict[str, Any]]]]
    tool_calls: List[Dict[str, Any]]
    tool_call_id: str
    name: str
    thinking: str
    cache_control: Dict[str, Any]


class LLMGatewayDecisionQuestionParam(TypedDict, total=False):
    """A decisions question. Pass as a plain dict: `{"type": "predicate", "name": "...", "instructions": "..."}`."""

    type: Required[str]
    '"predicate", "choice" (requires `choices`), or "score" (requires `levels`)'

    name: Required[str]
    "Unique within the request; echoed on the matching answer"

    instructions: Required[Union[str, List[Dict[str, Any]]]]
    choices: List[Dict[str, str]]
    '`[{"value": ..., "description": ...}]`, for "choice" questions'

    levels: List[Dict[str, str]]
    '`[{"label": ..., "description": ...}]`, for "score" questions'
