"""Data helpers for supervised soft-prefix tuning."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any

from skillopt.envs.docvqa.rollout import _build_messages as _build_docvqa_messages
from skillopt.envs.livemathematicianbench.rollout import _build_system as _build_livemath_system
from skillopt.envs.livemathematicianbench.rollout import _build_user as _build_livemath_user
from skillopt.envs.officeqa.rollout import _TOOL_SCHEMAS as OFFICEQA_TOOL_SCHEMAS
from skillopt.envs.officeqa.rollout import _build_system as _build_officeqa_system
from skillopt.envs.officeqa.rollout import _build_user as _build_officeqa_user
from skillopt.envs.searchqa.rollout import _build_system as _build_searchqa_system
from skillopt.envs.searchqa.rollout import _build_user as _build_searchqa_user
from skillopt.prompts import load_prompt

_SKILL_SECTION_PLACEHOLDER = "{skill_section}"
_SOFT_PREFIX_INSERT_MARKER = "<|skillopt_soft_prefix_insert|>"


def normalize_prefix_injection_position(value: str | None) -> str:
    raw = str(value or "prompt_start").strip().lower()
    if raw in {"prompt_start", "start", "prefix", "before_prompt"}:
        return "prompt_start"
    if raw in {"skill_section", "skill-section", "skill_placeholder", "skill"}:
        return "skill_section"
    raise ValueError(
        "soft_prefix.injection_position must be one of prompt_start or skill_section"
    )


def _apply_text_chat_template(
    tokenizer: Any,
    messages: list[dict],
    *,
    enable_thinking: bool = False,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template) and getattr(tokenizer, "chat_template", None):
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        if tools is not None:
            kwargs["tools"] = tools
        try:
            return apply_chat_template(_messages_for_text_chat_template(messages), **kwargs)
        except TypeError:
            # Tokenizer does not support enable_thinking (non-Qwen3 model).
            kwargs.pop("enable_thinking", None)
            return apply_chat_template(_messages_for_text_chat_template(messages), **kwargs)
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().title() or "User"
        content = str(message.get("content", "")).strip()
        rendered.append(f"{role}:\n{content}")
    if add_generation_prompt:
        rendered.append("Assistant:\n")
    return "\n\n".join(rendered)


def _messages_for_text_chat_template(messages: list[dict]) -> list[dict]:
    rendered_messages: list[dict] = []
    for message in messages:
        rendered_message = dict(message)
        tool_calls = rendered_message.get("tool_calls")
        if isinstance(tool_calls, list):
            rendered_message["tool_calls"] = [_tool_call_for_text_chat_template(tool_call) for tool_call in tool_calls]
        rendered_messages.append(rendered_message)
    return rendered_messages


def _tool_call_for_text_chat_template(tool_call: Any) -> Any:
    if not isinstance(tool_call, dict):
        return tool_call
    rendered_tool_call = dict(tool_call)
    function = rendered_tool_call.get("function")
    if isinstance(function, dict):
        rendered_function = dict(function)
        arguments = rendered_function.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = None
            if isinstance(parsed_arguments, dict):
                rendered_function["arguments"] = parsed_arguments
        rendered_tool_call["function"] = rendered_function
    return rendered_tool_call


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "ndim") and int(input_ids.ndim) == 2:
        return int(input_ids.shape[1])
    return len(input_ids)


def _strip_soft_prefix_marker_from_prompt(tokenizer: Any, prompt: str) -> tuple[str, int]:
    marker_counts = {
        _SOFT_PREFIX_INSERT_MARKER: prompt.count(_SOFT_PREFIX_INSERT_MARKER),
        _SKILL_SECTION_PLACEHOLDER: prompt.count(_SKILL_SECTION_PLACEHOLDER),
    }
    marker_total = sum(marker_counts.values())
    if marker_total != 1:
        raise ValueError(
            "soft_prefix.injection_position=skill_section requires exactly one "
            f"{_SKILL_SECTION_PLACEHOLDER} or {_SOFT_PREFIX_INSERT_MARKER} marker"
        )
    marker = (
        _SOFT_PREFIX_INSERT_MARKER
        if marker_counts[_SOFT_PREFIX_INSERT_MARKER]
        else _SKILL_SECTION_PLACEHOLDER
    )
    char_idx = prompt.index(marker)
    clean_prompt = prompt[:char_idx] + prompt[char_idx + len(marker):]
    return clean_prompt, _token_count(tokenizer, prompt[:char_idx])


def _render_messages_and_insert_idx(
    tokenizer: Any,
    messages: list[dict],
    *,
    enable_thinking: bool = False,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
) -> tuple[str, int]:
    rendered = _apply_text_chat_template(
        tokenizer,
        messages,
        enable_thinking=enable_thinking,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
    )
    return _strip_soft_prefix_marker_from_prompt(tokenizer, rendered)


def build_searchqa_prompt(tokenizer: Any, item: dict, *, enable_thinking: bool = False) -> str:
    """Build the same SearchQA prompt shape used by rollout, without a markdown skill.

    ``enable_thinking=False`` (default) passes Qwen3's ``enable_thinking=False`` flag to
    ``apply_chat_template``, which pre-closes the ``<think>`` block so the model answers
    directly.  Using the default (thinking enabled) causes the generation prompt to end
    with an *open* ``<think>`` token; training targets are then concatenated inside the
    unclosed thinking block, which is a malformed format.
    """
    system = _build_searchqa_system("")
    user = _build_searchqa_user(
        question=str(item["question"]),
        context=str(item.get("context", "")),
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _apply_text_chat_template(tokenizer, messages, enable_thinking=enable_thinking)


def build_searchqa_prompt_and_insert_idx(
    tokenizer: Any,
    item: dict,
    *,
    enable_thinking: bool = False,
    injection_position: str = "prompt_start",
) -> tuple[str, int | None]:
    """Build a SearchQA prompt and optional soft-prefix insertion token index."""
    injection_position = normalize_prefix_injection_position(injection_position)
    if injection_position == "prompt_start":
        return build_searchqa_prompt(tokenizer, item, enable_thinking=enable_thinking), None

    system = _build_searchqa_system(_SOFT_PREFIX_INSERT_MARKER)
    user = _build_searchqa_user(
        question=str(item["question"]),
        context=str(item.get("context", "")),
    )
    return _render_messages_and_insert_idx(
        tokenizer,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        enable_thinking=enable_thinking,
    )


def build_searchqa_target(tokenizer: Any, item: dict) -> str:
    """Return a compact supervised target using the first gold answer."""
    answers = item.get("answers") or []
    answer = str(answers[0] if answers else "").strip()
    target = f"<answer>{answer}</answer>"
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        target += eos
    return target


def build_officeqa_prompt(tokenizer: Any, item: dict, *, enable_thinking: bool = False) -> str:
    """Build an OfficeQA prompt for text-only supervised prefix tuning."""
    system = _build_officeqa_system("", use_local_tools=False)
    user = _build_officeqa_user(
        item,
        candidate_files=_officeqa_candidate_files_for_prompt(item),
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _apply_text_chat_template(tokenizer, messages, enable_thinking=enable_thinking)


def build_officeqa_prompt_and_insert_idx(
    tokenizer: Any,
    item: dict,
    *,
    enable_thinking: bool = False,
    injection_position: str = "prompt_start",
) -> tuple[str, int | None]:
    """Build an OfficeQA prompt and optional soft-prefix insertion token index."""
    injection_position = normalize_prefix_injection_position(injection_position)
    if injection_position == "prompt_start":
        return build_officeqa_prompt(tokenizer, item, enable_thinking=enable_thinking), None

    system = _build_officeqa_system(_SOFT_PREFIX_INSERT_MARKER, use_local_tools=False)
    user = _build_officeqa_user(
        item,
        candidate_files=_officeqa_candidate_files_for_prompt(item),
    )
    return _render_messages_and_insert_idx(
        tokenizer,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        enable_thinking=enable_thinking,
    )


def _officeqa_candidate_files_for_prompt(item: dict) -> list[str]:
    return item.get("resolved_source_paths") or item.get("source_files") or []


def build_officeqa_target(tokenizer: Any, item: dict) -> str:
    """Return a compact supervised target using the OfficeQA gold answer."""
    answers = item.get("answers") or []
    answer = str(item.get("ground_truth") or (answers[0] if answers else "")).strip()
    target = f"<answer>{answer}</answer>"
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        target += eos
    return target


def build_livemath_prompt(
    tokenizer: Any,
    item: dict,
    *,
    use_theorem: bool = False,
    use_sketch: bool = False,
    enable_thinking: bool = False,
) -> str:
    """Build the same LiveMath prompt shape used by rollout, without a markdown skill."""
    messages = [
        {"role": "system", "content": _build_livemath_system("")},
        {
            "role": "user",
            "content": _build_livemath_user(
                item,
                use_theorem=use_theorem,
                use_sketch=use_sketch,
            ),
        },
    ]
    return _apply_text_chat_template(tokenizer, messages, enable_thinking=enable_thinking)


def build_livemath_prompt_and_insert_idx(
    tokenizer: Any,
    item: dict,
    *,
    use_theorem: bool = False,
    use_sketch: bool = False,
    enable_thinking: bool = False,
    injection_position: str = "prompt_start",
) -> tuple[str, int | None]:
    """Build a LiveMath prompt and optional soft-prefix insertion token index."""
    injection_position = normalize_prefix_injection_position(injection_position)
    if injection_position == "prompt_start":
        return (
            build_livemath_prompt(
                tokenizer,
                item,
                use_theorem=use_theorem,
                use_sketch=use_sketch,
                enable_thinking=enable_thinking,
            ),
            None,
        )

    template = load_prompt("rollout_system", env="livemathematicianbench")
    if template.count(_SKILL_SECTION_PLACEHOLDER) != 1:
        raise ValueError(
            "LiveMath soft_prefix.injection_position=skill_section requires exactly "
            "one {skill_section} placeholder in rollout_system.md"
        )
    before_system, after_system = template.split(_SKILL_SECTION_PLACEHOLDER)
    system_with_marker = before_system + _SOFT_PREFIX_INSERT_MARKER + after_system
    user = _build_livemath_user(
        item,
        use_theorem=use_theorem,
        use_sketch=use_sketch,
    )
    return _render_messages_and_insert_idx(
        tokenizer,
        [
            {"role": "system", "content": system_with_marker},
            {"role": "user", "content": user},
        ],
        enable_thinking=enable_thinking,
    )


def build_spreadsheet_codegen_system_for_prefix(
    skill_content: str,
    *,
    injection_position: str = "prompt_start",
) -> str:
    """Build SpreadsheetBench codegen system prompt for soft-prefix training/eval."""
    base = load_prompt("codegen_system", env="spreadsheetbench")
    injection_position = normalize_prefix_injection_position(injection_position)
    if injection_position == "skill_section":
        return f"{base}\n\n## Skill\n{_SOFT_PREFIX_INSERT_MARKER}"
    if skill_content.strip():
        return f"{base}\n\n## Skill\n{skill_content.strip()}"
    return base


def build_spreadsheet_codegen_prompt_and_insert_idx(
    tokenizer: Any,
    *,
    user: str,
    skill_content: str = "",
    enable_thinking: bool = False,
    injection_position: str = "prompt_start",
) -> tuple[str, int | None]:
    """Build a SpreadsheetBench codegen prompt and optional soft-prefix insertion token index."""
    injection_position = normalize_prefix_injection_position(injection_position)
    system = build_spreadsheet_codegen_system_for_prefix(
        skill_content,
        injection_position=injection_position,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if injection_position == "prompt_start":
        return _apply_text_chat_template(tokenizer, messages, enable_thinking=enable_thinking), None

    return _render_messages_and_insert_idx(tokenizer, messages, enable_thinking=enable_thinking)


def build_livemath_target(tokenizer: Any, item: dict) -> str:
    """Return the supervised target as the remapped correct choice label."""
    answer = str((item.get("correct_choice") or {}).get("label", "")).strip()
    target = f"<answer>{answer}</answer>"
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        target += eos
    return target


def build_docvqa_messages(item: dict, *, image_detail: str = "auto") -> list[dict]:
    """Build Qwen-VL DocVQA messages in the rollout's question-then-image order."""
    _messages, system, user = _build_docvqa_messages(item, "", image_detail)
    image_path = str(item.get("image_path", "")).strip()
    if not image_path:
        raise ValueError(f"DocVQA item {item.get('id', '<unknown>')} is missing image_path")
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user},
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
            ],
        },
    ]


def qwen_image_patch_size(processor: Any) -> int:
    """Return the image patch size expected by qwen-vl-utils for this processor."""
    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(processor, "patch_size", None)
    return int(patch_size or 14)


def resolve_docvqa_image_token_budget(
    *,
    max_prompt_tokens: int,
    configured_max_image_tokens: int = 0,
) -> int:
    """Resolve a Qwen-VL image-token cap that leaves room for DocVQA text."""
    prompt_budget = max(1, int(max_prompt_tokens) - 128)
    if configured_max_image_tokens and int(configured_max_image_tokens) > 0:
        return max(1, min(int(configured_max_image_tokens), prompt_budget))
    return max(1, min(int(max_prompt_tokens * 0.75), prompt_budget))


def apply_docvqa_image_budget(
    messages: list[dict],
    *,
    max_image_tokens: int,
    image_patch_size: int,
) -> list[dict]:
    """Attach Qwen's per-image ``max_pixels`` cap without mutating messages."""
    updated = copy.deepcopy(messages)
    if max_image_tokens <= 0:
        return updated
    patch_factor = int(image_patch_size) * 2
    max_pixels = int(max_image_tokens) * patch_factor**2
    for message in updated:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                part["max_pixels"] = max_pixels
    return updated


def build_docvqa_target(tokenizer: Any, item: dict) -> str:
    """Return a compact supervised target using the first gold answer."""
    answers = item.get("answers") or []
    answer = str(answers[0] if answers else item.get("answer", "")).strip()
    target = f"<answer>{answer}</answer>"
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        target += eos
    return target


@dataclass(slots=True)
class EncodedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    extra_inputs: dict[str, Any] | None = None
    prefix_insert_idx: int | None = None


class SearchQAPrefixDataset:
    """Tokenized SearchQA examples for teacher-forced prefix tuning."""

    def __init__(
        self,
        items: list[dict],
        tokenizer: Any,
        *,
        max_prompt_tokens: int,
        max_target_tokens: int,
        injection_position: str = "prompt_start",
    ) -> None:
        self.items = list(items)
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_target_tokens = int(max_target_tokens)
        self.injection_position = normalize_prefix_injection_position(injection_position)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> EncodedExample:
        item = self.items[idx]
        prompt, prefix_insert_idx = build_searchqa_prompt_and_insert_idx(
            self.tokenizer,
            item,
            enable_thinking=False,
            injection_position=self.injection_position,
        )
        target = build_searchqa_target(self.tokenizer, item)
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_tokens,
        )["input_ids"]
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_tokens,
        )["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return EncodedExample(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            labels=labels,
            prefix_insert_idx=(
                min(prefix_insert_idx, len(prompt_ids))
                if prefix_insert_idx is not None
                else None
            ),
        )


class OfficeQAPrefixDataset:
    """Tokenized OfficeQA examples for teacher-forced prefix tuning."""

    def __init__(
        self,
        items: list[dict],
        tokenizer: Any,
        *,
        max_prompt_tokens: int,
        max_target_tokens: int,
    ) -> None:
        self.items = list(items)
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_target_tokens = int(max_target_tokens)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> EncodedExample:
        item = self.items[idx]
        prompt = build_officeqa_prompt(self.tokenizer, item, enable_thinking=False)
        target = build_officeqa_target(self.tokenizer, item)
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_tokens,
        )["input_ids"]
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_tokens,
        )["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return EncodedExample(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            labels=labels,
        )


class TextTrajectoryPrefixDataset:
    """Tokenized prompt/target trajectory examples for teacher-forced prefix tuning."""

    def __init__(
        self,
        items: list[dict],
        tokenizer: Any,
        *,
        max_prompt_tokens: int,
        max_target_tokens: int,
        injection_position: str = "prompt_start",
    ) -> None:
        self.items = list(items)
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_target_tokens = int(max_target_tokens)
        self.injection_position = normalize_prefix_injection_position(injection_position)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> EncodedExample:
        item = self.items[idx]
        if "target_message" in item:
            messages = list(item["messages"])
            target_message = dict(item["target_message"])
            tools = OFFICEQA_TOOL_SCHEMAS if _messages_include_tool_calls(messages + [target_message]) else None
            prefix_insert_idx = None
            if self.injection_position == "skill_section":
                prompt, prefix_insert_idx = _render_messages_and_insert_idx(
                    self.tokenizer,
                    messages,
                    enable_thinking=False,
                    tools=tools,
                    add_generation_prompt=True,
                )
            else:
                prompt = _apply_text_chat_template(
                    self.tokenizer,
                    messages,
                    enable_thinking=False,
                    tools=tools,
                    add_generation_prompt=True,
                )
            full = _apply_text_chat_template(
                self.tokenizer,
                messages + [target_message],
                enable_thinking=False,
                tools=tools,
                add_generation_prompt=False,
            )
            if self.injection_position == "skill_section":
                full = full.replace(_SOFT_PREFIX_INSERT_MARKER, "", 1).replace(_SKILL_SECTION_PLACEHOLDER, "", 1)
            target = full[len(prompt):] if full.startswith(prompt) else str(target_message.get("content") or "")
            eos = getattr(self.tokenizer, "eos_token", None)
            if eos and not target.endswith(eos):
                target += eos
            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_prompt_tokens,
            )["input_ids"]
            target_ids = self.tokenizer(
                target,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_target_tokens,
            )["input_ids"]
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            return EncodedExample(
                input_ids=input_ids,
                attention_mask=[1] * len(input_ids),
                labels=labels,
                prefix_insert_idx=(
                    min(prefix_insert_idx, len(prompt_ids))
                    if prefix_insert_idx is not None
                    else None
                ),
            )
        prefix_insert_idx = None
        if "messages" in item:
            messages = list(item["messages"])
            if self.injection_position == "skill_section":
                prompt, prefix_insert_idx = _render_messages_and_insert_idx(
                    self.tokenizer,
                    messages,
                    enable_thinking=False,
                )
            else:
                prompt = _apply_text_chat_template(
                    self.tokenizer,
                    messages,
                    enable_thinking=False,
                )
        else:
            prompt = str(item["prompt"])
            if self.injection_position == "skill_section":
                prompt, prefix_insert_idx = _strip_soft_prefix_marker_from_prompt(self.tokenizer, prompt)
        target = str(item["target"]).strip()
        eos = getattr(self.tokenizer, "eos_token", None)
        if eos and not target.endswith(eos):
            target += eos
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_tokens,
        )["input_ids"]
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_tokens,
        )["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return EncodedExample(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            labels=labels,
            prefix_insert_idx=(
                min(prefix_insert_idx, len(prompt_ids))
                if prefix_insert_idx is not None
                else None
            ),
        )


def _messages_include_tool_calls(messages: list[dict]) -> bool:
    return any(isinstance(message.get("tool_calls"), list) and message.get("tool_calls") for message in messages)


class LiveMathPrefixDataset:
    """Tokenized LiveMath examples for teacher-forced prefix tuning."""

    def __init__(
        self,
        items: list[dict],
        tokenizer: Any,
        *,
        max_prompt_tokens: int,
        max_target_tokens: int,
        use_theorem: bool = False,
        use_sketch: bool = False,
        injection_position: str = "prompt_start",
    ) -> None:
        self.items = list(items)
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_target_tokens = int(max_target_tokens)
        self.use_theorem = bool(use_theorem)
        self.use_sketch = bool(use_sketch)
        self.injection_position = normalize_prefix_injection_position(injection_position)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> EncodedExample:
        item = self.items[idx]
        prompt, prefix_insert_idx = build_livemath_prompt_and_insert_idx(
            self.tokenizer,
            item,
            use_theorem=self.use_theorem,
            use_sketch=self.use_sketch,
            enable_thinking=False,
            injection_position=self.injection_position,
        )
        target = build_livemath_target(self.tokenizer, item)
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_tokens,
        )["input_ids"]
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_tokens,
        )["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return EncodedExample(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            labels=labels,
            prefix_insert_idx=(
                min(prefix_insert_idx, len(prompt_ids))
                if prefix_insert_idx is not None
                else None
            ),
        )


class DocVQAPrefixDataset:
    """Processor-encoded DocVQA examples for Qwen-VL soft-prefix tuning."""

    def __init__(
        self,
        items: list[dict],
        processor: Any,
        tokenizer: Any,
        *,
        max_prompt_tokens: int,
        max_target_tokens: int,
        image_detail: str = "auto",
        max_image_tokens: int = 0,
    ) -> None:
        self.items = list(items)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_target_tokens = int(max_target_tokens)
        self.image_detail = image_detail
        self.max_image_tokens = resolve_docvqa_image_token_budget(
            max_prompt_tokens=self.max_prompt_tokens,
            configured_max_image_tokens=int(max_image_tokens or 0),
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> EncodedExample:
        item = self.items[idx]
        messages = apply_docvqa_image_budget(
            build_docvqa_messages(item, image_detail=self.image_detail),
            max_image_tokens=self.max_image_tokens,
            image_patch_size=qwen_image_patch_size(self.processor),
        )
        prompt_inputs = self._encode_prompt(messages)
        target_ids = self.tokenizer(
            build_docvqa_target(self.tokenizer, item),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_tokens,
        )["input_ids"]

        prompt_ids = prompt_inputs["input_ids"]
        if len(prompt_ids) > self.max_prompt_tokens:
            raise ValueError(
                f"DocVQA item {item.get('id', idx)!r} encoded to {len(prompt_ids)} prompt tokens, "
                f"exceeding soft_prefix.max_prompt_tokens={self.max_prompt_tokens}. "
                "Lower soft_prefix.docvqa_max_image_tokens to downscale images further."
            )

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = prompt_inputs["attention_mask"] + [1] * len(target_ids)
        mm_token_type_ids = prompt_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None and target_ids:
            import torch

            target_token_types = torch.zeros(
                (mm_token_type_ids.shape[0], len(target_ids)),
                dtype=mm_token_type_ids.dtype,
                device=mm_token_type_ids.device,
            )
            prompt_inputs["mm_token_type_ids"] = torch.cat(
                [mm_token_type_ids, target_token_types],
                dim=1,
            )
        extra_inputs = {
            key: value
            for key, value in prompt_inputs.items()
            if key not in {"input_ids", "attention_mask"}
        }
        return EncodedExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            extra_inputs=extra_inputs,
        )

    def _encode_prompt(self, messages: list[dict]) -> dict:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "DocVQA soft-prefix training requires qwen-vl-utils. "
                "Install with `pip install -e '.[softprefix]'`."
            ) from exc

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=qwen_image_patch_size(self.processor),
        )
        kwargs = {
            "text": [text],
            "images": image_inputs,
            "do_resize": False,
            "padding": False,
            "return_tensors": "pt",
        }
        if video_inputs:
            kwargs["videos"] = video_inputs
        encoded = self.processor(**kwargs)
        return {
            key: value[0].tolist() if key in {"input_ids", "attention_mask"} else value
            for key, value in encoded.items()
            if value is not None
        }


def _build_alfworld_trajectory_tokens(
    tokenizer: Any,
    steps: list[dict],
    max_total_tokens: int,
) -> tuple[list[int], list[int]]:
    """Build a multi-turn token sequence for a full ALFWorld episode.

    Each entry in `steps` must have:
        - "prompt": the formatted obs text shown to the model at this step
        - "model_response": the model's full response string

    Returns ``(input_ids, labels)`` where ``labels`` has ``-100`` for system
    and observation tokens and response token IDs for model-generated tokens.

    The per-turn masking uses progressive tokenisation: tokenising
    ``messages[:k+1]`` and ``messages[:k]`` (both without a generation prompt)
    gives us exactly the tokens contributed by turn ``k``, since chat-template
    special tokens act as BPE boundaries and prevent cross-turn merges.
    Truncation keeps the *most recent* tokens when the episode is too long.
    """
    from skillopt.envs.alfworld.rollout import SYSTEM_PROMPT as _ALFWORLD_SYSTEM

    messages: list[dict] = [{"role": "system", "content": _ALFWORLD_SYSTEM}]
    for step in steps:
        messages.append({"role": "user", "content": step["prompt"]})
        messages.append({"role": "assistant", "content": step["model_response"]})

    apply = getattr(tokenizer, "apply_chat_template", None)
    if not (callable(apply) and getattr(tokenizer, "chat_template", None)):
        return _build_alfworld_trajectory_tokens_simple(tokenizer, steps, max_total_tokens)

    try:
        input_ids: list[int] = []
        labels: list[int] = []
        prev_ids: list[int] = []

        for k, msg in enumerate(messages):
            try:
                cur_ids: list[int] = list(apply(messages[: k + 1], tokenize=True, add_generation_prompt=False))
            except Exception:
                return _build_alfworld_trajectory_tokens_simple(tokenizer, steps, max_total_tokens)

            turn_ids = cur_ids[len(prev_ids):]
            input_ids.extend(turn_ids)
            if msg["role"] == "assistant":
                labels.extend(list(turn_ids))
            else:
                labels.extend([-100] * len(turn_ids))
            prev_ids = cur_ids

    except Exception:
        return _build_alfworld_trajectory_tokens_simple(tokenizer, steps, max_total_tokens)

    # Truncate from the beginning so the most recent (decisive) steps are kept
    if len(input_ids) > max_total_tokens:
        input_ids = input_ids[-max_total_tokens:]
        labels = labels[-max_total_tokens:]

    return input_ids, labels


def _build_alfworld_trajectory_tokens_simple(
    tokenizer: Any,
    steps: list[dict],
    max_total_tokens: int,
) -> tuple[list[int], list[int]]:
    """Fallback: plain concatenation without chat-template markers."""
    from skillopt.envs.alfworld.rollout import SYSTEM_PROMPT as _ALFWORLD_SYSTEM

    eos = tokenizer.eos_token or ""
    sep = "\n\n"

    def _tok(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    input_ids: list[int] = []
    labels: list[int] = []

    sys_ids = _tok(_ALFWORLD_SYSTEM + sep)
    input_ids.extend(sys_ids)
    labels.extend([-100] * len(sys_ids))

    for step in steps:
        obs_ids = _tok(step["prompt"] + sep)
        input_ids.extend(obs_ids)
        labels.extend([-100] * len(obs_ids))

        response = step["model_response"].strip()
        if eos and not response.endswith(eos):
            response = response + eos
        resp_ids = _tok(response + sep)
        input_ids.extend(resp_ids)
        labels.extend(list(resp_ids))

    if len(input_ids) > max_total_tokens:
        input_ids = input_ids[-max_total_tokens:]
        labels = labels[-max_total_tokens:]

    return input_ids, labels


class AlfWorldTrajectoryPrefixDataset:
    """Multi-turn SFT dataset for ALFWorld episodes.

    One training example = one complete episode (all steps concatenated).
    Loss is computed only on model-generated response tokens; environment
    observations and the system prompt are masked with ``-100``.

    Each item dict must contain:
        ``"steps"``: list of ``{"prompt": str, "model_response": str}``
    """

    def __init__(
        self,
        items: list[dict],
        tokenizer: Any,
        *,
        max_prompt_tokens: int,
        max_target_tokens: int,
    ) -> None:
        self.items = list(items)
        self.tokenizer = tokenizer
        self.max_total_tokens = int(max_prompt_tokens) + int(max_target_tokens)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> EncodedExample:
        item = self.items[idx]
        steps = list(item.get("steps") or [])
        input_ids, labels = _build_alfworld_trajectory_tokens(
            self.tokenizer,
            steps,
            self.max_total_tokens,
        )
        return EncodedExample(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            labels=labels,
        )


class PrefixBatchCollator:
    """Pad examples for causal LM loss; prefix labels are added by the model wrapper."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples: list[EncodedExample]) -> dict:
        max_len = max(len(ex.input_ids) for ex in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for ex in examples:
            pad_n = max_len - len(ex.input_ids)
            batch["input_ids"].append(ex.input_ids + [self.pad_token_id] * pad_n)
            batch["attention_mask"].append(ex.attention_mask + [0] * pad_n)
            batch["labels"].append(ex.labels + [-100] * pad_n)
        if any(ex.prefix_insert_idx is not None for ex in examples):
            batch["prefix_insert_idx"] = [
                int(ex.prefix_insert_idx or 0)
                for ex in examples
            ]
        extra_keys = {
            key
            for ex in examples
            for key in (ex.extra_inputs or {})
        }
        for key in extra_keys:
            values = [(ex.extra_inputs or {}).get(key) for ex in examples]
            values = [value for value in values if value is not None]
            if not values:
                continue
            first = values[0]
            if hasattr(first, "dim") and callable(first.dim):
                import torch

                if key == "mm_token_type_ids":
                    padded_values = []
                    for value in values:
                        pad_n = max_len - int(value.shape[1])
                        if pad_n > 0:
                            padding = torch.zeros(
                                (value.shape[0], pad_n),
                                dtype=value.dtype,
                                device=value.device,
                            )
                            value = torch.cat([value, padding], dim=1)
                        padded_values.append(value)
                    values = padded_values
                batch[key] = torch.cat(values, dim=0)
            else:
                batch[key] = values
        return batch
