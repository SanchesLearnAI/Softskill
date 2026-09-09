"""Supervised soft-prefix trainers for frozen open-weight models."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from skillopt.envs.alfworld.dataloader import ALFWorldDataLoader
from skillopt.envs.alfworld.rollout import (
    SYSTEM_PROMPT as ALFWORLD_SYSTEM,
)
from skillopt.envs.alfworld.rollout import (
    _build_skill_prompt as build_alfworld_skill_prompt,
)
from skillopt.envs.alfworld.rollout import (
    build_alfworld_env,
    run_alfworld_batch,
)
from skillopt.envs.docvqa.dataloader import DocVQADataLoader
from skillopt.envs.docvqa.evaluator import evaluate as evaluate_docvqa
from skillopt.envs.livemathematicianbench.dataloader import LiveMathematicianBenchDataLoader
from skillopt.envs.livemathematicianbench.evaluator import evaluate as evaluate_livemath
from skillopt.envs.officeqa.dataloader import OfficeQADataLoader
from skillopt.envs.officeqa.evaluator import evaluate as evaluate_officeqa
from skillopt.envs.officeqa.rollout import _TOOL_SCHEMAS as OFFICEQA_TOOL_SCHEMAS
from skillopt.envs.officeqa.rollout import _build_system as build_officeqa_system
from skillopt.envs.officeqa.rollout import _build_user as build_officeqa_user
from skillopt.envs.officeqa.rollout import _extract_answer as extract_officeqa_answer
from skillopt.envs.officeqa.tool_runtime import (
    build_oracle_parsed_pages_context,
    resolve_candidate_files,
    resolve_docs_roots,
)
from skillopt.envs.officeqa.tool_runtime import (
    run_tool as run_officeqa_tool,
)
from skillopt.envs.searchqa.dataloader import SearchQADataLoader
from skillopt.envs.searchqa.evaluator import evaluate as evaluate_searchqa
from skillopt.envs.spreadsheetbench.codegen_agent import (
    _build_system as build_spreadsheet_system,
)
from skillopt.envs.spreadsheetbench.codegen_agent import (
    _build_user as build_spreadsheet_user,
)
from skillopt.envs.spreadsheetbench.codegen_agent import (
    extract_code as extract_spreadsheet_code,
)
from skillopt.envs.spreadsheetbench.dataloader import SpreadsheetBenchDataLoader
from skillopt.envs.spreadsheetbench.evaluator import evaluate as evaluate_spreadsheet
from skillopt.envs.spreadsheetbench.executor import run_generated_code as run_spreadsheet_generated_code
from skillopt.envs.spreadsheetbench.rollout import (
    _auto_verify_output as auto_verify_spreadsheet_output,
)
from skillopt.envs.spreadsheetbench.rollout import (
    _find_test_cases as find_spreadsheet_test_cases,
)
from skillopt.envs.spreadsheetbench.rollout import (
    run_spreadsheet_batch_codegen,
)
from skillopt.evaluation.gate import select_gate_score
from skillopt.softprefix.data import (
    AlfWorldTrajectoryPrefixDataset,
    DocVQAPrefixDataset,
    LiveMathPrefixDataset,
    OfficeQAPrefixDataset,
    PrefixBatchCollator,
    SearchQAPrefixDataset,
    TextTrajectoryPrefixDataset,
    _SOFT_PREFIX_INSERT_MARKER,
    _apply_text_chat_template,
    build_docvqa_messages,
    build_livemath_prompt_and_insert_idx,
    build_officeqa_prompt_and_insert_idx,
    build_officeqa_prompt,
    build_searchqa_prompt,
    build_searchqa_prompt_and_insert_idx,
    build_spreadsheet_codegen_prompt_and_insert_idx,
    normalize_prefix_injection_position,
)
from skillopt.softprefix.lora import LoraCausalLM, LoraSettings, LoraVisionLM
from skillopt.softprefix.model import SoftPrefixCausalLM, SoftPrefixVisionLM, _import_torch_and_transformers
from skillopt.utils import compute_score


def _normalize_trajectory_rollout_backend(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"", "openai", "openai_chat", "openai_compatible", "openai-compatible", "chat"}:
        return "openai"
    if raw == "vllm":
        return "vllm"
    if raw in {"local_hf", "local-hf", "local_hf_soft_prefix"}:
        return "local_hf"
    raise ValueError(
        "soft_prefix.trajectory_rollout_backend must be one of openai, openai_chat, "
        "openai_compatible, vllm, or local_hf"
    )


def _normalize_inference_backend(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"", "local", "local_hf", "local-hf"}:
        return "local_hf"
    if raw in {"vllm", "vllm_prompt_embeds", "prompt_embeds", "prompt-embeds"}:
        return "vllm_prompt_embeds"
    raise ValueError(
        "soft_prefix.inference_backend must be one of local_hf or vllm_prompt_embeds"
    )


@dataclass(slots=True)
class SoftPrefixSettings:
    model_name: str
    architecture: str = "auto"
    prefix_length: int = 32
    num_soft_skills: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_prompt_tokens: int = 2048
    max_target_tokens: int = 64
    max_new_tokens: int = 64
    trajectory_max_new_tokens: int = 0
    torch_dtype: str = "auto"
    device: str = "auto"
    trust_remote_code: bool = False
    init_text_path: str = ""
    init_strategy: str = "text"
    generation_temperature: float = 0.0
    training_data: str = "gold"
    trajectory_rollout_dir: str = ""
    trajectory_rollout_backend: str = "openai"
    trajectory_min_hard: float = 1.0
    trajectory_min_soft: float = 0.0
    trajectory_max_examples: int = 0
    trajectory_rollouts_per_task: int = 1
    trajectory_max_context_chars: int = 30000
    strip_trajectory_thoughts: bool = False
    trajectory_use_skill: bool = False
    train_on_final_only: bool = True
    docvqa_max_image_tokens: int = 0
    eval_init_prefix: bool = False
    eval_init_val: bool = True
    eval_plain_baseline: bool = False
    checkpoint_path: str = ""
    rewind_ckpt: bool = False
    inference_backend: str = "local_hf"
    inference_base_url: str = "http://127.0.0.1:8010"
    inference_timeout_seconds: float = 300.0
    injection_position: str = "prompt_start"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "SoftPrefixSettings":
        if not cfg.get("model_name"):
            raise ValueError("soft_prefix.model_name is required")
        return cls(
            model_name=str(cfg["model_name"]),
            architecture=str(cfg.get("architecture", "auto")),
            prefix_length=_parse_prefix_length(cfg.get("prefix_length", 32)),
            num_soft_skills=int(cfg.get("num_soft_skills", 2)),
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
            max_prompt_tokens=int(cfg.get("max_prompt_tokens", 2048)),
            max_target_tokens=int(cfg.get("max_target_tokens", 64)),
            max_new_tokens=int(cfg.get("max_new_tokens", 64)),
            trajectory_max_new_tokens=int(cfg.get("trajectory_max_new_tokens", 0) or 0),
            torch_dtype=str(cfg.get("torch_dtype", "auto")),
            device=str(cfg.get("device", "auto")),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            init_text_path=str(cfg.get("init_text_path", "")),
            init_strategy=str(cfg.get("init_strategy", "text")),
            generation_temperature=float(cfg.get("generation_temperature", 0.0)),
            training_data=str(cfg.get("training_data", "gold")),
            trajectory_rollout_dir=str(cfg.get("trajectory_rollout_dir", "")),
            trajectory_rollout_backend=_normalize_trajectory_rollout_backend(
                cfg.get("trajectory_rollout_backend", "openai")
            ),
            trajectory_min_hard=float(cfg.get("trajectory_min_hard", 1.0)),
            trajectory_min_soft=float(cfg.get("trajectory_min_soft", 0.0)),
            trajectory_max_examples=int(cfg.get("trajectory_max_examples", 0) or 0),
            trajectory_rollouts_per_task=_parse_trajectory_rollouts_per_task(
                cfg.get("trajectory_rollouts_per_task", 1)
            ),
            trajectory_max_context_chars=int(cfg.get("trajectory_max_context_chars", 30000) or 30000),
            strip_trajectory_thoughts=bool(cfg.get("strip_trajectory_thoughts", False)),
            trajectory_use_skill=bool(cfg.get("trajectory_use_skill", False)),
            train_on_final_only=bool(cfg.get("train_on_final_only", True)),
            docvqa_max_image_tokens=int(cfg.get("docvqa_max_image_tokens", 0) or 0),
            eval_init_prefix=bool(cfg.get("eval_init_prefix", False)),
            eval_init_val=_parse_bool(cfg.get("eval_init_val", True)),
            eval_plain_baseline=bool(cfg.get("eval_plain_baseline", False)),
            checkpoint_path=str(cfg.get("checkpoint_path", "")),
            rewind_ckpt=_parse_bool(cfg.get("rewind_ckpt", False)),
            inference_backend=_normalize_inference_backend(cfg.get("inference_backend", "local_hf")),
            inference_base_url=str(cfg.get("inference_base_url", "http://127.0.0.1:8010")),
            inference_timeout_seconds=float(cfg.get("inference_timeout_seconds", 300.0)),
            injection_position=normalize_prefix_injection_position(
                cfg.get("injection_position", "prompt_start")
            ),
        )


def _parse_prefix_length(value: Any) -> int:
    """Return 0 as a sentinel when the user sets prefix_length to 'auto'."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return 0
    return int(value)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _parse_trajectory_rollouts_per_task(value: Any) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        return 1
    rollouts = int(value)
    if rollouts < 1:
        raise ValueError("soft_prefix.trajectory_rollouts_per_task must be >= 1")
    return rollouts


def _expand_trajectory_rollout_items(items: list[dict], *, rollouts_per_task: int) -> list[dict]:
    """Return stable per-attempt items for repeated teacher trajectory rollouts."""
    rollouts = _parse_trajectory_rollouts_per_task(rollouts_per_task)
    if rollouts == 1:
        return items

    expanded: list[dict] = []
    for item_index, item in enumerate(items):
        source_id = str(item.get("id", f"item_{item_index:04d}"))
        for attempt_idx in range(rollouts):
            attempt = dict(item)
            attempt["trajectory_source_id"] = source_id
            attempt["trajectory_sample_idx"] = attempt_idx
            attempt["id"] = f"{source_id}__sample_{attempt_idx:02d}"
            expanded.append(attempt)
    return expanded


def _skill_content_fingerprint(skill_content: str) -> str:
    content = skill_content.strip()
    if not content:
        return ""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_rollout_metadata(*, rollout_dir: str, metadata: dict[str, Any]) -> None:
    os.makedirs(rollout_dir, exist_ok=True)
    meta_path = os.path.join(rollout_dir, "rollout_meta.json")
    if os.path.exists(meta_path):
        try:
            existing = _load_json_file(meta_path)
        except Exception:  # noqa: BLE001
            existing = None
        if isinstance(existing, dict) and existing != metadata:
            print(
                f"    [rollout-cache] warning: existing metadata differs in {meta_path}; "
                "reusing cache because soft_prefix.trajectory_rollout_dir was explicit",
                flush=True,
            )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _officeqa_rollout_metadata(
    *,
    cfg: dict[str, Any],
    settings: "SoftPrefixSettings",
    skill_content: str,
) -> dict[str, Any]:
    return {
        "env": "officeqa",
        "rollout_backend": settings.trajectory_rollout_backend,
        "target_backend": cfg.get("target_backend", ""),
        "target_model": cfg.get("target_model", ""),
        "max_tool_turns": int(cfg.get("max_tool_turns", 12) or 12),
        "max_completion_tokens": int(settings.trajectory_max_new_tokens or cfg.get("max_completion_tokens", 16384) or 16384),
        "trajectory_use_skill": bool(settings.trajectory_use_skill),
        "trajectory_skill_fingerprint": _skill_content_fingerprint(skill_content),
        "trajectory_rollouts_per_task": int(settings.trajectory_rollouts_per_task),
        "split_dir": cfg.get("split_dir", ""),
        "split_seed": int(cfg.get("split_seed", cfg.get("seed", 42)) or 42),
    }


def _spreadsheet_rollout_metadata(
    *,
    cfg: dict[str, Any],
    settings: "SoftPrefixSettings",
    skill_content: str,
) -> dict[str, Any]:
    return {
        "env": "spreadsheetbench",
        "rollout_backend": settings.trajectory_rollout_backend,
        "target_backend": cfg.get("target_backend", ""),
        "target_model": cfg.get("target_model", ""),
        "mode": str(cfg.get("mode", "multi") or "multi"),
        "max_turns": int(cfg.get("max_turns", 5) or 5),
        "max_completion_tokens": int(settings.trajectory_max_new_tokens or cfg.get("max_completion_tokens", 16384) or 16384),
        "use_eval_feedback": bool(cfg.get("use_eval_feedback", False)),
        "trajectory_use_skill": bool(settings.trajectory_use_skill),
        "trajectory_skill_fingerprint": _skill_content_fingerprint(skill_content),
        "trajectory_rollouts_per_task": int(settings.trajectory_rollouts_per_task),
        "split_dir": cfg.get("split_dir", ""),
        "split_seed": int(cfg.get("split_seed", cfg.get("seed", 42)) or 42),
        "data_root": cfg.get("data_root", ""),
    }


def _inject_alfworld_skill_into_prompt(prompt: str, skill_content: str) -> str:
    skill_prompt = build_alfworld_skill_prompt(skill_content)
    if not skill_prompt or "## Skill Knowledge" in prompt:
        return prompt
    return skill_prompt + "\n" + prompt


def _replace_skill_content_with_marker(text: str, skill_content: str) -> str:
    marker = _SOFT_PREFIX_INSERT_MARKER
    if marker in text:
        return text
    stripped_skill = skill_content.strip()
    if stripped_skill and stripped_skill in text:
        return text.replace(stripped_skill, marker, 1)
    if "## Skill\n" in text:
        return text.split("## Skill\n", 1)[0] + f"## Skill\n{marker}"
    if "## Rules" in text:
        return text.replace("## Rules", f"## Skill\n{marker}\n\n## Rules", 1)
    if "## Tools" in text:
        return text.replace("## Tools", f"## Skill\n{marker}\n\n## Tools", 1)
    return f"## Skill\n{marker}\n\n{text}"


def _resolve_alfworld_rollout_max_completion_tokens(
    cfg: dict[str, Any],
    settings: "SoftPrefixSettings",
) -> int:
    base = int(settings.trajectory_max_new_tokens or cfg.get("max_completion_tokens", 2048) or 2048)
    target_cap = cfg.get("target_qwen_chat_max_tokens", cfg.get("qwen_chat_max_tokens"))
    if target_cap is None or str(target_cap).strip() == "":
        return base
    return min(base, int(target_cap))


def _resolve_auto_prefix_length(model_name: str, init_text: str, trust_remote_code: bool) -> int:
    """Count the tokens in *init_text* using the model tokenizer and return that count."""
    _, _, AutoTokenizer = _import_torch_and_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    ids = tokenizer(init_text, add_special_tokens=False)["input_ids"]
    n = len(ids)
    if n == 0:
        raise ValueError(
            "prefix_length: auto requires a non-empty init_text_path document to count tokens from"
        )
    return n


def _set_seed(seed: int) -> None:
    torch, _, _ = _import_torch_and_transformers()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _trajectory_rollout_skill_content(*, settings: SoftPrefixSettings, init_text: str) -> str:
    """Skill markdown injected into trajectory rollouts when trajectory_use_skill is enabled."""
    if not settings.trajectory_use_skill:
        return ""
    return init_text.strip()


def _load_init_text(path: str) -> str:
    if not path:
        return ""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return ""
    with open(abs_path, encoding="utf-8") as f:
        return f.read()


def _batch_to_tensors(torch, batch: dict, device) -> dict:
    return {
        key: value.to(device) if hasattr(value, "to") else torch.tensor(value, dtype=torch.long, device=device)
        for key, value in batch.items()
    }


def _items_for_eval(dataloader: Any, split: str, env_num: int, seed: int) -> list[dict]:
    batch = dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed)
    return list(batch.payload or [])


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep_head = max_chars // 3
    keep_tail = max_chars - keep_head
    omitted = len(text) - keep_head - keep_tail
    return (
        text[:keep_head].rstrip()
        + f"\n\n[... {omitted} trajectory characters omitted ...]\n\n"
        + text[-keep_tail:].lstrip()
    )


def _strip_think_blocks(text: str) -> str:
    action_match = re.search(r"<action>(.*?)</action>", text, flags=re.DOTALL)
    if action_match:
        return f"<action>{action_match.group(1).strip()}</action>"
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _load_json_file(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_text_if_exists(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def _conversation_before_final(conversation: list[dict], final_response: str) -> list[dict]:
    final_idx = len(conversation)
    for idx in range(len(conversation) - 1, -1, -1):
        event = conversation[idx]
        content = str(event.get("content") or "")
        if event.get("type") == "message" and (
            "<answer>" in content.lower() or (final_response and content.strip() == final_response.strip())
        ):
            final_idx = idx
            break
    return conversation[:final_idx]


def _format_officeqa_trajectory_context(
    conversation: list[dict],
    *,
    final_response: str,
    max_chars: int,
) -> str:
    lines: list[str] = []
    prior = _conversation_before_final(conversation, final_response)
    for idx, event in enumerate(prior):
        role = str(event.get("role") or "")
        event_type = str(event.get("type") or "")
        content = str(event.get("content") or "").strip()
        if idx == 0 and role == "user":
            # The initial user prompt is already represented by target_user_prompt.
            continue
        if event_type == "message":
            if content:
                lines.append(f"Assistant turn:\n{content}")
            continue
        if event_type == "tool_call":
            cmd = str(event.get("cmd") or "").strip()
            obs = str(event.get("obs") or "").strip()
            lines.append(f"Tool call:\n{cmd}\n\nTool observation:\n{obs}")
            continue
        if role and content:
            lines.append(f"{role.title()} turn:\n{content}")
    if not lines:
        return ""
    context = "## Successful Retrieval Trajectory\n" + "\n\n".join(lines)
    return _truncate_text(context, max_chars)


def _parse_officeqa_tool_cmd(cmd: str) -> tuple[str, dict[str, Any]]:
    try:
        parsed = ast.parse(cmd.strip(), mode="eval")
    except SyntaxError:
        return "", {}
    call = parsed.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return "", {}
    arguments: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            continue
    return call.func.id, arguments


def _officeqa_tool_call_payload(cmd: str, *, turn_index: int, call_index: int) -> dict[str, Any] | None:
    name, arguments = _parse_officeqa_tool_cmd(cmd)
    if not name:
        return None
    return {
        "id": f"officeqa_tool_{turn_index}_{call_index}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _officeqa_base_messages(system: str, user: str, conversation: list[dict]) -> list[dict[str, Any]]:
    if not user:
        for event in conversation:
            if isinstance(event, dict) and str(event.get("role") or "") == "user":
                user = str(event.get("content") or "")
                break
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _truncate_text(user, 1_000_000)},
    ]


def _build_officeqa_trajectory_examples(
    rollout_dir: str,
    results: list[dict],
    settings: SoftPrefixSettings,
    skill_content: str = "",
) -> list[dict]:
    examples: list[dict] = []
    for row in results:
        hard = float(row.get("hard", 0) or 0)
        soft = float(row.get("soft", 0.0) or 0.0)
        if hard < settings.trajectory_min_hard or soft < settings.trajectory_min_soft:
            continue
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            continue
        final_response = str(row.get("response") or "").strip()
        if not final_response:
            continue
        conversation_path = os.path.join(rollout_dir, "predictions", item_id, "conversation.json")
        conversation = _load_json_file(conversation_path) if os.path.exists(conversation_path) else []
        if not isinstance(conversation, list):
            conversation = []
        system = str(row.get("target_system_prompt") or "").strip()
        if settings.injection_position == "skill_section":
            system = _replace_skill_content_with_marker(system, skill_content)
        user = str(row.get("target_user_prompt") or "").strip()
        messages = _officeqa_base_messages(system, user, conversation)
        turn_index = 0
        idx = 0
        while idx < len(conversation):
            event = conversation[idx]
            if not isinstance(event, dict):
                idx += 1
                continue
            role = str(event.get("role") or "")
            event_type = str(event.get("type") or "")
            if idx == 0 and role == "user":
                idx += 1
                continue
            if event_type != "message":
                if role in {"user", "tool"}:
                    messages.append({"role": role, "content": str(event.get("content") or "")})
                idx += 1
                continue

            content = str(event.get("content") or "")
            tool_events: list[dict] = []
            next_idx = idx + 1
            while next_idx < len(conversation):
                next_event = conversation[next_idx]
                if not isinstance(next_event, dict) or next_event.get("type") != "tool_call":
                    break
                tool_events.append(next_event)
                next_idx += 1
            if not content and not tool_events:
                idx = next_idx
                continue

            turn_index += 1
            target_message: dict[str, Any] = {"role": "assistant", "content": content}
            tool_call_payloads: list[dict[str, Any]] = []
            for call_index, tool_event in enumerate(tool_events, start=1):
                payload = _officeqa_tool_call_payload(
                    str(tool_event.get("cmd") or ""),
                    turn_index=turn_index,
                    call_index=call_index,
                )
                if payload is not None:
                    tool_call_payloads.append(payload)
            if tool_call_payloads:
                target_message["tool_calls"] = tool_call_payloads

            is_final_turn = "<answer>" in content.lower() or content.strip() == final_response
            if not settings.train_on_final_only or is_final_turn:
                examples.append(
                    {
                        "id": f"{item_id}__turn_{turn_index:02d}",
                        "source_id": item_id,
                        "turn_index": turn_index,
                        "messages": list(messages),
                        "target_message": target_message,
                        "hard": hard,
                        "soft": soft,
                        "rollout_dir": rollout_dir,
                    }
                )
            messages.append(target_message)
            for tool_call, tool_event in zip(tool_call_payloads, tool_events, strict=False):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call.get("id") or ""),
                        "content": str(tool_event.get("obs") or ""),
                    }
                )
            if is_final_turn:
                break
            if settings.trajectory_max_examples > 0 and len(examples) >= settings.trajectory_max_examples:
                break
            idx = next_idx
        if settings.trajectory_max_examples > 0 and len(examples) >= settings.trajectory_max_examples:
            break
    return examples


def _write_officeqa_local_rollout_files(
    *,
    pred_dir: str,
    system: str,
    user: str,
    conversation: list[dict],
) -> None:
    os.makedirs(pred_dir, exist_ok=True)
    with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(system)
    with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(user)
    with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)


def _run_officeqa_local_hf_rollout(
    *,
    prefix_model: SoftPrefixCausalLM,
    items: list[dict],
    out_root: str,
    skill_content: str,
    cfg: dict[str, Any],
    settings: SoftPrefixSettings,
) -> list[dict]:
    """Generate OfficeQA trajectories with the same local HF model being trained."""
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)
    rollout_backend = "local_hf_soft_prefix"
    rollout_model = settings.model_name
    rollout_max_new_tokens = int(
        settings.trajectory_max_new_tokens
        or cfg.get("max_completion_tokens", settings.max_new_tokens)
        or settings.max_new_tokens
    )
    rollout_max_turns = max(1, int(cfg.get("max_tool_turns", 1) or 1))

    done_ids: set[str] = set()
    results: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("rollout_backend") != rollout_backend or row.get("rollout_model") != rollout_model:
                    continue
                if int(row.get("rollout_max_new_tokens", 0) or 0) != rollout_max_new_tokens:
                    continue
                if int(row.get("rollout_max_turns", 0) or 0) != rollout_max_turns:
                    continue
                done_ids.add(str(row.get("id")))
                results.append(row)

    pending = [item for item in items if str(item["id"]) not in done_ids]
    total = len(results) + len(pending)
    completed = len(results)
    correct_count = sum(1 for row in results if row.get("hard", 0))
    if results:
        print(f"    [rollout-local-hf] resuming: {completed}/{total} already done", flush=True)

    docs_roots = resolve_docs_roots(cfg.get("data_dirs"))
    with open(results_path, "a", encoding="utf-8") as outf:
        for item in tqdm(pending, desc="    Rollout local HF", unit="ex", leave=False):
            item_id = str(item["id"])
            pred_dir = os.path.join(out_root, "predictions", item_id)
            candidate_files = resolve_candidate_files(item.get("source_files", []), docs_roots)
            oracle_context = build_oracle_parsed_pages_context(
                item.get("source_files", []),
                item.get("source_docs", []),
                docs_roots,
                evidence_note="Treat it as primary document evidence for answering the question.",
            )
            system = build_officeqa_system(skill_content, search_mode="offline", use_local_tools=False)
            user = build_officeqa_user(
                item,
                candidate_files,
                search_mode="offline",
                oracle_context=oracle_context,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            conversation = [{"role": "user", "content": user}]
            response = ""
            final_answer = ""
            for turn in range(1, rollout_max_turns + 1):
                prompt = _apply_text_chat_template(
                    prefix_model.tokenizer,
                    messages,
                    enable_thinking=False,
                )
                response = prefix_model.generate_from_prompt(
                    prompt,
                    max_prompt_tokens=settings.max_prompt_tokens,
                    max_new_tokens=rollout_max_new_tokens,
                    temperature=settings.generation_temperature,
                )
                conversation.append({"type": "message", "turn": turn, "content": response})
                messages.append({"role": "assistant", "content": response})
                if "<answer>" in response.lower():
                    final_answer = extract_officeqa_answer(response)
                    break
                if turn < rollout_max_turns:
                    followup = (
                        "Your previous response did not include a final answer in `<answer>...</answer>`. "
                        "Continue from where you stopped if needed, then return the final answer inside "
                        "`<answer>...</answer>`."
                    )
                    messages.append({"role": "user", "content": followup})
                    conversation.append({"role": "user", "turn": turn + 1, "content": followup})
            if not final_answer:
                final_answer = extract_officeqa_answer(response)
            scores = evaluate_officeqa(final_answer, item.get("ground_truth", ""))
            _write_officeqa_local_rollout_files(
                pred_dir=pred_dir,
                system=system,
                user=user,
                conversation=conversation,
            )
            row = {
                "id": item_id,
                "question": item.get("question", ""),
                "task_type": item.get("task_type", "officeqa"),
                "task_description": item.get("question", ""),
                "predicted_answer": scores["predicted_answer"],
                "response": response,
                "ground_truth": item.get("ground_truth", ""),
                "source_files": item.get("source_files", []),
                "resolved_source_paths": candidate_files,
                "oracle_parsed_pages_included": bool(oracle_context),
                "oracle_parsed_pages_chars": len(oracle_context),
                "use_local_tools": False,
                "hard": int(scores["em"]),
                "soft": scores["f1"],
                "em": scores["em"],
                "f1": scores["f1"],
                "fail_reason": "" if scores["em"] else f"predicted '{scores['predicted_answer']}' but expected '{item.get('ground_truth', '')}'",
                "agent_ok": True,
                "n_turns": len(conversation),
                "last_finish_reason": "",
                "target_system_prompt": system,
                "target_user_prompt": user,
                "rollout_backend": rollout_backend,
                "rollout_model": rollout_model,
                "rollout_max_new_tokens": rollout_max_new_tokens,
                "rollout_max_turns": rollout_max_turns,
            }
            results.append(row)
            completed += 1
            if row.get("hard", 0):
                correct_count += 1
            acc = correct_count / completed if completed else 0.0
            print(
                f"    [rollout-local-hf] {completed}/{total} "
                f"(acc={acc:.3f}) id={item_id} hard={row['hard']}",
                flush=True,
            )
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")
            outf.flush()
    return results


def _run_officeqa_vllm_rollout(
    *,
    items: list[dict],
    out_root: str,
    skill_content: str,
    cfg: dict[str, Any],
    settings: SoftPrefixSettings,
) -> list[dict]:
    """Run OfficeQA rollout against a vLLM-served model using the standard run_batch pipeline.

    This reuses the full local-tool loop (glob/read/grep) from the original SkillOpt
    rollout.  The model is queried via the already-configured target backend (qwen_chat
    pointing at a local vLLM endpoint).  Prefix embeddings are not injected; set
    soft_prefix.trajectory_use_skill to pass init_text_path markdown into the system prompt.
    """
    from skillopt.envs.officeqa.rollout import run_batch

    max_completion_tokens = int(
        settings.trajectory_max_new_tokens
        or cfg.get("max_completion_tokens", 16384)
        or 16384
    )
    return run_batch(
        items,
        out_root,
        skill_content=skill_content,
        workers=int(cfg.get("workers", 4) or 4),
        max_tool_turns=int(cfg.get("max_tool_turns", 12) or 12),
        max_completion_tokens=max_completion_tokens,
        search_mode=str(cfg.get("search_mode", "offline") or "offline"),
        max_queries_per_turn=int(cfg.get("max_queries_per_turn", 4) or 4),
        search_api_url=str(cfg.get("search_api_url", "") or ""),
        search_auth_env=str(cfg.get("search_auth_env", "OFFICEQA_CUSTOM_SEARCH_AUTH") or "OFFICEQA_CUSTOM_SEARCH_AUTH"),
        search_provider=str(cfg.get("search_provider", "duckduckgo") or "duckduckgo"),
        search_max_num_results=int(cfg.get("search_max_num_results", 4) or 4),
        search_timeout_seconds=int(cfg.get("search_timeout_seconds", 20) or 20),
        use_local_tools=True,
        data_dirs=cfg.get("data_dirs"),
    )


def _collect_officeqa_trajectory_examples(
    *,
    prefix_model: Any,
    train_items: list[dict],
    cfg: dict[str, Any],
    settings: SoftPrefixSettings,
    init_text: str,
    out_root: str,
) -> list[dict]:
    rollout_backend = settings.trajectory_rollout_backend
    default_subdir = f"rollout_{rollout_backend}"
    rollout_dir = settings.trajectory_rollout_dir.strip() or os.path.join(out_root, "trajectory_sft", default_subdir)
    rollout_dir = os.path.abspath(rollout_dir)
    skill_content = _trajectory_rollout_skill_content(settings=settings, init_text=init_text)
    rollout_items = _expand_trajectory_rollout_items(
        train_items,
        rollouts_per_task=settings.trajectory_rollouts_per_task,
    )
    _write_rollout_metadata(
        rollout_dir=rollout_dir,
        metadata=_officeqa_rollout_metadata(cfg=cfg, settings=settings, skill_content=skill_content),
    )
    if rollout_backend in {"vllm", "openai"}:
        results = _run_officeqa_vllm_rollout(
            items=rollout_items,
            out_root=rollout_dir,
            skill_content=skill_content,
            cfg=cfg,
            settings=settings,
        )
    elif rollout_backend == "local_hf":
        results = _run_officeqa_local_hf_rollout(
            prefix_model=prefix_model,
            items=rollout_items,
            out_root=rollout_dir,
            skill_content=skill_content,
            cfg=cfg,
            settings=settings,
        )
    else:
        raise ValueError(
            "OfficeQA trajectory SFT supports openai/openai_compatible, vllm, or local_hf rollout backends"
        )
    examples = _build_officeqa_trajectory_examples(
        rollout_dir,
        results,
        settings,
        skill_content=skill_content,
    )
    meta_path = os.path.join(out_root, "trajectory_sft", "examples.jsonl")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    if not examples:
        raise ValueError(
            "OfficeQA trajectory SFT found no successful rollout examples. "
            "Check rollout results, docs paths, or lower soft_prefix.trajectory_min_hard."
        )
    return examples


def evaluate_searchqa_prefix(
    prefix_model: SoftPrefixCausalLM,
    items: list[dict],
    *,
    out_dir: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool = False,
    desc: str = "Eval",
    generator: Any | None = None,
    injection_position: str = "prompt_start",
) -> tuple[float, float, list[dict]]:
    """Generate answers and score them with the existing SearchQA evaluator."""
    injection_position = normalize_prefix_injection_position(injection_position)
    os.makedirs(out_dir, exist_ok=True)
    results: list[dict] = []
    pred_path = os.path.join(out_dir, "results.jsonl")
    if injection_position == "skill_section":
        prompt_pairs = [
            build_searchqa_prompt_and_insert_idx(
                prefix_model.tokenizer,
                item,
                enable_thinking=enable_thinking,
                injection_position=injection_position,
            )
            for item in items
        ]
        prompts = [prompt for prompt, _insert_idx in prompt_pairs]
        prefix_insert_indices = [insert_idx for _prompt, insert_idx in prompt_pairs]
    else:
        prompts = [
            build_searchqa_prompt(prefix_model.tokenizer, item, enable_thinking=enable_thinking)
            for item in items
        ]
        prefix_insert_indices = None
    responses = _generate_prompt_responses(
        prefix_model,
        generator,
        prompts,
        max_prompt_tokens,
        max_new_tokens,
        temperature,
        prefix_insert_indices=prefix_insert_indices,
        desc=desc,
    )
    with open(pred_path, "w", encoding="utf-8") as f:
        for item, response in tqdm(list(zip(items, responses)), desc=desc, unit="ex", leave=False):
            scores = evaluate_searchqa(response, item.get("answers", []))
            row = {
                "id": str(item.get("id", "")),
                "question": item.get("question", ""),
                "response": response,
                "predicted_answer": scores["predicted_answer"],
                "gold_answers": scores["gold_answers"],
                "hard": int(scores["em"]),
                "soft": float(scores["f1"]),
                "em": scores["em"],
                "f1": scores["f1"],
                "sub_em": scores["sub_em"],
            }
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    hard, soft = compute_score(results)
    return hard, soft, results


def evaluate_officeqa_prefix(
    prefix_model: SoftPrefixCausalLM,
    items: list[dict],
    *,
    out_dir: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool = False,
    desc: str = "Eval",
    generator: Any | None = None,
    use_local_tools: bool = False,
    max_tool_turns: int = 12,
    data_dirs: list[str] | str | None = None,
    search_mode: str = "offline",
    injection_position: str = "prompt_start",
) -> tuple[float, float, list[dict]]:
    """Generate answers and score them with the existing OfficeQA evaluator."""
    injection_position = normalize_prefix_injection_position(injection_position)
    tool_generator = generator
    if tool_generator is None and hasattr(prefix_model, "generate_chat_completion"):
        tool_generator = prefix_model
    if use_local_tools and tool_generator is not None and hasattr(tool_generator, "generate_chat_completion"):
        return _evaluate_officeqa_prefix_with_local_tools(
            items,
            out_dir=out_dir,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            desc=desc,
            generator=tool_generator,
            max_tool_turns=max_tool_turns,
            data_dirs=data_dirs,
            search_mode=search_mode,
            injection_position=injection_position,
        )
    os.makedirs(out_dir, exist_ok=True)
    results: list[dict] = []
    pred_path = os.path.join(out_dir, "results.jsonl")
    if injection_position == "skill_section":
        prompt_pairs = [
            build_officeqa_prompt_and_insert_idx(
                prefix_model.tokenizer,
                item,
                enable_thinking=enable_thinking,
                injection_position=injection_position,
            )
            for item in items
        ]
        prompts = [prompt for prompt, _insert_idx in prompt_pairs]
        prefix_insert_indices = [insert_idx for _prompt, insert_idx in prompt_pairs]
    else:
        prompts = [
            build_officeqa_prompt(prefix_model.tokenizer, item, enable_thinking=enable_thinking)
            for item in items
        ]
        prefix_insert_indices = None
    responses = _generate_prompt_responses(
        prefix_model,
        generator,
        prompts,
        max_prompt_tokens,
        max_new_tokens,
        temperature,
        prefix_insert_indices=prefix_insert_indices,
        desc=desc,
    )
    with open(pred_path, "w", encoding="utf-8") as f:
        for item, response in tqdm(list(zip(items, responses)), desc=desc, unit="ex", leave=False):
            predicted_answer = extract_officeqa_answer(response)
            scores = evaluate_officeqa(predicted_answer, item.get("ground_truth", ""))
            row = {
                "id": str(item.get("id", "")),
                "question": item.get("question", ""),
                "response": response,
                "predicted_answer": scores["predicted_answer"],
                "gold_answer": scores["gold_answer"],
                "hard": int(scores["em"]),
                "soft": float(scores["f1"]),
                "em": scores["em"],
                "f1": scores["f1"],
            }
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    hard, soft = compute_score(results)
    return hard, soft, results


def _tool_call_payloads(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = message.get("tool_calls") or []
    return [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]


def _tool_call_name_and_arguments(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return "", {}
    name = str(function.get("name") or "")
    raw_arguments = function.get("arguments") or "{}"
    if isinstance(raw_arguments, dict):
        return name, raw_arguments
    try:
        arguments = json.loads(str(raw_arguments))
    except json.JSONDecodeError:
        arguments = {}
    return name, arguments if isinstance(arguments, dict) else {}


def _answer_from_tool_call(name: str, arguments: dict[str, Any], response: str) -> str:
    if name.strip().lower() != "answer":
        return ""
    for key in ("answer", "final_answer", "final", "result", "value", "text"):
        value = arguments.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return extract_officeqa_answer(response) if response.strip() else ""


def _evaluate_officeqa_prefix_with_local_tools(
    items: list[dict],
    *,
    out_dir: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    desc: str,
    generator: Any,
    max_tool_turns: int,
    data_dirs: list[str] | str | None,
    search_mode: str,
    injection_position: str,
) -> tuple[float, float, list[dict]]:
    os.makedirs(out_dir, exist_ok=True)
    results: list[dict] = []
    pred_path = os.path.join(out_dir, "results.jsonl")
    docs_roots = resolve_docs_roots(data_dirs)
    max_tool_turns = max(1, int(max_tool_turns or 1))

    with open(pred_path, "w", encoding="utf-8") as f:
        for item in tqdm(items, desc=desc, unit="ex", leave=False):
            item_id = str(item.get("id", ""))
            pred_dir = os.path.join(out_dir, "predictions", item_id)
            os.makedirs(pred_dir, exist_ok=True)
            candidate_files = resolve_candidate_files(item.get("source_files", []), docs_roots)
            oracle_context = build_oracle_parsed_pages_context(
                item.get("source_files", []),
                item.get("source_docs", []),
                docs_roots,
                evidence_note="Treat it as primary document evidence and combine it with local document tool evidence when useful.",
            )
            system = build_officeqa_system(
                _SOFT_PREFIX_INSERT_MARKER if injection_position == "skill_section" else "",
                search_mode=search_mode,
                use_local_tools=True,
                max_tool_turns=max_tool_turns,
            )
            user = build_officeqa_user(
                item,
                candidate_files,
                search_mode=search_mode,
                max_tool_turns=max_tool_turns,
                oracle_context=oracle_context,
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            conversation: list[dict[str, Any]] = [{"role": "user", "content": user}]
            allowed_files = [os.path.basename(path) for path in candidate_files]
            final_response = ""
            final_answer = ""
            fail_reason = ""
            last_response_metadata: dict[str, Any] = {}

            for turn in range(1, max_tool_turns + 1):
                message, last_response_metadata = generator.generate_chat_completion(
                    messages,
                    max_prompt_tokens=max_prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    tools=OFFICEQA_TOOL_SCHEMAS,
                    tool_choice="auto",
                    chat_template_kwargs={"enable_thinking": False},
                )
                response = str(message.get("content") or "")
                final_response = response
                tool_calls = _tool_call_payloads(message)
                assistant_message: dict[str, Any] = {"role": "assistant", "content": response}
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)
                conversation.append({"type": "message", "turn": turn, "content": response})

                if tool_calls:
                    answered = False
                    for tool_call in tool_calls:
                        tool_name, arguments = _tool_call_name_and_arguments(tool_call)
                        answer_text = _answer_from_tool_call(tool_name, arguments, response)
                        if answer_text:
                            final_answer = answer_text
                            final_response = response or f"<answer>{answer_text}</answer>"
                            answered = True
                            break
                        cmd, obs = run_officeqa_tool(
                            tool_name,
                            arguments,
                            allowed_roots=docs_roots,
                            allowed_files=allowed_files,
                        )
                        conversation.append({"type": "tool_call", "turn": turn, "cmd": cmd, "obs": obs})
                        messages.append({
                            "role": "tool",
                            "tool_call_id": str(tool_call.get("id", "")),
                            "content": obs,
                        })
                    if answered:
                        break
                    continue
                if "<answer>" in response.lower():
                    final_answer = extract_officeqa_answer(response)
                    break
                if turn == max_tool_turns:
                    fail_reason = f"Exceeded tool-turn budget ({max_tool_turns})"
                else:
                    fail_reason = "Model neither produced a tool request nor a final answer"
                    break

            scores = evaluate_officeqa(final_answer, item.get("ground_truth", "")) if final_answer else {
                "em": 0.0,
                "f1": 0.0,
                "predicted_answer": "",
                "gold_answer": item.get("ground_truth", ""),
            }
            row = {
                "id": item_id,
                "question": item.get("question", ""),
                "response": final_response,
                "predicted_answer": scores["predicted_answer"],
                "gold_answer": scores["gold_answer"],
                "hard": int(scores["em"]),
                "soft": float(scores["f1"]),
                "em": scores["em"],
                "f1": scores["f1"],
                "use_local_tools": True,
                "n_turns": len(conversation),
                "last_finish_reason": last_response_metadata.get("finish_reason", ""),
                "fail_reason": fail_reason,
            }
            with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as sf:
                sf.write(system)
            with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as uf:
                uf.write(user)
            with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as cf:
                json.dump(conversation, cf, ensure_ascii=False, indent=2)
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    hard, soft = compute_score(results)
    return hard, soft, results


def evaluate_livemath_prefix(
    prefix_model: SoftPrefixCausalLM,
    items: list[dict],
    *,
    out_dir: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    use_theorem: bool = False,
    use_sketch: bool = False,
    enable_thinking: bool = False,
    desc: str = "Eval",
    generator: Any | None = None,
    injection_position: str = "prompt_start",
) -> tuple[float, float, list[dict]]:
    """Generate answers and score them with the existing LiveMath evaluator."""
    os.makedirs(out_dir, exist_ok=True)
    results: list[dict] = []
    pred_path = os.path.join(out_dir, "results.jsonl")
    prompt_pairs = [
        build_livemath_prompt_and_insert_idx(
            prefix_model.tokenizer,
            item,
            use_theorem=use_theorem,
            use_sketch=use_sketch,
            enable_thinking=enable_thinking,
            injection_position=injection_position,
        )
        for item in items
    ]
    prompts = [prompt for prompt, _insert_idx in prompt_pairs]
    prefix_insert_indices = [insert_idx for _prompt, insert_idx in prompt_pairs]
    responses = _generate_prompt_responses(
        prefix_model,
        generator,
        prompts,
        max_prompt_tokens,
        max_new_tokens,
        temperature,
        prefix_insert_indices=prefix_insert_indices,
        desc=desc,
    )
    with open(pred_path, "w", encoding="utf-8") as f:
        for item, response in tqdm(list(zip(items, responses)), desc=desc, unit="ex", leave=False):
            scores = evaluate_livemath(response, item["correct_choice"], item["choices"])
            row = {
                "id": str(item.get("id", "")),
                "question": item.get("question", ""),
                "response": response,
                "predicted_answer": scores["predicted_answer"],
                "predicted_label": scores["predicted_label"],
                "predicted_text": scores["predicted_text"],
                "correct_label": scores["correct_label"],
                "correct_text": scores["correct_text"],
                "hard": int(scores["em"]),
                "soft": float(scores["f1"]),
                "em": scores["em"],
                "f1": scores["f1"],
                "sub_em": scores["sub_em"],
            }
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    hard, soft = compute_score(results)
    return hard, soft, results


def evaluate_docvqa_prefix(
    prefix_model: SoftPrefixVisionLM,
    items: list[dict],
    *,
    out_dir: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    image_detail: str = "auto",
    max_image_tokens: int = 0,
    desc: str = "Eval",
    generator: Any | None = None,
) -> tuple[float, float, list[dict]]:
    """Generate answers and score them with the existing DocVQA ANLS evaluator."""
    os.makedirs(out_dir, exist_ok=True)
    results: list[dict] = []
    pred_path = os.path.join(out_dir, "results.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for item in tqdm(items, desc=desc, unit="ex", leave=False):
            messages = build_docvqa_messages(item, image_detail=image_detail)
            if generator is None:
                response = prefix_model.generate_from_messages(
                    messages,
                    max_prompt_tokens=max_prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    max_image_tokens=max_image_tokens,
                )
            else:
                response = generator.generate_from_messages(
                    messages,
                    max_prompt_tokens=max_prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    max_image_tokens=max_image_tokens,
                )
            scores = evaluate_docvqa(response, item.get("answers", []))
            hard = int(scores["anls"] >= 0.999)
            row = {
                "id": str(item.get("id", "")),
                "question": item.get("question", ""),
                "response": response,
                "predicted_answer": scores["predicted_answer"],
                "gold_answers": scores["gold_answers"],
                "hard": hard,
                "soft": float(scores["anls"]),
                "anls": scores["anls"],
                "image_path": item.get("image_path", ""),
            }
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    hard, soft = compute_score(results)
    return hard, soft, results


def _spreadsheet_execution_feedback(err: str) -> str:
    return (
        "The code raised an error during execution:\n\n"
        f"```\n{err[:3000]}\n```\n\n"
        "Please fix the code and return a complete corrected Python script inside a ```python``` block."
    )


def _generate_spreadsheet_response_with_repair(
    generator: Any,
    record: dict[str, Any],
    *,
    pred_dir: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    exec_timeout: int,
    repair_turns: int,
) -> tuple[str, list[dict[str, str]]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": str(record["system"])},
        {"role": "user", "content": str(record["user"])},
    ]
    conversation: list[dict[str, str]] = []
    response = ""
    first_case = record["cases"][0]
    check_path = os.path.join(pred_dir, "_repair_check_pred.xlsx")

    for turn in range(1, max(1, repair_turns) + 1):
        response = generator.generate_from_messages(
            messages,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        conversation.append({"role": "assistant", "content": response})
        messages.append({"role": "assistant", "content": response})

        code = extract_spreadsheet_code(response)
        if not code.strip():
            feedback = "No Python code block was found. Return a complete corrected Python script inside a ```python``` block."
        else:
            ok_exec, err = run_spreadsheet_generated_code(
                code,
                first_case[1],
                check_path,
                timeout=exec_timeout,
            )
            if ok_exec:
                break
            feedback = _spreadsheet_execution_feedback(err)

        if turn >= repair_turns:
            break
        messages.append({"role": "user", "content": feedback})
        conversation.append({"role": "user", "content": feedback})

    return response, conversation


def evaluate_spreadsheet_prefix(
    prefix_model: SoftPrefixCausalLM,
    items: list[dict],
    *,
    out_dir: str,
    data_root: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    exec_timeout: int = 600,
    desc: str = "Eval",
    generator: Any | None = None,
    injection_position: str = "prompt_start",
    repair_turns: int = 1,
) -> tuple[float, float, list[dict]]:
    """Generate SpreadsheetBench Python solutions with the prefix model and score workbooks."""
    injection_position = normalize_prefix_injection_position(injection_position)
    os.makedirs(out_dir, exist_ok=True)
    pred_root = os.path.join(out_dir, "predictions")
    results_path = os.path.join(out_dir, "results.jsonl")
    prompt_records: list[dict[str, Any]] = []
    results_by_id: dict[str, dict] = {}

    for item in items:
        task_id = str(item.get("id", ""))
        instruction = str(item.get("instruction", ""))
        instruction_type = str(item.get("instruction_type", ""))
        answer_position = str(item.get("answer_position", ""))
        answer_sheet = str(item.get("answer_sheet", ""))
        answer_position_eval = (
            f"{answer_sheet}!{answer_position}"
            if answer_position and answer_sheet and "!" not in answer_position
            else answer_position
        )
        task_type = "cell_level" if "cell" in instruction_type.lower() else (
            "sheet_level" if "sheet" in instruction_type.lower() else "other"
        )
        sp = item.get("spreadsheet_path", f"spreadsheet/{task_id}")
        task_dir = str(sp) if os.path.isabs(str(sp)) else os.path.join(data_root, str(sp))
        cases = find_spreadsheet_test_cases(task_dir)
        base_result = {
            "id": task_id,
            "ok": False,
            "instruction_type": instruction_type,
            "task_type": task_type,
            "task_description": instruction,
            "n_cases": len(cases),
            "n_exec_pass": 0,
            "n_pass": 0,
            "soft": 0.0,
            "hard": 0,
            "n_turns": 0,
            "cases": [],
            "response": "",
            "fail_reason": "",
        }
        if not cases:
            base_result["fail_reason"] = "no-test-cases"
            results_by_id[task_id] = base_result
            continue

        first_input = cases[0][1]
        system = build_spreadsheet_system(
            _SOFT_PREFIX_INSERT_MARKER if injection_position == "skill_section" else ""
        )
        user = build_spreadsheet_user(
            instruction,
            first_input,
            instruction_type,
            answer_position_eval,
        )
        prompt, prefix_insert_idx = build_spreadsheet_codegen_prompt_and_insert_idx(
            prefix_model.tokenizer,
            user=user,
            enable_thinking=False,
            injection_position=injection_position,
        )
        prompt_records.append(
            {
                "item": item,
                "result": base_result,
                "cases": cases,
                "system": system,
                "user": user,
                "prompt": prompt,
                "prefix_insert_idx": prefix_insert_idx,
                "answer_position_eval": answer_position_eval,
            }
        )

    use_repair = repair_turns > 1 and generator is not None and hasattr(generator, "generate_from_messages")
    responses: list[str] = []
    if not use_repair:
        responses = _generate_prompt_responses(
            prefix_model,
            generator,
            [record["prompt"] for record in prompt_records],
            max_prompt_tokens,
            max_new_tokens,
            temperature,
            prefix_insert_indices=[record["prefix_insert_idx"] for record in prompt_records],
            desc=desc,
        )

    response_iter = zip(prompt_records, responses) if not use_repair else ((record, "") for record in prompt_records)
    for record, response in tqdm(list(response_iter), desc=desc, unit="ex", leave=False):
        item = record["item"]
        task_id = str(item.get("id", ""))
        result = record["result"]
        pred_dir = os.path.join(pred_root, task_id)
        os.makedirs(pred_dir, exist_ok=True)
        if use_repair:
            response, conversation = _generate_spreadsheet_response_with_repair(
                generator,
                record,
                pred_dir=pred_dir,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                exec_timeout=exec_timeout,
                repair_turns=repair_turns,
            )
        else:
            conversation = [{"role": "assistant", "content": response}]
        result["response"] = response
        result["n_turns"] = sum(1 for message in conversation if message.get("role") == "assistant") or 1

        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(record["system"])
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(record["user"])
        with open(os.path.join(pred_dir, "raw.txt"), "w", encoding="utf-8") as f:
            f.write(response)

        code = extract_spreadsheet_code(response)
        with open(os.path.join(pred_dir, "code.py"), "w", encoding="utf-8") as f:
            f.write(code)

        if not code.strip():
            result["fail_reason"] = "empty-code-block"
            with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
            results_by_id[task_id] = result
            continue

        enrichment_parts: list[str] = []
        for no, input_path, gold_path in record["cases"]:
            pred_path = os.path.join(pred_dir, f"{no}_pred.xlsx")
            ok_exec, err = run_spreadsheet_generated_code(code, input_path, pred_path, timeout=exec_timeout)
            if not ok_exec:
                result["cases"].append({"no": no, "stage": "exec", "ok": False, "error": err[:500]})
                if not result["fail_reason"]:
                    tail = err.strip().splitlines()[-1][:200] if err.strip() else "unknown"
                    result["fail_reason"] = f"exec-error: {tail}"
                enrichment_parts.append(f"## Execution (case {no})\nERROR: {err[:500]}")
                continue
            result["n_exec_pass"] += 1
            try:
                ev = evaluate_spreadsheet(
                    pred_path,
                    gold_path,
                    str(item.get("instruction_type", "")),
                    record["answer_position_eval"],
                )
            except Exception as exc:  # noqa: BLE001
                ev = {"ok": False, "reason": f"eval-exception: {type(exc).__name__}: {exc}"}
            if ev["ok"]:
                result["n_pass"] += 1
            elif not result["fail_reason"]:
                result["fail_reason"] = f"eval-mismatch: {ev.get('reason', '')[:200]}"
            result["cases"].append({"no": no, "stage": "eval", "ok": ev["ok"], "reason": ev.get("reason", "")})
            if record["answer_position_eval"]:
                verify_report = auto_verify_spreadsheet_output(pred_path, gold_path, record["answer_position_eval"])
                enrichment_parts.append(
                    f"## Eval Result (case {no}): {'PASS' if ev['ok'] else 'FAIL'}\n"
                    f"{ev.get('reason', '')}\n\n{verify_report}"
                )

        n_cases = result["n_cases"]
        result["soft"] = (result["n_pass"] / n_cases) if n_cases else 0.0
        result["hard"] = 1 if (n_cases > 0 and result["n_pass"] == n_cases) else 0
        result["ok"] = bool(result["hard"])
        if result["ok"]:
            result["fail_reason"] = ""
        if enrichment_parts:
            conversation.append({
                "role": "system",
                "content": "[POST-EXECUTION VERIFICATION]\n\n" + "\n\n---\n\n".join(enrichment_parts),
            })
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        results_by_id[task_id] = result

    results = [results_by_id.get(str(item.get("id", ""))) for item in items]
    results = [row for row in results if isinstance(row, dict)]
    with open(results_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    hard, soft = compute_score(results)
    return hard, soft, results


def _alfworld_eval_dataset_from_items(items: list[dict]) -> tuple[str, bool]:
    """Infer ALFWorld eval_dataset and is_train from gamefile paths."""
    gamefiles = [str(item.get("gamefile", "")) for item in items]
    if any("/valid_unseen/" in gf for gf in gamefiles):
        return "eval_out_of_distribution", False
    if any("/valid_seen/" in gf for gf in gamefiles):
        return "eval_in_distribution", False
    return "train", True


def evaluate_alfworld_prefix(
    prefix_model: SoftPrefixCausalLM,
    items: list[dict],
    *,
    out_dir: str,
    max_steps: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    desc: str = "Eval",
    generator: Any | None = None,
    injection_position: str = "prompt_start",
) -> tuple[float, float, list[dict]]:
    """Run ALFWorld episodes with the soft-prefix model generating actions step by step.

    At each step the current ``obs["text"]`` (which already encodes recent history
    via ALFWORLD_TEMPLATE) is formatted as a single-turn chat prompt and fed to the
    frozen+prefix model.  No skill text is injected — the soft prefix is the only
    learned signal.
    """
    os.makedirs(out_dir, exist_ok=True)
    injection_position = normalize_prefix_injection_position(injection_position)
    results_path = os.path.join(out_dir, "results.jsonl")

    gamefiles = [str(item.get("gamefile", "")) for item in items]
    eval_dataset, is_train = _alfworld_eval_dataset_from_items(items)
    env_num = len(items)

    env_manager = build_alfworld_env(
        env_num=env_num,
        eval_dataset=eval_dataset,
        seed=42,
        is_train=is_train,
        specific_gamefiles=gamefiles if any(gamefiles) else None,
    )
    try:
        obs, infos = env_manager.reset({})
        env_dones = [False] * env_num
        overall_success = [False] * env_num
        conversations: list[list[dict]] = [[] for _ in range(env_num)]

        env_meta: list[dict] = []
        for i in range(env_num):
            gamefile = infos[i].get("extra.gamefile", "") if isinstance(infos[i], dict) else ""
            anchor = obs["anchor"][i] if "anchor" in obs else ""
            task_idx = anchor.find("Your task is to: ")
            task_desc = anchor[task_idx + len("Your task is to: "):].strip() if task_idx != -1 else ""
            env_meta.append({"gamefile": gamefile, "task_desc": task_desc})

        for step_idx in tqdm(range(max_steps), desc=desc, unit="step", leave=False):
            if all(env_dones):
                break

            active = [i for i in range(env_num) if not env_dones[i]]
            actions = ["None"] * env_num

            prompts_by_env: dict[int, str] = {}
            insert_idx_by_env: dict[int, int | None] = {}
            for i in active:
                user_prompt = obs["text"][i]
                if injection_position == "skill_section":
                    user_prompt = _inject_alfworld_skill_into_prompt(
                        user_prompt,
                        _SOFT_PREFIX_INSERT_MARKER,
                    )
                messages = [
                    {"role": "system", "content": ALFWORLD_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ]
                rendered = _apply_text_chat_template(
                    prefix_model.tokenizer,
                    messages,
                    enable_thinking=False,
                )
                if injection_position == "skill_section":
                    marker_idx = rendered.index(_SOFT_PREFIX_INSERT_MARKER)
                    prompts_by_env[i] = rendered.replace(_SOFT_PREFIX_INSERT_MARKER, "", 1)
                    insert_idx_by_env[i] = len(
                        prefix_model.tokenizer(
                            rendered[:marker_idx],
                            add_special_tokens=False,
                        )["input_ids"]
                    )
                else:
                    prompts_by_env[i] = rendered
                    insert_idx_by_env[i] = None

            if generator is not None:
                active_prompts = [prompts_by_env[i] for i in active]
                kwargs: dict[str, Any] = {
                    "max_prompt_tokens": max_prompt_tokens,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                }
                if any(insert_idx_by_env[i] is not None for i in active):
                    kwargs["prefix_insert_indices"] = [insert_idx_by_env[i] for i in active]
                responses = generator.generate_from_prompts(
                    active_prompts,
                    **kwargs,
                )
                for i, response in zip(active, responses):
                    response = (response or "").strip()
                    actions[i] = response
            else:
                for i in active:
                    response = prefix_model.generate_from_prompt(
                        prompts_by_env[i],
                        max_prompt_tokens=max_prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        prefix_insert_idx=insert_idx_by_env[i],
                    )
                    response = (response or "").strip()
                    actions[i] = response

            model_responses = {i: actions[i] for i in active}
            obs, rewards, dones, infos = env_manager.step(actions)

            for i in active:
                conversations[i].append({
                    "step": step_idx,
                    "model_response": model_responses[i],
                    "env_feedback": obs["anchor"][i] if "anchor" in obs else "",
                    "reward": float(rewards[i]),
                    "done": bool(dones[i]),
                })

            for i in range(env_num):
                if not env_dones[i] and dones[i]:
                    env_dones[i] = True
                    overall_success[i] = bool(infos[i].get("won", False))

        results: list[dict] = []
        pred_dir = os.path.join(out_dir, "predictions")
        with open(results_path, "w", encoding="utf-8") as outf:
            for i, item in enumerate(items):
                won = overall_success[i]
                result = {
                    "id": str(item.get("id", f"env_{i:03d}")),
                    "gamefile": env_meta[i]["gamefile"],
                    "task_description": env_meta[i]["task_desc"],
                    "hard": 1 if won else 0,
                    "soft": 1.0 if won else 0.0,
                    "n_turns": len(conversations[i]),
                    "fail_reason": "" if won else f"not completed in {max_steps} steps",
                    "agent_ok": True,
                }
                results.append(result)
                outf.write(json.dumps(result, ensure_ascii=False) + "\n")

                conv_dir = os.path.join(pred_dir, result["id"])
                os.makedirs(conv_dir, exist_ok=True)
                with open(os.path.join(conv_dir, "conversation.json"), "w", encoding="utf-8") as cf:
                    json.dump(conversations[i], cf, ensure_ascii=False, indent=2)

        hard, soft = compute_score(results)
        return hard, soft, results
    finally:
        close = getattr(env_manager, "close", None)
        if callable(close):
            close()


def _resolve_alfworld_trajectory_rollout_dir(
    settings: "SoftPrefixSettings",
    *,
    out_root: str,
) -> str:
    if settings.trajectory_rollout_dir.strip():
        return os.path.abspath(settings.trajectory_rollout_dir.strip())
    return os.path.abspath(
        os.path.join(out_root, "trajectory_sft", f"rollout_{settings.trajectory_rollout_backend}")
    )


def _alfworld_rollout_metadata(
    *,
    cfg: dict[str, Any],
    settings: "SoftPrefixSettings",
    skill_content: str = "",
) -> dict[str, Any]:
    return {
        "env": "alfworld",
        "rollout_backend": settings.trajectory_rollout_backend,
        "target_backend": cfg.get("target_backend", ""),
        "target_model": cfg.get("target_model", ""),
        "target_qwen_chat_base_url": cfg.get("target_qwen_chat_base_url", ""),
        "target_azure_openai_endpoint": cfg.get("target_azure_openai_endpoint", ""),
        "target_azure_openai_auth_mode": cfg.get("target_azure_openai_auth_mode", ""),
        "max_steps": int(cfg.get("max_steps", 50) or 50),
        "max_completion_tokens": _resolve_alfworld_rollout_max_completion_tokens(cfg, settings),
        "temperature": float(cfg.get("rollout_temperature", 0.4)),
        "invalid_response_retries": int(cfg.get("invalid_response_retries", 3) or 3),
        "trajectory_use_skill": bool(settings.trajectory_use_skill),
        "trajectory_skill_fingerprint": _skill_content_fingerprint(skill_content),
        "trajectory_rollouts_per_task": int(settings.trajectory_rollouts_per_task),
        "split_dir": cfg.get("split_dir", ""),
        "split_seed": int(cfg.get("split_seed", cfg.get("seed", 42)) or 42),
        "train_size": int(cfg.get("train_size", 0) or 0),
    }


def _write_alfworld_rollout_metadata(
    *,
    rollout_dir: str,
    cfg: dict[str, Any],
    settings: "SoftPrefixSettings",
    skill_content: str = "",
) -> None:
    meta_path = os.path.join(rollout_dir, "rollout_meta.json")
    current = _alfworld_rollout_metadata(cfg=cfg, settings=settings, skill_content=skill_content)
    if os.path.exists(meta_path):
        try:
            existing = _load_json_file(meta_path)
        except Exception:  # noqa: BLE001
            existing = None
        if isinstance(existing, dict) and existing != current:
            print(
                f"    [rollout-cache] warning: existing metadata differs in {meta_path}; "
                "reusing cache because soft_prefix.trajectory_rollout_dir was explicit",
                flush=True,
            )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)


def _alfworld_metadata_matches_except_max_steps(existing: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    if not isinstance(existing, dict):
        return False
    existing_cmp = dict(existing)
    current_cmp = dict(current)
    existing_cmp.pop("max_steps", None)
    current_cmp.pop("max_steps", None)
    return existing_cmp == current_cmp


def _load_alfworld_cached_conversation(rollout_dir: str, result_id: str) -> list[dict] | None:
    conv_path = os.path.join(rollout_dir, "predictions", result_id, "conversation.json")
    try:
        with open(conv_path, encoding="utf-8") as f:
            conversation = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    return conversation if isinstance(conversation, list) else None


def _alfworld_cached_timeout_steps(row: dict[str, Any]) -> int | None:
    fail_reason = str(row.get("fail_reason", ""))
    prefix = "Timeout after "
    suffix = " steps"
    if not fail_reason.startswith(prefix) or not fail_reason.endswith(suffix):
        return None
    try:
        return int(fail_reason[len(prefix):-len(suffix)])
    except ValueError:
        return None


def _should_extend_alfworld_cached_rollout(
    row: dict[str, Any],
    *,
    rollout_dir: str,
    rollout_backend: str,
    cached_max_steps: int,
    target_max_steps: int,
) -> bool:
    if target_max_steps <= cached_max_steps:
        return False
    if row.get("rollout_backend") != rollout_backend:
        return False
    if int(row.get("hard", 0) or 0) != 0:
        return False
    if _alfworld_cached_timeout_steps(row) != cached_max_steps:
        return False

    result_id = str(row.get("id", ""))
    if not result_id:
        return False
    conversation = _load_alfworld_cached_conversation(rollout_dir, result_id)
    if not conversation:
        return False
    step_records = [record for record in conversation if int(record.get("step", -1)) >= 0]
    if not step_records:
        return False
    return not bool(step_records[-1].get("done", False))


def _run_alfworld_chat_rollout(
    *,
    items: list[dict],
    out_root: str,
    cfg: dict[str, Any],
    settings: "SoftPrefixSettings",
    skill_content: str = "",
) -> list[dict]:
    """Run ALFWorld rollout via the configured chat target backend.

    Reuses ``run_alfworld_batch`` which already calls ``chat_target``; the target
    backend is configured by ``train_soft_prefix.py`` before this function is reached.
    """
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)
    rollout_backend = settings.trajectory_rollout_backend
    log_prefix = f"rollout-{rollout_backend}"

    max_steps = int(cfg.get("max_steps", 50) or 50)
    max_completion_tokens = _resolve_alfworld_rollout_max_completion_tokens(cfg, settings)
    temperature = float(cfg.get("rollout_temperature", 0.4))
    api_timeout_seconds = float(
        cfg.get("target_qwen_chat_timeout_seconds", cfg.get("qwen_chat_timeout_seconds", 30.0)) or 30.0
    )
    generator = None
    prompt_renderer = None
    if settings.inference_backend == "vllm_prompt_embeds" and str(cfg.get("target_backend", "")).strip().lower() == "qwen_chat":
        from skillopt.softprefix.vllm_prompt_embeds import SoftPrefixVllmClient

        _, _, AutoTokenizer = _import_torch_and_transformers()
        tokenizer = AutoTokenizer.from_pretrained(
            settings.model_name,
            trust_remote_code=settings.trust_remote_code,
        )
        generator = SoftPrefixVllmClient(
            settings.inference_base_url,
            timeout_seconds=settings.inference_timeout_seconds,
        )
        generator.set_prefix([])

        def prompt_renderer(system: str, user: str) -> str:
            return _apply_text_chat_template(
                tokenizer,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                enable_thinking=False,
            )

    current_meta = _alfworld_rollout_metadata(cfg=cfg, settings=settings, skill_content=skill_content)
    meta_path = os.path.join(out_root, "rollout_meta.json")
    existing_meta = None
    if os.path.exists(meta_path):
        try:
            existing_meta = _load_json_file(meta_path)
        except Exception:  # noqa: BLE001
            existing_meta = None
    allow_timeout_extension = _alfworld_metadata_matches_except_max_steps(existing_meta, current_meta)
    _write_alfworld_rollout_metadata(
        rollout_dir=out_root,
        cfg=cfg,
        settings=settings,
        skill_content=skill_content,
    )

    # Resume: load already-completed episode IDs
    done_ids: set[str] = set()
    extend_ids: set[str] = set()
    results: list[dict] = []
    existing_rows: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing_rows.append(row)
                if row.get("rollout_backend") != rollout_backend:
                    continue
                row_id = str(row.get("id"))
                timeout_steps = _alfworld_cached_timeout_steps(row)
                if (
                    allow_timeout_extension
                    and timeout_steps is not None
                    and _should_extend_alfworld_cached_rollout(
                        row,
                        rollout_dir=out_root,
                        rollout_backend=rollout_backend,
                        cached_max_steps=timeout_steps,
                        target_max_steps=max_steps,
                    )
                ):
                    extend_ids.add(row_id)
                    continue
                done_ids.add(row_id)
                results.append(row)

    pending = [item for item in items if str(item.get("id", "")) not in done_ids]
    if not pending:
        print(f"    [{log_prefix}] all {len(results)} episodes already done", flush=True)
        return results
    if results:
        print(f"    [{log_prefix}] resuming: {len(results)}/{len(items)} already done", flush=True)
    if extend_ids:
        print(
            f"    [{log_prefix}] extending {len(extend_ids)} timed-out episodes to max_steps={max_steps}",
            flush=True,
        )

    def run_pending_group(group: list[dict], *, use_cached_conversations: bool) -> list[dict]:
        if not group:
            return []
        workers = max(1, min(int(cfg.get("workers", 8) or 8), len(group)))
        max_api_workers = max(1, int(cfg.get("max_api_workers", 8) or 8))
        group_results: list[dict] = []
        for start in range(0, len(group), workers):
            chunk = group[start:start + workers]
            gamefiles = [str(item.get("gamefile", "")) for item in chunk]
            eval_dataset, is_train = _alfworld_eval_dataset_from_items(chunk)
            result_ids = [
                str(item.get("id", f"env_{start + i:03d}"))
                for i, item in enumerate(chunk)
            ]
            cached_conversations = None
            if use_cached_conversations:
                cached_conversations = {
                    result_id: _load_alfworld_cached_conversation(out_root, result_id) or []
                    for result_id in result_ids
                }

            env_manager = build_alfworld_env(
                env_num=len(chunk),
                eval_dataset=eval_dataset,
                seed=int(cfg.get("seed", 42)) + start,
                is_train=is_train,
                specific_gamefiles=gamefiles if any(gamefiles) else None,
            )
            try:
                group_results.extend(
                    run_alfworld_batch(
                        env_manager,
                        skill_content=skill_content,
                        max_steps=max_steps,
                        out_root=out_root,
                        max_api_workers=min(max_api_workers, len(chunk)),
                        temperature=temperature,
                        max_completion_tokens=max_completion_tokens,
                        api_timeout_seconds=api_timeout_seconds,
                        fallback_on_invalid_response=False,
                        invalid_response_retries=int(cfg.get("invalid_response_retries", 3) or 3),
                        result_ids=result_ids,
                        cached_conversations=cached_conversations,
                        generator=generator,
                        prompt_renderer=prompt_renderer,
                        max_prompt_tokens=settings.max_prompt_tokens,
                    )
                )
            finally:
                close = getattr(env_manager, "close", None)
                if callable(close):
                    close()
        return group_results

    extend_pending = [item for item in pending if str(item.get("id", "")) in extend_ids]
    new_pending = [item for item in pending if str(item.get("id", "")) not in extend_ids]
    batch_results = []
    batch_results.extend(run_pending_group(extend_pending, use_cached_conversations=True))
    batch_results.extend(run_pending_group(new_pending, use_cached_conversations=False))

    # Tag each result with rollout metadata for resume support
    open_mode = "w" if extend_ids else "a"
    retained_existing_rows = [
        row
        for row in existing_rows
        if not (row.get("rollout_backend") == rollout_backend and str(row.get("id")) in extend_ids)
    ]
    with open(results_path, open_mode, encoding="utf-8") as outf:
        if extend_ids:
            for row in retained_existing_rows:
                outf.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in batch_results:
            row["rollout_backend"] = rollout_backend
            results.append(row)
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")

    won = sum(1 for r in batch_results if r.get("hard", 0))
    print(
        f"    [{log_prefix}] done {len(batch_results)} episodes, "
        f"acc={won/max(len(batch_results),1):.3f}",
        flush=True,
    )
    return results


def _build_alfworld_trajectory_examples(
    rollout_dir: str,
    results: list[dict],
    settings: "SoftPrefixSettings",
    skill_content: str = "",
) -> list[dict]:
    """Convert successful ALFWorld rollout episodes into per-step SFT examples.

    Each example is a **single step**: a single-turn (system + obs) prompt paired
    with the model's response as the supervised target.  This aligns training with
    rollout and evaluation, which both present one obs turn at a time rather than
    the full multi-turn episode history.

    ``trajectory_max_examples`` is interpreted as a cap on the number of *episodes*
    (not individual steps) to include, preserving its original semantics.
    """
    examples: list[dict] = []
    episodes_included = 0
    for row in results:
        hard = float(row.get("hard", 0) or 0)
        soft = float(row.get("soft", 0.0) or 0.0)
        if hard < settings.trajectory_min_hard or soft < settings.trajectory_min_soft:
            continue
        task_id = str(row.get("id", "")).strip()
        if not task_id:
            continue

        conv_path = os.path.join(rollout_dir, "predictions", task_id, "conversation.json")
        if not os.path.exists(conv_path):
            continue
        conversation = _load_json_file(conv_path)
        if not isinstance(conversation, list):
            continue

        episode_steps: list[dict] = []
        for record in conversation:
            if not isinstance(record, dict):
                continue
            if record.get("type") == "initial_obs":
                continue
            prompt = str(record.get("prompt", "")).strip()
            model_response = str(record.get("model_response", "")).strip()
            if settings.strip_trajectory_thoughts:
                model_response = _strip_think_blocks(model_response)
            if not prompt or not model_response:
                continue
            if settings.trajectory_max_context_chars > 0:
                prompt = _truncate_text(prompt, settings.trajectory_max_context_chars)
            if settings.injection_position == "skill_section":
                prompt = _replace_skill_content_with_marker(prompt, skill_content)
            elif settings.trajectory_use_skill:
                prompt = _inject_alfworld_skill_into_prompt(prompt, skill_content)
            episode_steps.append({"prompt": prompt, "model_response": model_response})

        if not episode_steps:
            continue

        for step_idx, step in enumerate(episode_steps):
            examples.append(
                {
                    "id": f"{task_id}_step_{step_idx:03d}",
                    "messages": [
                        {"role": "system", "content": ALFWORLD_SYSTEM},
                        {"role": "user", "content": step["prompt"]},
                    ],
                    "target": step["model_response"],
                    "hard": hard,
                    "soft": soft,
                    "task_type": row.get("task_type", ""),
                    "gamefile": row.get("gamefile", ""),
                    "task_description": row.get("task_description", ""),
                    "rollout_dir": rollout_dir,
                }
            )

        episodes_included += 1
        if settings.trajectory_max_examples > 0 and episodes_included >= settings.trajectory_max_examples:
            break
    return examples


def _collect_alfworld_trajectory_examples(
    *,
    items: list[dict],
    cfg: dict[str, Any],
    settings: "SoftPrefixSettings",
    out_root: str,
    init_text: str = "",
) -> list[dict]:
    """Run (or resume) chat-backend rollouts and return filtered SFT examples."""
    rollout_dir = _resolve_alfworld_trajectory_rollout_dir(settings, out_root=out_root)

    if settings.trajectory_rollout_backend == "local_hf":
        raise ValueError("ALFWorld trajectory SFT supports openai/openai_compatible or vllm rollout backends")

    skill_content = _trajectory_rollout_skill_content(settings=settings, init_text=init_text)
    rollout_items = _expand_trajectory_rollout_items(
        items,
        rollouts_per_task=settings.trajectory_rollouts_per_task,
    )
    results = _run_alfworld_chat_rollout(
        items=rollout_items,
        out_root=rollout_dir,
        cfg=cfg,
        settings=settings,
        skill_content=skill_content,
    )

    examples = _build_alfworld_trajectory_examples(
        rollout_dir,
        results,
        settings,
        skill_content=skill_content,
    )
    meta_path = os.path.join(out_root, "trajectory_sft", "examples.jsonl")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    if not examples:
        raise ValueError(
            "ALFWorld trajectory SFT found no successful rollout episodes. "
            "Check vLLM server config or lower soft_prefix.trajectory_min_hard."
        )
    return examples


def _resolve_spreadsheet_trajectory_rollout_dir(
    settings: "SoftPrefixSettings",
    *,
    out_root: str,
) -> str:
    if settings.trajectory_rollout_dir.strip():
        return os.path.abspath(settings.trajectory_rollout_dir.strip())
    return os.path.abspath(
        os.path.join(out_root, "trajectory_sft", f"rollout_{settings.trajectory_rollout_backend}")
    )


def _build_spreadsheet_trajectory_examples(
    rollout_dir: str,
    results: list[dict],
    settings: "SoftPrefixSettings",
    skill_content: str = "",
) -> list[dict]:
    """Convert successful SpreadsheetBench codegen rollouts into prompt/target examples."""
    examples: list[dict] = []
    for row in results:
        hard = float(row.get("hard", 0) or 0)
        soft = float(row.get("soft", 0.0) or 0.0)
        if hard < settings.trajectory_min_hard or soft < settings.trajectory_min_soft:
            continue
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            continue

        pred_dir = os.path.join(rollout_dir, "predictions", item_id)
        system_path = os.path.join(pred_dir, "target_system_prompt.txt")
        user_path = os.path.join(pred_dir, "target_user_prompt.txt")
        conv_path = os.path.join(pred_dir, "conversation.json")
        raw_path = os.path.join(pred_dir, "raw.txt")

        system = _read_text_if_exists(system_path).strip()
        if settings.injection_position == "skill_section":
            system = _replace_skill_content_with_marker(system, skill_content)
        user = _read_text_if_exists(user_path).strip()
        if not system or not user:
            continue

        conversation = _load_json_file(conv_path) if os.path.exists(conv_path) else []
        if not isinstance(conversation, list):
            conversation = []

        assistant_indices: list[int] = []
        for idx, message in enumerate(conversation):
            if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant":
                assistant_indices.append(idx)

        if assistant_indices:
            indices_to_train = (
                [assistant_indices[-1]]
                if settings.train_on_final_only
                else assistant_indices
            )
            for turn_num, assistant_idx in enumerate(indices_to_train, start=1):
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                for message in conversation[:assistant_idx]:
                    if not isinstance(message, dict):
                        continue
                    role = str(message.get("role", "")).lower()
                    if role not in {"assistant", "user"}:
                        continue
                    content = str(message.get("content") or "").strip()
                    if content:
                        messages.append({"role": role, "content": content})
                target = str(conversation[assistant_idx].get("content") or "").strip()
                if settings.strip_trajectory_thoughts:
                    target = _strip_think_blocks(target)
                if not target:
                    continue
                examples.append(
                    {
                        "id": (
                            item_id
                            if settings.train_on_final_only or len(assistant_indices) == 1
                            else f"{item_id}__turn_{turn_num:02d}"
                        ),
                        "messages": messages,
                        "target": target,
                        "hard": hard,
                        "soft": soft,
                        "task_type": row.get("task_type", ""),
                        "task_description": row.get("task_description", ""),
                        "rollout_dir": rollout_dir,
                    }
                )
                if settings.trajectory_max_examples > 0 and len(examples) >= settings.trajectory_max_examples:
                    break
        else:
            target = _read_text_if_exists(raw_path).strip()
            if settings.strip_trajectory_thoughts:
                target = _strip_think_blocks(target)
            if not target:
                continue
            examples.append(
                {
                    "id": item_id,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "target": target,
                    "hard": hard,
                    "soft": soft,
                    "task_type": row.get("task_type", ""),
                    "task_description": row.get("task_description", ""),
                    "rollout_dir": rollout_dir,
                }
            )
        if settings.trajectory_max_examples > 0 and len(examples) >= settings.trajectory_max_examples:
            break
    return examples


def _spreadsheet_task_type(instruction_type: str) -> str:
    lowered = instruction_type.lower()
    if "cell" in lowered:
        return "cell_level"
    if "sheet" in lowered:
        return "sheet_level"
    return "other"


def _spreadsheet_cached_success_case_count(conversation: list[dict]) -> int:
    verification_blocks = [
        str(message.get("content") or "")
        for message in conversation
        if isinstance(message, dict)
        and str(message.get("role", "")).lower() == "system"
        and "[POST-EXECUTION VERIFICATION]" in str(message.get("content") or "")
    ]
    if not verification_blocks:
        return 0
    verification_text = "\n".join(verification_blocks)
    if "FAIL" in verification_text:
        return 0
    pass_count = len(re.findall(r"Eval Result \(case \d+\): PASS", verification_text))
    return max(1, pass_count) if "PASS" in verification_text else 0


def _reconstruct_spreadsheet_result_from_prediction(
    *,
    rollout_dir: str,
    item: dict,
) -> dict[str, Any] | None:
    item_id = str(item.get("id", "")).strip()
    if not item_id:
        return None
    pred_dir = os.path.join(rollout_dir, "predictions", item_id)
    system = _read_text_if_exists(os.path.join(pred_dir, "target_system_prompt.txt")).strip()
    user = _read_text_if_exists(os.path.join(pred_dir, "target_user_prompt.txt")).strip()
    if not system or not user:
        return None
    conv_path = os.path.join(pred_dir, "conversation.json")
    if not os.path.exists(conv_path):
        return None
    conversation = _load_json_file(conv_path)
    if not isinstance(conversation, list):
        return None
    n_pass = _spreadsheet_cached_success_case_count(conversation)
    if n_pass <= 0:
        return None

    response = ""
    for message in conversation:
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant":
            response = str(message.get("content") or "").strip()
    if not response:
        response = _read_text_if_exists(os.path.join(pred_dir, "raw.txt")).strip()
    if not response:
        return None

    instruction_type = str(item.get("instruction_type", ""))
    return {
        "id": item_id,
        "ok": True,
        "task_type": _spreadsheet_task_type(instruction_type),
        "task_description": str(item.get("instruction", "")),
        "n_cases": n_pass,
        "n_pass": n_pass,
        "soft": 1.0,
        "hard": 1,
        "n_turns": sum(
            1
            for message in conversation
            if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant"
        ),
        "cases": [{"stage": "eval", "ok": True} for _ in range(n_pass)],
        "response": response,
        "fail_reason": "",
        "cache_source": "predictions",
    }


def _reconstruct_spreadsheet_results_jsonl_from_predictions(
    *,
    rollout_dir: str,
    items: list[dict],
) -> None:
    results_path = os.path.join(rollout_dir, "results.jsonl")
    existing: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_id = str(row.get("id", "")).strip()
                if row_id:
                    done_ids.add(row_id)
                    existing.append(row)

    reconstructed: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item.get("id", "")).strip()
        if not item_id or item_id in done_ids:
            continue
        row = _reconstruct_spreadsheet_result_from_prediction(rollout_dir=rollout_dir, item=item)
        if row is not None:
            done_ids.add(item_id)
            reconstructed.append(row)

    if not reconstructed:
        return
    os.makedirs(rollout_dir, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        for row in existing + reconstructed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"    [rollout-cache] reconstructed {len(reconstructed)} SpreadsheetBench results from predictions",
        flush=True,
    )


def _collect_spreadsheet_trajectory_examples(
    *,
    items: list[dict],
    cfg: dict[str, Any],
    settings: "SoftPrefixSettings",
    out_root: str,
    init_text: str = "",
) -> list[dict]:
    """Run (or resume) SpreadsheetBench codegen rollouts and return SFT examples."""
    if settings.trajectory_rollout_backend == "local_hf":
        raise ValueError(
            "SpreadsheetBench trajectory SFT supports openai/openai_compatible or vllm rollout backends"
        )

    rollout_dir = _resolve_spreadsheet_trajectory_rollout_dir(settings, out_root=out_root)
    skill_content = _trajectory_rollout_skill_content(settings=settings, init_text=init_text)
    rollout_items = _expand_trajectory_rollout_items(
        items,
        rollouts_per_task=getattr(settings, "trajectory_rollouts_per_task", 1),
    )
    _write_rollout_metadata(
        rollout_dir=rollout_dir,
        metadata=_spreadsheet_rollout_metadata(cfg=cfg, settings=settings, skill_content=skill_content),
    )
    _reconstruct_spreadsheet_results_jsonl_from_predictions(
        rollout_dir=rollout_dir,
        items=rollout_items,
    )
    results = run_spreadsheet_batch_codegen(
        items=rollout_items,
        data_root=str(cfg.get("data_root", "")),
        out_root=rollout_dir,
        skill_content=skill_content,
        mode=str(cfg.get("mode", "multi") or "multi"),
        max_turns=int(cfg.get("max_turns", 5) or 5),
        max_completion_tokens=int(settings.trajectory_max_new_tokens or cfg.get("max_completion_tokens", 16384) or 16384),
        max_api_workers=max(1, int(cfg.get("workers", cfg.get("max_api_workers", 8)) or 8)),
        task_timeout=int(cfg.get("exec_timeout", cfg.get("task_timeout", 600)) or 600),
        use_eval_feedback=bool(cfg.get("use_eval_feedback", False)),
    )

    examples = _build_spreadsheet_trajectory_examples(
        rollout_dir,
        results,
        settings,
        skill_content=skill_content,
    )
    meta_path = os.path.join(out_root, "trajectory_sft", "examples.jsonl")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    if not examples:
        raise ValueError(
            "SpreadsheetBench trajectory SFT found no successful rollout examples. "
            "Check target backend config or lower soft_prefix.trajectory_min_hard."
        )
    return examples


def _build_dataloader(env: str, cfg: dict[str, Any], seed: int) -> Any:
    common = {
        "split_dir": str(cfg.get("split_dir", "")),
        "data_path": str(cfg.get("data_path", "")),
        "split_mode": str(cfg.get("split_mode", "split_dir")),
        "split_ratio": str(cfg.get("split_ratio", "2:1:7")),
        "split_seed": int(cfg.get("split_seed", seed)),
        "split_output_dir": str(cfg.get("split_output_dir", "")),
        "seed": seed,
        "limit": int(cfg.get("limit", 0) or 0),
    }
    if env == "alfworld":
        return ALFWorldDataLoader(**common)
    if env == "searchqa":
        return SearchQADataLoader(**common)
    if env == "officeqa":
        return OfficeQADataLoader(**common)
    if env == "livemathematicianbench":
        return LiveMathematicianBenchDataLoader(
            **common,
            shuffle_choices=bool(cfg.get("shuffle_choices", True)),
        )
    if env == "docvqa":
        return DocVQADataLoader(**common)
    if env == "spreadsheetbench":
        return SpreadsheetBenchDataLoader(
            **common,
            data_root=str(cfg.get("data_root", "")),
        )
    raise ValueError(f"Unsupported soft-prefix env: {env!r}")


def _build_prefix_model(env: str, settings: SoftPrefixSettings, init_text: str) -> SoftPrefixCausalLM | SoftPrefixVisionLM:
    architecture = settings.architecture.strip().lower()
    if architecture == "auto":
        use_vision_model = env == "docvqa"
    elif architecture in {"vision_lm", "vlm", "multimodal"}:
        use_vision_model = True
    elif architecture in {"causal_lm", "text", "text_lm"}:
        use_vision_model = False
    else:
        raise ValueError(
            "soft_prefix.architecture must be one of auto, causal_lm, or vision_lm"
        )
    model_cls = SoftPrefixVisionLM if use_vision_model else SoftPrefixCausalLM
    return model_cls(
        settings.model_name,
        prefix_length=settings.prefix_length,
        num_soft_skills=settings.num_soft_skills,
        init_text=init_text,
        init_strategy=settings.init_strategy,
        torch_dtype=settings.torch_dtype,
        device=settings.device,
        trust_remote_code=settings.trust_remote_code,
    )


def _build_lora_model(env: str, settings: LoraSettings) -> LoraCausalLM | LoraVisionLM:
    architecture = settings.architecture.strip().lower()
    if architecture == "auto":
        use_vision_model = env == "docvqa"
    elif architecture in {"vision_lm", "vlm", "multimodal"}:
        use_vision_model = True
    elif architecture in {"causal_lm", "text", "text_lm"}:
        use_vision_model = False
    else:
        raise ValueError("lora.architecture must be one of auto, causal_lm, or vision_lm")
    return LoraVisionLM(settings) if use_vision_model else LoraCausalLM(settings)


def _build_dataset(
    env: str,
    items: list[dict],
    prefix_model: Any,
    cfg: dict[str, Any],
    settings: SoftPrefixSettings,
) -> AlfWorldTrajectoryPrefixDataset | SearchQAPrefixDataset | OfficeQAPrefixDataset | TextTrajectoryPrefixDataset | LiveMathPrefixDataset | DocVQAPrefixDataset:
    if env == "alfworld":
        # Per-step items (from trajectory_sft) have a "messages" key and use a
        # single-turn prompt, matching rollout and evaluation format.
        # Legacy per-episode items (with "steps") fall back to the multi-turn dataset.
        if items and "messages" in items[0]:
            return TextTrajectoryPrefixDataset(
                items,
                prefix_model.tokenizer,
                max_prompt_tokens=settings.max_prompt_tokens,
                max_target_tokens=settings.max_target_tokens,
                injection_position=settings.injection_position,
            )
        return AlfWorldTrajectoryPrefixDataset(
            items,
            prefix_model.tokenizer,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_target_tokens=settings.max_target_tokens,
        )
    if env == "searchqa":
        return SearchQAPrefixDataset(
            items,
            prefix_model.tokenizer,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_target_tokens=settings.max_target_tokens,
            injection_position=settings.injection_position,
        )
    if env == "officeqa":
        if settings.training_data.strip().lower() in {"trajectory", "trajectory_sft"}:
            return TextTrajectoryPrefixDataset(
                items,
                prefix_model.tokenizer,
                max_prompt_tokens=settings.max_prompt_tokens,
                max_target_tokens=settings.max_target_tokens,
                injection_position=settings.injection_position,
            )
        return OfficeQAPrefixDataset(
            _resolve_officeqa_training_items(items, cfg),
            prefix_model.tokenizer,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_target_tokens=settings.max_target_tokens,
        )
    if env == "livemathematicianbench":
        return LiveMathPrefixDataset(
            items,
            prefix_model.tokenizer,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_target_tokens=settings.max_target_tokens,
            use_theorem=bool(cfg.get("use_theorem", False)),
            use_sketch=bool(cfg.get("use_sketch", False)),
            injection_position=settings.injection_position,
        )
    if env == "docvqa":
        if not hasattr(prefix_model, "processor"):
            raise TypeError("DocVQA requires a vision-language adapter model")
        return DocVQAPrefixDataset(
            items,
            prefix_model.processor,
            prefix_model.tokenizer,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_target_tokens=settings.max_target_tokens,
            image_detail=str(cfg.get("image_detail", "auto")),
            max_image_tokens=settings.docvqa_max_image_tokens,
        )
    if env == "spreadsheetbench":
        if settings.training_data.strip().lower() not in {"trajectory", "trajectory_sft"}:
            raise ValueError("SpreadsheetBench soft-prefix training requires soft_prefix.training_data=trajectory_sft")
        return TextTrajectoryPrefixDataset(
            items,
            prefix_model.tokenizer,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_target_tokens=settings.max_target_tokens,
            injection_position=settings.injection_position,
        )
    raise ValueError(f"Unsupported soft-prefix env: {env!r}")


def _resolve_officeqa_training_items(items: list[dict], cfg: dict[str, Any]) -> list[dict]:
    docs_roots = resolve_docs_roots(cfg.get("data_dirs"))
    resolved_items: list[dict] = []
    for item in items:
        resolved_item = dict(item)
        if not resolved_item.get("resolved_source_paths"):
            resolved_item["resolved_source_paths"] = resolve_candidate_files(
                resolved_item.get("source_files", []),
                docs_roots,
            )
        resolved_items.append(resolved_item)
    return resolved_items


class _PlainInferenceModel:
    """Adapter that reuses the same loaded model without injecting soft-prefix rows."""

    def __init__(self, prefix_model: Any) -> None:
        self._prefix_model = prefix_model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._prefix_model, name)

    def generate_from_prompt(self, prompt: str, **kwargs) -> str:
        kwargs["use_prefix"] = False
        return self._prefix_model.generate_from_prompt(prompt, **kwargs)

    def generate_from_prompts(self, prompts: list[str], **kwargs) -> list[str]:
        kwargs["use_prefix"] = False
        return self._prefix_model.generate_from_prompts(prompts, **kwargs)

    def generate_from_messages(self, messages: list[dict], **kwargs) -> str:
        kwargs["use_prefix"] = False
        return self._prefix_model.generate_from_messages(messages, **kwargs)


def _generate_prompt_responses(
    prefix_model: Any,
    generator: Any | None,
    prompts: list[str],
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    prefix_insert_indices: list[int | None] | None = None,
    desc: str = "Eval",
) -> list[str]:
    has_insert_indices = bool(prefix_insert_indices) and any(
        idx is not None for idx in prefix_insert_indices
    )
    if generator is not None:
        kwargs: dict[str, Any] = {
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if has_insert_indices:
            kwargs["prefix_insert_indices"] = prefix_insert_indices
        return generator.generate_from_prompts(prompts, **kwargs)
    indices = prefix_insert_indices or [None] * len(prompts)
    responses: list[str] = []
    progress_desc = f"{desc} Generate" if desc else "Generate"
    for prompt, insert_idx in zip(tqdm(prompts, desc=progress_desc, unit="ex", leave=False), indices):
        responses.append(
            prefix_model.generate_from_prompt(
                prompt,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prefix_insert_idx=insert_idx,
            )
        )
    return responses


def _build_vllm_eval_generator(prefix_model: Any, settings: SoftPrefixSettings, *, plain: bool) -> Any | None:
    if settings.inference_backend != "vllm_prompt_embeds":
        return None
    from skillopt.softprefix.vllm_prompt_embeds import SoftPrefixVllmClient

    if not hasattr(prefix_model, "prefix_embeddings"):
        raise TypeError("vLLM prompt-embeds inference requires a soft-prefix model")
    generator = SoftPrefixVllmClient(
        settings.inference_base_url,
        timeout_seconds=settings.inference_timeout_seconds,
    )
    prefix = prefix_model.prefix_embeddings.detach()
    if prefix.dim() == 3:
        prefix = prefix.flatten(0, 1)
    if plain:
        prefix = prefix[:0]
    generator.set_prefix(prefix, injection_position=settings.injection_position)
    return generator


def _evaluate_prefix(
    env: str,
    prefix_model: Any,
    items: list[dict],
    *,
    cfg: dict[str, Any],
    settings: SoftPrefixSettings,
    out_dir: str,
    desc: str,
    vllm_plain: bool = False,
) -> tuple[float, float, list[dict]]:
    generator = _build_vllm_eval_generator(prefix_model, settings, plain=vllm_plain)
    if env == "alfworld":
        return evaluate_alfworld_prefix(
            prefix_model,
            items,
            out_dir=out_dir,
            max_steps=int(cfg.get("max_steps", 50) or 50),
            max_prompt_tokens=settings.max_prompt_tokens,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.generation_temperature,
            desc=desc,
            generator=generator,
            injection_position=settings.injection_position,
        )
    if env == "searchqa":
        return evaluate_searchqa_prefix(
            prefix_model,
            items,
            out_dir=out_dir,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.generation_temperature,
            desc=desc,
            generator=generator,
            injection_position=settings.injection_position,
        )
    if env == "officeqa":
        return evaluate_officeqa_prefix(
            prefix_model,
            items,
            out_dir=out_dir,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.generation_temperature,
            desc=desc,
            generator=generator,
            use_local_tools=bool(cfg.get("use_local_tools", False)),
            max_tool_turns=int(cfg.get("max_tool_turns", 12) or 12),
            data_dirs=cfg.get("data_dirs"),
            search_mode=str(cfg.get("search_mode", "offline") or "offline"),
            injection_position=settings.injection_position,
        )
    if env == "livemathematicianbench":
        return evaluate_livemath_prefix(
            prefix_model,
            items,
            out_dir=out_dir,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.generation_temperature,
            use_theorem=bool(cfg.get("use_theorem", False)),
            use_sketch=bool(cfg.get("use_sketch", False)),
            desc=desc,
            generator=generator,
            injection_position=settings.injection_position,
        )
    if env == "docvqa":
        if not hasattr(prefix_model, "generate_from_messages"):
            raise TypeError("DocVQA eval requires a vision-language adapter model")
        return evaluate_docvqa_prefix(
            prefix_model,
            items,
            out_dir=out_dir,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.generation_temperature,
            image_detail=str(cfg.get("image_detail", "auto")),
            max_image_tokens=settings.docvqa_max_image_tokens,
            desc=desc,
            generator=generator,
        )
    if env == "spreadsheetbench":
        return evaluate_spreadsheet_prefix(
            prefix_model,
            items,
            out_dir=out_dir,
            data_root=str(cfg.get("data_root", "")),
            max_prompt_tokens=settings.max_prompt_tokens,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.generation_temperature,
            exec_timeout=int(cfg.get("exec_timeout", cfg.get("task_timeout", 600)) or 600),
            desc=desc,
            generator=generator,
            injection_position=settings.injection_position,
            repair_turns=int(cfg.get("max_turns", cfg.get("max_tool_turns", 1)) or 1),
        )
    raise ValueError(f"Unsupported soft-prefix env: {env!r}")


def _evaluate_plain_baseline(
    env: str,
    prefix_model: Any,
    items: list[dict],
    *,
    cfg: dict[str, Any],
    settings: SoftPrefixSettings,
    out_dir: str,
    desc: str,
) -> tuple[float, float, list[dict]]:
    plain_model = _PlainInferenceModel(prefix_model)
    if env == "alfworld":
        generator = _build_vllm_eval_generator(prefix_model, settings, plain=True)
        return evaluate_alfworld_prefix(
            plain_model,
            items,
            out_dir=out_dir,
            max_steps=int(cfg.get("max_steps", 50) or 50),
            max_prompt_tokens=settings.max_prompt_tokens,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.generation_temperature,
            desc=desc,
            generator=generator,
        )
    return _evaluate_prefix(
        env,
        plain_model,
        items,
        cfg=cfg,
        settings=settings,
        out_dir=out_dir,
        desc=desc,
        vllm_plain=True,
    )


def _save_torch_state(torch, adapter_model: Any, path: str) -> None:
    torch.save(adapter_model.state_dict(), path)


def _load_torch_state(torch, adapter_model: Any, path: str) -> None:
    adapter_model.load_state_dict(torch.load(path, map_location="cpu"))


def _save_lora_adapter(_torch, adapter_model: Any, path: str) -> None:
    adapter_model.save_adapter(path)


def _load_lora_adapter(_torch, adapter_model: Any, path: str) -> None:
    adapter_model.load_adapter(path)


def _train_adapter(
    *,
    cfg: dict[str, Any],
    settings: Any,
    adapter_model: Any,
    best_path: str,
    latest_path: str,
    best_summary_key: str,
    latest_summary_key: str,
    save_checkpoint,
    load_checkpoint,
) -> dict[str, Any]:
    """Train a prompt-conditioned adapter while preserving SkillOpt split/eval settings."""
    torch, _, _ = _import_torch_and_transformers()
    env = str(cfg.get("env", "")).strip()
    if env not in {"alfworld", "searchqa", "officeqa", "livemathematicianbench", "docvqa", "spreadsheetbench"}:
        raise ValueError(
            "adapter training supports env=alfworld, env=searchqa, env=officeqa, "
            "env=livemathematicianbench, env=docvqa, or env=spreadsheetbench"
        )

    out_root = os.path.abspath(str(cfg["out_root"]))
    os.makedirs(out_root, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    dataloader = _build_dataloader(env, cfg, seed)
    dataloader.setup(cfg)

    init_text = _load_init_text(settings.init_text_path or str(cfg.get("skill_init", "")))

    train_size = int(cfg.get("train_size", 0) or 0)
    batch_size = int(cfg.get("batch_size", 1))
    num_epochs = int(cfg.get("num_epochs", 1))
    accumulation = int(cfg.get("accumulation", 1))
    sel_items = _items_for_eval(dataloader, "valid_seen", int(cfg.get("sel_env_num", 0) or 0), seed)
    test_items = _items_for_eval(dataloader, "valid_unseen", int(cfg.get("test_env_num", 0) or 0), seed)
    gate_metric = str(cfg.get("gate_metric", "hard") or "hard")
    gate_mixed_weight = float(cfg.get("gate_mixed_weight", 0.5) or 0.5)
    rewind_ckpt = bool(getattr(settings, "rewind_ckpt", False))
    best_score = -math.inf
    history: list[dict[str, Any]] = []

    summary: dict[str, Any] = {
        "best_score": None,
        best_summary_key: "",
        latest_summary_key: "",
        "history": history,
    }
    if getattr(settings, "eval_init_prefix", False):
        if settings.eval_init_val and sel_items:
            init_val_dir = os.path.join(out_root, "eval", "init", "valid_seen")
            init_val_hard, init_val_soft, _ = _evaluate_prefix(
                env,
                adapter_model,
                sel_items,
                cfg=cfg,
                settings=settings,
                out_dir=init_val_dir,
                desc="  Init Val",
            )
            summary["init_valid_seen_hard"] = init_val_hard
            summary["init_valid_seen_soft"] = init_val_soft
            print(f"[init prefix] valid_hard={init_val_hard:.4f} valid_soft={init_val_soft:.4f}", flush=True)
        if cfg.get("eval_test", True) and test_items:
            init_test_dir = os.path.join(out_root, "eval", "init", "valid_unseen")
            init_test_hard, init_test_soft, _ = _evaluate_prefix(
                env,
                adapter_model,
                test_items,
                cfg=cfg,
                settings=settings,
                out_dir=init_test_dir,
                desc="  Init Test",
            )
            summary["init_test_hard"] = init_test_hard
            summary["init_test_soft"] = init_test_soft
            print(f"[init prefix] test_hard={init_test_hard:.4f} test_soft={init_test_soft:.4f}", flush=True)
    if getattr(settings, "eval_plain_baseline", False):
        if sel_items:
            plain_val_dir = os.path.join(out_root, "eval", "plain", "valid_seen")
            plain_val_hard, plain_val_soft, _ = _evaluate_plain_baseline(
                env,
                adapter_model,
                sel_items,
                cfg=cfg,
                settings=settings,
                out_dir=plain_val_dir,
                desc="  Plain Val",
            )
            summary["plain_valid_seen_hard"] = plain_val_hard
            summary["plain_valid_seen_soft"] = plain_val_soft
            print(f"[plain baseline] valid_hard={plain_val_hard:.4f} valid_soft={plain_val_soft:.4f}", flush=True)
        if cfg.get("eval_test", True) and test_items:
            plain_test_dir = os.path.join(out_root, "eval", "plain", "valid_unseen")
            plain_test_hard, plain_test_soft, _ = _evaluate_plain_baseline(
                env,
                adapter_model,
                test_items,
                cfg=cfg,
                settings=settings,
                out_dir=plain_test_dir,
                desc="  Plain Test",
            )
            summary["plain_test_hard"] = plain_test_hard
            summary["plain_test_soft"] = plain_test_soft
            print(f"[plain baseline] test_hard={plain_test_hard:.4f} test_soft={plain_test_soft:.4f}", flush=True)
    checkpoint_path = str(getattr(settings, "checkpoint_path", "") or "").strip()
    if checkpoint_path:
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"soft_prefix.checkpoint_path not found: {checkpoint_path}")
        load_checkpoint(torch, adapter_model, checkpoint_path)
        summary[best_summary_key] = checkpoint_path
        if sel_items:
            checkpoint_val_dir = os.path.join(out_root, "eval", "checkpoint", "valid_seen")
            checkpoint_val_hard, checkpoint_val_soft, _ = _evaluate_prefix(
                env,
                adapter_model,
                sel_items,
                cfg=cfg,
                settings=settings,
                out_dir=checkpoint_val_dir,
                desc="  Checkpoint Val",
            )
            summary["checkpoint_valid_seen_hard"] = checkpoint_val_hard
            summary["checkpoint_valid_seen_soft"] = checkpoint_val_soft
            print(
                f"[checkpoint prefix] valid_hard={checkpoint_val_hard:.4f} "
                f"valid_soft={checkpoint_val_soft:.4f}",
                flush=True,
            )
        if cfg.get("eval_test", True) and test_items:
            checkpoint_test_dir = os.path.join(out_root, "eval", "checkpoint", "valid_unseen")
            checkpoint_test_hard, checkpoint_test_soft, _ = _evaluate_prefix(
                env,
                adapter_model,
                test_items,
                cfg=cfg,
                settings=settings,
                out_dir=checkpoint_test_dir,
                desc="  Checkpoint Test",
            )
            summary["checkpoint_test_hard"] = checkpoint_test_hard
            summary["checkpoint_test_soft"] = checkpoint_test_soft
            print(
                f"[checkpoint prefix] test_hard={checkpoint_test_hard:.4f} "
                f"test_soft={checkpoint_test_soft:.4f}",
                flush=True,
            )
    if num_epochs <= 0:
        with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    train_items = _items_for_eval(dataloader, "train", 0, seed)
    if train_size > 0:
        train_items = train_items[:train_size]

    training_data = settings.training_data.strip().lower()
    if training_data in {"trajectory", "trajectory_sft"}:
        if env == "alfworld":
            train_items = _collect_alfworld_trajectory_examples(
                items=train_items,
                cfg=cfg,
                settings=settings,
                out_root=out_root,
                init_text=init_text,
            )
            print(f"  [trajectory_sft] using {len(train_items)} single-step ALFWorld SFT examples", flush=True)
        elif env == "officeqa":
            train_items = _collect_officeqa_trajectory_examples(
                prefix_model=adapter_model,
                train_items=train_items,
                cfg=cfg,
                settings=settings,
                init_text=init_text,
                out_root=out_root,
            )
            print(f"  [trajectory_sft] using {len(train_items)} successful OfficeQA rollout examples", flush=True)
        elif env == "spreadsheetbench":
            train_items = _collect_spreadsheet_trajectory_examples(
                items=train_items,
                cfg=cfg,
                settings=settings,
                out_root=out_root,
                init_text=init_text,
            )
            print(f"  [trajectory_sft] using {len(train_items)} successful SpreadsheetBench rollout examples", flush=True)
        else:
            raise ValueError(
                f"soft_prefix.training_data=trajectory_sft is not supported for env={env!r}. "
                "Supported: alfworld, officeqa, spreadsheetbench."
            )

    dataset = _build_dataset(env, train_items, adapter_model, cfg, settings)
    collator = PrefixBatchCollator(adapter_model.tokenizer.pad_token_id)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(
        adapter_model.trainable_parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        t0 = time.time()
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        step_bar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"  Train {epoch}/{num_epochs}", unit="batch", leave=False)
        for step, batch in step_bar:
            tensor_batch = _batch_to_tensors(torch, batch, adapter_model.device)
            outputs = adapter_model.forward(tensor_batch)
            loss = outputs.loss / accumulation
            loss.backward()
            if step % accumulation == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            cur_loss = float(outputs.loss.detach().cpu())
            losses.append(cur_loss)
            step_bar.set_postfix(loss=f"{cur_loss:.4f}")

        val_dir = os.path.join(out_root, "eval", f"epoch_{epoch:02d}", "valid_seen")
        val_hard, val_soft, _ = _evaluate_prefix(
            env,
            adapter_model,
            sel_items,
            cfg=cfg,
            settings=settings,
            out_dir=val_dir,
            desc=f"  Val   {epoch}/{num_epochs}",
        )
        val_score = select_gate_score(val_hard, val_soft, gate_metric, gate_mixed_weight)
        action = "accept_new_best" if val_score > best_score else "reject"
        if val_score > best_score:
            best_score = val_score
            save_checkpoint(torch, adapter_model, best_path)
        save_checkpoint(torch, adapter_model, latest_path)
        if action == "reject" and rewind_ckpt and os.path.exists(best_path):
            load_checkpoint(torch, adapter_model, best_path)
        rec = {
            "epoch": epoch,
            "loss": sum(losses) / max(len(losses), 1),
            "valid_seen_hard": val_hard,
            "valid_seen_soft": val_soft,
            "gate_metric": gate_metric,
            "valid_seen_score": val_score,
            "action": action,
            "wall_time_s": round(time.time() - t0, 1),
        }
        history.append(rec)
        with open(os.path.join(out_root, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        epoch_bar.set_postfix(
            loss=f"{rec['loss']:.4f}",
            val_hard=f"{val_hard:.4f}",
            val_soft=f"{val_soft:.4f}",
            action=action,
        )
        tqdm.write(
            f"[epoch {epoch}/{num_epochs}] loss={rec['loss']:.4f} "
            f"valid_hard={val_hard:.4f} valid_soft={val_soft:.4f} action={action}",
        )

    summary["best_score"] = best_score if history else None
    summary[best_summary_key] = best_path if os.path.exists(best_path) else ""
    summary[latest_summary_key] = latest_path
    if os.path.exists(best_path):
        load_checkpoint(torch, adapter_model, best_path)
    if cfg.get("eval_test", True):
        test_dir = os.path.join(out_root, "eval", "best", "valid_unseen")
        test_hard, test_soft, _ = _evaluate_prefix(
            env,
            adapter_model,
            test_items,
            cfg=cfg,
            settings=settings,
            out_dir=test_dir,
            desc="  Test",
        )
        summary["test_hard"] = test_hard
        summary["test_soft"] = test_soft

    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def train_soft_prefix(
    *,
    cfg: dict[str, Any],
    soft_prefix_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Train a soft prefix while preserving SkillOpt split/eval settings."""
    settings = SoftPrefixSettings.from_dict(soft_prefix_cfg)
    env = str(cfg.get("env", "")).strip()
    if env not in {"alfworld", "searchqa", "officeqa", "livemathematicianbench", "docvqa", "spreadsheetbench"}:
        raise ValueError(
            "soft-prefix training supports env=alfworld, env=searchqa, env=officeqa, "
            "env=livemathematicianbench, env=docvqa, or env=spreadsheetbench"
        )
    _set_seed(int(cfg.get("seed", 42)))
    init_text = _load_init_text(settings.init_text_path or str(cfg.get("skill_init", "")))
    if settings.prefix_length == 0:
        settings.prefix_length = _resolve_auto_prefix_length(
            settings.model_name, init_text, settings.trust_remote_code
        )
        print(f"  [soft-prefix] using auto-prefix length {settings.prefix_length}", flush=True)
    prefix_model = _build_prefix_model(env, settings, init_text)
    out_root = os.path.abspath(str(cfg["out_root"]))
    return _train_adapter(
        cfg=cfg,
        settings=settings,
        adapter_model=prefix_model,
        best_path=os.path.join(out_root, "best_prefix.pt"),
        latest_path=os.path.join(out_root, "latest_prefix.pt"),
        best_summary_key="best_prefix_path",
        latest_summary_key="latest_prefix_path",
        save_checkpoint=_save_torch_state,
        load_checkpoint=_load_torch_state,
    )


def train_lora(
    *,
    cfg: dict[str, Any],
    lora_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Train a LoRA adapter on the same supervised examples as soft-prefix tuning."""
    env = str(cfg.get("env", "")).strip()
    if env not in {"alfworld", "searchqa", "officeqa", "livemathematicianbench", "docvqa", "spreadsheetbench"}:
        raise ValueError(
            "LoRA baseline training supports env=alfworld, env=searchqa, env=officeqa, "
            "env=livemathematicianbench, env=docvqa, or env=spreadsheetbench"
        )
    settings = LoraSettings.from_dict(lora_cfg)
    _set_seed(int(cfg.get("seed", 42)))
    lora_model = _build_lora_model(env, settings)
    out_root = os.path.abspath(str(cfg["out_root"]))
    return _train_adapter(
        cfg=cfg,
        settings=settings,
        adapter_model=lora_model,
        best_path=os.path.join(out_root, "best_lora"),
        latest_path=os.path.join(out_root, "latest_lora"),
        best_summary_key="best_lora_path",
        latest_summary_key="latest_lora_path",
        save_checkpoint=_save_lora_adapter,
        load_checkpoint=_load_lora_adapter,
    )


def train_searchqa_soft_prefix(
    *,
    cfg: dict[str, Any],
    soft_prefix_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Backward-compatible SearchQA entrypoint."""
    if cfg.get("env") != "searchqa":
        raise ValueError("train_searchqa_soft_prefix requires env=searchqa; use train_soft_prefix for dispatch")
    return train_soft_prefix(cfg=cfg, soft_prefix_cfg=soft_prefix_cfg)
