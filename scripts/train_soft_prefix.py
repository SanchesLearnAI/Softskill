#!/usr/bin/env python3
"""Train a soft prefix on a frozen open-weight LM/VLM."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_OPENAI_DEFAULT_MODEL_SENTINELS = {"gpt-5.4", "gpt-5.5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoftSkill soft-prefix trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--cfg-options", nargs="+", default=[], help="Override config entries, e.g. soft_prefix.learning_rate=1e-3")
    parser.add_argument("--out_root", type=str, help="Output directory")
    parser.add_argument("--split_dir", type=str, help="Dataset split directory")
    parser.add_argument("--model_name", type=str, help="Hugging Face model name/path for the frozen target")
    return parser.parse_args()


def _configure_rollout_model(cfg: dict, cfg_options: list[str]) -> None:
    """Apply target backend settings before trajectory rollouts call skillopt.model."""
    from skillopt.model import (
        configure_azure_openai,
        configure_claude_code_exec,
        configure_codex_exec,
        configure_minimax_chat,
        configure_qwen_chat,
        set_optimizer_backend,
        set_optimizer_deployment,
        set_reasoning_effort,
        set_target_backend,
        set_target_deployment,
    )
    from skillopt.model.common import default_model_for_backend, normalize_backend_name

    def has_override(dotted_key: str) -> bool:
        for option in cfg_options or []:
            key = str(option).split("=", 1)[0].strip()
            if key == dotted_key:
                return True
        return False

    backend = normalize_backend_name(cfg.get("model_backend") or cfg.get("target_backend") or "openai_chat")
    optimizer_backend = cfg.get("optimizer_backend")
    target_backend = cfg.get("target_backend")
    if not optimizer_backend or not target_backend:
        if backend in {"claude", "claude_chat"}:
            optimizer_backend = optimizer_backend or "claude_chat"
            target_backend = target_backend or "claude_chat"
        elif backend in {"codex", "codex_exec"}:
            optimizer_backend = optimizer_backend or "openai_chat"
            target_backend = target_backend or "codex_exec"
        elif backend == "claude_code_exec":
            optimizer_backend = optimizer_backend or "openai_chat"
            target_backend = target_backend or "claude_code_exec"
        elif backend in {"qwen", "qwen_chat"}:
            optimizer_backend = optimizer_backend or "openai_chat"
            target_backend = target_backend or "qwen_chat"
        elif backend in {"minimax", "minimax_chat"}:
            optimizer_backend = optimizer_backend or "openai_chat"
            target_backend = target_backend or "minimax_chat"
        else:
            optimizer_backend = optimizer_backend or "openai_chat"
            target_backend = target_backend or "openai_chat"

    cfg["optimizer_backend"] = optimizer_backend
    cfg["target_backend"] = target_backend
    if (
        optimizer_backend == "qwen_chat"
        and str(cfg.get("optimizer_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
        and not has_override("model.optimizer")
    ):
        cfg["optimizer_model"] = default_model_for_backend("qwen_chat")
    if (
        target_backend == "qwen_chat"
        and str(cfg.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
        and not has_override("model.target")
    ):
        cfg["target_model"] = default_model_for_backend("qwen_chat")

    configure_azure_openai(
        endpoint=cfg.get("azure_openai_endpoint") or None,
        api_version=cfg.get("azure_openai_api_version") or None,
        api_key=cfg.get("azure_openai_api_key") or None,
        auth_mode=cfg.get("azure_openai_auth_mode") or None,
        ad_scope=cfg.get("azure_openai_ad_scope") or None,
        managed_identity_client_id=cfg.get("azure_openai_managed_identity_client_id") or None,
        optimizer_endpoint=cfg.get("optimizer_azure_openai_endpoint") or None,
        optimizer_api_version=cfg.get("optimizer_azure_openai_api_version") or None,
        optimizer_api_key=cfg.get("optimizer_azure_openai_api_key") or None,
        optimizer_auth_mode=cfg.get("optimizer_azure_openai_auth_mode") or None,
        optimizer_ad_scope=cfg.get("optimizer_azure_openai_ad_scope") or None,
        optimizer_managed_identity_client_id=cfg.get("optimizer_azure_openai_managed_identity_client_id") or None,
        target_endpoint=cfg.get("target_azure_openai_endpoint") or None,
        target_api_version=cfg.get("target_azure_openai_api_version") or None,
        target_api_key=cfg.get("target_azure_openai_api_key") or None,
        target_auth_mode=cfg.get("target_azure_openai_auth_mode") or None,
        target_ad_scope=cfg.get("target_azure_openai_ad_scope") or None,
        target_managed_identity_client_id=cfg.get("target_azure_openai_managed_identity_client_id") or None,
    )
    set_optimizer_backend(optimizer_backend)
    set_target_backend(target_backend)
    set_optimizer_deployment(str(cfg.get("optimizer_model") or default_model_for_backend(optimizer_backend)))
    set_target_deployment(str(cfg.get("target_model") or default_model_for_backend(target_backend)))
    configure_codex_exec(
        path=cfg.get("codex_exec_path", "codex"),
        sandbox=cfg.get("codex_exec_sandbox", "workspace-write"),
        profile=cfg.get("codex_exec_profile", ""),
        full_auto=cfg.get("codex_exec_full_auto", False),
        reasoning_effort=cfg.get("codex_exec_reasoning_effort", "none"),
        use_sdk=cfg.get("codex_exec_use_sdk", None),
        network_access=cfg.get("codex_exec_network_access", False),
        web_search=cfg.get("codex_exec_web_search", False),
        approval_policy=cfg.get("codex_exec_approval_policy", "never"),
    )
    configure_claude_code_exec(
        path=cfg.get("claude_code_exec_path", "claude"),
        profile=cfg.get("claude_code_exec_profile", ""),
        use_sdk=cfg.get("claude_code_exec_use_sdk", None),
        effort=cfg.get("claude_code_exec_effort", cfg.get("reasoning_effort", "medium")),
        max_thinking_tokens=cfg.get("claude_code_exec_max_thinking_tokens", 16384),
    )
    configure_qwen_chat(
        base_url=cfg.get("qwen_chat_base_url") or None,
        api_key=cfg.get("qwen_chat_api_key") or None,
        temperature=cfg.get("qwen_chat_temperature"),
        timeout_seconds=cfg.get("qwen_chat_timeout_seconds"),
        max_tokens=cfg.get("qwen_chat_max_tokens"),
        enable_thinking=cfg.get("qwen_chat_enable_thinking"),
        optimizer_base_url=cfg.get("optimizer_qwen_chat_base_url") or None,
        optimizer_api_key=cfg.get("optimizer_qwen_chat_api_key") or None,
        optimizer_temperature=cfg.get("optimizer_qwen_chat_temperature"),
        optimizer_timeout_seconds=cfg.get("optimizer_qwen_chat_timeout_seconds"),
        optimizer_max_tokens=cfg.get("optimizer_qwen_chat_max_tokens"),
        optimizer_enable_thinking=cfg.get("optimizer_qwen_chat_enable_thinking"),
        target_base_url=cfg.get("target_qwen_chat_base_url") or None,
        target_api_key=cfg.get("target_qwen_chat_api_key") or None,
        target_temperature=cfg.get("target_qwen_chat_temperature"),
        target_timeout_seconds=cfg.get("target_qwen_chat_timeout_seconds"),
        target_max_tokens=cfg.get("target_qwen_chat_max_tokens"),
        target_enable_thinking=cfg.get("target_qwen_chat_enable_thinking"),
    )
    configure_minimax_chat(
        base_url=cfg.get("minimax_base_url") or None,
        api_key=cfg.get("minimax_api_key") or None,
        temperature=cfg.get("minimax_temperature"),
        max_tokens=cfg.get("minimax_max_tokens"),
        enable_thinking=cfg.get("minimax_enable_thinking"),
    )
    set_reasoning_effort(cfg.get("reasoning_effort", "") or None)


def main() -> None:
    args = parse_args()
    from skillopt.config import flatten_config, is_structured, load_config
    from skillopt.softprefix import train_lora, train_soft_prefix
    from skillopt.softprefix.trainer import _normalize_trajectory_rollout_backend

    raw_cfg = load_config(args.config, overrides=args.cfg_options)
    flat_cfg = flatten_config(raw_cfg) if is_structured(raw_cfg) else dict(raw_cfg)
    soft_prefix_cfg = dict(raw_cfg.get("soft_prefix", {}) if isinstance(raw_cfg, dict) else {})
    lora_cfg = dict(raw_cfg.get("lora", {}) if isinstance(raw_cfg, dict) else {})
    training_method = "lora" if lora_cfg else "soft_prefix"
    adapter_cfg = lora_cfg if training_method == "lora" else soft_prefix_cfg

    if args.split_dir:
        flat_cfg["split_dir"] = args.split_dir
        flat_cfg["split_mode"] = "split_dir"
    if args.model_name:
        adapter_cfg["model_name"] = args.model_name

    if not flat_cfg.get("out_root"):
        env = flat_cfg.get("env", "unknown")
        model = str(adapter_cfg.get("model_name", "unknown")).replace("/", "-")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        flat_cfg["out_root"] = os.path.join("outputs", f"{training_method}_{env}_{model}_{ts}")
    if args.out_root:
        flat_cfg["out_root"] = args.out_root
    flat_cfg["out_root"] = os.path.abspath(flat_cfg["out_root"])

    os.makedirs(flat_cfg["out_root"], exist_ok=True)
    training_data = str(adapter_cfg.get("training_data", "")).strip().lower()
    trajectory_rollout_backend = _normalize_trajectory_rollout_backend(
        adapter_cfg.get("trajectory_rollout_backend", "openai")
    )
    if training_data not in {"trajectory", "trajectory_sft"} or trajectory_rollout_backend in {"openai", "vllm"}:
        _configure_rollout_model(flat_cfg, args.cfg_options)
    with open(os.path.join(flat_cfg["out_root"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "runtime": flat_cfg,
                "soft_prefix": soft_prefix_cfg,
                "lora": lora_cfg,
                "training_method": training_method,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 60)
    print(f"  SoftSkill {training_method.replace('_', ' ').title()}")
    print("=" * 60)
    print(f"  env:          {flat_cfg.get('env')}")
    print(f"  model:        {adapter_cfg.get('model_name')}")
    if training_method == "soft_prefix":
        num_soft_skills = int(adapter_cfg.get("num_soft_skills", 2))
        prefix_length = adapter_cfg.get("prefix_length")
        print(f"  soft_skills:  {num_soft_skills}")
        print(f"  prefix_len:   {prefix_length} each ({num_soft_skills} x {prefix_length} total)")
    else:
        print(f"  lora_r:       {adapter_cfg.get('r')}")
        print(f"  lora_alpha:   {adapter_cfg.get('alpha')}")
    print(f"  epochs:       {flat_cfg.get('num_epochs')}")
    print(f"  batch_size:   {flat_cfg.get('batch_size')}")
    if training_data in {"trajectory", "trajectory_sft"}:
        if trajectory_rollout_backend == "vllm":
            vllm_url = flat_cfg.get("target_qwen_chat_base_url") or flat_cfg.get("qwen_chat_base_url") or "http://localhost:8000/v1"
            print(f"  rollout:      {flat_cfg.get('target_model')} (vllm @ {vllm_url})")
            print(f"  rollout_new:  {adapter_cfg.get('trajectory_max_new_tokens') or flat_cfg.get('max_completion_tokens')}")
            if adapter_cfg.get("trajectory_use_skill"):
                skill_path = adapter_cfg.get("init_text_path") or flat_cfg.get("skill_init") or "(init_text_path)"
                print(f"  note:         trajectory rollout uses vLLM with skill from {skill_path}")
            else:
                print("  note:         trajectory rollout uses vLLM — plain inference with local tools, no skill/prefix")
        elif trajectory_rollout_backend == "openai":
            rollout_dir = adapter_cfg.get("trajectory_rollout_dir") or os.path.join(
                flat_cfg["out_root"],
                "trajectory_sft",
                "rollout_openai",
            )
            print(f"  rollout:      {flat_cfg.get('target_model')} ({flat_cfg.get('target_backend')})")
            print(f"  rollout_dir:  {os.path.abspath(str(rollout_dir))}")
            print(f"  rollout_new:  {adapter_cfg.get('trajectory_max_new_tokens') or flat_cfg.get('max_completion_tokens')}")
            print("  note:         trajectory rollout uses the configured OpenAI-compatible target; set trajectory_rollout_dir to share the cache")
        else:
            print(f"  rollout:      {adapter_cfg.get('model_name')} (local_hf_adapter)")
            print(f"  rollout_new:  {adapter_cfg.get('trajectory_max_new_tokens') or flat_cfg.get('max_completion_tokens')}")
            print("  note:         trajectory rollout and training use the same local HF model instance")
    else:
        print(f"  rollout:      {flat_cfg.get('target_model')} ({flat_cfg.get('target_backend')})")
    print("=" * 60)

    if training_method == "lora":
        summary = train_lora(cfg=flat_cfg, lora_cfg=lora_cfg)
    else:
        summary = train_soft_prefix(cfg=flat_cfg, soft_prefix_cfg=soft_prefix_cfg)
    print(f"\nOutput saved to: {flat_cfg['out_root']}")
    if summary.get("test_hard") is not None:
        print(f"Final test hard={summary['test_hard']:.4f} soft={summary.get('test_soft', 0.0):.4f}")


if __name__ == "__main__":
    main()
