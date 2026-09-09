"""Custom vLLM prompt-embeds service for soft-prefix inference."""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
from typing import Any


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


_SOFTPREFIX_DEBUG = _env_bool("SOFTPREFIX_DEBUG")
_SOFTPREFIX_DEBUG_LOGPROBS = _env_bool("SOFTPREFIX_DEBUG_LOGPROBS")
_SOFTPREFIX_DEBUG_NO_PROMPT_TOKEN_IDS = _env_bool("SOFTPREFIX_DEBUG_NO_PROMPT_TOKEN_IDS")
_SOFTPREFIX_DEBUG_PREFIX_MODE = os.environ.get("SOFTPREFIX_DEBUG_PREFIX_MODE", "learned").strip().lower()
_SOFTPREFIX_DEBUG_STRONG_TOKEN_TEXT = os.environ.get(
    "SOFTPREFIX_DEBUG_STRONG_TOKEN_TEXT",
    " banana banana banana banana",
)
_SOFTPREFIX_DEBUG_PREFIX_SCALE = _env_float("SOFTPREFIX_DEBUG_PREFIX_SCALE", 10.0)
_SOFTPREFIX_DEBUG_RANDOM_SEED = _env_int("SOFTPREFIX_DEBUG_RANDOM_SEED", 0)


_QWEN_REASONING_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*", re.DOTALL)
_QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_QWEN_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
_QWEN_PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_SKILL_SECTION_PLACEHOLDER = "{skill_section}"
_SOFT_PREFIX_INSERT_MARKER = "<|skillopt_soft_prefix_insert|>"


def _parse_qwen3_reasoning_content(text: str) -> tuple[str, str | None]:
    match = _QWEN_REASONING_RE.match(text)
    if match is None:
        return text, None
    return text[match.end() :].lstrip(), match.group(1).strip()


def _normalize_prefix_injection_position(value: str | None) -> str:
    raw = str(value or "prompt_start").strip().lower()
    if raw in {"prompt_start", "start", "prefix", "before_prompt"}:
        return "prompt_start"
    if raw in {"skill_section", "skill-section", "skill_placeholder", "skill"}:
        return "skill_section"
    raise ValueError("prefix_injection_position must be one of prompt_start or skill_section")


def _content_part_image_ref(part: Any) -> str | None:
    """Return an image reference from common OpenAI/Qwen chat content parts."""
    if not isinstance(part, dict):
        return None
    part_type = str(part.get("type", "")).lower()
    if part_type in {"video", "input_video"}:
        raise ValueError("video inputs are not supported by this soft-prefix service")
    if part_type == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
            return str(url) if url is not None else None
        if image_url is not None:
            return str(image_url)
    if part_type in {"image", "input_image"}:
        for key in ("image", "image_url", "url"):
            value = part.get(key)
            if isinstance(value, dict):
                value = value.get("url")
            if value is not None:
                return str(value)
    return None


def _chat_template_content_part(part: Any) -> Any:
    """Normalize OpenAI image_url parts to the Qwen/vLLM chat-template shape."""
    image_ref = _content_part_image_ref(part)
    if image_ref is None:
        return part
    rendered = {"type": "image", "image": image_ref}
    if isinstance(part, dict):
        for key in ("min_pixels", "max_pixels", "resized_height", "resized_width"):
            if key in part:
                rendered[key] = part[key]
    return rendered


def _messages_for_multimodal_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered_messages: list[dict[str, Any]] = []
    for message in messages:
        rendered_message = dict(message)
        content = rendered_message.get("content")
        if isinstance(content, list):
            rendered_message["content"] = [_chat_template_content_part(part) for part in content]
        rendered_messages.append(rendered_message)
    return rendered_messages


def _load_image_from_reference(image_ref: str, *, allow_local_image_paths: bool = False) -> Any:
    """Load a PIL image from a data URL, HTTP(S) URL, raw base64 string, or opt-in local path."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Multimodal image inputs require `Pillow` to be installed.") from exc

    ref = image_ref.strip()
    if ref.startswith("data:"):
        if "," not in ref:
            raise ValueError("invalid image data URL")
        header, encoded = ref.split(",", 1)
        if ";base64" not in header:
            raise ValueError("only base64 image data URLs are supported")
        raw = base64.b64decode(encoded)
    elif ref.startswith(("http://", "https://")):
        with urllib.request.urlopen(ref, timeout=30.0) as response:
            raw = response.read()
    elif allow_local_image_paths and Path(ref).expanduser().is_file():
        raw = Path(ref).expanduser().read_bytes()
    else:
        try:
            raw = base64.b64decode(ref, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "image references must be data URLs, HTTP(S) URLs, raw base64 strings,"
                " or local paths when --allow-local-image-paths is set"
            ) from exc
    image = Image.open(io.BytesIO(raw))
    return image.convert("RGB")


def _extract_images_from_messages(
    messages: list[dict[str, Any]],
    *,
    allow_local_image_paths: bool = False,
    max_images: int = 1,
) -> list[Any]:
    images: list[Any] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            image_ref = _content_part_image_ref(part)
            if image_ref is None:
                continue
            images.append(_load_image_from_reference(image_ref, allow_local_image_paths=allow_local_image_paths))
            if max_images > 0 and len(images) > max_images:
                raise ValueError(f"received {len(images)} images, but max_images_per_prompt={max_images}")
    return images


def _contains_multimodal_content(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if _content_part_image_ref(part) is not None:
                    return True
    return False


def _tool_parameter_schemas(tools: list[dict[str, Any]] | None, function_name: str) -> dict[str, Any]:
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or function.get("name") != function_name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            return {}
        properties = parameters.get("properties")
        return properties if isinstance(properties, dict) else {}
    return {}


def _coerce_tool_argument(value: str, schema: dict[str, Any] | None) -> Any:
    value = value.strip("\n")
    raw_type = schema.get("type") if isinstance(schema, dict) else None
    param_type = str(raw_type or "string").lower()
    if param_type in {"integer", "int"}:
        try:
            return int(value)
        except ValueError:
            return value
    if param_type in {"number", "float"}:
        try:
            return float(value)
        except ValueError:
            return value
    if param_type in {"boolean", "bool"}:
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return value
    if param_type in {"object", "array"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _parse_qwen_tool_calls(text: str, tools: list[dict[str, Any]] | None) -> tuple[str, list[dict[str, Any]]]:
    matches = list(_QWEN_TOOL_CALL_RE.finditer(text))
    if not matches:
        return text, []
    allowed_names = {
        str((tool.get("function") or {}).get("name") or "").strip()
        for tool in (tools or [])
        if isinstance(tool, dict)
    }
    content = text[: matches[0].start()]
    tool_calls: list[dict[str, Any]] = []
    for match in matches:
        for function_match in _QWEN_FUNCTION_RE.finditer(match.group(1)):
            function_name = function_match.group(1).strip()
            if allowed_names and function_name not in allowed_names and function_name != "answer":
                continue
            parameter_schemas = _tool_parameter_schemas(tools, function_name)
            arguments: dict[str, Any] = {}
            for parameter_match in _QWEN_PARAMETER_RE.finditer(function_match.group(2)):
                parameter_name = parameter_match.group(1).strip()
                arguments[parameter_name] = _coerce_tool_argument(
                    parameter_match.group(2),
                    parameter_schemas.get(parameter_name),
                )
            tool_calls.append(
                {
                    "id": f"call_softprefix_{len(tool_calls) + 1}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
    if not tool_calls:
        return text, []
    return content, tool_calls


def _messages_for_chat_template(
    messages: list[dict[str, Any]],
    *,
    normalize_multimodal: bool = False,
) -> list[dict[str, Any]]:
    source_messages = (
        _messages_for_multimodal_chat_template(messages) if normalize_multimodal else messages
    )
    rendered_messages: list[dict[str, Any]] = []
    for message in source_messages:
        rendered_message = dict(message)
        tool_calls = rendered_message.get("tool_calls")
        if isinstance(tool_calls, list):
            rendered_message["tool_calls"] = [_tool_call_for_chat_template(tool_call) for tool_call in tool_calls]
        rendered_messages.append(rendered_message)
    return rendered_messages


_EMBED_TOKEN_WEIGHT_KEYS = (
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)


def _try_load_embed_tokens_weight(model_name: str, *, trust_remote_code: bool) -> Any | None:
    """Load only the token embedding matrix for multimodal checkpoints."""
    try:
        import json

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError:
        return None

    def _load_from_tensors(tensors: dict[str, Any]) -> Any | None:
        for key in _EMBED_TOKEN_WEIGHT_KEYS:
            tensor = tensors.get(key)
            if tensor is not None:
                return tensor
        return None

    try:
        index_path = hf_hub_download(
            model_name,
            "model.safetensors.index.json",
            trust_remote_code=trust_remote_code,
        )
        weight_map = json.loads(Path(index_path).read_text(encoding="utf-8")).get("weight_map", {})
        for key in _EMBED_TOKEN_WEIGHT_KEYS:
            shard_name = weight_map.get(key)
            if shard_name is None:
                continue
            shard_path = hf_hub_download(
                model_name,
                shard_name,
                trust_remote_code=trust_remote_code,
            )
            return load_file(shard_path)[key]
    except Exception:  # noqa: BLE001
        pass

    try:
        shard_path = hf_hub_download(
            model_name,
            "model.safetensors",
            trust_remote_code=trust_remote_code,
        )
        return _load_from_tensors(load_file(shard_path))
    except Exception:  # noqa: BLE001
        return None


def _load_embedding_layer(
    *,
    model_name: str,
    dtype: str,
    trust_remote_code: bool,
    embedding_device: str,
) -> Any:
    """Return an embedding layer for soft-prefix prompt construction."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    embed_weight = _try_load_embed_tokens_weight(model_name, trust_remote_code=trust_remote_code)
    if embed_weight is not None:
        embedding = torch.nn.Embedding(
            int(embed_weight.shape[0]),
            int(embed_weight.shape[1]),
            dtype=embed_weight.dtype,
        )
        embedding.weight.data.copy_(embed_weight)
        embedding.eval()
        return embedding.to(embedding_device)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_config = getattr(config, "text_config", None) or config
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "config": model_config,
    }
    if hasattr(torch, dtype):
        model_kwargs["dtype"] = getattr(torch, dtype)
    try:
        embedding_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except ValueError:
        vlm_kwargs = dict(model_kwargs)
        vlm_kwargs["config"] = config
        embedding_model = AutoModelForImageTextToText.from_pretrained(model_name, **vlm_kwargs)
    embedding_model.eval()
    embedding_layer = embedding_model.get_input_embeddings().to(embedding_device)
    del embedding_model
    return embedding_layer


def _tool_call_for_chat_template(tool_call: Any) -> Any:
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
                parsed_arguments = {}
            rendered_function["arguments"] = parsed_arguments if isinstance(parsed_arguments, dict) else {}
        rendered_tool_call["function"] = rendered_function
    return rendered_tool_call


class SoftPrefixVllmClient:
    """HTTP client for the custom soft-prefix vLLM service."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"soft-prefix vLLM service returned HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"soft-prefix vLLM service request failed: {e}") from e
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"soft-prefix vLLM service returned non-object JSON: {data!r}")
        return data

    def set_prefix(self, prefix_embeddings: Any, *, injection_position: str | None = None) -> None:
        if hasattr(prefix_embeddings, "detach"):
            tensor = prefix_embeddings.detach().float().cpu()
            if tensor.ndim == 3:
                tensor = tensor.flatten(0, 1)
            if tensor.ndim == 2 and tensor.shape[0] == 0:
                prefix_embeddings = []
            else:
                prefix_embeddings = tensor.tolist()
        payload = {"prefix_embeddings": prefix_embeddings}
        if injection_position is not None:
            payload["prefix_injection_position"] = injection_position
        self._post("/set_prefix", payload)

    def generate_from_prompts(
        self,
        prompts: list[str],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        prefix_insert_indices: list[int | None] | None = None,
    ) -> list[str]:
        if not prompts:
            return []
        if prefix_insert_indices is not None and len(prefix_insert_indices) != len(prompts):
            raise ValueError(
                f"prefix_insert_indices has {len(prefix_insert_indices)} entries for {len(prompts)} prompts"
            )
        payload: dict[str, Any] = {
            "prompts": prompts,
            "max_prompt_tokens": int(max_prompt_tokens),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }
        if prefix_insert_indices is not None:
            payload["prefix_insert_indices"] = prefix_insert_indices
        data = self._post(
            "/generate",
            payload,
        )
        texts = data.get("texts")
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise RuntimeError(f"soft-prefix vLLM service returned invalid texts: {data!r}")
        if len(texts) != len(prompts):
            raise RuntimeError(
                f"soft-prefix vLLM service returned {len(texts)} texts for {len(prompts)} prompts"
            )
        return texts

    def generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        max_image_tokens: int = 0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "messages": messages,
            "max_prompt_tokens": int(max_prompt_tokens),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "max_image_tokens": int(max_image_tokens),
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = chat_template_kwargs
        data = self._post("/generate_messages", payload)
        text = data.get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"soft-prefix vLLM service returned invalid text: {data!r}")
        return text

    def generate_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload: dict[str, Any] = {
            "messages": messages,
            "max_prompt_tokens": int(max_prompt_tokens),
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "use_prefix": True,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = chat_template_kwargs
        data = self._post("/v1/chat/completions", payload)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"soft-prefix vLLM service returned invalid chat choices: {data!r}")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError(f"soft-prefix vLLM service returned invalid chat choice: {data!r}")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"soft-prefix vLLM service returned invalid chat message: {data!r}")
        return message, {"finish_reason": choice.get("finish_reason", "")}


class SoftPrefixVllmEngine:
    """Owns vLLM generation plus the learned prefix tensor."""

    def __init__(
        self,
        *,
        model_name: str,
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        prefix_path: str = "",
        embedding_device: str = "cpu",
        enable_auto_tool_choice: bool = False,
        tool_call_parser: str = "",
        reasoning_parser: str = "",
        max_images_per_prompt: int = 0,
        allow_local_image_paths: bool = False,
        language_model_only: bool = False,
        enable_prefix_caching: bool = False,
        enable_chunked_prefill: bool = False,
        enforce_eager: bool = False,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
        tensor_parallel_size: int | None = None,
        prefix_injection_position: str = "prompt_start",
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "The soft-prefix vLLM service requires `torch`, `transformers`, and `vllm`."
            ) from exc

        self.torch = torch
        self.SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name
        self.enable_auto_tool_choice = enable_auto_tool_choice
        self.tool_call_parser = tool_call_parser.strip()
        self.reasoning_parser = reasoning_parser.strip()
        self.max_images_per_prompt = int(max_images_per_prompt)
        self.allow_local_image_paths = bool(allow_local_image_paths)
        self.language_model_only = bool(language_model_only)
        self.embedding_device = embedding_device
        self.prefix_injection_position = _normalize_prefix_injection_position(prefix_injection_position)
        self.embedding_layer = _load_embedding_layer(
            model_name=model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            embedding_device=embedding_device,
        )
        llm_kwargs: dict[str, Any] = {
            "model": model_name,
            "enable_prompt_embeds": True,
            "enable_prefix_caching": enable_prefix_caching,
            "enable_chunked_prefill": enable_chunked_prefill,
            "enforce_eager": enforce_eager,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = int(max_model_len)
        if tensor_parallel_size is not None:
            llm_kwargs["tensor_parallel_size"] = int(tensor_parallel_size)
        if language_model_only:
            # Disable vision inputs without vLLM's --language-model-only flag so we
            # keep the same weight-loading path while using direct prompt_embeds.
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        elif max_images_per_prompt > 0:
            llm_kwargs["limit_mm_per_prompt"] = {"image": int(max_images_per_prompt)}
        self.llm = LLM(**llm_kwargs)
        if language_model_only:
            self.supports_mm_inputs = False
        else:
            try:
                from vllm.multimodal.registry import MULTIMODAL_REGISTRY

                self.supports_mm_inputs = MULTIMODAL_REGISTRY.supports_multimodal_inputs(
                    self.llm.model_config
                )
            except Exception:  # noqa: BLE001
                self.supports_mm_inputs = False
        if _SOFTPREFIX_DEBUG:
            print(
                f"[softprefix-debug] supports_mm_inputs={self.supports_mm_inputs}",
                flush=True,
            )
        self.prefix_embeddings = None
        if prefix_path:
            self.set_prefix_from_checkpoint(prefix_path)

    def set_prefix_from_checkpoint(self, path: str) -> None:
        state = self.torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "prefix_embeddings" in state:
            self.set_prefix(state["prefix_embeddings"])
            return
        raise ValueError(f"checkpoint {path!r} does not contain prefix_embeddings")

    def set_prefix(self, prefix_embeddings: Any, *, injection_position: str | None = None) -> None:
        if not hasattr(self, "prefix_injection_position"):
            self.prefix_injection_position = "prompt_start"
        if injection_position is not None:
            self.prefix_injection_position = _normalize_prefix_injection_position(injection_position)
        hidden = int(self.embedding_layer.weight.shape[1])
        dtype = self.embedding_layer.weight.dtype
        if prefix_embeddings is None or (isinstance(prefix_embeddings, list) and not prefix_embeddings):
            tensor = self.torch.empty((0, hidden), dtype=dtype)
        else:
            tensor = self.torch.as_tensor(prefix_embeddings, dtype=dtype)
            if tensor.ndim == 3:
                tensor = tensor.flatten(0, 1)
            if tensor.ndim == 1 and tensor.numel() == 0:
                tensor = tensor.reshape(0, hidden)
            elif tensor.ndim != 2:
                raise ValueError(f"prefix_embeddings must be rank 2, got shape={tuple(tensor.shape)}")
            elif int(tensor.shape[1]) != hidden:
                raise ValueError(
                    f"prefix_embeddings hidden dim {int(tensor.shape[1])} != model hidden dim {hidden}"
                )
        self.prefix_embeddings = tensor.cpu()

    def _debug_tensor_stats(self, name: str, tensor: Any) -> str:
        if tensor is None:
            return f"{name}=None"
        detached = tensor.detach().float().cpu()
        shape = tuple(int(dim) for dim in tensor.shape)
        if detached.numel() == 0:
            return f"{name}: shape={shape} numel=0"
        flat = detached.reshape(-1)
        weights = self.torch.arange(1, flat.numel() + 1, dtype=flat.dtype)
        fingerprint = float((flat * weights).sum().item())
        std = float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0
        return (
            f"{name}: shape={shape} dtype={tensor.dtype} "
            f"mean={float(flat.mean().item()):.6g} std={std:.6g} "
            f"absmax={float(flat.abs().max().item()):.6g} l2={float(flat.norm().item()):.6g} "
            f"fingerprint={fingerprint:.6g}"
        )

    def _debug_row_norm_stats(self, name: str, tensor: Any) -> str:
        if tensor is None or tensor.numel() == 0:
            return f"{name}_row_norms=empty"
        row_norms = tensor.detach().float().norm(dim=-1)
        return (
            f"{name}_row_norms: mean={float(row_norms.mean().item()):.6g} "
            f"max={float(row_norms.max().item()):.6g}"
        )

    def _debug_prefix_for_prompt(self, prefix: Any, text_embeds: Any) -> tuple[Any, str]:
        mode = _SOFTPREFIX_DEBUG_PREFIX_MODE
        if mode in {"", "learned", "checkpoint"} or prefix.numel() == 0:
            return prefix, "learned"
        if mode in {"zero", "zeros"}:
            return self.torch.zeros_like(prefix), "zeros"
        if mode in {"random", "random_scaled"}:
            generator = self.torch.Generator(device=prefix.device)
            generator.manual_seed(_SOFTPREFIX_DEBUG_RANDOM_SEED)
            random_prefix = self.torch.randn(
                prefix.shape,
                generator=generator,
                dtype=prefix.dtype,
                device=prefix.device,
            )
            random_prefix = random_prefix / random_prefix.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            if text_embeds is not None and text_embeds.numel() > 0:
                target_norm = float(text_embeds.detach().float().norm(dim=-1).mean().item())
            else:
                target_norm = float(prefix.detach().float().norm(dim=-1).mean().item()) or 1.0
            scale = _SOFTPREFIX_DEBUG_PREFIX_SCALE if mode == "random_scaled" else 1.0
            return random_prefix.to(prefix.dtype) * target_norm * scale, mode
        if mode in {"strong_token", "token", "token_repeat"}:
            token_text = _SOFTPREFIX_DEBUG_STRONG_TOKEN_TEXT
            encoded = self.tokenizer(token_text, add_special_tokens=False, return_tensors="pt")
            token_ids = encoded["input_ids"].to(self.embedding_device)
            with self.torch.no_grad():
                token_embeds = self.embedding_layer(token_ids)[0].detach().cpu().to(dtype=prefix.dtype)
            if token_embeds.numel() == 0:
                return prefix, "token(empty-fallback-learned)"
            repeats = (int(prefix.shape[0]) + int(token_embeds.shape[0]) - 1) // int(token_embeds.shape[0])
            return token_embeds.repeat((repeats, 1))[: int(prefix.shape[0])], f"token({token_text!r})"
        print(
            f"[softprefix-debug] unknown SOFTPREFIX_DEBUG_PREFIX_MODE={mode!r}; using learned prefix",
            flush=True,
        )
        return prefix, "learned"

    def _debug_prompt_input(self, label: str, prompt_input: dict[str, Any]) -> None:
        if not _SOFTPREFIX_DEBUG:
            return
        prompt_embeds = prompt_input.get("prompt_embeds")
        prompt_token_ids = prompt_input.get("prompt_token_ids")
        prompt_is_token_ids = prompt_input.get("prompt_is_token_ids")
        mm_kwargs = prompt_input.get("mm_kwargs")
        token_id_len = len(prompt_token_ids) if isinstance(prompt_token_ids, list) else None
        embed_rows = int(prompt_embeds.shape[0]) if prompt_embeds is not None else None
        is_token_counts = ""
        if isinstance(prompt_is_token_ids, list):
            is_token_counts = f" prompt_is_token_ids=true:{sum(bool(x) for x in prompt_is_token_ids)} false:{sum(not bool(x) for x in prompt_is_token_ids)}"
        print(
            f"[softprefix-debug] {label} type={prompt_input.get('type', 'prompt_dict')} "
            f"keys={sorted(prompt_input.keys())} "
            f"prompt_token_ids_len={token_id_len} prompt_embeds_rows={embed_rows}"
            f"{is_token_counts}",
            flush=True,
        )
        if prompt_embeds is not None:
            print(f"[softprefix-debug] {self._debug_tensor_stats(label + '.prompt_embeds', prompt_embeds)}", flush=True)
        if isinstance(mm_kwargs, dict) and mm_kwargs.get("prompt_embeds"):
            pe_items = mm_kwargs["prompt_embeds"]
            if pe_items:
                pe_tensor = pe_items[0]["embedding"].data
                print(
                    f"[softprefix-debug] {self._debug_tensor_stats(label + '.mm_kwargs.prompt_embeds', pe_tensor)}",
                    flush=True,
                )
        if isinstance(prompt_token_ids, list):
            print(
                "[softprefix-debug] "
                f"{label}.prompt_token_ids first16={prompt_token_ids[:16]} last16={prompt_token_ids[-16:]}",
                flush=True,
            )

    def _sampling_params(self, *, temperature: float, max_tokens: int) -> Any:
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if _SOFTPREFIX_DEBUG_LOGPROBS:
            kwargs["logprobs"] = 20
        return self.SamplingParams(**kwargs)

    def _debug_first_token_logprobs(self, label: str, output: Any) -> None:
        if not (_SOFTPREFIX_DEBUG and _SOFTPREFIX_DEBUG_LOGPROBS):
            return
        if not output.outputs:
            print(f"[softprefix-debug] {label} no output candidates", flush=True)
            return
        logprobs = getattr(output.outputs[0], "logprobs", None)
        if not logprobs:
            print(f"[softprefix-debug] {label} first-token logprobs unavailable", flush=True)
            return
        first = logprobs[0]
        if not isinstance(first, dict):
            print(f"[softprefix-debug] {label} first-token logprobs unexpected={first!r}", flush=True)
            return
        rows = []
        for token_id, entry in first.items():
            logprob = getattr(entry, "logprob", entry)
            decoded = getattr(entry, "decoded_token", None)
            if decoded is None:
                try:
                    decoded = self.tokenizer.decode([int(token_id)])
                except Exception:  # noqa: BLE001
                    decoded = ""
            rows.append((float(logprob), int(token_id), decoded))
        rows.sort(reverse=True)
        formatted = ", ".join(
            f"{token_id}:{decoded!r}:{logprob:.4f}" for logprob, token_id, decoded in rows[:20]
        )
        print(f"[softprefix-debug] {label} first_token_top_logprobs={formatted}", flush=True)

    def _build_mm_modality_prompt_input(
        self,
        prefix: Any,
        flat_token_ids: list[int],
        *,
        placeholder_id: int,
        prefix_insert_idx: int | None = None,
    ) -> dict[str, Any]:
        """Build a vLLM multimodal engine input for soft-prefix on MM-capable models.

        Qwen3.5 registers as multimodal in vLLM. Its worker always takes the
        multimodal prefill path, which embeds every ``prompt_token_id`` and
        ignores the separate ``enable_prompt_embeds`` / ``prompt_is_token_ids``
        branch. The supported way to inject custom prefix rows is the
        ``prompt_embeds`` multimodal modality (spliced via ``is_mm_embed``).
        """
        from vllm.multimodal.hasher import MultiModalHasher
        from vllm.multimodal.inputs import (
            MultiModalFieldElem,
            MultiModalKwargsItem,
            MultiModalSharedField,
            PlaceholderRange,
        )

        prefix_len = int(prefix.shape[0])
        prefix_cpu = prefix.detach().cpu()
        insert_idx = 0 if prefix_insert_idx is None else max(0, min(int(prefix_insert_idx), len(flat_token_ids)))
        prompt_token_ids = (
            [int(token_id) for token_id in flat_token_ids[:insert_idx]]
            + [placeholder_id] * prefix_len
            + [int(token_id) for token_id in flat_token_ids[insert_idx:]]
        )
        pe_item = MultiModalKwargsItem(
            {
                "embedding": MultiModalFieldElem(
                    data=prefix_cpu,
                    field=MultiModalSharedField(batch_size=1),
                )
            }
        )
        return {
            "type": "multimodal",
            "prompt_token_ids": prompt_token_ids,
            "mm_kwargs": {"prompt_embeds": [pe_item]},
            "mm_hashes": {"prompt_embeds": [MultiModalHasher.hash_kwargs(prompt_embeds=prefix_cpu)]},
            "mm_placeholders": {
                "prompt_embeds": [PlaceholderRange(offset=insert_idx, length=prefix_len, is_embed=None)]
            },
        }

    def _prompt_input(
        self,
        prompt: str,
        max_prompt_tokens: int,
        *,
        prefix_insert_idx: int | None = None,
    ) -> dict[str, Any]:
        if self.prefix_embeddings is None:
            raise RuntimeError("soft prefix has not been set")
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        flat_token_ids = input_ids[0].tolist()
        with self.torch.no_grad():
            text_embeds = self.embedding_layer(input_ids.to(self.embedding_device))[0].detach().cpu()
        prefix = self.prefix_embeddings.to(dtype=text_embeds.dtype)
        if _SOFTPREFIX_DEBUG:
            prefix, prefix_mode = self._debug_prefix_for_prompt(prefix, text_embeds)
        else:
            prefix_mode = "learned"
        placeholder_id = int(getattr(self.tokenizer, "pad_token_id", None) or 0)
        supports_mm_inputs = (
            bool(getattr(self, "supports_mm_inputs", False))
            and not bool(getattr(self, "language_model_only", False))
        )
        if _SOFTPREFIX_DEBUG:
            print(
                "[softprefix-debug] _prompt_input"
                f" path={'mm_modality' if supports_mm_inputs else 'embeds_mixed'}"
                f" prefix_mode={prefix_mode} prefix_rows={len(prefix)}"
                f" prefix_insert_idx={prefix_insert_idx}"
                f" text_tokens={len(flat_token_ids)} total={len(prefix) + len(flat_token_ids)}\n"
                f"[softprefix-debug] prompt_repr={prompt!r}\n"
                f"[softprefix-debug] first_text_token_ids={flat_token_ids[:12]}\n"
                f"[softprefix-debug] {self._debug_tensor_stats('prefix', prefix)}\n"
                f"[softprefix-debug] {self._debug_tensor_stats('text_embeds', text_embeds)}\n"
                f"[softprefix-debug] {self._debug_row_norm_stats('prefix', prefix)}\n"
                f"[softprefix-debug] {self._debug_row_norm_stats('text_embeds', text_embeds)}",
                flush=True,
            )
        if supports_mm_inputs:
            prompt_input = self._build_mm_modality_prompt_input(
                prefix,
                flat_token_ids,
                placeholder_id=placeholder_id,
                prefix_insert_idx=prefix_insert_idx,
            )
            self._debug_prompt_input("_prompt_input.return", prompt_input)
            return prompt_input

        insert_idx = 0 if prefix_insert_idx is None else max(0, min(int(prefix_insert_idx), len(flat_token_ids)))
        prompt_embeds = self.torch.cat(
            [text_embeds[:insert_idx], prefix, text_embeds[insert_idx:]],
            dim=0,
        )
        prompt_token_ids = (
            [int(token_id) for token_id in flat_token_ids[:insert_idx]]
            + [placeholder_id] * len(prefix)
            + [int(token_id) for token_id in flat_token_ids[insert_idx:]]
        )
        prompt_input = {
            "prompt_embeds": prompt_embeds,
        }
        if not _SOFTPREFIX_DEBUG_NO_PROMPT_TOKEN_IDS:
            prompt_input["prompt_token_ids"] = prompt_token_ids
            prompt_input["prompt_is_token_ids"] = (
                [True] * insert_idx
                + [False] * len(prefix)
                + [True] * (len(flat_token_ids) - insert_idx)
            )
        self._debug_prompt_input("_prompt_input.return", prompt_input)
        return prompt_input

    def _full_prompt_embed_input(
        self,
        prompt: str,
        max_prompt_tokens: int,
        *,
        prefix_insert_idx: int | None = None,
    ) -> dict[str, Any]:
        """Embed the whole templated prompt, matching vLLM's prompt_embed example."""
        if self.prefix_embeddings is None:
            raise RuntimeError("soft prefix has not been set")
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        flat_token_ids = [int(token_id) for token_id in input_ids[0].tolist()]
        with self.torch.no_grad():
            text_embeds = self.embedding_layer(input_ids.to(self.embedding_device))[0].detach().cpu()
        prefix = self.prefix_embeddings.to(dtype=text_embeds.dtype)
        if _SOFTPREFIX_DEBUG:
            prefix, prefix_mode = self._debug_prefix_for_prompt(prefix, text_embeds)
            print(
                "[softprefix-debug] _full_prompt_embed_input"
                f" prefix_mode={prefix_mode} prefix_rows={len(prefix)}"
                f" text_tokens={len(flat_token_ids)} total={len(prefix) + len(flat_token_ids)}",
                flush=True,
            )
        insert_idx = 0 if prefix_insert_idx is None else max(0, min(int(prefix_insert_idx), len(flat_token_ids)))
        prompt_embeds = self.torch.cat(
            [text_embeds[:insert_idx], prefix, text_embeds[insert_idx:]],
            dim=0,
        )
        placeholder_id = int(getattr(self.tokenizer, "pad_token_id", None) or 0)
        prompt_token_ids = (
            flat_token_ids[:insert_idx]
            + [placeholder_id] * len(prefix)
            + flat_token_ids[insert_idx:]
        )
        prompt_input = {
            "prompt_embeds": prompt_embeds,
            "prompt_token_ids": prompt_token_ids,
            "prompt_is_token_ids": [False] * len(prompt_token_ids),
        }
        self._debug_prompt_input("_full_prompt_embed_input.return", prompt_input)
        return prompt_input

    def _prompt_input_with_native_prompt(
        self,
        prompt: str,
        *,
        images: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Use vLLM's native text/multimodal preprocessing and prepend prefix embeddings.

        This is the important path for VLMs: vLLM owns image preprocessing,
        placeholder replacement, position metadata, and model-specific multimodal
        handling, while prompt_embeds supplies only the learned soft prefix.
        """
        if self.prefix_embeddings is None:
            raise RuntimeError("soft prefix has not been set")
        prompt_embeds = self.prefix_embeddings.to(dtype=self.embedding_layer.weight.dtype).detach().cpu()
        if _SOFTPREFIX_DEBUG:
            # This path lets vLLM tokenize/process the native prompt itself; only
            # the prefix rows are visible in prompt_embeds at this boundary.
            empty_text = self.torch.empty((0, prompt_embeds.shape[1]), dtype=prompt_embeds.dtype)
            prompt_embeds, prefix_mode = self._debug_prefix_for_prompt(prompt_embeds, empty_text)
            print(
                "[softprefix-debug] _prompt_input_with_native_prompt"
                f" prefix_mode={prefix_mode} prefix_rows={len(prompt_embeds)}"
                f" has_images={bool(images)} prompt_repr={prompt!r}\n"
                f"[softprefix-debug] {self._debug_tensor_stats('native_prefix', prompt_embeds)}\n"
                f"[softprefix-debug] {self._debug_row_norm_stats('native_prefix', prompt_embeds)}",
                flush=True,
            )
        prompt_input: dict[str, Any] = {
            "prompt": prompt,
            "prompt_embeds": prompt_embeds,
        }
        if images:
            prompt_input["multi_modal_data"] = {
                "image": images[0] if len(images) == 1 else images,
            }
        self._debug_prompt_input("_prompt_input_with_native_prompt.return", prompt_input)
        return prompt_input

    def _plain_prompt_input(
        self,
        prompt: str,
        *,
        images: list[Any] | None = None,
    ) -> str | dict[str, Any]:
        if not images:
            return prompt
        return {
            "prompt": prompt,
            "multi_modal_data": {
                "image": images[0] if len(images) == 1 else images,
            },
        }

    def _token_count(self, text: str) -> int:
        encoded = self.tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "ndim") and int(input_ids.ndim) == 2:
            return int(input_ids.shape[1])
        return len(input_ids)

    def _strip_soft_prefix_marker(self, prompt: str) -> tuple[str, int]:
        marker_counts = {
            _SOFT_PREFIX_INSERT_MARKER: prompt.count(_SOFT_PREFIX_INSERT_MARKER),
            _SKILL_SECTION_PLACEHOLDER: prompt.count(_SKILL_SECTION_PLACEHOLDER),
        }
        marker_total = sum(marker_counts.values())
        if marker_total != 1:
            raise ValueError(
                "soft prefix injection_position=skill_section requires exactly one "
                f"{_SKILL_SECTION_PLACEHOLDER} or {_SOFT_PREFIX_INSERT_MARKER} marker"
            )
        marker = (
            _SOFT_PREFIX_INSERT_MARKER
            if marker_counts[_SOFT_PREFIX_INSERT_MARKER]
            else _SKILL_SECTION_PLACEHOLDER
        )
        char_idx = prompt.index(marker)
        clean_prompt = prompt[:char_idx] + prompt[char_idx + len(marker):]
        return clean_prompt, self._token_count(prompt[:char_idx])

    def _prompt_and_insert_idx(
        self,
        prompt: str,
        prefix_insert_idx: int | None,
    ) -> tuple[str, int | None]:
        if prefix_insert_idx is not None:
            return prompt, prefix_insert_idx
        if getattr(self, "prefix_injection_position", "prompt_start") != "skill_section":
            return prompt, None
        return self._strip_soft_prefix_marker(prompt)

    def _message_content_with_insert_marker(self, content: Any) -> tuple[Any, int]:
        if isinstance(content, str):
            count = content.count(_SOFT_PREFIX_INSERT_MARKER) + content.count(_SKILL_SECTION_PLACEHOLDER)
            if count == 0:
                return content, 0
            if count > 1:
                return content, count
            normalized = content.replace(_SKILL_SECTION_PLACEHOLDER, _SOFT_PREFIX_INSERT_MARKER, 1)
            return normalized, 1
        if isinstance(content, list):
            marker_count = 0
            rendered_parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    rendered_part = dict(part)
                    rendered_text, count = self._message_content_with_insert_marker(part["text"])
                    rendered_part["text"] = rendered_text
                    marker_count += count
                    rendered_parts.append(rendered_part)
                else:
                    rendered_parts.append(part)
            return rendered_parts, marker_count
        return content, 0

    def _messages_with_insert_marker(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        marker_count = 0
        rendered_messages = []
        for message in messages:
            rendered_message = dict(message)
            content, count = self._message_content_with_insert_marker(rendered_message.get("content"))
            rendered_message["content"] = content
            marker_count += count
            rendered_messages.append(rendered_message)
        if marker_count != 1:
            raise ValueError(
                "soft prefix injection_position=skill_section requires exactly one "
                f"{_SKILL_SECTION_PLACEHOLDER} or {_SOFT_PREFIX_INSERT_MARKER} marker"
            )
        return rendered_messages

    def _messages_to_prompt_and_insert_idx(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        normalize_multimodal: bool = False,
    ) -> tuple[str, int | None]:
        if getattr(self, "prefix_injection_position", "prompt_start") != "skill_section":
            return (
                self._messages_to_prompt(
                    messages,
                    tools=tools,
                    chat_template_kwargs=chat_template_kwargs,
                    normalize_multimodal=normalize_multimodal,
                ),
                None,
            )
        marked_messages = self._messages_with_insert_marker(messages)
        marked_prompt = self._messages_to_prompt(
            marked_messages,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            normalize_multimodal=normalize_multimodal,
        )
        return self._strip_soft_prefix_marker(marked_prompt)

    def generate_from_prompts(
        self,
        prompts: list[str],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        prefix_insert_indices: list[int | None] | None = None,
    ) -> list[str]:
        if prefix_insert_indices is not None and len(prefix_insert_indices) != len(prompts):
            raise ValueError(
                f"prefix_insert_indices has {len(prefix_insert_indices)} entries for {len(prompts)} prompts"
            )
        insert_indices = prefix_insert_indices or [None] * len(prompts)
        prompt_pairs = [
            self._prompt_and_insert_idx(prompt, insert_idx)
            for prompt, insert_idx in zip(prompts, insert_indices)
        ]
        prompt_builder = self._full_prompt_embed_input if self.language_model_only else self._prompt_input
        prompt_inputs = [
            prompt_builder(prompt, max_prompt_tokens, prefix_insert_idx=insert_idx)
            for prompt, insert_idx in prompt_pairs
        ]
        sampling = self._sampling_params(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(prompt_inputs, sampling)
        texts = [output.outputs[0].text if output.outputs else "" for output in outputs]
        if _SOFTPREFIX_DEBUG and texts:
            print(f"[softprefix-debug] WITH prefix    output[0]={texts[0]!r}", flush=True)
            self._debug_first_token_logprobs("WITH prefix", outputs[0])
            # Decisive A/B: same token ids, NO prefix embeds. If this matches the
            # WITH-prefix output above, vLLM is effectively ignoring the injected
            # soft prefix on this path.
            try:
                encoded = self.tokenizer(
                    prompts[0],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_prompt_tokens,
                )
                plain_ids = [int(token_id) for token_id in encoded["input_ids"]]
                plain_outputs = self.llm.generate([{"prompt_token_ids": plain_ids}], sampling)
                placeholder_id = int(getattr(self.tokenizer, "pad_token_id", None) or 0)
                prefix_rows = int(self.prefix_embeddings.shape[0]) if self.prefix_embeddings is not None else 0
                length_matched_ids = [placeholder_id] * prefix_rows + plain_ids
                length_matched_outputs = self.llm.generate(
                    [{"prompt_token_ids": length_matched_ids}],
                    sampling,
                )
                plain_text = (
                    plain_outputs[0].outputs[0].text
                    if plain_outputs and plain_outputs[0].outputs
                    else ""
                )
                length_matched_text = (
                    length_matched_outputs[0].outputs[0].text
                    if length_matched_outputs and length_matched_outputs[0].outputs
                    else ""
                )
                print(f"[softprefix-debug] WITHOUT prefix output[0]={plain_text!r}", flush=True)
                if plain_outputs:
                    self._debug_first_token_logprobs("WITHOUT prefix", plain_outputs[0])
                print(
                    f"[softprefix-debug] TOKEN prefix output[0]={length_matched_text!r}",
                    flush=True,
                )
                if length_matched_outputs:
                    self._debug_first_token_logprobs("TOKEN prefix", length_matched_outputs[0])
                print(
                    f"[softprefix-debug] prefix_changed_output={plain_text != texts[0]}",
                    flush=True,
                )
                print(
                    f"[softprefix-debug] token_prefix_changed_output={length_matched_text != texts[0]}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[softprefix-debug] no-prefix A/B failed: {type(exc).__name__}: {exc}", flush=True)
        return texts

    def generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        max_image_tokens: int = 0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        del max_image_tokens, tool_choice
        has_multimodal = _contains_multimodal_content(messages)
        if not has_multimodal:
            prompt, prefix_insert_idx = self._messages_to_prompt_and_insert_idx(
                messages,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
            )
            return self.generate_from_prompts(
                [prompt],
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prefix_insert_indices=[prefix_insert_idx],
            )[0]

        # For image prompts, do not pre-embed the text ourselves. Let vLLM's
        # multimodal processor handle image placeholders and model-specific
        # metadata, and prepend only the learned prefix via prompt_embeds.
        if getattr(self, "prefix_injection_position", "prompt_start") == "skill_section":
            raise ValueError(
                "soft prefix injection_position=skill_section is not supported for multimodal chat prompts"
            )
        prompt = self._messages_to_prompt(
            messages,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            normalize_multimodal=True,
        )
        images = _extract_images_from_messages(
            messages,
            allow_local_image_paths=self.allow_local_image_paths,
            max_images=self.max_images_per_prompt,
        )
        sampling = self._sampling_params(temperature=temperature, max_tokens=max_new_tokens)
        prompt_input = self._prompt_input_with_native_prompt(prompt, images=images)
        outputs = self.llm.generate([prompt_input], sampling)
        if _SOFTPREFIX_DEBUG and outputs:
            self._debug_first_token_logprobs("WITH native-prefix", outputs[0])
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text

    def _messages_to_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        normalize_multimodal: bool = False,
    ) -> str:
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            # Match the training prompt format exactly. Soft-prefix training
            # (skillopt.softprefix.data._apply_text_chat_template) renders the
            # chat template with enable_thinking=False, which pre-closes the
            # <think> block. Without this, Qwen3 defaults to enable_thinking=True
            # and the generation prompt ends with an *open* <think>, a different
            # format from what the prefix was trained on. Callers may override via
            # chat_template_kwargs.
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            if tools:
                template_kwargs["tools"] = tools
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)
            rendered_messages = _messages_for_chat_template(
                messages, normalize_multimodal=normalize_multimodal
            )
            try:
                return str(apply_chat_template(rendered_messages, **template_kwargs))
            except TypeError:
                # Tokenizer does not support enable_thinking (non-Qwen3 model).
                template_kwargs.pop("enable_thinking", None)
                return str(apply_chat_template(rendered_messages, **template_kwargs))
        chunks = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    "<image>" if _content_part_image_ref(part) is not None else str(part.get("text", part))
                    for part in content
                )
            chunks.append(f"{role}: {content}")
        chunks.append("assistant:")
        return "\n".join(chunks)

    def generate_plain_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        del tool_choice
        sampling = self._sampling_params(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        has_multimodal = _contains_multimodal_content(messages)
        prompt = self._messages_to_prompt(
            messages,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            normalize_multimodal=has_multimodal,
        )
        images = (
            _extract_images_from_messages(
                messages,
                allow_local_image_paths=self.allow_local_image_paths,
                max_images=self.max_images_per_prompt,
            )
            if has_multimodal
            else []
        )
        outputs = self.llm.generate([self._plain_prompt_input(prompt, images=images)], sampling)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text


class _Handler(BaseHTTPRequestHandler):
    engine: SoftPrefixVllmEngine

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8")
        data = json.loads(body or "{}")
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(200, {"ok": True})
            return
        if self.path in {"/v1/models", "/models"}:
            model_name = str(getattr(self.engine, "model_name", "soft-prefix-vllm"))
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model_name,
                            "object": "model",
                            "owned_by": "skillopt",
                        }
                    ],
                },
            )
            return
        self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            data = self._read_json()
            if self.path == "/set_prefix":
                self.engine.set_prefix(
                    data.get("prefix_embeddings"),
                    injection_position=data.get("prefix_injection_position", data.get("injection_position")),
                )
                self._write_json(200, {"ok": True})
                return
            if self.path == "/generate":
                prompts = [str(prompt) for prompt in data.get("prompts", [])]
                raw_insert_indices = data.get("prefix_insert_indices")
                prefix_insert_indices = None
                if raw_insert_indices is not None:
                    if not isinstance(raw_insert_indices, list):
                        raise ValueError("prefix_insert_indices must be a list")
                    if len(raw_insert_indices) != len(prompts):
                        raise ValueError("prefix_insert_indices must match prompts length")
                    prefix_insert_indices = [
                        None if idx is None else int(idx) for idx in raw_insert_indices
                    ]
                texts = self.engine.generate_from_prompts(
                    prompts,
                    max_prompt_tokens=int(data.get("max_prompt_tokens", 2048)),
                    max_new_tokens=int(data.get("max_new_tokens", 64)),
                    temperature=float(data.get("temperature", 0.0)),
                    prefix_insert_indices=prefix_insert_indices,
                )
                self._write_json(200, {"texts": texts})
                return
            if self.path == "/generate_messages":
                messages = data.get("messages", [])
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
                tools = data.get("tools")
                if tools is not None and not isinstance(tools, list):
                    raise ValueError("tools must be a list")
                chat_template_kwargs = data.get("chat_template_kwargs")
                if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
                    raise ValueError("chat_template_kwargs must be an object")
                text = self.engine.generate_from_messages(
                    [message for message in messages if isinstance(message, dict)],
                    max_prompt_tokens=int(data.get("max_prompt_tokens", 2048)),
                    max_new_tokens=int(data.get("max_new_tokens", 64)),
                    temperature=float(data.get("temperature", 0.0)),
                    max_image_tokens=int(data.get("max_image_tokens", 0)),
                    tools=tools,
                    tool_choice=data.get("tool_choice"),
                    chat_template_kwargs=chat_template_kwargs,
                )
                self._write_json(200, {"text": text})
                return
            if self.path in {"/v1/chat/completions", "/chat/completions"}:
                messages = data.get("messages", [])
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
                tools = data.get("tools")
                if tools is not None and not isinstance(tools, list):
                    raise ValueError("tools must be a list")
                chat_template_kwargs = data.get("chat_template_kwargs")
                if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
                    raise ValueError("chat_template_kwargs must be an object")
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": int(data.get("max_tokens", data.get("max_completion_tokens", 64)) or 64),
                    "temperature": float(data.get("temperature", 0.0) or 0.0),
                }
                if tools is not None:
                    generation_kwargs["tools"] = tools
                if "tool_choice" in data:
                    generation_kwargs["tool_choice"] = data.get("tool_choice")
                if chat_template_kwargs is not None:
                    generation_kwargs["chat_template_kwargs"] = chat_template_kwargs
                engine_messages = [message for message in messages if isinstance(message, dict)]
                if bool(data.get("use_prefix", False)):
                    text = self.engine.generate_from_messages(
                        engine_messages,
                        max_prompt_tokens=int(data.get("max_prompt_tokens", 2048)),
                        **generation_kwargs,
                    )
                else:
                    text = self.engine.generate_plain_from_messages(
                        engine_messages,
                        **generation_kwargs,
                    )
                model_name = str(data.get("model") or getattr(self.engine, "model_name", "soft-prefix-vllm"))
                message: dict[str, Any] = {"role": "assistant", "content": text}
                finish_reason = "stop"
                if str(getattr(self.engine, "reasoning_parser", "") or "").strip() == "qwen3":
                    content, reasoning_content = _parse_qwen3_reasoning_content(text)
                    if reasoning_content is not None:
                        message = {
                            "role": "assistant",
                            "content": content,
                            "reasoning_content": reasoning_content,
                        }
                        text = content
                parser_name = str(getattr(self.engine, "tool_call_parser", "") or "").strip()
                if (
                    tools
                    and bool(getattr(self.engine, "enable_auto_tool_choice", False))
                    and parser_name in {"qwen3_coder", "qwen3_xml"}
                ):
                    content, tool_calls = _parse_qwen_tool_calls(text, tools)
                    if tool_calls:
                        parsed_message: dict[str, Any] = {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": tool_calls,
                        }
                        if "reasoning_content" in message:
                            parsed_message["reasoning_content"] = message["reasoning_content"]
                        message = parsed_message
                        finish_reason = "tool_calls"
                self._write_json(
                    200,
                    {
                        "id": "chatcmpl-soft-prefix-vllm",
                        "object": "chat.completion",
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": finish_reason,
                            }
                        ],
                    },
                )
                return
            self._write_json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._write_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve soft-prefix inference via vLLM prompt_embeds")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--prefix_path", default="", help="Optional prefix checkpoint to load at startup")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--embedding_device", default="cpu")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enable-auto-tool-choice", action="store_true")
    parser.add_argument("--tool-call-parser", default="", choices=["", "qwen3_coder", "qwen3_xml"])
    parser.add_argument("--reasoning-parser", default="", choices=["", "qwen3"])
    parser.add_argument("--max-images-per-prompt", type=int, default=1)
    parser.add_argument(
        "--allow-local-image-paths",
        action="store_true",
        help="Allow request JSON to reference server-local image paths. Prefer data URLs for exposed services.",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        help=(
            "Text-only soft-prefix mode for vision-capable checkpoints: disable image/video "
            "inputs in vLLM and build full prompt_embeds like test_vllm.py."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of GPU memory vLLM may use (for example 0.85).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "vLLM context window. Vision-capable models like Qwen3.5-4B default to "
            "262144 and can OOM at startup unless capped (for example 8192)."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of GPUs to shard the vLLM model across.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable vLLM prefix caching (off by default; incompatible with per-request soft-prefix embeds).",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        help="Enable vLLM chunked prefill (off by default while debugging prompt_embeds).",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph execution in vLLM for prompt_embeds debugging.",
    )
    parser.add_argument(
        "--prefix-injection-position",
        default="prompt_start",
        choices=["prompt_start", "skill_section"],
        help=(
            "Default placement for a prefix loaded at startup. Runtime /set_prefix "
            "requests may override this."
        ),
    )
    args = parser.parse_args()

    _Handler.engine = SoftPrefixVllmEngine(
        model_name=args.model_name,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        prefix_path=args.prefix_path,
        embedding_device=args.embedding_device,
        enable_auto_tool_choice=args.enable_auto_tool_choice,
        tool_call_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        max_images_per_prompt=args.max_images_per_prompt,
        allow_local_image_paths=args.allow_local_image_paths,
        language_model_only=args.language_model_only,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        prefix_injection_position=args.prefix_injection_position,
    )
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"soft-prefix vLLM service listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
