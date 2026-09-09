"""Frozen causal LM wrapper with trainable soft-prefix embeddings."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from skillopt.softprefix.data import (
    apply_docvqa_image_budget,
    qwen_image_patch_size,
    resolve_docvqa_image_token_budget,
)


def _import_torch_and_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Soft-prefix training requires the optional dependencies "
            "`torch` and `transformers`. Install them with `pip install -e '.[softprefix]'`."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _import_torch_and_vlm_transformers():
    try:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as AutoVLM
            except ImportError:
                from transformers import AutoModelForCausalLM as AutoVLM
    except ImportError as exc:
        raise ImportError(
            "DocVQA soft-prefix training requires the optional dependencies "
            "`torch`, `transformers`, and `qwen-vl-utils`. Install them with "
            "`pip install -e '.[softprefix]'`."
        ) from exc
    return torch, AutoVLM, AutoProcessor


def _initialize_prefix_from_vocab_mean(torch, model: Any, prefix_embeddings) -> None:
    """Initialize every prefix row from the mean token embedding."""
    with torch.no_grad():
        vocab_mean = model.get_input_embeddings().weight.detach().mean(dim=0)
        vocab_mean = vocab_mean.to(device=prefix_embeddings.device, dtype=prefix_embeddings.dtype)
        prefix_embeddings.copy_(vocab_mean.unsqueeze(0).expand_as(prefix_embeddings))


def _flatten_prefix_embeddings(prefix_embeddings):
    """Return all soft skills as one contiguous sequence of virtual tokens."""
    if prefix_embeddings.dim() == 2:
        return prefix_embeddings
    if prefix_embeddings.dim() == 3:
        return prefix_embeddings.flatten(0, 1)
    raise ValueError(
        "prefix_embeddings must have shape [prefix_length, hidden_size] or "
        "[num_soft_skills, prefix_length, hidden_size]"
    )


def _masked_causal_lm_loss(torch, logits, labels):
    """Compute CE only for supervised target positions, avoiding full-logit fp32 upcast."""
    ignore_index = -100
    shift_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    if logits.shape[1] != shift_labels.shape[1]:
        shift_labels = shift_labels[:, -logits.shape[1] :]
    active = shift_labels != ignore_index
    if not bool(active.any()):
        return logits.sum() * 0.0
    active_logits = logits[active].float()
    active_labels = shift_labels[active].to(active_logits.device)
    return torch.nn.functional.cross_entropy(active_logits, active_labels, ignore_index=ignore_index)


def _logits_to_keep_from_labels(torch, labels) -> int:
    """Return the smallest suffix of logits needed to score supervised labels."""
    if labels is None or labels.ndim < 2:
        return 0
    active = labels != -100
    if not bool(active.any()):
        return 1
    active_positions = active.nonzero(as_tuple=False)[:, 1]
    first_logit_pos = max(int(active_positions.min().item()) - 1, 0)
    return max(int(labels.shape[1]) - first_logit_pos, 1)


class SoftPrefixCausalLM:
    """Owns a frozen causal LM and a trainable prefix embedding parameter."""

    def __init__(
        self,
        model_name: str,
        *,
        prefix_length: int,
        num_soft_skills: int = 2,
        init_text: str = "",
        init_strategy: str = "text",
        torch_dtype: str = "auto",
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer = _import_torch_and_transformers()
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        n_gpus = torch.cuda.device_count()
        use_device_map = device == "auto" and n_gpus > 1

        dtype = None
        if torch_dtype == "auto":
            dtype = torch.bfloat16 if (use_device_map or torch.cuda.is_available()) else torch.float32
        elif torch_dtype:
            dtype = getattr(torch, torch_dtype)

        model_kwargs = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        if use_device_map:
            model_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
            if resolved_device == "auto":
                resolved_device = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(resolved_device)

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        # With device_map="auto" the model spans multiple GPUs; anchor our
        # trainable prefix on the same device as the token embedding layer.
        self.device = self.model.get_input_embeddings().weight.device
        self.prefix_length = int(prefix_length)
        self.num_soft_skills = int(num_soft_skills)
        if self.num_soft_skills < 1:
            raise ValueError("num_soft_skills must be >= 1")
        hidden_size = self.model.get_input_embeddings().embedding_dim
        self.prefix_embeddings = torch.nn.Parameter(
            torch.empty(
                self.num_soft_skills,
                self.prefix_length,
                hidden_size,
                device=self.device,
                dtype=self.model.dtype,
            )
        )
        torch.nn.init.normal_(self.prefix_embeddings, mean=0.0, std=0.02)
        init_strategy = init_strategy.strip().lower()
        if init_strategy in {"text", "skill_text"}:
            if init_text.strip():
                self.initialize_from_text(init_text)
        elif init_strategy in {"vocab_mean", "mean_vocab", "embedding_mean"}:
            _initialize_prefix_from_vocab_mean(self.torch, self.model, self.prefix_embeddings)
        elif init_strategy in {"random", "normal", "normal_random"}:
            pass
        else:
            raise ValueError("soft_prefix.init_strategy must be one of text, vocab_mean, or random")

    def trainable_parameters(self):
        return [self.prefix_embeddings]

    def active_prefix_embeddings(self):
        return _flatten_prefix_embeddings(self.prefix_embeddings)

    def state_dict(self) -> dict[str, Any]:
        return {
            "prefix_embeddings": self.prefix_embeddings.detach().cpu(),
            "prefix_length": self.prefix_length,
            "num_soft_skills": self.num_soft_skills,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        value = state["prefix_embeddings"].to(device=self.device, dtype=self.prefix_embeddings.dtype)
        if value.dim() == 2 and tuple(value.shape) == tuple(self.prefix_embeddings.shape[1:]):
            value = value.unsqueeze(0).expand_as(self.prefix_embeddings).clone()
        if tuple(value.shape) != tuple(self.prefix_embeddings.shape):
            raise ValueError(
                f"prefix shape mismatch: checkpoint {tuple(value.shape)} vs model {tuple(self.prefix_embeddings.shape)}"
            )
        with self.torch.no_grad():
            self.prefix_embeddings.copy_(value)

    def initialize_from_text(self, text: str) -> None:
        """Initialize all soft skills from consecutive token embeddings of one text seed."""
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.numel() == 0:
            return
        with self.torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)[0].to(self.prefix_embeddings.dtype)
            total_length = self.num_soft_skills * self.prefix_length
            repeats = (total_length + token_embeds.shape[0] - 1) // token_embeds.shape[0]
            tiled = token_embeds.repeat((repeats, 1))[:total_length]
            self.prefix_embeddings.copy_(
                tiled.reshape(self.num_soft_skills, self.prefix_length, -1)
            )

    def _with_prefix(self, input_ids, attention_mask, labels=None, prefix_insert_idx=None):
        batch_size = input_ids.shape[0]
        token_embeds = self.model.get_input_embeddings()(input_ids.to(self.device))
        flat_prefix = self.active_prefix_embeddings()
        effective_prefix_length = int(flat_prefix.shape[0])
        prefix = flat_prefix.unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = attention_mask.to(self.device)
        labels_on_device = labels.to(self.device) if labels is not None else None
        prefix_mask = self.torch.ones(
            effective_prefix_length,
            dtype=attention_mask.dtype,
            device=self.device,
        )

        if prefix_insert_idx is None:
            inputs_embeds = self.torch.cat([prefix, token_embeds], dim=1)
            full_attention_mask = self.torch.cat(
                [prefix_mask.unsqueeze(0).expand(batch_size, -1), attention_mask],
                dim=1,
            )
            full_labels = None
            if labels_on_device is not None:
                prefix_labels = self.torch.full(
                    (batch_size, effective_prefix_length),
                    -100,
                    dtype=labels_on_device.dtype,
                    device=self.device,
                )
                full_labels = self.torch.cat([prefix_labels, labels_on_device], dim=1)
            return inputs_embeds, full_attention_mask, full_labels

        insert_indices = self.torch.as_tensor(prefix_insert_idx, device=self.device).view(-1)
        if int(insert_indices.numel()) != batch_size:
            raise ValueError(
                f"prefix_insert_idx must have one entry per batch row, got {int(insert_indices.numel())} for {batch_size}"
            )
        embed_rows = []
        mask_rows = []
        label_rows = [] if labels_on_device is not None else None
        prefix_labels_row = None
        if labels_on_device is not None:
            prefix_labels_row = self.torch.full(
                (effective_prefix_length,),
                -100,
                dtype=labels_on_device.dtype,
                device=self.device,
            )
        seq_len = int(input_ids.shape[1])
        for row, raw_idx in enumerate(insert_indices.tolist()):
            idx = max(0, min(int(raw_idx), seq_len))
            embed_rows.append(
                self.torch.cat(
                    [token_embeds[row, :idx], prefix[row], token_embeds[row, idx:]],
                    dim=0,
                )
            )
            mask_rows.append(
                self.torch.cat(
                    [attention_mask[row, :idx], prefix_mask, attention_mask[row, idx:]],
                    dim=0,
                )
            )
            if label_rows is not None and prefix_labels_row is not None:
                label_rows.append(
                    self.torch.cat(
                        [labels_on_device[row, :idx], prefix_labels_row, labels_on_device[row, idx:]],
                        dim=0,
                    )
                )
        inputs_embeds = self.torch.stack(embed_rows, dim=0)
        full_attention_mask = self.torch.stack(mask_rows, dim=0)
        full_labels = self.torch.stack(label_rows, dim=0) if label_rows is not None else None
        return inputs_embeds, full_attention_mask, full_labels

    def forward(self, batch: dict):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        inputs_embeds, full_attention_mask, full_labels = self._with_prefix(
            input_ids,
            attention_mask,
            labels,
            prefix_insert_idx=batch.get("prefix_insert_idx"),
        )
        logits_to_keep = _logits_to_keep_from_labels(self.torch, full_labels)
        try:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                logits_to_keep=logits_to_keep,
            )
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
            )
        return SimpleNamespace(loss=_masked_causal_lm_loss(self.torch, outputs.logits, full_labels))

    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        use_prefix: bool = True,
        prefix_insert_idx: int | None = None,
        stop_strings: list[str] | tuple[str, ...] | None = None,
        use_cache: bool | None = None,
    ) -> str:
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if use_cache is not None:
            generate_kwargs["use_cache"] = bool(use_cache)
        if stop_strings:
            generate_kwargs["stop_strings"] = list(stop_strings)
            generate_kwargs["tokenizer"] = self.tokenizer
        if use_prefix:
            inputs_embeds, full_attention_mask, _ = self._with_prefix(
                input_ids,
                attention_mask,
                prefix_insert_idx=(
                    self.torch.tensor([prefix_insert_idx], device=self.device)
                    if prefix_insert_idx is not None
                    else None
                ),
            )
            generate_kwargs["inputs_embeds"] = inputs_embeds
            generate_kwargs["attention_mask"] = full_attention_mask
        else:
            generate_kwargs["input_ids"] = input_ids
            generate_kwargs["attention_mask"] = attention_mask
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        if not use_prefix:
            output_ids = output_ids[:, input_ids.shape[1]:]
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def generate_from_prompts(
        self,
        prompts: list[str],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        use_prefix: bool = True,
        prefix_insert_indices: list[int | None] | None = None,
    ) -> list[str]:
        if not prompts:
            return []
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                prompts,
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_tokens,
                padding=True,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = old_padding_side
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if use_prefix:
            insert_tensor = None
            if prefix_insert_indices is not None and any(idx is not None for idx in prefix_insert_indices):
                insert_tensor = self.torch.tensor(
                    [int(idx or 0) for idx in prefix_insert_indices],
                    device=self.device,
                )
            inputs_embeds, full_attention_mask, _ = self._with_prefix(
                input_ids,
                attention_mask,
                prefix_insert_idx=insert_tensor,
            )
            generate_kwargs["inputs_embeds"] = inputs_embeds
            generate_kwargs["attention_mask"] = full_attention_mask
        else:
            generate_kwargs["input_ids"] = input_ids
            generate_kwargs["attention_mask"] = attention_mask
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        if not use_prefix:
            output_ids = output_ids[:, input_ids.shape[1]:]
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


class SoftPrefixVisionLM:
    """Frozen Qwen-style vision-language LM with a trainable text-side prefix."""

    def __init__(
        self,
        model_name: str,
        *,
        prefix_length: int,
        num_soft_skills: int = 2,
        init_text: str = "",
        init_strategy: str = "text",
        torch_dtype: str = "auto",
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        torch, AutoVLM, AutoProcessor = _import_torch_and_vlm_transformers()
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError(f"Processor for {model_name} does not expose a tokenizer")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        n_gpus = torch.cuda.device_count()
        use_device_map = device == "auto" and n_gpus > 1

        dtype = None
        if torch_dtype == "auto":
            dtype = torch.bfloat16 if (use_device_map or torch.cuda.is_available()) else torch.float32
        elif torch_dtype:
            dtype = getattr(torch, torch_dtype)

        model_kwargs = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        if use_device_map:
            model_kwargs["device_map"] = "auto"
            self.model = AutoVLM.from_pretrained(model_name, **model_kwargs)
        else:
            resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
            if resolved_device == "auto":
                resolved_device = "cpu"
            self.model = AutoVLM.from_pretrained(model_name, **model_kwargs).to(resolved_device)

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.device = self.model.get_input_embeddings().weight.device
        self.prefix_length = int(prefix_length)
        self.num_soft_skills = int(num_soft_skills)
        if self.num_soft_skills < 1:
            raise ValueError("num_soft_skills must be >= 1")
        hidden_size = self.model.get_input_embeddings().embedding_dim
        self.prefix_embeddings = torch.nn.Parameter(
            torch.empty(
                self.num_soft_skills,
                self.prefix_length,
                hidden_size,
                device=self.device,
                dtype=self.model.dtype,
            )
        )
        torch.nn.init.normal_(self.prefix_embeddings, mean=0.0, std=0.02)
        init_strategy = init_strategy.strip().lower()
        if init_strategy in {"text", "skill_text"}:
            if init_text.strip():
                self.initialize_from_text(init_text)
        elif init_strategy in {"vocab_mean", "mean_vocab", "embedding_mean"}:
            _initialize_prefix_from_vocab_mean(self.torch, self.model, self.prefix_embeddings)
        elif init_strategy in {"random", "normal", "normal_random"}:
            pass
        else:
            raise ValueError("soft_prefix.init_strategy must be one of text, vocab_mean, or random")

    def trainable_parameters(self):
        return [self.prefix_embeddings]

    def active_prefix_embeddings(self):
        return _flatten_prefix_embeddings(self.prefix_embeddings)

    def state_dict(self) -> dict[str, Any]:
        return {
            "prefix_embeddings": self.prefix_embeddings.detach().cpu(),
            "prefix_length": self.prefix_length,
            "num_soft_skills": self.num_soft_skills,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        value = state["prefix_embeddings"].to(device=self.device, dtype=self.prefix_embeddings.dtype)
        if value.dim() == 2 and tuple(value.shape) == tuple(self.prefix_embeddings.shape[1:]):
            value = value.unsqueeze(0).expand_as(self.prefix_embeddings).clone()
        if tuple(value.shape) != tuple(self.prefix_embeddings.shape):
            raise ValueError(
                f"prefix shape mismatch: checkpoint {tuple(value.shape)} vs model {tuple(self.prefix_embeddings.shape)}"
            )
        with self.torch.no_grad():
            self.prefix_embeddings.copy_(value)

    def initialize_from_text(self, text: str) -> None:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.numel() == 0:
            return
        with self.torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)[0].to(self.prefix_embeddings.dtype)
            total_length = self.num_soft_skills * self.prefix_length
            repeats = (total_length + token_embeds.shape[0] - 1) // token_embeds.shape[0]
            tiled = token_embeds.repeat((repeats, 1))[:total_length]
            self.prefix_embeddings.copy_(
                tiled.reshape(self.num_soft_skills, self.prefix_length, -1)
            )

    def _embed_with_vision(self, batch: dict):
        input_ids = batch["input_ids"].to(self.device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        if pixel_values is None:
            return inputs_embeds
        if not hasattr(self.model, "visual"):
            raise ValueError("Vision soft-prefix training requires a Qwen-style model with a `visual` module")

        pixel_values = pixel_values.to(device=self.device, dtype=self.model.dtype)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)
        try:
            image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        except TypeError:
            image_embeds = self.model.visual(pixel_values, image_grid_thw)
        image_embeds = image_embeds.to(inputs_embeds.dtype)

        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("Vision model config does not define image_token_id")
        image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

    def _uses_native_vision_forward(self, batch: dict) -> bool:
        return batch.get("mm_token_type_ids") is not None and (
            batch.get("pixel_values") is not None or batch.get("pixel_values_videos") is not None
        )

    def _native_vision_kwargs(self, batch: dict) -> dict:
        kwargs = {}
        for key in ("image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
            value = batch.get(key)
            if value is not None:
                kwargs[key] = value.to(self.device)
        for key in ("pixel_values", "pixel_values_videos"):
            value = batch.get(key)
            if value is not None:
                kwargs[key] = value.to(device=self.device, dtype=self.model.dtype)
        return kwargs

    def _qwen3_position_inputs(self, batch: dict, full_attention_mask):
        model = getattr(self.model, "model", None)
        get_rope_index = getattr(model, "get_rope_index", None)
        mm_token_type_ids = batch.get("mm_token_type_ids")
        if not callable(get_rope_index) or mm_token_type_ids is None:
            return None, None

        input_ids = batch["input_ids"].to(self.device)
        batch_size = input_ids.shape[0]
        effective_prefix_length = int(self.active_prefix_embeddings().shape[0])
        prefix_ids = self.torch.full(
            (batch_size, effective_prefix_length),
            int(self.tokenizer.pad_token_id or 0),
            dtype=input_ids.dtype,
            device=self.device,
        )
        full_input_ids = self.torch.cat([prefix_ids, input_ids], dim=1)
        prefix_token_types = self.torch.zeros(
            (batch_size, effective_prefix_length),
            dtype=mm_token_type_ids.dtype,
            device=self.device,
        )
        full_mm_token_type_ids = self.torch.cat(
            [prefix_token_types, mm_token_type_ids.to(self.device)],
            dim=1,
        )
        position_ids, _rope_deltas = get_rope_index(
            full_input_ids,
            image_grid_thw=(
                batch.get("image_grid_thw").to(self.device)
                if batch.get("image_grid_thw") is not None
                else None
            ),
            video_grid_thw=(
                batch.get("video_grid_thw").to(self.device)
                if batch.get("video_grid_thw") is not None
                else None
            ),
            attention_mask=full_attention_mask,
            mm_token_type_ids=full_mm_token_type_ids,
        )
        return position_ids, full_mm_token_type_ids

    def _with_prefix(self, batch: dict, labels=None):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size = input_ids.shape[0]
        use_native_vision = self._uses_native_vision_forward(batch)
        if use_native_vision:
            token_embeds = self.model.get_input_embeddings()(input_ids.to(self.device))
        else:
            token_embeds = self._embed_with_vision(batch)
        flat_prefix = self.active_prefix_embeddings()
        effective_prefix_length = int(flat_prefix.shape[0])
        prefix = flat_prefix.unsqueeze(0).expand(batch_size, -1, -1)
        prefix_mask_row = self.torch.ones(
            effective_prefix_length,
            dtype=attention_mask.dtype,
            device=self.device,
        )
        attention_mask = attention_mask.to(self.device)
        labels_on_device = labels.to(self.device) if labels is not None else None
        insert_indices = batch.get("prefix_insert_idx")
        if insert_indices is None or use_native_vision:
            inputs_embeds = self.torch.cat([prefix, token_embeds], dim=1)
            prefix_mask = prefix_mask_row.unsqueeze(0).expand(batch_size, -1)
            full_attention_mask = self.torch.cat([prefix_mask, attention_mask], dim=1)
            full_labels = None
            if labels_on_device is not None:
                prefix_labels = self.torch.full(
                    (batch_size, effective_prefix_length),
                    -100,
                    dtype=labels_on_device.dtype,
                    device=self.device,
                )
                full_labels = self.torch.cat([prefix_labels, labels_on_device], dim=1)
        else:
            insert_indices = self.torch.as_tensor(insert_indices, device=self.device).view(-1)
            if int(insert_indices.numel()) != batch_size:
                raise ValueError("prefix_insert_idx must have one entry per batch row")
            embed_rows = []
            mask_rows = []
            label_rows = [] if labels_on_device is not None else None
            prefix_label_row = (
                self.torch.full(
                    (effective_prefix_length,),
                    -100,
                    dtype=labels_on_device.dtype,
                    device=self.device,
                )
                if labels_on_device is not None
                else None
            )
            seq_len = int(input_ids.shape[1])
            for row, raw_idx in enumerate(insert_indices.tolist()):
                idx = max(0, min(int(raw_idx), seq_len))
                embed_rows.append(self.torch.cat([token_embeds[row, :idx], prefix[row], token_embeds[row, idx:]], dim=0))
                mask_rows.append(self.torch.cat([attention_mask[row, :idx], prefix_mask_row, attention_mask[row, idx:]], dim=0))
                if label_rows is not None and prefix_label_row is not None:
                    label_rows.append(self.torch.cat([labels_on_device[row, :idx], prefix_label_row, labels_on_device[row, idx:]], dim=0))
            inputs_embeds = self.torch.stack(embed_rows)
            full_attention_mask = self.torch.stack(mask_rows)
            full_labels = self.torch.stack(label_rows) if label_rows is not None else None
        model_kwargs = {}
        if use_native_vision:
            model_kwargs = self._native_vision_kwargs(batch)
            position_ids, full_mm_token_type_ids = self._qwen3_position_inputs(
                batch,
                full_attention_mask,
            )
            if full_mm_token_type_ids is not None:
                model_kwargs["mm_token_type_ids"] = full_mm_token_type_ids
            if position_ids is not None:
                model_kwargs["position_ids"] = position_ids
        return inputs_embeds, full_attention_mask, full_labels, model_kwargs

    def forward(self, batch: dict):
        inputs_embeds, full_attention_mask, full_labels, model_kwargs = self._with_prefix(
            batch,
            labels=batch["labels"],
        )
        logits_to_keep = _logits_to_keep_from_labels(self.torch, full_labels)
        try:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                logits_to_keep=logits_to_keep,
                **model_kwargs,
            )
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                **model_kwargs,
            )
        return SimpleNamespace(loss=_masked_causal_lm_loss(self.torch, outputs.logits, full_labels))

    def generate_from_messages(
        self,
        messages: list[dict],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        max_image_tokens: int = 0,
        use_prefix: bool = True,
        stop_strings: list[str] | tuple[str, ...] | None = None,
        use_cache: bool | None = None,
    ) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "DocVQA soft-prefix evaluation requires qwen-vl-utils. "
                "Install with `pip install -e '.[softprefix]'`."
            ) from exc

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        patch_size = qwen_image_patch_size(self.processor)
        budgeted_messages = apply_docvqa_image_budget(
            messages,
            max_image_tokens=resolve_docvqa_image_token_budget(
                max_prompt_tokens=max_prompt_tokens,
                configured_max_image_tokens=max_image_tokens,
            ),
            image_patch_size=patch_size,
        )
        image_inputs, video_inputs = process_vision_info(
            budgeted_messages,
            image_patch_size=patch_size,
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
        batch = {}
        for key, value in encoded.items():
            if value is None:
                continue
            if key in {"input_ids", "attention_mask"} and value.shape[1] > max_prompt_tokens:
                raise ValueError(
                    f"DocVQA prompt encoded to {value.shape[1]} tokens, exceeding max_prompt_tokens={max_prompt_tokens}. "
                    "Lower soft_prefix.docvqa_max_image_tokens to downscale images further."
                )
            batch[key] = value.to(self.device) if hasattr(value, "to") else value

        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if use_cache is not None:
            generate_kwargs["use_cache"] = bool(use_cache)
        if stop_strings:
            generate_kwargs["stop_strings"] = list(stop_strings)
            generate_kwargs["tokenizer"] = self.tokenizer
        if use_prefix:
            inputs_embeds, full_attention_mask, _, model_kwargs = self._with_prefix(batch)
            generate_kwargs["inputs_embeds"] = inputs_embeds
            generate_kwargs["attention_mask"] = full_attention_mask
            generate_kwargs.update(model_kwargs)
        else:
            generate_kwargs.update(batch)
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        if not use_prefix:
            output_ids = output_ids[:, batch["input_ids"].shape[1]:]
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        use_prefix: bool = True,
        prefix_insert_idx: int | None = None,
        stop_strings: list[str] | tuple[str, ...] | None = None,
        use_cache: bool | None = None,
    ) -> str:
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        )
        batch = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }
        if prefix_insert_idx is not None:
            batch["prefix_insert_idx"] = self.torch.tensor([prefix_insert_idx], device=self.device)
        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if use_cache is not None:
            generate_kwargs["use_cache"] = bool(use_cache)
        if stop_strings:
            generate_kwargs["stop_strings"] = list(stop_strings)
            generate_kwargs["tokenizer"] = self.tokenizer
        if use_prefix:
            inputs_embeds, full_attention_mask, _, model_kwargs = self._with_prefix(batch)
            generate_kwargs["inputs_embeds"] = inputs_embeds
            generate_kwargs["attention_mask"] = full_attention_mask
            generate_kwargs.update(model_kwargs)
        else:
            generate_kwargs.update(batch)
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        if not use_prefix:
            output_ids = output_ids[:, batch["input_ids"].shape[1]:]
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


class ResidualPromptMLP:
    """Factory for the bottleneck residual reparameterizer from Residual Prompt Tuning."""

    @staticmethod
    def build(torch, embedding_dim: int, bottleneck_size: int):
        if int(bottleneck_size) < 1:
            raise ValueError("residual MLP bottleneck_size must be >= 1")
        module = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, int(bottleneck_size)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(bottleneck_size), embedding_dim),
            torch.nn.LayerNorm(embedding_dim),
        )
        # Start from Phi(P) = P exactly. The down projection remains normally
        # initialized; the zero up projection lets the residual branch grow
        # smoothly without destroying the Markdown embedding initialization.
        torch.nn.init.zeros_(module[2].weight)
        torch.nn.init.zeros_(module[2].bias)
        torch.nn.init.ones_(module[3].weight)
        torch.nn.init.zeros_(module[3].bias)
        return module

    @staticmethod
    def apply(module, prompt):
        return prompt + module(prompt)


def _module_state_to_cpu(module) -> dict[str, Any]:
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
    }


class TaskSpecificMultiPrefixVisionLM(SoftPrefixVisionLM):
    """One independent multi-prefix parameter per task, with no cross-task sharing."""

    def __init__(
        self,
        model_name: str,
        *,
        prefix_length: int,
        num_soft_skills: int,
        task_init_texts: dict[str, str],
        residual_bottleneck_size: int = 400,
        use_residual_reparameterization: bool = True,
        torch_dtype: str = "auto",
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        if not task_init_texts:
            raise ValueError("task_init_texts must contain at least one task")
        first_text = next(iter(task_init_texts.values()))
        super().__init__(
            model_name,
            prefix_length=prefix_length,
            num_soft_skills=num_soft_skills,
            init_text=first_text,
            init_strategy="text",
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        initial_parameter = self.prefix_embeddings
        self.task_prefix_embeddings = self.torch.nn.ParameterDict()
        self.task_residual_mlps = self.torch.nn.ModuleDict()
        self.use_residual_reparameterization = bool(use_residual_reparameterization)
        embedding_dim = int(initial_parameter.shape[-1])
        for index, (task_name, text) in enumerate(task_init_texts.items()):
            if index == 0:
                parameter = initial_parameter
            else:
                parameter = self.torch.nn.Parameter(self.torch.empty_like(initial_parameter))
                self.torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
                self._initialize_multi_prefix_from_text(parameter, text)
            self.task_prefix_embeddings[str(task_name)] = parameter
            if self.use_residual_reparameterization:
                self.task_residual_mlps[str(task_name)] = ResidualPromptMLP.build(
                    self.torch,
                    embedding_dim,
                    residual_bottleneck_size,
                ).to(device=self.device, dtype=parameter.dtype)
        self.active_task = next(iter(self.task_prefix_embeddings))
        self.residual_bottleneck_size = int(residual_bottleneck_size)

    def _initialize_multi_prefix_from_text(self, parameter, text: str) -> None:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.numel() == 0:
            return
        with self.torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)[0].to(parameter.dtype)
            total_length = self.num_soft_skills * self.prefix_length
            repeats = (total_length + token_embeds.shape[0] - 1) // token_embeds.shape[0]
            parameter.copy_(
                token_embeds.repeat((repeats, 1))[:total_length].reshape_as(parameter)
            )

    def set_active_task(self, task_name: str) -> None:
        if task_name not in self.task_prefix_embeddings:
            raise KeyError(f"unknown task prefix: {task_name!r}")
        self.active_task = task_name

    def active_prefix_embeddings(self):
        prompt = _flatten_prefix_embeddings(self.task_prefix_embeddings[self.active_task])
        if not self.use_residual_reparameterization:
            return prompt
        return ResidualPromptMLP.apply(self.task_residual_mlps[self.active_task], prompt)

    def trainable_parameters(self):
        return (
            list(self.task_prefix_embeddings.values())
            + list(self.task_residual_mlps.parameters())
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "task_prefix_embeddings": {
                name: parameter.detach().cpu()
                for name, parameter in self.task_prefix_embeddings.items()
            },
            "task_residual_mlps": {
                name: _module_state_to_cpu(module)
                for name, module in self.task_residual_mlps.items()
            },
            "prefix_length": self.prefix_length,
            "num_soft_skills": self.num_soft_skills,
            "residual_bottleneck_size": self.residual_bottleneck_size,
            "use_residual_reparameterization": self.use_residual_reparameterization,
            "active_task": self.active_task,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        task_state = state["task_prefix_embeddings"]
        if set(task_state) != set(self.task_prefix_embeddings):
            raise ValueError("task prefix checkpoint keys do not match configured tasks")
        with self.torch.no_grad():
            for name, parameter in self.task_prefix_embeddings.items():
                value = task_state[name].to(device=self.device, dtype=parameter.dtype)
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"task prefix shape mismatch for {name!r}")
                parameter.copy_(value)
        if self.use_residual_reparameterization:
            mlp_state = state["task_residual_mlps"]
            if set(mlp_state) != set(self.task_residual_mlps):
                raise ValueError("task residual MLP checkpoint keys do not match configured tasks")
            for name, module in self.task_residual_mlps.items():
                module.load_state_dict(mlp_state[name])
        self.set_active_task(str(state.get("active_task", self.active_task)))


class SharedTaskSoftPrefixVisionLM(SoftPrefixVisionLM):
    """One shared prefix plus one Markdown-initialized prefix per task."""

    def __init__(
        self,
        model_name: str,
        *,
        prefix_length: int,
        shared_init_text: str,
        task_init_texts: dict[str, str],
        residual_bottleneck_size: int = 400,
        use_residual_reparameterization: bool = True,
        torch_dtype: str = "auto",
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__(
            model_name,
            prefix_length=prefix_length,
            num_soft_skills=1,
            init_text=shared_init_text,
            init_strategy="text",
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        if not task_init_texts:
            raise ValueError("task_init_texts must contain at least one task")
        self.task_prefix_embeddings = {}
        embedding_dim = int(self.prefix_embeddings.shape[-1])
        self.use_residual_reparameterization = bool(use_residual_reparameterization)
        if self.use_residual_reparameterization:
            self.shared_residual_mlp = ResidualPromptMLP.build(
                self.torch,
                embedding_dim,
                residual_bottleneck_size,
            ).to(device=self.device, dtype=self.prefix_embeddings.dtype)
        self.task_residual_mlps = self.torch.nn.ModuleDict()
        for task_name, text in task_init_texts.items():
            parameter = self.torch.nn.Parameter(self.torch.empty_like(self.prefix_embeddings[0]))
            self.torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
            self._initialize_parameter_from_text(parameter, text)
            self.task_prefix_embeddings[str(task_name)] = parameter
            if self.use_residual_reparameterization:
                self.task_residual_mlps[str(task_name)] = ResidualPromptMLP.build(
                    self.torch,
                    embedding_dim,
                    residual_bottleneck_size,
                ).to(device=self.device, dtype=parameter.dtype)
        self.active_task = next(iter(self.task_prefix_embeddings))
        self.residual_bottleneck_size = int(residual_bottleneck_size)

    def _initialize_parameter_from_text(self, parameter, text: str) -> None:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.numel() == 0:
            return
        with self.torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)[0].to(parameter.dtype)
            repeats = (self.prefix_length + token_embeds.shape[0] - 1) // token_embeds.shape[0]
            parameter.copy_(token_embeds.repeat((repeats, 1))[: self.prefix_length])

    def set_active_task(self, task_name: str) -> None:
        if task_name not in self.task_prefix_embeddings:
            raise KeyError(f"unknown task prefix: {task_name!r}")
        self.active_task = task_name

    def active_prefix_embeddings(self):
        shared = _flatten_prefix_embeddings(self.prefix_embeddings)
        task = self.task_prefix_embeddings[self.active_task]
        if not self.use_residual_reparameterization:
            return self.torch.cat([shared, task], dim=0)
        shared = ResidualPromptMLP.apply(self.shared_residual_mlp, shared)
        task = ResidualPromptMLP.apply(self.task_residual_mlps[self.active_task], task)
        return self.torch.cat([shared, task], dim=0)

    def shared_parameters(self):
        parameters = [self.prefix_embeddings]
        if self.use_residual_reparameterization:
            parameters += list(self.shared_residual_mlp.parameters())
        return parameters

    def task_parameters(self):
        return (
            list(self.task_prefix_embeddings.values())
            + list(self.task_residual_mlps.parameters())
        )

    def trainable_parameters(self):
        return self.shared_parameters() + self.task_parameters()

    def state_dict(self) -> dict[str, Any]:
        state = {
            "shared_prefix_embeddings": self.prefix_embeddings.detach().cpu(),
            "task_prefix_embeddings": {
                name: parameter.detach().cpu()
                for name, parameter in self.task_prefix_embeddings.items()
            },
            "task_residual_mlps": {
                name: _module_state_to_cpu(module)
                for name, module in self.task_residual_mlps.items()
            },
            "prefix_length": self.prefix_length,
            "residual_bottleneck_size": self.residual_bottleneck_size,
            "use_residual_reparameterization": self.use_residual_reparameterization,
            "active_task": self.active_task,
        }
        if self.use_residual_reparameterization:
            state["shared_residual_mlp"] = _module_state_to_cpu(self.shared_residual_mlp)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        shared = state["shared_prefix_embeddings"].to(
            device=self.device,
            dtype=self.prefix_embeddings.dtype,
        )
        if shared.dim() == 2:
            shared = shared.unsqueeze(0)
        if tuple(shared.shape) != tuple(self.prefix_embeddings.shape):
            raise ValueError(
                f"shared prefix shape mismatch: {tuple(shared.shape)} vs "
                f"{tuple(self.prefix_embeddings.shape)}"
            )
        task_state = state["task_prefix_embeddings"]
        if set(task_state) != set(self.task_prefix_embeddings):
            raise ValueError("task prefix checkpoint keys do not match configured tasks")
        with self.torch.no_grad():
            self.prefix_embeddings.copy_(shared)
            for name, parameter in self.task_prefix_embeddings.items():
                value = task_state[name].to(device=self.device, dtype=parameter.dtype)
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"task prefix shape mismatch for {name!r}")
                parameter.copy_(value)
        if self.use_residual_reparameterization:
            self.shared_residual_mlp.load_state_dict(state["shared_residual_mlp"])
            mlp_state = state["task_residual_mlps"]
            if set(mlp_state) != set(self.task_residual_mlps):
                raise ValueError("task residual MLP checkpoint keys do not match configured tasks")
            for name, module in self.task_residual_mlps.items():
                module.load_state_dict(mlp_state[name])
        self.set_active_task(str(state.get("active_task", self.active_task)))

    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        use_prefix: bool = True,
        prefix_insert_idx: int | None = None,
        stop_strings: list[str] | tuple[str, ...] | None = None,
        use_cache: bool | None = None,
    ) -> str:
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        )
        batch = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }
        if prefix_insert_idx is not None:
            batch["prefix_insert_idx"] = self.torch.tensor([prefix_insert_idx], device=self.device)
        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if use_cache is not None:
            generate_kwargs["use_cache"] = bool(use_cache)
        if stop_strings:
            generate_kwargs["stop_strings"] = list(stop_strings)
            generate_kwargs["tokenizer"] = self.tokenizer
        if use_prefix:
            inputs_embeds, full_attention_mask, _, model_kwargs = self._with_prefix(batch)
            generate_kwargs["inputs_embeds"] = inputs_embeds
            generate_kwargs["attention_mask"] = full_attention_mask
            generate_kwargs.update(model_kwargs)
        else:
            generate_kwargs.update(batch)
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        if not use_prefix:
            output_ids = output_ids[:, batch["input_ids"].shape[1]:]
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
