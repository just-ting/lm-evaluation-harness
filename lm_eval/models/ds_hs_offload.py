from __future__ import annotations

import atexit
import logging
import os
from contextlib import nullcontext
from typing import Any, Literal

import torch
import transformers
from deepspeed.runtime.hs_offload import profiler as hs_profiler
from packaging.version import parse as vparse
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import get_dtype


eval_logger = logging.getLogger(__name__)


def _hs_profile_enabled() -> bool:
    return bool(hs_profiler.enabled())


def _hs_profile_section(name: str, sync_cuda: bool = False):
    if not hs_profiler.enabled():
        return nullcontext()
    return hs_profiler.section(name, sync_cuda=sync_cuda)


def _hs_profile_set(name: str, value: Any) -> None:
    hs_profiler.set_value(name, value)


def _hs_profile_add(name: str, value: float) -> None:
    hs_profiler.add_value(name, value)


def _hs_profile_count(name: str, amount: float = 1.0) -> None:
    hs_profiler.add_counter(name, amount)


def _hs_profile_dump(path: str | None = None) -> None:
    hs_profiler.dump(path)


@register_model("hs-offload-hf", "ds_hs_offload")
class HSOffloadHFLM(HFLM):
    """HFLM adapter that builds the underlying HF model via DeepSpeed HS offload."""

    def __init__(
        self,
        pretrained: str | transformers.PreTrainedModel,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        revision: str | None = "main",
        subfolder: str = "",
        tokenizer: str
        | transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast
        | None = None,
        truncation: bool | None = False,
        logits_cache: bool = True,
        max_length: int | None = None,
        device: str | None = "cuda",
        dtype: str | torch.dtype | None = "auto",
        softmax_dtype: str | torch.dtype | None = None,
        mixed_precision_dtype: str | torch.dtype | None = None,
        batch_size: int | str | None = 1,
        max_batch_size: int | None = 64,
        trust_remote_code: bool | None = False,
        use_fast_tokenizer: bool | None = True,
        add_bos_token: bool | None = None,
        prefix_token_id: int | None = None,
        parallelize: bool | None = False,
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | os.PathLike | None = "./offload",
        peft: str | None = None,
        delta: str | None = None,
        autogptq: bool | str | None = False,
        gptqmodel: bool | None = False,
        gguf_file: str | None = None,
        think_end_token: str | int | None = None,
        enable_thinking: bool | None = None,
        chat_template_args: dict[str, Any] | None = None,
        hs_offload: bool = True,
        hs_max_gen_len: int | None = None,
        hs_device: str = "cpu",
        hs_codec_name: str = "fp16",
        hs_outlier_k: int | None = None,
        short_context_exact_threshold: int = 16,
        ds_batch_size: int | None = None,
        cpu_offload: bool = False,
        disk_offload: bool = False,
        offload_dir: str | None = None,
        pin_memory: bool = False,
        quant_bits: int = 16,
        quant_group_size: int = 64,
        kv_offload: bool = False,
        pin_kv_cache: bool = False,
        async_kv_offload: bool = False,
        **kwargs,
    ) -> None:
        if hs_offload and kv_offload:
            raise ValueError("hs_offload and standard kv_offload cannot be enabled at the same time.")

        self._ds_engine = None
        self._hf_ds_config = None
        self._hs_runtime = None
        self._hs_config = None
        self._hs_profile_dump_registered = False
        self._kv_offload_enabled = bool(kv_offload)
        self._pin_kv_cache = bool(pin_kv_cache)
        self._async_kv_offload = bool(async_kv_offload)

        if isinstance(pretrained, str):
            _hs_profile_set("adapter.pretrained", pretrained)
            _hs_profile_set("adapter.hs_codec_name", hs_codec_name)
            _hs_profile_set("adapter.hs_outlier_k", hs_outlier_k)
            _hs_profile_set("adapter.hs_device", hs_device)
            _hs_profile_set("adapter.short_context_exact_threshold", int(short_context_exact_threshold))
            _hs_profile_set("adapter.quant_bits", int(quant_bits))
            _hs_profile_set("adapter.quant_group_size", int(quant_group_size))
            _hs_profile_set("adapter.kv_offload", bool(kv_offload))
            _hs_profile_set("adapter.pin_kv_cache", bool(pin_kv_cache))
            _hs_profile_set("adapter.async_kv_offload", bool(async_kv_offload))
            with _hs_profile_section("adapter.tokenizer_load"):
                tokenizer_obj = self._load_tokenizer(
                    pretrained=pretrained,
                    tokenizer=tokenizer,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast_tokenizer=use_fast_tokenizer,
                    add_bos_token=add_bos_token,
                    subfolder=subfolder,
                )
            self._validate_hs_load_args(
                backend=backend,
                parallelize=parallelize,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                gptqmodel=gptqmodel,
                gguf_file=gguf_file,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
            )
            (
                self._ds_engine,
                model,
                self._hf_ds_config,
                self._hs_config,
            ) = self._build_hs_offload_model(
                pretrained,
                backend=backend,
                revision=revision,
                subfolder=subfolder,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                batch_size=batch_size,
                ds_batch_size=ds_batch_size,
                cpu_offload=cpu_offload,
                disk_offload=disk_offload,
                offload_dir=offload_dir,
                pin_memory=pin_memory,
                quant_bits=quant_bits,
                quant_group_size=quant_group_size,
                hs_offload=hs_offload,
                hs_max_gen_len=hs_max_gen_len,
                hs_device=hs_device,
                hs_codec_name=hs_codec_name,
                hs_outlier_k=hs_outlier_k,
                short_context_exact_threshold=short_context_exact_threshold,
                **kwargs,
            )
            self._hs_runtime = getattr(self._ds_engine, "hs_runtime", None)
            if self._hs_runtime is not None:
                self._hs_runtime.reset_profiler()

            with _hs_profile_section("adapter.hflm_super_init"):
                super().__init__(
                    pretrained=model,
                    backend=backend,
                    revision=revision,
                    subfolder=subfolder,
                    tokenizer=tokenizer_obj,
                    truncation=truncation,
                    logits_cache=logits_cache,
                    max_length=max_length,
                    device=device,
                    dtype=dtype,
                    softmax_dtype=softmax_dtype,
                    mixed_precision_dtype=mixed_precision_dtype,
                    batch_size=batch_size,
                    max_batch_size=max_batch_size,
                    trust_remote_code=trust_remote_code,
                    use_fast_tokenizer=use_fast_tokenizer,
                    add_bos_token=add_bos_token,
                    prefix_token_id=prefix_token_id,
                    parallelize=False,
                    max_memory_per_gpu=max_memory_per_gpu,
                    max_cpu_memory=max_cpu_memory,
                    offload_folder=offload_folder,
                    peft=peft,
                    delta=delta,
                    autogptq=autogptq,
                    gptqmodel=gptqmodel,
                    gguf_file=gguf_file,
                    think_end_token=think_end_token,
                    enable_thinking=enable_thinking,
                    chat_template_args=chat_template_args,
                    **kwargs,
                )
            self.pretrained = pretrained
            self._device = self._resolve_runtime_device(device)
            self._sync_distributed_state()
            self._register_hs_profile_dump()
            return

        super().__init__(
            pretrained=pretrained,
            backend=backend,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
            truncation=truncation,
            logits_cache=logits_cache,
            max_length=max_length,
            device=device,
            dtype=dtype,
            softmax_dtype=softmax_dtype,
            mixed_precision_dtype=mixed_precision_dtype,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            add_bos_token=add_bos_token,
            prefix_token_id=prefix_token_id,
            parallelize=parallelize,
            max_memory_per_gpu=max_memory_per_gpu,
            max_cpu_memory=max_cpu_memory,
            offload_folder=offload_folder,
            peft=peft,
            delta=delta,
            autogptq=autogptq,
            gptqmodel=gptqmodel,
            gguf_file=gguf_file,
            think_end_token=think_end_token,
            enable_thinking=enable_thinking,
            chat_template_args=chat_template_args,
            **kwargs,
        )

    @staticmethod
    def _load_tokenizer(
        *,
        pretrained: str,
        tokenizer: str
        | transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast
        | None,
        revision: str | None = "main",
        trust_remote_code: bool | None = False,
        use_fast_tokenizer: bool | None = True,
        add_bos_token: bool | None = None,
        subfolder: str = "",
    ):
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "use_fast": use_fast_tokenizer,
        }
        if add_bos_token is not None:
            kwargs["add_bos_token"] = add_bos_token
        if subfolder:
            kwargs["subfolder"] = subfolder
        if tokenizer is None:
            return transformers.AutoTokenizer.from_pretrained(pretrained, **kwargs)
        if isinstance(tokenizer, str):
            return transformers.AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        return tokenizer

    @classmethod
    def _build_hs_offload_model(
        cls,
        pretrained: str,
        *,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        revision: str | None = "main",
        subfolder: str = "",
        device: str | None = "cuda",
        dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool | None = False,
        batch_size: int | str | None = 1,
        ds_batch_size: int | None = None,
        cpu_offload: bool = False,
        disk_offload: bool = False,
        offload_dir: str | None = None,
        pin_memory: bool = False,
        quant_bits: int = 16,
        quant_group_size: int = 64,
        hs_offload: bool = True,
        hs_max_gen_len: int | None = None,
        hs_device: str = "cpu",
        hs_codec_name: str = "fp16",
        hs_outlier_k: int | None = None,
        short_context_exact_threshold: int = 16,
        **model_kwargs,
    ):
        import deepspeed
        from deepspeed.runtime.hs_offload.config import HSOffloadConfig
        from transformers.deepspeed import HfDeepSpeedConfig

        with _hs_profile_section("adapter.distributed_init"):
            cls._maybe_init_distributed(deepspeed=deepspeed, device=device)
        with _hs_profile_section("adapter.config_load"):
            config = cls._load_model_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                subfolder=subfolder,
            )
        cls._ensure_causal_backend(backend=backend, config=config)

        hidden_size = cls._get_hidden_size(config)
        effective_hs_max_gen_len = cls._resolve_hs_staging_capacity(
            config=config,
            hs_max_gen_len=hs_max_gen_len,
        )
        resolved_dtype = cls._resolve_model_dtype(dtype=dtype, config=config)
        resolved_batch_size = cls._resolve_ds_batch_size(
            batch_size=batch_size,
            ds_batch_size=ds_batch_size,
        )
        ds_config = cls._build_deepspeed_config(
            hidden_size=hidden_size,
            dtype=resolved_dtype,
            batch_size=resolved_batch_size,
            cpu_offload=cpu_offload,
            disk_offload=disk_offload,
            offload_dir=offload_dir,
            pin_memory=pin_memory,
            quant_bits=quant_bits,
            quant_group_size=quant_group_size,
        )

        with _hs_profile_section("adapter.hf_deepspeed_config"):
            hf_ds_config = HfDeepSpeedConfig(ds_config)

        dtype_arg = (
            "dtype"
            if vparse(transformers.__version__) >= vparse("4.56.0")
            else "torch_dtype"
        )
        load_kwargs = dict(model_kwargs)
        load_kwargs.update(
            {
                "revision": revision,
                "trust_remote_code": trust_remote_code,
                dtype_arg: resolved_dtype,
            }
        )
        if subfolder:
            load_kwargs["subfolder"] = subfolder

        with _hs_profile_section("adapter.model_from_pretrained", sync_cuda=True):
            model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                **load_kwargs,
            ).eval()
        with _hs_profile_section("adapter.deepspeed_initialize", sync_cuda=True):
            ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
        ds_engine.module.eval()

        hs_config = None
        if hs_offload:
            hs_config = HSOffloadConfig(
                enabled=True,
                max_gen_len=effective_hs_max_gen_len,
                device=hs_device,
                codec_name=hs_codec_name,
                outlier_k=hs_outlier_k,
                hidden_dim=hidden_size,
                short_context_exact_threshold=int(short_context_exact_threshold),
            )
            with _hs_profile_section("adapter.set_hidden_state_offload", sync_cuda=True):
                ds_engine.set_hidden_state_offload(enabled=True, config=hs_config)

        return ds_engine, ds_engine.module, hf_ds_config, hs_config

    def _register_hs_profile_dump(self) -> None:
        if self._hs_profile_dump_registered or not _hs_profile_enabled():
            return
        atexit.register(self._write_hs_profile_snapshot)
        self._hs_profile_dump_registered = True

    def _write_hs_profile_snapshot(self) -> None:
        if not _hs_profile_enabled():
            return
        if self._hs_runtime is not None:
            _hs_profile_set("hs_runtime", self._hs_runtime.profiler_summary())
            _hs_profile_set("hs_runtime.decode_backend_stats", self._hs_runtime.decode_backend_stats())
            _hs_profile_set("hs_runtime.decode_backend_policy", self._hs_runtime.decode_backend_policy())
        if self._hs_config is not None:
            _hs_profile_set(
                "hs_config",
                {
                    "codec_name": getattr(self._hs_config, "codec_name", None),
                    "outlier_k": getattr(self._hs_config, "outlier_k", None),
                    "max_gen_len": getattr(self._hs_config, "max_gen_len", None),
                    "device": getattr(self._hs_config, "device", None),
                    "hidden_dim": getattr(self._hs_config, "hidden_dim", None),
                    "block_size": getattr(self._hs_config, "block_size", None),
                    "stage_slots": getattr(self._hs_config, "stage_slots", None),
                },
            )
        _hs_profile_dump()

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: int | None = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _hs_profile_section("lm_eval.tok_batch_encode"):
            input_ids, attention_mask = super().tok_batch_encode(
                strings=strings,
                padding_side=padding_side,
                left_truncate_len=left_truncate_len,
                truncation=truncation,
            )
        _hs_profile_count("lm_eval.tok_batch_encode.calls")
        _hs_profile_add("lm_eval.tok_batch_encode.batch_size_total", float(len(strings)))
        _hs_profile_add("lm_eval.tok_batch_encode.input_token_slots", float(input_ids.numel()))
        _hs_profile_add("lm_eval.tok_batch_encode.attention_tokens", float(attention_mask.sum().item()))
        return input_ids, attention_mask

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        _hs_profile_count("lm_eval.generate_until.calls")
        _hs_profile_add("lm_eval.generate_until.request_count", float(len(requests)))
        with _hs_profile_section("lm_eval.generate_until_total", sync_cuda=True):
            return super().generate_until(requests=requests, disable_tqdm=disable_tqdm)

    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs,
    ) -> torch.Tensor:
        _hs_profile_count("lm_eval.model_generate.calls")
        _hs_profile_add("lm_eval.model_generate.context_token_slots", float(context.numel()))
        _hs_profile_add("lm_eval.model_generate.max_new_token_budget", float(max(max_length - context.shape[1], 0)))
        with _hs_profile_section("lm_eval.model_generate", sync_cuda=True):
            self._maybe_enable_kv_offload(max_length=max_length, context=context)
            output = super()._model_generate(
                context=context,
                max_length=max_length,
                stop=stop,
                **generation_kwargs,
            )
        generated_tokens = max(int(output.shape[1]) - int(context.shape[1]), 0)
        _hs_profile_add("lm_eval.model_generate.generated_tokens", float(generated_tokens))
        _hs_profile_set("lm_eval.model_generate.last_generated_tokens", float(generated_tokens))
        _hs_profile_set("lm_eval.model_generate.last_context_len", float(context.shape[1]))
        _hs_profile_set("lm_eval.model_generate.last_output_len", float(output.shape[1]))
        return output

    def _maybe_enable_kv_offload(self, *, max_length: int, context: torch.Tensor) -> None:
        if not self._kv_offload_enabled:
            return
        set_kv_cache_offload = getattr(self.model, "set_kv_cache_offload", None)
        if set_kv_cache_offload is None:
            raise ValueError(
                "kv_offload=True requires the loaded model to expose set_kv_cache_offload()."
            )
        gen_len = max(int(max_length) - int(context.shape[1]), 0)
        set_kv_cache_offload(
            True,
            gen_len,
            self._pin_kv_cache,
            self._async_kv_offload,
        )

    @staticmethod
    def _validate_hs_load_args(
        *,
        backend: Literal["default", "causal", "seq2seq"],
        parallelize: bool | None,
        peft: str | None,
        delta: str | None,
        autogptq: bool | str | None,
        gptqmodel: bool | None,
        gguf_file: str | None,
        max_memory_per_gpu: int | str | None,
        max_cpu_memory: int | str | None,
    ) -> None:
        if backend == "seq2seq":
            raise NotImplementedError(
                "HSOffloadHFLM currently supports causal decoder-only models only."
            )
        unsupported = {
            "parallelize": parallelize,
            "peft": peft,
            "delta": delta,
            "autogptq": autogptq,
            "gptqmodel": gptqmodel,
            "gguf_file": gguf_file,
            "max_memory_per_gpu": max_memory_per_gpu,
            "max_cpu_memory": max_cpu_memory,
        }
        enabled = [name for name, value in unsupported.items() if value not in (None, False)]
        if enabled:
            raise ValueError(
                "HSOffloadHFLM does not support these HFLM load-time options: "
                + ", ".join(enabled)
            )

    @staticmethod
    def _load_model_config(
        pretrained: str,
        *,
        revision: str | None = "main",
        trust_remote_code: bool | None = False,
        subfolder: str = "",
    ) -> transformers.PretrainedConfig:
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        if subfolder:
            kwargs["subfolder"] = subfolder
        return transformers.AutoConfig.from_pretrained(pretrained, **kwargs)

    @staticmethod
    def _ensure_causal_backend(
        *,
        backend: Literal["default", "causal", "seq2seq"],
        config: transformers.PretrainedConfig,
    ) -> None:
        if backend == "seq2seq":
            raise NotImplementedError(
                "HSOffloadHFLM currently supports causal decoder-only models only."
            )
        model_type = getattr(config, "model_type", None)
        if model_type in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
            raise NotImplementedError(
                f"HSOffloadHFLM does not support seq2seq backend for model type '{model_type}'."
            )
        if backend == "default" and model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            eval_logger.info(
                "Model type '%s' is not explicitly listed as causal in transformers; "
                "continuing with AutoModelForCausalLM for HS offload.",
                model_type,
            )

    @staticmethod
    def _get_hidden_size(config: transformers.PretrainedConfig) -> int:
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None and hasattr(config, "text_config"):
            hidden_size = getattr(config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "Unable to infer hidden_size from model config for DeepSpeed HS offload."
            )
        return int(hidden_size)

    @staticmethod
    def _resolve_model_dtype(
        *,
        dtype: str | torch.dtype | None,
        config: transformers.PretrainedConfig,
    ) -> torch.dtype:
        if dtype in (None, "auto"):
            config_dtype = getattr(config, "torch_dtype", None)
            if config_dtype is None:
                return torch.float16
            return get_dtype(config_dtype)
        return get_dtype(dtype)

    @staticmethod
    def _resolve_hs_staging_capacity(
        *,
        config: transformers.PretrainedConfig,
        hs_max_gen_len: int | None,
    ) -> int:
        candidate_lengths = [
            getattr(config, "max_position_embeddings", None),
            getattr(config, "n_positions", None),
            getattr(config, "max_sequence_length", None),
            getattr(config, "seq_length", None),
        ]
        if hasattr(config, "text_config"):
            candidate_lengths.extend(
                [
                    getattr(config.text_config, "max_position_embeddings", None),
                    getattr(config.text_config, "n_positions", None),
                    getattr(config.text_config, "max_sequence_length", None),
                    getattr(config.text_config, "seq_length", None),
                ]
            )
        supported_len = max(
            [int(length) for length in candidate_lengths if length is not None],
            default=1024,
        )
        if hs_max_gen_len is None:
            eval_logger.info(
                "hs_max_gen_len was not provided; using model sequence capacity %s for HS staging.",
                supported_len,
            )
            return supported_len

        configured_len = int(hs_max_gen_len)
        effective_len = max(configured_len, supported_len)
        if effective_len != configured_len:
            eval_logger.info(
                "Increasing HS staging capacity from %s to %s to cover the model sequence length for generate_until workloads.",
                configured_len,
                effective_len,
            )
        return effective_len

    @staticmethod
    def _resolve_ds_batch_size(
        *,
        batch_size: int | str | None,
        ds_batch_size: int | None,
    ) -> int:
        if ds_batch_size is not None:
            return int(ds_batch_size)
        if isinstance(batch_size, int):
            return batch_size
        eval_logger.info(
            "Using DeepSpeed train_batch_size=1 because lm-eval batch_size was set to '%s'.",
            batch_size,
        )
        return 1

    @staticmethod
    def _build_deepspeed_config(
        *,
        hidden_size: int,
        dtype: torch.dtype,
        batch_size: int,
        cpu_offload: bool,
        disk_offload: bool,
        offload_dir: str | None,
        pin_memory: bool,
        quant_bits: int = 16,
        quant_group_size: int = 64,
    ) -> dict[str, Any]:
        if cpu_offload and disk_offload:
            raise ValueError("cpu_offload and disk_offload cannot both be enabled.")
        quant_bits = int(quant_bits)
        quant_group_size = int(quant_group_size)
        if quant_bits not in (4, 8, 16):
            raise ValueError("quant_bits must be one of 4, 8, or 16.")

        ds_config: dict[str, Any] = {
            "fp16": {"enabled": dtype == torch.float16},
            "bf16": {"enabled": dtype == torch.bfloat16},
            "zero_optimization": {
                "stage": 3,
                "stage3_prefetch_bucket_size": 2 * hidden_size * hidden_size,
                "stage3_param_persistence_threshold": hidden_size,
                "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
            },
            "steps_per_print": 2000,
            "train_batch_size": int(batch_size),
            "wall_clock_breakdown": False,
        }

        if quant_bits != 16:
            ds_config["weight_quantization"] = {
                "quantized_initialization": {
                    "num_bits": quant_bits,
                    "group_size": quant_group_size,
                    "group_dim": 1,
                    "symmetric": False,
                }
            }

        if cpu_offload:
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": bool(pin_memory),
            }

        if disk_offload:
            if not offload_dir:
                raise ValueError(
                    "offload_dir must be provided when disk_offload is enabled."
                )
            resolved_offload_dir = os.path.abspath(os.path.expanduser(offload_dir))
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "nvme",
                "pin_memory": bool(pin_memory),
                "nvme_path": resolved_offload_dir,
                "buffer_count": 5,
                "buffer_size": 2 * 1024 * 1024 * 1024,
            }
            ds_config["aio"] = {
                "block_size": 1048576 * 16,
                "queue_depth": 64,
                "thread_count": 8,
                "single_submit": False,
                "overlap_events": True,
            }

        return ds_config

    @staticmethod
    def _maybe_init_distributed(*, deepspeed, device: str | None) -> None:
        if not torch.distributed.is_available():
            return
        if torch.distributed.is_initialized():
            return
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1:
            return
        backend = "nccl"
        if device is not None and str(device).startswith("cpu"):
            backend = "gloo"
        deepspeed.init_distributed(backend)

    @staticmethod
    def _resolve_runtime_device(device: str | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _sync_distributed_state(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()

    def all_gather(self, tensor):
        if self.world_size <= 1:
            return tensor
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return tensor
        tensor = tensor.contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, tensor)
        if tensor.ndim == 0:
            return torch.stack(gathered)
        return torch.cat(gathered, dim=0)

    def gather_object(self, obj, dst=0):
        if self.world_size <= 1:
            return [obj]
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return [obj]
        result = [None] * self.world_size if self.rank == dst else None
        torch.distributed.gather_object(obj=obj, object_gather_list=result, dst=dst)
        return result

    def barrier(self) -> None:
        if self.world_size > 1 and torch.distributed.is_available():
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    def get_model_info(self) -> dict:
        model_info = super().get_model_info()
        hs_compression = self._hs_compression_result_info()
        if hs_compression:
            model_info["hs_compression"] = hs_compression
        return model_info

    def _hs_compression_result_info(self) -> dict[str, Any]:
        if self._hs_runtime is None:
            return {}
        summary = self._hs_runtime.compression_summary() or {}

        return {
            "codec_name": getattr(self._hs_config, "codec_name", None),
            "outlier_k": getattr(self._hs_config, "outlier_k", None),
            "input_bytes": float(summary.get("compressed_input_bytes", 0.0)),
            "output_bytes": float(summary.get("compressed_output_bytes", 0.0)),
            "block_count": float(summary.get("compressed_block_count", 0.0)),
            "compression_ratio": float(summary.get("compression_ratio", 0.0)),
            "compressed_size_ratio": float(summary.get("compressed_size_ratio", 0.0)),
            "compression_savings": float(summary.get("compression_savings", 0.0)),
        }
